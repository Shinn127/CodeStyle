from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Utils.fid_feature_extractor import (
    JOINT_SET_SELECTED24,
    build_fid_feature_extractor_from_checkpoint,
    motion375_clip_to_classifier_input,
)
from Utils.motion375 import reconstruct_root_track_from_375
from Utils.motion_eval import MotionModelName, load_motion_model, reconstruct_motion_batch
from Utils.runtime import load_checkpoint, load_norm_stats, resolve_device, set_seed, validate_memmap_db_dir, configure_torch_runtime
from dataset import Motion375Dataset, build_train_val_test_datasets, norm_stats_fingerprint


from Utils.reproducibility import case_noise_plan, cluster_interval, file_sha256
from Utils.style_code_prior import sparse_code_histogram, recipe_probabilities_from_indices, recipe_conditioning_metadata

SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ci95_half_width(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    return float(1.96 * np.std(finite, ddof=1) / np.sqrt(finite.size))


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _resolve_norm_stats_path(flow_ckpt: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    ckpt = load_checkpoint(flow_ckpt, map_location="cpu")
    flow_args = ckpt.get("args", {}) or {}
    raw = flow_args.get("norm_stats", None)
    if raw is not None:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (flow_ckpt.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    base_raw = flow_args.get("vqvae_ckpt", None)
    if base_raw is not None:
        base = Path(base_raw).expanduser()
        if not base.is_absolute():
            base = (flow_ckpt.parent / base).resolve()
        else:
            base = base.resolve()
        candidate = base.parent / "norm_stats.npz"
        if candidate.exists():
            return candidate.resolve()
    candidate = flow_ckpt.parent / "norm_stats.npz"
    if candidate.exists():
        return candidate.resolve()
    return None


def _load_style_names(db: Path) -> list[str]:
    path = db / "StyleNames.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def _validate_style_mapping(db_style_names: list[str], classifier_ckpt: dict, *, strict: bool) -> None:
    class_names = list(classifier_ckpt.get("class_names", []))
    if not class_names or not db_style_names:
        return
    if len(class_names) < len(db_style_names):
        raise ValueError(
            f"Classifier has fewer classes ({len(class_names)}) than generator DB styles ({len(db_style_names)})."
        )
    prefix = class_names[: len(db_style_names)]
    if prefix != db_style_names:
        message = (
            "Classifier class_names do not match DB StyleNames.json order. "
            "Target-style accuracy/confidence would be invalid."
        )
        if strict:
            raise ValueError(message)
        print(f"[Warn] {message}")


def _validate_classifier_capacity(classifier_ckpt: dict, *, num_styles: int) -> None:
    class_num = int(classifier_ckpt.get("class_num", len(classifier_ckpt.get("class_names", []))))
    if class_num < int(num_styles):
        raise ValueError(
            f"Classifier has {class_num} output classes but evaluation needs labels [0,{int(num_styles) - 1}]."
        )


def _denormalize(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if mean is None or std is None:
        return x
    return x * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _root_ade_cm(reference_375: np.ndarray, prediction_375: np.ndarray, *, dt: float, context_frames: int = 0) -> float:
    ref_root_pos, _ = reconstruct_root_track_from_375(reference_375, dt=dt, root_wy_in_degrees=True)
    pred_root_pos, _ = reconstruct_root_track_from_375(prediction_375, dt=dt, root_wy_in_degrees=True)
    values = np.linalg.norm(pred_root_pos[:, [0, 2]] - ref_root_pos[:, [0, 2]], axis=-1) * 100.0
    return float(np.mean(values[context_frames:]))


def _resolve_evaluation_horizons(horizons, classifier_window):
    resolved = [classifier_window] if horizons is None else list(horizons)
    if classifier_window < 1 or not resolved or any(h < classifier_window for h in resolved):
        raise ValueError(f'Evaluation horizons must provide at least {classifier_window} classifier frames')
    return resolved


def _classifier_logits_for_clips(
    classifier: torch.nn.Module,
    clips_375: np.ndarray,
    *,
    device: torch.device,
    classifier_window: int,
    classifier_stride: int,
    batch_size: int,
    joint_set: str = JOINT_SET_SELECTED24,
) -> torch.Tensor:
    clips = np.asarray(clips_375, dtype=np.float32)
    if clips.ndim != 3 or clips.shape[-1] != 375:
        raise ValueError(f"Expected clips shape [B,T,375], got {clips.shape}")
    if classifier_window < 1:
        raise ValueError(f"classifier_window must be >= 1, got {classifier_window}")
    if classifier_stride < 1:
        raise ValueError(f"classifier_stride must be >= 1, got {classifier_stride}")

    logits_per_clip: list[torch.Tensor] = []
    with torch.no_grad():
        for clip in clips:
            total_frames = int(clip.shape[0])
            if total_frames < classifier_window:
                raise ValueError(
                    f"Clip length {total_frames} is shorter than classifier window {classifier_window}."
                )
            starts = list(range(0, total_frames - classifier_window + 1, classifier_stride))
            if starts[-1] != total_frames - classifier_window:
                starts.append(total_frames - classifier_window)
            window_logits: list[torch.Tensor] = []
            for begin in range(0, len(starts), batch_size):
                batch_starts = starts[begin : begin + batch_size]
                windows = np.stack([clip[s : s + classifier_window] for s in batch_starts], axis=0)
                x_cls = motion375_clip_to_classifier_input(windows, joint_set=joint_set).to(device=device, non_blocking=True)
                _, logits = classifier(x_cls)
                window_logits.append(logits.detach())
            logits_clip = torch.cat(window_logits, dim=0).mean(dim=0)
            logits_per_clip.append(logits_clip.cpu())
    return torch.stack(logits_per_clip, dim=0)


def _classifier_stats(
    classifier: torch.nn.Module,
    clips_375: np.ndarray,
    target_labels: np.ndarray,
    *,
    device: torch.device,
    classifier_window: int,
    classifier_stride: int,
    classifier_batch_size: int,
    joint_set: str = JOINT_SET_SELECTED24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = _classifier_logits_for_clips(
        classifier,
        clips_375,
        device=device,
        classifier_window=classifier_window,
        classifier_stride=classifier_stride,
        batch_size=classifier_batch_size,
        joint_set=joint_set,
    )
    probs = F.softmax(logits, dim=1)
    target = torch.from_numpy(np.asarray(target_labels, dtype=np.int64).reshape(-1))
    if target.numel() != probs.size(0):
        raise ValueError(f"Expected {probs.size(0)} target labels, got {target.numel()}")
    pred = probs.argmax(dim=1)
    target_conf = probs[torch.arange(probs.size(0)), target]
    return (
        pred.numpy().astype(np.int64),
        target_conf.numpy().astype(np.float64),
        probs.max(dim=1).values.numpy().astype(np.float64),
    )


def _resolve_recipe_style_map(
    *,
    recipe_style_names: list[str],
    target_style_names: list[str],
    classifier_class_names: list[str],
) -> dict[int, int]:
    target_name_to_label = {name: idx for idx, name in enumerate(target_style_names)}
    classifier_name_to_label = {name: idx for idx, name in enumerate(classifier_class_names)}
    mapping: dict[int, int] = {}
    for local_label, style_name in enumerate(recipe_style_names):
        if style_name in target_name_to_label:
            mapping[local_label] = int(target_name_to_label[style_name])
        elif style_name in classifier_name_to_label:
            mapping[local_label] = int(classifier_name_to_label[style_name])
    return mapping


def _build_target_style_recipe_probs(
    *,
    model: torch.nn.Module,
    recipe_db: Path,
    target_style_names: list[str],
    classifier_class_names: list[str],
    target_style_count: int,
    seq_len: int,
    stride: int,
    exemplars_per_style: int,
    batch_size: int,
    seed: int,
    norm_mean: np.ndarray | None,
    norm_std: np.ndarray | None,
    device: torch.device,
    recipe_profile: str = "canonical",
) -> tuple[torch.Tensor, dict[str, Any]]:
    if min(seq_len,stride,exemplars_per_style,batch_size,target_style_count)<1:
        raise ValueError('Recipe window/stride, exemplar count, batch size and target count must be positive')
    if not bool(getattr(model, "use_style_code_selector", False)):
        raise RuntimeError("Target-style recipe requires a FlowMatch checkpoint with use_style_code_selector=True.")
    selected_slots = tuple(int(slot) for slot in getattr(model, "selected_style_slots", ()))
    if not selected_slots:
        raise RuntimeError("FlowMatch model has no selected_style_slots for recipe conditioning.")
    num_embeddings = int(model.vqvae.quantizer.num_embeddings)
    recipe_style_names = _load_style_names(recipe_db)
    if not recipe_style_names:
        raise RuntimeError(f"Recipe DB has no StyleNames.json entries: {recipe_db}")
    style_map = _resolve_recipe_style_map(
        recipe_style_names=recipe_style_names,
        target_style_names=target_style_names,
        classifier_class_names=classifier_class_names,
    )
    if not style_map:
        raise RuntimeError("Could not map any recipe DB style names to target/classifier labels.")
    if recipe_profile not in {'canonical','legacy'}:
        raise ValueError('Recipe profile must be canonical or legacy')
    if recipe_profile == 'canonical':
        if norm_mean is None or norm_std is None:
            raise ValueError('Canonical recipes require saved VQ normalization')
        segments = json.loads((recipe_db/'segments.json').read_text())
        meta = json.loads((recipe_db/'meta.json').read_text())
        source = np.memmap(recipe_db/'X.f32',mode='r',dtype=np.float32,shape=(meta['num_frames'],375))
        rng = np.random.default_rng(seed)
        probs = np.zeros((target_style_count,len(selected_slots),num_embeddings),dtype=np.float32)
        manifests = []
        for target_label in range(target_style_count):
            independent = {}
            for index, segment in enumerate(segments):
                if (style_map.get(segment['label']) == target_label and segment.get('split')=='train'
                        and segment['num_frames']>=seq_len):
                    if 'source_entry_id' not in segment:
                        raise ValueError('Canonical recipes require original source_entry_id')
                    independent.setdefault(segment['source_entry_id'],(index,segment))
            candidates = list(independent.values())
            if len(candidates)<exemplars_per_style:
                raise ValueError(f'Target {target_label} has fewer than {exemplars_per_style} independent train exemplars')
            chosen = rng.choice(len(candidates),exemplars_per_style,replace=False)
            clips=[]; selected=[]
            for index in chosen:
                segment_id,segment = candidates[index]
                start = segment['frame_start'] + int(rng.integers(segment['num_frames']-seq_len+1))
                clips.append(torch.from_numpy((np.array(source[start:start+seq_len],copy=True)-norm_mean)/norm_std))
                selected.append(dict(segment_id=segment_id,source_entry_id=segment['source_entry_id'],
                                     frame_start=start,frames=seq_len))
            parts=[]
            for begin in range(0,len(clips),batch_size):
                batch = torch.stack(clips[begin:begin+batch_size]).to(device=device,dtype=torch.float32)
                with torch.no_grad():
                    _,indices,_ = model.vqvae.encode(batch)
                parts.append(indices.cpu().numpy().reshape(-1,int(model.vqvae.num_latents)))
            probs[target_label] = recipe_probabilities_from_indices(np.concatenate(parts), model,
                clip_lengths=[seq_len]*len(clips))
            manifests.append(dict(target_label=target_label,clips=selected))
        return torch.from_numpy(probs),dict(**recipe_conditioning_metadata(model), recipe_profile='canonical',representation='topk_sparse',
            sampling='independent_train_clips',topk=model.style_selector.topk,recipe_db=str(recipe_db),
            selected_style_slots=list(selected_slots),seq_len=seq_len,seed=seed,
            exemplars_per_style=exemplars_per_style,chosen_clips=manifests)

    dataset = Motion375Dataset(
        db_path=recipe_db,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=True,
        load_into_memory=False,
        normalize=True,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    local_labels = np.asarray([int(dataset.labels[int(start)]) for start in dataset.sample_starts], dtype=np.int64)
    sample_ids_by_target: dict[int, list[int]] = {}
    for sample_idx, local_label in enumerate(local_labels):
        target_label = style_map.get(int(local_label))
        if target_label is None or target_label < 0 or target_label >= int(target_style_count):
            continue
        sample_ids_by_target.setdefault(int(target_label), []).append(sample_idx)

    rng = np.random.default_rng(seed)
    counts = np.zeros((target_style_count, len(selected_slots), num_embeddings), dtype=np.float64)
    selected_style_ids: list[int] = []

    for target_label in range(target_style_count):
        sample_ids = sample_ids_by_target.get(target_label, [])
        if not sample_ids:
            continue
        keep = min(int(exemplars_per_style), len(sample_ids))
        chosen = rng.choice(np.asarray(sample_ids, dtype=np.int64), size=keep, replace=False).tolist()
        selected_style_ids.append(target_label)
        for begin in range(0, len(chosen), batch_size):
            batch_ids = chosen[begin : begin + batch_size]
            clips = torch.stack([dataset[int(sample_id)]["x"] for sample_id in batch_ids], dim=0)
            clips = clips.to(device=device, dtype=torch.float32, non_blocking=True)
            _, indices, _ = model.vqvae.encode(clips)
            indices_np = indices.detach().cpu().numpy().astype(np.int64)
            selected = indices_np[..., np.asarray(selected_slots, dtype=np.int64)]
            for slot_idx in range(len(selected_slots)):
                slot_codes = selected[..., slot_idx].reshape(-1)
                counts[target_label, slot_idx] += np.bincount(slot_codes, minlength=num_embeddings)

    missing = [idx for idx in range(target_style_count) if idx not in selected_style_ids]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        raise RuntimeError(
            f"Recipe source DB did not provide exemplars for {len(missing)} target styles. "
            f"Missing label preview: {preview}"
        )
    probs = counts / np.clip(counts.sum(axis=-1, keepdims=True), 1e-8, None)
    meta = {
        "recipe_profile": "legacy", "representation": "dense", "sampling": "all_overlapping_windows",
        "recipe_db": str(recipe_db),
        "recipe_style_count": len(recipe_style_names),
        "target_style_count": int(target_style_count),
        "selected_style_slots": list(selected_slots),
        "num_embeddings": num_embeddings,
        "seq_len": int(seq_len),
        "stride": int(stride),
        "exemplars_per_style": int(exemplars_per_style),
        "seed": int(seed),
        "mapped_styles": int(len(selected_style_ids)),
    }
    return torch.from_numpy(probs.astype(np.float32)), meta


def _make_summary(setting: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    root_values = [float(row["root_ade_cm"]) for row in rows if np.isfinite(float(row["root_ade_cm"]))]
    correct = [1.0 if int(row["pred_style"]) == int(row["target_style"]) else 0.0 for row in rows]
    target_conf = [float(row["target_conf"]) for row in rows]
    pred_conf = [float(row["pred_conf"]) for row in rows]
    root_mean = _mean(root_values)
    groups = [row.get("source_entry_id", row.get("source_start", idx)) for idx, row in enumerate(rows)]
    all_roots = [float(row["root_ade_cm"]) for row in rows]
    root_low, root_high = cluster_interval(all_roots, groups)
    root_ci = .5 * (root_high - root_low)
    acc_pct = 100.0 * _mean(correct)
    acc_low, acc_high = cluster_interval(correct, groups)
    acc_ci = 100.0 * .5 * (acc_high - acc_low)
    target_conf_pct = 100.0 * _mean(target_conf)
    conf_low, conf_high = cluster_interval(target_conf, groups)
    target_conf_ci = 100.0 * .5 * (conf_high - conf_low)
    uses_recipe = any(bool(row.get("uses_recipe", False)) for row in rows)
    recipe_strength = _mean([float(row.get("recipe_strength", 0.0)) for row in rows])
    return {
        "ci_unit": "source_entry_id", "bootstrap_samples": 1000,
        "setting": setting,
        "num_pairs": len(rows),
        "uses_recipe": uses_recipe,
        "recipe_strength": recipe_strength,
        "root_ade_cm": root_mean,
        "root_ade_cm_ci95": root_ci,
        "root_ade_cm_ci95_bounds": [root_low, root_high],
        "target_style_acc_pct": acc_pct,
        "target_style_acc_pct_ci95": acc_ci,
        "target_style_acc_pct_ci95_bounds": [100 * acc_low, 100 * acc_high],
        "target_conf_pct": target_conf_pct,
        "target_conf_pct_ci95": target_conf_ci,
        "pred_conf_pct": 100.0 * _mean(pred_conf),
        "root_ade_cm_text": "-" if not np.isfinite(root_mean) else f"{root_mean:.3f} +/- {root_ci:.3f}",
        "target_style_acc_pct_text": f"{acc_pct:.2f} +/- {acc_ci:.2f}",
        "target_conf_pct_text": f"{target_conf_pct:.2f} +/- {target_conf_ci:.2f}",
    }


def _resolve_flow_model_name(flow_ckpt: Path, explicit: str) -> MotionModelName:
    if explicit != "auto":
        return explicit  # type: ignore[return-value]
    ckpt = load_checkpoint(flow_ckpt, map_location="cpu")
    prediction_mode = str((ckpt.get("args", {}) or {}).get("prediction_mode", "velocity")).lower()
    if prediction_mode == "target":
        return "flowmatch_target"
    return "flowmatch_velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minor-revision generalization diagnostic: Root ADE + target style accuracy/confidence."
    )
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--flow-ckpt", type=Path, default=SCRIPT_DIR / "checkpoints" / "flowmatch_vqvae_controller" / "best.pt")
    parser.add_argument("--flow-vqvae-ckpt", type=Path, default=None)
    parser.add_argument("--classifier-ckpt", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, default=None)
    parser.add_argument("--legacy-inference-order", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "eval_results" / "minor_revision_generalization_generated")
    parser.add_argument(
        "--flow-model-name",
        choices=["auto", "flowmatch_velocity", "flowmatch_target"],
        default="auto",
        help="Use auto to infer velocity/target from the FlowMatch checkpoint args.",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Generated frames per case; defaults to the classifier window (240 for the 4-second classifier)")
    parser.add_argument('--include-gt-context', action='store_true',
                        help='Legacy replay: count the initial GT row in the horizon and classifier input')
    parser.add_argument('--source-split', choices=['val', 'test'], default='test',
                        help='Use val for checkpoint selection; keep test sealed until selection is complete')
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--num-sources", type=int, default=1000)
    parser.add_argument("--targets-per-source", type=int, default=5)
    parser.add_argument("--classifier-stride", type=int, default=10)
    parser.add_argument("--classifier-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument("--use-target-style-recipe", action="store_true",
                        help="Legacy seen-target recipe comparison (label+recipe); not used by the label runs")
    parser.add_argument('--recipe-only', action='store_true',
                        help='Evaluate matched/cross using pure recipes; automatic for recipe-only checkpoints')
    parser.add_argument('--recipe-profile',choices=['canonical','legacy'],default='canonical',
                        help='canonical uses independent train clips/TopK; legacy replays all-window dense recipes')
    parser.add_argument("--recipe-source-db", type=Path, default=None)
    parser.add_argument("--recipe-exemplars-per-style", type=int, default=5)
    parser.add_argument("--recipe-window", type=int, default=240)
    parser.add_argument("--recipe-stride", type=int, default=30)
    parser.add_argument("--recipe-batch-size", type=int, default=64)
    parser.add_argument("--recipe-strength", type=float, default=1.0)
    parser.add_argument("--recipe-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--flow-eval-mode", choices=["rollout", "reconstruct"], default="rollout")
    parser.add_argument("--flow-solver", choices=["euler"], default="euler")
    parser.add_argument("--flow-ode-steps", type=int, default=0)
    parser.add_argument("--flow-noise-std", type=float, default=-1.0)
    parser.add_argument('--flow-microbatch-size', type=int, default=512,
                        help='Flattened teacher-forced pure-recipe frame batch size')
    parser.add_argument("--flow-requantize-z", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flow-latent-clip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--no-strict-style-names", dest="strict_style_names", action="store_false", default=True)
    return parser.parse_args()


def _build_test_loader(args: argparse.Namespace, *, horizon: int, mean: np.ndarray | None, std: np.ndarray | None):
    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(
        db_path=args.db,
        seq_len=horizon + (0 if args.include_gt_context else 1),
        stride=args.stride,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        load_into_memory=args.load_into_memory,
        normalize=True,
        norm_mean=mean,
        norm_std=std,
    )
    source_split = getattr(args, 'source_split', 'test')
    if source_split == 'val':
        test_dataset = val_dataset
    if test_dataset is None or len(test_dataset) == 0:
        raise RuntimeError(f"{source_split} split is empty for horizon={horizon}")
    rng = np.random.default_rng(args.seed + int(horizon))
    eligible = np.arange(len(test_dataset))
    if not len(eligible):
        raise ValueError(f'No eligible {source_split} source windows')
    keep = min(int(args.num_sources), len(eligible))
    existing = getattr(args, "case_data", {}).get("horizons", {}).get(str(horizon))
    if existing is not None:
        lookup = {int(test_dataset.sample_starts[i]): int(i) for i in eligible}
        if len(existing) != keep:
            raise ValueError("Case manifest source count differs from requested protocol")
        try:
            indices = [lookup[int(row["source_start"])] for row in existing]
        except KeyError as exc:
            raise ValueError(f"Manifest source is not in this {source_split} split/window protocol") from exc
    else:
        indices = rng.choice(eligible, size=keep, replace=False).tolist()
    subset = Subset(test_dataset, indices)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(subset, **loader_kwargs), len(eligible), keep, train_dataset.norm_mean, train_dataset.norm_std


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
    args.classifier_ckpt = args.classifier_ckpt.expanduser().resolve()
    if args.recipe_source_db is None:
        args.recipe_source_db = args.db
    else:
        args.recipe_source_db = args.recipe_source_db.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.case_manifest = args.case_manifest or args.out_dir / "case_manifest.json"
    protocol_version = 2 if args.include_gt_context else 3
    context_frames = 0 if args.include_gt_context else 1
    args.case_data = json.loads(args.case_manifest.read_text()) if args.case_manifest.exists() else {
        "protocol_version": protocol_version, "seed": args.seed, "targets_per_source": args.targets_per_source,
        "source_split": args.source_split,
        "split_sha256": file_sha256(args.db / "segments.json"), "horizons": {}}
    for name, expected in (("protocol_version", protocol_version), ("seed", args.seed), ("targets_per_source", args.targets_per_source),
                           ("split_sha256", file_sha256(args.db / "segments.json"))):
        if args.case_data.get(name) != expected:
            raise ValueError(f"Case manifest protocol mismatch: {name}")
    if args.case_data.get('source_split', 'test') != args.source_split:
        raise ValueError('Case manifest source split differs')
    args.case_data['source_split'] = args.source_split
    db_segments = json.loads((args.db / "segments.json").read_text())

    validate_memmap_db_dir(args.db)
    if args.use_target_style_recipe:
        validate_memmap_db_dir(args.recipe_source_db)
    if not args.flow_ckpt.exists():
        raise FileNotFoundError(f"--flow-ckpt not found: {args.flow_ckpt}")
    if not args.classifier_ckpt.exists():
        raise FileNotFoundError(f"--classifier-ckpt not found: {args.classifier_ckpt}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision='highest')

    flow_model_name = _resolve_flow_model_name(args.flow_ckpt, args.flow_model_name)
    norm_stats_path = _resolve_norm_stats_path(args.flow_ckpt, args.norm_stats)
    motion_mean = motion_std = None
    if norm_stats_path is not None:
        motion_mean, motion_std = load_norm_stats(norm_stats_path)

    if motion_mean is None or motion_std is None:
        raise ValueError("Evaluation requires saved VQ/controller normalization")
    db_style_names = _load_style_names(args.db)
    classifier, classifier_ckpt = build_fid_feature_extractor_from_checkpoint(args.classifier_ckpt, map_location="cpu")
    classifier_class_names = list(classifier_ckpt.get("class_names", []))
    classifier_window = int(classifier_ckpt.get("window_size", 60))
    args.horizons = _resolve_evaluation_horizons(args.horizons, classifier_window)
    classifier_fps = float(classifier_ckpt.get('fps', 1. / args.dt))
    if not np.isclose(classifier_fps, 1. / args.dt):
        raise ValueError('Classifier fps differs from evaluation fps; temporal resampling must be explicit')
    classifier_joint_set = str(classifier_ckpt.get("joint_set", JOINT_SET_SELECTED24))
    print(f"[Info] classifier joints  : {classifier_joint_set} ({int(classifier_ckpt.get('in_feat', 0))}-dim input)")
    _validate_style_mapping(db_style_names, classifier_ckpt, strict=args.strict_style_names)
    classifier = classifier.to(device)
    classifier.eval()

    num_styles = len(db_style_names)
    if num_styles <= 0:
        class_num = int(classifier_ckpt.get("class_num", 0))
        if class_num <= 0:
            raise RuntimeError("Could not infer number of styles from DB or classifier checkpoint.")
        num_styles = class_num
    _validate_classifier_capacity(classifier_ckpt, num_styles=num_styles)

    flow_model = load_motion_model(
        model_name=flow_model_name,
        ckpt_path=args.flow_ckpt,
        num_styles=num_styles,
        flow_vqvae_ckpt=args.flow_vqvae_ckpt,
    ).to(device)
    flow_model.eval()
    args.recipe_only = args.recipe_only or getattr(flow_model, 'conditioning_mode', 'label') in ('recipe', 'clip_recipe')
    if getattr(flow_model, 'conditioning_mode', 'label') == 'clip_recipe' and args.recipe_profile != 'canonical':
        raise ValueError('clip_recipe checkpoints require canonical learned recipe export')
    if args.recipe_only:
        args.use_target_style_recipe = True
        if args.recipe_profile != 'canonical' or args.recipe_strength != 1.:
            raise ValueError('Recipe-only evaluation requires canonical pure recipes with strength=1')
        validate_memmap_db_dir(args.recipe_source_db)
    if args.flow_microbatch_size < 1:
        raise ValueError('Flow evaluation microbatch size must be positive')
    flow_model.eval_microbatch_size = args.flow_microbatch_size
    recorded_norm = getattr(flow_model, 'eval_fingerprints', {}).get('norm_fingerprint')
    if recorded_norm and recorded_norm != norm_stats_fingerprint(motion_mean, motion_std):
        raise ValueError('Evaluation normalization differs from the controller/VQ manifold')
    recipe_probs: torch.Tensor | None = None
    recipe_meta: dict[str, Any] | None = None
    if args.use_target_style_recipe and motion_mean is not None and motion_std is not None:
        recipe_probs, recipe_meta = _build_target_style_recipe_probs(
            model=flow_model,
            recipe_db=args.recipe_source_db,
            target_style_names=db_style_names,
            classifier_class_names=classifier_class_names,
            target_style_count=num_styles,
            seq_len=args.recipe_window,
            stride=args.recipe_stride,
            exemplars_per_style=args.recipe_exemplars_per_style,
            batch_size=args.recipe_batch_size,
            seed=args.recipe_seed,
            norm_mean=motion_mean,
            norm_std=motion_std,
            device=device,
            recipe_profile=args.recipe_profile,
        )
        recipe_probs = recipe_probs.to(device=device)

    print(f"[Info] db                 : {args.db}")
    print(f"[Info] flow ckpt          : {args.flow_ckpt}")
    print(f"[Info] flow model         : {flow_model_name}")
    print(f"[Info] classifier ckpt    : {args.classifier_ckpt}")
    print(f"[Info] classifier window  : {classifier_window}")
    print(f"[Info] classifier span    : {classifier_window / classifier_fps:g} s @ {classifier_fps:g} fps")
    print(f"[Info] GT context in acc  : {args.include_gt_context}")
    print(f"[Info] styles             : {num_styles}")
    print(f"[Info] device             : {device}")
    print(f"[Info] horizons           : {args.horizons}")
    print(f"[Info] num sources        : {args.num_sources}")
    print(f"[Info] targets/source     : {args.targets_per_source}")
    print(f"[Info] norm stats         : {norm_stats_path if norm_stats_path is not None else 'none'}")
    print(f"[Info] target recipe      : {args.use_target_style_recipe}")
    if args.use_target_style_recipe:
        print(f"[Info] recipe source db   : {args.recipe_source_db}")
        print(f"[Info] recipe strength    : {args.recipe_strength}")
    print(
        f"[Info] flow eval          : mode={args.flow_eval_mode} "
        f"ode_steps={'ckpt' if args.flow_ode_steps <= 0 else args.flow_ode_steps} "
        f"noise_std={'ckpt' if args.flow_noise_std < 0.0 else args.flow_noise_std} "
        f"requantize_z={args.flow_requantize_z} latent_clip={args.flow_latent_clip}"
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for horizon in args.horizons:
            source_frames = int(horizon) + context_frames
            loader, total_test, kept_sources, loader_mean, loader_std = _build_test_loader(
                args,
                horizon=int(horizon),
                mean=motion_mean,
                std=motion_std,
            )
            if motion_mean is None or motion_std is None:
                motion_mean = loader_mean
                motion_std = loader_std
                print("[Warn] norm stats not found in checkpoint; using stats estimated from the DB train split.")
                if args.use_target_style_recipe and recipe_probs is None:
                    recipe_probs, recipe_meta = _build_target_style_recipe_probs(
                        model=flow_model,
                        recipe_db=args.recipe_source_db,
                        target_style_names=db_style_names,
                        classifier_class_names=classifier_class_names,
                        target_style_count=num_styles,
                        seq_len=args.recipe_window,
                        stride=args.recipe_stride,
                        exemplars_per_style=args.recipe_exemplars_per_style,
                        batch_size=args.recipe_batch_size,
                        seed=args.recipe_seed,
                        norm_mean=motion_mean,
                        norm_std=motion_std,
                        device=device,
                        recipe_profile=args.recipe_profile,
                    )
                    recipe_probs = recipe_probs.to(device=device)
            print(f"[Info] horizon={horizon} {args.source_split} windows={total_test} sampled sources={kept_sources}")

            real_rows: list[dict[str, Any]] = []
            matched_rows: list[dict[str, Any]] = []
            cross_rows: list[dict[str, Any]] = []
            cross_recipe_rows: list[dict[str, Any]] = []

            for batch in tqdm(loader, desc=f"horizon {horizon}", leave=False):
                x = batch["x"].to(device=device, non_blocking=True).float()
                source_labels = batch["label"].to(device=device, non_blocking=True).long()
                starts = batch["start"].detach().cpu().numpy().astype(np.int64)
                segment_ids = batch["segment_id"].tolist()
                source_groups = [int(db_segments[int(s)]["source_entry_id"]) for s in segment_ids]
                source_rows = args.case_data["horizons"].setdefault(str(horizon), [])
                rows_by_start = {int(row["source_start"]): row for row in source_rows}
                for i, start in enumerate(starts):
                    recorded = rows_by_start.get(int(start))
                    if recorded is not None and (recorded['source_label'] != int(source_labels[i])
                            or recorded['source_entry_id'] != source_groups[i] or recorded['noise_seed'] != args.seed):
                        raise ValueError('Manifest source identity/label/noise seed differs')
                if args.targets_per_source <= 0:
                    for i, start in enumerate(starts):
                        if int(start) not in rows_by_start:
                            source_rows.append({"source_start": int(start), "source_entry_id": source_groups[i],
                                "source_label": int(source_labels[i]), "targets": [], "noise_seed": args.seed})
                batch_size = int(x.size(0))
                ref_np = _denormalize(x.detach().cpu().numpy(), motion_mean, motion_std)

                real_pred, real_target_conf, real_pred_conf = _classifier_stats(
                    classifier,
                    ref_np[:, context_frames:],
                    source_labels.detach().cpu().numpy(),
                    device=device,
                    classifier_window=classifier_window,
                    classifier_stride=args.classifier_stride,
                    classifier_batch_size=args.classifier_batch_size,
                    joint_set=classifier_joint_set,
                )
                for i in range(batch_size):
                    row = {
                        "setting": f"real_motion_upper_bound_h{horizon}",
                        "horizon": int(horizon),
                        "source_start": int(starts[i]), "source_entry_id": source_groups[i],
                        "source_style": int(source_labels[i].item()),
                        "target_style": int(source_labels[i].item()),
                        "pred_style": int(real_pred[i]),
                        "target_conf": float(real_target_conf[i]),
                        "pred_conf": float(real_pred_conf[i]),
                        "root_ade_cm": float("nan"),
                    }
                    real_rows.append(row)

                t0 = time.perf_counter()
                matched_pred = reconstruct_motion_batch(
                    model_name=flow_model_name,
                    model=flow_model,
                    x=x,
                    style_label=None if args.recipe_only else source_labels,
                    manual_style_probs=recipe_probs.index_select(0, source_labels) if args.recipe_only else None,
                    flow_eval_mode=args.flow_eval_mode,
                    flow_solver=args.flow_solver,
                    flow_ode_steps=None if args.flow_ode_steps <= 0 else args.flow_ode_steps,
                    flow_noise_std=None if args.flow_noise_std < 0.0 else args.flow_noise_std,
                    flow_requantize_z=args.flow_requantize_z,
                    flow_latent_clip=args.flow_latent_clip,
                    legacy_inference_order=args.legacy_inference_order,
                    noise_plan=case_noise_plan(flow_model, starts, source_labels.cpu().tolist(), source_frames,
                        seed=args.seed, noise_std=flow_model.eval_noise_std if args.flow_noise_std < 0 else args.flow_noise_std),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                matched_latency_ms = (time.perf_counter() - t0) * 1000.0 / max(batch_size, 1)
                matched_np = _denormalize(matched_pred.detach().cpu().numpy(), motion_mean, motion_std)
                matched_cls_pred, matched_target_conf, matched_pred_conf = _classifier_stats(
                    classifier,
                    matched_np[:, context_frames:],
                    source_labels.detach().cpu().numpy(),
                    device=device,
                    classifier_window=classifier_window,
                    classifier_stride=args.classifier_stride,
                    classifier_batch_size=args.classifier_batch_size,
                    joint_set=classifier_joint_set,
                )
                for i in range(batch_size):
                    row = {
                        "setting": f"matched_style_{'recipe_' if args.recipe_only else ''}{'teacher_forced' if args.flow_eval_mode == 'reconstruct' and not args.include_gt_context else 'rollout'}_h{horizon}",
                        "horizon": int(horizon),
                        "source_start": int(starts[i]), "source_entry_id": source_groups[i],
                        "source_style": int(source_labels[i].item()),
                        "target_style": int(source_labels[i].item()),
                        "pred_style": int(matched_cls_pred[i]),
                        "target_conf": float(matched_target_conf[i]),
                        "pred_conf": float(matched_pred_conf[i]),
                        "root_ade_cm": _root_ade_cm(ref_np[i], matched_np[i], dt=args.dt, context_frames=context_frames),
                        "latency_ms_per_sequence": float(matched_latency_ms),
                        "uses_recipe": args.recipe_only,
                        "recipe_strength": 1. if args.recipe_only else 0.,
                    }
                    matched_rows.append(row)

                if args.targets_per_source > 0:
                    source_np = source_labels.detach().cpu().numpy().astype(np.int64)
                    target_labels: list[int] = []
                    source_indices: list[int] = []
                    for i, src_label in enumerate(source_np):
                        candidates = np.arange(num_styles, dtype=np.int64)
                        candidates = candidates[candidates != int(src_label)]
                        if candidates.size == 0:
                            raise RuntimeError("Cross-style transfer requires at least two styles.")
                        replace = args.targets_per_source > candidates.size
                        if int(starts[i]) in rows_by_start:
                            sampled = np.asarray(rows_by_start[int(starts[i])]["targets"], dtype=np.int64)
                            if len(sampled) != args.targets_per_source or np.any(sampled == src_label) or np.any(sampled < 0) or np.any(sampled >= num_styles):
                                raise ValueError("Invalid manifest cross targets")
                        else:
                            sampled = rng.choice(candidates, size=args.targets_per_source, replace=replace)
                            source_rows.append({"source_start": int(starts[i]), "source_entry_id": source_groups[i],
                                "source_label": int(src_label), "targets": sampled.tolist(), "noise_seed": args.seed})
                        target_labels.extend(int(v) for v in sampled)
                        source_indices.extend([i] * args.targets_per_source)
                    source_indices_np = np.asarray(source_indices, dtype=np.int64)
                    target_t = torch.tensor(target_labels, device=device, dtype=torch.long)
                    source_index_t = torch.as_tensor(source_indices_np, device=device, dtype=torch.long)
                    x_cross = x.index_select(0, source_index_t)
                    t0 = time.perf_counter()
                    cross_pred = reconstruct_motion_batch(
                        model_name=flow_model_name,
                        model=flow_model,
                        x=x_cross,
                        style_label=None if args.recipe_only else target_t,
                        manual_style_probs=recipe_probs.index_select(0, target_t) if args.recipe_only else None,
                        flow_eval_mode=args.flow_eval_mode,
                        flow_solver=args.flow_solver,
                        flow_ode_steps=None if args.flow_ode_steps <= 0 else args.flow_ode_steps,
                        flow_noise_std=None if args.flow_noise_std < 0.0 else args.flow_noise_std,
                        flow_requantize_z=args.flow_requantize_z,
                        flow_latent_clip=args.flow_latent_clip,
                        legacy_inference_order=args.legacy_inference_order,
                        noise_plan=case_noise_plan(flow_model, starts[source_indices_np], target_labels, source_frames,
                            seed=args.seed, noise_std=flow_model.eval_noise_std if args.flow_noise_std < 0 else args.flow_noise_std),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    cross_latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(target_labels), 1)
                    cross_np = _denormalize(cross_pred.detach().cpu().numpy(), motion_mean, motion_std)
                    ref_cross_np = ref_np[source_indices_np]
                    cross_cls_pred, cross_target_conf, cross_pred_conf = _classifier_stats(
                        classifier,
                        cross_np[:, context_frames:],
                        np.asarray(target_labels, dtype=np.int64),
                        device=device,
                        classifier_window=classifier_window,
                        classifier_stride=args.classifier_stride,
                        classifier_batch_size=args.classifier_batch_size,
                        joint_set=classifier_joint_set,
                    )
                    for j, src_i in enumerate(source_indices_np):
                        row = {
                            "setting": (f"cross_style_transfer_recipe_{'teacher_forced' if args.flow_eval_mode == 'reconstruct' else 'rollout'}_h{horizon}"
                                        if args.recipe_only else f"cross_style_transfer_h{horizon}"),
                            "horizon": int(horizon),
                            "source_start": int(starts[src_i]), "source_entry_id": source_groups[src_i],
                            "source_style": int(source_np[src_i]),
                            "target_style": int(target_labels[j]),
                            "pred_style": int(cross_cls_pred[j]),
                            "target_conf": float(cross_target_conf[j]),
                            "pred_conf": float(cross_pred_conf[j]),
                            "root_ade_cm": _root_ade_cm(ref_cross_np[j], cross_np[j], dt=args.dt, context_frames=context_frames),
                            "latency_ms_per_sequence": float(cross_latency_ms),
                            "uses_recipe": args.recipe_only,
                            "recipe_strength": 1. if args.recipe_only else 0.,
                        }
                        cross_rows.append(row)

                    if args.use_target_style_recipe and not args.recipe_only:
                        if recipe_probs is None:
                            raise RuntimeError("Recipe probabilities were not initialized.")
                        manual_probs_batch = recipe_probs.index_select(0, target_t)
                        t0 = time.perf_counter()
                        cross_recipe_pred = reconstruct_motion_batch(
                            model_name=flow_model_name,
                            model=flow_model,
                            x=x_cross,
                            style_label=None if args.recipe_profile=='canonical' and args.recipe_strength==1. else target_t,
                            manual_style_probs=manual_probs_batch,
                            manual_style_strength=args.recipe_strength,
                            flow_eval_mode=args.flow_eval_mode,
                            flow_solver=args.flow_solver,
                            flow_ode_steps=None if args.flow_ode_steps <= 0 else args.flow_ode_steps,
                            flow_noise_std=None if args.flow_noise_std < 0.0 else args.flow_noise_std,
                            flow_requantize_z=args.flow_requantize_z,
                            flow_latent_clip=args.flow_latent_clip,
                            legacy_inference_order=args.legacy_inference_order,
                            noise_plan=case_noise_plan(flow_model, starts[source_indices_np], target_labels, source_frames,
                                seed=args.seed, noise_std=flow_model.eval_noise_std if args.flow_noise_std < 0 else args.flow_noise_std),
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        cross_recipe_latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(target_labels), 1)
                        cross_recipe_np = _denormalize(
                            cross_recipe_pred.detach().cpu().numpy(),
                            motion_mean,
                            motion_std,
                        )
                        (
                            cross_recipe_cls_pred,
                            cross_recipe_target_conf,
                            cross_recipe_pred_conf,
                        ) = _classifier_stats(
                            classifier,
                            cross_recipe_np[:, context_frames:],
                            np.asarray(target_labels, dtype=np.int64),
                            device=device,
                            classifier_window=classifier_window,
                            classifier_stride=args.classifier_stride,
                            classifier_batch_size=args.classifier_batch_size,
                            joint_set=classifier_joint_set,
                        )
                        for j, src_i in enumerate(source_indices_np):
                            row = {
                                "setting": f"cross_style_transfer_recipe_h{horizon}",
                                "horizon": int(horizon),
                                "source_start": int(starts[src_i]), "source_entry_id": source_groups[src_i],
                                "source_style": int(source_np[src_i]),
                                "target_style": int(target_labels[j]),
                                "pred_style": int(cross_recipe_cls_pred[j]),
                                "target_conf": float(cross_recipe_target_conf[j]),
                                "pred_conf": float(cross_recipe_pred_conf[j]),
                                "root_ade_cm": _root_ade_cm(ref_cross_np[j], cross_recipe_np[j], dt=args.dt, context_frames=context_frames),
                                "latency_ms_per_sequence": float(cross_recipe_latency_ms),
                                "uses_recipe": True,
                                "recipe_strength": float(args.recipe_strength),
                            }
                            cross_recipe_rows.append(row)

            for setting_rows in (real_rows, matched_rows, cross_rows, cross_recipe_rows):
                if not setting_rows:
                    continue
                all_rows.extend(setting_rows)
                summary = _make_summary(setting_rows[0]["setting"], setting_rows)
                summaries.append(summary)
                print(
                    f"[{summary['setting']}] N={summary['num_pairs']} "
                    f"RootADE={summary['root_ade_cm_text']} cm | "
                    f"TargetAcc={summary['target_style_acc_pct_text']}% | "
                    f"TargetConf={summary['target_conf_pct_text']}%"
                )

    for row in all_rows:
        row['context_frames'] = context_frames
        row['classification_start'] = row['source_start'] + context_frames
    summary_payload = {
        "config": {
            "db": str(args.db),
            "flow_ckpt": str(args.flow_ckpt),
            "flow_vqvae_ckpt": str(args.flow_vqvae_ckpt) if args.flow_vqvae_ckpt is not None else None,
            "classifier_ckpt": str(args.classifier_ckpt),
            "classifier_joint_set": classifier_joint_set,
            "flow_model_name": flow_model_name,
            "norm_stats": str(norm_stats_path) if norm_stats_path is not None else None,
            "horizons": [int(v) for v in args.horizons],
            "stride": args.stride,
            "num_sources": args.num_sources,
            "targets_per_source": args.targets_per_source,
            "classifier_window": classifier_window,
            "classifier_fps": classifier_fps,
            "classifier_window_seconds": classifier_window / classifier_fps,
            "include_gt_context": args.include_gt_context,
            "source_split": args.source_split,
            "conditioning_mode": getattr(flow_model, 'conditioning_mode', 'label'),
            "recipe_only": args.recipe_only,
            "flow_microbatch_size": args.flow_microbatch_size,
            "protocol_version": protocol_version,
            "classifier_stride": args.classifier_stride,
            "seed": args.seed,
            "dt": args.dt,
            "flow_eval_mode": args.flow_eval_mode,
            "flow_solver": args.flow_solver,
            "flow_ode_steps": args.flow_ode_steps,
            "flow_noise_std": args.flow_noise_std,
            "flow_requantize_z": args.flow_requantize_z,
            "effective_flow_requantize_z": getattr(flow_model, 'inference_requantize', False) if args.flow_requantize_z is None else args.flow_requantize_z,
            "flow_latent_clip": args.flow_latent_clip,
            "use_target_style_recipe": args.use_target_style_recipe,
            "recipe_source_db": str(args.recipe_source_db) if args.use_target_style_recipe else None,
            "recipe_exemplars_per_style": args.recipe_exemplars_per_style,
            "recipe_window": args.recipe_window,
            "recipe_stride": args.recipe_stride,
            "recipe_strength": args.recipe_strength,
            "recipe_seed": args.recipe_seed,
            "recipe_profile": args.recipe_profile,
            "recipe_pure_conditioning": args.use_target_style_recipe and args.recipe_profile=='canonical' and args.recipe_strength==1.,
            "recipe_meta": recipe_meta,
        },
        "summaries": summaries,
    }

    args.case_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.case_manifest.with_suffix(".json.partial")
    temporary_manifest.write_text(json.dumps(args.case_data, ensure_ascii=False, indent=2))
    temporary_manifest.replace(args.case_manifest)
    summary_path = args.out_dir / "minor_revision_generalization_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2, default=_json_default)

    summary_csv = args.out_dir / "minor_revision_generalization_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            extrasaction="ignore",
            fieldnames=[
                "setting",
                "num_pairs",
                "uses_recipe",
                "recipe_strength",
                "root_ade_cm",
                "root_ade_cm_ci95",
                "target_style_acc_pct",
                "target_style_acc_pct_ci95",
                "target_conf_pct",
                "target_conf_pct_ci95",
                "pred_conf_pct",
                "root_ade_cm_text",
                "target_style_acc_pct_text",
                "target_conf_pct_text",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    details_csv = args.out_dir / "minor_revision_generalization_per_sample.csv"
    detail_fields = [
        "source_entry_id",
        "setting",
        "horizon",
        "source_start",
        "classification_start",
        "context_frames",
        "source_style",
        "target_style",
        "pred_style",
        "target_conf",
        "pred_conf",
        "root_ade_cm",
        "latency_ms_per_sequence",
        "uses_recipe",
        "recipe_strength",
    ]
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[Done] summary json : {summary_path}")
    print(f"[Done] summary csv  : {summary_csv}")
    print(f"[Done] details csv  : {details_csv}")


if __name__ == "__main__":
    main()
