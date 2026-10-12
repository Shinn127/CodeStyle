"""
Build a VQ style-recipe JSON from a zero-shot exemplar clip.

The generated recipe can be passed directly to:
  run_simulate_realtime_flowmatch_control.sh  via STYLE_RECIPE=...

Workflow:
  1) resolve the VQ encoder / norm stats / selected style slots from a flow config
  2) pick a zero-shot exemplar segment from the memmap DB
  3) encode the exemplar frames into VQ indices
  4) keep the top-k codes per selected slot and export them as a style recipe
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.realtime import parse_style_selector
from Utils.runtime import (
    MEMMAP_META_FILE,
    MEMMAP_STYLES_FILE,
    MEMMAP_X_FILE,
    configure_torch_runtime,
    load_checkpoint,
    load_norm_stats,
    resolve_device,
    set_seed,
    validate_memmap_db_dir,
    sanitize_state_dict_keys,
)
from Utils.style_code_prior import (load_style_code_prior, sparse_code_histogram,
    recipe_probabilities_from_indices, recipe_slots_from_probs, recipe_conditioning_metadata)
from Utils.reproducibility import state_fingerprint
from dataset import norm_stats_fingerprint


MEMMAP_SEGMENTS_FILE = "segments.json"
SCRIPT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FLOW_CKPT = SCRIPT_DIR / "checkpoints" / "flowmatch_tf150_staged_noise" / "best.pt"
DEFAULT_ZEROSHOT_DB = SCRIPT_DIR / "database_100sty_375_memmap_zeroshot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a zero-shot style recipe from a memmap exemplar clip")
    parser.add_argument("--db", type=Path, default=DEFAULT_ZEROSHOT_DB, help="Zero-shot exemplar memmap DB")
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        default=DEFAULT_FLOW_CKPT,
        help="VQ-based flow checkpoint used for recipe playback; used to resolve defaults",
    )
    parser.add_argument(
        "--flow-config",
        type=Path,
        default=None,
        help="Optional config.json override. Default: sibling of --flow-ckpt when available",
    )
    parser.add_argument(
        "--style-code-prior",
        type=Path,
        default=None,
        help="Optional style code prior (.npz) used to resolve selected slots",
    )
    parser.add_argument("--vqvae-ckpt", type=Path, default=None, help="Optional frozen VQ checkpoint override")
    parser.add_argument("--norm-stats", type=Path, default=None, help="Optional normalization stats override")
    parser.add_argument(
        "--selected-slots",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit slot list override; bypasses flow-config/style-prior discovery",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="Optional style selector in the zero-shot DB. Empty means all segments are eligible",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help="Optional global index into segments.json. If omitted, the script picks one automatically",
    )
    parser.add_argument(
        "--pick",
        type=str,
        choices=["longest", "first", "random"],
        default="longest",
        help="Selection rule used when --segment-index is omitted",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=240,
        help="Frames used from the exemplar segment; 0 means the whole segment",
    )
    parser.add_argument(
        "--crop",
        type=str,
        choices=["start", "center", "random"],
        default="center",
        help="How to crop the exemplar segment when --num-frames is shorter than the clip",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride applied inside the cropped exemplar clip")
    parser.add_argument("--topk", type=int, default=4, help="Top-k codes kept for each controllable slot")
    parser.add_argument("--strength", type=float, default=1.0, help="Recipe strength written into the JSON")
    parser.add_argument(
        "--base-style",
        type=str,
        default="",
        help="Optional seen-style anchor written into the recipe as base_style",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path. Default: auto under cache/")
    parser.add_argument("--chunk-rows", type=int, default=4096, help="Frames encoded per VQ chunk")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA")
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default="highest",
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    return parser.parse_args()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()
    return slug or "recipe"


def _resolve_saved_path(path_like: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_flow_config_dict(
    *,
    flow_ckpt: Path | None,
    flow_config: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if flow_config is not None:
        resolved = flow_config.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--flow-config not found: {resolved}")
        payload = _load_json(resolved)
        if not isinstance(payload, dict):
            raise ValueError(f"{resolved} must contain a JSON object")
        return payload, resolved

    if flow_ckpt is not None:
        ckpt = load_checkpoint(flow_ckpt, map_location="cpu")
        payload = ckpt.get("args", {}) or {}
        if payload:
            if not isinstance(payload, dict):
                raise ValueError(f"Checkpoint args must be a dict: {flow_ckpt}")
            return payload, None
        sibling = flow_ckpt.parent / "config.json"
        if sibling.exists():
            payload = _load_json(sibling)
            if not isinstance(payload, dict):
                raise ValueError(f"{sibling} must contain a JSON object")
            return payload, sibling

        ckpt = load_checkpoint(flow_ckpt, map_location="cpu")
        payload = ckpt.get("args", {}) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint args must be a dict: {flow_ckpt}")
        return payload, None

    return {}, None


def _load_style_names(db_path: Path) -> list[str]:
    style_names_path = db_path / MEMMAP_STYLES_FILE
    if not style_names_path.exists():
        raise FileNotFoundError(f"Style names not found: {style_names_path}")
    payload = _load_json(style_names_path)
    if not isinstance(payload, list):
        raise ValueError(f"{style_names_path} must contain a JSON list")
    return [str(name) for name in payload]


def _load_segments(db_path: Path) -> list[dict[str, Any]]:
    segments_path = db_path / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        raise FileNotFoundError(f"segments.json not found: {segments_path}")
    payload = _load_json(segments_path)
    if not isinstance(payload, list):
        raise ValueError(f"{segments_path} must contain a JSON list")
    segments: list[dict[str, Any]] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid segment at index {idx}: expected object, got {type(raw)}")
        if "frame_start" not in raw or "num_frames" not in raw or "label" not in raw:
            raise KeyError(f"Segment {idx} must contain frame_start, num_frames, label")
        segments.append(raw)
    return segments


def _resolve_selected_slots(
    *,
    args: argparse.Namespace,
    flow_config: dict[str, Any],
    flow_config_path: Path | None,
    flow_ckpt: Path | None,
) -> tuple[int, ...]:
    if args.selected_slots:
        slots = tuple(int(slot) for slot in args.selected_slots)
    else:
        style_code_prior_path = args.style_code_prior
        if style_code_prior_path is None:
            raw = flow_config.get("style_code_prior", None)
            if raw is not None:
                base_dir = flow_config_path.parent if flow_config_path is not None else (flow_ckpt.parent if flow_ckpt is not None else Path.cwd())
                style_code_prior_path = _resolve_saved_path(raw, base_dir=base_dir)
        if style_code_prior_path is not None:
            prior_payload = load_style_code_prior(style_code_prior_path)
            slots = tuple(int(slot) for slot in np.asarray(prior_payload["selected_slots"], dtype=np.int64).tolist())
        else:
            slots = tuple(int(slot) for slot in flow_config.get("style_slot_indices", []))

    if not slots:
        raise ValueError(
            "Could not resolve controllable style slots. "
            "Pass --selected-slots explicitly or use a flow config/checkpoint trained with --use-style-code-selector."
        )
    if len(set(slots)) != len(slots):
        raise ValueError(f"selected slots contain duplicates: {slots}")
    if any(slot < 0 for slot in slots):
        raise ValueError(f"selected slots must be non-negative, got {slots}")

    if "use_style_code_selector" in flow_config and not bool(flow_config["use_style_code_selector"]):
        raise ValueError(
            "The resolved flow config disables use_style_code_selector, so the generated recipe would not be accepted "
            "by simulate_realtime_flowmatch_control.py."
        )
    return slots


def _resolve_vqvae_ckpt(
    *,
    args: argparse.Namespace,
    flow_config: dict[str, Any],
    flow_config_path: Path | None,
    flow_ckpt: Path | None,
) -> Path:
    if args.vqvae_ckpt is not None:
        path = args.vqvae_ckpt.expanduser().resolve()
    else:
        raw = flow_config.get("vqvae_ckpt", None)
        if raw is None:
            raise ValueError("Could not resolve vqvae_ckpt. Pass --vqvae-ckpt explicitly.")
        base_dir = flow_config_path.parent if flow_config_path is not None else (flow_ckpt.parent if flow_ckpt is not None else Path.cwd())
        path = _resolve_saved_path(raw, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"VQ checkpoint not found: {path}")
    return path


def _resolve_norm_stats(
    *,
    args: argparse.Namespace,
    flow_config: dict[str, Any],
    flow_config_path: Path | None,
    flow_ckpt: Path | None,
    vqvae_ckpt: Path,
) -> Path:
    if args.norm_stats is not None:
        path = args.norm_stats.expanduser().resolve()
    else:
        raw = flow_config.get("norm_stats", None)
        if raw is not None:
            base_dir = flow_config_path.parent if flow_config_path is not None else (flow_ckpt.parent if flow_ckpt is not None else Path.cwd())
            path = _resolve_saved_path(raw, base_dir=base_dir)
        else:
            path = (vqvae_ckpt.parent / "norm_stats.npz").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {path}")
    return path


def _filter_segments_by_style(
    segments: list[dict[str, Any]],
    *,
    style_label: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    if style_label is None:
        return [(idx, segment) for idx, segment in enumerate(segments)]
    return [
        (idx, segment)
        for idx, segment in enumerate(segments)
        if int(segment["label"]) == int(style_label)
    ]


def _pick_segment(
    *,
    segments: list[dict[str, Any]],
    style_label: int | None,
    segment_index: int | None,
    pick: str,
    seed: int,
) -> tuple[int, dict[str, Any], int]:
    candidates = _filter_segments_by_style(segments, style_label=style_label)
    if not candidates:
        raise RuntimeError(f"No segments available for style_label={style_label}")

    if segment_index is not None:
        if segment_index < 0 or segment_index >= len(segments):
            raise IndexError(
                f"--segment-index out of range for segments.json: got {segment_index}, "
                f"valid [0, {len(segments) - 1}]"
            )
        segment = segments[segment_index]
        if style_label is not None and int(segment["label"]) != int(style_label):
            raise ValueError(
                f"--segment-index={segment_index} has label={int(segment['label'])}, "
                f"which does not match --style label={int(style_label)}"
            )
        filtered_index = next(
            idx for idx, (global_index, _) in enumerate(candidates) if int(global_index) == int(segment_index)
        )
        return int(segment_index), segment, filtered_index

    if pick == "first":
        filtered_index = 0
    elif pick == "longest":
        filtered_index = max(
            range(len(candidates)),
            key=lambda idx: (int(candidates[idx][1]["num_frames"]), -idx),
        )
    else:
        rng = np.random.default_rng(seed)
        filtered_index = int(rng.integers(0, len(candidates)))
    global_index, segment = candidates[filtered_index]
    return global_index, segment, filtered_index


def _resolve_crop(
    *,
    segment_start: int,
    segment_frames: int,
    wanted_frames: int,
    crop: str,
    seed: int,
) -> tuple[int, int, int]:
    if wanted_frames <= 0 or wanted_frames >= segment_frames:
        return segment_start, segment_frames, 0

    max_offset = segment_frames - wanted_frames
    if crop == "start":
        offset = 0
    elif crop == "center":
        offset = max_offset // 2
    else:
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(0, max_offset + 1))
    return segment_start + offset, wanted_frames, offset


@torch.no_grad()
def _encode_vq_indices(
    *,
    clip_norm: np.ndarray,
    vqvae,
    device: torch.device,
    chunk_rows: int,
) -> np.ndarray:
    if clip_norm.ndim != 2 or clip_norm.shape[1] != 375:
        raise ValueError(f"Expected normalized clip shape [T,375], got {clip_norm.shape}")
    if clip_norm.shape[0] < 1:
        raise ValueError("Exemplar clip has zero frames after cropping / frame-stride")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    parts: list[np.ndarray] = []
    for start in range(0, clip_norm.shape[0], chunk_rows):
        end = min(start + chunk_rows, clip_norm.shape[0])
        x_t = torch.from_numpy(np.asarray(clip_norm[start:end], dtype=np.float32)).to(device=device)
        _, indices_t, _ = vqvae.encode(x_t)
        parts.append(indices_t.detach().cpu().numpy().astype(np.int64, copy=False))
    return np.concatenate(parts, axis=0)


def _build_recipe_slots(
    *,
    indices: np.ndarray,
    selected_slots: tuple[int, ...],
    num_embeddings: int,
    topk: int,
) -> list[dict[str, Any]]:
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    if indices.ndim != 2:
        raise ValueError(f"indices must have rank 2 [T,M], got {indices.shape}")
    if indices.shape[0] < 1:
        raise ValueError("indices must contain at least one frame")
    if indices.shape[1] <= max(selected_slots):
        raise ValueError(
            f"Selected slot {max(selected_slots)} is out of range for num_codebooks={indices.shape[1]}"
        )

    probabilities = sparse_code_histogram(indices, selected_slots=selected_slots,
        num_embeddings=num_embeddings, topk=topk)
    slot_entries: list[dict[str, Any]] = []
    for local, slot in enumerate(selected_slots):
        probs = probabilities[local]
        ranked = np.argsort(-probs, kind='stable')
        ranked = ranked[probs[ranked] > 0]
        weights = probs[ranked]
        slot_entries.append(
            {
                "slot": int(slot),
                "codes": [int(code) for code in ranked.tolist()],
                "weights": [float(weight) for weight in weights.tolist()],
            }
        )
    return slot_entries


def _default_out_path(
    *,
    db_path: Path,
    segment: dict[str, Any],
    global_segment_index: int,
    frame_start: int,
    frame_stop: int,
    topk: int,
) -> Path:
    style_name = str(segment.get("style_name", f"label_{int(segment['label'])}"))
    style_slug = _slugify(style_name)
    file_name = (
        f"{style_slug}_seg{global_segment_index:03d}_"
        f"frames{frame_start}-{frame_stop}_top{topk}.json"
    )
    return (SCRIPT_DIR / "cache" / "zeroshot_style_recipes" / file_name).resolve()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve() if args.flow_ckpt is not None else None
    args.flow_config = args.flow_config.expanduser().resolve() if args.flow_config is not None else None
    args.style_code_prior = args.style_code_prior.expanduser().resolve() if args.style_code_prior is not None else None
    args.vqvae_ckpt = args.vqvae_ckpt.expanduser().resolve() if args.vqvae_ckpt is not None else None
    args.norm_stats = args.norm_stats.expanduser().resolve() if args.norm_stats is not None else None
    args.out = args.out.expanduser().resolve() if args.out is not None else None

    validate_memmap_db_dir(args.db)
    if args.flow_ckpt is not None and not args.flow_ckpt.exists():
        raise FileNotFoundError(f"--flow-ckpt not found: {args.flow_ckpt}")
    if args.num_frames < 0:
        raise ValueError(f"--num-frames must be >= 0, got {args.num_frames}")
    if args.frame_stride < 1:
        raise ValueError(f"--frame-stride must be >= 1, got {args.frame_stride}")
    if not (0.0 <= args.strength <= 1.0):
        raise ValueError(f"--strength must lie in [0,1], got {args.strength}")
    if args.strength != 1.0 and not args.base_style.strip():
        raise ValueError('Pure recipes require strength=1; specify a seen base-style for partial mixing')

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    flow_config, flow_config_path = _resolve_flow_config_dict(
        flow_ckpt=args.flow_ckpt,
        flow_config=args.flow_config,
    )
    if flow_config.get('conditioning_mode') == 'clip_recipe' and args.flow_ckpt is None:
        raise ValueError('Learned recipe export requires --flow-ckpt, not only its config')
    selected_slots = _resolve_selected_slots(
        args=args,
        flow_config=flow_config,
        flow_config_path=flow_config_path,
        flow_ckpt=args.flow_ckpt,
    )
    vqvae_ckpt = _resolve_vqvae_ckpt(
        args=args,
        flow_config=flow_config,
        flow_config_path=flow_config_path,
        flow_ckpt=args.flow_ckpt,
    )
    norm_stats = _resolve_norm_stats(
        args=args,
        flow_config=flow_config,
        flow_config_path=flow_config_path,
        flow_ckpt=args.flow_ckpt,
        vqvae_ckpt=vqvae_ckpt,
    )
    mean, std = load_norm_stats(norm_stats)

    style_names = _load_style_names(args.db)
    style_label = parse_style_selector(args.style, style_names) if args.style.strip() else None
    segments = _load_segments(args.db)
    global_segment_index, segment, filtered_segment_index = _pick_segment(
        segments=segments,
        style_label=style_label,
        segment_index=args.segment_index,
        pick=args.pick,
        seed=args.seed,
    )

    with (args.db / MEMMAP_META_FILE).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    total_frames = int(meta.get("num_frames", -1))
    feature_dim = int(meta.get("feature_dim", -1))
    if total_frames <= 0 or feature_dim != 375:
        raise ValueError(f"Invalid memmap metadata in {args.db / MEMMAP_META_FILE}: num_frames={total_frames}, feature_dim={feature_dim}")

    segment_start = int(segment["frame_start"])
    segment_frames = int(segment["num_frames"])
    frame_start, used_frames, crop_offset = _resolve_crop(
        segment_start=segment_start,
        segment_frames=segment_frames,
        wanted_frames=int(args.num_frames),
        crop=args.crop,
        seed=args.seed,
    )
    frame_stop = frame_start + used_frames
    frame_ids = np.arange(frame_start, frame_stop, args.frame_stride, dtype=np.int64)
    if frame_ids.size == 0:
        raise RuntimeError("No exemplar frames remain after applying --frame-stride")

    x_source = np.memmap(args.db / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(total_frames, 375))
    clip_raw = np.asarray(x_source[frame_ids], dtype=np.float32)
    clip_norm = (clip_raw - mean[None, :]) / std[None, :]

    vqvae = build_vqvae_from_checkpoint(vqvae_ckpt).to(device)
    vqvae.eval()
    if args.flow_ckpt is not None:
        checkpoint = load_checkpoint(args.flow_ckpt, map_location='cpu')
        state = sanitize_state_dict_keys(checkpoint['model'])
        bundled = {k[6:]: v for k, v in state.items() if k.startswith('vqvae.')}
        if not bundled or state_fingerprint(bundled) != state_fingerprint(vqvae.state_dict()):
            raise ValueError('Recipe VQ override differs from the controller bundled VQ')
        saved_args = checkpoint.get('args', {})
        slots_in_state = state.get('style_slot_indices', saved_args.get('style_slot_indices', []))
        if tuple(int(s) for s in slots_in_state) != selected_slots:
            raise ValueError('Recipe selected slots/order differ from the controller')
        saved_norm = saved_args.get('fingerprints', {}).get('norm_fingerprint')
        if saved_norm is not None and saved_norm != norm_stats_fingerprint(mean, std):
            raise ValueError('Recipe normalization differs from the controller/VQ manifold')
    indices = _encode_vq_indices(
        clip_norm=clip_norm,
        vqvae=vqvae,
        device=device,
        chunk_rows=args.chunk_rows,
    )
    slot_entries = _build_recipe_slots(
        indices=indices,
        selected_slots=selected_slots,
        num_embeddings=int(vqvae.quantizer.num_embeddings),
        topk=args.topk,
    )
    conditioning_metadata = dict(probability_space='control', recipe_source='topk_histogram')
    if args.flow_ckpt is not None and saved_args.get('conditioning_mode') == 'clip_recipe':
        from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
        if args.topk != int(saved_args.get('style_selector_topk', 4)):
            raise ValueError('Learned export TopK must match the controller checkpoint')
        flow_model = build_flowmatch_controller_from_config(saved_args, vqvae=vqvae,
            num_styles=int(saved_args['num_styles'])).to(device)
        flow_model.load_state_dict(state, strict=True)
        flow_model.eval()
        probs = recipe_probabilities_from_indices(indices, flow_model)
        slot_entries = recipe_slots_from_probs(probs, selected_slots)
        conditioning_metadata = recipe_conditioning_metadata(flow_model)

    out_path = args.out
    if out_path is None:
        out_path = _default_out_path(
            db_path=args.db,
            segment=segment,
            global_segment_index=global_segment_index,
            frame_start=frame_start,
            frame_stop=frame_stop,
            topk=args.topk,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    style_name = str(segment.get("style_name", style_names[int(segment["label"])] if style_names else f"label_{int(segment['label'])}"))
    recipe: dict[str, Any] = {
        "strength": float(args.strength),
        "slots": slot_entries,
        "metadata": {
            **conditioning_metadata,
            "source_db": str(args.db),
            "style_name": style_name,
            "style_label": int(segment["label"]),
            "global_segment_index": int(global_segment_index),
            "filtered_segment_index": int(filtered_segment_index),
            "source_path": str(segment.get("source_path", "")),
            "segment_frame_start": int(segment_start),
            "segment_num_frames": int(segment_frames),
            "crop_frame_start": int(frame_start),
            "crop_frame_stop": int(frame_stop),
            "crop_offset": int(crop_offset),
            "frames_used": int(frame_ids.size),
            "frame_stride": int(args.frame_stride),
            "pick": "segment_index" if args.segment_index is not None else str(args.pick),
            "crop": str(args.crop),
            "topk": int(args.topk),
            "selected_slots": [int(slot) for slot in selected_slots],
            "flow_ckpt": None if args.flow_ckpt is None else str(args.flow_ckpt),
            "flow_config": None if flow_config_path is None else str(flow_config_path),
            "vqvae_ckpt": str(vqvae_ckpt),
            "vq_state_sha256": state_fingerprint(vqvae.state_dict()),
            "norm_stats": str(norm_stats),
            "norm_fingerprint": norm_stats_fingerprint(mean, std),
        },
    }
    if args.base_style.strip():
        recipe["base_style"] = str(args.base_style)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    print(f"[Info] device          : {device}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] flow ckpt       : {args.flow_ckpt}")
    if flow_config_path is not None:
        print(f"[Info] flow config     : {flow_config_path}")
    print(f"[Info] vq ckpt         : {vqvae_ckpt}")
    print(f"[Info] norm stats      : {norm_stats}")
    print(f"[Info] style name      : {style_name}")
    print(f"[Info] segment idx     : global={global_segment_index} filtered={filtered_segment_index}")
    print(f"[Info] crop frames     : {frame_start}:{frame_stop} stride={args.frame_stride} used={frame_ids.size}")
    print(f"[Info] selected slots  : {list(selected_slots)}")
    print(f"[Info] topk            : {args.topk}")
    if args.base_style.strip():
        print(f"[Info] base style      : {args.base_style}")
    print(f"[Done] recipe          : {out_path}")


if __name__ == "__main__":
    main()
