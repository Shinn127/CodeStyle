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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.fid_feature_extractor import (
    JOINT_SET_SELECTED24,
    JOINT_SETS,
    build_fid_feature_extractor_from_checkpoint,
    motion375_clip_to_classifier_input,
)
from Utils.motion_eval import (
    MotionEvalConfig,
    MotionModelName,
    camdm_clip_fid,
    camdm_style_fid,
    compute_sequence_metric_sums,
    frechet_distance,
    denormalize_motion_batch,
    load_motion_model,
    reconstruct_motion_batch,
    smooth_motion_block,
)
from Utils.runtime import (
    load_checkpoint,
    load_norm_stats,
    resolve_device,
    save_norm_stats_if_present,
    set_seed,
    validate_memmap_db_dir,
)
from Utils.runtime import subset_dataset_by_fraction
from dataset import build_train_val_test_datasets


from Utils.reproducibility import case_noise_plan, cluster_interval

SCRIPT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CKPTS = {
    "basic_vqvae": SCRIPT_DIR / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror" / "best.pt",
    "camdm": SCRIPT_DIR / "checkpoints" / "camdm_controller" / "best.pt",
    "diffusion_lmm_x0": SCRIPT_DIR / "checkpoints" / "diffusion_lmm_controller" / "best.pt",
    "diffusion_vqvae_x0": SCRIPT_DIR / "checkpoints" / "diffusion_vqvae_controller" / "best.pt",
    "flowmatch_velocity": SCRIPT_DIR / "checkpoints" / "flowmatch_vqvae_controller_velocity_head10" / "best.pt",
    "flowmatch_target": SCRIPT_DIR / "checkpoints" / "flowmatch_vqvae_controller_target_head10" / "best.pt",
    "flowmatch_lmm_velocity": SCRIPT_DIR / "checkpoints" / "flowmatch_lmm_controller_velocity_head10" / "best.pt",
    "lmm": SCRIPT_DIR / "checkpoints" / "lmm_375to291_head10" / "best.pt",
    "mann": SCRIPT_DIR / "checkpoints" / "mann_controller" / "best.pt",
    "mvae": SCRIPT_DIR / "checkpoints" / "mvae_375to291_head10" / "best.pt",
}


def warn_if_checkpoint_db_differs(
    model_names: list[MotionModelName],
    ckpt_paths: list[Path],
    eval_db: Path,
) -> None:
    """Flag checkpoints that were trained on a different database than the one evaluated.

    The retired `*_head10` baselines are still the defaults for some models; evaluated on the
    90-style database they are out of distribution and their numbers are not comparable.
    """
    eval_db_name = eval_db.name
    for model_name, ckpt_path in zip(model_names, ckpt_paths, strict=True):
        try:
            ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        except Exception:  # noqa: BLE001 - a missing/unreadable ckpt is reported elsewhere
            continue
        train_db = str((ckpt.get("args") or {}).get("db", "") or "")
        if not train_db:
            continue
        train_db_name = Path(train_db).name
        if train_db_name and train_db_name != eval_db_name:
            print(
                f"[Warn] {model_name}: checkpoint was trained on '{train_db_name}' but is being "
                f"evaluated on '{eval_db_name}' - the metrics are out of distribution and not "
                f"comparable with the other rows"
            )


def _resolve_norm_stats_path(
    args: argparse.Namespace,
    model_names: list[MotionModelName],
    ckpt_paths: list[Path],
) -> Path | None:
    if args.norm_stats is not None:
        return Path(args.norm_stats).expanduser().resolve()
    for model_name, ckpt_path in zip(model_names, ckpt_paths, strict=True):
        if model_name in {
            "flowmatch_velocity",
            "flowmatch_target",
            "flowmatch_lmm_velocity",
            "flowmatch_lmm_target",
            "diffusion_vqvae_x0",
            "diffusion_lmm_x0",
        }:
            flow_ckpt = load_checkpoint(ckpt_path, map_location="cpu")
            flow_args = flow_ckpt.get("args", {}) or {}
            norm_stats_raw = flow_args.get("norm_stats", None)
            if norm_stats_raw is not None:
                candidate = Path(norm_stats_raw).expanduser()
                if not candidate.is_absolute():
                    candidate = (ckpt_path.parent / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if candidate.exists():
                    return candidate
            base_ckpt_key = (
                "vqvae_ckpt"
                if model_name in {"flowmatch_velocity", "flowmatch_target", "diffusion_vqvae_x0"}
                else "lmm_ckpt"
            )
            base_raw = flow_args.get(base_ckpt_key, None)
            if base_raw is not None:
                base_candidate = Path(base_raw).expanduser()
                if not base_candidate.is_absolute():
                    base_candidate = (ckpt_path.parent / base_candidate).resolve()
                else:
                    base_candidate = base_candidate.resolve()
                candidate = base_candidate.parent / "norm_stats.npz"
                if candidate.exists():
                    return candidate.resolve()
            continue
        candidate = ckpt_path.parent / "norm_stats.npz"
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_ckpt_path(model_name: str, args: argparse.Namespace) -> Path:
    explicit = getattr(args, f"{model_name}_ckpt")
    if explicit is not None:
        ckpt = Path(explicit)
    elif model_name in DEFAULT_CKPTS:
        ckpt = DEFAULT_CKPTS[model_name]
    else:
        raise ValueError(f"Checkpoint must be provided for {model_name} via --{model_name.replace('_', '-')}-ckpt")
    ckpt = ckpt.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt}")
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified motion evaluation for 375-to-291 models")
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "eval_results" / "motion_375to291")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "basic_vqvae",
            "camdm",
            "diffusion_lmm_x0",
            "diffusion_vqvae_x0",
            "lmm",
            "mann",
            "mvae",
            "flowmatch_velocity",
            "flowmatch_target",
            "flowmatch_lmm_velocity",
            "flowmatch_lmm_target",
        ],
        default=["basic_vqvae", "lmm", "mvae"],
        help="Model(s) to evaluate",
    )
    parser.add_argument("--vq-eval-path", choices=["pre_vq", "quantized"], default="quantized")
    parser.add_argument("--legacy-inference-order", action="store_true")
    parser.add_argument("--basic-vqvae-ckpt", type=Path, default=None)
    parser.add_argument("--camdm-ckpt", type=Path, default=None)
    parser.add_argument("--diffusion-lmm-x0-ckpt", type=Path, default=None)
    parser.add_argument("--diffusion-vqvae-x0-ckpt", type=Path, default=None)
    parser.add_argument("--lmm-ckpt", type=Path, default=None)
    parser.add_argument("--mann-ckpt", type=Path, default=None)
    parser.add_argument("--mvae-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-velocity-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-target-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-lmm-velocity-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-lmm-target-ckpt", type=Path, default=None)
    parser.add_argument("--flow-vqvae-ckpt", type=Path, default=None, help="Optional shared VQ checkpoint override for flowmatch models")
    parser.add_argument("--flow-lmm-ckpt", type=Path, default=None, help="Optional shared LMM checkpoint override for LMM flowmatch models")
    parser.add_argument("--norm-stats", type=Path, default=None, help="Optional shared norm_stats.npz path")
    parser.add_argument("--metric-start-frame", type=int, default=0, help="Start frame (inclusive) for metrics")
    parser.add_argument("--metric-end-frame", type=int, default=0, help="End frame (exclusive); 0 means all frames")
    parser.add_argument("--foot-contact-height-cm", type=float, default=20.0)
    parser.add_argument("--foot-contact-speed-cm-s", type=float, default=20.0)
    parser.add_argument("--velocity-dir-min-speed-cm-s", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--fid-eps", type=float, default=1e-6)
    parser.add_argument(
        "--decode-smooth-taps",
        type=int,
        default=1,
        help=(
            "Binomial temporal filter taps applied to the decoded 291-dim motion block "
            "before metrics/FID (odd, 1 disables). 3 = [1,2,1]/4, 5 = [1,4,6,4,1]/16."
        ),
    )
    parser.add_argument("--fid-extractor-ckpt", type=Path, default=None, help="Optional classifier checkpoint for FID features")
    parser.add_argument("--fid-bootstrap-samples", type=int, default=100, help="Bootstrap samples for FID std; <=1 disables")
    parser.add_argument(
        "--fid-camdm-aggregation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also report CAMDM's per-style FID aggregation (Fréchet per style group divided by "
            "the feature dimension, plus the per-clip variant of their exp1 protocol)."
        ),
    )
    parser.add_argument(
        "--fid-joint-set",
        type=str,
        choices=("auto", JOINT_SET_SELECTED24, "ybot15"),
        default="auto",
        help=(
            "Which joints the FID extractor consumes. 'auto' (default) reads the checkpoint's "
            "joint_set/in_feat: 72 dims = this project's 24 joints, 45 dims = CAMDM's 15 ybot joints."
        ),
    )
    parser.add_argument(
        "--allow-legacy-quantizer",
        action="store_true",
        help=(
            "Allow loading VQ checkpoints whose codebook uses the pre-library quantizer keys "
            "(quantizer.embedding.weight / quantizer.ema_*). Only the autoencoder path "
            "(encode_pre_vq -> decode) is meaningful for those checkpoints."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=123)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-windows-for-stats", type=int, default=0)
    parser.add_argument("--stats-windows-chunk-size", type=int, default=4096)
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--flow-eval-mode", type=str, choices=["rollout", "reconstruct"], default="rollout")
    parser.add_argument("--flow-solver", type=str, choices=["euler"], default="euler")
    parser.add_argument("--infer-steps", type=int, default=0, help="0 means infer from checkpoint args")
    parser.add_argument("--flow-ode-steps", type=int, default=0, help="0 means infer from flow checkpoint args")
    parser.add_argument("--flow-noise-std", type=float, default=-1.0, help="<0 means infer from flow checkpoint args")
    parser.add_argument("--flow-requantize-z", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flow-latent-clip", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _evaluate_one_model(
    *,
    model_name: str,
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    cfg: MotionEvalConfig,
    motion_mean: np.ndarray | None,
    motion_std: np.ndarray | None,
    fid_extractor: torch.nn.Module | None = None,
    fid_bootstrap_samples: int = 100,
    fid_bootstrap_seed: int = 42,
    num_styles: int | None = None,
    flow_vqvae_ckpt: Path | None = None,
    flow_lmm_ckpt: Path | None = None,
    flow_eval_mode: str = "rollout",
    flow_solver: str = "euler",
    flow_ode_steps: int = 0,
    flow_noise_std: float = -1.0,
    flow_requantize_z: bool | None = None,
    flow_latent_clip: bool | None = None,
    allow_legacy_quantizer: bool = False,
    decode_smooth_taps: int = 1,
    fid_joint_set: str = JOINT_SET_SELECTED24,
    fid_camdm_aggregation: bool = True,
    vq_eval_path="quantized", legacy_inference_order=False,
) -> dict[str, float | int | str]:
    model = load_motion_model(
        model_name=model_name,
        ckpt_path=ckpt_path,
        num_styles=num_styles,
        flow_vqvae_ckpt=flow_vqvae_ckpt,
        flow_lmm_ckpt=flow_lmm_ckpt,
        allow_legacy_quantizer=bool(allow_legacy_quantizer),
    ).to(device)
    model.eval()
    if getattr(model, "eval_fingerprints", None):
        from dataset import norm_stats_fingerprint
        if motion_mean is None or model.eval_fingerprints["norm_fingerprint"] != norm_stats_fingerprint(motion_mean, motion_std):
            raise ValueError("Evaluation normalization differs from the controller VQ normalization")

    mpjpe_sum = 0.0
    mpjpe_count = 0
    root_ade_sum = 0.0
    root_ade_count = 0
    heading_error_sum = 0.0
    heading_error_count = 0
    velocity_dir_error_sum = 0.0
    velocity_dir_error_count = 0
    foot_sum = 0.0
    foot_count = 0
    world_sum = 0.0
    world_count = 0
    possible_contacts = 0
    case_manifest = []
    source_groups = []
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    segments = getattr(dataset, "segments", None)
    jitter_sum = 0.0
    jitter_count = 0
    real_fid_features: list[np.ndarray] = []
    pred_fid_features: list[np.ndarray] = []
    fid_labels: list[int] = []
    fid_segments: list[int] = []
    per_sequence_mpjpe: list[float] = []
    per_sequence_root_ade: list[float] = []
    per_sequence_heading_error: list[float] = []
    per_sequence_velocity_dir_error: list[float] = []
    per_sequence_foot_sliding: list[float] = []
    per_sequence_foot_contacts: list[int] = []
    per_sequence_world_sliding: list[float] = []
    per_sequence_world_contacts: list[int] = []
    per_sequence_jitter: list[float] = []
    per_sequence_latency_ms: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            x = batch["x"].to(device=device, non_blocking=True).float()
            style_label = batch["label"].to(device=device, non_blocking=True).long()
            batch_segments = batch.get("segment_id")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred = reconstruct_motion_batch(
                model_name=model_name,
                model=model,
                x=x,
                style_label=style_label,
                flow_eval_mode=flow_eval_mode,
                flow_solver=flow_solver,
                flow_ode_steps=None if flow_ode_steps <= 0 else flow_ode_steps,
                flow_noise_std=None if flow_noise_std < 0.0 else flow_noise_std,
                flow_requantize_z=flow_requantize_z,
                flow_latent_clip=flow_latent_clip, vq_eval_path=vq_eval_path,
                legacy_inference_order=legacy_inference_order,
                noise_plan=case_noise_plan(model, batch["start"].tolist(), style_label.cpu().tolist(),
                    x.size(1), seed=fid_bootstrap_seed,
                    noise_std=model.eval_noise_std if flow_noise_std < 0 else flow_noise_std)
                    if model_name in ("flowmatch_target", "flowmatch_velocity") else None,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_latency_ms = (time.perf_counter() - t0) * 1000.0
            per_sequence_latency_ms.extend([batch_latency_ms / max(int(x.size(0)), 1)] * int(x.size(0)))
            if decode_smooth_taps > 1:
                pred = smooth_motion_block(pred, decode_smooth_taps)
            ref_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
            pred_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            ref_np = denormalize_motion_batch(ref_np, mean=motion_mean, std=motion_std)
            pred_np = denormalize_motion_batch(pred_np, mean=motion_mean, std=motion_std)

            for seq_idx, (ref_seq, pred_seq) in enumerate(zip(ref_np, pred_np, strict=True)):
                seq_label = style_label[seq_idx]
                sums = compute_sequence_metric_sums(
                    reference_375=ref_seq,
                    prediction_375=pred_seq,
                    cfg=cfg,
                )
                mpjpe_sum += sums.mpjpe_sum_cm
                mpjpe_count += sums.mpjpe_count
                root_ade_sum += sums.root_ade_sum_cm
                root_ade_count += sums.root_ade_count
                heading_error_sum += sums.heading_error_sum_deg
                heading_error_count += sums.heading_error_count
                velocity_dir_error_sum += sums.velocity_dir_error_sum_deg
                velocity_dir_error_count += sums.velocity_dir_error_count
                foot_sum += sums.foot_sliding_sum_cm
                foot_count += sums.foot_sliding_count
                world_sum += sums.foot_sliding_world_sum_cm
                world_count += sums.foot_sliding_world_count
                possible_contacts += sums.foot_contact_possible_count
                segment_id = int(batch_segments[seq_idx]) if batch_segments is not None else -1
                group = int(segments[segment_id]["source_entry_id"]) if segments and segment_id >= 0 else len(source_groups)
                source_groups.append(group)
                case_manifest.append({"source_start": int(batch["start"][seq_idx]), "source_entry_id": group,
                    "target_label": int(seq_label), "noise_seed": fid_bootstrap_seed})
                jitter_sum += sums.jitter_sum_cm
                jitter_count += sums.jitter_count
                per_sequence_mpjpe.append(
                    sums.mpjpe_sum_cm / sums.mpjpe_count if sums.mpjpe_count > 0 else float("nan")
                )
                per_sequence_root_ade.append(
                    sums.root_ade_sum_cm / sums.root_ade_count if sums.root_ade_count > 0 else float("nan")
                )
                per_sequence_heading_error.append(
                    sums.heading_error_sum_deg / sums.heading_error_count
                    if sums.heading_error_count > 0
                    else float("nan")
                )
                per_sequence_velocity_dir_error.append(
                    sums.velocity_dir_error_sum_deg / sums.velocity_dir_error_count
                    if sums.velocity_dir_error_count > 0
                    else float("nan")
                )
                per_sequence_foot_sliding.append(
                    sums.foot_sliding_sum_cm / sums.foot_sliding_count if sums.foot_sliding_count > 0 else float("nan")
                )
                per_sequence_foot_contacts.append(sums.foot_sliding_count)
                per_sequence_world_sliding.append(sums.foot_sliding_world_sum_cm / sums.foot_sliding_world_count
                    if sums.foot_sliding_world_count else float('nan'))
                per_sequence_world_contacts.append(sums.foot_sliding_world_count)
                per_sequence_jitter.append(
                    sums.jitter_sum_cm / sums.jitter_count if sums.jitter_count > 0 else float("nan")
                )
                if fid_extractor is None:
                    real_fid_features.append(sums.fid_ref_frames)
                    pred_fid_features.append(sums.fid_pred_frames)
                else:
                    ref_input = motion375_clip_to_classifier_input(ref_seq, joint_set=fid_joint_set)
                    pred_input = motion375_clip_to_classifier_input(pred_seq, joint_set=fid_joint_set)
                    ref_feat = fid_extractor.get_feats(ref_input.to(device=device))
                    pred_feat = fid_extractor.get_feats(pred_input.to(device=device))
                    real_fid_features.append(ref_feat.detach().cpu().numpy())
                    pred_fid_features.append(pred_feat.detach().cpu().numpy())
                if fid_camdm_aggregation:
                    # one entry per evaluated sequence, aligned with the appended features
                    fid_labels.append(int(seq_label))
                    fid_segments.append(-1 if batch_segments is None else int(batch_segments[seq_idx]))

    real_fid = np.concatenate(real_fid_features, axis=0)
    pred_fid = np.concatenate(pred_fid_features, axis=0)
    fid_value = frechet_distance(real_fid, pred_fid, eps=cfg.fid_eps)
    fid_camdm_style_value = float("nan")
    fid_camdm_style_styles = 0
    fid_camdm_clip_value = float("nan")
    fid_camdm_clip_clips = 0
    fid_camdm_clip_floor_value = float("nan")
    if fid_camdm_aggregation and len(fid_labels) == real_fid.shape[0]:
        labels_arr = np.asarray(fid_labels)
        fid_camdm_style_value, _, style_counts = camdm_style_fid(
            real_fid, pred_fid, labels_arr, eps=cfg.fid_eps
        )
        fid_camdm_style_styles = len(style_counts)
        fid_camdm_clip_value, fid_camdm_clip_clips = camdm_clip_fid(
            real_fid, pred_fid, labels_arr, np.asarray(fid_segments), eps=cfg.fid_eps
        )
        # Reference floor: what the per-clip variant reports for a perfect predictor, i.e. the
        # distance between each clip's own Gaussian and its style's pooled Gaussian.
        fid_camdm_clip_floor_value, _ = camdm_clip_fid(
            real_fid, real_fid, labels_arr, np.asarray(fid_segments), eps=cfg.fid_eps
        )
    fid_ci_half = float("nan")
    if fid_bootstrap_samples > 1 and len(real_fid_features) >= 2 and len(pred_fid_features) >= 2:
        rng = np.random.default_rng(fid_bootstrap_seed)
        fid_boot = []
        groups = np.asarray(source_groups)
        unique_groups = np.unique(groups)
        by_group = [np.flatnonzero(groups == group) for group in unique_groups]
        for _ in range(fid_bootstrap_samples):
            selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
            sample_ids = np.concatenate([by_group[int(i)] for i in selected])
            real_boot = np.concatenate([real_fid_features[i] for i in sample_ids], axis=0)
            pred_boot = np.concatenate([pred_fid_features[i] for i in sample_ids], axis=0)
            fid_boot.append(frechet_distance(real_boot, pred_boot, eps=cfg.fid_eps))
        fid_boot = np.asarray(fid_boot, dtype=np.float64)
        fid_ci_low = float(np.quantile(fid_boot, 0.025))
        fid_ci_high = float(np.quantile(fid_boot, 0.975))
        fid_ci_half = 0.5 * (fid_ci_high - fid_ci_low)

    mpjpe_values = np.asarray(per_sequence_mpjpe, dtype=np.float64)
    root_ade_values = np.asarray(per_sequence_root_ade, dtype=np.float64)
    heading_error_values = np.asarray(per_sequence_heading_error, dtype=np.float64)
    velocity_dir_error_values = np.asarray(per_sequence_velocity_dir_error, dtype=np.float64)
    foot_values = np.asarray(per_sequence_foot_sliding, dtype=np.float64)
    jitter_values = np.asarray(per_sequence_jitter, dtype=np.float64)
    latency_values = np.asarray(per_sequence_latency_ms, dtype=np.float64)

    def _ci95_half_width(values, weights=None):
        low, high = cluster_interval(values, source_groups, samples=1000, seed=fid_bootstrap_seed, weights=weights)
        return .5 * (high - low)

    mpjpe_mean = mpjpe_sum / mpjpe_count if mpjpe_count > 0 else float("nan")
    root_ade_mean = root_ade_sum / root_ade_count if root_ade_count > 0 else float("nan")
    heading_error_mean = heading_error_sum / heading_error_count if heading_error_count > 0 else float("nan")
    velocity_dir_error_mean = (
        velocity_dir_error_sum / velocity_dir_error_count if velocity_dir_error_count > 0 else float("nan")
    )
    foot_mean = foot_sum / foot_count if foot_count > 0 else float("nan")
    jitter_mean = jitter_sum / jitter_count if jitter_count > 0 else float("nan")
    mpjpe_ci_half = _ci95_half_width(mpjpe_values)
    root_ade_ci_half = _ci95_half_width(root_ade_values)
    heading_error_ci_half = _ci95_half_width(heading_error_values)
    velocity_dir_error_ci_half = _ci95_half_width(velocity_dir_error_values)
    foot_ci_half = _ci95_half_width(foot_values, per_sequence_foot_contacts)
    jitter_ci_half = _ci95_half_width(jitter_values)
    latency_p95_ms = float(np.quantile(latency_values[np.isfinite(latency_values)], 0.95)) if latency_values.size > 0 else float("nan")
    latency_mean_ms = float(np.mean(latency_values[np.isfinite(latency_values)])) if latency_values.size > 0 else float("nan")

    metrics = {
        "model": model_name,
        "ckpt": str(ckpt_path),
        "mpjpe_cm": mpjpe_mean,
        "mpjpe_cm_ci95": mpjpe_ci_half,
        "root_ade_cm": root_ade_mean,
        "root_ade_cm_ci95": root_ade_ci_half,
        "heading_error_deg": heading_error_mean,
        "heading_error_deg_ci95": heading_error_ci_half,
        "velocity_dir_error_deg": velocity_dir_error_mean,
        "velocity_dir_error_deg_ci95": velocity_dir_error_ci_half,
        "fid": fid_value,
        "fid_global": fid_value,
        "fid_extractor_sha256": getattr(fid_extractor, 'eval_sha256', None),
        "fid_ci95": fid_ci_half,
        "fid_camdm_style": fid_camdm_style_value,
        "fid_style": fid_camdm_style_value,
        "fid_camdm_style_groups": int(fid_camdm_style_styles),
        "fid_camdm_clip": fid_camdm_clip_value,
        "fid_clip": fid_camdm_clip_value,
        "fid_camdm_clip_groups": int(fid_camdm_clip_clips),
        "fid_camdm_clip_floor": fid_camdm_clip_floor_value,
        "foot_sliding_cm": foot_mean,
        "foot_sliding_local_cm": foot_mean,
        "foot_sliding_world_cm": world_sum / world_count if world_count else float("nan"),
        "foot_sliding_world_count": world_count,
        "foot_contact_coverage": world_count / possible_contacts if possible_contacts else float("nan"),
        "foot_sliding_local_available": foot_count > 0,
        "foot_sliding_world_available": world_count > 0,
        "per_sequence": [{**case, 'mpjpe_cm': float(mpjpe_values[i]), 'root_ade_cm': float(root_ade_values[i]),
            'foot_sliding_local_cm': float(foot_values[i]), 'foot_sliding_local_count': per_sequence_foot_contacts[i],
            'foot_sliding_world_cm': per_sequence_world_sliding[i], 'foot_sliding_world_count': per_sequence_world_contacts[i],
            'jitter_cm': float(jitter_values[i])} for i, case in enumerate(case_manifest)],
        "bootstrap_samples": 1000,
        "mpjpe_cm_ci95_bounds": cluster_interval(mpjpe_values, source_groups),
        "root_ade_cm_ci95_bounds": cluster_interval(root_ade_values, source_groups),
        "foot_sliding_local_cm_ci95_bounds": cluster_interval(foot_values, source_groups, weights=per_sequence_foot_contacts),
        "foot_sliding_world_cm_ci95_bounds": cluster_interval(per_sequence_world_sliding, source_groups, weights=per_sequence_world_contacts),
        "case_manifest": case_manifest, "ci_unit": "source_entry_id", "vq_eval_path": vq_eval_path,
        "foot_sliding_cm_ci95": foot_ci_half,
        "jitteriness_cm": jitter_mean,
        "jitteriness_cm_ci95": jitter_ci_half,
        "latency_ms_mean": latency_mean_ms,
        "latency_ms_p95": latency_p95_ms,
        "mpjpe_count": int(mpjpe_count),
        "root_ade_count": int(root_ade_count),
        "heading_error_count": int(heading_error_count),
        "velocity_dir_error_count": int(velocity_dir_error_count),
        "foot_sliding_count": int(foot_count),
        "jitteriness_count": int(jitter_count),
        "fid_frames_real": int(real_fid.shape[0]),
        "fid_frames_pred": int(pred_fid.shape[0]),
        "num_sequences": int(len(per_sequence_mpjpe)),
        "mpjpe_cm_text": f"{mpjpe_mean:.3f} ± {mpjpe_ci_half:.3f}",
        "root_ade_cm_text": f"{root_ade_mean:.3f} ± {root_ade_ci_half:.3f}",
        "heading_error_deg_text": f"{heading_error_mean:.3f} ± {heading_error_ci_half:.3f}",
        "velocity_dir_error_deg_text": f"{velocity_dir_error_mean:.3f} ± {velocity_dir_error_ci_half:.3f}",
        "fid_text": f"{fid_value:.3f} ± {fid_ci_half:.3f}" if np.isfinite(fid_ci_half) else f"{fid_value:.3f}",
        "fid_camdm_style_text": f"{fid_camdm_style_value:.4f}" if np.isfinite(fid_camdm_style_value) else "n/a",
        "fid_camdm_clip_text": f"{fid_camdm_clip_value:.4f}" if np.isfinite(fid_camdm_clip_value) else "n/a",
        "fid_camdm_clip_floor_text": (
            f"{fid_camdm_clip_floor_value:.4f}" if np.isfinite(fid_camdm_clip_floor_value) else "n/a"
        ),
        "foot_sliding_cm_text": f"{foot_mean:.3f} ± {foot_ci_half:.3f}",
        "jitteriness_cm_text": f"{jitter_mean:.3f} ± {jitter_ci_half:.3f}",
        "latency_ms_p95_text": f"{latency_p95_ms:.3f}",
    }
    return metrics


def main() -> None:
    args = parse_args()
    args.db = Path(args.db).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    validate_memmap_db_dir(args.db)
    if int(args.decode_smooth_taps) < 1 or int(args.decode_smooth_taps) % 2 == 0:
        raise ValueError(
            f"--decode-smooth-taps must be a positive odd integer, got {args.decode_smooth_taps}"
        )
    set_seed(args.seed)
    device = resolve_device(args.device)
    from Utils.runtime import configure_torch_runtime
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision='highest')

    ckpt_paths = [_resolve_ckpt_path(model_name, args) for model_name in args.models]
    warn_if_checkpoint_db_differs(args.models, ckpt_paths, args.db)
    norm_stats_path = _resolve_norm_stats_path(args, args.models, ckpt_paths)
    motion_mean = None
    motion_std = None
    if norm_stats_path is not None:
        motion_mean, motion_std = load_norm_stats(norm_stats_path)

    fid_extractor = None
    fid_joint_set = JOINT_SET_SELECTED24
    if args.fid_extractor_ckpt is not None:
        requested = None if args.fid_joint_set == "auto" else str(args.fid_joint_set)
        fid_extractor, fid_ckpt = build_fid_feature_extractor_from_checkpoint(
            Path(args.fid_extractor_ckpt), map_location="cpu", joint_set=requested
        )
        fid_joint_set = str(fid_ckpt["joint_set"])
        print(f"[Info] fid joint set   : {fid_joint_set} ({int(fid_ckpt['in_feat'])}-dim input)")
        fid_extractor = fid_extractor.to(device)
        from Utils.reproducibility import file_sha256
        fid_extractor.eval_sha256 = file_sha256(args.fid_extractor_ckpt)
        fid_extractor.eval()
        print(f"[Info] fid extractor  : {Path(args.fid_extractor_ckpt).expanduser().resolve()}")
        if int(fid_ckpt.get("window_size", args.seq_len)) != int(args.seq_len):
            raise ValueError(
                f"FID extractor window_size={fid_ckpt.get('window_size')} does not match --seq-len={args.seq_len}"
            )

    if args.normalize and (motion_mean is None or motion_std is None):
        raise ValueError("Evaluation requires saved training normalization; pass --norm-stats")
    train_dataset, _, test_dataset = build_train_val_test_datasets(
        db_path=args.db,
        seq_len=args.seq_len,
        stride=args.stride,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        load_into_memory=args.load_into_memory,
        normalize=args.normalize,
        norm_mean=motion_mean,
        norm_std=motion_std,
        max_windows_for_stats=args.max_windows_for_stats,
        stats_windows_chunk_size=args.stats_windows_chunk_size,
    )

    style_names = getattr(train_dataset, "style_names", None)
    if style_names:
        num_styles = len(style_names)
    else:
        labels = getattr(train_dataset, "labels", None)
        if labels is None:
            num_styles = 0
        else:
            num_styles = int(np.max(labels)) + 1

    test_dataset = subset_dataset_by_fraction(test_dataset, args.test_fraction, args.subset_seed + 2)
    if test_dataset is None or len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty after splitting/subsetting")

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
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    saved_stats_path = save_norm_stats_if_present(
        out_dir=args.out_dir,
        norm_mean=train_dataset.norm_mean,
        norm_std=train_dataset.norm_std,
    )

    cfg = MotionEvalConfig(
        dt=args.dt,
        foot_contact_height_cm=args.foot_contact_height_cm,
        foot_contact_speed_cm_s=args.foot_contact_speed_cm_s,
        velocity_dir_min_speed_cm_s=args.velocity_dir_min_speed_cm_s,
        metric_start_frame=args.metric_start_frame,
        metric_end_frame=args.metric_end_frame,
        fid_eps=args.fid_eps,
    )

    print(f"[Info] db              : {args.db}")
    print(f"[Info] device          : {device}")
    print(f"[Info] seq_len         : {args.seq_len}")
    print(f"[Info] stride          : {args.stride}")
    print(f"[Info] test samples    : {len(test_dataset)}")
    print(f"[Info] models          : {', '.join(args.models)}")
    if norm_stats_path is not None:
        print(f"[Info] norm stats      : {norm_stats_path}")
    elif saved_stats_path is not None:
        print(f"[Info] norm stats      : {saved_stats_path}")
    else:
        print("[Warn] norm stats      : not provided; using stats computed from the eval split")
    print(
        f"[Info] metric slice    : start={cfg.metric_start_frame} "
        f"end={cfg.metric_end_frame if cfg.metric_end_frame > 0 else 'all'}"
    )
    print(
        f"[Info] foot contact    : height<={cfg.foot_contact_height_cm} cm "
        f"speed<={cfg.foot_contact_speed_cm_s} cm/s"
    )
    print(f"[Info] vel dir min spd : {cfg.velocity_dir_min_speed_cm_s} cm/s")
    print("[Info] position space  : MPJPE=global, F.S./Jitter=local")
    print("[Info] jitter metric  : second-order local-position difference (no dt^2)")
    print(
        "[Info] decode filter  : "
        + ("off" if int(args.decode_smooth_taps) <= 1 else f"{int(args.decode_smooth_taps)}-tap binomial on the 291-dim motion block")
    )
    print("[Info] uncertainty     : reported as mean ± 95% CI")
    resolved_infer_steps = args.infer_steps if args.infer_steps > 0 else args.flow_ode_steps
    if any(
        model_name in {
            "flowmatch_velocity",
            "flowmatch_target",
            "flowmatch_lmm_velocity",
            "flowmatch_lmm_target",
            "diffusion_vqvae_x0",
            "diffusion_lmm_x0",
        }
        for model_name in args.models
    ):
        print(
            f"[Info] flow eval       : mode={args.flow_eval_mode} solver={args.flow_solver} "
            f"ode_steps={'ckpt' if resolved_infer_steps <= 0 else resolved_infer_steps} "
            f"noise_std={'ckpt' if args.flow_noise_std < 0.0 else args.flow_noise_std} "
            f"requantize_z={args.flow_requantize_z} latent_clip={args.flow_latent_clip}"
        )
    if "mvae" in args.models and cfg.metric_start_frame == 0:
        print(
            "[Tip] MVAE includes condition frames in the reported metrics by default. "
            "Use --metric-start-frame 1 (or your condition length) for prediction-only scoring."
        )
    if fid_extractor is not None:
        print("[Info] FID mode       : classifier feature extractor")
    else:
        print("[Info] FID mode       : raw motion frames")

    results: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        ckpt_path = _resolve_ckpt_path(model_name, args)
        metrics = _evaluate_one_model(
            model_name=model_name,
            ckpt_path=ckpt_path,
            loader=test_loader,
            device=device,
            cfg=cfg,
            motion_mean=motion_mean,
            motion_std=motion_std,
            fid_extractor=fid_extractor,
            fid_bootstrap_samples=args.fid_bootstrap_samples,
            fid_bootstrap_seed=args.seed,
            num_styles=num_styles,
            flow_vqvae_ckpt=args.flow_vqvae_ckpt,
            flow_lmm_ckpt=args.flow_lmm_ckpt,
            flow_eval_mode=args.flow_eval_mode,
            flow_solver=args.flow_solver,
            flow_ode_steps=resolved_infer_steps,
            flow_noise_std=args.flow_noise_std,
            flow_requantize_z=args.flow_requantize_z,
            flow_latent_clip=args.flow_latent_clip,
            allow_legacy_quantizer=args.allow_legacy_quantizer,
            decode_smooth_taps=args.decode_smooth_taps,
            fid_joint_set=fid_joint_set,
            fid_camdm_aggregation=bool(args.fid_camdm_aggregation),
            vq_eval_path=args.vq_eval_path, legacy_inference_order=args.legacy_inference_order,
        )
        results.append(metrics)
        print(
            f"[{model_name}] "
            f"MPJPE={metrics['mpjpe_cm_text']} cm | "
            f"RootADE={metrics['root_ade_cm_text']} cm | "
            f"Heading={metrics['heading_error_deg_text']} deg | "
            f"VelDir={metrics['velocity_dir_error_deg_text']} deg | "
            f"FID={metrics['fid_text']} | "
            f"FID_camdm(style)={metrics['fid_camdm_style_text']} | "
            f"FID_camdm(clip)={metrics['fid_camdm_clip_text']} (floor {metrics['fid_camdm_clip_floor_text']}) | "
            f"F.S.={metrics['foot_sliding_cm_text']} cm | "
            f"Jitter={metrics['jitteriness_cm_text']} cm | "
            f"P95Latency={metrics['latency_ms_p95_text']} ms"
        )

    summary = {
        "config": {
            "db": str(args.db),
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "test_fraction": args.test_fraction,
            "metric_start_frame": cfg.metric_start_frame,
            "metric_end_frame": cfg.metric_end_frame,
            "foot_contact_height_cm": cfg.foot_contact_height_cm,
            "foot_contact_speed_cm_s": cfg.foot_contact_speed_cm_s,
            "velocity_dir_min_speed_cm_s": cfg.velocity_dir_min_speed_cm_s,
            "dt": cfg.dt,
            "models": list(args.models),
            "norm_stats": str(norm_stats_path) if norm_stats_path is not None else None,
            "fid_extractor_ckpt": str(args.fid_extractor_ckpt) if args.fid_extractor_ckpt is not None else None,
            "fid_bootstrap_samples": args.fid_bootstrap_samples,
            "seed": args.seed,
            "flow_vqvae_ckpt": str(args.flow_vqvae_ckpt) if args.flow_vqvae_ckpt is not None else None,
            "flow_lmm_ckpt": str(args.flow_lmm_ckpt) if args.flow_lmm_ckpt is not None else None,
            "flow_eval_mode": args.flow_eval_mode,
            "flow_solver": args.flow_solver,
            "infer_steps": resolved_infer_steps,
            "flow_ode_steps": args.flow_ode_steps,
            "flow_noise_std": args.flow_noise_std,
            "vq_eval_path": args.vq_eval_path, "legacy_inference_order": args.legacy_inference_order,
            "flow_requantize_z": args.flow_requantize_z,
            "flow_latent_clip": args.flow_latent_clip,
            "allow_legacy_quantizer": args.allow_legacy_quantizer,
            "decode_smooth_taps": args.decode_smooth_taps,
            "fid_joint_set": fid_joint_set,
            "fid_camdm_aggregation": bool(args.fid_camdm_aggregation),
        },
        "results": results,
    }

    summary_path = args.out_dir / "motion_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    csv_path = args.out_dir / "motion_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, extrasaction="ignore",
            fieldnames=[
                "model",
                "ckpt",
                "mpjpe_cm",
                "mpjpe_cm_ci95",
                "root_ade_cm",
                "root_ade_cm_ci95",
                "heading_error_deg",
                "heading_error_deg_ci95",
                "velocity_dir_error_deg",
                "velocity_dir_error_deg_ci95",
                "fid",
                "fid_ci95",
                "fid_camdm_style",
                "fid_camdm_style_groups",
                "fid_camdm_clip",
                "fid_camdm_clip_groups",
                "fid_camdm_clip_floor",
                "fid_camdm_style_text",
                "fid_camdm_clip_text",
                "fid_camdm_clip_floor_text",
                "foot_sliding_local_cm", "foot_sliding_world_cm", "foot_sliding_world_count", "foot_contact_coverage", "fid_global",
                "foot_sliding_cm",
                "foot_sliding_cm_ci95",
                "jitteriness_cm",
                "jitteriness_cm_ci95",
                "latency_ms_mean",
                "latency_ms_p95",
                "mpjpe_cm_text",
                "root_ade_cm_text",
                "heading_error_deg_text",
                "velocity_dir_error_deg_text",
                "fid_text",
                "foot_sliding_cm_text",
                "jitteriness_cm_text",
                "latency_ms_p95_text",
                "mpjpe_count",
                "root_ade_count",
                "heading_error_count",
                "velocity_dir_error_count",
                "foot_sliding_count",
                "jitteriness_count",
                "fid_frames_real",
                "fid_frames_pred",
                "num_sequences",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[Done] summary saved : {summary_path}")
    print(f"[Done] csv saved     : {csv_path}")


if __name__ == "__main__":
    main()
