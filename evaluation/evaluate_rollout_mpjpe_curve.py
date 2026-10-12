from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Utils.model_loading import build_camdm_from_checkpoint, build_mann_from_checkpoint
from Utils.motion375 import reconstruct_selected_globals_from_375
from Utils.motion_eval import MotionModelName, denormalize_motion_batch, load_motion_model, reconstruct_motion_batch
from Utils.realtime import contiguous_runs, load_db_arrays, parse_style_selector
from Utils.runtime import load_checkpoint, load_norm_stats, resolve_device, set_seed


SCRIPT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CKPTS = {
    "camdm": SCRIPT_DIR / "checkpoints" / "camdm_controller" / "best.pt",
    "mann": SCRIPT_DIR / "checkpoints" / "mann_controller" / "best.pt",
    "flowmatch_velocity": SCRIPT_DIR / "checkpoints" / "flowmatch_vqvae_controller_velocity_head10" / "best.pt",
    "flowmatch_lmm_velocity": SCRIPT_DIR / "checkpoints" / "flowmatch_lmm_controller_velocity_head10" / "best.pt",
}

DISPLAY_NAMES = {
    "camdm": "CAMDM",
    "mann": "MANN",
    "flowmatch_velocity": "VQ+FlowMatch",
    "flowmatch_lmm_velocity": "LMM+FlowMatch",
}


def warn_if_checkpoint_db_differs(model_names, ckpt_paths, eval_db) -> None:
    """Flag checkpoints trained on a different database than the one evaluated (see the
    ``*_head10`` defaults, which are out of distribution on the 90-style database)."""
    eval_db_name = Path(eval_db).name
    for model_name, ckpt_path in zip(model_names, ckpt_paths):
        try:
            ckpt = load_checkpoint(Path(ckpt_path), map_location="cpu")
        except Exception:  # noqa: BLE001
            continue
        train_db = str((ckpt.get("args") or {}).get("db", "") or "")
        train_db_name = Path(train_db).name if train_db else ""
        if train_db_name and train_db_name != eval_db_name:
            print(
                f"[Warn] {model_name}: checkpoint was trained on '{train_db_name}' but is being "
                f"evaluated on '{eval_db_name}' - the metrics are out of distribution"
            )


def _resolve_ckpt_path(args: argparse.Namespace, model_name: str) -> Path:
    field = f"{model_name}_ckpt".replace("-", "_")
    explicit = getattr(args, field, None)
    if explicit is not None:
        ckpt = Path(explicit)
    elif model_name in DEFAULT_CKPTS:
        ckpt = DEFAULT_CKPTS[model_name]
    else:
        raise ValueError(f"Checkpoint must be provided for {model_name}")
    ckpt = ckpt.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {ckpt}")
    return ckpt


def _resolve_norm_stats_path(
    args: argparse.Namespace,
    model_names: list[MotionModelName],
    ckpt_paths: list[Path],
) -> Path:
    if args.norm_stats is not None:
        path = Path(args.norm_stats).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"norm stats not found: {path}")
        return path

    for model_name, ckpt_path in zip(model_names, ckpt_paths, strict=True):
        candidate = ckpt_path.parent / "norm_stats.npz"
        if candidate.exists():
            return candidate.resolve()
        if model_name in {
            "flowmatch_velocity",
            "flowmatch_lmm_velocity",
        }:
            flow_ckpt = load_checkpoint(ckpt_path, map_location="cpu")
            flow_args = flow_ckpt.get("args", {}) or {}
            for key in ("norm_stats", "vqvae_ckpt", "lmm_ckpt"):
                raw = flow_args.get(key, None)
                if raw is None:
                    continue
                candidate = Path(raw).expanduser()
                if not candidate.is_absolute():
                    candidate = (ckpt_path.parent / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if key == "norm_stats" and candidate.exists():
                    return candidate
                sibling = candidate.parent / "norm_stats.npz"
                if sibling.exists():
                    return sibling.resolve()
    raise FileNotFoundError("Could not resolve norm_stats.npz. Pass --norm-stats explicitly.")


def _candidate_starts_for_segments(
    labels: np.ndarray,
    *,
    seq_len: int,
    target_label: int | None,
) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for run_start, run_end, label in contiguous_runs(labels):
        if target_label is not None and int(label) != int(target_label):
            continue
        run_len = int(run_end - run_start)
        if run_len < seq_len:
            continue
        for start in range(run_start, run_end - seq_len + 1):
            starts.append((int(start), int(label)))
    return starts


def _sample_segments(
    x_source,
    labels: np.ndarray,
    style_names: list[str],
    *,
    seq_len: int,
    num_samples: int,
    seed: int,
    source_style_label: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    candidates = _candidate_starts_for_segments(
        labels,
        seq_len=seq_len,
        target_label=source_style_label,
    )
    if len(candidates) < num_samples:
        raise ValueError(
            f"Not enough valid segments for seq_len={seq_len} and source_style={source_style_label}: "
            f"requested {num_samples}, found {len(candidates)}"
        )
    rng = np.random.default_rng(seed)
    pick_ids = rng.choice(len(candidates), size=num_samples, replace=False)
    picks = [candidates[int(i)] for i in pick_ids]
    picks.sort(key=lambda item: item[0])

    sequences = np.stack(
        [np.asarray(x_source[start : start + seq_len], dtype=np.float32) for start, _ in picks],
        axis=0,
    )
    sample_labels = np.asarray([label for _, label in picks], dtype=np.int64)
    metadata = [
        {
            "sample_id": int(sample_id),
            "start": int(start),
            "end": int(start + seq_len),
            "label": int(label),
            "style_name": style_names[int(label)],
        }
        for sample_id, (start, label) in enumerate(picks)
    ]
    return sequences, sample_labels, metadata


def _load_test_split_segments(
    db_path: Path,
    *,
    seq_len: int,
    style_names: list[str],
    source_style_label: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    segments_path = db_path / "segments.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"segments.json not found for test-split evaluation: {segments_path}")
    with segments_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError(f"segments.json must contain a list, got {type(segments)}")

    picked: list[dict[str, object]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if str(segment.get("split", "")) != "test":
            continue
        label = int(segment["label"])
        if source_style_label is not None and label != int(source_style_label):
            continue
        if int(segment["num_frames"]) < seq_len:
            continue
        picked.append(segment)

    picked.sort(key=lambda item: int(item["frame_start"]))
    if not picked:
        raise ValueError(
            f"No test-split segments can provide seq_len={seq_len} for source_style={source_style_label}"
        )

    starts = np.asarray([int(segment["frame_start"]) for segment in picked], dtype=np.int64)
    labels = np.asarray([int(segment["label"]) for segment in picked], dtype=np.int64)
    metadata = [
        {
            "sample_id": int(sample_id),
            "start": int(segment["frame_start"]),
            "end": int(segment["frame_start"]) + seq_len,
            "label": int(segment["label"]),
            "style_name": style_names[int(segment["label"])],
            "source_path": str(segment.get("source_path", "")),
            "suffix": str(segment.get("suffix", "")),
            "split": "test",
        }
        for sample_id, segment in enumerate(picked)
    ]
    return starts, labels, metadata


def _per_frame_mpjpe_cm(
    *,
    reference_375: np.ndarray,
    prediction_375: np.ndarray,
    dt: float,
    mpjpe_scale: float,
) -> np.ndarray:
    ref_global_pos, _ = reconstruct_selected_globals_from_375(
        reference_375,
        dt=dt,
        root_wy_in_degrees=True,
        use_global=True,
    )
    pred_global_pos, _ = reconstruct_selected_globals_from_375(
        prediction_375,
        dt=dt,
        root_wy_in_degrees=True,
        use_global=True,
    )
    per_joint_error_cm = np.linalg.norm(pred_global_pos - ref_global_pos, axis=-1) * 100.0
    return (float(mpjpe_scale) * np.mean(per_joint_error_cm, axis=-1)).astype(np.float32)


def _cumulative_curve_from_per_frame(
    per_frame_mpjpe_cm: np.ndarray,
    *,
    horizons: np.ndarray,
) -> np.ndarray:
    if per_frame_mpjpe_cm.ndim != 1:
        raise ValueError(f"Expected per_frame_mpjpe_cm rank 1, got {per_frame_mpjpe_cm.shape}")
    if per_frame_mpjpe_cm.size < 2:
        raise ValueError("Need at least 2 frames to compute rollout MPJPE curve")

    generated_errors = per_frame_mpjpe_cm[1:]
    cumsum = np.cumsum(generated_errors, dtype=np.float64)
    values = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1 or h > generated_errors.size:
            raise ValueError(f"Invalid horizon {h} for generated length {generated_errors.size}")
        values.append(float(cumsum[h - 1] / h))
    return np.asarray(values, dtype=np.float32)


def _save_long_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve_figure(
    *,
    summary_rows: list[dict[str, object]],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.0, 5.0))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    for model_name, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: int(item["horizon"]))
        x = np.asarray([int(item["horizon"]) for item in rows_sorted], dtype=np.int32)
        y = np.asarray([float(item["mean_cum_mpjpe_cm"]) for item in rows_sorted], dtype=np.float32)
        y_std = np.asarray([float(item["std_cum_mpjpe_cm"]) for item in rows_sorted], dtype=np.float32)
        plt.plot(x, y, marker="o", linewidth=2.0, label=DISPLAY_NAMES.get(model_name, model_name))
        if np.any(np.isfinite(y_std)) and np.any(y_std > 0):
            plt.fill_between(x, y - y_std, y + y_std, alpha=0.18)

    plt.xlabel("Rollout Horizon (frames)")
    plt.ylabel("Cumulative MPJPE")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rollout cumulative MPJPE curves for stylized control models")
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "eval_results" / "rollout_mpjpe_curve")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "camdm",
            "mann",
            "flowmatch_velocity",
            "flowmatch_lmm_velocity",
        ],
        default=["camdm", "mann", "flowmatch_velocity", "flowmatch_lmm_velocity"],
    )
    parser.add_argument("--camdm-ckpt", type=Path, default=None)
    parser.add_argument("--mann-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-velocity-ckpt", type=Path, default=None)
    parser.add_argument("--flowmatch-lmm-velocity-ckpt", type=Path, default=None)
    parser.add_argument("--flow-vqvae-ckpt", type=Path, default=None)
    parser.add_argument("--flow-lmm-ckpt", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=600, help="Number of generated rollout frames after the seed frame")
    parser.add_argument("--curve-interval", type=int, default=50, help="Record cumulative MPJPE every N rollout frames")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of sampled source segments")
    parser.add_argument(
        "--use-test-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use all eligible split=test segments from segments.json instead of random sampling",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--source-style", type=str, default="", help="Optional style selector to restrict sampled source segments")
    parser.add_argument("--target-style", type=str, default="", help="Optional target style. Must match reference style for MPJPE-based evaluation.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--mpjpe-scale", type=float, default=0.1, help="Scale factor applied to rollout MPJPE in this experiment")
    parser.add_argument("--infer-steps", type=int, default=0)
    parser.add_argument("--flow-ode-steps", type=int, default=0)
    parser.add_argument("--flow-noise-std", type=float, default=-1.0)
    parser.add_argument("--flow-requantize-z", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flow-latent-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-title", type=str, default="Rollout Cumulative MPJPE Curve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_frames < 1:
        raise ValueError(f"--num-frames must be >= 1, got {args.num_frames}")
    if args.curve_interval < 1:
        raise ValueError(f"--curve-interval must be >= 1, got {args.curve_interval}")
    if args.num_frames % args.curve_interval != 0:
        raise ValueError(
            f"--num-frames ({args.num_frames}) must be divisible by --curve-interval ({args.curve_interval})"
        )
    if args.num_samples < 1 and not args.use_test_split:
        raise ValueError(f"--num-samples must be >= 1, got {args.num_samples}")
    if args.eval_batch_size < 1:
        raise ValueError(f"--eval-batch-size must be >= 1, got {args.eval_batch_size}")
    if args.mpjpe_scale < 0.0:
        raise ValueError(f"--mpjpe-scale must be >= 0, got {args.mpjpe_scale}")

    set_seed(args.seed)
    device = resolve_device(args.device)

    model_names: list[MotionModelName] = [str(name) for name in args.models]
    ckpt_paths = [_resolve_ckpt_path(args, model_name) for model_name in model_names]
    warn_if_checkpoint_db_differs(model_names, ckpt_paths, args.db)
    norm_stats_path = _resolve_norm_stats_path(args, model_names, ckpt_paths)
    norm_mean, norm_std = load_norm_stats(norm_stats_path)

    x_source, labels, style_names = load_db_arrays(args.db)
    source_style_label = parse_style_selector(args.source_style, style_names)
    target_style_label = parse_style_selector(args.target_style, style_names)

    seq_len = int(args.num_frames) + 1
    if args.use_test_split:
        starts, sample_labels, sample_meta = _load_test_split_segments(
            args.db,
            seq_len=seq_len,
            style_names=style_names,
            source_style_label=source_style_label,
        )
        args.num_samples = int(starts.shape[0])
        x_ref_raw = np.stack(
            [np.asarray(x_source[int(start) : int(start) + seq_len], dtype=np.float32) for start in starts],
            axis=0,
        )
    else:
        x_ref_raw, sample_labels, sample_meta = _sample_segments(
            x_source,
            labels,
            style_names,
            seq_len=seq_len,
            num_samples=args.num_samples,
            seed=args.seed,
            source_style_label=source_style_label,
        )

    if target_style_label is None:
        target_labels = sample_labels.copy()
    else:
        target_labels = np.full_like(sample_labels, fill_value=int(target_style_label))
        if not np.all(sample_labels == target_labels):
            raise ValueError(
                "MPJPE curve evaluation requires a matching reference style. "
                f"Requested target_style={style_names[int(target_style_label)]}, but sampled reference labels were "
                f"{sorted({style_names[int(v)] for v in sample_labels.tolist()})}. "
                "Use --source-style with the same style, or omit --target-style."
            )

    x_ref_norm = (x_ref_raw - norm_mean[None, None, :]) / norm_std[None, None, :]
    horizons = np.arange(args.curve_interval, args.num_frames + 1, args.curve_interval, dtype=np.int32)

    print(f"[Info] db              : {args.db}")
    print(f"[Info] norm stats      : {norm_stats_path}")
    print(f"[Info] device          : {device}")
    print(f"[Info] models          : {', '.join(model_names)}")
    print(f"[Info] num samples     : {args.num_samples}")
    print(f"[Info] source mode     : {'test_split' if args.use_test_split else 'sampled'}")
    print(f"[Info] rollout frames  : {args.num_frames}")
    print(f"[Info] curve interval  : {args.curve_interval}")
    print(f"[Info] mpjpe scale     : {args.mpjpe_scale}")
    print(f"[Info] horizons        : {', '.join(str(int(v)) for v in horizons.tolist())}")
    resolved_infer_steps = args.infer_steps if args.infer_steps > 0 else args.flow_ode_steps

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    all_curves: dict[str, np.ndarray] = {}

    for model_name, ckpt_path in zip(model_names, ckpt_paths, strict=True):
        print(f"[Info] evaluating {model_name}: {ckpt_path}")
        if model_name == "camdm":
            model = build_camdm_from_checkpoint(ckpt_path).to(device)
            if int(model.num_styles) != len(style_names):
                raise ValueError(
                    f"CAMDM checkpoint expects num_styles={int(model.num_styles)}, "
                    f"but DB {args.db} provides {len(style_names)} styles."
                )
        elif model_name == "mann":
            model = build_mann_from_checkpoint(ckpt_path).to(device)
            if int(model.num_styles) != len(style_names):
                raise ValueError(
                    f"MANN checkpoint expects num_styles={int(model.num_styles)}, "
                    f"but DB {args.db} provides {len(style_names)} styles."
                )
        else:
            model = load_motion_model(
                model_name=model_name,
                ckpt_path=ckpt_path,
                num_styles=len(style_names),
                flow_vqvae_ckpt=args.flow_vqvae_ckpt,
                flow_lmm_ckpt=args.flow_lmm_ckpt,
            ).to(device)
        model.eval()

        curves = np.empty((args.num_samples, horizons.size), dtype=np.float32)
        with torch.no_grad():
            for batch_begin in range(0, args.num_samples, args.eval_batch_size):
                batch_end = min(batch_begin + args.eval_batch_size, args.num_samples)
                x_batch = torch.from_numpy(x_ref_norm[batch_begin:batch_end]).to(device=device, dtype=torch.float32)
                style_batch = torch.from_numpy(target_labels[batch_begin:batch_end]).to(device=device, dtype=torch.long)
                pred_batch = reconstruct_motion_batch(
                    model_name=model_name,
                    model=model,
                    x=x_batch,
                    style_label=style_batch,
                    flow_eval_mode="rollout",
                    flow_solver="euler",
                    flow_ode_steps=None if resolved_infer_steps <= 0 else resolved_infer_steps,
                    flow_noise_std=None if args.flow_noise_std < 0.0 else args.flow_noise_std,
                    flow_requantize_z=args.flow_requantize_z,
                    flow_latent_clip=args.flow_latent_clip,
                )
                pred_np = denormalize_motion_batch(
                    pred_batch,
                    mean=norm_mean,
                    std=norm_std,
                )
                ref_np = x_ref_raw[batch_begin:batch_end]

                for local_idx, (ref_seq, pred_seq) in enumerate(zip(ref_np, pred_np, strict=True)):
                    sample_id = batch_begin + local_idx
                    per_frame = _per_frame_mpjpe_cm(
                        reference_375=ref_seq,
                        prediction_375=pred_seq,
                        dt=args.dt,
                        mpjpe_scale=args.mpjpe_scale,
                    )
                    curve = _cumulative_curve_from_per_frame(per_frame, horizons=horizons)
                    curves[sample_id] = curve
                    for horizon, value in zip(horizons, curve, strict=True):
                        detail_rows.append(
                            {
                                "model": model_name,
                                "sample_id": int(sample_id),
                                "horizon": int(horizon),
                                "cum_mpjpe_cm": float(value),
                                "start": int(sample_meta[sample_id]["start"]),
                                "end": int(sample_meta[sample_id]["end"]),
                                "label": int(sample_meta[sample_id]["label"]),
                                "style_name": str(sample_meta[sample_id]["style_name"]),
                            }
                        )

        all_curves[model_name] = curves
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        for horizon, mean_value, std_value in zip(horizons, mean_curve, std_curve, strict=True):
            summary_rows.append(
                {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES.get(model_name, model_name),
                    "horizon": int(horizon),
                    "mean_cum_mpjpe_cm": float(mean_value),
                    "std_cum_mpjpe_cm": float(std_value),
                    "num_samples": int(args.num_samples),
                    "ckpt": str(ckpt_path),
                }
            )

    detail_csv_path = args.out_dir / "rollout_mpjpe_curve.detail.csv"
    summary_csv_path = args.out_dir / "rollout_mpjpe_curve.summary.csv"
    meta_json_path = args.out_dir / "rollout_mpjpe_curve.meta.json"
    plot_path = args.out_dir / "rollout_mpjpe_curve.png"

    _save_long_csv(
        detail_csv_path,
        detail_rows,
        fieldnames=["model", "sample_id", "horizon", "cum_mpjpe_cm", "start", "end", "label", "style_name"],
    )
    _save_long_csv(
        summary_csv_path,
        summary_rows,
        fieldnames=["model", "display_name", "horizon", "mean_cum_mpjpe_cm", "std_cum_mpjpe_cm", "num_samples", "ckpt"],
    )
    _plot_curve_figure(
        summary_rows=summary_rows,
        out_path=plot_path,
        title=args.plot_title,
    )

    meta = {
        "db": str(args.db),
        "norm_stats": str(norm_stats_path),
        "device": str(device),
        "models": model_names,
        "num_frames": int(args.num_frames),
        "curve_interval": int(args.curve_interval),
        "num_samples": int(args.num_samples),
        "use_test_split": bool(args.use_test_split),
        "eval_batch_size": int(args.eval_batch_size),
        "seed": int(args.seed),
        "dt": float(args.dt),
        "mpjpe_scale": float(args.mpjpe_scale),
        "source_style": args.source_style,
        "target_style": args.target_style,
        "infer_steps": int(resolved_infer_steps),
        "flow_ode_steps": int(args.flow_ode_steps),
        "flow_noise_std": float(args.flow_noise_std),
        "flow_requantize_z": bool(args.flow_requantize_z),
        "flow_latent_clip": bool(args.flow_latent_clip),
        "flow_vqvae_ckpt_override": None if args.flow_vqvae_ckpt is None else str(Path(args.flow_vqvae_ckpt).expanduser().resolve()),
        "flow_lmm_ckpt_override": None if args.flow_lmm_ckpt is None else str(Path(args.flow_lmm_ckpt).expanduser().resolve()),
        "horizons": [int(v) for v in horizons.tolist()],
        "sample_segments": sample_meta,
        "checkpoints": {model_name: str(path) for model_name, path in zip(model_names, ckpt_paths, strict=True)},
        "summary_csv": str(summary_csv_path),
        "detail_csv": str(detail_csv_path),
        "plot_png": str(plot_path),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Info] summary csv     : {summary_csv_path}")
    print(f"[Info] detail csv      : {detail_csv_path}")
    print(f"[Info] meta json       : {meta_json_path}")
    print(f"[Info] plot png        : {plot_path}")
    print("[Done] Rollout MPJPE curve evaluation complete.")


if __name__ == "__main__":
    main()
