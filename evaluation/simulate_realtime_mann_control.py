"""
Simulate real-time stylized motion control with the NewCode MANN controller.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from Utils.model_loading import build_mann_from_checkpoint
from Utils.realtime import (
    choose_segment,
    load_db_arrays,
    parse_style_selector,
    prepare_output_dir,
    resolve_db_coordinate_convert,
    write_generated_reference_npz,
    write_meta_json,
)
from Utils.runtime import load_norm_stats, resolve_device, set_seed
from Visualization.backends.dispatch_backend import canonicalize_vis_backend, visualize_pair_from_375


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent


def _sync_device_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_rollout_benchmark(
    *,
    frame_latencies_sec: np.ndarray,
    warmup_frames: int,
    target_fps: float,
) -> dict[str, float | int]:
    if frame_latencies_sec.ndim != 1:
        raise ValueError(f"Expected frame_latencies_sec rank 1, got {frame_latencies_sec.shape}")
    if frame_latencies_sec.size == 0:
        raise ValueError("No frame latencies collected for benchmark")

    warmup = min(int(warmup_frames), int(frame_latencies_sec.size))
    measured = frame_latencies_sec[warmup:]
    if measured.size == 0:
        raise ValueError("No measured frames left after warmup")

    lat_ms = measured * 1000.0
    fps_inst = 1.0 / np.maximum(measured, 1e-12)
    total_sec = float(measured.sum())
    target_frame_ms = 1000.0 / float(target_fps)
    over_budget = lat_ms > target_frame_ms

    return {
        "warmup_frames": int(warmup),
        "measured_frames": int(measured.size),
        "target_fps": float(target_fps),
        "target_frame_ms": float(target_frame_ms),
        "latency_ms_mean": float(lat_ms.mean()),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)),
        "latency_ms_p99": float(np.percentile(lat_ms, 99)),
        "fps_mean_effective": float(measured.size / max(total_sec, 1e-12)),
        "fps_p05": float(np.percentile(fps_inst, 5)),
        "fps_p95": float(np.percentile(fps_inst, 95)),
        "over_budget_ratio": float(over_budget.mean()),
    }


def _print_benchmark_summary(summary: dict[str, float | int]) -> None:
    print(
        "[Benchmark] "
        f"frames={summary['measured_frames']} warmup={summary['warmup_frames']} "
        f"lat_mean={summary['latency_ms_mean']:.3f}ms "
        f"lat_p95={summary['latency_ms_p95']:.3f}ms "
        f"lat_p99={summary['latency_ms_p99']:.3f}ms "
        f"fps_eff={summary['fps_mean_effective']:.2f} "
        f"fps_p05={summary['fps_p05']:.2f} "
        f"fps_p95={summary['fps_p95']:.2f} "
        f"over_budget={100.0 * float(summary['over_budget_ratio']):.2f}%"
    )


def reconstruct_sequence(
    *,
    model,
    x_ref_norm: np.ndarray,
    style_label: int,
    device: torch.device,
    benchmark: bool = False,
    warmup_frames: int = 0,
    target_fps: float = 60.0,
) -> tuple[np.ndarray, dict[str, float | int] | None]:
    if x_ref_norm.ndim != 2 or x_ref_norm.shape[1] != 375:
        raise ValueError(f"x_ref_norm must be shape [T,375], got {x_ref_norm.shape}")
    if x_ref_norm.shape[0] < 2:
        raise ValueError(f"x_ref_norm must contain at least 2 frames, got {x_ref_norm.shape[0]}")

    x_ref_t = torch.from_numpy(x_ref_norm.astype(np.float32, copy=False)).to(device=device)
    seq = torch.empty_like(x_ref_t)
    seq[0] = x_ref_t[0]
    style_t = torch.tensor([int(style_label)], device=device, dtype=torch.long)
    frame_latencies_sec: list[float] = []

    model.eval()
    with torch.no_grad():
        for frame_idx in range(1, x_ref_t.size(0)):
            x_curr = x_ref_t[frame_idx - 1 : frame_idx]
            traj_next = x_ref_t[frame_idx : frame_idx + 1, :84]
            if benchmark:
                _sync_device_if_needed(device)
                t0 = time.perf_counter()
            x_next = model.predict_next_frame(
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_t,
            )
            if benchmark:
                _sync_device_if_needed(device)
                frame_latencies_sec.append(time.perf_counter() - t0)
            seq[frame_idx] = x_next[0]

    benchmark_summary = None
    if benchmark:
        benchmark_summary = _summarize_rollout_benchmark(
            frame_latencies_sec=np.asarray(frame_latencies_sec, dtype=np.float64),
            warmup_frames=warmup_frames,
            target_fps=target_fps,
        )
    return seq.detach().cpu().numpy().astype(np.float32, copy=False), benchmark_summary


def rollout_sequence(
    *,
    model,
    x_seed_norm: np.ndarray,
    traj_plan_norm: np.ndarray,
    style_label: int,
    device: torch.device,
    benchmark: bool = False,
    warmup_frames: int = 0,
    target_fps: float = 60.0,
) -> tuple[np.ndarray, dict[str, float | int] | None]:
    if x_seed_norm.shape != (375,):
        raise ValueError(f"x_seed_norm must be shape [375], got {x_seed_norm.shape}")
    if traj_plan_norm.ndim != 2 or traj_plan_norm.shape[1] != 84:
        raise ValueError(f"traj_plan_norm must be shape [T,84], got {traj_plan_norm.shape}")

    style_t = torch.tensor([int(style_label)], device=device, dtype=torch.long)
    traj_plan_t = torch.from_numpy(traj_plan_norm.astype(np.float32, copy=False)).to(device=device)
    x_curr = torch.from_numpy(x_seed_norm.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)
    seq = torch.empty((traj_plan_t.size(0), 375), device=device, dtype=torch.float32)
    seq[0] = x_curr[0]
    frame_latencies_sec: list[float] = []

    model.eval()
    with torch.no_grad():
        for frame_idx in range(1, traj_plan_t.size(0)):
            traj_next = traj_plan_t[frame_idx : frame_idx + 1]
            if benchmark:
                _sync_device_if_needed(device)
                t0 = time.perf_counter()
            x_next = model.predict_next_frame(
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_t,
            )
            if benchmark:
                _sync_device_if_needed(device)
                frame_latencies_sec.append(time.perf_counter() - t0)
            seq[frame_idx] = x_next[0]
            x_curr = x_next

    benchmark_summary = None
    if benchmark:
        benchmark_summary = _summarize_rollout_benchmark(
            frame_latencies_sec=np.asarray(frame_latencies_sec, dtype=np.float64),
            warmup_frames=warmup_frames,
            target_fps=target_fps,
        )
    return seq.detach().cpu().numpy().astype(np.float32, copy=False), benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate stylized motion control with the NewCode MANN controller")
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror",
        help="Input DB path (memmap directory)",
    )
    parser.add_argument(
        "--mann-ckpt",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "mann_controller" / "best.pt",
        help="Trained MANN checkpoint",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz. Default: sibling of MANN checkpoint (norm_stats.npz)",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Generated rollout length")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for segment sampling")
    parser.add_argument(
        "--source-style",
        type=str,
        default="",
        help="Style selector for source trajectory segment (name or label id)",
    )
    parser.add_argument(
        "--target-style",
        type=str,
        default="",
        help="Target style for generation (name or label id). Empty means same as source style.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional fixed start index in DB (must keep same label for num_frames)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reconstruct", "rollout"],
        default="rollout",
        help="Generation mode. 'reconstruct' uses teacher-forced previous motion; 'rollout' is autoregressive.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--benchmark", action="store_true", help="Measure per-frame generation latency")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Warmup frames excluded from benchmark summary")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "rollout_vis",
        help="Directory for generated npz, metadata, and visualization outputs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Default: mann_<mode>_<num_frames>",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Show side-by-side reference/prediction visualization after generation (default true)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization window",
    )
    parser.add_argument("--fps", type=int, default=60, help="Visualization fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Visualization dt")
    parser.add_argument("--local-only", action="store_true", help="Visualize local positions only")
    parser.add_argument(
        "--vis-backend",
        type=str,
        choices=["genoview", "quick", "matplotlib"],
        default="genoview",
        help="Visualization backend",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="Optional GIF output path for quick visualization backend.",
    )
    parser.add_argument(
        "--geno-gap",
        type=float,
        default=2.4,
        help="Horizontal spacing between the two GenoView characters.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="100STYLE BVH root used for side-by-side visualization",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional BVH ref used for side-by-side visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.mann_ckpt = args.mann_ckpt.expanduser().resolve()
    args.vis_backend = canonicalize_vis_backend(args.vis_backend)
    if args.norm_stats is None:
        args.norm_stats = (args.mann_ckpt.parent / "norm_stats.npz").resolve()
    else:
        args.norm_stats = args.norm_stats.expanduser().resolve()

    if not args.mann_ckpt.exists():
        raise FileNotFoundError(f"MANN checkpoint not found: {args.mann_ckpt}")
    if not args.norm_stats.exists():
        raise FileNotFoundError(f"norm stats not found: {args.norm_stats}")

    if args.prefix is None or args.prefix.strip() == "":
        args.prefix = f"mann_{args.mode}_{args.num_frames}"

    set_seed(args.seed)
    device = resolve_device(args.device)
    coordinate_convert = resolve_db_coordinate_convert(args.db)
    x_source, labels, style_names = load_db_arrays(args.db)
    if labels.size == 0:
        raise RuntimeError("DB is empty")

    src_style_label = parse_style_selector(args.source_style, style_names)
    start, end, sampled_label, src_run = choose_segment(
        labels=labels,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=src_style_label,
        start_index=args.start_index,
    )
    target_label = parse_style_selector(args.target_style, style_names)
    if target_label is None:
        target_label = int(sampled_label)

    x_ref_raw = np.asarray(x_source[start:end], dtype=np.float32)
    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    x_ref_norm = (x_ref_raw - norm_mean[None, :]) / norm_std[None, :]

    model = build_mann_from_checkpoint(
        args.mann_ckpt,
        num_styles=len(style_names),
    ).to(device)

    benchmark_summary = None
    if args.mode == "reconstruct":
        x_gen_norm, benchmark_summary = reconstruct_sequence(
            model=model,
            x_ref_norm=x_ref_norm,
            style_label=target_label,
            device=device,
            benchmark=args.benchmark,
            warmup_frames=args.warmup_frames,
            target_fps=float(args.fps),
        )
    else:
        x_gen_norm, benchmark_summary = rollout_sequence(
            model=model,
            x_seed_norm=x_ref_norm[0],
            traj_plan_norm=x_ref_norm[:, :84],
            style_label=target_label,
            device=device,
            benchmark=args.benchmark,
            warmup_frames=args.warmup_frames,
            target_fps=float(args.fps),
        )

    x_gen_raw = x_gen_norm * norm_std[None, :] + norm_mean[None, :]
    motion_mse_norm = float(np.mean((x_gen_norm[:, 84:] - x_ref_norm[:, 84:]) ** 2))
    motion_mae_norm = float(np.mean(np.abs(x_gen_norm[:, 84:] - x_ref_norm[:, 84:])))

    args.out_dir = prepare_output_dir(args.out_dir)
    gen_npz_path, ref_npz_path = write_generated_reference_npz(
        out_dir=args.out_dir,
        prefix=args.prefix,
        generated_x=x_gen_raw,
        reference_x=x_ref_raw,
        generated_label=target_label,
        reference_label=sampled_label,
        style_names=style_names,
        coordinate_convert=coordinate_convert,
    )

    meta = {
        "model": "mann",
        "task": args.mode,
        "mann_ckpt": str(args.mann_ckpt),
        "norm_stats": str(args.norm_stats),
        "db": str(args.db),
        "db_format": "memmap",
        "segment_start": int(start),
        "segment_end": int(end),
        "source_label": int(sampled_label),
        "source_style": style_names[sampled_label],
        "target_label": int(target_label),
        "target_style": style_names[target_label],
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "vis_backend": str(args.vis_backend),
        "coordinate_convert": bool(coordinate_convert),
        "generated_npz": str(gen_npz_path),
        "reference_npz": str(ref_npz_path),
        "motion_mse_norm": motion_mse_norm,
        "motion_mae_norm": motion_mae_norm,
    }
    if benchmark_summary is not None:
        meta["benchmark"] = benchmark_summary
    if args.gif_path is not None:
        meta["visualization_gif"] = str(args.gif_path)
    meta_path = write_meta_json(
        out_dir=args.out_dir,
        prefix=args.prefix,
        meta=meta,
    )

    print(f"[Info] model           : mann")
    print(f"[Info] mode            : {args.mode}")
    print(f"[Info] device          : {device}")
    print(f"[Info] ckpt            : {args.mann_ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] source segment  : [{start}, {end}) label={sampled_label} ({style_names[sampled_label]})")
    print(f"[Info] target style    : {style_names[target_label]} (label={target_label})")
    print(f"[Info] motion mse(norm): {motion_mse_norm:.6f}")
    print(f"[Info] motion mae(norm): {motion_mae_norm:.6f}")
    print(f"[Info] generated npz   : {gen_npz_path}")
    print(f"[Info] reference npz   : {ref_npz_path}")
    print(f"[Info] meta json       : {meta_path}")
    if benchmark_summary is not None:
        _print_benchmark_summary(benchmark_summary)

    if args.visualize or args.gif_path is not None:
        vis_title = (
            f"MANN {args.mode} | src={style_names[sampled_label]} | "
            f"target={style_names[target_label]} | seg=[{start},{end})"
        )
        vis_info = visualize_pair_from_375(
            ref_features_375=x_ref_raw,
            pred_features_375=x_gen_raw,
            vis_backend=args.vis_backend,
            db_path=args.db,
            data_root=args.data_root,
            bvh_ref=args.bvh_ref,
            fps=args.fps,
            dt=args.dt,
            use_global=not args.local_only,
            root_wy_in_degrees=True,
            coordinate_convert=coordinate_convert,
            trail_len=min(120, args.num_frames),
            title=vis_title,
            gif_path=args.gif_path,
            show=args.visualize,
            geno_gap=args.geno_gap,
        )
        if vis_info["reference_bvh"] is not None:
            print(f"[Info] visualize bvh  : {vis_info['reference_bvh']}")
        if args.gif_path is not None:
            print(f"[Info] gif path       : {args.gif_path}")


if __name__ == "__main__":
    main()
