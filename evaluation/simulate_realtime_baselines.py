"""
Unified reconstruction visualization test script for the cleaned baseline set.

Supported models:
  - basic_vqvae
  - lmm
  - mvae

The script will:
  1) sample a contiguous segment from DB (default 300 frames)
  2) reconstruct motion for the selected model
  3) write reconstructed/reference NPZ files
  4) show side-by-side visualization:
     left = reference, right = reconstruction
     backend = GenoView or Matplotlib quick view
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch

from Utils.model_loading import (
    build_lmm_from_checkpoint,
    build_mvae_from_checkpoint,
    build_vqvae_from_checkpoint,
)
from Utils.realtime import (
    choose_segment,
    load_db_arrays,
    parse_style_selector,
    prepare_output_dir,
    resolve_db_coordinate_convert,
    write_reconstructed_reference_npz,
    write_meta_json,
)
from Utils.runtime import load_norm_stats, resolve_device, set_seed
from arch import (
    BasicVQVAE375to291,
    LMM375to291,
    MVAE375to291,
)
from Visualization.backends.dispatch_backend import canonicalize_vis_backend, visualize_pair_from_375


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
VQVAEModel = BasicVQVAE375to291


def run_vqvae_reconstruct(
    model: VQVAEModel,
    x_norm: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x_ref_t = torch.from_numpy(x_norm).to(device=device, dtype=torch.float32)
        motion_hat = model.reconstruct(x_ref_t)  # [T,291]
        x_hat = torch.cat((x_ref_t[:, :84], motion_hat), dim=-1)  # [T,375]
    return x_hat.detach().cpu().numpy().astype(np.float32, copy=False)


def run_lmm_reconstruct(model: LMM375to291, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    traj = torch.from_numpy(x_norm[:, :84]).to(device=device, dtype=torch.float32)
    motion = torch.from_numpy(x_norm[:, 84:]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        motion_hat = model(motion=motion, traj=traj)  # [T,291]
        x_hat = torch.cat((traj, motion_hat), dim=-1)
    return x_hat.detach().cpu().numpy().astype(np.float32, copy=False)


def _prefer_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_mvae_condition_window(sequence: np.ndarray, end_idx: int, c: int) -> np.ndarray:
    """
    Return reference condition [C,375] ending right before `end_idx`.
    """
    start = end_idx - c
    return sequence[start:end_idx]


def run_mvae_reconstruct(model: MVAE375to291, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    c = model.num_condition_frames
    f = model.num_future_predictions
    if f != 1:
        raise ValueError(
            f"Current unified script supports MVAE only when num_future_predictions=1, got {f}"
        )
    t_total = x_norm.shape[0]
    if t_total <= c:
        raise ValueError(f"Need num_frames > num_condition_frames ({c}), got {t_total}")

    seq = np.array(x_norm, copy=True)

    with torch.no_grad():
        for i in range(c, t_total):
            cond_np = _build_mvae_condition_window(x_norm, end_idx=i, c=c)  # [C,375]
            cond = torch.from_numpy(cond_np[None, ...]).to(device=device, dtype=torch.float32)  # [1,C,375]
            x_future = torch.from_numpy(x_norm[i : i + 1][None, ...]).to(device=device, dtype=torch.float32)  # [1,1,375]
            traj_next = torch.from_numpy(x_norm[i : i + 1, :84]).to(device=device, dtype=torch.float32)  # [1,84]

            mu, _ = model.encode(x_future=x_future, condition=cond)  # deterministic latent
            motion_next = model.sample(z=mu, future_traj=traj_next, condition=cond)  # [1,291]
            x_next = torch.cat((traj_next, motion_next), dim=-1)
            seq[i] = x_next[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return seq


def default_ckpt_for_model(model_name: str) -> Path:
    if model_name == "basic_vqvae":
        return SCRIPT_DIR / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048" / "best.pt"
    if model_name == "lmm":
        return _prefer_existing_path(
            SCRIPT_DIR / "checkpoints" / "lmm_375to291" / "best.pt",
            SCRIPT_DIR / "checkpoints" / "lmm_ae_291to84plus32" / "best.pt",
        )
    if model_name == "mvae":
        return SCRIPT_DIR / "checkpoints" / "mvae_375to291" / "best.pt"
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified baseline motion reconstruction visualization test")
    parser.add_argument(
        "--model",
        type=str,
        choices=["basic_vqvae", "lmm", "mvae"],
        required=True,
        help="Model type to test",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint path for selected model. Default uses standard checkpoints/<model>/best.pt",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror",
        help="Input DB path (memmap directory)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz. Default: sibling of ckpt (norm_stats.npz)",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Segment length")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for segment sampling")
    parser.add_argument(
        "--style",
        type=str,
        default="",
        help="Style selector for sampled segment (name or label id)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional fixed start index in DB (must keep same label for num_frames)",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "reconstruction_vis",
        help="Directory for reconstructed npz and metadata",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Default: <model>_reconstruct_<num_frames>",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="Optional output GIF path for side-by-side visualization",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Show side-by-side visualization after generation (default true)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization window",
    )
    parser.add_argument("--fps", type=int, default=60, help="Visualization fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Visualization dt")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Visualize local positions only (skip global root reconstruction)",
    )
    parser.add_argument(
        "--vis-backend",
        type=str,
        choices=["genoview", "quick", "matplotlib"],
        default="genoview",
        help="Visualization backend. 'quick' uses the Matplotlib quick-visualize path; 'matplotlib' is kept as an alias of 'quick'.",
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
        help="100STYLE BVH root used to auto-pick a reference BVH for topology",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional BVH ref for deriving 24-joint parent topology",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vis_backend = canonicalize_vis_backend(args.vis_backend)
    if args.ckpt is None:
        args.ckpt = default_ckpt_for_model(args.model)
    args.ckpt = args.ckpt.expanduser().resolve()
    if not args.ckpt.exists():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if args.norm_stats is None:
        args.norm_stats = args.ckpt.parent / "norm_stats.npz"
    args.norm_stats = args.norm_stats.expanduser().resolve()
    if args.gif_path is not None:
        args.gif_path = args.gif_path.expanduser().resolve()
    if args.prefix is None:
        args.prefix = f"{args.model}_reconstruct_{args.num_frames}"

    if args.num_frames < 2:
        raise ValueError("--num-frames must be >= 2")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")

    set_seed(args.seed)

    device = resolve_device(args.device)
    coordinate_convert = resolve_db_coordinate_convert(args.db)
    task = "reconstruct"
    print(f"[Info] model           : {args.model}")
    print(f"[Info] task            : {task}")
    print(f"[Info] device          : {device}")
    print(f"[Info] ckpt            : {args.ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] vis backend     : {args.vis_backend}")
    print(f"[Info] coord convert   : {coordinate_convert}")
    x_source, labels, style_names = load_db_arrays(args.db)
    if labels.size == 0:
        raise RuntimeError("DB is empty")
    print("[Info] db format       : memmap")
    print(f"[Info] db frames       : {labels.shape[0]}")
    print(f"[Info] num styles      : {len(style_names)}")

    style_label = parse_style_selector(args.style, style_names)
    start, end, sampled_label, src_run = choose_segment(
        labels=labels,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=style_label,
        start_index=args.start_index,
    )
    x_ref_raw = np.asarray(x_source[start:end], dtype=np.float32)
    if x_ref_raw.shape != (args.num_frames, 375):
        raise RuntimeError(
            f"Unexpected sampled segment shape {x_ref_raw.shape}, expected {(args.num_frames, 375)}"
        )

    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    x_ref_norm = (x_ref_raw - norm_mean[None, :]) / norm_std[None, :]

    if args.model == "basic_vqvae":
        model = build_vqvae_from_checkpoint(args.ckpt).to(device)
        x_recon_norm = run_vqvae_reconstruct(
            model=model,
            x_norm=x_ref_norm,
            device=device,
        )
    elif args.model == "lmm":
        model = build_lmm_from_checkpoint(args.ckpt).to(device)
        x_recon_norm = run_lmm_reconstruct(model=model, x_norm=x_ref_norm, device=device)
    elif args.model == "mvae":
        model = build_mvae_from_checkpoint(args.ckpt).to(device)
        x_recon_norm = run_mvae_reconstruct(model=model, x_norm=x_ref_norm, device=device)
    else:
        raise RuntimeError(f"Unhandled model: {args.model}")

    x_recon_raw = x_recon_norm * norm_std[None, :] + norm_mean[None, :]

    mse_norm = float(np.mean((x_recon_norm[:, 84:] - x_ref_norm[:, 84:]) ** 2))
    mae_norm = float(np.mean(np.abs(x_recon_norm[:, 84:] - x_ref_norm[:, 84:])))
    print(f"[Info] motion mse(norm): {mse_norm:.6f}")
    print(f"[Info] motion mae(norm): {mae_norm:.6f}")

    args.out_dir = prepare_output_dir(args.out_dir)
    recon_npz_path, ref_npz_path = write_reconstructed_reference_npz(
        out_dir=args.out_dir,
        prefix=args.prefix,
        reconstructed_x=x_recon_raw,
        reference_x=x_ref_raw,
        reconstructed_label=sampled_label,
        reference_label=sampled_label,
        style_names=style_names,
        coordinate_convert=coordinate_convert,
    )

    meta = {
        "model": args.model,
        "task": task,
        "ckpt": str(args.ckpt),
        "norm_stats": str(args.norm_stats),
        "db": str(args.db),
        "db_format": "memmap",
        "segment_start": int(start),
        "segment_end": int(end),
        "sampled_label": int(sampled_label),
        "sampled_style": style_names[sampled_label] if 0 <= sampled_label < len(style_names) else f"label_{sampled_label}",
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "num_frames": int(args.num_frames),
        "vis_backend": str(args.vis_backend),
        "coordinate_convert": bool(coordinate_convert),
        "reconstructed_npz": str(recon_npz_path),
        "reference_npz": str(ref_npz_path),
        "motion_mse_norm": mse_norm,
        "motion_mae_norm": mae_norm,
    }
    if args.gif_path is not None:
        meta["visualization_gif"] = str(args.gif_path)
    meta_path = write_meta_json(
        out_dir=args.out_dir,
        prefix=args.prefix,
        meta=meta,
    )

    print(f"[Info] source segment  : [{start}, {end}) label={sampled_label} ({meta['sampled_style']})")
    print(f"[Info] reconstructed npz: {recon_npz_path}")
    print(f"[Info] reference npz   : {ref_npz_path}")
    print(f"[Info] meta json       : {meta_path}")

    if args.visualize or args.gif_path is not None:
        vis_title = (
            f"Baseline reconstruction | model={args.model} | "
            f"style={meta['sampled_style']} (label={sampled_label}) | "
            f"seg=[{start},{end})"
        )
        print("[Info] visualize pair : left=reference | right=reconstruction")
        vis_info = visualize_pair_from_375(
            ref_features_375=x_ref_raw,
            pred_features_375=x_recon_raw,
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
            print(f"[Done] GIF saved      : {args.gif_path}")
        if args.visualize:
            print("[Done] Visualization complete.")
        else:
            print("[Done] GIF export complete.")
    else:
        print("[Done] Generation complete (visualization skipped).")


if __name__ == "__main__":
    main()
