"""VQ realtime-quality diagnostics: per-frame code switching and quantized-vs-continuous decode.

Measures, per clip and aggregated:
  - per-head / any-head / all-head code-switch rates between consecutive frames
  - run lengths of unchanged code assignments (32 heads x 4 dims = 128 latent dims)
  - the latent gap E[(z_e - z_q)^2] over the sequence
  - per-frame decode jump at switch vs no-switch frames
  - the paper metric triple (MPJPE / Jitter / FootSliding) for both decode paths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.motion_eval import (
    MotionEvalConfig,
    compute_sequence_metric_sums,
    denormalize_motion_batch,
    reconstruct_selected_globals_from_375,
)
from Utils.runtime import resolve_device, set_seed

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VQ = SCRIPT_DIR / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    p.add_argument("--vq-ckpt", type=Path, default=DEFAULT_VQ)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--clips", type=int, default=30)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vq_ckpt = args.vq_ckpt.expanduser().resolve()
    norm_path = args.norm_stats or args.vq_ckpt.parent / "norm_stats.npz"
    norm_path = Path(norm_path).expanduser().resolve()
    device = resolve_device(args.device)
    set_seed(args.seed)

    meta = json.loads((args.db / "meta.json").read_text())
    segments = json.loads((args.db / "segments.json").read_text())
    names = json.loads((args.db / "StyleNames.json").read_text())
    norm = np.load(norm_path)
    mean, std = norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
    x_all = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))

    candidates = [s for s in segments if s["split"] == args.split and s["num_frames"] >= args.frames]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(candidates))
    chosen = [candidates[i] for i in order[: args.clips]]

    model = build_vqvae_from_checkpoint(args.vq_ckpt).to(device).eval()
    cfg = MotionEvalConfig()

    per_clip = []
    switch_runs_all = []
    with torch.no_grad():
        for seg in chosen:
            start = seg["frame_start"] + (seg["num_frames"] - args.frames) // 2
            raw = np.asarray(x_all[start : start + args.frames], dtype=np.float32)
            x = torch.from_numpy((raw - mean) / std).to(device)[None]  # [1,T,375]

            z_e = model.encode_pre_vq(x)[0]                       # [T,128]
            z_q, indices, perp = model.encode(x)                  # [1,T,128], [1,T,32]
            indices = indices[0].cpu().numpy()
            motion_q = model.decode(z_q)[0].cpu().numpy()          # [T,291]
            motion_ae = model.decode(z_e[None])[0].cpu().numpy()

            changed = indices[1:] != indices[:-1]                  # [T-1, heads]
            per_head = changed.mean(0)
            any_head = float(changed.any(-1).mean())
            all_head = float(changed.all(-1).mean())
            same_all = (~changed.any(-1))
            # longest run of frames where the full code vector is unchanged
            runs, run = [], 0
            for ok in same_all:
                if ok:
                    run += 1
                else:
                    runs.append(run + 1); run = 0
            runs.append(run + 1)
            switch_runs_all.append(np.asarray(runs))

            gap = float(((z_e - z_q) ** 2).mean())

            motion_std = std[84:].astype(np.float32)
            delta = np.diff(motion_q, axis=0) * motion_std[None] * 100.0  # cm
            per_frame = np.linalg.norm(delta[:, 3:75].reshape(-1, 24, 3), axis=-1).mean(-1)
            idle = ~changed.any(-1)
            jump_switch = float(per_frame[~idle].mean()) if (~idle).any() else float("nan")
            jump_steady = float(per_frame[idle].mean()) if idle.any() else float("nan")
            decode_diff = float(np.linalg.norm(((motion_q - motion_ae)[:, 3:75] * motion_std[3:75]).reshape(-1, 24, 3), axis=-1).mean() * 100.0)
            row = dict(style=names[seg["label"]], suffix=seg["suffix"], frames=int(args.frames),
                       perp=float(perp), gap=gap, any_head_switch=any_head, all_head_switch=all_head,
                       per_head_switch=float(per_head.mean()), jump_switch_cm=jump_switch,
                       jump_steady_cm=jump_steady, decode_diff_cm=decode_diff)
            for tag, motion in (("quantized", motion_q), ("continuous", motion_ae)):
                pred = np.concatenate((raw[:, :84], motion * motion_std + mean[84:]), axis=-1)
                s = compute_sequence_metric_sums(reference_375=raw, prediction_375=pred, cfg=cfg)
                row[f"mpjpe_{tag}"] = s.mpjpe_sum_cm / s.mpjpe_count
                row[f"jitter_{tag}"] = s.jitter_sum_cm / s.jitter_count
                row[f"fs_{tag}"] = s.foot_sliding_sum_cm / max(s.foot_sliding_count, 1)
            per_clip.append(row)
            print(f"[Clip] {row['style']:<16} {seg['suffix']:<4} switch(a/h)={any_head:.2f} "
                  f"jump sw/st={jump_switch:.3f}/{jump_steady:.3f}cm "
                  f"mpjpe q={row['mpjpe_quantized']:.3f} ae={row['mpjpe_continuous']:.3f}", flush=True)

    def agg(key):
        vals = [r[key] for r in per_clip if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    runs = np.concatenate(switch_runs_all)
    summary = dict(
        split=args.split, clips=len(per_clip), frames=args.frames, checkpoint=str(args.vq_ckpt),
        code_switch=dict(
            per_head_rate=agg("per_head_switch"),
            any_head_rate=agg("any_head_switch"),
            all_head_rate=agg("all_head_switch"),
            unchanged_run_frames_mean=float(runs.mean()),
            unchanged_run_frames_p50=float(np.percentile(runs, 50)),
            unchanged_run_frames_p95=float(np.percentile(runs, 95)),
            fraction_runs_len1=float((runs == 1).mean()),
        ),
        latent_gap=agg("gap"),
        decode=dict(
            jump_at_switch_cm=agg("jump_switch_cm"),
            jump_steady_cm=agg("jump_steady_cm"),
            quantized_vs_continuous_diff_cm=agg("decode_diff_cm"),
        ),
        metrics=dict(
            mpjpe_quantized=agg("mpjpe_quantized"), mpjpe_continuous=agg("mpjpe_continuous"),
            jitter_quantized=agg("jitter_quantized"), jitter_continuous=agg("jitter_continuous"),
            foot_sliding_quantized=agg("fs_quantized"), foot_sliding_continuous=agg("fs_continuous"),
        ),
        per_clip=per_clip,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, indent=2, ensure_ascii=False, default=float))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
        print(f"[Done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
