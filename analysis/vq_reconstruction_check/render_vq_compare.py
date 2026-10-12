"""Render reference vs VQ-reconstructed motion: side-by-side GIF, diagnostic panels, live GenoView.

Picks a random `--frames`-long clip segment for each requested style (fresh seed every run unless
`--seed` is given), then compares decode paths:
  - quantized : z_e -> codebook -> decode   (feedback path used by the controller rollout)
  - continuous: z_e -> decode               (AE upper bound; no code-switch noise)

Examples:
  # live mesh window per style: random 240-frame clip, ref | quantized side-by-side, loops until ESC
  python analysis/vq_reconstruction_check/render_vq_compare.py --backend genoview --styles Neutral March

  # same but continuous AE path, fixed seed for a reproducible clip
  python analysis/vq_reconstruction_check/render_vq_compare.py --backend genoview --styles March --path continuous --seed 42

  # headless: GIFs + static metrics into analysis/vq_reconstruction_check/render
  python analysis/vq_reconstruction_check/render_vq_compare.py --styles Neutral --frames 240
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.motion_eval import compute_sequence_metric_sums, MotionEvalConfig
from Utils.motion375 import reconstruct_selected_globals_from_375
from Utils.runtime import resolve_device, set_seed

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VQ = SCRIPT_DIR / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    p.add_argument("--vq-ckpt", type=Path, default=DEFAULT_VQ)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--split", default="any", choices=("train", "val", "test", "any"))
    p.add_argument("--styles", nargs="*", default=None, help="Style names or label ids; default picks spread-out segments")
    p.add_argument("--clips-per-style", type=int, default=1)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--seed", type=int, default=None,
                   help="None draws a fresh random 240-frame clip every run; set an int to reproduce a run")
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", default="quick", choices=("quick", "genoview", "none"), help="quick=GIF files, genoview=live mesh window")
    p.add_argument("--path", default="quantized", choices=("quantized", "continuous"),
                   help="GenoView window only: which reconstruction to show on the right")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "analysis" / "vq_reconstruction_check" / "render")
    return p.parse_args()


def pick_clips(segments, names, styles, frames, clips_per_style, split, rng):
    """Pick `clips_per_style` random segments per requested style and a random `frames`-long window in each."""
    pool = [s for s in segments if s["num_frames"] >= frames]
    if split != "any":
        pool = [s for s in pool if s["split"] == split]
    if styles:
        wanted = []
        for entry in styles:
            wanted.append(int(entry) if str(entry).isdigit() else names.index(entry))
        pool = [s for s in pool if s["label"] in wanted]
    chosen, used = [], {}
    for seg in (pool[i] for i in rng.permutation(len(pool))):
        count = used.get(seg["label"], 0)
        if count < clips_per_style:
            start = int(seg["frame_start"]) + int(rng.integers(0, seg["num_frames"] - frames + 1))
            chosen.append((seg, start))
            used[seg["label"]] = count + 1
    return chosen


def main():
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vq_ckpt = args.vq_ckpt.expanduser().resolve()
    norm_path = Path(args.norm_stats).expanduser().resolve() if args.norm_stats else args.vq_ckpt.parent / "norm_stats.npz"
    device = resolve_device(args.device)
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    print(f"[Info] seed={seed} split={args.split} frames={args.frames}", flush=True)

    meta = json.loads((args.db / "meta.json").read_text())
    segments = json.loads((args.db / "segments.json").read_text())
    names = json.loads((args.db / "StyleNames.json").read_text())
    norm = np.load(norm_path)
    mean, std = norm["mean"].astype(np.float32), norm["std"].astype(np.float32)
    x_all = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))

    chosen = pick_clips(segments, names, args.styles, args.frames, args.clips_per_style, args.split, rng)
    if not chosen:
        raise SystemExit("No segments matched the request")

    model = build_vqvae_from_checkpoint(args.vq_ckpt).to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = MotionEvalConfig()
    summary = []

    for seg, start in chosen:
        raw = np.asarray(x_all[start : start + args.frames], dtype=np.float32)
        x = torch.from_numpy((raw - mean) / std).to(device)[None]
        with torch.no_grad():
            z_e = model.encode_pre_vq(x)
            z_q, indices, _ = model.encode(x)
            motion_q = model.decode(z_q)[0].cpu().numpy()
            motion_ae = model.decode(z_e)[0].cpu().numpy()
        indices = indices[0].cpu().numpy()
        switch = (indices[1:] != indices[:-1])

        row = dict(style=names[seg["label"]], suffix=seg["suffix"], segment_id=seg["source_entry_id"],
                   window_start=int(start), seed=int(seed))
        for tag, motion in (("quantized", motion_q), ("continuous", motion_ae)):
            pred = np.concatenate((raw[:, :84], motion * std[84:] + mean[84:]), axis=-1)
            sums = compute_sequence_metric_sums(reference_375=raw, prediction_375=pred, cfg=cfg)
            row[f"mpjpe_{tag}"] = sums.mpjpe_sum_cm / sums.mpjpe_count
            row[f"jitter_{tag}"] = sums.jitter_sum_cm / sums.jitter_count
        row["any_head_switch_rate"] = float(switch.any(-1).mean())
        row["switch_frames"] = int(switch.any(-1).sum())
        summary.append(row)
        print(f"[Clip] {row['style']:<16} {seg['suffix']:<4} mpjpe q={row['mpjpe_quantized']:.3f} ae={row['mpjpe_continuous']:.3f} "
              f"jitter q={row['jitter_quantized']:.3f} ae={row['jitter_continuous']:.3f} switch={row['any_head_switch_rate']:.2f}",
              flush=True)

        stem = f"{row['style'].replace(' ', '_')}_{seg['suffix']}_{seg['source_entry_id']}_s{int(start)}"
        if args.backend == "quick":
            from Visualization.io.genoview_bvh import _load_template, GENOVIEW_RESOURCES_DIR
            from Visualization.io.reference_bvh import build_reduced_parents
            from Visualization.backends.matplotlib_backend import visualize_side_by_side_from_375
            from Utils.motion375 import SELECTED_24_NAMES
            template = _load_template(GENOVIEW_RESOURCES_DIR / "Geno_bind.bvh")
            mapped = [list(template["names"]).index(n) for n in SELECTED_24_NAMES]
            parents = build_reduced_parents(template["parents"], mapped)
            for tag, motion in (("quantized", motion_q), ("continuous", motion_ae)):
                pred = np.concatenate((raw[:, :84], motion * std[84:] + mean[84:]), axis=-1)
                visualize_side_by_side_from_375(
                    ref_features_375=raw, pred_features_375=pred, parents=parents,
                    fps=60, dt=1 / 60, use_global=True,
                    title=f"{row['style']} {seg['suffix']} | VQ {tag} | ref left / reconstructed right",
                    save_path=args.out_dir / f"{stem}.{tag}.gif", show=False)
            print(f"[Render] gifs -> {args.out_dir}/{stem}.*.gif", flush=True)
        elif args.backend == "genoview":
            from Visualization.backends.dispatch_backend import visualize_pair_from_375
            motion = motion_q if args.path == "quantized" else motion_ae
            pred = np.concatenate((raw[:, :84], motion * std[84:] + mean[84:]), axis=-1)
            visualize_pair_from_375(
                ref_features_375=raw, pred_features_375=pred, vis_backend="genoview",
                fps=60, dt=1 / 60, use_global=True,
                title=f"VQ {args.path} reconstruction | {row['style']} {seg['suffix']} | left=ref right=recon",
                db_path=args.db, show=True)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    agg = {key: float(np.mean([r[key] for r in summary])) for key in summary[0] if key.endswith(("quantized", "continuous", "rate"))}
    print("[Summary]", json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"[Done] summary -> {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
