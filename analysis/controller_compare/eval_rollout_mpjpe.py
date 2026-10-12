"""Authoritative MPJPE on the sealed-test matched sources (same manifest, same noise plan).

The final style/ADE report does not include MPJPE, and the 90-clip quick probe uses a different
source sample; this runs the exact evaluation path (`reconstruct_motion_batch` + `case_noise_plan`,
240-frame autoregressive rollout, Euler8, std=1, clip->requantize->decode) over the 1000 matched
sources recorded in `case_manifest.json`, then reports global and local MPJPE per generated frame
(GT context frame excluded) with clustered bootstrap CIs, plus RootADE as a sanity check.

Usage:
  python analysis/controller_compare/eval_rollout_mpjpe.py \
      --flow-ckpt checkpoints/<your_run>/best.pt --tag myrun \
      --out-json analysis/motion_quality_diagnosis/probes/mpjpe_myrun.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.motion_eval import MotionEvalConfig, compute_sequence_metric_sums, load_motion_model, reconstruct_motion_batch
from Utils.motion375 import reconstruct_root_track_from_375, reconstruct_selected_globals_from_375
from Utils.reproducibility import case_noise_plan, cluster_interval
from Utils.runtime import configure_torch_runtime, load_checkpoint, load_norm_stats, resolve_device, set_seed
from evaluation.simulate_realtime_flowmatch_control import _resolve_base_ckpt_path, _resolve_norm_stats_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-ckpt", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "eval_results/6tok_rollout/seen/case_manifest.json")
    p.add_argument("--db", type=Path, default=ROOT / "database_100sty_375_memmap_nomirror")
    p.add_argument("--horizon", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=0, help="Euler steps; 0 follows the checkpoint solver_steps")
    p.add_argument("--noise-std", type=float, default=1.0,
                   help="Sampling noise scale. 1.0 is the formal protocol; 0.5/0.25 describe the display settings.")
    p.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=True,
                   help="Feedback clip->quantize->decode (formal); --no-requantize uses continuous AE feedback")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.flow_ckpt = args.flow_ckpt.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    set_seed(42)
    device = resolve_device(args.device)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision="highest")

    rows = json.loads(args.manifest.read_text())["horizons"][str(args.horizon)]
    starts = np.array([int(r["source_start"]) for r in rows], dtype=np.int64)
    labels = np.array([int(r["source_label"]) for r in rows], dtype=np.int64)
    groups = np.array([int(r["source_entry_id"]) for r in rows], dtype=np.int64)
    n_sources = len(starts)
    print(f"[Info] manifest sources: {n_sources} (horizon={args.horizon}, split={json.loads(args.manifest.read_text()).get('source_split')}, "
          f"noise_std={args.noise_std})")

    checkpoint = load_checkpoint(args.flow_ckpt, map_location="cpu")
    config = checkpoint.get("args", {})
    kind, base_path = _resolve_base_ckpt_path(None, None, flow_args=config, flow_ckpt_path=args.flow_ckpt)
    norm_path = _resolve_norm_stats_path(None, flow_args=config, flow_ckpt_path=args.flow_ckpt, base_ckpt_path=base_path)
    mean, std = load_norm_stats(norm_path)
    num_styles = int(config.get("num_styles", 90))
    model = load_motion_model("flowmatch_target", args.flow_ckpt, num_styles=num_styles).to(device).eval()

    meta = json.loads((args.db / "meta.json").read_text())
    x_all = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))
    frames = args.horizon + 1  # one GT context frame + horizon generated frames

    mpjpe_global, mpjpe_local, root_ade, jitter_cm, fs_cm = [], [], [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, n_sources, args.batch_size):
            end = min(begin + args.batch_size, n_sources)
            idx = np.arange(begin, end)
            raw = np.stack([np.asarray(x_all[s : s + frames], dtype=np.float32) for s in starts[idx]])
            x = torch.from_numpy((raw - mean) / std).to(device=device, dtype=torch.float32)
            style = torch.from_numpy(labels[idx]).to(device).long()
            noise_plan = case_noise_plan(model, starts[idx].tolist(), labels[idx].tolist(), frames,
                                         seed=42, noise_std=args.noise_std)
            pred = reconstruct_motion_batch(
                model_name="flowmatch_target", model=model, x=x, style_label=style,
                flow_eval_mode="rollout", flow_solver="euler",
                flow_ode_steps=int(config.get("solver_steps", 8)) if args.ode_steps <= 0 else args.ode_steps,
                flow_noise_std=args.noise_std, flow_requantize_z=args.requantize, flow_latent_clip=True,
                noise_plan=noise_plan)
            ref_np = x.detach().cpu().numpy() * std[None, None, :] + mean[None, None, :]
            pred_np = pred.detach().cpu().numpy() * std[None, None, :] + mean[None, None, :]
            for i in range(end - begin):
                ref_seq = np.asarray(ref_np[i], np.float32)
                pred_seq = np.asarray(pred_np[i], np.float32)
                ref_g, _ = reconstruct_selected_globals_from_375(ref_seq, dt=1 / 60, root_wy_in_degrees=True, use_global=True)
                pred_g, _ = reconstruct_selected_globals_from_375(pred_seq, dt=1 / 60, root_wy_in_degrees=True, use_global=True)
                ref_l, _ = reconstruct_selected_globals_from_375(ref_seq, dt=1 / 60, root_wy_in_degrees=True, use_global=False)
                pred_l, _ = reconstruct_selected_globals_from_375(pred_seq, dt=1 / 60, root_wy_in_degrees=True, use_global=False)
                mpjpe_global.append(float(np.linalg.norm(pred_g[1:] - ref_g[1:], axis=-1).mean() * 100.0))
                mpjpe_local.append(float(np.linalg.norm(pred_l[1:] - ref_l[1:], axis=-1).mean() * 100.0))
                ref_p, _ = reconstruct_root_track_from_375(ref_seq, dt=1 / 60, root_wy_in_degrees=True)
                pred_p, _ = reconstruct_root_track_from_375(pred_seq, dt=1 / 60, root_wy_in_degrees=True)
                root_ade.append(float(np.linalg.norm(pred_p[1:, [0, 2]] - ref_p[1:, [0, 2]], axis=-1).mean() * 100.0))
                sums = compute_sequence_metric_sums(reference_375=ref_seq, prediction_375=pred_seq, cfg=MotionEvalConfig())
                jitter_cm.append(float(sums.jitter_sum_cm / max(sums.jitter_count, 1)))
                fs_cm.append(float(sums.foot_sliding_sum_cm / max(sums.foot_sliding_count, 1)))
            print(f"  batch {begin//args.batch_size+1}/{(n_sources+args.batch_size-1)//args.batch_size}: "
                  f"mpjpe={np.mean(mpjpe_global[begin:]):.2f}cm", flush=True)

    def summarize(values):
        values = np.asarray(values, np.float64)
        lo, hi = cluster_interval(values, groups, samples=1000, seed=42)
        return dict(mean=float(values.mean()), ci95=[float(lo), float(hi)])

    summary = dict(tag=args.tag, checkpoint=str(args.flow_ckpt), horizon=int(args.horizon),
                   noise_std=float(args.noise_std),
                   num_sources=int(n_sources), frames_evaluated=int(args.horizon),
                   mpjpe_cm=summarize(mpjpe_global), local_mpjpe_cm=summarize(mpjpe_local),
                   root_ade_cm=summarize(root_ade), jitter_cm=summarize(jitter_cm),
                   foot_sliding_cm=summarize(fs_cm), wall_sec=time.perf_counter() - t0,
                   per_source=dict(groups=groups.tolist(), mpjpe=mpjpe_global,
                                   local_mpjpe=mpjpe_local, root_ade=root_ade,
                                   jitter=jitter_cm, foot_sliding=fs_cm))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Summary] {args.tag} (std={args.noise_std}): MPJPE={summary['mpjpe_cm']['mean']:.2f}cm "
          f"CI[{summary['mpjpe_cm']['ci95'][0]:.2f},{summary['mpjpe_cm']['ci95'][1]:.2f}] "
          f"local={summary['local_mpjpe_cm']['mean']:.2f}cm "
          f"RootADE={summary['root_ade_cm']['mean']:.2f}cm "
          f"jitter={summary['jitter_cm']['mean']:.3f} F.S.={summary['foot_sliding_cm']['mean']:.3f} "
          f"({summary['wall_sec']:.0f}s, n={n_sources})")
    print(f"[Done] {args.out_json}")


if __name__ == "__main__":
    main()
