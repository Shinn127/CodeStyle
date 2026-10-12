"""Why does TF→rollout training degrade displayed motion quality? Pure-inference diagnosis.

For each checkpoint (and the fresh VQ encode/decode ceiling on the same clips) this captures,
with no gradients and the exact inference path (clip → requantize → decode):

  rollout (240 frames): per-step z_hat, clipped latent, quantized latent q, displayed decode
      D(q) and continuous decode D(clip(z_hat))
  teacher-forced (one step per frame from GT history): same latents + code agreement with the
      GT quantized latent

Reported per clip set:
  - disp_jitter : paper jitter metric on the displayed motion (2nd local-position difference)
  - cont_jitter : same metric on the continuous (no-quantize) decode
  - step switch/steady : mean local-position frame step (cm) on code-switch vs steady frames
  - any_switch  : fraction of frames with >=1 head changing code
  - resid       : mean per-dim |z_clip - q| / scale (normalized quantization residual)
  - tf_agree    : TF-only, per-head and full-vector code agreement with the GT latent codes
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
from Utils.runtime import load_checkpoint, load_norm_stats, resolve_device, configure_torch_runtime, set_seed
from evaluation.simulate_realtime_flowmatch_control import (
    _resolve_base_ckpt_path, _resolve_norm_stats_path, _build_flow_controller,
)


def pick_clips(segments, names, styles, frames, clips_per_style, split, rng):
    pool = [s for s in segments if s["num_frames"] >= frames]
    if split != "any":
        pool = [s for s in pool if s["split"] == split]
    if styles:
        wanted = [int(e) if str(e).isdigit() else names.index(e) for e in styles]
        pool = [s for s in pool if s["label"] in wanted]
    chosen, used = [], {}
    for seg in (pool[i] for i in rng.permutation(len(pool))):
        count = used.get(seg["label"], 0)
        if count < clips_per_style:
            start = int(seg["frame_start"]) + int(rng.integers(0, seg["num_frames"] - frames + 1))
            chosen.append((seg, start))
            used[seg["label"]] = count + 1
    return chosen


def nearest_indices(quantizer, z: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Per-head nearest-code indices for latents [B, heads*code_dim] (shared codebook)."""
    embed = quantizer.embedding_weight
    batch = z.size(0)
    heads = quantizer.num_codebooks
    dim = quantizer.code_dim
    x = z.reshape(batch * heads, dim)
    embed_norm = embed.square().sum(-1)
    out = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for begin in range(0, x.size(0), chunk * heads):
        piece = x[begin : begin + chunk * heads]
        dist = piece.square().sum(-1, keepdim=True) + embed_norm[None] - 2.0 * (piece @ embed.t())
        out[begin : begin + chunk * heads] = dist.argmin(dim=-1)
    return out.reshape(batch, heads)


def local_jitter_cm(motion_real: np.ndarray) -> float:
    """Paper jitter: mean 2nd difference of the 24 selected local positions (cm)."""
    local = motion_real[:, 3:75].reshape(-1, 24, 3) * 100.0
    if local.shape[0] < 3:
        return float("nan")
    d2 = local[2:] - 2.0 * local[1:-1] + local[:-2]
    return float(np.linalg.norm(d2, axis=-1).mean())


def local_steps_cm(motion_real: np.ndarray) -> np.ndarray:
    local = motion_real[:, 3:75].reshape(-1, 24, 3) * 100.0
    return np.linalg.norm(np.diff(local, axis=0), axis=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ROOT / "database_100sty_375_memmap_nomirror")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noise-std", type=float, default=1.0,
                    help="Scale of the per-frame sampling noise (paper protocol uses 1.0).")
    ap.add_argument("--z0-zero", action="store_true",
                    help="Start Euler from the zero normalized latent instead of fresh noise, to "
                         "separate per-frame sampling noise from the learned map's own jitter.")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.db = args.db.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    device = resolve_device(args.device)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision="highest")
    set_seed(args.seed)

    checkpoint = load_checkpoint(args.ckpt, map_location="cpu")
    config = checkpoint.get("args", {})
    kind, base_path = _resolve_base_ckpt_path(None, None, flow_args=config, flow_ckpt_path=args.ckpt)
    norm_path = _resolve_norm_stats_path(None, flow_args=config, flow_ckpt_path=args.ckpt, base_ckpt_path=base_path)
    mean, std = load_norm_stats(norm_path)
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)

    names = json.loads((args.db / "StyleNames.json").read_text())
    meta = json.loads((args.db / "meta.json").read_text())
    segments = json.loads((args.db / "segments.json").read_text())
    x_all = np.memmap(args.db / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))

    vq = build_vqvae_from_checkpoint(base_path).to(device)
    model = _build_flow_controller(flow_args=config, flow_ckpt=checkpoint, base_model=vq,
                                   base_model_type=kind, num_styles=int(config.get("num_styles", len(names))),
                                   device=device)
    model.eval()
    quantizer = vq.quantizer
    scale_t = model.latent_scale

    rng = np.random.default_rng(args.seed)
    chosen = pick_clips(segments, names, None, args.frames + 1, 1, "any", rng)[: args.clips]

    rollout_rows, tf_rows, ceiling_rows = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for seg, start in chosen:
            raw = np.asarray(x_all[start : start + args.frames + 1], dtype=np.float32)
            xr = ((raw - mean) / std).astype(np.float32)
            xref = torch.from_numpy(xr).to(device)
            label = int(seg["label"])
            style_t = torch.tensor([label], device=device)

            # ---------------- GT / VQ ceiling ----------------
            z_q_gt, idx_gt, _ = vq.encode(xref[None])                 # [1,T,128], [1,T,32]
            disp_gt = (vq.decode(z_q_gt)[0] * std_t[84:] + mean_t[84:]).cpu().numpy()
            z_e_gt = vq.encode_pre_vq(xref[None])[0]
            cont_gt = (vq.decode(z_e_gt[None])[0] * std_t[84:] + mean_t[84:]).cpu().numpy()
            idx_gt = idx_gt[0]
            gt_switch = (idx_gt[1:] != idx_gt[:-1]).any(-1).cpu().numpy()
            steps_gt = local_steps_cm(disp_gt)
            ceiling_rows.append(dict(
                disp_jitter=local_jitter_cm(disp_gt), cont_jitter=local_jitter_cm(cont_gt),
                any_switch=float(gt_switch.mean()),
                step_switch=float(steps_gt[gt_switch].mean()) if gt_switch.any() else float("nan"),
                step_steady=float(steps_gt[~gt_switch].mean()) if (~gt_switch).any() else float("nan")))

            # ---------------- rollout (inference path) ----------------
            gen = torch.Generator(device=device).manual_seed(args.seed + label)
            x_curr = xref[0:1]
            prev_latent = model.frame_latent(x_curr)
            style_feature = model.encode_conditions(x_curr=x_curr, style_label=style_t)[2]
            disp, cont, resid, switch = [], [], [], []
            prev_q = None
            for i in range(1, args.frames + 1):
                traj_next = xref[i : i + 1, :84]
                if args.z0_zero:
                    z0 = model.latent_offset.detach().reshape(1, -1).clone()
                else:
                    noise = torch.randn((1, model.latent_dim), device=device, generator=gen)
                    z0 = model.latent_offset + model.latent_scale * model.latent_transport_std * float(args.noise_std) * noise
                z_hat = model.sample_latent(x_curr=x_curr, style_label=None, style_feature=style_feature,
                                            prev_latent=prev_latent, z0=z0, num_steps=8, solver="euler")
                z_clip = model.clip_latent_to_data_bounds(z_hat)
                q = model.requantize_latent(z_clip)
                m_disp = model.vqvae.decode(q)
                m_cont = model.vqvae.decode(z_clip)
                disp.append(m_disp[0].cpu().numpy())
                cont.append(m_cont[0].cpu().numpy())
                resid.append(float(((z_clip - q) / scale_t).abs().mean()))
                if prev_q is not None:
                    switch.append(bool((q != prev_q).any()))
                prev_q = q
                x_curr = torch.cat((traj_next, m_disp), dim=-1)
                prev_latent = q
            disp = np.asarray(disp, np.float32) * std[84:] + mean[84:]
            cont = np.asarray(cont, np.float32) * std[84:] + mean[84:]
            switch = np.asarray(switch)
            steps = local_steps_cm(disp)
            rollout_rows.append(dict(
                disp_jitter=local_jitter_cm(disp), cont_jitter=local_jitter_cm(cont),
                any_switch=float(switch.mean()), resid=float(np.mean(resid)),
                step_switch=float(steps[switch].mean()) if switch.any() else float("nan"),
                step_steady=float(steps[~switch].mean()) if (~switch).any() else float("nan")))

            # ---------------- teacher-forced one-step ----------------
            gen_tf = torch.Generator(device=device).manual_seed(args.seed + 1000 + label)
            agree_head, agree_full, tf_resid = [], [], []
            for i in range(1, args.frames + 1):
                xc = xref[i - 1 : i]
                if args.z0_zero:
                    z0 = model.latent_offset.detach().reshape(1, -1).clone()
                else:
                    noise = torch.randn((1, model.latent_dim), device=device, generator=gen_tf)
                    z0 = model.latent_offset + model.latent_scale * model.latent_transport_std * float(args.noise_std) * noise
                z_hat = model.sample_latent(x_curr=xc, style_label=style_t, prev_latent=None,
                                            z0=z0, num_steps=8, solver="euler")
                z_clip = model.clip_latent_to_data_bounds(z_hat)
                q = model.requantize_latent(z_clip)
                idx_hat = nearest_indices(quantizer, q)
                match = idx_hat == idx_gt[i]
                agree_head.append(float(match.float().mean()))
                agree_full.append(float(match.all(-1).float().mean()))
                tf_resid.append(float(((z_clip - q) / scale_t).abs().mean()))
            tf_rows.append(dict(tf_agree_head=float(np.mean(agree_head)),
                                tf_agree_full=float(np.mean(agree_full)),
                                tf_resid=float(np.mean(tf_resid))))

    def agg(rows, key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = dict(
        tag=args.tag, checkpoint=str(args.ckpt), clips=len(chosen), frames=args.frames, seed=args.seed,
        z0_zero=bool(args.z0_zero), noise_std=float(args.noise_std),
        wall_sec=time.perf_counter() - t0,
        rollout={k: agg(rollout_rows, k) for k in rollout_rows[0]},
        tf={k: agg(tf_rows, k) for k in tf_rows[0]},
        vq_ceiling={k: agg(ceiling_rows, k) for k in ceiling_rows[0]},
        per_clip=dict(rollout=rollout_rows, tf=tf_rows, ceiling=ceiling_rows),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: summary[k] for k in ("tag", "clips", "rollout", "tf", "vq_ceiling")},
                     indent=2, ensure_ascii=False, default=float))
    print(f"[Done] {args.out_json}  ({summary['wall_sec']:.0f}s)")


if __name__ == "__main__":
    main()
