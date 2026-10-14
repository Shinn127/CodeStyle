"""Decompose the single-step code error into solver / noise / multimodality components.

Companion to codebook_cause.py. Everything here is TF-anchored (GT trajectory, GT prev
latent, GT history, std0 unless stated), 12 test clips.

A. solver sweep   : single-step code match vs Euler step count K in {2,4,8,16,32}
                    (the last Euler step of this target-mode sampler evaluates the model
                    at t=(K-1)/K and the final state equals that prediction exactly, so K
                    shifts where the flow is queried; K=4 queries t=0.75).
B. start noise    : K=4 / K=16 with stochastic z0 (std1) vs deterministic z0 (std0).
C. sample spread  : N=8 independent z0 per frame at K=4. Pairwise code agreement and
                    "any sample matches GT" measure how much of the flip is genuine
                    conditional multimodality vs a shared bias (mean-averaging).
D. GT frame-to-frame per-slot displacement vs code spacing (why GT switches 63%/frame).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/shinn/桌面/multimedia/revision/NewCode')
sys.path.insert(0, str(ROOT))
from Utils.motion_eval import load_motion_model
from Utils.runtime import load_norm_stats

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'


def code_idx(vq, z_raw):
    embed = vq.quantizer.embedding_weight.detach().float()
    x = z_raw.reshape(-1, vq.code_dim).float()
    return torch.cdist(x, embed).argmin(-1).reshape(z_raw.shape[0], vq.num_latents)


def pick_clips(n, frames):
    segs = json.loads((DB / 'segments.json').read_text())
    elig = [s for s in segs if s.get('split') == 'test' and not s.get('mirror', False)
            and s['num_frames'] >= frames + 5]
    by = {}
    for s in elig:
        by.setdefault(s['label'], s)
    styles = sorted(by)
    picks = np.linspace(0, len(styles) - 1, n).round().astype(int)
    return [by[styles[i]] for i in sorted(set(picks))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow-ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--spread-frames', type=int, default=60)
    ap.add_argument('--spread-samples', type=int, default=8)
    ap.add_argument('--ks', nargs='+', type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda')
    model = load_motion_model('flowmatch_target', ROOT / args.flow_ckpt, num_styles=90).to(dev)
    model.eval().requires_grad_(False)
    vq = model.vqvae
    mean, std = load_norm_stats(NORM)
    mean_t = torch.as_tensor(mean, device=dev)
    std_t = torch.as_tensor(std, device=dev)
    segs = pick_clips(args.clips, args.frames + 1)
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    res = dict(tag=args.tag, checkpoint=args.flow_ckpt, frames=args.frames, seed=args.seed)

    def z0_from(gen, noise):
        if noise <= 0:
            return torch.zeros(1, vq.latent_dim, device=dev)  # normalizes to 0
        eps = torch.randn((1, vq.latent_dim), device=dev, generator=gen)
        return model.latent_offset + model.latent_scale * model.latent_transport_std * noise * eps

    def single_step(x_norm, z_prev, hist, e_s, K, noise, gen):
        out = model.sample_latent(x_curr=x_norm, style_label=None, style_feature=e_s,
                                  prev_latent=z_prev, history_latent=hist,
                                  z0=z0_from(gen, noise), num_steps=K, solver='euler')
        return out

    # ---------- A: solver sweep ----------
    res['solver_sweep'] = {}
    for K in args.ks:
        t0 = time.perf_counter()
        accs = []
        for ci, s in enumerate(segs):
            rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
            x = torch.as_tensor((rows - mean) / std, device=dev)
            with torch.no_grad():
                z_all, _, _ = vq.encode(x)
                e_s = model.encode_conditions(x_curr=x[0:1], style_label=torch.tensor([s['label']], device=dev))[2]
                gen = torch.Generator(device=dev).manual_seed(args.seed + ci)
                idxs = []
                for k in range(1, args.frames + 1):
                    out = single_step(x[k - 1:k], z_all[k - 1:k],
                                      z_all[max(k - 2, 0):max(k - 2, 0) + 1], e_s, K, 0.0, gen)
                    idxs.append(code_idx(vq, out)[0])
                idxs = torch.stack(idxs)
            gt = code_idx(vq, z_all[1:args.frames + 1])
            accs.append(float((idxs == gt).float().mean()))
        # per-slot error of the final prediction, for the same K
        res['solver_sweep'][str(K)] = dict(match=float(np.mean(accs)), match_std=float(np.std(accs)),
                                           wall_sec=time.perf_counter() - t0)
        print(f"[solver K={K:2d}] match={np.mean(accs):.3f} wall={time.perf_counter()-t0:.0f}s")

    # ---------- B: start noise at K=4 and K=16 ----------
    res['noise'] = {}
    for K in (4, 16):
        for noise in (0.0, 1.0):
            accs = []
            for ci, s in enumerate(segs):
                rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
                x = torch.as_tensor((rows - mean) / std, device=dev)
                with torch.no_grad():
                    z_all, _, _ = vq.encode(x)
                    e_s = model.encode_conditions(x_curr=x[0:1], style_label=torch.tensor([s['label']], device=dev))[2]
                    gen = torch.Generator(device=dev).manual_seed(args.seed + ci)
                    idxs = []
                    for k in range(1, args.frames + 1):
                        out = single_step(x[k - 1:k], z_all[k - 1:k],
                                          z_all[max(k - 2, 0):max(k - 2, 0) + 1], e_s, K, noise, gen)
                        idxs.append(code_idx(vq, out)[0])
                    idxs = torch.stack(idxs)
                gt = code_idx(vq, z_all[1:args.frames + 1])
                accs.append(float((idxs == gt).float().mean()))
            res['noise'][f'K{K}_std{noise:g}'] = float(np.mean(accs))
            print(f"[noise K={K} std={noise:g}] match={np.mean(accs):.3f}")

    # ---------- C: sample spread ----------
    N = args.spread_samples
    pair_all, any_all = [], []
    pair_stable, pair_trans, any_stable, any_trans = [], [], [], []
    for ci, s in enumerate(segs):
        rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            z_all, _, _ = vq.encode(x)
            e_s = model.encode_conditions(x_curr=x[0:1], style_label=torch.tensor([s['label']], device=dev))[2]
            gt = code_idx(vq, z_all[1:args.frames + 1])
            gen = torch.Generator(device=dev).manual_seed(9000 + ci)
            for k in range(1, args.spread_frames + 1):
                samples = []
                for _ in range(N):
                    out = single_step(x[k - 1:k], z_all[k - 1:k],
                                      z_all[max(k - 2, 0):max(k - 2, 0) + 1], e_s, 4, 1.0, gen)
                    samples.append(code_idx(vq, out)[0])
                S = torch.stack(samples)                      # [N,32]
                pair_slot = (S[:, None, :] == S[None, :, :]).float().mean(0).mean(0)  # [32] pairwise agreement per slot
                any_slot = (S == gt[k - 1]).any(0).float()                            # [32]
                is_stable = (gt[k - 2] == gt[k - 1]) if k >= 2 else torch.ones(32, dtype=torch.bool, device=dev)
                pair_all.append(pair_slot.mean().item()); any_all.append(any_slot.mean().item())
                if bool(is_stable.any()):
                    pair_stable.append(pair_slot[is_stable].mean().item())
                    any_stable.append(any_slot[is_stable].mean().item())
                if bool((~is_stable).any()):
                    pair_trans.append(pair_slot[~is_stable].mean().item())
                    any_trans.append(any_slot[~is_stable].mean().item())
    res['spread'] = dict(
        samples_per_frame=N, frames=args.spread_frames * len(segs),
        pairwise_code_agreement=float(np.mean(pair_all)),
        any_sample_matches_gt=float(np.mean(any_all)),
        pairwise_agreement_stable=float(np.mean(pair_stable)) if pair_stable else None,
        pairwise_agreement_transition=float(np.mean(pair_trans)) if pair_trans else None,
        any_match_stable=float(np.mean(any_stable)) if any_stable else None,
        any_match_transition=float(np.mean(any_trans)) if any_trans else None,
    )
    print('[spread]', json.dumps(res['spread']))

    # ---------- D: GT frame-to-frame per-slot displacement ----------
    # NOTE: z_all is the QUANTIZED latent, so a slot whose code is unchanged has displacement
    # exactly 0. Fractions therefore move in steps of 1/32 (or larger at code ties), and the
    # median lands on the 36.9% mass of stable slots -> use quantiles on NONZERO displacement.
    disp, margin, switch_all = [], [], []
    for ci, s in enumerate(segs):
        rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            z_all, idx, _ = vq.encode(x)
            d = (z_all[1:] - z_all[:-1]).reshape(-1, 4).norm(dim=-1)
            dd = torch.cdist(z_all.reshape(-1, 4).float(), vq.quantizer.embedding_weight.detach().float())
            m = dd.topk(2, largest=False).values
            m = (m[:, 1] - m[:, 0]).reshape(-1)[vq.num_latents:]
            sw = (idx[1:] != idx[:-1]).float().reshape(-1)[:len(d)]
            disp.append(d); margin.append(m); switch_all.append(sw)
    disp = torch.cat(disp); margin = torch.cat(margin); switch_all = torch.cat(switch_all)
    nz = disp > 0
    q = lambda t: [float(v) for v in torch.quantile(t, torch.tensor([.1, .5, .9], device=t.device))]
    res['gt_motion'] = dict(
        switch_rate=float(switch_all.mean()),
        zero_displacement_frac=float((~nz).float().mean()),
        displacement_nonzero_p10_p50_p90=q(disp[nz]),
        displacement_all_p10_p50_p90=q(disp),
        margin_p10_p50_p90=q(margin),
    )
    print('[gt_motion]', json.dumps(res['gt_motion']))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print('[Done]', args.out)


if __name__ == '__main__':
    main()
