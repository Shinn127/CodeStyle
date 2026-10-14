"""Focused cause analysis for quantized-code divergence (码本跳变) of the label controller.

Three nested questions, measured separately:

1. geometry   -- codec-only: how fragile is the discrete code identity under its own
                 decode->re-encode loop? (nearest-code rank of the flipped destination,
                 code-to-code packing distance, flip rate vs displacement/margin ratio)
2. stability  -- decompose code match into GT-stable slots vs GT-transition slots
                 (roundtrip / tf / rollout), to separate "cannot hold a stable code"
                 from "cannot produce the next code".
3. refresh    -- closed-loop accumulation sweep: run the official rollout but re-anchor
                 the feedback state to the GT quantized latent every K frames
                 (K=1 => single-step manifold, K=len => full rollout). The match-vs-K
                 curve measures how many steps of self-feedback until the state leaves
                 the model's GT-anchored recovery basin.

Trajectory block is GT in every mode (same convention as code_divergence.py), so the
numbers exclude trajectory prediction error by construction.

Runs a single model per invocation; writes one JSON. No repository files are modified.
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


def load_assets(flow_ckpt, device):
    model = load_motion_model('flowmatch_target', Path(flow_ckpt), num_styles=90).to(device)
    model.eval().requires_grad_(False)
    steps = int(getattr(model, 'eval_solver_steps', 4))
    mean, std = load_norm_stats(NORM)
    return model, model.vqvae, mean.astype(np.float32), std.astype(np.float32), steps


def pick_clips(clips, frames):
    segs = json.loads((DB / 'segments.json').read_text())
    eligible = [s for s in segs if s.get('split') == 'test' and not s.get('mirror', False)
                and s['num_frames'] >= frames + 5]
    by_style = {}
    for s in eligible:
        by_style.setdefault(s['label'], s)
    styles = sorted(by_style)
    picks = np.linspace(0, len(styles) - 1, clips).round().astype(int)
    return [by_style[styles[i]] for i in sorted(set(picks))]


def code_indices(vq, z_raw):
    b = z_raw.shape[0]
    x = z_raw.reshape(b * vq.num_latents, vq.code_dim).float()
    embed = vq.quantizer.embedding_weight.detach().float()
    d = torch.cdist(x, embed)
    idx = d.argmin(-1)
    top2 = d.topk(2, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]).reshape(b, vq.num_latents)
    return idx.reshape(b, vq.num_latents), margin


def rank_of_code(vq, z_ref, idx_gen, chunk=8192):
    """Rank (0 = nearest) of the generated code in the code-distance ordering of z_ref."""
    embed = vq.quantizer.embedding_weight.detach().float()
    z = z_ref.reshape(-1, vq.num_latents, vq.code_dim)
    n = z.shape[0]
    out = torch.empty(n, vq.num_latents, dtype=torch.long, device=z.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(z[s:e].reshape(-1, vq.code_dim), embed)
        dg = d.gather(1, idx_gen[s:e].reshape(-1)[:, None])
        out[s:e] = (d < dg).sum(-1).reshape(e - s, vq.num_latents)
    return out


def code_pairwise_nn(vq):
    embed = vq.quantizer.embedding_weight.detach().float()
    d = torch.cdist(embed, embed)
    d.fill_diagonal_(float('inf'))
    nn = d.min(-1).values.cpu().numpy()
    q = np.quantile(nn, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(median=float(q[2]), p05=float(q[0]), p25=float(q[1]), p75=float(q[3]), p95=float(q[4]))


def ratio_curve(move, margin, flip, nbins=6):
    r = (move.reshape(-1) / (margin.reshape(-1) / 2 + 1e-9)).cpu().numpy()
    f = flip.reshape(-1).cpu().numpy()
    qs = np.quantile(r, np.linspace(0, 1, nbins + 1))
    out = []
    for i in range(nbins):
        sel = (r >= qs[i]) & (r <= qs[i + 1]) if i == nbins - 1 else (r >= qs[i]) & (r < qs[i + 1])
        if sel.sum() >= 8:
            out.append(dict(ratio_lo=float(qs[i]), ratio_hi=float(qs[i + 1]), n=int(sel.sum()),
                            flip_rate=float(f[sel].mean())))
    return out


def stability_split(match_ind, gen_idx, gt_idx):
    """match_ind: 1 = generated code equals GT code. Split by GT-stable vs GT-transition slots."""
    prev, cur, f, gen = gt_idx[:-1], gt_idx[1:], match_ind[1:], gen_idx[1:]
    stable = prev == cur
    trans = ~stable
    return dict(
        stable_frac=float(stable.float().mean()),
        match_stable=float(f[stable].mean()) if bool(stable.any()) else None,
        match_transition=float(f[trans].mean()) if bool(trans.any()) else None,
        gen_keeps_prev=float((gen[trans] == prev[trans]).float().mean()) if bool(trans.any()) else None,
        gen_new=float((gen[trans] == cur[trans]).float().mean()) if bool(trans.any()) else None,
    )


def run_roundtrip(vq, z_all, mean, std, rows, frames):
    """encode(GT) -> decode -> re-encode on the GT trajectory block (matches code_divergence.py)."""
    dev = z_all.device
    mean_t = torch.as_tensor(mean, device=dev)
    std_t = torch.as_tensor(std, device=dev)
    x_norm = (torch.as_tensor(rows, device=dev) - mean_t) / std_t
    with torch.no_grad():
        motion_re = vq.decode(z_all)                 # normalized-space motion (291)
        x_re = torch.cat((x_norm[:, :84], motion_re), -1)
        z2, _, _ = vq.encode(x_re)
    gt_idx, gt_margin = code_indices(vq, z_all)
    gen_idx, _ = code_indices(vq, z2)
    move = (z2 - z_all).reshape(len(z2), -1, vq.code_dim).norm(dim=-1)
    match_ind = (gen_idx == gt_idx).float()
    rank = rank_of_code(vq, z_all, gen_idx)
    s = slice(1, frames + 1)
    motion_re_cm = motion_re * std_t[84:] + mean_t[84:]
    gt_motion = torch.as_tensor(rows[:, 84:], device=dev)
    gen_motion_cm = vq.decode(z2) * std_t[84:] + mean_t[84:]
    return dict(
        match=float(match_ind[s].mean()),
        move_mean=float(move[s].mean()), margin_mean=float(gt_margin[s].mean()),
        flip_rate=float((1 - match_ind)[s].mean()),
        rank_hist={str(r): float((rank[s] == r).float().mean()) for r in (0, 1, 2, 3)},
        rank_ge3=float((rank[s] >= 3).float().mean()),
        ratio_curve=ratio_curve(move[s], gt_margin[s], 1 - match_ind[s]),
        decode_roundtrip_cm=float((motion_re_cm[s] - gt_motion[s]).abs().mean() * 100),
        code_impact_cm=float((gen_motion_cm[s] - motion_re_cm[s]).abs().mean() * 100),
        per_slot_flip=(1 - match_ind)[s].mean(0).cpu().tolist(),
        per_slot_move=move[s].mean(0).cpu().tolist(),
        per_slot_margin=gt_margin[s].mean(0).cpu().tolist(),
        frac=match_ind[s], gen_idx=gen_idx[s], gt_idx=gt_idx[s],
    )


def rollout_once(model, vq, mean_t, std_t, rows, label, frames, mode, noise_std, steps, gen, refresh=0):
    dev = next(model.parameters()).device
    x_norm = torch.as_tensor((rows - mean_t.cpu().numpy()) / std_t.cpu().numpy(), device=dev)
    with torch.no_grad():
        z_all, _, _ = vq.encode(x_norm)
        e_s = model.encode_conditions(x_curr=x_norm[0:1], style_label=torch.tensor([label], device=dev))[2]

    def noise_z0():
        eps = torch.randn((1, vq.latent_dim), device=dev, generator=gen)
        return model.latent_offset + model.latent_scale * model.latent_transport_std * noise_std * eps

    def step(x_curr, prev_z_raw, hist_z):
        z_cont = model.sample_latent(x_curr=x_curr, style_label=None, style_feature=e_s,
                                     prev_latent=prev_z_raw, history_latent=hist_z,
                                     z0=noise_z0(), num_steps=steps, solver='euler')
        z_clip = model.clip_latent_to_data_bounds(z_cont)
        z_q = model.requantize_latent(z_clip)
        return z_cont, z_q

    gen_cont, gen_q = [], []
    prev_q = z_all[0:1]
    hist = z_all[0:1]
    motion_prev = vq.decode(prev_q)
    for k in range(1, frames + 1):
        if mode == 'tf':
            zc, zq = step(x_norm[k - 1:k], z_all[k - 1:k], z_all[max(k - 2, 0):max(k - 2, 0) + 1])
        elif mode == 'reconstruct':
            x_curr = torch.cat((x_norm[k - 1:k, :84], vq.decode(z_all[k - 1:k])), -1)
            zc, zq = step(x_curr, z_all[k - 1:k], z_all[max(k - 2, 0):max(k - 2, 0) + 1])
        else:  # rollout, optionally re-anchored to the GT quantized state every `refresh` frames
            x_curr = torch.cat((x_norm[k - 1:k, :84], motion_prev), -1)
            zc, zq = step(x_curr, prev_q, hist)
            if refresh and k % refresh == 0 and k < frames:
                prev_q = z_all[k:k + 1]
                hist = z_all[max(k - 1, 0):max(k - 1, 0) + 1]
                motion_prev = vq.decode(prev_q)
            else:
                motion_prev = vq.decode(zq)
                hist = prev_q
                prev_q = zq
        gen_cont.append(zc[0]); gen_q.append(zq[0])
    gen_cont = torch.stack(gen_cont)
    gen_q = torch.stack(gen_q)
    z_gt = z_all[1:frames + 1]
    gt_idx, gt_margin = code_indices(vq, z_gt)
    gen_idx, _ = code_indices(vq, gen_q)
    frac = (gen_idx == gt_idx).float()
    gen_raw = vq.decode(gen_q) * std_t[84:] + mean_t[84:]
    gt_raw = torch.as_tensor(rows[1:frames + 1, 84:], device=dev)
    gt_codes_motion = vq.decode(z_gt) * std_t[84:] + mean_t[84:]
    return dict(
        match=float(frac.mean()), hamming=float((1 - frac).sum(-1).mean()),
        l2_sig=float((gen_cont - z_gt).norm(dim=-1).mean()),
        motion_abs_cm=float((gen_raw - gt_raw).abs().mean() * 100),
        code_impact_cm=float((gen_raw - gt_codes_motion).abs().mean() * 100),
        step_l2=(gen_cont - z_gt).norm(dim=-1).cpu().numpy().tolist(),
        frac=frac, gen_idx=gen_idx, gt_idx=gt_idx, gt_margin=gt_margin,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow-ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--refresh-ks', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64, 240])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    model, vq, mean, std, steps = load_assets(args.flow_ckpt, device)
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)
    segs = pick_clips(args.clips, args.frames)
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    print(f'[Info] tag={args.tag} ckpt={args.flow_ckpt} steps={steps} clips={len(segs)} frames={args.frames}')

    results = dict(tag=args.tag, checkpoint=args.flow_ckpt, solver_steps=steps, frames=args.frames,
                   seed=args.seed,
                   clips=[dict(style=s['style_name'], source_entry_id=s['source_entry_id']) for s in segs])

    def read_seg(s):
        return np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)

    # ---- section 1: geometry (codec-only roundtrip)
    t0 = time.perf_counter()
    geo, rt_cat = [], None
    for s in segs:
        rows = read_seg(s)
        x_norm = torch.as_tensor((rows - mean) / std, device=device)
        with torch.no_grad():
            z_all, _, _ = vq.encode(x_norm)
        g = run_roundtrip(vq, z_all, mean, std, rows, args.frames)
        geo.append(g)
        rt_cat = (g['frac'], g['gen_idx'], g['gt_idx']) if rt_cat is None else (
            torch.cat([rt_cat[0], g['frac']]), torch.cat([rt_cat[1], g['gen_idx']]), torch.cat([rt_cat[2], g['gt_idx']]))
    results['c2c_nn'] = code_pairwise_nn(vq)
    agg_keys = ('match', 'move_mean', 'margin_mean', 'flip_rate', 'rank_ge3',
                'decode_roundtrip_cm', 'code_impact_cm')
    results['geometry'] = {k: [float(np.mean([g[k] for g in geo])), float(np.std([g[k] for g in geo]))] for k in agg_keys}
    results['geometry']['rank_hist'] = {r: float(np.mean([g['rank_hist'][r] for g in geo])) for r in ('0', '1', '2', '3')}
    results['geometry']['ratio_curve'] = geo[0]['ratio_curve']
    results['geometry']['per_slot_flip_mean'] = np.mean([g['per_slot_flip'] for g in geo], 0).tolist()
    results['geometry']['per_slot_move_mean'] = np.mean([g['per_slot_move'] for g in geo], 0).tolist()
    results['geometry']['per_slot_margin_mean'] = np.mean([g['per_slot_margin'] for g in geo], 0).tolist()
    results['roundtrip_stability'] = stability_split(*rt_cat)
    print(f"[geometry] match={results['geometry']['match'][0]:.3f} flip={results['geometry']['flip_rate'][0]:.3f} "
          f"rank_ge3={results['geometry']['rank_ge3'][0]:.4f} c2c_nn_med={results['c2c_nn']['median']:.3f} "
          f"wall={time.perf_counter()-t0:.0f}s")
    print(f"[roundtrip] stable match={results['roundtrip_stability']['match_stable']:.3f} "
          f"transition match={results['roundtrip_stability']['match_transition']:.3f}")

    # ---- section 2: tf baseline with rank + stability
    t0 = time.perf_counter()
    ms = []
    for ci, s in enumerate(segs):
        rows = read_seg(s)
        x_norm = torch.as_tensor((rows - mean) / std, device=device)
        with torch.no_grad():
            z_all, _, _ = vq.encode(x_norm)
        gen = torch.Generator(device=device).manual_seed(args.seed + ci)
        m = rollout_once(model, vq, mean_t, std_t, rows, s['label'], args.frames, 'tf', 0.0, steps, gen)
        r = rank_of_code(vq, z_all[1:args.frames + 1], m['gen_idx'])
        m['rank_hist'] = {str(x): float((r == x).float().mean()) for x in (0, 1, 2, 3)}
        m['rank_ge3'] = float((r >= 3).float().mean())
        m['stability'] = stability_split(m['frac'], m['gen_idx'], m['gt_idx'])
        ms.append(m)
    results['tf_std0'] = dict(
        match=float(np.mean([m['match'] for m in ms])),
        motion_abs_cm=float(np.mean([m['motion_abs_cm'] for m in ms])),
        code_impact_cm=float(np.mean([m['code_impact_cm'] for m in ms])),
        l2_sig=float(np.mean([m['l2_sig'] for m in ms])),
        rank_hist={r: float(np.mean([m['rank_hist'][r] for m in ms])) for r in ('0', '1', '2', '3')},
        rank_ge3=float(np.mean([m['rank_ge3'] for m in ms])),
        stability={k: float(np.mean([m['stability'][k] for m in ms if m['stability'][k] is not None]))
                   for k in ('stable_frac', 'match_stable', 'match_transition', 'gen_keeps_prev', 'gen_new')},
    )
    t = results['tf_std0']
    print(f"[tf_std0] match={t['match']:.3f} rank_ge3={t['rank_ge3']:.4f} "
          f"stable={t['stability']['match_stable']:.3f} transition={t['stability']['match_transition']:.3f} "
          f"wall={time.perf_counter()-t0:.0f}s")

    # ---- section 3: rollout stability split
    t0 = time.perf_counter()
    ms = []
    for ci, s in enumerate(segs):
        rows = read_seg(s)
        gen = torch.Generator(device=device).manual_seed(args.seed + ci)
        m = rollout_once(model, vq, mean_t, std_t, rows, s['label'], args.frames, 'rollout', 0.0, steps, gen)
        m['stability'] = stability_split(m['frac'], m['gen_idx'], m['gt_idx'])
        ms.append(m)
    results['rollout_stability'] = {k: float(np.mean([m['stability'][k] for m in ms]))
                                    for k in ('stable_frac', 'match_stable', 'match_transition', 'gen_keeps_prev', 'gen_new')}
    results['rollout_std0'] = dict(
        match=float(np.mean([m['match'] for m in ms])),
        l2_sig=float(np.mean([m['l2_sig'] for m in ms])),
        motion_abs_cm=float(np.mean([m['motion_abs_cm'] for m in ms])),
        code_impact_cm=float(np.mean([m['code_impact_cm'] for m in ms])),
    )
    onset = np.mean([np.asarray(m['step_l2']) for m in ms], 0)
    results['onset'] = dict(
        l2_sig_at={str(k): float(onset[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64, 128, 240) if k <= args.frames})
    r = results['rollout_stability']
    print(f"[rollout] match={results['rollout_std0']['match']:.3f} stable={r['match_stable']:.3f} "
          f"transition={r['match_transition']:.3f} L2sig@1/8/32/240="
          f"{onset[0]:.1f}/{onset[7]:.1f}/{onset[31]:.1f}/{onset[-1]:.1f} wall={time.perf_counter()-t0:.0f}s")

    # ---- section 4: refresh sweep
    results['refresh'] = {}
    for K in args.refresh_ks:
        t0 = time.perf_counter()
        ms = []
        for ci, s in enumerate(segs):
            rows = read_seg(s)
            gen = torch.Generator(device=device).manual_seed(args.seed + ci)
            ms.append(rollout_once(model, vq, mean_t, std_t, rows, s['label'], args.frames,
                                   'rollout', 0.0, steps, gen, refresh=K))
        results['refresh'][str(K)] = dict(
            match=float(np.mean([m['match'] for m in ms])),
            l2_sig=float(np.mean([m['l2_sig'] for m in ms])),
            motion_abs_cm=float(np.mean([m['motion_abs_cm'] for m in ms])),
            code_impact_cm=float(np.mean([m['code_impact_cm'] for m in ms])),
            wall_sec=time.perf_counter() - t0,
        )
        r = results['refresh'][str(K)]
        print(f"[refresh K={K:3d}] match={r['match']:.3f} L2sig={r['l2_sig']:.2f} "
              f"motion|d|={r['motion_abs_cm']:.2f}cm code|d|={r['code_impact_cm']:.2f}cm")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print('[Done]', args.out)


if __name__ == '__main__':
    main()
