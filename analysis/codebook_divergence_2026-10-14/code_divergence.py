"""GT-clip quantized-code divergence probe for the label flow controller.

For each test clip, use the GT first frame as the seed and the clip's GT trajectory block
as the per-step trajectory condition, then compare 32-slot quantized code indices against
the GT codes under four regimes:

  roundtrip  : encode(GT) -> decode -> re-encode; no controller (codec stability reference)
  tf         : teacher forcing with the GT motion row as the observation (training manifold)
  reconstruct: teacher forcing with decode(z_gt) as the observation (official single-step
               inference manifold; prev_latent = z_gt, history = z_gt sequence)
  rollout    : the model's own requantized feedback (official rollout semantics)

Also reports the GT code-boundary margin (d2-d1 in raw latent space) and the per-slot
distance the model actually moves the latent, plus a flip-rate-vs-margin curve, so code
mismatch can be separated into "codec boundary fragility" vs "model error".

Runs from /tmp; touches no repository files.
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
    embed = vq.quantizer.embedding_weight.float()
    d = torch.cdist(x, embed)
    idx = d.argmin(-1)
    top2 = d.topk(2, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]).reshape(b, vq.num_latents)
    return idx.reshape(b, vq.num_latents), margin


def idx_switch(seq):
    return (seq[1:] != seq[:-1]).float().sum(-1).mean().item() if len(seq) > 1 else 0.


def bin_curve(gen_idx, gt_idx, nbins=8):
    F = gen_idx.shape[0]
    edges = np.linspace(0, F, nbins + 1).astype(int)
    return [dict(frames=[int(edges[i]), int(edges[i + 1])],
                 match=float((gen_idx[edges[i]:edges[i + 1]] == gt_idx[edges[i]:edges[i + 1]]).float().mean()),
                 hamming=float((gen_idx[edges[i]:edges[i + 1]] != gt_idx[edges[i]:edges[i + 1]]).float().sum(-1).mean()))
            for i in range(nbins)]


def lag_match(gen_idx, gt_idx, max_off=6):
    """Per-slot code match of gen[t] against gt[t+off]; peaks off 0 if time-aligned."""
    F = gen_idx.shape[0]
    out = {}
    for off in range(-max_off, max_off + 1):
        a0, b0 = max(0, -off), max(0, off)
        a1, b1 = F - max(0, off), F - max(0, -off)
        out[off] = float((gen_idx[a0:a1] == gt_idx[b0:b1]).float().mean())
    return out


def verdicts(gen_idx, gt_idx, gen_margin, gen_move, gen_idx_full=None):
    n_slots = gen_idx.shape[-1]
    # flip-rate vs GT margin buckets (pooled across frames/slots)
    m = gen_margin.reshape(-1).cpu().numpy()
    f = (gen_idx != gt_idx).reshape(-1).cpu().numpy()
    mv = gen_move.reshape(-1).cpu().numpy()
    qs = np.quantile(m, [0, .2, .4, .6, .8, 1.0])
    buckets = []
    for i in range(5):
        sel = (m >= qs[i]) & (m <= qs[i + 1]) if i == 4 else (m >= qs[i]) & (m < qs[i + 1])
        if sel.sum():
            buckets.append(dict(margin_lo=float(qs[i]), margin_hi=float(qs[i + 1]),
                                n=int(sel.sum()), flip_rate=float(f[sel].mean()), mean_move=float(mv[sel].mean())))
    return dict(
        match=float((gen_idx == gt_idx).float().mean()),
        hamming=float((gen_idx != gt_idx).float().sum(-1).mean()),
        exact_all=float((gen_idx == gt_idx).all(-1).float().mean()),
        near2=float(((gen_idx != gt_idx).float().sum(-1) <= 2).float().mean()),
        slot_match=(gen_idx == gt_idx).float().mean(0).cpu().tolist(),
        margin_mean=float(m.mean()), move_mean=float(mv.mean()),
        flip_by_margin=buckets,
        switch_gen=idx_switch(gen_idx), switch_gt=idx_switch(gt_idx),
        bins=bin_curve(gen_idx, gt_idx),
        lag=lag_match(gen_idx, gt_idx),
    )


def run_clip(model, vq, mean_t, std_t, rows, label, frames, mode, noise_std, steps, gen):
    dev = next(model.parameters()).device
    x_norm = torch.as_tensor((rows - mean_t.cpu().numpy()) / std_t.cpu().numpy(), device=dev)
    with torch.no_grad():
        z_all, idx_all, _ = vq.encode(x_norm)
        gt_idx, gt_margin = code_indices(vq, z_all)

    gt_slice = slice(1, frames + 1)

    if mode == 'roundtrip':
        with torch.no_grad():
            motion_re = vq.decode(z_all)
            x_re = torch.cat((x_norm[:, :84], motion_re), -1)
            z2, idx2, _ = vq.encode(x_re)
            idx2_full, _ = code_indices(vq, z2)
            move = (z2.reshape(len(z2), vq.num_latents, vq.code_dim)
                    - z_all.reshape(len(z_all), vq.num_latents, vq.code_dim)).norm(dim=-1)
        out = verdicts(idx2_full[gt_slice], gt_idx[gt_slice], gt_margin[gt_slice], move[gt_slice])
        out['l2_sig'] = float((model.normalize_latent(z2[gt_slice]) - model.normalize_latent(z_all[gt_slice])).norm(dim=-1).mean())
        # Codec's own reconstruction error: decode(GT codes) vs the GT motion block (cm).
        re_motion = motion_re[gt_slice] * std_t[84:] + mean_t[84:]
        gt_motion = torch.as_tensor(rows[1:frames + 1, 84:], device=dev)
        out['motion_abs_cm'] = float((re_motion - gt_motion).abs().mean() * 100)
        # Impact of the flipped codes themselves: decode(z2) vs decode(GT codes), in cm.
        gen_motion = vq.decode(z2[gt_slice]) * std_t[84:] + mean_t[84:]
        out['code_impact_cm'] = float((gen_motion - re_motion).abs().mean() * 100)
        return out

    with torch.no_grad():
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
    if mode == 'tf':
        for i in range(1, frames + 1):
            hist = z_all[max(i - 2, 0):max(i - 2, 0) + 1]
            zc, zq = step(x_norm[i - 1:i], z_all[i - 1:i], hist)
            gen_cont.append(zc[0]); gen_q.append(zq[0])
    elif mode == 'reconstruct':
        with torch.no_grad():
            decoded_motion = vq.decode(z_all.reshape(-1, vq.latent_dim)).reshape(-1, vq.motion_dim)
        for i in range(1, frames + 1):
            x_curr = torch.cat((x_norm[i - 1:i, :84], decoded_motion[i - 1:i]), -1)
            hist = z_all[max(i - 2, 0):max(i - 2, 0) + 1]
            zc, zq = step(x_curr, z_all[i - 1:i], hist)
            gen_cont.append(zc[0]); gen_q.append(zq[0])
    else:  # rollout
        prev_q = z_all[0:1]
        hist = z_all[0:1]
        motion_prev = vq.decode(prev_q)
        for k in range(1, frames + 1):
            x_curr = torch.cat((x_norm[k - 1:k, :84], motion_prev), -1)
            zc, zq = step(x_curr, prev_q, hist)
            motion_prev = vq.decode(zq)
            hist = prev_q
            prev_q = zq
            gen_cont.append(zc[0]); gen_q.append(zq[0])

    gen_cont = torch.stack(gen_cont)   # [F,128] raw continuous
    gen_q = torch.stack(gen_q)         # [F,128] raw quantized
    gen_idx, _ = code_indices(vq, gen_q)
    move = (gen_cont.reshape(len(gen_cont), vq.num_latents, vq.code_dim)
            - z_all[gt_slice].reshape(frames, vq.num_latents, vq.code_dim)).norm(dim=-1)
    out = verdicts(gen_idx, gt_idx[gt_slice], gt_margin[gt_slice], move)
    out['l2_sig'] = float((model.normalize_latent(gen_cont) - model.normalize_latent(z_all[gt_slice])).norm(dim=-1).mean())
    gen_raw = vq.decode(gen_q) * std_t[84:] + mean_t[84:]
    gt_raw = torch.as_tensor(rows[1:frames + 1, 84:], device=dev)
    out['motion_abs_cm'] = float((gen_raw - gt_raw).abs().mean() * 100)
    # Impact of the code differences alone: decode(gen codes) vs decode(GT codes), in cm.
    gt_codes_motion = vq.decode(z_all[gt_slice]) * std_t[84:] + mean_t[84:]
    out['code_impact_cm'] = float((gen_raw - gt_codes_motion).abs().mean() * 100)
    return out


def aggregate(per_clip, key):
    vals = [c[key] for c in per_clip]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow-ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--modes', nargs='+', default=['roundtrip', 'reconstruct', 'tf', 'rollout'])
    ap.add_argument('--noise-stds', nargs='+', type=float, default=[0.0, 1.0])
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    model, vq, mean, std, steps = load_assets(args.flow_ckpt, device)
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)
    segs = pick_clips(args.clips, args.frames)
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(3534258, 375))
    print(f'[Info] tag={args.tag} ckpt={args.flow_ckpt} steps={steps} clips={len(segs)} frames={args.frames}')

    results = {'tag': args.tag, 'checkpoint': args.flow_ckpt, 'solver_steps': steps,
               'frames': args.frames, 'seed': args.seed,
               'clips': [dict(style=s['style_name'], source_entry_id=s['source_entry_id'],
                              frame_start=int(s['frame_start'])) for s in segs],
               'runs': {}}
    for mode in args.modes:
        stds = [0.0] if mode == 'roundtrip' else args.noise_stds
        for noise in stds:
            t0 = time.perf_counter()
            per_clip = []
            for ci, s in enumerate(segs):
                rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
                gen = torch.Generator(device=device).manual_seed(args.seed + ci)
                per_clip.append(run_clip(model, vq, mean_t, std_t, rows, s['label'],
                                         args.frames, mode, noise, steps, gen))
            wall = time.perf_counter() - t0
            agg = {k: aggregate(per_clip, k) for k in
                   ('match', 'hamming', 'exact_all', 'near2', 'margin_mean', 'move_mean',
                    'switch_gen', 'switch_gt', 'l2_sig', 'motion_abs_cm', 'code_impact_cm')}
            agg['slot_match_mean'] = np.mean([c['slot_match'] for c in per_clip], 0).tolist()
            agg['flip_by_margin'] = [
                dict(margin_lo=np.mean([c['flip_by_margin'][i]['margin_lo'] for c in per_clip]),
                     margin_hi=np.mean([c['flip_by_margin'][i]['margin_hi'] for c in per_clip]),
                     flip_rate=np.mean([c['flip_by_margin'][i]['flip_rate'] for c in per_clip]),
                     mean_move=np.mean([c['flip_by_margin'][i]['mean_move'] for c in per_clip]))
                for i in range(len(per_clip[0]['flip_by_margin']))]
            agg['bins'] = [dict(frames=per_clip[0]['bins'][i]['frames'],
                                match=float(np.mean([p['bins'][i]['match'] for p in per_clip])),
                                hamming=float(np.mean([p['bins'][i]['hamming'] for p in per_clip])))
                           for i in range(len(per_clip[0]['bins']))]
            agg['lag'] = {str(off): float(np.mean([c['lag'][off] for c in per_clip])) for off in per_clip[0]['lag']}
            agg['wall_sec'] = wall
            key = f'{mode}_std{noise:g}'
            results['runs'][key] = {'aggregate': agg, 'per_clip': per_clip}
            a = agg
            print(f'[{key}] match={a["match"][0]:.3f}±{a["match"][1]:.3f} '
                  f'hamming={a["hamming"][0]:.2f}/32 exact={a["exact_all"][0]:.4f} '
                  f'margin={a["margin_mean"][0]:.3f} move={a["move_mean"][0]:.3f} '
                  f'switch g/gt={a["switch_gen"][0]:.2f}/{a["switch_gt"][0]:.2f} '
                  f'L2sig={a["l2_sig"][0]:.2f} motion|d|={a["motion_abs_cm"][0]:.2f}cm '
                  f'code|d|={a["code_impact_cm"][0]:.2f}cm wall={wall:.0f}s')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print('[Done]', args.out)


if __name__ == '__main__':
    main()
