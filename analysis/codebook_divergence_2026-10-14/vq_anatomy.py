"""VQ roundtrip instability anatomy: where does the code flip actually come from?

Decomposes the encode->decode->re-encode code flip (measured 62% of slots) into:
  r1      : encoder-output distance to its assigned code (off-centering)
  bdist   : exact distance from the encoder output to its Voronoi boundary
  delta   : re-encode drift ||z_e2 - z_e|| (and its control: same-norm input noise)
  flip    : argmin code after re-encode differs
  decode  : motion-space reconstruction error of the decoded path

Control experiment: perturb the input by Gaussian noise scaled to the SAME per-frame
norm as the decoder artifact, and compare latent drift. If the artifact drift is much
larger, the encoder is specifically sensitive to decoder outputs (exposure bias);
if comparable, the instability is ordinary sensitivity + tight cells (geometry).

Also measures the continuous-decode roundtrip (no quantization on the decode side) to
separate "decoder artifact" from "quantization noise" contributions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/shinn/桌面/multimedia/revision/NewCode')
sys.path.insert(0, str(ROOT))
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import load_norm_stats

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'
VQ_CKPT = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=str(Path(__file__).with_name('vq_anatomy.json')))
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    vq = build_vqvae_from_checkpoint(VQ_CKPT).to(dev).eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    mean, std = load_norm_stats(NORM)
    segs = json.loads((DB / 'segments.json').read_text())
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    pool = [s for s in segs if s.get('split') == args.split and not s.get('mirror', False)
            and s['num_frames'] >= args.frames + 1]
    rng = np.random.RandomState(0)
    picks = [pool[i] for i in rng.choice(len(pool), min(args.clips, len(pool)), replace=False)]
    embed = vq.quantizer.embedding_weight.detach().float()
    if embed.dim() == 3:
        embed = embed[0]
    K, D = embed.shape
    print(f'[Info] split={args.split} clips={len(picks)} frames={args.frames} K={K} D={D}')

    r1s, bdists, dmags, dctrl, dcont, recon_cms = [], [], [], [], [], []
    flips, flips_ctrl_list, ratios, per_head_flip, per_head_n = [], [], [], np.zeros(32), np.zeros(32)
    switch_gts = []
    for s in picks:
        rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            z_e = vq.encode_pre_vq(x)                    # [T,128] continuous
            z_q, idx, _ = vq.encode(x)                   # quantized path
            idx = idx.reshape(len(x), 32)
            motion_hat = vq.decode(z_q)
            x2 = torch.cat((x[:, :84], motion_hat), -1)
            z_e2 = vq.encode_pre_vq(x2)
            # continuous decode roundtrip (no quantization)
            motion_c = vq.decode(z_e)
            x3 = torch.cat((x[:, :84], motion_c), -1)
            z_e3 = vq.encode_pre_vq(x3)
            # same-norm noise control on the real input
            diff = (x2 - x)[:, 84:]
            norm_per_frame = diff.norm(dim=-1, keepdim=True)              # [T,1]
            eps = torch.randn_like(diff)
            eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-9) * norm_per_frame
            x_ctrl = torch.cat((x[:, :84], x[:, 84:] + eps), -1)
            z_ectrl = vq.encode_pre_vq(x_ctrl)
        # per-slot geometry of the ORIGINAL encode
        ze = z_e.reshape(-1, 32, D); c1 = z_q.reshape(-1, 32, D)
        r1 = (ze - c1).norm(dim=-1)                                       # [T,32]
        d2_all = torch.cdist(ze.reshape(-1, D).float(), embed)            # [T*32,K]
        own = idx.reshape(-1)
        d1_all = d2_all.gather(1, own[:, None])
        d2_all.scatter_(1, own[:, None], float('inf'))
        c2v = d2_all.argmin(-1)
        d2v = d2_all.gather(1, c2v[:, None])
        c2 = embed[c2v]
        cc = (c1.reshape(-1, D) - c2).norm(dim=-1)
        bdist_exact = ((d2v.squeeze(1) ** 2 - d1_all.squeeze(1) ** 2) / (2 * cc.clamp_min(1e-9)))
        r1 = r1.reshape(-1); bdist_exact = bdist_exact.reshape(-1)
        # roundtrip drift and flips
        dq = (z_e2 - z_e).reshape(-1, 32, D).norm(dim=-1).reshape(-1)
        dc = (z_e3 - z_e).reshape(-1, 32, D).norm(dim=-1).reshape(-1)
        dk = (z_ectrl - z_e).reshape(-1, 32, D).norm(dim=-1).reshape(-1)
        idx2 = (torch.cdist(z_e2.reshape(-1, D).float(), embed).argmin(-1))
        flip = (idx2 != own).float()
        d2_all = torch.cdist(z_ectrl.reshape(-1, D).float(), embed)
        idxc = d2_all.argmin(-1)
        flip_ctrl = (idxc != own).float()
        rec_cm = ((motion_hat - x[:, 84:]) * torch.as_tensor(std, device=dev)[84:]).abs().mean() * 100
        r1s.append(r1.cpu()); bdists.append(bdist_exact.cpu()); dmags.append(dq.cpu())
        dctrl.append(dk.cpu()); dcont.append(dc.cpu()); flips.append(flip.cpu())
        flips_ctrl_list.append(flip_ctrl.cpu()); recon_cms.append(float(rec_cm))
        ratios.append((dq / bdist_exact.clamp_min(1e-9)).cpu())
        ph = flip.reshape(-1, 32).mean(0).cpu().numpy()
        per_head_flip += ph; per_head_n += 1
        sw = (idx[1:] != idx[:-1]).float().sum(-1).mean().item()
        switch_gts.append(sw)

    cat = lambda L: torch.cat(L)
    q = lambda t, p: [round(v, 4) for v in torch.quantile(t, torch.tensor(p)).tolist()]
    out = {
        'split': args.split, 'clips': len(picks), 'frames_total': args.frames * len(picks),
        'r1': {'mean': float(cat(r1s).mean()), 'p50': q(cat(r1s), [.5])[0], 'p90': q(cat(r1s), [.9])[0]},
        'bdist': {'mean': float(cat(bdists).mean()), 'p10': q(cat(bdists), [.1])[0],
                  'p50': q(cat(bdists), [.5])[0], 'p90': q(cat(bdists), [.9])[0]},
        'drift_quant': {'mean': float(cat(dmags).mean()), 'p50': q(cat(dmags), [.5])[0], 'p90': q(cat(dmags), [.9])[0]},
        'drift_cont': {'mean': float(cat(dcont).mean()), 'p50': q(cat(dcont), [.5])[0]},
        'drift_ctrl_noise': {'mean': float(cat(dctrl).mean()), 'p50': q(cat(dctrl), [.5])[0]},
        'flip_rate': float(cat(flips).mean()),
        'flip_rate_noise_control': float(cat(flips_ctrl_list).mean()),
        'flip_given_drift_gt_bdist': float(cat(flips)[cat(dmags) > cat(bdists)].mean()),
        'flip_given_drift_le_bdist': float(cat(flips)[cat(dmags) <= cat(bdists)].mean()),
        'r1_over_bdist_mean': float(cat(ratios).clamp_max(100).mean()),
        'recon_cm_mean': float(np.mean(recon_cms)),
        'gt_switch_per_frame': float(np.mean(switch_gts)),
        'per_head_flip_rate': (per_head_flip / per_head_n).tolist(),
    }
    print(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
