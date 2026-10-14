"""VQ-level levers for code fidelity: error->match curve and codebook coarsening trade-off.

Answers: if the controller prediction carries error eps, what code match does the CURRENT
codebook support, and how would a coarser/better-separated codebook change it?

* error curve: GT quantized latents + isotropic noise at the measured model-error scale,
  requantized against the frozen codebook -> match vs sigma.
* coarsening: iteratively merge nearest code pairs (weighted by EMA cluster usage) to
  K' in {1024, 512, 256}; for each K' report the introduced code displacement (recon proxy,
  in raw latent units) and the match under the same error sigmas.
* margin structure: NN-spacing distribution of the current book (why the p05 tail matters).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/shinn/桌面/multimedia/revision/NewCode')
sys.path.insert(0, str(ROOT))
from Utils.motion_eval import load_motion_model
from Utils.runtime import load_norm_stats

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'


def main():
    dev = torch.device('cuda')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_motion_model('flowmatch_target', ROOT / 'checkpoints/flowmatch_label_history_obsnoise_tf10_l6_film/best.pt', num_styles=90).to(dev)
    model.eval().requires_grad_(False)
    vq = model.vqvae
    q = vq.quantizer
    embed = q.embedding_weight.detach().float()
    if embed.dim() == 3:
        embed = embed[0]
    K, D = embed.shape
    cluster = None
    try:
        cluster = q.vq._codebook.cluster_size.detach().float().reshape(-1)
    except Exception:
        pass
    mean, std = load_norm_stats(NORM)
    segs = json.loads((DB / 'segments.json').read_text())
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    test = [s for s in segs if s.get('split') == 'test' and not s.get('mirror', False) and s['num_frames'] >= 121]
    rng = np.random.RandomState(0)
    picks = [test[i] for i in rng.choice(len(test), 12, replace=False)]
    zs = []
    for s in picks:
        rows = np.array(data[s['frame_start']:s['frame_start'] + 121], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            _, idx, _ = vq.encode(x)
        idx = idx.reshape(-1, 32)
        zs.append(embed[idx.reshape(-1)].reshape(-1, 32 * D))
    Z = torch.cat(zs)                       # [N, 128] raw quantized latents (code vectors)
    print(f'[Info] {Z.shape[0]} slot-vectors, K={K}, D={D}')

    def requant(book, z):
        a = z.reshape(-1, D)
        d = torch.cdist(a.float(), book.float())
        i = d.argmin(-1)
        return i, book[i]

    def match_at(sigma, book, base_idx):
        torch.manual_seed(0)
        noise = torch.randn_like(Z) * float(sigma)
        pred = Z + noise
        idx2, z2 = requant(book, pred)
        # per-slot code match: need per-slot index for both; recompute base with same book
        return idx2

    # per-slot codes under the ORIGINAL book for GT
    gt_slots, _ = requant(embed, Z)

    print('\n--- error->match curve on the frozen codebook ---')
    curve = {}
    for sigma in [0.0, 0.03, 0.05, 0.077, 0.10, 0.141, 0.20, 0.30]:
        torch.manual_seed(0)
        pred = Z + torch.randn_like(Z) * sigma
        idx2, _ = requant(embed, pred)
        m = float((idx2 == gt_slots).float().mean())
        curve[sigma] = m
        print(f'sigma={sigma:.3f}: match={m:.3f}')

    print('\n--- codebook coarsening: merge nearest pairs ---')
    results = {'K': K, 'curve': {str(k): v for k, v in curve.items()}, 'coarsen': {}}
    book = embed.clone()
    usage = cluster.clone() if cluster is not None else torch.ones(K, device=dev)
    # map from current row -> original index for displacement tracking
    origin = torch.arange(K, device=dev)
    for target in [1024, 512, 256]:
        with torch.no_grad():
            while book.shape[0] > target:
                d = torch.cdist(book.float(), book.float())
                d.fill_diagonal_(float('inf'))
                # merge the globally closest pair, weighted by combined usage
                pair_cost = d
                flat = pair_cost.argmin()
                i, j = int(flat // book.shape[0]), int(flat % book.shape[0])
                w_i, w_j = usage[i], usage[j]
                new = (book[i] * w_i + book[j] * w_j) / (w_i + w_j).clamp_min(1e-9)
                keep = torch.ones(book.shape[0], dtype=torch.bool, device=dev)
                keep[j] = False
                book = book[keep].clone()
                usage = usage[keep].clone()
                book[i] = new
                usage[i] = w_i + w_j
                origin = origin[keep].clone()
                origin[i] = origin[i]  # merged row keeps i's origin (not exact; displacement uses book rows only)
        # introduced displacement: each original code vs its nearest merged centroid, weighted
        # by the original per-code usage (usage tracks merged rows, so weight from cluster_size).
        usage0 = cluster.clone() if cluster is not None else torch.ones(K, device=dev)
        d0 = torch.cdist(embed.float(), book.float()).min(-1).values
        disp = float((d0 * usage0).sum() / usage0.sum())
        # match under the SAME GT code identity mapped to merged book
        gt_merged, _ = requant(book, Z)
        rows = {}
        for sigma in [0.05, 0.077, 0.10, 0.141]:
            torch.manual_seed(0)
            pred = Z + torch.randn_like(Z) * sigma
            idx2, _ = requant(book, pred)
            m = float((idx2 == gt_merged).float().mean())
            rows[sigma] = m
        results['coarsen'][str(target)] = {'weighted_disp': disp, 'match': rows}
        print(f"K'={target}: weighted_disp={disp:.4f} | " +
              ' '.join(f'sig{s}:{m:.3f}' for s, m in rows.items()))

    # margin structure
    with torch.no_grad():
        dd = torch.cdist(embed.float(), embed.float())
        dd.fill_diagonal_(float('inf'))
        nn = dd.min(-1).values
    qs = np.quantile(nn.cpu().numpy(), [.05, .25, .5, .75, .95])
    results['nn_spacing'] = {k: float(v) for k, v in zip(['p05', 'p25', 'p50', 'p75', 'p95'], qs)}
    print('\nNN spacing p05/p25/p50/p75/p95:', np.round(qs, 4))
    Path('/tmp/vq_levers.json').write_text(json.dumps(results, indent=1))
    print('[Done] /tmp/vq_levers.json')


if __name__ == '__main__':
    main()
