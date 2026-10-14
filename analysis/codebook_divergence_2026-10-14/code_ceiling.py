"""Code-predictability ceiling: what can be achieved from code history alone?

Motivation: the controller reaches ~0.38/0.29 per-slot single-step code match, and the
closed loop decays to ~0.05. Before investing in training changes, quantify how much of
the mismatch is information-limited (mode not the GT code) versus model-limited, by
fitting simple per-slot n-gram predictors on train codes and evaluating them on test.

Predictors (all per slot, fit on train, evaluated on test):
  persistence : next = prev                       (floor)
  bigram      : argmax P(next | prev)             (backoff: persistence)
  trigram     : argmax P(next | prev2, prev)      (backoff: bigram)
  bayes_top1  : mean over test slots of the smoothed train probability of the GT code
                under the bigram context  (expected match of an oracle n-gram decoder)
Also reports stable/transition splits to line up with the controller probes.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/shinn/桌面/multimedia/revision/NewCode')
sys.path.insert(0, str(ROOT))
from Utils.motion_eval import load_motion_model
from Utils.runtime import load_norm_stats

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'
K = 2048


def code_idx(vq, z_raw):
    x = z_raw.reshape(-1, vq.code_dim).float()
    return torch.cdist(x, vq.quantizer.embedding_weight.detach().float()).argmin(-1).reshape(
        z_raw.shape[0], vq.num_latents)


def main():
    clips_train = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    frames = 240
    dev = torch.device('cuda')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_motion_model('flowmatch_target', ROOT / 'checkpoints/flowmatch_label_history_obsnoise_tf100/best.pt', num_styles=90).to(dev)
    model.eval().requires_grad_(False)
    vq = model.vqvae
    mean, std = load_norm_stats(NORM)
    segs = json.loads((DB / 'segments.json').read_text())
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))

    def codes_of(s, n=240):
        rows = np.array(data[s['frame_start']:s['frame_start'] + n + 1], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            _, idx, _ = vq.encode(x)
        return idx.reshape(len(x), 32).cpu().numpy()

    train_segs = [s for s in segs if s.get('split') == 'train' and not s.get('mirror', False)
                  and s['num_frames'] >= frames + 1]
    test_segs = [s for s in segs if s.get('split') == 'test' and not s.get('mirror', False)
                 and s['num_frames'] >= frames + 1]
    rng = np.random.RandomState(0)
    train_pick = [train_segs[i] for i in rng.choice(len(train_segs), min(clips_train, len(train_segs)), replace=False)]
    test_pick = test_segs if len(test_segs) <= 60 else [test_segs[i] for i in rng.choice(len(test_segs), 60, replace=False)]
    print(f'[Info] train clips {len(train_pick)} / test clips {len(test_pick)}')

    # per-slot counts
    bi = [defaultdict(lambda: defaultdict(int)) for _ in range(32)]   # [slot][prev][next]
    tri = [defaultdict(lambda: defaultdict(int)) for _ in range(32)]  # [slot][(p2,p1)][next]
    for s in train_pick:
        idx = codes_of(s)
        for t in range(2, len(idx)):
            for slot in range(32):
                p2, p1, nx = int(idx[t - 2, slot]), int(idx[t - 1, slot]), int(idx[t, slot])
                bi[slot][p1][nx] += 1
                tri[slot][(p2, p1)][nx] += 1

    ALPHA = 0.5
    def bi_argmax(slot, p1):
        d = bi[slot].get(p1)
        return max(d.items(), key=lambda kv: kv[1])[0] if d else None

    def tri_argmax(slot, p2, p1):
        d = tri[slot].get((p2, p1))
        return max(d.items(), key=lambda kv: kv[1])[0] if d else None

    def bi_prob(slot, p1, target):
        d = bi[slot].get(p1)
        if not d:
            return 1.0 / K
        tot = sum(d.values())
        return (d.get(target, 0) + ALPHA) / (tot + ALPHA * K)

    stats = {name: {'all': [], 'stable': [], 'trans': []} for name in ('persistence', 'bigram', 'trigram')}
    bayes_top1 = []
    for s in test_pick:
        idx = codes_of(s)
        n = len(idx)
        for t in range(2, n):
            for slot in range(32):
                p2, p1, gt = int(idx[t - 2, slot]), int(idx[t - 1, slot]), int(idx[t, slot])
                stable = (p1 == gt)
                bucket = 'stable' if p1 == gt else 'trans'
                preds = {
                    'persistence': p1,
                    'bigram': bi_argmax(slot, p1) if bi_argmax(slot, p1) is not None else p1,
                    'trigram': tri_argmax(slot, p2, p1) if tri_argmax(slot, p2, p1) is not None else (
                        bi_argmax(slot, p1) if bi_argmax(slot, p1) is not None else p1),
                }
                for name, pred in preds.items():
                    hit = float(pred == gt)
                    stats[name]['all'].append(hit)
                    stats[name]['stable'].append(hit if bucket == 'stable' else np.nan)
                    stats[name]['trans'].append(hit if bucket == 'trans' else np.nan)
                bayes_top1.append(bi_prob(slot, p1, gt))

    def nanmean(v):
        v = np.asarray(v, dtype=np.float64)
        return float(np.nanmean(v)) if len(v) else None

    out = {name: {'all': float(np.mean(v['all'])), 'stable': nanmean(v['stable']), 'trans': nanmean(v['trans'])}
           for name, v in stats.items()}
    out['bayes_top1_bigram'] = float(np.mean(bayes_top1))
    out['stable_frac'] = float(np.mean([not np.isnan(x) for x in stats['persistence']['stable']])) if stats['persistence']['stable'] else None
    print(json.dumps(out, indent=1))
    Path('/tmp/code_ceiling.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
