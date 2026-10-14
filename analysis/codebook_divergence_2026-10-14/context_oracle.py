"""Continuous-context memorization oracle: how deterministic is the next code given the
model's actual input context (previous frame features + trajectory)?

For each test frame, find the k nearest TRAIN frames in normalized feature space (the same
375-dim vector the controller sees as x_curr) and vote their next-frame codes. The vote
accuracy is an upper-bound estimate for any predictor limited to this context: if train
frames with nearly identical context disagree on the next code, the mismatch is
information-limited (multimodality / codec fragility), not a training deficiency.

Also reports the same oracle on the *latent* context (z_prev code only, matching the
n-gram probe) for a like-for-like comparison against the controller and n-gram ceilings.
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
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    frames = 240
    k = 8
    dev = torch.device('cuda')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_motion_model('flowmatch_target', ROOT / 'checkpoints/flowmatch_label_history_obsnoise_tf100/best.pt', num_styles=90).to(dev)
    model.eval().requires_grad_(False)
    vq = model.vqvae
    mean, std = load_norm_stats(NORM)
    mean_t = torch.tensor(mean, device=dev); std_t = torch.tensor(std, device=dev)
    segs = json.loads((DB / 'segments.json').read_text())
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))

    train_segs = [s for s in segs if s.get('split') == 'train' and not s.get('mirror', False) and s['num_frames'] >= frames + 1]
    test_segs = [s for s in segs if s.get('split') == 'test' and not s.get('mirror', False) and s['num_frames'] >= frames + 1]
    rng = np.random.RandomState(0)
    train_pick = [train_segs[i] for i in rng.choice(len(train_segs), min(n_train, len(train_segs)), replace=False)]
    test_pick = [test_segs[i] for i in rng.choice(len(test_segs), min(n_test, len(test_segs)), replace=False)]

    def load_clip(s):
        rows = np.array(data[s['frame_start']:s['frame_start'] + frames + 1], dtype=np.float32)
        return rows

    # build train bank: normalized context features (375) and next codes
    bank_x, bank_next = [], []
    for s in train_pick:
        rows = load_clip(s)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            _, idx, _ = vq.encode(x)
        idx = idx.reshape(len(x), 32)
        bank_x.append(x[:-1])            # context at t
        bank_next.append(idx[1:])        # code at t+1
    bank_x = torch.cat(bank_x); bank_next = torch.cat(bank_next)
    print(f'[Info] train bank {tuple(bank_x.shape)}')

    acc_nearest, acc_vote = [], []
    acc_nearest_stable, acc_nearest_trans = [], []
    dists = []
    vote_stable, vote_trans = [], []
    for s in test_pick:
        rows = load_clip(s)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            _, idx, _ = vq.encode(x)
        idx = idx.reshape(len(x), 32)
        q = x[:-1]                       # queries: contexts t=0..frames-1
        target = idx[1:]                 # GT next codes
        # chunked distance search
        best_d = None; best_idx = None
        for b in range(0, len(q), 256):
            d = torch.cdist(q[b:b + 256], bank_x)
            dv, di = d.topk(k, largest=False)
            if best_d is None:
                best_d, best_idx = dv, di
            else:
                best_d = torch.cat([best_d, dv]); best_idx = torch.cat([best_idx, di])
        nn_codes = bank_next[best_idx]                       # [Q,k,32]
        nearest = nn_codes[:, 0]
        # majority vote per slot
        votes = torch.zeros(len(q), 32, 2048, device=dev, dtype=torch.float32)
        for j in range(k):
            votes.scatter_(2, nn_codes[:, j][:, :, None], 1.0)
        vote = votes.argmax(-1)
        prev = idx[:-1]
        stable = prev == target
        acc_nearest.append(float((nearest == target).float().mean()))
        acc_vote.append(float((vote == target).float().mean()))
        acc_nearest_stable.append(float((nearest == target).float()[stable].mean()))
        acc_nearest_trans.append(float((nearest == target).float()[~stable].mean()))
        vote_stable.append(float((vote == target).float()[stable].mean()))
        vote_trans.append(float((vote == target).float()[~stable].mean()))
        dists.append(best_d[:, 0].mean().item())
    out = dict(
        k=k, train_clips=len(train_pick), test_clips=len(test_pick), frames=frames,
        nearest_1nn=float(np.mean(acc_nearest)),
        vote_knn=float(np.mean(acc_vote)),
        nearest_stable=float(np.mean(acc_nearest_stable)),
        nearest_transition=float(np.mean(acc_nearest_trans)),
        vote_stable=float(np.mean(vote_stable)),
        vote_transition=float(np.mean(vote_trans)),
        mean_nearest_context_distance=float(np.mean(dists)),
    )
    print(json.dumps(out, indent=1))
    Path('/tmp/context_oracle.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
