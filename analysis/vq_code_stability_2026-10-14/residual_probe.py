"""Quantized-decode residual probe: r = P(D(q)) - P(D(z_e)) and its time derivatives.

Companion to docs/VQ_DECODE_CONTINUITY_PLAN.md sections 1/5. The quantity to shrink is the
EXTRA temporal error quantization adds on the decode side, not the code-switch count:

    r_t  = p_q(t) - p_c(t)        (quantized decode minus continuous decode)
    Δr_t = r_t - r_{t-1}          (extra per-frame motion)
    Δ²r  = Δr_t - Δr_{t-1}        (extra acceleration / jitter band)

Computed on the 24 selected joints in two frames of reference:
* local  : positions in the root frame (motion291 channels 3:75), root motion excluded;
* world  : after integrating the root motion with the GT trajectory block
           (Utils.motion375.reconstruct_selected_globals_from_375).
All values in cm. Also reports quantized/continuous jitter and second differences directly,
because J_q - J_c is NOT J(r) (the plan's caveat), and GT second differences so a "frozen but
smooth" decoder cannot pass: a static output still errs against a dynamic GT.

Runs one model per invocation; writes one JSON. No repository files are modified.
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
from Utils.motion375 import reconstruct_selected_globals_from_375, SELECTED_24_NAMES

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'
N_JOINTS = len(SELECTED_24_NAMES)  # 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vq-ckpt', type=Path, required=True)
    ap.add_argument('--teacher-ckpt', type=Path, default=None,
                    help='optional frozen teacher; adds r_fixed = P(D(q)) - P(D0(E0(x)))')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    vq = build_vqvae_from_checkpoint(args.vq_ckpt).to(dev).eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    mean, std = load_norm_stats(NORM)
    mean_t = torch.as_tensor(mean, device=dev)
    std_t = torch.as_tensor(std, device=dev)
    segs = json.loads((DB / 'segments.json').read_text())
    meta = json.loads((DB / 'meta.json').read_text())
    data = np.memmap(DB / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    pool = [s for s in segs if s.get('split') == args.split and not s.get('mirror', False)
            and s['num_frames'] >= args.frames + 2]
    rng = np.random.RandomState(42)
    picks = [pool[i] for i in rng.choice(len(pool), min(args.clips, len(pool)), replace=False)]

    def frames_of(motion_norm: torch.Tensor):
        """normalized motion291 [T,291] -> (local_cm, world_cm) for the 24 selected joints.

        The 24-joint globals are rebuilt from the motion block's own root-motion channels
        (motion291 channels 0:3), so the 84-dim trajectory slot is filled with zeros purely to
        satisfy the 375 layout - the reconstruction reads only the motion block, and mixing a
        normalized trajectory block with raw motion units would be misleading.
        """
        motion_raw = motion_norm * std_t[84:] + mean_t[84:]
        local = motion_raw[:, 3:3 + N_JOINTS * 3].reshape(-1, N_JOINTS, 3) * 100.0
        layout = torch.cat((torch.zeros_like(motion_raw[:, :84]), motion_raw), -1).cpu().numpy()
        world, _ = reconstruct_selected_globals_from_375(layout)
        return local, torch.as_tensor(world, device=motion_norm.device) * 100.0

    # Optional frozen-teacher target: the alignment term B actually optimized is
    # ||D_theta(q) - sg(D0(E0(x)))||^2, so the residual against D0 is the term's own error
    # metric (r_fixed), distinct from the self-consistency residual r_self above.
    teacher = None
    if args.teacher_ckpt is not None:
        teacher = build_vqvae_from_checkpoint(args.teacher_ckpt, freeze=True).to(dev).eval()

    loc_r, loc_dr, loc_d2r = [], [], []
    wor_r, wor_dr, wor_d2r = [], [], []
    tch_r, tch_dr, tch_d2r = [], [], []
    jq_loc, jc_loc, jgt_loc = [], [], []
    d2q_loc, d2c_loc, d2gt_loc = [], [], []
    per_clip = []
    for s in picks:
        rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 2], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            z_e = vq.encode_pre_vq(x)
            z_q, _, _ = vq.encode(x)
            motion_q = vq.decode(z_q)
            motion_c = vq.decode(z_e)
            motion_t = teacher.decode(teacher.encode_pre_vq(x)) if teacher is not None else None
        lq, wq = frames_of(motion_q)
        lc, wc = frames_of(motion_c)
        lg, wg = frames_of(x[:, 84:])
        for rq, rc, name, store in ((lq, lc, 'local', (loc_r, loc_dr, loc_d2r)),
                                    (wq, wc, 'world', (wor_r, wor_dr, wor_d2r))):
            r = rq - rc
            dr = r[1:] - r[:-1]
            d2r = dr[1:] - dr[:-1]
            store[0].append(r.norm(dim=-1).reshape(-1).cpu())
            store[1].append(dr.norm(dim=-1).reshape(-1).cpu())
            store[2].append(d2r.norm(dim=-1).reshape(-1).cpu())
        if motion_t is not None:
            lt, _ = frames_of(motion_t)
            r = lq - lt
            dr = r[1:] - r[:-1]
            d2r = dr[1:] - dr[:-1]
            tch_r.append(r.norm(dim=-1).reshape(-1).cpu())
            tch_dr.append(dr.norm(dim=-1).reshape(-1).cpu())
            tch_d2r.append(d2r.norm(dim=-1).reshape(-1).cpu())
        # first-order displacement and second-difference magnitudes (local frame); the former
        # is NOT the project's official Jitter metric, only a cheap kinematic proxy.
        jq_loc.append((lq[1:] - lq[:-1]).norm(dim=-1).reshape(-1).cpu())
        jc_loc.append((lc[1:] - lc[:-1]).norm(dim=-1).reshape(-1).cpu())
        jgt_loc.append((lg[1:] - lg[:-1]).norm(dim=-1).reshape(-1).cpu())
        d2q_loc.append((lq[2:] - 2 * lq[1:-1] + lq[:-2]).norm(dim=-1).reshape(-1).cpu())
        d2c_loc.append((lc[2:] - 2 * lc[1:-1] + lc[:-2]).norm(dim=-1).reshape(-1).cpu())
        d2gt_loc.append((lg[2:] - 2 * lg[1:-1] + lg[:-2]).norm(dim=-1).reshape(-1).cpu())
        per_clip.append(dict(style=s.get('style_name'),
                             r_local_cm=float((lq - lc).norm(dim=-1).mean()),
                             dr_local_cm=float(((lq - lc)[1:] - (lq - lc)[:-1]).norm(dim=-1).mean()),
                             r_world_cm=float((wq - wc).norm(dim=-1).mean())))

    cat = torch.cat
    stat = lambda t: dict(mean=float(t.mean()), p50=float(t.median()),
                          p95=float(torch.quantile(t, torch.tensor(.95))), max=float(t.max()))
    out = dict(
        tag=args.tag, checkpoint=str(args.vq_ckpt), split=args.split,
        clips=len(picks), frames=args.frames, joints=N_JOINTS,
        teacher_ckpt=None if args.teacher_ckpt is None else str(args.teacher_ckpt),
        r_local_cm=stat(cat(loc_r)), dr_local_cm=stat(cat(loc_dr)), d2r_local_cm=stat(cat(loc_d2r)),
        r_world_cm=stat(cat(wor_r)), dr_world_cm=stat(cat(wor_dr)), d2r_world_cm=stat(cat(wor_d2r)),
        # first-order displacement magnitudes (NOT the project's official Jitter metric)
        step1_quant_local_cm=stat(cat(jq_loc)),
        step1_cont_local_cm=stat(cat(jc_loc)),
        step1_gt_local_cm=stat(cat(jgt_loc)),
        d2_quant_local_cm=stat(cat(d2q_loc)),
        d2_cont_local_cm=stat(cat(d2c_loc)),
        d2_gt_local_cm=stat(cat(d2gt_loc)),
        identity_note=('J_q - J_c != J(r); quantized/continuous/GT reported directly; '
                       'step1_* are first-order displacement proxies, not formal Jitter'),
        per_clip=per_clip,
    )
    if tch_r:
        out['r_fixed_local_cm'] = stat(cat(tch_r))
        out['dr_fixed_local_cm'] = stat(cat(tch_dr))
        out['d2r_fixed_local_cm'] = stat(cat(tch_d2r))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('[Done]', args.out)


if __name__ == '__main__':
    main()
