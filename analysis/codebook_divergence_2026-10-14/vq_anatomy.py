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
from Utils.runtime import load_norm_stats, load_checkpoint
from arch.basic_vqvae_375to291 import codebook_boundary_distance
from Utils.motion_eval import compute_sequence_metric_sums, MotionEvalConfig
from Utils.reproducibility import file_sha256, state_fingerprint

DB = ROOT / 'database_100sty_375_memmap_nomirror'
NORM = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz'
VQ_CKPT = ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--split', choices=('train','val','test'), default='val')
    ap.add_argument('--db',type=Path,default=DB)
    ap.add_argument('--norm-stats',type=Path,default=None)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--vq-ckpt', type=Path, default=VQ_CKPT,
                    help='VQ checkpoint to probe (default: the frozen 150ep K=2048 baseline)')
    ap.add_argument('--out', default=str(Path(__file__).with_name('vq_anatomy.json')))
    args = ap.parse_args()
    if args.clips<1 or args.frames<2:ap.error('Need clips>=1 and frames>=2')
    dev = torch.device(args.device)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    vq = build_vqvae_from_checkpoint(args.vq_ckpt).to(dev).eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    before=state_fingerprint(vq.state_dict())
    checkpoint_meta=load_checkpoint(args.vq_ckpt,map_location='cpu')
    checkpoint_args=checkpoint_meta.get('args') or {}
    checkpoint_epoch=checkpoint_meta.get('epoch');del checkpoint_meta
    norm=args.norm_stats or args.vq_ckpt.parent/'norm_stats.npz'
    if not norm.is_file():ap.error('Checkpoint normalization is missing; pass --norm-stats explicitly')
    mean, std = load_norm_stats(norm)
    segs = json.loads((args.db / 'segments.json').read_text())
    meta = json.loads((args.db / 'meta.json').read_text())
    data = np.memmap(args.db / 'X.f32', mode='r', dtype=np.float32, shape=(int(meta['num_frames']), 375))
    pool = [s for s in segs if s.get('split') == args.split and not s.get('mirror', False)
            and s['num_frames'] >= args.frames + 1]
    if not pool:ap.error('No eligible clips in the requested split')
    rng = np.random.RandomState(args.seed)
    noise_generator=torch.Generator(device=dev).manual_seed(args.seed+1)
    picks = [pool[i] for i in rng.choice(len(pool), min(args.clips, len(pool)), replace=False)]
    embed = vq.quantizer.embedding_weight.detach().float()
    if embed.dim() == 3:
        embed = embed[0]
    K, D = embed.shape
    print(f'[Info] split={args.split} clips={len(picks)} frames={args.frames} K={K} D={D}')

    r1s, bdists, dmags, dctrl, dcont, recon_cms = [], [], [], [], [], []
    n_heads = int(vq.num_latents)
    flips, flips_ctrl_list, ratios, per_head_flip, per_head_n = [], [], [], np.zeros(n_heads), np.zeros(n_heads)
    switch_gts = []
    legacy_distances=[];valid_boundaries=[];relative_boundaries=[];motion_metrics=[];decode_displacements=[]
    continuous_metrics=[];gt_metrics=[]
    frame_counts=[];geometry_disagreements=[]
    for s in picks:
        rows = np.array(data[s['frame_start']:s['frame_start'] + args.frames + 1], dtype=np.float32)
        x = torch.as_tensor((rows - mean) / std, device=dev)
        with torch.no_grad():
            z_e = vq.encode_pre_vq(x)                    # [T,128] continuous
            z_q, idx, _ = vq.encode(x)                   # quantized path
            idx = idx.reshape(len(x), n_heads)
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
            eps = torch.randn(diff.shape,device=dev,generator=noise_generator)
            eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-9) * norm_per_frame
            x_ctrl = torch.cat((x[:, :84], x[:, 84:] + eps), -1)
            z_ectrl = vq.encode_pre_vq(x_ctrl)
        # per-slot geometry of the ORIGINAL encode
        ze = z_e.reshape(-1, n_heads, D); c1 = z_q.reshape(-1, n_heads, D)
        r1 = (ze - c1).norm(dim=-1)                                       # [T,32]
        d2_all = torch.cdist(ze.reshape(-1, D).float(), embed)            # [T*32,K]
        own = idx.reshape(-1)
        geometry_disagreements.append(float((d2_all.argmin(-1)!=own).float().mean()))
        bdist_exact,valid=codebook_boundary_distance(ze.reshape(-1,D),embed,assigned_indices=own)
        legacy,_=codebook_boundary_distance(ze.reshape(-1,D),embed,mode='nearest_code',assigned_indices=own)
        # Scale-normalized radius: raw latent rescaling alone must not count as stability.
        separations=torch.cdist(embed,embed,compute_mode='donot_use_mm_for_euclid_dist')
        separations.masked_fill_(separations<=1e-9,float('inf'))
        spacing=separations.min(-1).values[own]
        legacy_distances.append(legacy.cpu());valid_boundaries.append(valid.cpu())
        relative_boundaries.append((bdist_exact/(.5*spacing).clamp_min(1e-9)).cpu())
        r1 = r1.reshape(-1); bdist_exact = bdist_exact.reshape(-1)
        # roundtrip drift and flips
        dq = (z_e2 - z_e).reshape(-1, n_heads, D).norm(dim=-1).reshape(-1)
        dc = (z_e3 - z_e).reshape(-1, n_heads, D).norm(dim=-1).reshape(-1)
        dk = (z_ectrl - z_e).reshape(-1, n_heads, D).norm(dim=-1).reshape(-1)
        idx2 = (torch.cdist(z_e2.reshape(-1, D).float(), embed).argmin(-1))
        flip = (idx2 != own).float()
        d2_all = torch.cdist(z_ectrl.reshape(-1, D).float(), embed)
        idxc = d2_all.argmin(-1)
        flip_ctrl = (idxc != own).float()
        # Only positions have length units. A motion291 MAE mixes rotations and velocities.
        pos_delta=(motion_hat[:,3:75]-x[:,87:159])*torch.as_tensor(std,device=dev)[87:159]
        rec_cm=pos_delta.reshape(-1,24,3).norm(dim=-1).mean()*100
        raw=np.array(rows,copy=True);reconstructed=np.array(raw,copy=True)
        reconstructed[:,84:]=motion_hat.cpu().numpy()*std[84:]+mean[84:]
        sums=compute_sequence_metric_sums(reference_375=raw,prediction_375=reconstructed,cfg=MotionEvalConfig())
        motion_metrics.append(sums);frame_counts.append(len(x))
        continuous=np.array(raw,copy=True);continuous[:,84:]=motion_c.cpu().numpy()*std[84:]+mean[84:]
        continuous_metrics.append(compute_sequence_metric_sums(reference_375=raw,prediction_375=continuous,cfg=MotionEvalConfig()))
        gt_metrics.append(compute_sequence_metric_sums(reference_375=raw,prediction_375=raw,cfg=MotionEvalConfig()))
        with torch.no_grad():
            _,idx_rt,_=vq.encode(x2)
            motion_rt=vq.decode(vq.quantizer.lookup(idx_rt))
            impact=(motion_rt[:,3:75]-motion_hat[:,3:75])*torch.as_tensor(std,device=dev)[87:159]
            decode_displacements.append(float(impact.reshape(-1,24,3).norm(dim=-1).mean()*100))
        r1s.append(r1.cpu()); bdists.append(bdist_exact.cpu()); dmags.append(dq.cpu())
        dctrl.append(dk.cpu()); dcont.append(dc.cpu()); flips.append(flip.cpu())
        flips_ctrl_list.append(flip_ctrl.cpu()); recon_cms.append(float(rec_cm))
        ratios.append((dq / bdist_exact.clamp_min(1e-9)).cpu())
        ph = flip.reshape(-1, n_heads).mean(0).cpu().numpy()
        per_head_flip += ph; per_head_n += 1
        sw = (idx[1:] != idx[:-1]).float().sum(-1).mean().item()
        switch_gts.append(sw)

    cat = lambda L: torch.cat(L)
    q = lambda t, p: [round(v, 4) for v in torch.quantile(t, torch.tensor(p)).tolist()]
    out = {
        'schema':'vq_anatomy_exact_boundary_v2','split': args.split, 'clips': len(picks),
        'frames_total':sum(frame_counts),'transition_intervals_total':sum(n-1 for n in frame_counts),
        'seed':args.seed,'starts':[s['frame_start'] for s in picks],
        'source_entry_ids':[s['source_entry_id'] for s in picks],
        'vq_sha256':file_sha256(args.vq_ckpt),'vq_state_sha256':before,'norm_sha256':file_sha256(norm),
        'checkpoint_epoch':checkpoint_epoch,
        'trained_margin_mode':checkpoint_args.get('latent_margin_mode','nearest_code') if checkpoint_args.get('latent_margin_weight',0)>0 else 'disabled',
        'r1': {'mean': float(cat(r1s).mean()), 'p50': q(cat(r1s), [.5])[0], 'p90': q(cat(r1s), [.9])[0]},
        'bdist': {'mean': float(cat(bdists).mean()), 'p10': q(cat(bdists), [.1])[0],
                  'p50': q(cat(bdists), [.5])[0], 'p90': q(cat(bdists), [.9])[0]},
        'drift_quant': {'mean': float(cat(dmags).mean()), 'p50': q(cat(dmags), [.5])[0], 'p90': q(cat(dmags), [.9])[0]},
        'drift_cont': {'mean': float(cat(dcont).mean()), 'p50': q(cat(dcont), [.5])[0]},
        'drift_ctrl_noise': {'mean': float(cat(dctrl).mean()), 'p50': q(cat(dctrl), [.5])[0]},
        'flip_rate': float(cat(flips).mean()),
        'flip_rate_noise_control': float(cat(flips_ctrl_list).mean()),
        'flip_given_drift_gt_bdist': float(cat(flips)[cat(dmags)>cat(bdists)].mean()) if (cat(dmags)>cat(bdists)).any() else None,
        'flip_given_drift_le_bdist': float(cat(flips)[cat(dmags)<=cat(bdists)].mean()) if (cat(dmags)<=cat(bdists)).any() else None,
        'r1_over_bdist_mean': float(cat(ratios).clamp_max(100).mean()),
        'recon_cm_mean': float(np.mean(recon_cms)),
        'gt_switch_per_frame': float(np.mean(switch_gts)),
        'gt_switch_fraction_per_slot':float(np.mean(switch_gts)/n_heads),
        'legacy_second_code_bdist_mean':float(cat(legacy_distances).mean()),
        'legacy_overestimates_fraction':float((cat(legacy_distances)>cat(bdists)+1e-6).float().mean()),
        'boundary_valid_fraction':float(cat(valid_boundaries).float().mean()),
        'relative_boundary_p50':q(cat(relative_boundaries),[.5])[0],
        'boundary_below_005_fraction':float((cat(bdists)<.05).float().mean()),
        'quantizer_search_disagreement_fraction':float(np.mean(geometry_disagreements)),
        'reconstruction_metric':'local_position_mpjpe_24j_cm; legacy motion291 MAE was not cm',
        'cycle_pose_displacement_cm':float(np.mean(decode_displacements)),
        'per_head_flip_rate': (per_head_flip / per_head_n).tolist(),
    }
    for name,metrics in (('motion_quality',motion_metrics),('continuous_motion_quality',continuous_metrics),('gt_motion_quality',gt_metrics)):
        total=lambda field:sum(getattr(m,field) for m in metrics)
        out[name]=dict(mpjpe_cm=total('mpjpe_sum_cm')/total('mpjpe_count'),
            root_ade_cm=total('root_ade_sum_cm')/total('root_ade_count'),
            jitter_cm=total('jitter_sum_cm')/total('jitter_count'),
            local_fs_cm=total('foot_sliding_sum_cm')/total('foot_sliding_count') if total('foot_sliding_count') else None,
            world_fs_cm=total('foot_sliding_world_sum_cm')/total('foot_sliding_world_count') if total('foot_sliding_world_count') else None,
            world_contact_count=total('foot_sliding_world_count'))
    assert state_fingerprint(vq.state_dict())==before
    print(json.dumps(out, indent=1))
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
