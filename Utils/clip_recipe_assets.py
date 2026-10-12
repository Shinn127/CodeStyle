"""Fixed assets shared by the bounded V2 metric/controller entry points."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch

from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import configure_torch_runtime, load_norm_stats, set_seed
from Utils.reproducibility import artifact_fingerprints, validate_fingerprints
from Utils.vq_target_cache import VQTargetCache
from Utils.style_code_prior import load_style_code_prior

ROOT=Path(__file__).resolve().parents[1]
VQ=ROOT/'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror'


def add_asset_arguments(parser):
    parser.add_argument('--db',type=Path,default=ROOT/'database_100sty_375_memmap_nomirror')
    parser.add_argument('--vqvae-ckpt',type=Path,default=VQ/'best.pt')
    parser.add_argument('--norm-stats',type=Path,default=VQ/'norm_stats.npz')
    parser.add_argument('--vq-target-cache',type=Path,default=ROOT/'cache/vq_targets_vq150ep_nomirror_repair_v2')
    parser.add_argument('--style-code-prior',type=Path,default=ROOT/'cache/style_code_prior_vq150ep_nomirror_repair_v2.npz')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42)


def load_assets(args):
    device=torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    configure_torch_runtime(device=device,tf32=False,cudnn_benchmark=False,matmul_precision='highest')
    set_seed(args.seed)
    mean,std=load_norm_stats(args.norm_stats)
    fp=artifact_fingerprints(args.db,args.vqvae_ckpt,args.norm_stats)
    vq=build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device).eval().requires_grad_(False)
    cache=VQTargetCache(args.vq_target_cache)
    cache.validate(vqvae=vq,fingerprints=fp)
    prior=load_style_code_prior(args.style_code_prior)
    validate_fingerprints(prior.get('fingerprints'),fp)
    if prior.get('cache_sha256')!=cache.meta.indices_sha256:raise ValueError('Prior/cache contents differ')
    segments=json.loads((args.db/'segments.json').read_text())
    names=json.loads((args.db/'StyleNames.json').read_text())
    if len(names)!=len(prior['style_names']) or names!=prior['style_names']:raise ValueError('Prior style names/order differ')
    if len(names)<2 or any(s.get('split') not in ('train','val','test') for s in segments):
        raise ValueError('V2 requires an explicit IND train/val/test database')
    meta=json.loads((args.db/'meta.json').read_text())
    if meta.get('dataset_role')!='in_distribution':raise ValueError('OOD motion cannot enter generator/metric training')
    return device,vq,cache,mean,std,fp,prior,segments,names


def independent_train_clips(segments,frames=240):
    result={};seen=set()
    for s in segments:
        identity=s['source_entry_id']
        if s['split']=='train' and not s.get('mirror') and s['num_frames']>=frames and identity not in seen:
            seen.add(identity);result.setdefault(s['label'],[]).append(s)
    return result


def serializable_args(args):
    return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
