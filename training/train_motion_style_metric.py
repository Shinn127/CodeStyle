"""Bounded IND-only metric pretraining, with held-style independent-clip retrieval."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from arch.clip_recipe_v2 import MotionStyleMetric
from Utils.runtime import require_unmirrored_db
from Utils.clip_recipe_assets import ROOT,add_asset_arguments,load_assets,independent_train_clips,serializable_args
from Utils.reproducibility import isolated_rng,state_fingerprint,rng_state


def supervised_contrastive(features,labels):
    similarity=features@features.T/.1
    diagonal=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    positive=(labels[:,None]==labels[None])&~diagonal
    if not positive.any(1).all():raise ValueError('Every metric anchor needs an independent same-style positive')
    logp=similarity-similarity.masked_fill(diagonal,-float('inf')).logsumexp(1,keepdim=True)
    return -(logp.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()


def read_windows(plans,source,cache,vq,mean,std,device):
    points=np.linspace(0,238,64).astype(np.int64)+1
    rows=np.array([start for _,start in plans])[:,None]+points
    gt=torch.from_numpy((np.array(source[rows,84:],copy=True)-mean[84:])/std[84:]).to(device)
    codes=torch.from_numpy(np.array(cache.indices[rows],dtype=np.int64)).to(device)
    with torch.no_grad():
        z=vq.quantizer.lookup(codes.reshape(-1,codes.shape[-1])).reshape(len(plans),64,vq.latent_dim)
        decoded=vq.decode(z)
    stamp=torch.as_tensor(points/60.,dtype=torch.float32,device=device).expand(len(plans),-1)
    return gt,decoded,stamp


@torch.no_grad()
def retrieval(metric,held,clips,source,cache,vq,mean,std,device,seed):
    metric.eval();rng=np.random.default_rng(seed);support=[];query=[];query_labels=[]
    for i,label in enumerate(held):
        choices=clips[label]
        if len(choices)<4:raise ValueError('Held-style retrieval needs independent support/query clips')
        plans=[]
        for s in choices[:6]:
            start=s['frame_start']+int(rng.integers(s['num_frames']-240+1));plans.append((label,start))
        gt,ae,times=read_windows(plans,source,cache,vq,mean,std,device)
        # Three source clips form a prototype; all remaining original clips are queries.
        support.append(F.normalize(metric(gt[:3],times[:3],dense=True).mean(0),dim=0))
        query.extend((metric(gt[3:],times[3:],dense=True),metric(ae[3:],times[3:],dense=True)))
        query_labels.extend([i]*(2*(len(plans)-3)))
    features=torch.cat(query);targets=torch.tensor(query_labels,device=device)
    accuracy=float(((features@torch.stack(support).T).argmax(1)==targets).float().mean())
    return dict(held_style_retrieval=accuracy,held_styles=held,held_style_count=len(held),
        independent_query_cases=len(targets),chance=1/len(held),protocol='IND-held-style_GT_and_VQ_AE_64anchors_240f')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);add_asset_arguments(parser)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'checkpoints/clip_recipe_v2/style_metric')
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--held-styles',type=int,default=10)
    parser.add_argument('--hidden',type=int,default=128)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--styles-per-batch',type=int,default=8)
    parser.add_argument('--val-every',type=int,default=100)
    args=parser.parse_args(argv)
    require_unmirrored_db(args.db)
    if not 1<=args.updates<=1000 or args.val_every<1:parser.error('Metric pretraining is bounded to 1..1000 updates')
    if (args.out_dir/'latest.pt').exists():raise ValueError('Metric run already exists; choose an independent output directory')
    device,vq,cache,mean,std,fp,_,segments,names=load_assets(args)
    clips=independent_train_clips(segments)
    if set(clips)!=set(range(len(names))) or min(map(len,clips.values()))<6:
        raise ValueError('Metric training needs six independent train clips/style')
    if not 1<=args.held_styles<len(names):raise ValueError('Held-style count leaves no metric training styles')
    rng=np.random.default_rng(args.seed)
    held=sorted(rng.choice(len(names),args.held_styles,replace=False).tolist())
    trained=[i for i in range(len(names)) if i not in held]
    source=np.memmap(args.db/'X.f32',mode='r',dtype=np.float32,shape=(cache.meta.num_frames,375))
    metric=MotionStyleMetric(args.hidden,args.layers).to(device)
    optimizer=torch.optim.AdamW(metric.parameters(),lr=3e-4,weight_decay=1e-4,
        fused=device.type=='cuda',foreach=False if device.type=='cuda' else None)
    frozen=state_fingerprint(vq.state_dict());best=-1.;started=time.perf_counter();args.out_dir.mkdir(parents=True,exist_ok=True)
    for update in range(1,args.updates+1):
        metric.train();plans=[];labels=[]
        styles=torch.randperm(len(trained))[:min(args.styles_per_batch,len(trained))].tolist()
        for index in styles:
            label=trained[index];candidates=clips[label]
            first=int(torch.randint(len(candidates),()))
            alternate=[i for i,s in enumerate(candidates) if i!=first and s.get('suffix')!=candidates[first].get('suffix')]
            if not alternate:alternate=[i for i in range(len(candidates)) if i!=first]
            second=alternate[int(torch.randint(len(alternate),()))]
            for pick in (first,second):
                s=candidates[pick];start=s['frame_start']+int(torch.randint(s['num_frames']-240+1,()))
                plans.append((label,start));labels.append(label)
        gt,ae,stamp=read_windows(plans,source,cache,vq,mean,std,device)
        mix=torch.rand((len(gt),1,1),device=device)<.5
        features=metric(torch.where(mix,gt,ae),stamp,dense=True)
        loss=supervised_contrastive(features,torch.tensor(labels,device=device))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(metric.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update%args.val_every==0 or update==args.updates:
            with isolated_rng(args.seed):validation=retrieval(metric,held,clips,source,cache,vq,mean,std,device,args.seed)
            state=dict(format='motion_style_metric_v1',model=metric.state_dict(),architecture=dict(hidden=args.hidden,layers=args.layers,tokens=64),
                optimizer=optimizer.state_dict(),rng_state=rng_state(),
                fingerprints=fp,train_styles=trained,validation=validation,args=serializable_args(args),updates=update,
                input_spec='normalized_root_frame_position_rotation_and_time_difference',
                fp32=True,fps=60,window_frames=240,vq_unchanged=state_fingerprint(vq.state_dict())==frozen)
            assert state['vq_unchanged'] and all(p.grad is None for p in vq.parameters())
            if validation['held_style_retrieval']>best:best=validation['held_style_retrieval'];torch.save(state,args.out_dir/'best.pt')
            torch.save(state,args.out_dir/'latest.pt')
            row=dict(update=update,loss=float(loss.detach()),validation=validation,elapsed_seconds=time.perf_counter()-started)
            with (args.out_dir/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
