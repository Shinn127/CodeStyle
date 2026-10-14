"""Bounded FP32 forward/backward timing, no optimizer updates or EMA changes."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import load_norm_stats
from Utils.reproducibility import state_fingerprint


def main():
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(2)
    checkpoint=ROOT/'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt'
    model=build_vqvae_from_checkpoint(checkpoint,freeze=False).cuda().train()
    # Keep the training STE/rotation-trick graph, freeze only the library EMA update.
    forward=model.quantizer.vq.forward
    model.quantizer.vq.forward=lambda *a,**kw:forward(*a,freeze_codebook=True,**kw)
    before=state_fingerprint(model.state_dict())
    db=ROOT/'database_100sty_375_memmap_nomirror'
    meta=json.load(open(db/'meta.json'));segments=json.load(open(db/'segments.json'))
    data=np.memmap(db/'X.f32',mode='r',dtype=np.float32,shape=(meta['num_frames'],375))
    train=[s for s in segments if s['split']=='train' and s['num_frames']>=3]
    rng=np.random.default_rng(42);rows=[]
    for _ in range(512):
        s=train[int(rng.integers(len(train)))];start=s['frame_start']+int(rng.integers(s['num_frames']-1))
        rows.append(np.array(data[start:start+2]))
    mean,std=load_norm_stats(checkpoint.parent/'norm_stats.npz')
    x=torch.from_numpy((np.stack(rows)-mean)/std).cuda()
    common=dict(recon_weight=3,vq_weight=1,vel_weight=15,root_weight=3,
        code_align_weight=2,latent_l1_weight=.01,latent_vel_weight=.1)
    modes=[('baseline',{}),('legacy_margin_rt',dict(latent_margin_weight=1,roundtrip_weight=1,latent_margin_mode='nearest_code')),
           ('exact_margin_rt',dict(latent_margin_weight=1,roundtrip_weight=1))]
    results=[]
    for name,extra in modes:
        durations=[];torch.cuda.reset_peak_memory_stats()
        for iteration in range(13):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.perf_counter()
            loss=model.compute_loss(x,**common,**extra);loss['loss'].backward()
            torch.cuda.synchronize()
            if iteration>=3:durations.append((time.perf_counter()-t0)*1000)
        results.append(dict(mode=name,mean_ms=float(np.mean(durations)),p95_ms=float(np.percentile(durations,95)),
            allocated_peak_gb=torch.cuda.max_memory_allocated()/2**30))
    # Gradient scale is independent from a loss's numerical share.
    model.zero_grad(set_to_none=True);loss=model.compute_loss(x[:32],**common,latent_margin_weight=1,roundtrip_weight=1)
    parameters=list(model.encoder.parameters())+list(model.decoder.parameters());nenc=len(list(model.encoder.parameters()))
    scales={}
    parts={'baseline':loss['loss']-loss['latent_margin_loss']-loss['roundtrip_loss'],
           'margin':loss['latent_margin_loss'],'roundtrip':loss['roundtrip_loss']}
    for name,value in parts.items():
        grad=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        norm=lambda pieces:float(torch.stack([g.square().sum() for g in pieces if g is not None]).sum().sqrt()) if any(g is not None for g in pieces) else 0.
        scales[name]=dict(encoder_l2=norm(grad[:nenc]),decoder_l2=norm(grad[nenc:]))
    assert state_fingerprint(model.state_dict())==before
    out=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,batch=512,seq_len=2,seed=42,
        fp32=True,tf32=False,measured_iterations=10,results=results,gradient_norms_batch32=scales,
        state_unchanged=True,protocol='training_graph_frozen_ema_fwd_backward_no_optimizer_not_epoch_timing')
    Path(__file__).with_name('performance.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
