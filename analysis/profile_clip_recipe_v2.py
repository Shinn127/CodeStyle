"""Disposable real-data V2 smoke; random metric checks cost/interfaces, not style quality."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arch.clip_recipe_v2 import build_clip_recipe_v2,MotionStyleMetric
from dataset import build_train_val_test_dataloaders
from Utils.clip_recipe_assets import add_asset_arguments,load_assets
from Utils.clip_recipe_training import TemporalReferencePool,ReferenceMetricCache,reference_latents
from Utils.reproducibility import validate_fingerprints,state_fingerprint
from training.train_clip_recipe_v2 import run_epoch,balanced_window_weights
from training.train_flowmatch_vqvae_controller import _load_latent_stats,configure_controller_loader,_seed_loader


def gpu_process_inventory():
    try:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],
            capture_output=True,text=True,timeout=10,check=True)
        return dict(benchmark_pid=os.getpid(),processes=result.stdout.strip().splitlines())
    except (OSError,subprocess.SubprocessError) as error:return dict(unavailable=str(error))


def main():
    p=argparse.ArgumentParser(description=__doc__);add_asset_arguments(p)
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--microbatch',type=int,default=512)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--backward-mode',choices=('combined','split'),default='combined')
    p.add_argument('--reference-cache-device',choices=('gpu','cpu'),default='gpu')
    p.add_argument('--cuda-field-graph',action='store_true')
    p.add_argument('--gpu-data-cache-gb',type=float,default=0.)
    p.add_argument('--compare',action='store_true',help='Alternate baseline/optimized 25-update blocks under the same GPU load')
    p.add_argument('--out',type=Path,default=ROOT/'analysis/clip_recipe_v2_implementation/smoke.json')
    p.add_argument('--latent-stats',type=Path,default=ROOT/'checkpoints/clip_recipe_gt_history50/latent_stats.npz')
    args=p.parse_args()
    if not 1<=args.updates<=100 or not 1<=args.warmup<=100:p.error('Smoke is bounded to <=100 timed updates')
    if args.out.exists():p.error('Choose a new output path; existing results are preserved')
    torch.set_num_threads(2)
    device,vq,cache,mean,std,fp,prior,segments,names=load_assets(args)
    config=dict(style_slot_indices=prior['selected_slots'].tolist(),conditioning_version=2,
        flow_hidden_dim=256,flow_layers=4,flow_ffn_dim=1024,style_guidance_scale=1.5)
    model=build_clip_recipe_v2(config,vqvae=vq,num_styles=len(names)).to(device)
    model.backward_mode=args.backward_mode
    stats=_load_latent_stats(args.latent_stats,latent_dim=model.latent_dim)
    validate_fingerprints(stats['fingerprints'],fp)
    if stats['split']!='train':raise ValueError('Smoke latent stats must be train-only')
    model.set_latent_prior_stats(stats['offset'],stats['scale'],transport_std=stats['transport_std'],
        data_min=stats['data_min'],data_max=stats['data_max'])
    model.set_style_code_prior(torch.as_tensor(prior['prior'],device=device))
    baseline=None
    if args.compare:
        import copy,types
        from arch.flowmatch_vqvae_controller import FlowMatchVQVAEController
        baseline=copy.deepcopy(model);baseline.backward_mode='split'
        baseline.requantize_latent=types.MethodType(FlowMatchVQVAEController.requantize_latent,baseline)
    if args.cuda_field_graph:
        from Utils.clip_recipe_cuda import graph_training_field
        graph_training_field(model,args.microbatch)
    metric_cache=ReferenceMetricCache(MotionStyleMetric().to(device),device_cache=args.reference_cache_device=='gpu')
    vq_before=state_fingerprint(vq.state_dict());metric_before=state_fingerprint(metric_cache.metric.state_dict())
    bundle=build_train_val_test_dataloaders(db_path=args.db,seq_len=240,stride=1,batch_size=8,num_workers=0,
        norm_mean=mean,norm_std=std,normalize=True,drop_last=True)
    data=bundle.train_dataset;sampler=WeightedRandomSampler(balanced_window_weights(data),args.warmup*8,replacement=True)
    loader=DataLoader(data,batch_size=8,sampler=sampler,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4)
    configure_controller_loader(loader,cache)
    original_loader=loader
    pool=TemporalReferencePool(db=args.db,model=model,cache=cache,mean=mean,std=std,
        frames=240,exemplar_choices=(1,3,5),bank_size=64,seed=args.seed)
    resident_info=None
    if args.gpu_data_cache_gb>0:
        from Utils.clip_recipe_resident import ResidentTrainingData,ResidentWindowLoader
        resident=ResidentTrainingData(data,cache,mean,std,device=device,budget_gb=args.gpu_data_cache_gb)
        resident_info=resident.info
        if resident.indices is not None:pool.resident=resident
        if resident.features is not None:loader=ResidentWindowLoader(data,sampler,resident)
    optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=5e-4,weight_decay=1e-4,
        fused=device.type=='cuda',foreach=False if device.type=='cuda' else None)
    run=SimpleNamespace(epochs=50,aux_warmup_epochs=0,microbatch=args.microbatch,style_dropout=.1,
        unpaired_probability=.2,style_weight=.1,root_weight=.1,
        prior_weight=.02,entropy_weight=.003,codec_probability=.2,codec_noise=.15)
    mean_t=torch.as_tensor(mean,device=device);std_t=torch.as_tensor(std,device=device)
    if args.compare:
        import copy
        from Utils.reproducibility import rng_state,restore_rng_state
        base_pool=TemporalReferencePool(db=args.db,model=baseline,cache=cache,mean=mean,std=std,
            frames=240,exemplar_choices=(1,3,5),bank_size=64,seed=args.seed)
        base_metric=ReferenceMetricCache(copy.deepcopy(metric_cache.metric),device_cache=False)
        base_optimizer=torch.optim.AdamW([x for x in baseline.parameters() if x.requires_grad],lr=5e-4,
            weight_decay=1e-4,fused=device.type=='cuda',foreach=False if device.type=='cuda' else None)
        variants=dict(baseline=(baseline,original_loader,base_pool,base_metric,base_optimizer),
            optimized=(model,loader,pool,metric_cache,optimizer))
        measurements={name:[] for name in variants};gpu_before=gpu_process_inventory()
        for block in range(5):
            initial=rng_state();sampler.num_samples=(10 if block==0 else 25)*8
            order=('baseline','optimized') if block%2==0 else ('optimized','baseline')
            for name in order:
                restore_rng_state(initial);current,data_loader,refs,metrics,opt=variants[name]
                _seed_loader(data_loader,seed=args.seed,epoch=20+block)
                row=run_epoch(current,data_loader,refs,metrics,opt,args=run,mean=mean_t,std=std_t,epoch=10)
                if block:measurements[name].append(row)
        result=dict(protocol='alternating_real_new_samples_FP32',random_metric=True,measurements=measurements,
            resident_data=resident_info,cuda_field_graph=args.cuda_field_graph,
            gpu_processes_before=gpu_before,gpu_processes_after=gpu_process_inventory(),
            limitations='Alternates each variant on the same case/RNG block; external GPU load can still vary.')
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
        print(json.dumps({name:[r['ms_per_update'] for r in rows] for name,rows in measurements.items()},indent=2))
        return
    _seed_loader(loader,seed=args.seed,epoch=9)
    warm=run_epoch(model,loader,pool,metric_cache,optimizer,args=run,mean=mean_t,std=std_t,epoch=10)
    sampler.num_samples=args.updates*8;_seed_loader(loader,seed=args.seed,epoch=10)
    timed=run_epoch(model,loader,pool,metric_cache,optimizer,args=run,mean=mean_t,std=std_t,epoch=10)
    if args.profile:
        import cProfile,io,pstats
        def instrument(obj,name,label):
            original=getattr(obj,name)
            def wrapped(*a,**kw):
                with torch.profiler.record_function(label):return original(*a,**kw)
            setattr(obj,name,wrapped)
        instrument(pool,'sample_windows','reference_plan_CPU')
        instrument(metric_cache,'embeddings','frozen_reference_Phi')
        instrument(model.field,'forward','flow_field_forward')
        instrument(model.clip_encoder,'forward','reference_Es_forward')
        instrument(metric_cache.metric,'forward','Phi_forward')
        instrument(model.vqvae,'decode','frozen_VQ_decode')
        sampler.num_samples=8*8
        python_profile=cProfile.Profile()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            python_profile.enable()
            run_epoch(model,loader,pool,metric_cache,optimizer,args=run,mean=mean_t,std=std_t,epoch=10)
            python_profile.disable()
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.with_suffix('.profile.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=40))
        profile.export_chrome_trace(str(args.out.with_suffix('.trace.json')))
        summary=io.StringIO();pstats.Stats(python_profile,stream=summary).sort_stats('cumtime').print_stats(40)
        args.out.with_suffix('.python.txt').write_text(summary.getvalue())
    # Fixed validation trajectory, GT only for initialization; explicit same-noise sampling.
    segment=next(s for s in segments if s['split']=='val' and not s.get('mirror') and s['num_frames']>=601)
    source=np.memmap(args.db/'X.f32',mode='r',dtype=np.float32,shape=(cache.meta.num_frames,375))
    x=torch.from_numpy((np.array(source[segment['frame_start']:segment['frame_start']+601])-mean)/std).to(device)
    codes,valid,_,_=pool.sample_windows(torch.tensor([segment['label']]),exclusions=[{segment['source_entry_id']}])
    model.eval();rollouts={}
    with torch.no_grad():
        z,t,mask=reference_latents(model,codes.to(device),valid.to(device))
        probs,_=model.probabilities_from_references(z,t,mask)
        style=model.condition_from_probabilities(probs,validate=False)
        generator=torch.Generator(device=device).manual_seed(args.seed)
        noise=torch.randn((600,1,model.latent_dim),device=device,generator=generator)
        noise=model.latent_offset+model.latent_scale*model.latent_transport_std*noise
        for frames in (60,600):
            current=x[:1];previous=model.frame_latent(current);started=time.perf_counter()
            for step in range(frames):
                latent=model.sample_latent(x_curr=current,style_label=None,style_feature=style,prev_latent=previous,
                    z0=noise[step],num_steps=8)
                motion,previous=model.feedback_state(latent)
                assert torch.isfinite(motion).all() and torch.isfinite(previous).all()
                if step%50==0 or step==frames-1:
                    torch.testing.assert_close(motion,model.vqvae.decode(previous),atol=0,rtol=0)
                current=torch.cat((x[step+1:step+2,:84],motion),-1)
            if device.type=='cuda':torch.cuda.synchronize()
            rollouts[frames]=dict(seconds=time.perf_counter()-started,frames=frames,feedback_matches=True,finite=True)
    assert state_fingerprint(vq.state_dict())==vq_before and all(x.grad is None for x in vq.parameters())
    assert state_fingerprint(metric_cache.metric.state_dict())==metric_before and all(x.grad is None for x in metric_cache.metric.parameters())
    report=dict(protocol='real_train_new_samples_disposable_V2',random_metric=True,style_quality_verified=False,
        backward_mode=args.backward_mode,reference_cache_device=args.reference_cache_device,
        cuda_field_graph=args.cuda_field_graph,
        resident_data=resident_info,
        fp32=True,tf32=False,device=str(device),seed=args.seed,warmup=warm,timed=timed,rollouts=rollouts,
        vq_unchanged=True,metric_unchanged=True,main_training_epochs=0,style_guidance_scale=1.5,
        limitations='Random frozen metric is only a resource/interface fixture; no official weights or accuracy claims.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
