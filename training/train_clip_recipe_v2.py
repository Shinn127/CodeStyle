"""From-scratch, bounded FP32 temporal clip/recipe training; rollout selection only."""
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
from torch.utils.data import DataLoader,WeightedRandomSampler
from tqdm import tqdm

from arch.clip_recipe_v2 import build_clip_recipe_v2,MotionStyleMetric
from dataset import build_train_val_test_dataloaders
from Utils.clip_recipe_assets import ROOT,add_asset_arguments,load_assets,serializable_args
from Utils.clip_recipe_training import TemporalReferencePool,ReferenceMetricCache,window_objective
from Utils.controller_validation import TeacherForcedStyleValidation,teacher_forced_selection_key
from Utils.reproducibility import (validate_fingerprints,file_sha256,state_fingerprint,
    rng_state,restore_rng_state,isolated_rng)
from training.train_flowmatch_vqvae_controller import (_load_latent_stats,_save_latent_stats,_estimate_raw_latent_stats,
    _build_latent_stats_payload,configure_controller_loader,_seed_loader)

VERSION='clip_recipe_v2_window_fm_quantized_style_2'


def balanced_window_weights(dataset):
    keys=[(s['label'],s['suffix']) for s in dataset.segments]
    unique=sorted(set(keys));lookup={key:i for i,key in enumerate(unique)}
    by_segment=np.array([lookup[key] for key in keys])
    groups=by_segment[dataset.segment_of_start]
    counts=np.bincount(groups,minlength=len(unique))
    actions={label:sum(counts[i]>0 for i,key in enumerate(unique) if key[0]==label) for label,_ in unique}
    factors=np.array([actions[label] for label,_ in unique])
    return torch.from_numpy(1./(counts[groups]*factors[groups]))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_asset_arguments(p)
    p.add_argument('--out-dir',type=Path,default=ROOT/'checkpoints/clip_recipe_v2/controller')
    p.add_argument('--style-metric-ckpt',type=Path,default=ROOT/'checkpoints/clip_recipe_v2/style_metric/best.pt')
    p.add_argument('--metric-min-retrieval',type=float,default=.6)
    p.add_argument('--latent-stats',type=Path,default=None)
    p.add_argument('--resume',type=Path,default=None)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--stop-after-epoch',type=int,default=0)
    p.add_argument('--updates-per-epoch',type=int,default=0,help='0 uses the original seq_len=2/batch512 update budget')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--microbatch',type=int,default=512)
    p.add_argument('--backward-mode',choices=('combined','split'),default=None)
    p.add_argument('--reference-cache-device',choices=('gpu','cpu'),default=None)
    p.add_argument('--gpu-data-cache-gb',type=float,default=None)
    p.add_argument('--cuda-field-graph',action=argparse.BooleanOptionalAction,default=None,
        help='Capture the FP32 field for a single batch512 forward/backward; accumulation uses eager')
    p.add_argument('--flow-hidden-dim',type=int,default=256)
    p.add_argument('--flow-layers',type=int,default=4)
    p.add_argument('--flow-ffn-dim',type=int,default=1024)
    p.add_argument('--clip-hidden',type=int,default=128)
    p.add_argument('--clip-layers',type=int,default=2)
    p.add_argument('--slot-dim',type=int,default=128)
    p.add_argument('--global-dim',type=int,default=256)
    p.add_argument('--style-guidance-scale',type=float,default=1.5)
    p.add_argument('--style-dropout',type=float,default=.1)
    p.add_argument('--unpaired-probability',type=float,default=.2)
    p.add_argument('--style-weight',type=float,default=.1)
    p.add_argument('--aux-warmup-epochs',type=int,default=5)
    p.add_argument('--root-weight',type=float,default=.1)
    p.add_argument('--prior-weight',type=float,default=.02)
    p.add_argument('--entropy-weight',type=float,default=.003)
    p.add_argument('--codec-probability',type=float,default=.2)
    p.add_argument('--codec-noise',type=float,default=.15)
    p.add_argument('--val-every',type=int,default=5)
    p.add_argument('--val-clips',type=int,default=90)
    p.add_argument('--val-windows',type=int,default=3)
    p.add_argument('--val-batch-size',type=int,default=64)
    p.add_argument('--classifier',type=Path,default=ROOT/'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth')
    p.add_argument('--diagnostic-epoch',type=int,default=10)
    p.add_argument('--lr',type=float,default=5e-4)
    args=p.parse_args(argv)
    previous=torch.load(args.resume,map_location='cpu',weights_only=False)['args'] if args.resume else {}
    defaults=dict(backward_mode='combined',reference_cache_device='gpu',gpu_data_cache_gb=10.,cuda_field_graph=True)
    legacy=dict(backward_mode='split',reference_cache_device='cpu',gpu_data_cache_gb=0.,cuda_field_graph=False)
    for key,value in defaults.items():
        if getattr(args,key) is None:setattr(args,key,previous.get(key,legacy[key]) if args.resume else value)
    if (not 1<=args.epochs<=50 or args.val_every<0 or args.workers<0 or args.updates_per_epoch<0
            or args.aux_warmup_epochs<0 or args.diagnostic_epoch<0 or args.stop_after_epoch<0):
        p.error('Training is bounded to 1..50 epochs with nonnegative update/worker/validation settings')
    if args.microbatch not in (128,256,512):p.error('Effective batch512 supports microbatch128/256/512')
    probabilities=(args.style_dropout,args.unpaired_probability,args.codec_probability)
    if not all(np.isfinite(x) and 0<=x<=1 for x in probabilities) or sum(probabilities[1:])>1:
        p.error('Invalid task/dropout probabilities')
    if not all(np.isfinite(x) and x>=0 for x in (args.style_weight,args.root_weight,args.prior_weight,args.entropy_weight,args.codec_noise)):
        p.error('Loss/noise weights must be finite and nonnegative')
    if not np.isfinite(args.metric_min_retrieval) or not 0<=args.metric_min_retrieval<=1:p.error('Invalid metric retrieval gate')
    if not np.isfinite(args.gpu_data_cache_gb) or args.gpu_data_cache_gb<0:p.error('Invalid immutable data cache budget')
    if args.cuda_field_graph and args.backward_mode!='combined':p.error('Field CUDA graphs require combined backward')
    return args


def initialize(args):
    device,vq,cache,mean,std,fp,prior,segments,names=load_assets(args)
    config=serializable_args(args)
    config.update(conditioning_version=2,conditioning_mode='clip_recipe',prediction_mode='target',
        flow_loss_space='normalized',style_slot_indices=prior['selected_slots'].tolist(),num_styles=len(names),
        style_selector_topk=4,style_selector_temperature=1.,reference_frames=240,reference_tokens=60,
        inference_requantize=True,inference_latent_clip=True,fp32_math_attention=True,
        fp32=True,tf32=False,matmul_precision='highest',
        solver_steps=8,noise_std=1.,fingerprints=fp,format=VERSION,
        source_window_frames=240,excluded_short_train_segments=sum(s['split']=='train' and s['num_frames']<240 for s in segments),
        style_metric_sha256=file_sha256(args.style_metric_ckpt),prior_sha256=file_sha256(args.style_code_prior))
    model=build_clip_recipe_v2(config,vqvae=vq,num_styles=len(names)).to(device)
    model.backward_mode=args.backward_mode
    path=args.latent_stats or args.out_dir/'latent_stats.npz'
    if path.exists():
        stats=_load_latent_stats(path,latent_dim=model.latent_dim)
        validate_fingerprints(stats['fingerprints'],fp)
        if stats.get('split')!='train':raise ValueError('Latent statistics must be train-only')
    else:
        values=_estimate_raw_latent_stats(db_path=args.db,vqvae=vq,device=device,norm_mean=mean,norm_std=std,
            vq_target_cache=cache,chunk_rows=8192)
        stats=_build_latent_stats_payload(raw_mean=values[0],raw_std=values[1],raw_min=values[2],raw_max=values[3])
        stats.update(fingerprints=fp,split='train');_save_latent_stats(path,stats)
    model.set_latent_prior_stats(stats['offset'],stats['scale'],transport_std=stats['transport_std'],
        data_min=stats['data_min'],data_max=stats['data_max'])
    config['latent_statistics_sha256']=state_fingerprint({k:torch.as_tensor(stats[k],dtype=torch.float32)
        for k in ('offset','scale','transport_std','data_min','data_max')})
    model.set_style_code_prior(torch.as_tensor(prior['prior'],device=device))
    metric_state=torch.load(args.style_metric_ckpt,map_location='cpu',weights_only=False)
    if metric_state.get('format')!='motion_style_metric_v1':raise ValueError('Expected the independent train-only style metric')
    validate_fingerprints(metric_state['fingerprints'],fp)
    if metric_state['validation']['held_style_retrieval']<args.metric_min_retrieval:
        raise ValueError('Style metric failed held-style retrieval gate; inspect its report before controller training')
    metric=MotionStyleMetric(**metric_state['architecture']).to(device)
    metric.load_state_dict(metric_state['model']);metric.eval().requires_grad_(False)
    return config,model,ReferenceMetricCache(metric,device_cache=args.reference_cache_device=='gpu'),cache,mean,std,segments,device


def run_epoch(model,loader,pool,metric_cache,optimizer,*,args,mean,std,epoch):
    model.train();device=model.latent_offset.device;sums={};updates=0;started=time.perf_counter()
    ramp=min(max((epoch-args.aux_warmup_epochs)/5,0.),1.) if args.aux_warmup_epochs else 1.
    for batch in tqdm(loader,desc=f'V2 {epoch}/{args.epochs}',unit='update',dynamic_ncols=True):
        window=batch['x'].to(device,non_blocking=True)
        labels=batch['label'].to(device,non_blocking=True)
        source_labels=labels.clone();segments=batch['segment_id']
        codes=batch['vq_indices'].to(device,non_blocking=True)
        with torch.no_grad():
            latents=model.vqvae.quantizer.lookup(codes.reshape(-1,codes.shape[-1])).reshape(8,240,-1)
            draw=torch.rand(8,device=device)
            unpaired=draw<args.unpaired_probability*ramp
            target=torch.randint(model.num_styles-1,(8,),device=device)
            target+=target>=source_labels
            labels=torch.where(unpaired,target,labels)
        source_entries=[loader.dataset.segments[int(i)]['source_entry_id'] for i in segments.tolist()]
        refs,valid,records,keys=pool.sample_windows(labels.cpu(),exclusions=[{s} for s in source_entries])
        refs=refs.to(device,non_blocking=True);valid=valid.to(device,non_blocking=True)
        state=rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            micro=getattr(args,'active_microbatch',args.microbatch)
            try:
                metrics=window_objective(model,window,latents,labels,refs,valid,
                    metric_cache=metric_cache,reference_keys=keys,unpaired=unpaired,epoch=epoch,microbatch=micro,
                    style_weight=args.style_weight*ramp,root_weight=args.root_weight,
                    prior_weight=args.prior_weight,entropy_weight=args.entropy_weight,style_dropout=args.style_dropout,
                    codec_probability=args.codec_probability,codec_noise=args.codec_noise)
                break
            except torch.OutOfMemoryError:
                if micro<=128:raise
                args.active_microbatch=micro//2
                restore_rng_state(state);torch.cuda.empty_cache()
                print(f'Retry the same effective512 batch with microbatch={args.active_microbatch}',flush=True)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();updates+=1
        metrics.update(paired_fraction=(~unpaired).float().mean())
        for k,v in metrics.items():sums[k]=sums.get(k,window.new_zeros(()))+v.detach()
        if not torch.isfinite(metrics['loss']):raise ValueError('Nonfinite training loss')
    if not updates:raise ValueError('No training updates')
    if device.type=='cuda':torch.cuda.synchronize()
    result={k:float(v.cpu()/updates) for k,v in sums.items()}
    result.update(updates=updates,effective_batch=512,independent_windows=8,ms_per_update=(time.perf_counter()-started)*1000/updates,
        active_microbatch=getattr(args,'active_microbatch',args.microbatch),
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0.)
    return result


def main(argv=None):
    args=parse_args(argv)
    if (args.out_dir/'latest.pt').exists() and args.resume is None:raise ValueError('Run already exists; use its same-version resume or a new directory')
    config,model,metric_cache,cache,mean,std,segments,device=initialize(args)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    bundle=build_train_val_test_dataloaders(db_path=args.db,seq_len=240,stride=1,batch_size=8,num_workers=0,
        norm_mean=mean,norm_std=std,normalize=True,drop_last=True,pin_memory=device.type=='cuda')
    dataset=bundle.train_dataset
    # A window is the sampling unit; an epoch keeps the old 512-transition update budget.
    budget=args.updates_per_epoch or sum(max(s['num_frames']-1,0) for s in segments if s['split']=='train')//512
    if budget<1:raise ValueError('Not enough train transitions for batch512')
    sampler=WeightedRandomSampler(balanced_window_weights(dataset),num_samples=budget*8,replacement=True)
    loader=DataLoader(dataset,batch_size=8,sampler=sampler,num_workers=args.workers,
        pin_memory=device.type=='cuda',persistent_workers=args.workers>0,
        **({'prefetch_factor':4} if args.workers else {}))
    configure_controller_loader(loader,cache)
    pool=TemporalReferencePool(db=args.db,model=model,cache=cache,mean=mean,std=std,frames=240,
        exemplar_choices=(1,3,5),bank_size=64,seed=args.seed)
    if device.type=='cuda' and args.gpu_data_cache_gb>0:
        from Utils.clip_recipe_resident import ResidentTrainingData,ResidentWindowLoader
        resident=ResidentTrainingData(dataset,cache,mean,std,device=device,budget_gb=args.gpu_data_cache_gb)
        if resident.indices is not None:pool.resident=resident
        if resident.features is not None:loader=ResidentWindowLoader(dataset,sampler,resident)
        print('[GPU immutable data cache]',resident.info,flush=True)
    mean_t=torch.as_tensor(mean,device=device);std_t=torch.as_tensor(std,device=device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4,
        fused=device.type=='cuda',foreach=False if device.type=='cuda' else None)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5)
    start=1;best_key=None;best_selection=None;best_loss=float('inf')
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('format')!=VERSION:raise ValueError('Only the same V2 training state can resume')
        # Path moves are allowed only because all corresponding content hashes below
        # are checked. A changed file at the same path is still rejected.
        ignored={'resume','stop_after_epoch','out_dir','workers','db','vqvae_ckpt','norm_stats',
                 'vq_target_cache','style_code_prior','style_metric_ckpt','latent_stats'}
        legacy=dict(backward_mode='split',reference_cache_device='cpu',gpu_data_cache_gb=0.,cuda_field_graph=False)
        for k,v in config.items():
            if k not in ignored and saved['args'].get(k,legacy.get(k))!=v:raise ValueError(f'Resume configuration differs: {k}')
        bundled={k[6:]:v for k,v in saved['model'].items() if k.startswith('vqvae.')}
        if state_fingerprint(bundled)!=config['fingerprints']['vq_state_sha256']:
            raise ValueError('Resume checkpoint embedded VQ differs from the fixed VQ')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['epoch']+1;best_key=saved['best_key'];best_selection=saved['selection'];best_loss=saved['best_loss']
        args.active_microbatch=saved.get('active_microbatch',args.microbatch)
        restore_rng_state(saved['rng_state'])
    if args.cuda_field_graph and device.type=='cuda':
        if getattr(args,'active_microbatch',args.microbatch)==512:
            from Utils.clip_recipe_cuda import graph_training_field
            with isolated_rng(args.seed):graph_training_field(model,512)
        else:print('[CUDA field graph] eager field for gradient accumulation',flush=True)
    (args.out_dir/'config.json').write_text(json.dumps(config,indent=2))
    frozen=state_fingerprint(model.vqvae.state_dict());metric_frozen=state_fingerprint(metric_cache.metric.state_dict())
    validator=None
    if args.val_every:
        with isolated_rng(args.seed):
            validator=TeacherForcedStyleValidation(model,db=args.db,mean=mean,std=std,classifier_path=args.classifier,
                seed=args.seed,clips=args.val_clips,windows=args.val_windows,batch_size=args.val_batch_size,out_dir=args.out_dir,
                eval_mode='rollout')
    for epoch in range(start,args.epochs+1):
        _seed_loader(loader,seed=args.seed,epoch=epoch)
        metrics=run_epoch(model,loader,pool,metric_cache,optimizer,args=args,mean=mean_t,std=std_t,epoch=epoch)
        scheduler.step();validation=None;diagnostics=None;is_best=False
        if validator is not None and (epoch%args.val_every==0 or epoch==args.epochs or epoch==args.diagnostic_epoch):
            with isolated_rng(args.seed):validation=validator.evaluate(model)
            if epoch in (5,args.diagnostic_epoch):
                with isolated_rng(args.seed):diagnostics=validator.condition_diagnostics(model)
            key=teacher_forced_selection_key(validation)
            if best_key is None or key<tuple(best_key):
                best_key=key;is_best=True
                best_selection=dict(epoch=epoch,metrics=validation,selection_key=key,constraints_passed=key[0]==0,
                    seen_targets_passed=validation['matched_accuracy']>=.85 and validation['cross_accuracy']>=.8,
                    protocol='rollout_60fps_240f',style_guidance_scale=args.style_guidance_scale)
        is_loss=metrics['loss']<best_loss
        if is_loss:best_loss=metrics['loss']
        assert state_fingerprint(model.vqvae.state_dict())==frozen and all(p.grad is None for p in model.vqvae.parameters())
        assert state_fingerprint(metric_cache.metric.state_dict())==metric_frozen and all(p.grad is None for p in metric_cache.metric.parameters())
        saved=dict(format=VERSION,training_objective_version=VERSION,implementation_version=2,epoch=epoch,args=config,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng_state=rng_state(),
            best_key=best_key,selection=best_selection,best_loss=best_loss,training_metrics=metrics,validation=validation,
            condition_diagnostics=diagnostics)
        saved['active_microbatch']=getattr(args,'active_microbatch',args.microbatch)
        if is_best:torch.save(saved,args.out_dir/'best.pt')
        if is_loss:torch.save(saved,args.out_dir/'best_loss.pt')
        torch.save(saved,args.out_dir/'latest.pt')
        row=dict(epoch=epoch,train=metrics,validation=validation,condition_diagnostics=diagnostics,selection=best_selection)
        with (args.out_dir/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:break
        if epoch==args.diagnostic_epoch and validation is not None:
            passed=(validation['matched_accuracy']>=.75 and validation['cross_accuracy']>=.55
                and validation['matched_root_ade_cm']<=15 and validation['cross_root_ade_cm']<=20)
            if not passed:
                (args.out_dir/'stopped_diagnostic.json').write_text(json.dumps(row,indent=2))
                print('Stopped at the predefined diagnostic gate; inspect rollout before further training.',flush=True)
                break


if __name__=='__main__':main()
