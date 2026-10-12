"""Real-data FP32 benchmark: warm-prefix replay by default; --streaming times new samples."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train_flowmatch_vqvae_controller as trainer
from dataset import build_train_val_test_dataloaders
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import configure_torch_runtime,load_norm_stats,set_seed
from Utils.style_code_prior import TrainStyleRecipePool
from Utils.style_donor_bank import StyleDonorBank
from Utils.vq_target_cache import VQTargetCache
from Utils.reproducibility import rng_state,restore_rng_state,state_fingerprint


def snapshot(name,filename):
    path=ROOT/'analysis/training_speed_2026-10-09/before'/filename
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--warmup',type=int,default=100)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--reference',action='store_true',help='Use the saved 2026-10-09 pre-optimization implementation')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--streaming',action='store_true',help='Time new shuffled samples instead of replaying the warm-cache prefix')
    parser.add_argument('--worker-batches',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--config-checkpoint',type=Path,default=ROOT/'checkpoints/clip_recipe_gt_history50/latest.pt')
    args=parser.parse_args()
    if min(args.warmup,args.updates)<1 or (not args.streaming and args.warmup<args.updates):
        parser.error('updates/warmup must be positive; warm-cache replay also requires warmup >= updates')
    if not torch.cuda.is_available():parser.error('CUDA is not visible; run in the host GPU environment')
    if Path(args.name).name!=args.name:parser.error('name must be a single filename component')
    folder=ROOT/'analysis/performance_runs';folder.mkdir(exist_ok=True)
    output=folder/(args.name+'.json')
    if output.exists():parser.error(f'Result already exists: {output}; choose a new name')
    train=trainer;Pool=TrainStyleRecipePool;Bank=StyleDonorBank
    if args.reference:
        args.worker_batches=False  # The snapshot predates worker tensor batches.
        train=snapshot('reference_trainer','train_flowmatch_vqvae_controller.py')
        train.build_flowmatch_controller_from_config=snapshot('arch.reference_controller','flowmatch_vqvae_controller.py').build_flowmatch_controller_from_config
        Pool=snapshot('reference_pool','style_code_prior.py').TrainStyleRecipePool
        Bank=snapshot('reference_bank','style_donor_bank.py').StyleDonorBank
    checkpoint=torch.load(args.config_checkpoint,map_location='cpu',weights_only=False)
    config=dict(checkpoint['args']);device=torch.device('cuda')
    configure_torch_runtime(device=device,tf32=False,cudnn_benchmark=False,matmul_precision='highest')
    set_seed(42);mean,std=load_norm_stats(config['norm_stats'])
    config['fp32_math_attention']=not args.reference
    vq=build_vqvae_from_checkpoint(config['vqvae_ckpt']).to(device)
    model=train.build_flowmatch_controller_from_config(config,vqvae=vq,num_styles=config['num_styles']).to(device)
    saved=checkpoint['model']
    model.set_latent_prior_stats(saved['latent_offset'],saved['latent_scale'],
        transport_std=saved['latent_transport_std'],data_min=saved['latent_data_min'],data_max=saved['latent_data_max'])
    model.set_style_code_prior(saved['style_prior_probs'])
    del checkpoint
    frozen=state_fingerprint(vq.state_dict())
    def unused(*a,**kw):raise AssertionError('Latent-only smoke must not decode or integrate')
    model.vqvae.decode=unused;model._integrate=unused
    model.vqvae.decoder.register_forward_pre_hook(lambda *a: unused())
    cache=VQTargetCache(config['vq_target_cache'])
    bundle=build_train_val_test_dataloaders(db_path=config['db'],seq_len=2,stride=1,
        enforce_same_label=True,batch_size=512,num_workers=8,pin_memory=True,drop_last=True,
        normalize=True,norm_mean=mean,norm_std=std,persistent_workers=True,prefetch_factor=4)
    if args.worker_batches:
        train.configure_controller_loader(bundle.train_loader,cache)
    else:
        bundle.train_loader.collate_fn=train.controller_collate
    pool=Pool(db=config['db'],model=model,cache=cache,mean=mean,std=std,frames=240,
        exemplars=5,exemplar_choices=(1,3,5),bank_size=64,seed=42)
    bank=Bank(config['db'],window=16,windows_per_clip=16,device=device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,
        weight_decay=1e-4,fused=True,foreach=False)
    kwargs=dict(model=model,loader=bundle.train_loader,device=device,prediction_mode='target',
        recon_weight=0.,style_hist_weight=.02,style_entropy_weight=.003,noise_std=1.,
        prev_latent_noise_std=.3,prev_motion_noise_std=.2,prev_latent_dropout=0.,prev_motion_dropout=0.,
        solver_steps=8,optimizer=optimizer,vq_target_cache=cache,cache_vqvae=vq,grad_clip=1.,
        log_interval=1000,epoch=5,total_epochs=50,split='smoke',decouple_prob=0.,
        norm_mean=mean,norm_std=std,microbatch_size=512,recipe_pool=pool,recipe_probability=1.,
        history_noise_mode='structured',history_noise_probability=.3,history_noise_start_epoch=51,
        donor_bank=bank,history_style_probability=.3,history_mask_probability=.2,
        history_condition_warmup_epochs=5,flow_low_t_probability=.5,flow_low_t_max=.2)
    initial=rng_state()
    train._seed_loader(bundle.train_loader,seed=42,epoch=5)
    warm_started=time.perf_counter()
    warm_metrics=train._run_epoch(**kwargs,max_batches=args.warmup);torch.cuda.synchronize()
    warm_ms=(time.perf_counter()-warm_started)*1000/args.warmup
    if not args.streaming:
        restore_rng_state(initial);train._seed_loader(bundle.train_loader,seed=42,epoch=5)
    started=time.perf_counter()
    metrics=train._run_epoch(**kwargs,max_batches=args.updates);torch.cuda.synchronize()
    result=dict(device=torch.cuda.get_device_name(),reference=args.reference,fp32=True,tf32=False,
        warmup_updates=args.warmup,timed_updates=args.updates,ms_per_update=(time.perf_counter()-started)*1000/args.updates,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,metrics=metrics)
    result['worker_batches']=args.worker_batches
    result['streaming']=args.streaming
    result['warmup_ms_per_update']=warm_ms
    result['warmup_recipe_bank_hit_fraction']=warm_metrics['recipe_bank_hit_fraction']
    assert state_fingerprint(vq.state_dict())==frozen and all(p.grad is None for p in vq.parameters())
    assert metrics['gt_target_fraction']==1. and metrics['recipe_fraction']==1.
    if args.profile:
        import cProfile
        import io
        import pstats
        python_profile=cProfile.Profile()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            python_profile.enable()
            train._run_epoch(**kwargs,max_batches=8)
            python_profile.disable()
        (folder/(args.name+'_profile.txt')).write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=30))
        summary=io.StringIO();pstats.Stats(python_profile,stream=summary).sort_stats('cumtime').print_stats(35)
        (folder/(args.name+'_python_profile.txt')).write_text(summary.getvalue())
    result['frozen_vq_unchanged']=True
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))


if __name__=='__main__':main()
