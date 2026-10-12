"""At most 3 x 100 disposable updates: CPU loader, cold cache, repeated warm cache.

Runs the real training entry point but exits before validation/checkpoint saving.
Warm mode deliberately repeats the same source batches; it is a cache ceiling,
not a measured full-epoch speedup. All variants reset weights/optimizer/RNG.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
import training.train_flowmatch_vqvae_controller as train
from Utils.reproducibility import rng_state,restore_rng_state,state_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--report',type=Path,default=Path(__file__).with_name('performance.json'))
    args=parser.parse_args()
    if not 1<=args.updates<=100:parser.error('Updates must be in [1,100]')
    if not torch.cuda.is_available():parser.error('CUDA is required')
    env=dict(os.environ,DRY_RUN='1',OUT=tempfile.mkdtemp(prefix='newcode_history_benchmark_'),DEVICE='cuda',
        FEEDBACK_START_EPOCH='1',FEEDBACK_RAMP_EPOCHS='0',FEEDBACK_PROBABILITY='.25')
    command=subprocess.check_output(['bash',str(ROOT/'scripts'/'run_train_flowmatch_history_phase10.sh'),
        '--epochs','1','--max-train-batches',str(args.updates),'--max-val-batches','1',
        '--tf-val-every','0','--no-motion-quality-selection'],env=env,text=True)
    tokens=shlex.split(next(line for line in command.splitlines() if line.startswith('[Run] '))[6:])
    sys.argv=tokens[1:]
    original=train._run_epoch;results=[];started=time.perf_counter()
    class Complete(Exception):pass
    def benchmark(**kwargs):
        if kwargs['optimizer'] is None:raise AssertionError('This benchmark must never evaluate')
        setup_sec=time.perf_counter()-started
        model,optimizer,fast=(kwargs[k] for k in ('model','optimizer','loader'))
        before=state_fingerprint(model.vqvae.state_dict())
        initial=copy.deepcopy(model.state_dict());opt=copy.deepcopy(optimizer.state_dict());random=rng_state()
        cpu=torch.utils.data.DataLoader(fast.dataset,batch_size=fast.batch_size,sampler=fast.sampler,
            num_workers=8,pin_memory=True,drop_last=True,persistent_workers=True,prefetch_factor=4)
        train.configure_controller_loader(cpu,kwargs['vq_target_cache'])
        cache_info=copy.deepcopy(fast.resident.info)
        for name,loader in (('cpu_online',cpu),('gpu_cache_cold',fast),('gpu_cache_warm_repeat',fast)):
            model.load_state_dict(initial);optimizer.load_state_dict(copy.deepcopy(opt));restore_rng_state(random)
            train._seed_loader(loader,seed=42,epoch=1)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
            metrics=original(**dict(kwargs,loader=loader))
            torch.cuda.synchronize();elapsed=time.perf_counter()-t0
            if state_fingerprint(model.vqvae.state_dict())!=before:raise AssertionError('VQ changed')
            row=dict(mode=name,updates=args.updates,seconds=elapsed,updates_per_sec=args.updates/elapsed,
                cuda_peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                cache_ready_fraction=0.0,metrics=metrics,
                model_sha256=state_fingerprint(model.state_dict()),rng_sha256=state_fingerprint({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}))
            results.append(row);print('[Benchmark]',json.dumps(row),flush=True)
        report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,fp32=True,tf32=False,
            seed=42,batch_size=fast.batch_size,unroll=4,feedback_probability=.25,feedback_euler_steps=8,
            setup_seconds=setup_sec,cache=cache_info,
            modes=results,warm_speedup=results[0]['seconds']/results[-1]['seconds'],
            cold_warm_weights_equal=results[1]['model_sha256']==results[2]['model_sha256'],
            cold_warm_rng_equal=results[1]['rng_sha256']==results[2]['rng_sha256'],
            caveats=['Repeated warm batches are a cache ceiling, not full-epoch timing.',
                     'Peak memory includes immutable GPU caches held for all comparison modes.',
                     'FP32 decoder batching and CUDA backward reductions can change last-bit rounding; model hashes need not match.'])
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
        raise Complete()
    train._run_epoch=benchmark
    try:train.main()
    except Complete:pass
    finally:train._run_epoch=original


if __name__=='__main__':main()
