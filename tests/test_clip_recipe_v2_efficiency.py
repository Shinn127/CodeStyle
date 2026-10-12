"""Performance paths preserve inputs/RNG, gradients, and frozen-code decisions."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_clip_recipe_v2 import model_fixture,asset_fixture
from arch.clip_recipe_v2 import MotionStyleMetric
from Utils.clip_recipe_training import ReferenceMetricCache,TemporalReferencePool,window_objective
from Utils.clip_recipe_resident import ResidentTrainingData,ResidentWindowLoader
from Utils.clip_recipe_cuda import graph_training_field
from Utils.vq_target_cache import VQTargetCache
from Utils.reproducibility import rng_state,restore_rng_state,state_fingerprint
from dataset import Motion375Dataset
from training.train_flowmatch_vqvae_controller import configure_controller_loader


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_frozen_quantize_only_preserves_codes_and_state(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,_=model_fixture(device);q=model.vqvae.quantizer
    before=state_fingerprint(q.state_dict());z=torch.randn(128,8,device=device)
    torch.testing.assert_close(q.quantize_only(z),q(z).quantized,atol=0,rtol=0)
    assert state_fingerprint(q.state_dict())==before


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_combined_backward_and_device_cache_preserve_loss_gradients_rng(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    base,_=model_fixture(device);base.train();metric=MotionStyleMetric(hidden=16,layers=1).to(device)
    x=torch.randn(2,240,375,device=device)*.1
    z=base.frame_latent(x.reshape(-1,375)).reshape(2,240,8)
    refs=torch.randint(16,(2,3,240,4),device=device);mask=torch.ones(2,3,dtype=torch.bool,device=device)
    labels=torch.tensor([0,1],device=device);initial=rng_state();outputs=[]
    for mode,cpu_cache in [('split',True),('combined',False)]:
        model=copy.deepcopy(base);model.backward_mode=mode
        cache=ReferenceMetricCache(copy.deepcopy(metric),device_cache=not cpu_cache)
        restore_rng_state(initial)
        out=window_objective(model,x,z,labels,refs,mask,metric_cache=cache,reference_keys=[(0,frozenset(),3,0),(1,frozenset(),3,0)],
            unpaired=torch.tensor([False,True],device=device),epoch=10,microbatch=128,style_dropout=0,aux_fraction=1)
        outputs.append((model,out,rng_state()))
    for key in outputs[0][1]:torch.testing.assert_close(outputs[0][1][key],outputs[1][1][key],atol=1e-6,rtol=1e-6)
    for a,b in zip(outputs[0][0].parameters(),outputs[1][0].parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=3e-4)
    torch.testing.assert_close(outputs[0][2]['torch'],outputs[1][2]['torch'],atol=0,rtol=0)
    for a,b in zip(outputs[0][2]['cuda'],outputs[1][2]['cuda']):torch.testing.assert_close(a,b,atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_resident_inputs_and_reference_crops_are_identical(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    argv=asset_fixture(tmp_path);db=Path(argv[1]);cache=VQTargetCache(Path(argv[7]))
    import json
    segments=json.loads((db/'segments.json').read_text())
    starts=np.concatenate([np.arange(s['frame_start'],s['frame_start']+17) for s in segments if s['split']=='train'])
    mean=np.linspace(-.2,.3,375,dtype=np.float32);std=np.linspace(.7,1.4,375,dtype=np.float32)
    data=Motion375Dataset(db,seq_len=240,sample_starts=starts,normalize=True,norm_mean=mean,norm_std=std)
    original=DataLoader(data,batch_size=8);configure_controller_loader(original,cache)
    expected=next(iter(original))
    resident=ResidentTrainingData(data,cache,mean,std,device=device,budget_gb=.1,reserve_gb=0)
    actual=next(iter(ResidentWindowLoader(data,list(range(8)),resident)))
    for key in ('x','label','start','segment_id','vq_indices'):torch.testing.assert_close(actual[key].cpu(),expected[key],atol=0,rtol=0)
    model,_=model_fixture(device);args=dict(db=db,model=model,cache=cache,mean=mean,std=std,frames=240,exemplar_choices=(1,3,5),bank_size=64)
    pools=[TemporalReferencePool(**args),TemporalReferencePool(**args)];pools[1].resident=resident
    results=[]
    for pool in pools:
        torch.manual_seed(99);results.append((*pool.sample_windows(torch.tensor([0,1]),exclusions=[{0},{8}]),torch.get_rng_state()))
    for i in (0,1,4):torch.testing.assert_close(results[0][i].cpu(),results[1][i].cpu(),atol=0,rtol=0)
    assert results[0][2:4]==results[1][2:4]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('microbatch',[128,64])
def test_field_cuda_graph_matches_eager_gradients_and_parameters(microbatch):
    eager,_=model_fixture('cuda');eager.train();graphed=copy.deepcopy(eager)
    graph_training_field(graphed,microbatch)
    x=torch.randn(2,240,375,device='cuda')*.1
    z=eager.frame_latent(x.reshape(-1,375)).reshape(2,240,8)
    refs=torch.randint(16,(2,1,240,4),device='cuda');mask=torch.ones(2,1,dtype=torch.bool,device='cuda')
    metric=MotionStyleMetric(hidden=16,layers=1).cuda();initial=rng_state();outputs=[]
    for model in (eager,graphed):
        restore_rng_state(initial);model.zero_grad(set_to_none=True)
        out=window_objective(model,x,z,torch.tensor([0,1],device='cuda'),refs,mask,
            metric_cache=ReferenceMetricCache(copy.deepcopy(metric)),reference_keys=['a','b'],
            unpaired=torch.tensor([False,True],device='cuda'),epoch=10,microbatch=microbatch,style_dropout=0,aux_fraction=1.)
        outputs.append(out)
    torch.testing.assert_close(outputs[0]['loss'],outputs[1]['loss'],atol=2e-5,rtol=2e-5)
    for a,b in zip(eager.parameters(),graphed.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=3e-4)


def test_resume_preserves_old_execution_defaults(tmp_path):
    from training.train_clip_recipe_v2 import parse_args
    path=tmp_path/'old.pt';torch.save({'args':{}},path)
    old=parse_args(['--resume',str(path)])
    assert (old.backward_mode,old.reference_cache_device,old.gpu_data_cache_gb,old.cuda_field_graph)==('split','cpu',0.,False)
    current=parse_args([])
    assert (current.backward_mode,current.reference_cache_device,current.gpu_data_cache_gb,current.cuda_field_graph)==('combined','gpu',10.,True)
    torch.save({'args':vars(current)},path)
    resumed=parse_args(['--resume',str(path)])
    for key in ('backward_mode','reference_cache_device','gpu_data_cache_gb','cuda_field_graph'):
        assert getattr(resumed,key)==getattr(current,key)


def test_resident_cache_gracefully_falls_back_after_allocation_oom(tmp_path,monkeypatch):
    argv=asset_fixture(tmp_path);db=Path(argv[1]);cache=VQTargetCache(Path(argv[7]))
    mean=np.zeros(375,np.float32);std=np.ones(375,np.float32)
    data=Motion375Dataset(db,seq_len=240,normalize=True,norm_mean=mean,norm_std=std)
    original=torch.empty
    def fail_features(shape,*args,**kwargs):
        if len(shape)==2 and shape[-1]==375:raise torch.OutOfMemoryError('fixture')
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(torch,'empty',fail_features)
    resident=ResidentTrainingData(data,cache,mean,std,device='cpu',budget_gb=.1)
    assert resident.indices is not None and resident.features is None
    assert resident.info['mode']=='indices' and resident.info['fallback']=='feature_allocation_oom'
