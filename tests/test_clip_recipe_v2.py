"""Unified temporal/manual conditioning, quantized auxiliary gradients, and the real entry."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arch.clip_recipe_v2 import build_clip_recipe_v2,MotionStyleMetric
from test_controller_repair import controller
from Utils.clip_recipe_training import ReferenceMetricCache,window_objective
from Utils.clip_recipe_training import TemporalReferencePool
from Utils.reproducibility import state_fingerprint,artifact_fingerprints,file_sha256
from Utils.style_code_prior import save_style_code_prior,recipe_probabilities_from_indices
from Utils.vq_target_cache import VQTargetCacheMeta,save_vq_target_cache_meta
from _memmap_fixtures import write_db


def model_fixture(device='cpu'):
    config=dict(conditioning_version=2,style_slot_indices=[0,1,2,3],flow_hidden_dim=16,flow_layers=2,
        flow_ffn_dim=32,clip_hidden=16,clip_layers=1,slot_dim=8,global_dim=16)
    model=build_clip_recipe_v2(config,vqvae=controller().vqvae,num_styles=3).to(device)
    model.set_latent_prior_stats(torch.zeros(8,device=device),torch.ones(8,device=device))
    model.set_style_code_prior(torch.ones(3,4,16,device=device)/16)
    return model,config


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_temporal_reference_export_manual_condition_and_cfg(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,_=model_fixture(device);model.eval()
    codes=np.random.default_rng(5).integers(0,16,(480,4))
    probs=recipe_probabilities_from_indices(codes,model,clip_lengths=[240,240])
    points=np.linspace(0,239,60).astype(np.int64)
    selected=torch.from_numpy(codes.reshape(1,2,240,4)[:,:,points]).to(device)
    z=model.vqvae.quantizer.lookup(selected.reshape(-1,4)).reshape(1,2,60,8)
    times=torch.as_tensor(points/60.,device=device,dtype=torch.float32).expand(1,2,60)
    valid=torch.ones_like(times,dtype=torch.bool)
    with torch.no_grad():direct,logits=model.probabilities_from_references(z,times,valid)
    torch.testing.assert_close(direct[0],torch.from_numpy(probs).to(device),atol=0,rtol=0)
    with torch.no_grad():_,reversed_logits=model.probabilities_from_references(z.flip(2),times,valid)
    assert not torch.equal(logits,reversed_logits)
    x=torch.randn(2,375,device=device);manual=torch.from_numpy(probs).to(device)
    style=model.encode_conditions(x_curr=x,style_label=None,manual_style_probs=manual)[2]
    torch.testing.assert_close(style,model.condition_from_probabilities(direct.expand(2,-1,-1)))
    previous=model.frame_latent(x);noise=torch.randn(2,8,device=device)
    model.style_guidance_scale=0
    actual=model.sample_latent(x_curr=x,style_label=None,style_feature=style,prev_latent=previous,z0=noise,num_steps=3)
    model.style_guidance_scale=1
    expected=model.sample_latent(x_curr=x,style_label=None,style_feature=model.null_style[None].expand(2,-1),
        prev_latent=previous,z0=noise,num_steps=3)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    model.style_guidance_scale=1.5
    motion,latent=model.predict_next_motion(x_curr=x,style_label=None,manual_style_probs=manual,
        prev_latent=previous,z0=noise,num_steps=2)
    torch.testing.assert_close(motion,model.vqvae.decode(latent),atol=0,rtol=0)
    torch.testing.assert_close(latent,model.requantize_latent(latent),atol=0,rtol=0)
    with pytest.raises(ValueError,match='unordered'):model.recipe_from_histogram(torch.ones(1,4,16))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_quantized_window_objective_gradients_no_ode_frozen_vq_and_metric(device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,_=model_fixture(device);model.train();window=torch.randn(2,240,375,device=device)*.1
    latents=model.frame_latent(window.reshape(-1,375)).reshape(2,240,8)
    refs=torch.randint(16,(2,1,240,4),device=device);valid=torch.ones(2,1,dtype=torch.bool,device=device)
    metric=MotionStyleMetric(hidden=16,layers=1).to(device);cache=ReferenceMetricCache(metric)
    original=state_fingerprint(model.vqvae.state_dict());metric_before=state_fingerprint(metric.state_dict())
    def forbidden(*a,**k):raise AssertionError('Auxiliary training must never execute Euler')
    monkeypatch.setattr(model,'_integrate',forbidden)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        metrics=window_objective(model,window,latents,torch.tensor([0,1],device=device),refs,valid,
            metric_cache=cache,reference_keys=['a','b'],unpaired=torch.tensor([False,True],device=device),
            epoch=10,microbatch=128,style_dropout=0,aux_fraction=1.)
        assert all(torch.isfinite(v).all() for v in metrics.values())
        optimizer.step()
    assert model.condition_encoder.code[0].weight.grad.abs().sum()>0
    assert model.clip_selector.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.field.parameters())
    assert all(p.grad is None for p in model.vqvae.parameters()) and state_fingerprint(model.vqvae.state_dict())==original
    assert all(p.grad is None for p in metric.parameters()) and state_fingerprint(metric.state_dict())==metric_before
    assert all(p.dtype==torch.float32 for p in model.parameters())


def asset_fixture(tmp_path):
    db=tmp_path/'db';db.mkdir();segments=[]
    for label in range(3):
        for clip in range(8):
            segments.append(dict(label=label,source_entry_id=len(segments),suffix='walk',
                frame_start=len(segments)*256,num_frames=256,split='train' if clip<6 else 'val' if clip==6 else 'test'))
    raw=write_db(db,num_frames=24*256,num_styles=3,segments=segments)*.1;raw.tofile(db/'X.f32')
    vq=controller().vqvae;vq.eval()
    path=tmp_path/'vq.pt'
    torch.save(dict(model=vq.state_dict(),args=dict(hidden_dim=16,decoder_hidden_dim=16,encoder_layers=2,
        decoder_layers=2,num_latents=4,code_dim=2,num_embeddings=16,kmeans_init=False,dead_code_threshold=0)),path)
    norm=tmp_path/'norm.npz';np.savez(norm,mean=np.zeros(375,np.float32),std=np.ones(375,np.float32))
    _,codes,_=vq.encode(torch.from_numpy(raw));folder=tmp_path/'cache';folder.mkdir()
    codes.numpy().astype(np.uint16).tofile(folder/'indices.u16');fp=artifact_fingerprints(db,path,norm)
    sha=file_sha256(folder/'indices.u16')
    save_vq_target_cache_meta(folder,VQTargetCacheMeta(db=db,vqvae_ckpt=path,norm_stats=norm,num_frames=len(raw),
        feature_dim=375,num_codebooks=4,code_dim=2,latent_dim=8,num_embeddings=16,indices_dtype='uint16',
        indices_file='indices.u16',fingerprints=fp,indices_sha256=sha,complete=True))
    prior=tmp_path/'prior.npz';p=np.ones((3,4,16),np.float32)/16
    save_style_code_prior(prior,prior=p,global_prior=p.mean(0),slot_saliency=np.ones(4),selected_slots=np.arange(4),
        style_names=[f'style_{i}' for i in range(3)],fingerprints=fp,cache_sha256=sha,split='train')
    return ['--db',str(db),'--vqvae-ckpt',str(path),'--norm-stats',str(norm),'--vq-target-cache',str(folder),
        '--style-code-prior',str(prior),'--device','cpu']


def test_real_metric_and_controller_entry_exact_resume(tmp_path):
    from training.train_motion_style_metric import main as metric_main
    from training.train_clip_recipe_v2 import main
    common=asset_fixture(tmp_path);metric_dir=tmp_path/'metric'
    metric_main(common+['--out-dir',str(metric_dir),'--updates','2','--val-every','2','--held-styles','1','--hidden','16','--layers','1'])
    def argv(out):return common+['--out-dir',str(out),'--style-metric-ckpt',str(metric_dir/'best.pt'),
        '--epochs','2','--updates-per-epoch','1','--workers','0','--val-every','0','--metric-min-retrieval','0',
        '--flow-hidden-dim','16','--flow-layers','2','--flow-ffn-dim','32','--clip-hidden','16','--clip-layers','1',
        '--slot-dim','8','--global-dim','16','--aux-warmup-epochs','0']
    full=tmp_path/'full';split=tmp_path/'split'
    main(argv(full));main(argv(split)+['--stop-after-epoch','1']);main(argv(split)+['--resume',str(split/'latest.pt')])
    a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
    for name,value in a['model'].items():torch.testing.assert_close(value,b['model'][name],atol=0,rtol=0)
    torch.testing.assert_close(a['rng_state']['torch'],b['rng_state']['torch'],atol=0,rtol=0)
    for key,state in a['optimizer']['state'].items():
        for name,value in state.items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b['optimizer']['state'][key][name],atol=0,rtol=0)
    history=[json.loads(x) for x in (full/'history.jsonl').read_text().splitlines()]
    assert all(row['train']['num_anchors']==512 for row in history)
    assert a['args']['conditioning_version']==2 and a['epoch']==2
    from Utils.motion_eval import load_motion_model
    restored=load_motion_model(model_name='flowmatch_target',ckpt_path=full/'latest.pt',num_styles=3)
    assert restored.conditioning_version==2
    torch.testing.assert_close(restored.clip_selector.weight.cpu(),a['model']['clip_selector.weight'].cpu(),atol=0,rtol=0)
    for left,right in zip(a['rng_state']['cuda'],b['rng_state']['cuda']):torch.testing.assert_close(left,right,atol=0,rtol=0)
    import sys
    from unittest.mock import patch
    import tools.build_zeroshot_style_recipe as builder
    exported=tmp_path/'recipe.json'
    with patch.object(sys,'argv',['builder','--db',common[1],'--flow-ckpt',str(full/'latest.pt'),
        '--segment-index','0','--num-frames','240','--topk','4','--device','cpu','--out',str(exported)]):builder.main()
    payload=json.loads(exported.read_text())
    assert payload['metadata']['recipe_source']=='temporal_clip_selector'
    raw=np.fromfile(Path(common[1])/'X.f32',dtype=np.float32).reshape(-1,375)
    start=payload['metadata']['crop_frame_start']
    indices=restored.vqvae.encode(torch.from_numpy(raw[start:start+240]))[1].numpy()
    expected=recipe_probabilities_from_indices(indices,restored)
    actual=np.zeros_like(expected)
    for row,slot in enumerate(payload['slots']):actual[row,slot['codes']]=slot['weights']
    np.testing.assert_array_equal(actual,expected)


def test_metric_excludes_root_and_root_dependent_velocity():
    metric=MotionStyleMetric(hidden=16,layers=1).eval()
    x=torch.randn(2,64,291);times=torch.linspace(0,4,64).expand(2,-1)
    changed=x.clone();changed[:,:,:3]+=100;changed[:,:,219:]+=100
    torch.testing.assert_close(metric(x,times),metric(changed,times),atol=0,rtol=0)


def test_v2_rollout_validation_uses_temporal_refs_and_excludes_gt_initial_frame(tmp_path,monkeypatch):
    import Utils.controller_validation as validation
    import evaluation.evaluate_minor_revision_generalization as evaluator
    common=asset_fixture(tmp_path);model,_=model_fixture()
    class Classifier(torch.nn.Module):
        def forward(self,x):return x.new_zeros((len(x),8)),x.new_zeros((len(x),3))
    metadata=dict(window_size=240,fps=60,class_names=['style_0','style_1','style_2'],joint_set='ybot15')
    monkeypatch.setattr(validation,'build_fid_feature_extractor_from_checkpoint',lambda *a,**k:(Classifier(),metadata))
    classifier=tmp_path/'classifier.pt';classifier.write_bytes(b'fixture')
    original=evaluator._classifier_stats;seen=[]
    def check(clf,raw,*a,**kw):
        assert raw.shape[1]==240;seen.append(len(raw));return original(clf,raw,*a,**kw)
    monkeypatch.setattr(evaluator,'_classifier_stats',check)
    obj=validation.TeacherForcedStyleValidation(model,db=Path(common[1]),mean=np.zeros(375),std=np.ones(375),
        classifier_path=classifier,clips=3,windows=1,batch_size=3,out_dir=tmp_path)
    result=obj.evaluate(model)
    assert result['matched_cases']==3 and result['cross_cases']==6 and sum(seen)==9
    assert obj.manifest['protocol']=='rollout_60fps_240f' and obj.manifest['conditioning_version']==2
    assert all(s['frames']==240 for profile in obj.manifest['recipes'] for s in profile['clips'])
    diagnostic=obj.condition_diagnostics(model,cases=1)
    assert diagnostic['cases']==1 and diagnostic['recipe_condition_identical']
    assert set(diagnostic['variants'])=={'correct_reference','wrong_reference','null_style'}
    assert diagnostic['targets']!=diagnostic['wrong_targets']


def test_sampler_balances_styles_actions_and_epoch_budget_and_wrapper(tmp_path):
    import os,subprocess,shlex
    from dataset import Motion375Dataset
    from training.train_clip_recipe_v2 import balanced_window_weights,parse_args
    common=asset_fixture(tmp_path)
    data=Motion375Dataset(Path(common[1]),seq_len=240)
    weight=balanced_window_weights(data).numpy()
    for label in range(3):assert np.isclose(weight[data.labels[data.sample_starts]==label].sum(),1.)
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(root/'scripts'/'run_train_clip_recipe_v2.sh')],env={**os.environ,'DRY_RUN':'1'},capture_output=True,text=True,check=True)
    command=shlex.split(result.stdout.splitlines()[1])[3:]
    args=parse_args(command)
    assert args.epochs==50 and args.flow_layers==4 and args.style_guidance_scale==1.5
    assert args.resume is None and args.microbatch==512


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_cuda_metric_controller_exact_resume(tmp_path,monkeypatch):
    import sys
    original=asset_fixture
    monkeypatch.setattr(sys.modules[__name__],'asset_fixture',lambda folder:original(folder)+['--device','cuda'])
    test_real_metric_and_controller_entry_exact_resume(tmp_path)


def test_unpaired_windows_have_zero_fm_supervision(monkeypatch):
    model,_=model_fixture();window=torch.randn(2,240,375)*.1
    latent=model.frame_latent(window.reshape(-1,375)).reshape(2,240,8)
    refs=torch.randint(16,(2,1,240,4));valid=torch.ones(2,1,dtype=torch.bool)
    cache=ReferenceMetricCache(MotionStyleMetric(hidden=16,layers=1))
    result=window_objective(model,window,latent,torch.tensor([0,1]),refs,valid,metric_cache=cache,
        reference_keys=['a','b'],unpaired=torch.ones(2,dtype=torch.bool),epoch=10,
        style_weight=0,root_weight=0,prior_weight=0,entropy_weight=0,style_dropout=0)
    assert result['flow_loss']==0 and result['loss']==0
    assert all(p.grad is None or not p.grad.any() for p in model.field.parameters())
