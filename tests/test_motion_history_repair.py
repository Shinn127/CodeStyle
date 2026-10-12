"""Causal history, latent-only feedback, phase-cache and real training entry checks."""
import copy
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from test_controller_repair import controller, loss_args, small_training_fixture, training_argv
from test_clip_recipe_revision import bank_fixture
from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
from Utils.reproducibility import state_fingerprint, case_noise_plan
from Utils.motion_eval import reconstruct_motion_batch
from Utils.controller_validation import motion_quality_selection_key
from Utils.interactive_control import LabelControlSession, InteractiveTrajectory
from Utils.clip_recipe_resident import ResidentTrainingData, ResidentWindowLoader
from Utils.vq_target_cache import VQTargetCache, VQTargetCacheMeta, save_vq_target_cache_meta
from dataset import Motion375Dataset
import training.train_flowmatch_vqvae_controller as trainer


def history_model():
    old = controller()
    config = dict(history_conditioning=True,prediction_mode='target',flow_loss_space='normalized',
        use_style_code_selector=True,style_slot_indices=[0,1,2,3],style_dim=8,
        style_selector_hidden_dim=16,style_selector_topk=2,flow_hidden_dim=16,flow_layers=1,
        flow_ffn_dim=32,dropout=0.,inference_requantize=True)
    new = build_flowmatch_controller_from_config(config,vqvae=copy.deepcopy(old.vqvae),num_styles=3)
    missing, unexpected = new.load_state_dict(old.state_dict(),strict=False)
    assert set(missing)=={'history_delta_proj.weight','history_delta_proj.bias'} and not unexpected
    return old,new


def common_epoch(model,x):
    return dict(model=model,loader=[{'x':x,'label':torch.tensor([0,1,2]),'start':torch.arange(3)}],
        device=torch.device('cpu'),prediction_mode='target',recon_weight=0,style_hist_weight=.02,
        style_entropy_weight=.003,noise_std=1,solver_steps=8,vq_target_cache=None,cache_vqvae=model.vqvae,
        grad_clip=1,log_interval=100,epoch=1,total_epochs=1,split='fixture',unroll_steps=4)


def test_zero_history_projection_preserves_existing_model_exactly_and_trains():
    old,new=history_model();old.eval();new.eval();x,labels=loss_args()
    current=new.frame_latent(x[:,1]);previous=new.frame_latent(x[:,0]);target=new.frame_latent(x[:,2])
    assert new.field.num_tokens==6
    z0=torch.randn_like(current)
    kwargs=dict(x_curr=x[:,1],style_label=labels,prev_latent=current,z0=z0,num_steps=8)
    torch.testing.assert_close(old.sample_latent(**kwargs),new.sample_latent(**kwargs,history_latent=previous),atol=0,rtol=0)
    new.train();before=state_fingerprint(new.vqvae.state_dict())
    def forbidden(*a,**k):raise AssertionError('FM-only must not decode')
    hook=new.vqvae.decoder.register_forward_pre_hook(forbidden)
    out=new.compute_loss(x_curr=x[:,1],x_next=x[:,2],style_label=labels,prev_latent=current,
        target_latent=target,history_latent=previous)
    out['loss'].backward();hook.remove()
    assert new.history_delta_proj.weight.grad.abs().sum()>0
    assert state_fingerprint(new.vqvae.state_dict())==before
    assert all(p.grad is None for p in new.vqvae.parameters())
    with pytest.raises(ValueError,match='shape'):
        new.sample_latent(**kwargs,history_latent=previous[:1])


def test_history_projection_is_cached_once_per_euler_solve():
    _,model=history_model();x,labels=loss_args();calls=[]
    hook=model.history_delta_proj.register_forward_hook(lambda *a:calls.append(1))
    model.sample_latent(x_curr=x[:,0],style_label=labels,num_steps=8)
    hook.remove();assert len(calls)==1


def test_latent_only_feedback_preserves_loss_gradient_rng_and_halves_decode_calls():
    model=controller().train();x,labels=loss_args();latent=model.frame_latent(x.flatten(0,1)).reshape(3,4,-1)
    outputs=[];original=model.compute_loss
    for legacy in (False,True):
        calls=[];hook=model.vqvae.decoder.register_forward_pre_hook(lambda *a:calls.append(1))
        def replay(**kwargs):
            kwargs['return_prediction']=kwargs['return_prediction'] or kwargs.pop('return_rollout_latent')
            return original(**kwargs)
        model.compute_loss=replay if legacy else original
        model.zero_grad();torch.manual_seed(84)
        out=trainer.unrolled_flow_loss(model=model,x_window=x,style_label=labels,prev_latent_0=latent[:,0],
            target_latents=latent[:,1:],recon_targets=x[:,1:,84:],steps=3,solver_steps=8,
            style_hist_weight=.02,style_entropy_weight=.003)
        out['loss'].backward();hook.remove()
        outputs.append((out,{k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None},torch.get_rng_state(),len(calls)))
    model.compute_loss=original
    assert [o[3] for o in outputs]==[2,4]
    for key in outputs[0][0]:torch.testing.assert_close(outputs[0][0][key],outputs[1][0][key],atol=0,rtol=0)
    for key in outputs[0][1]:torch.testing.assert_close(outputs[0][1][key],outputs[1][1][key],atol=0,rtol=0)
    torch.testing.assert_close(outputs[0][2],outputs[1][2],atol=0,rtol=0)


def test_real_epoch_codec_teacher_history_keeps_targets_immutable():
    _,model=history_model();x,_=loss_args();x=x[:,:1].repeat(1,6,1)+torch.randn(3,6,375)*.03
    before=x.clone();lat=model.frame_latent(x.flatten(0,1)).reshape(3,6,-1)
    expected_motion=model.vqvae.decode(lat[:,1:5].flatten(0,1)).reshape(3,4,291)
    captured=[];compute=model.compute_loss
    def capture(**kwargs):
        captured.append({k:kwargs[k].detach().clone() for k in ('x_curr','x_next','history_latent','prev_latent','target_latent')})
        return compute(**kwargs)
    model.compute_loss=capture
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    result=trainer._run_epoch(**common_epoch(model,x),optimizer=optimizer,feedback_probability=0,
        feedback_batch_schedule='batch',train_feedback_context=True)
    assert result['matched_samples']==3 and result['feedback_fraction']==0
    for step,c in enumerate(captured):
        torch.testing.assert_close(c['history_latent'],lat[:,step],atol=0,rtol=0)
        torch.testing.assert_close(c['prev_latent'],lat[:,step+1],atol=0,rtol=0)
        torch.testing.assert_close(c['target_latent'],lat[:,step+2],atol=0,rtol=0)
        torch.testing.assert_close(c['x_curr'][:,84:],expected_motion[:,step],atol=0,rtol=0)
        torch.testing.assert_close(c['x_next'],before[:,step+2],atol=0,rtol=0)
    torch.testing.assert_close(x,before,atol=0,rtol=0)


def test_batch_schedule_skips_euler_on_teacher_batches():
    model=controller();x,_=loss_args();x=x[:,:1].repeat(1,5,1)
    kwargs=common_epoch(model,x);kwargs['loader']*=4
    integrations=[];original=model._integrate
    def count(**kw):integrations.append(kw['num_steps']);return original(**kw)
    model._integrate=count
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    result=trainer._run_epoch(**kwargs,optimizer=optimizer,feedback_probability=.25,feedback_batch_schedule='batch')
    assert integrations==[8,8,8] and result['feedback_fraction']==.25


def test_history_validation_loss_uses_same_codec_observation_and_original_targets():
    _,model=history_model();x,_=loss_args();x=x[:,:1].repeat(1,6,1)+torch.randn(3,6,375)*.03
    q=model.frame_latent(x.flatten(0,1)).reshape(3,6,-1)
    expected=model.vqvae.decode(q[:,1:-1].reshape(-1,model.latent_dim))
    seen=[];compute=model.compute_loss
    def capture(**kw):
        seen.append(kw);return compute(**kw)
    model.compute_loss=capture
    trainer._run_epoch(**common_epoch(model,x),optimizer=None,train_feedback_context=True)
    assert len(seen)==1
    torch.testing.assert_close(seen[0]['x_curr'][:,84:],expected,atol=0,rtol=0)
    torch.testing.assert_close(seen[0]['x_next'],x[:,2:].reshape(-1,375),atol=0,rtol=0)


def test_interactive_and_offline_rollout_use_actual_past_latents():
    _,model=history_model();session=LabelControlSession(model,np.zeros(375,np.float32),np.zeros(375,np.float32),
        np.ones(375,np.float32),['a','b','c'],steps=2)
    captured=[];sample=model.sample_latent
    def capture(**kwargs):captured.append((kwargs['prev_latent'].clone(),kwargs['history_latent'].clone()));return sample(**kwargs)
    model.sample_latent=capture
    first=session.latent.clone();session.step();second=session.latent.clone();session.set_style(1);session.step()
    torch.testing.assert_close(captured[0][1],first,atol=0,rtol=0)
    torch.testing.assert_close(captured[1][0],second,atol=0,rtol=0)
    torch.testing.assert_close(captured[1][1],first,atol=0,rtol=0)
    captured.clear();x,labels=loss_args()
    noise=case_noise_plan(model,[0,10,20],labels.tolist(),4,seed=42)
    reconstruct_motion_batch(model_name='flowmatch_target',model=model,x=x,style_label=labels,
        noise_plan=noise,flow_eval_mode='rollout',flow_ode_steps=2)
    assert len(captured)==3
    torch.testing.assert_close(captured[1][1],captured[0][0],atol=0,rtol=0)
    torch.testing.assert_close(captured[2][1],captured[1][0],atol=0,rtol=0)


def test_quality_selection_requires_contact_and_joint_constraints():
    metrics=dict(matched_accuracy=.90,cross_accuracy=.85,matched_root_ade_cm=4,cross_root_ade_cm=8,
        jitter_cm=.3,cross_jitter_cm=.3,foot_sliding_local_cm=.5,foot_sliding_world_cm=.5,world_contact_count=100)
    assert motion_quality_selection_key(metrics)[0]==0
    assert motion_quality_selection_key(dict(metrics,world_contact_count=0))[0]==1
    assert motion_quality_selection_key(dict(metrics,jitter_cm=1.5))[0]==1


def test_smoothed_command_future_changes_less_on_strafe_switch():
    plans=[]
    for smooth in (False,True):
        p=InteractiveTrajectory(smooth_commands=smooth)
        for _ in range(60):a,_=p.condition(forward=1)
        b,_=p.condition(side=1)
        plans.append(np.linalg.norm(b[:24]-a[:24]))
    assert plans[1]<plans[0]*.5


def test_history_cli_save_resume_is_identical(tmp_path,monkeypatch):
    db,vq,norm=small_training_fixture(tmp_path)
    def argv(out):
        args=training_argv(db,vq,norm,out,4)
        args[args.index('--seq-len')+1]='6'
        return args+['--field-arch','transformer','--history-conditioning','--train-feedback-context','--feedback-batch-schedule','batch']
    full=tmp_path/'full';split=tmp_path/'split'
    monkeypatch.setattr(sys,'argv',argv(full));trainer.main()
    monkeypatch.setattr(sys,'argv',argv(split)+['--stop-after-epoch','1']);trainer.main()
    monkeypatch.setattr(sys,'argv',argv(split)+['--resume',str(split/'latest.pt')]);trainer.main()
    a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
    assert a['training_objective_version']==11 and a['args']['history_conditioning']
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
    assert a['scheduler']==b['scheduler']


def test_frozen_codec_resident_cache_is_train_only_fp32_and_matches_decode(tmp_path):
    raw=bank_fixture(tmp_path);model=controller().eval()
    before=state_fingerprint(model.vqvae.state_dict())
    codes=model.vqvae.encode(torch.from_numpy(raw))[1].cpu().numpy().astype(np.uint16)
    folder=tmp_path/'cache';folder.mkdir();codes.tofile(folder/'indices.u16')
    save_vq_target_cache_meta(folder,VQTargetCacheMeta(db=tmp_path,vqvae_ckpt=tmp_path/'vq',norm_stats=tmp_path/'norm',
        num_frames=40,feature_dim=375,num_codebooks=4,code_dim=2,latent_dim=8,num_embeddings=16,
        indices_dtype='uint16',indices_file='indices.u16',complete=True))
    cache=VQTargetCache(folder)
    data=Motion375Dataset(tmp_path,seq_len=6,sample_starts=np.arange(3),normalize=True,
        norm_mean=np.zeros(375,np.float32),norm_std=np.ones(375,np.float32))
    resident=ResidentTrainingData(data,cache,np.zeros(375,np.float32),np.ones(375,np.float32),
        device='cpu',budget_gb=.01,reserve_gb=0,decoder=model.vqvae)
    loader=ResidentWindowLoader(data,torch.utils.data.SequentialSampler(data),resident,batch_size=3)
    batch=next(iter(loader))
    assert resident.segment_offsets[3]==-1 and resident.teacher_motion.dtype==torch.float32
    expected=model.vqvae.decode(model.vqvae.quantizer.lookup(batch['vq_indices'].flatten(0,1))).reshape(3,6,291)
    torch.testing.assert_close(batch['teacher_motion'],expected,atol=1e-6,rtol=1e-6)
    assert state_fingerprint(model.vqvae.state_dict())==before


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_history_flow_gradient_and_feedback():
    old,new=history_model();old.cuda().eval();new.cuda().eval()
    x,labels=loss_args();x=x.cuda();labels=labels.cuda()
    q=new.frame_latent(x[:,1]);h=new.frame_latent(x[:,0]);z0=torch.randn_like(q)
    args=dict(x_curr=x[:,1],style_label=labels,prev_latent=q,z0=z0,num_steps=8)
    torch.testing.assert_close(old.sample_latent(**args),new.sample_latent(**args,history_latent=h),atol=0,rtol=0)
    new.train();out=new.compute_loss(x_curr=x[:,1],x_next=x[:,2],style_label=labels,history_latent=h,prev_latent=q)
    out['loss'].backward()
    assert new.history_delta_proj.weight.grad.abs().sum()>0 and all(p.grad is None for p in new.vqvae.parameters())
    motion,latent=new.feedback_state(new.sample_latent(**args,history_latent=h))
    torch.testing.assert_close(motion,new.vqvae.decode(latent),atol=0,rtol=0)


