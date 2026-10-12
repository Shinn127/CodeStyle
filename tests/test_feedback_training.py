from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_controller_repair import small_training_fixture, training_argv, controller, loss_args
import training.train_flowmatch_vqvae_controller as trainer


def test_real_cli_four_step_entry_and_slow_cosine_prefix(tmp_path,monkeypatch):
    db,vq,norm = small_training_fixture(tmp_path)
    out = tmp_path/'recovery'
    argv = training_argv(db,vq,norm,out,2)
    argv[argv.index('--seq-len')+1] = '5'
    argv[argv.index('--unroll-steps')+1] = '4'
    argv += ['--lr-schedule-epochs','50','--train-feedback-context']
    monkeypatch.setattr(sys,'argv',argv)
    trainer.main()
    cp = torch.load(out/'latest.pt',weights_only=False)
    assert cp['scheduler']['T_max'] == 50 and cp['args']['unroll_steps'] == 4
    assert cp['training_objective_version'] == 11  # donor-free objective family
    assert cp['optimizer']['param_groups'][0]['lr'] > cp['args']['lr']*.99
    records = [json.loads(s) for s in (out/'history.jsonl').read_text().splitlines()]
    assert len(records) == 2 and all(r['train']['num_samples']>0 for r in records)
    original = torch.load(vq,weights_only=False)['model']
    for key,tensor in original.items():
        torch.testing.assert_close(cp['model']['vqvae.'+key],tensor,rtol=0,atol=0)


def test_short_schedule_resume_is_numerically_identical(tmp_path,monkeypatch):
    db,vq,norm = small_training_fixture(tmp_path)
    def argv(out):
        return training_argv(db,vq,norm,out,2)+['--lr-schedule-epochs','50','--train-feedback-context']
    full = tmp_path/'full'; resumed = tmp_path/'resumed'
    monkeypatch.setattr(sys,'argv',argv(full)); trainer.main()
    save = trainer._save_checkpoint
    class Interrupted(Exception):
        pass
    def stop(**kwargs):
        save(**kwargs)
        if kwargs['path'].name=='latest.pt' and kwargs['epoch']==1:
            raise Interrupted()
    monkeypatch.setattr(trainer,'_save_checkpoint',stop)
    monkeypatch.setattr(sys,'argv',argv(resumed))
    with pytest.raises(Interrupted):
        trainer.main()
    monkeypatch.setattr(trainer,'_save_checkpoint',save)
    monkeypatch.setattr(sys,'argv',argv(resumed)+['--resume',str(resumed/'latest.pt')]); trainer.main()
    a = torch.load(full/'latest.pt',weights_only=False)
    b = torch.load(resumed/'latest.pt',weights_only=False)
    assert a['scheduler']==b['scheduler']
    for key,tensor in a['model'].items():
        torch.testing.assert_close(tensor,b['model'][key],rtol=0,atol=0)


def test_feedback_context_changes_only_observation_and_keeps_gt_supervision():
    model = controller().train(); x,labels = loss_args()
    original = x.clone()
    expected_latent = model.frame_latent(x[:,0])
    expected_motion = model.vqvae.decode(expected_latent).detach()
    recorded = []
    compute = model.compute_loss
    def capture(**kwargs):
        recorded.append({k:kwargs[k].detach().clone() for k in ['x_curr','x_next','prev_latent','target_latent']})
        return compute(**kwargs)
    model.compute_loss = capture
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    common = dict(model=model,loader=[{'x':x[:,:2],'label':labels,'start':torch.arange(3)}],
        device=torch.device('cpu'),prediction_mode='target',recon_weight=.1,style_hist_weight=.02,
        style_entropy_weight=.003,noise_std=1.,
        solver_steps=2,vq_target_cache=None,
        cache_vqvae=model.vqvae,grad_clip=1.,log_interval=100,epoch=1,total_epochs=1,split='fixture',
        train_feedback_context=True)
    trainer._run_epoch(**common,optimizer=optimizer)
    torch.testing.assert_close(recorded[0]['x_curr'][:,84:],expected_motion,rtol=0,atol=0)
    torch.testing.assert_close(recorded[0]['x_curr'][:,:84],original[:,0,:84],rtol=0,atol=0)
    torch.testing.assert_close(recorded[0]['x_next'],original[:,1],rtol=0,atol=0)
    torch.testing.assert_close(recorded[0]['prev_latent'],expected_latent,rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    assert all(p.grad is None for p in model.vqvae.parameters())
    recorded.clear()
    metrics = trainer._run_epoch(**common,optimizer=None)
    torch.testing.assert_close(recorded[0]['x_curr'],original[:,0],rtol=0,atol=0)
    assert metrics['matched_samples'] == metrics['num_samples'] == 3
    assert metrics['cross_samples'] == 0
    assert metrics['matched_flow_loss'] == pytest.approx(metrics['flow_loss'])
    assert metrics['cross_flow_loss'] == 0
