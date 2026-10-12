from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config, FlowMatchVQVAEController
from arch.basic_vqvae_375to291 import BasicVQVAE375to291
from Utils.motion_eval import reconstruct_motion_batch, _predict_flowmatch_step, load_motion_model
from Utils.reproducibility import case_noise_plan, state_fingerprint
from Utils.style_code_prior import TrainStyleRecipePool
from test_controller_repair import controller, loss_args
from test_style_refinement import style_cli_fixture
from _memmap_fixtures import write_db
import training.train_flowmatch_vqvae_controller as trainer
import evaluation.evaluate_minor_revision_generalization as evaluator


def pure_controller(mode='target'):
    config = dict(prediction_mode=mode, conditioning_mode='recipe', flow_loss_space='normalized',
        use_style_code_selector=True, style_slot_indices=[0, 1, 2, 3], style_dim=8,
        style_selector_hidden_dim=16, style_selector_topk=2, style_proto_normalization='none',
        flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32, dropout=0., inference_requantize=True)
    model = build_flowmatch_controller_from_config(config, vqvae=controller().vqvae, num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8))
    model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return model


@pytest.mark.parametrize('mode', ['target', 'velocity'])
def test_pure_fm_gradients_without_embedding_prior_decoder_or_integration(mode, monkeypatch):
    model = pure_controller(mode).train()
    x, _ = loss_args()
    previous, target = model.frame_latent(x[:, 0]), model.frame_latent(x[:, 1])
    recipe = torch.eye(16)[:4]
    before = state_fingerprint(model.vqvae.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure cached FM must not use label conditioning, encoding, Euler or decoder')
    monkeypatch.setattr(model.style_selector.style_embedding, 'forward', forbidden)
    monkeypatch.setattr(model.style_selector.to_logits, 'forward', forbidden)
    monkeypatch.setattr(model, 'frame_latent', forbidden)
    monkeypatch.setattr(model, '_integrate', forbidden)
    monkeypatch.setattr(model.vqvae, 'decode', forbidden)
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=None,
        manual_style_probs=recipe, prev_latent=previous, target_latent=target)
    out['loss'].backward()
    assert out['loss'] == out['flow_loss'] and all(torch.isfinite(v) for v in out.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.style_selector.proto_to_style.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.style_selector.to_logits.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.style_selector.style_embedding.parameters())
    assert state_fingerprint(model.vqvae.state_dict()) == before
    with pytest.raises(ValueError, match='zero prior/entropy'):
        model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=None,
            manual_style_probs=recipe, style_hist_weight=.02)
    with pytest.raises(ValueError, match='Recipe-only'):
        model.encode_conditions(x_curr=x[:, 0], style_label=torch.arange(3), manual_style_probs=recipe)


def pool_fixture(tmp_path):
    segments = []
    for label in range(3):
        for clip in range(6):
            segments.append(dict(label=label, source_entry_id=6*label+clip,
                frame_start=len(segments)*8, num_frames=8, split='train'))
    segments.append(dict(label=0, source_entry_id=100, frame_start=144,
        num_frames=8, split='val'))
    write_db(tmp_path, num_frames=152, num_styles=3, segments=segments)
    model = pure_controller().eval()
    indices = np.empty((152, 4), dtype=np.int64)
    for s in segments:
        indices[s['frame_start']:s['frame_start']+8] = s['source_entry_id'] % 6 + 1 if s['split']=='train' else 100
    return model, segments, SimpleNamespace(indices=indices)


def test_per_sample_exclusion_avoids_full_batch_starvation_and_count_fallback(tmp_path):
    model, segments, cache = pool_fixture(tmp_path)
    pool = TrainStyleRecipePool(db=tmp_path, model=model, cache=cache, mean=np.zeros(375),
        std=np.ones(375), frames=4, exemplar_choices=[1, 3, 5])
    labels = torch.tensor([0, 1, 2])
    counts = set()
    for _ in range(40):
        recipes, fraction = pool.sample(labels, probability=1.,
            per_sample_exclusions=[{0, 6}, {6, 12}, {12, 0}], require_all=True)
        assert fraction == 1 and pool.last_sample_stats['shortfall_fraction'] == 0
        assert recipes[:, :, 1].sum() == 0  # Each supervised clip's unique code is excluded.
        assert torch.isfinite(recipes).all()
        torch.testing.assert_close(recipes.sum(-1), torch.ones(3, 4))
        counts.update((recipes > 0).sum(-1).flatten().tolist())
    assert 1 in counts and 2 in counts  # Single clip vs multi-clip TopK2 conditions.
    with pytest.raises(ValueError, match='no independent'):
        pool.sample(labels, probability=1., excluded_source_ids=range(18), require_all=True)
    fixed = TrainStyleRecipePool(db=tmp_path, model=model, cache=cache, mean=np.zeros(375),
        std=np.ones(375), frames=4, exemplars=5)
    with pytest.raises(ValueError, match='only 4 remain'):
        fixed.sample(labels[:1], probability=1., per_sample_exclusions=[{0, 1}], require_all=True)


def test_new_sampler_cached_and_online_match_for_identical_exemplars(tmp_path):
    model, _, _ = pool_fixture(tmp_path)
    raw = np.fromfile(tmp_path/'X.f32', dtype=np.float32).reshape(-1, 375)
    _, indices, _ = model.vqvae.encode(torch.from_numpy(raw))
    pools = [TrainStyleRecipePool(db=tmp_path, model=model, cache=cache,
        mean=np.zeros(375), std=np.ones(375), frames=4, exemplar_choices=[1, 3, 5], bank_size=8)
        for cache in (SimpleNamespace(indices=indices.numpy()), None)]
    outputs = []
    for pool in pools:
        torch.manual_seed(71)
        outputs.append(pool.sample(torch.arange(3), probability=1.,
            per_sample_exclusions=[{0}, {6}, {12}], require_all=True)[0])
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


def recipe_cli(argv, out):
    return argv(out) + ['--conditioning-mode', 'recipe', '--recipe-exemplar-choices', '1',
        '--recipe-bank-size', '8',
        '--seq-len', '2', '--unroll-steps', '1', '--root-recon-weight', '0']


def test_pure_actual_train_val_and_resume_with_cross_targets(tmp_path, monkeypatch):
    _, vq, _, argv = style_cli_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure TF entry must not decode or integrate')
    monkeypatch.setattr(BasicVQVAE375to291, 'decode', forbidden)
    monkeypatch.setattr(FlowMatchVQVAEController, '_integrate', forbidden)
    original = FlowMatchVQVAEController.encode_conditions
    calls = []
    def check(self, **kwargs):
        assert kwargs['style_label'] is None and kwargs['manual_style_probs'] is not None
        calls.append(self.training)
        return original(self, **kwargs)
    monkeypatch.setattr(FlowMatchVQVAEController, 'encode_conditions', check)
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    monkeypatch.setattr(sys, 'argv', recipe_cli(argv, full)); trainer.main()
    rows = [json.loads(row) for row in (full/'history.jsonl').read_text().splitlines()]
    assert True in calls and False in calls
    assert all(row['train']['recipe_fraction'] == row['val']['recipe_fraction'] == 1 for row in rows)
    assert all(row['train']['recipe_exemplars_mean'] == 1 for row in rows)
    assert all(row['train']['style_hist_loss'] == row['train']['style_entropy'] == 0 for row in rows)
    save = trainer._save_checkpoint
    class Interrupted(Exception): pass
    def stop(**kwargs):
        save(**kwargs)
        if kwargs['path'].name == 'latest.pt' and kwargs['epoch'] == 1:
            raise Interrupted()
    monkeypatch.setattr(trainer, '_save_checkpoint', stop)
    monkeypatch.setattr(sys, 'argv', recipe_cli(argv, resumed))
    with pytest.raises(Interrupted): trainer.main()
    monkeypatch.setattr(trainer, '_save_checkpoint', save)
    monkeypatch.setattr(sys, 'argv', recipe_cli(argv, resumed)+['--resume', str(resumed/'latest.pt')]); trainer.main()
    a, b = [torch.load(path/'latest.pt', weights_only=False) for path in (full, resumed)]
    assert a['training_objective_version'] == b['training_objective_version'] == 11
    assert a['args']['style_hist_weight'] == a['args']['style_entropy_weight'] == 0
    for key, tensor in a['model'].items():
        torch.testing.assert_close(tensor, b['model'][key], rtol=0, atol=0)
    for key, tensor in torch.load(vq, weights_only=False)['model'].items():
        torch.testing.assert_close(a['model']['vqvae.'+key], tensor, rtol=0, atol=0)
    monkeypatch.setattr(FlowMatchVQVAEController, 'encode_conditions', original)
    loaded = load_motion_model(model_name='flowmatch_target', ckpt_path=full/'best.pt', num_styles=3)
    assert loaded.conditioning_mode == 'recipe'
    with pytest.raises(ValueError, match='Recipe-only'):
        loaded.encode_conditions(x_curr=torch.zeros(1,375), style_label=torch.zeros(1,dtype=torch.long))
    monkeypatch.setattr(sys, 'argv', recipe_cli(argv, resumed)+['--resume', str(resumed/'latest.pt'),
        '--recipe-exemplar-choices', '3'])
    with pytest.raises(ValueError, match='recipe_exemplar_choices'): trainer.main()


@pytest.mark.parametrize('requantize', [False, True])
def test_vectorized_teacher_forced_matches_sequential_reference_and_policy(requantize):
    model = pure_controller().eval()
    x, _ = loss_args(); x = x[:2]
    model.latent_data_min.fill_(-.1); model.latent_data_max.fill_(.1)
    model.eval_noise_std = 1.
    probs = torch.eye(16)[:4][None].repeat(2,1,1); probs[1] = probs[1].roll(2,-1)
    noise = case_noise_plan(model,[0,10],[0,1],4,seed=42)
    with torch.no_grad():
        feature = model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=probs)[2]
        expected = x.clone()
        for frame in range(1,4):
            expected[:,frame],_ = _predict_flowmatch_step(model_name='flowmatch_target',model=model,
                x_curr=x[:,frame-1],traj_next=x[:,frame,:84],style_label=None,
                manual_style_probs=probs,manual_style_strength=1.,prev_latent=None,
                solver='euler',ode_steps=2,noise_std=1.,requantize_z=requantize,latent_clip=True,
                z0=noise[:,frame-1],style_feature=feature)
    for micro in (1,3,6):
        model.eval_microbatch_size = micro
        actual = reconstruct_motion_batch(model_name='flowmatch_target',model=model,x=x,
            style_label=None,manual_style_probs=probs,noise_plan=noise,flow_eval_mode='reconstruct',
            flow_ode_steps=2,flow_requantize_z=requantize,flow_latent_clip=True)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_actual_pure_seen_evaluation_uses_val_and_reports_matched_cross(tmp_path, monkeypatch):
    db, _, norm, argv = style_cli_fixture(tmp_path)
    out = tmp_path/'train'
    monkeypatch.setattr(sys,'argv',recipe_cli(argv,out)); trainer.main()
    classifier = tmp_path/'classifier.pth'; classifier.write_bytes(b'fixture')
    names = json.loads((db/'StyleNames.json').read_text())
    metadata = dict(class_names=names,window_size=4,fps=60,joint_set='ybot15',class_num=3,in_feat=45)
    monkeypatch.setattr(evaluator,'build_fid_feature_extractor_from_checkpoint',
        lambda *args,**kwargs:(torch.nn.Identity(),metadata))
    lengths = []
    def stats(classifier,clips,targets,**kwargs):
        lengths.extend(clip.shape[0] for clip in clips)
        return np.asarray(targets),np.ones(len(clips)),np.ones(len(clips))
    monkeypatch.setattr(evaluator,'_classifier_stats',stats)
    evaluated = tmp_path/'eval'
    command = ['evaluate','--db',str(db),'--flow-ckpt',str(out/'best.pt'),
        '--classifier-ckpt',str(classifier),'--source-split','val',
        '--recipe-window','4','--recipe-exemplars-per-style','1','--num-sources','3',
        '--targets-per-source','1','--flow-eval-mode','reconstruct','--flow-ode-steps','2',
        '--num-workers','0','--device','cpu','--out-dir',str(evaluated)]
    monkeypatch.setattr(sys,'argv',command); evaluator.main()
    report = json.loads((evaluated/'minor_revision_generalization_summary.json').read_text())
    assert report['config']['source_split']=='val' and report['config']['recipe_only']
    assert report['config']['recipe_pure_conditioning']
    settings = [row['setting'] for row in report['summaries']]
    assert len(settings)==3 and any('matched_style_recipe_teacher_forced' in s for s in settings)
    assert any('cross_style_transfer_recipe_teacher_forced' in s for s in settings)
    assert lengths and set(lengths)=={4}
    manifest = json.loads((evaluated/'case_manifest.json').read_text())
    assert manifest['source_split']=='val'
    segments = json.loads((db/'segments.json').read_text())
    val_ids = {s['source_entry_id'] for s in segments if s['split']=='val'}
    assert all(row['source_entry_id'] in val_ids for row in manifest['horizons']['4'])
    monkeypatch.setattr(sys,'argv',command+['--source-split','test'])
    with pytest.raises(ValueError,match='source split'): evaluator.main()


def test_recipe_bank_warm_and_cold_have_identical_probs_and_rng(tmp_path):
    model, _, cache = pool_fixture(tmp_path)
    pools = [TrainStyleRecipePool(db=tmp_path,model=model,cache=cache,mean=np.zeros(375),
        std=np.ones(375),frames=4,exemplar_choices=[1,3,5],bank_size=8,seed=42) for _ in range(2)]
    kwargs = dict(probability=1.,per_sample_exclusions=[{0},{6},{12}],require_all=True)
    torch.manual_seed(51)
    expected,_ = pools[0].sample(torch.arange(3),**kwargs)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(51)
    warmed,_ = pools[0].sample(torch.arange(3),**kwargs)
    assert pools[0].last_sample_stats['bank_hit_fraction']==1
    torch.testing.assert_close(expected,warmed,rtol=0,atol=0)
    assert torch.equal(expected_rng,torch.get_rng_state())
    torch.manual_seed(51)
    cold,_ = pools[1].sample(torch.arange(3),**kwargs)
    assert pools[1].last_sample_stats['bank_hit_fraction']==0
    torch.testing.assert_close(expected,cold,rtol=0,atol=0)
    assert torch.equal(expected_rng,torch.get_rng_state())




def test_actual_epoch_excludes_original_source_per_sample(monkeypatch):
    model=pure_controller().train(); x,labels=loss_args(); x=x[:,:2]; x[:,:,84:87]=0
    class Loader(list): pass
    loader=Loader([dict(x=x,label=labels,start=torch.arange(3),segment_id=torch.arange(3))])
    loader.dataset=SimpleNamespace(segments=[dict(source_entry_id=10+i,suffix='FW') for i in range(3)])
    class Pool:
        last_sample_stats=dict(exemplars_mean=1.,shortfall_fraction=0.)
        def sample(self,target_labels,**kwargs):
            assert kwargs['require_all'] and kwargs['probability']==1
            assert kwargs['per_sample_exclusions']==[{10+i} for i in range(3)]
            torch.testing.assert_close(target_labels,labels)
            return torch.eye(16)[:4][None].repeat(3,1,1),1.
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    out=trainer._run_epoch(model=model,loader=loader,device=torch.device('cpu'),prediction_mode='target',
        recon_weight=0.,style_hist_weight=0.,style_entropy_weight=0.,noise_std=1.,
        solver_steps=2,optimizer=optimizer,vq_target_cache=None,cache_vqvae=model.vqvae,
        grad_clip=1.,log_interval=10,epoch=1,total_epochs=1,split='fixture',recipe_pool=Pool())
    assert out['recipe_fraction']==1
