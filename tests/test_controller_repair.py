from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arch.basic_vqvae_375to291 import BasicVQVAE375to291
from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
from training.train_flowmatch_vqvae_controller import _run_epoch, unrolled_flow_loss, _save_checkpoint
from Utils.motion_eval import reconstruct_motion_batch, compute_sequence_metric_sums, MotionEvalConfig
from Utils.reproducibility import state_fingerprint, isolated_rng, rng_state, restore_rng_state, case_noise_plan, cluster_interval, validate_fingerprints
from Utils.controller_validation import classifier_label_mapping, selection_key
from Utils.style_code_prior import validate_recipe_metadata
from Utils.vq_target_cache import VQTargetCache, VQTargetCacheMeta, save_vq_target_cache_meta
from _memmap_fixtures import write_db
import training.train_flowmatch_vqvae_controller as trainer


def controller(mode='target'):
    torch.manual_seed(31)
    vq = BasicVQVAE375to291(hidden_dim=16, decoder_hidden_dim=16, encoder_layers=2,
        decoder_layers=2, num_latents=4, code_dim=2, num_embeddings=16,
        kmeans_init=False, dead_code_threshold=0)
    model = build_flowmatch_controller_from_config(dict(prediction_mode=mode,
        flow_loss_space='normalized', use_style_code_selector=True,
        style_slot_indices=[0, 1, 2, 3], style_dim=8, style_selector_hidden_dim=16,
        style_selector_topk=2, flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32,
        dropout=0., inference_requantize=True), vqvae=vq, num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return model


def loss_args():
    x = torch.randn(3, 4, 375) * .1
    return x, torch.tensor([0, 1, 2])


def test_reconstruction_only_reaches_flow_and_selector_but_vq_stays_frozen():
    model = controller().train()
    original = state_fingerprint(model.vqvae.state_dict())
    x, labels = loss_args()
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
        flow_weight=0., recon_weight=.1, solver_steps=2)
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.style_selector.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.vqvae.parameters())
    AdamW([p for p in model.parameters() if p.requires_grad], lr=.001).step()
    assert state_fingerprint(model.vqvae.state_dict()) == original


def test_reconstruction_weight_changes_gradient():
    model = controller()
    x, labels = loss_args()
    gradients = []
    for weight in (0., .1):
        model.zero_grad()
        torch.manual_seed(8)
        out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
            recon_weight=weight, solver_steps=2)
        out['loss'].backward()
        gradients.append(torch.cat([p.grad.flatten() for p in model.field.parameters()]))
    assert not torch.equal(*gradients)


@pytest.mark.parametrize('mode', ['target', 'velocity'])
@pytest.mark.parametrize('steps', [1, 2, 3])
def test_real_epoch_entry_and_validation_cover_all_metrics(mode, steps):
    model = controller(mode)
    x, labels = loss_args()
    batches = [{'x': x[:, :steps + 1], 'label': labels, 'start': torch.tensor([0, 10, 20])}]
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    vq_before = state_fingerprint(model.vqvae.state_dict())
    common = dict(model=model, loader=batches, device=torch.device('cpu'), prediction_mode=mode,
        recon_weight=.1, style_hist_weight=.02, style_entropy_weight=.02, noise_std=1.,
        solver_steps=2, vq_target_cache=None, cache_vqvae=model.vqvae,
        grad_clip=1., log_interval=10, epoch=1, total_epochs=1, split='fixture',
        unroll_steps=steps, microbatch_size=2)
    out = _run_epoch(**common, optimizer=optimizer)
    assert all(np.isfinite(value) for value in out.values())
    assert out['num_samples'] == 3
    _run_epoch(**common, optimizer=None)
    assert state_fingerprint(model.vqvae.state_dict()) == vq_before


def test_steps_one_matches_all_losses_and_regularizer_does_not_shrink():
    model = controller().eval()
    x, labels = loss_args()
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(3, 4, 8)
    kwargs = dict(recon_weight=.1, root_recon_weight=.1, style_hist_weight=.02,
                  style_entropy_weight=.02, solver_steps=2)
    torch.manual_seed(7)
    direct = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
        prev_latent=latents[:, 0], target_latent=latents[:, 1], **kwargs)
    torch.manual_seed(7)
    out = unrolled_flow_loss(model=model, x_window=x[:, :2], style_label=labels,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:2], recon_targets=x[:, 1:2, 84:],
        steps=1, **kwargs)
    for key in direct:
        torch.testing.assert_close(out[key], direct[key])
    two = unrolled_flow_loss(model=model, x_window=x[:, :3], style_label=labels,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:3], recon_targets=x[:, 1:3, 84:],
        steps=2, **kwargs)
    expected = two['flow_loss'] + .1 * two['recon_loss'] + .1 * two['root_recon_loss'] + .02 * two['style_hist_loss'] + .02 * two['style_entropy']
    torch.testing.assert_close(two['loss'], expected)


def test_unroll_trajectory_and_feedback_are_identical_to_evaluation_policy():
    model = controller().eval()
    x, labels = loss_args()
    x[:, :, 0] = torch.arange(4)
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(3, 4, 8)
    original = model.compute_loss
    seen = []
    def spy(**kwargs):
        seen.append(kwargs['x_curr'].clone())
        if len(seen) > 1:
            torch.testing.assert_close(kwargs['x_curr'][:, 84:], model.vqvae.decode(kwargs['prev_latent']))
        return original(**kwargs)
    model.compute_loss = spy
    unrolled_flow_loss(model=model, x_window=x, style_label=labels, prev_latent_0=latents[:, 0],
        target_latents=latents[:, 1:], recon_targets=x[:, 1:, 84:], steps=3, solver_steps=2)
    assert [float(row[0, 0]) for row in seen] == [0., 1., 2.]


def test_feedback_clips_before_quantizing_and_motion_matches_latent():
    model = controller().eval()
    model.latent_data_min.fill_(0.)
    model.latent_data_max.fill_(.1)
    motion, z = model.feedback_state(torch.full((2, 8), 50.))
    torch.testing.assert_close(motion, model.vqvae.decode(z))
    torch.testing.assert_close(z, model.requantize_latent(z))


def test_shared_and_pure_recipes_and_style_schedule():
    model = controller().eval()
    x, labels = loss_args()
    recipe = torch.ones(4, 16)
    a = model.encode_conditions(x_curr=x[:, 0], style_label=labels, manual_style_probs=recipe)[2]
    b = model.encode_conditions(x_curr=x[:, 0], style_label=labels, manual_style_probs=recipe[None].expand(3, -1, -1))[2]
    torch.testing.assert_close(a, b)
    model.encode_conditions(x_curr=x[:, 0], style_label=None, manual_style_probs=recipe)
    model.encode_conditions(x_curr=x[:, 0], style_label=None, manual_style_probs=recipe * 1e-30)
    model.encode_conditions(x_curr=x[:, 0], style_label=None, manual_style_probs=recipe * 1e38)
    for invalid in (float('nan'), -1.):
        bad = recipe.clone(); bad[0, 0] = invalid
        with pytest.raises(ValueError):
            model.encode_conditions(x_curr=x[:, 0], style_label=None, manual_style_probs=bad)
    bad = recipe.clone(); bad[0] = 0
    with pytest.raises(ValueError):
        model.encode_conditions(x_curr=x[:, 0], style_label=None, manual_style_probs=bad)
    schedule = labels[:, None].expand(-1, 4).clone(); schedule[:, 2:] = (schedule[:, 2:] + 1) % 3
    noise = case_noise_plan(model, [1, 2, 3], labels.tolist(), 4)
    result = reconstruct_motion_batch(model_name='flowmatch_target', model=model, x=x,
        style_label=schedule, noise_plan=noise, flow_ode_steps=2)
    assert torch.isfinite(result).all()
    one = reconstruct_motion_batch(model_name='flowmatch_target', model=model, x=x[:1],
        style_label=None, manual_style_probs=recipe, noise_plan=noise[:1], flow_ode_steps=2)
    assert one.shape == (1, 4, 375)


def test_case_noise_and_outputs_are_batch_independent():
    model = controller().eval()
    x, labels = loss_args()
    noise = case_noise_plan(model, [1, 2, 3], labels.tolist(), 4)
    small = case_noise_plan(model, [2], [1], 4)
    torch.testing.assert_close(noise[1:2], small, rtol=0, atol=0)
    full = reconstruct_motion_batch(model_name='flowmatch_target', model=model, x=x, style_label=labels,
        noise_plan=noise, flow_ode_steps=2)
    single = reconstruct_motion_batch(model_name='flowmatch_target', model=model, x=x[1:2], style_label=labels[1:2],
        noise_plan=small, flow_ode_steps=2)
    torch.testing.assert_close(full[1:2], single, atol=1e-5, rtol=1e-5)


def test_rng_isolation_checkpoint_and_epoch_boundary_continuation(tmp_path):
    model = controller(); optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=2)
    args = SimpleNamespace(prediction_mode='target')
    state = rng_state()
    expected = torch.randn(3)
    restore_rng_state(state)
    with isolated_rng(123):
        torch.randn(100)
    torch.testing.assert_close(torch.randn(3), expected, rtol=0, atol=0)
    _save_checkpoint(path=tmp_path / 'fixture.pt', model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, best_metric=1., args=args)
    saved = torch.load(tmp_path / 'fixture.pt', weights_only=False)
    assert saved['implementation_version'] == 2 and 'rng_state' in saved
    other = controller(); other.load_state_dict(saved['model'])
    assert state_fingerprint(other.state_dict()) == state_fingerprint(model.state_dict())


def small_training_fixture(tmp_path):
    db = tmp_path / 'db'; db.mkdir()
    segments = []
    start = 0
    for split, size in [('train', 8), ('val', 5), ('test', 5)]:
        for label in range(3):
            segments.append(dict(frame_start=start, num_frames=size, label=label,
                split=split, source_entry_id=len(segments), suffix='FW'))
            start += size
    x = write_db(db, num_frames=start, num_styles=3, segments=segments)
    x[:, 84:87] = 0.; x[:, 87:159] *= .1; x.tofile(db / 'X.f32')
    model = controller()
    ckpt = tmp_path / 'vq.pt'
    torch.save({'model': model.vqvae.state_dict(), 'args': dict(hidden_dim=16,
        decoder_hidden_dim=16, encoder_layers=2, decoder_layers=2, num_latents=4,
        code_dim=2, num_embeddings=16, kmeans_init=False, dead_code_threshold=0)}, ckpt)
    norm = tmp_path / 'norm_stats.npz'
    np.savez(norm, mean=np.zeros(375, np.float32), std=np.ones(375, np.float32))
    return db, ckpt, norm


def training_argv(db, vq, norm, out, steps):
    return ['train', '--db', str(db), '--vqvae-ckpt', str(vq), '--norm-stats', str(norm),
        '--out-dir', str(out), '--no-use-style-code-selector', '--epochs', '2',
        '--batch-size', '6', '--microbatch-size', '2', '--num-workers', '0', '--device', 'cpu',
        '--rollout-val-every', '0', '--max-train-batches', '1', '--max-val-batches', '1',
        '--solver-steps', '2', '--flow-hidden-dim', '16', '--flow-layers', '1',
        '--flow-ffn-dim', '32', '--style-dim', '8', '--time-embed-dim', '8',
        '--seq-len', str(steps + 1), '--unroll-steps', str(steps)]


def test_teacher_forced_defaults_keep_stride_one_and_disable_motion_losses_and_rollout(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trainer'])
    args = trainer.parse_args()
    assert args.epochs == 50 and args.seq_len == 2 and args.stride == 1 and args.unroll_steps == 1
    assert args.recon_weight == args.root_recon_weight == args.style_consistency_weight == 0.
    assert args.rollout_val_every == 0 and not args.train_feedback_context
    assert args.matmul_precision == 'highest' and not args.tf32
    assert args.out_dir.name == 'flowmatch_tf50_fp32_norecon'
    monkeypatch.setattr(sys, 'argv', ['trainer', '--recon-weight', '.1', '--rollout-val-every', '5'])
    args = trainer.parse_args()
    assert args.recon_weight == .1 and args.rollout_val_every == 5 and args.stride == 1


def test_latent_only_cli_saves_zero_loss_flags_without_decoder_or_rollout(tmp_path, monkeypatch):
    from arch.flowmatch_vqvae_controller import FlowMatchVQVAEController
    from Utils import controller_validation
    db, vq, norm = small_training_fixture(tmp_path)
    out = tmp_path / 'latent_only'
    argv = training_argv(db, vq, norm, out, 1)
    index = argv.index('--rollout-val-every')
    del argv[index:index + 2]  # Exercise the new default rather than an explicit opt-out.

    def unused(*args, **kwargs):
        raise AssertionError('Latent-only CLI must not decode, integrate, or evaluate rollout')

    monkeypatch.setattr(BasicVQVAE375to291, 'decode', unused)
    monkeypatch.setattr(FlowMatchVQVAEController, '_integrate', unused)
    monkeypatch.setattr(controller_validation, 'evaluate_validation', unused)
    monkeypatch.setattr(sys, 'argv', argv)
    trainer.main()
    saved = torch.load(out / 'latest.pt', weights_only=False)
    assert saved['args']['recon_weight'] == 0 and saved['args']['stride'] == 1
    assert saved['args']['rollout_val_every'] == 0 and saved['epoch'] == 2
    records = [json.loads(row) for row in (out / 'history.jsonl').read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        for split in ('train', 'val'):
            assert record[split]['recon_loss'] == record[split]['root_recon_loss'] == 0.
    assert (out / 'best_loss.pt').exists() and (out / 'best.pt').exists()
    original = torch.load(vq, weights_only=False)['model']
    for key, tensor in original.items():
        torch.testing.assert_close(saved['model']['vqvae.' + key], tensor, rtol=0, atol=0)


@pytest.mark.parametrize('steps', [1, 2])
def test_cli_fixture_entry_saves_and_logs(tmp_path, monkeypatch, steps):
    db, vq, norm = small_training_fixture(tmp_path)
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', training_argv(db, vq, norm, out, steps))
    trainer.main()
    saved = torch.load(out / 'latest.pt', weights_only=False)
    assert saved['epoch'] == 2 and (out / 'best_loss.pt').exists() and (out / 'best.pt').exists()
    bundled = {k[6:]: v for k, v in saved['model'].items() if k.startswith('vqvae.')}
    assert state_fingerprint(bundled) == state_fingerprint(torch.load(vq, weights_only=False)['model'])
    records = [json.loads(row) for row in (out / 'history.jsonl').read_text().splitlines()]
    assert len(records) == 2 and all(r['val']['style_consistency_loss'] == 0 for r in records)


def test_actual_cli_resume_matches_continuous_epoch_training(tmp_path, monkeypatch):
    db, vq, norm = small_training_fixture(tmp_path)
    full = tmp_path / 'full'; resumed = tmp_path / 'resumed'
    def argv(out):
        return training_argv(db, vq, norm, out, 2)
    monkeypatch.setattr(sys, 'argv', argv(full))
    trainer.main()
    save = trainer._save_checkpoint
    class EpochInterrupted(Exception):
        pass
    def stop_after_epoch_one(**kwargs):
        save(**kwargs)
        if kwargs['path'].name == 'latest.pt' and kwargs['epoch'] == 1:
            raise EpochInterrupted()
    monkeypatch.setattr(trainer, '_save_checkpoint', stop_after_epoch_one)
    monkeypatch.setattr(sys, 'argv', argv(resumed))
    with pytest.raises(EpochInterrupted):
        trainer.main()
    monkeypatch.setattr(trainer, '_save_checkpoint', save)
    monkeypatch.setattr(sys, 'argv', argv(resumed) +
                        ['--resume', str(resumed / 'latest.pt')])
    trainer.main()
    a = torch.load(full / 'latest.pt', weights_only=False)
    b = torch.load(resumed / 'latest.pt', weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    for index in a['optimizer']['state']:
        for key in a['optimizer']['state'][index]:
            torch.testing.assert_close(a['optimizer']['state'][index][key], b['optimizer']['state'][index][key], rtol=0, atol=0)
    assert a['scheduler'] == b['scheduler']
    torch.testing.assert_close(a['rng_state']['torch'], b['rng_state']['torch'], rtol=0, atol=0)


def test_loader_shuffle_is_independent_of_worker_base_seed_consumption():
    from torch.utils.data import RandomSampler
    a = SimpleNamespace(sampler=RandomSampler(range(20)))
    b = SimpleNamespace(sampler=RandomSampler(range(20)))
    trainer._seed_loader(a, seed=42, epoch=2)
    trainer._seed_loader(b, seed=42, epoch=2)
    saved = torch.get_rng_state()
    # A resumed worker pool consumes a new base seed; an existing persistent pool does not.
    torch.empty((), dtype=torch.int64).random_(generator=b.generator)
    assert list(a.sampler) == list(b.sampler)
    torch.testing.assert_close(torch.get_rng_state(), saved, rtol=0, atol=0)


def test_latent_statistics_exclude_val_and_test(tmp_path):
    db, _, _ = small_training_fixture(tmp_path)
    source = np.memmap(db / 'X.f32', mode='r+', dtype='float32', shape=(54, 375))
    source[24:, :8] = 10000.; source.flush()
    fake_vq = SimpleNamespace(latent_dim=8, encode=lambda x: (x[:, :8], None, None))
    mean, std, low, high = trainer._estimate_raw_latent_stats(db_path=db, vqvae=fake_vq,
        device=torch.device('cpu'), norm_mean=np.zeros(375), norm_std=np.ones(375),
        vq_target_cache=None, chunk_rows=7)
    np.testing.assert_allclose(mean, np.asarray(source[:24, :8]).mean(0), atol=1e-6)
    assert high.max() < 10000
    np.testing.assert_allclose(std, np.asarray(source[:24, :8]).std(0), atol=1e-6)


def test_cache_content_change_and_relocation(tmp_path):
    from Utils.reproducibility import artifact_fingerprints, file_sha256
    db, vq, norm = small_training_fixture(tmp_path)
    recorded = artifact_fingerprints(db, vq, norm)
    cache_dir = tmp_path / 'cache'; cache_dir.mkdir()
    indices = np.zeros((54, 4), dtype='uint16'); indices.tofile(cache_dir / 'indices.u16')
    save_vq_target_cache_meta(cache_dir, VQTargetCacheMeta(db=db, vqvae_ckpt=vq, norm_stats=norm,
        num_frames=54, feature_dim=375, num_codebooks=4, code_dim=2, latent_dim=8,
        num_embeddings=16, indices_dtype='uint16', indices_file='indices.u16', fingerprints=recorded,
        indices_sha256=file_sha256(cache_dir / 'indices.u16')))
    cache = VQTargetCache(cache_dir)
    cache.validate(db=tmp_path / 'moved_db', vqvae_ckpt=tmp_path / 'moved_vq.pt', fingerprints=recorded)
    source = np.memmap(db / 'X.f32', mode='r+', dtype='float32', shape=(54, 375))
    source[0, 0] += 1.; source.flush()
    with pytest.raises(ValueError, match='fingerprint'):
        cache.validate(fingerprints=artifact_fingerprints(db, vq, norm))


def test_style_loss_uses_motion_slice_name_mapping_and_frozen_classifier():
    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(45, 100)
        def forward(self, positions):
            positions = positions.squeeze(-1)
            return positions, self.linear(positions)
    classifier = Classifier().train()
    model = controller().train(); x, labels = loss_args()
    # Exercise the full unroll style-loss branch, not just CE on an unrelated input tensor.
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(3, 4, 8)
    out = unrolled_flow_loss(model=model, x_window=x[:, :3], style_label=labels,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:3], recon_targets=x[:, 1:3, 84:],
        steps=2, flow_weight=0., style_consistency_weight=1., style_classifier=classifier,
        motion_mean=torch.arange(291) / 1000., motion_std=torch.ones(291),
        joint_indices=list(range(15)), classifier_label_map=torch.tensor([99, 91, 95]), solver_steps=2)
    out['loss'].backward()
    assert out['style_consistency_loss'] > 0
    assert not classifier.training and all(p.grad is None and not p.requires_grad for p in classifier.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
    pred = torch.randn(3, 291, requires_grad=True)
    full_mean = torch.cat((torch.ones(84) * 1000, torch.arange(291) / 1000.))
    args = dict(pred_motion=pred, style_label=labels, classifier=classifier, joint_indices=list(range(15)))
    a = trainer.style_consistency_loss(**args, motion_mean=full_mean, motion_std=torch.ones(375))
    b = trainer.style_consistency_loss(**args, motion_mean=full_mean[84:], motion_std=torch.ones(291))
    torch.testing.assert_close(a, b)


def test_normalized_target_loss_and_euler_conversion():
    model = controller().eval(); x, labels = loss_args()
    target = model.normalize_latent(model.frame_latent(x[:, 1]))
    field_out = torch.arange(8, dtype=x.dtype)[None].expand(3, -1)
    original = model._predict_velocity
    def capture(**kwargs):
        velocity, output, prediction = original(**kwargs)
        capture.prediction = prediction
        return velocity, output, prediction
    model._predict_velocity = capture
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    torch.testing.assert_close(out['flow_loss'], torch.nn.functional.mse_loss(capture.prediction, target))
    model.field.forward = lambda **kwargs: field_out
    kwargs = dict(z_t=torch.ones(3, 8), z_prev=torch.zeros(3, 8), t=torch.full((3, 1), .75),
                  e_c=x[:, 0, 84:], e_traj=x[:, 0, :84], e_s=torch.zeros(3, 8))
    velocity, _, prediction = original(**kwargs)
    torch.testing.assert_close(prediction, field_out * model.latent_transport_std)
    torch.testing.assert_close(kwargs['z_t'] + .25 * velocity, prediction)
    legacy = build_flowmatch_controller_from_config(dict(flow_hidden_dim=16, flow_layers=1,
        flow_ffn_dim=32, prediction_mode='velocity'), vqvae=copy.deepcopy(model.vqvae), num_styles=3)
    assert legacy.flow_loss_space == 'normalized'


def test_explicit_transport_velocity_loss_uses_transport_space():
    model = controller('velocity').eval(); model.flow_loss_space = 'transport'
    x, labels = loss_args()
    original = model._predict_velocity
    def capture(**kwargs):
        result = original(**kwargs)
        capture.output = result[1]
        return result
    model._predict_velocity = capture
    state = torch.get_rng_state()
    torch.manual_seed(8)
    expected_noise = model.sample_transport_noise_like(torch.zeros(3, 8))
    torch.manual_seed(8)
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    target = model.normalize_latent(model.frame_latent(x[:, 1])) - expected_noise
    torch.testing.assert_close(out['flow_loss'], torch.nn.functional.mse_loss(capture.output,
        model.transport_normalize_latent(target)))
    torch.set_rng_state(state)


def test_latest_has_updated_rollout_selection(tmp_path, monkeypatch):
    from Utils import controller_validation
    db, vq, norm = small_training_fixture(tmp_path); out = tmp_path / 'run'
    classifier = tmp_path / 'classifier.pt'; classifier.write_bytes(b'fixture')
    argv = training_argv(db, vq, norm, out, 1)
    argv[argv.index('--rollout-val-every') + 1] = '1'
    monkeypatch.setattr(sys, 'argv', argv + ['--rollout-val-classifier', str(classifier)])
    calls = []
    def validation(*args, **kwargs):
        calls.append(1)
        return dict(mpjpe_cm=8 - len(calls), jitter_cm=.3, foot_sliding_local_cm=.5,
            matched_root_ade_cm=3., cross_root_ade_cm=4., matched_accuracy=.8, cross_accuracy=.7)
    monkeypatch.setattr(controller_validation, 'evaluate_validation', validation)
    trainer.main()
    latest = torch.load(out / 'latest.pt', weights_only=False)
    best = torch.load(out / 'best.pt', weights_only=False)
    assert latest['selection'] == best['selection'] and best['epoch'] == 2
    assert latest['best_metric'] == best['best_metric']


def test_legacy_read_and_vq_override_consistency(tmp_path, monkeypatch):
    from Utils.motion_eval import load_motion_model
    db, vq, norm = small_training_fixture(tmp_path); out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', training_argv(db, vq, norm, out, 1))
    trainer.main()
    cp = torch.load(out / 'latest.pt', weights_only=False)
    for field in ('flow_loss_space', 'inference_requantize', 'inference_latent_clip'):
        cp['args'].pop(field)
    cp.pop('implementation_version'); cp.pop('training_objective_version'); cp.pop('rng_state')
    old = tmp_path / 'legacy.pt'; torch.save(cp, old)
    restored = load_motion_model('flowmatch_target', old, num_styles=3)
    assert restored.flow_loss_space == 'transport' and not restored.inference_requantize and restored.inference_latent_clip
    modified = torch.load(vq, weights_only=False)
    key = next(k for k, value in modified['model'].items() if value.dtype == torch.float32)
    modified['model'][key] += .01
    other = tmp_path / 'different_vq.pt'; torch.save(modified, other)
    with pytest.raises(ValueError, match='VQ'):
        load_motion_model('flowmatch_target', old, num_styles=3, flow_vqvae_ckpt=other)


def test_protocol_bvh_and_plot_export_uses_existing_exporter(tmp_path):
    from analysis.export_examples import export_example
    root = Path(__file__).resolve().parents[1]
    db, _, _ = small_training_fixture(tmp_path)
    out = tmp_path / 'export'; out.mkdir()
    ref = np.zeros((8, 375), np.float32)
    ref[:, 87:159].reshape(8, 24, 3)[:, :, 1] = .1
    ref[:, 159:231].reshape(8, 24, 3)[:, :, 0] = 1.
    ref[:, 231:303].reshape(8, 24, 3)[:, :, 1] = 1.
    export_example(SimpleNamespace(out=out, db=db, bvh_template=root.parent / 'GenoViewPython-main/resources/Geno_stance.bvh'),
                   'fixture', ref, ref, case={'source_start': 0})
    assert (out / 'examples/fixture_generated.bvh').is_file()
    assert (out / 'examples/fixture_root.png').is_file()


def test_metadata_class_mapping_and_cluster_intervals(tmp_path):
    mapping = classifier_label_mapping(['a', 'b'], {'class_names': ['b', 'ood', 'a']})
    assert mapping.tolist() == [2, 0]
    with pytest.raises(ValueError):
        validate_recipe_metadata({'vq_state_sha256': 'old', 'selected_slots': [1]}, vq_fingerprint='new', selected_slots=[1])
    with pytest.raises(ValueError):
        validate_fingerprints({'db_sha256': 'a'}, {'db_sha256': 'b'})
    low, high = cluster_interval([1., 1., 0., 0.], [1, 1, 2, 2])
    assert low <= .5 <= high
    meta = VQTargetCacheMeta(db=tmp_path, vqvae_ckpt=tmp_path / 'vq.pt', norm_stats=tmp_path / 'norm.npz',
        num_frames=2, feature_dim=375, num_codebooks=4, code_dim=2, latent_dim=8,
        num_embeddings=16, indices_dtype='uint16', indices_file='indices.u16', complete=False)
    save_vq_target_cache_meta(tmp_path, meta)
    with pytest.raises(ValueError, match='incomplete'):
        VQTargetCache(tmp_path)


def test_world_foot_sliding_detects_moving_world_feet():
    ref = np.zeros((8, 375), dtype=np.float32)
    ref[:, 84] = 1.
    positions = ref[:, 87:159].reshape(8, 24, 3)
    positions[:, :, 1] = .1
    positions[:, :, 0] = -np.arange(8, dtype=np.float32)[:, None] / 60
    # Valid rotation axes avoid degenerate zero-vector conversions.
    ref[:, 159:231].reshape(8, 24, 3)[:, :, 0] = 1.
    ref[:, 231:303].reshape(8, 24, 3)[:, :, 1] = 1.
    perfect = compute_sequence_metric_sums(reference_375=ref, prediction_375=ref, cfg=MotionEvalConfig())
    assert perfect.foot_sliding_world_count > 0
    assert perfect.foot_sliding_world_sum_cm < 1e-4
    moving = ref.copy(); moving[:, 87:159].reshape(8, 24, 3)[:, :, 0] = 0
    bad = compute_sequence_metric_sums(reference_375=ref, prediction_375=moving, cfg=MotionEvalConfig())
    assert bad.foot_sliding_world_sum_cm / bad.foot_sliding_world_count > 1.
    high = ref.copy(); high[:, 87:159].reshape(8, 24, 3)[:, :, 1] = 3.
    missing = compute_sequence_metric_sums(reference_375=high, prediction_375=high, cfg=MotionEvalConfig())
    assert missing.foot_sliding_count == missing.foot_sliding_world_count == 0
