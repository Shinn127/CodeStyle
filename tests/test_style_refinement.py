from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller, loss_args, small_training_fixture, training_argv
from arch.flowmatch_vqvae_controller import StyleCodeSelector, build_flowmatch_controller_from_config
from Utils.reproducibility import state_fingerprint, artifact_fingerprints
from Utils.style_code_prior import TrainStyleRecipePool, sparse_code_histogram, save_style_code_prior
from tools.build_zeroshot_style_recipe import _build_recipe_slots
from _memmap_fixtures import write_meta
import training.train_flowmatch_vqvae_controller as trainer


def test_current_training_defaults_keep_explicit_prior_and_fp32_settings(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trainer'])
    defaults = trainer.parse_args()
    assert defaults.recipe_prob == 0 and defaults.style_consistency_weight == 0
    assert defaults.style_selector_init == 'prior' and defaults.style_proto_normalization == 'none'
    assert defaults.style_entropy_weight == .003
    assert defaults.fp32_math_attention and defaults.style_val_batch_size == 64


def test_prior_initialization_fits_dense_logits_without_vq_updates():
    model = controller().train()
    prior = torch.rand_like(model.style_prior_probs) + .01
    prior /= prior.sum(-1, keepdim=True)
    before = state_fingerprint(model.vqvae.state_dict())
    result = model.style_selector.initialize_from_prior(prior)
    labels = torch.arange(3)
    logits = model.style_selector.to_logits(model.style_selector.style_embedding(labels)).reshape_as(prior)
    torch.testing.assert_close(logits.softmax(-1), prior, atol=1e-6, rtol=1e-5)
    assert result['rank'] == 3 and model.style_selector.training
    assert state_fingerprint(model.vqvae.state_dict()) == before


def test_prior_initialization_supports_actual_90style_8slot_2048code_shape():
    torch.manual_seed(10)
    selector = StyleCodeSelector(num_styles=90, num_embeddings=2048, code_dim=4,
        selected_slots=range(8), style_dim=64, hidden_dim=128)
    prior = torch.rand(90, 8, 2048) + .001
    prior /= prior.sum(-1, keepdim=True)
    result = selector.initialize_from_prior(prior)
    assert result['rank'] == 90 and result['max_probability_error'] < 1e-6


def test_reduced_entropy_preserves_uniform_support_local_stability():
    p = torch.full((16,), .8 / 12); p[:4] = .2 / 4
    def objective(logits, entropy_weight):
        r = logits.softmax(-1)
        q = torch.cat((.2 * r, torch.full((12,), .8 / 12)))
        return .02 * -(p * q.log()).sum() + entropy_weight * -(r * r.log()).sum()
    equal = torch.zeros(4)
    perturbed = torch.tensor([.1, -.1, 0., 0.])
    assert objective(perturbed, .003) > objective(equal, .003)
    assert objective(perturbed, .02) < objective(equal, .02)


def test_raw_prototype_preserves_scale_and_missing_config_keeps_legacy_norm():
    selector = StyleCodeSelector(num_styles=3, num_embeddings=16, code_dim=2,
        selected_slots=[0, 1, 2, 3], style_dim=8, prototype_normalization='none')
    selector.proto_to_style = torch.nn.Identity()
    codebook = torch.randn(16, 2)
    recipe = torch.eye(16)[:4]
    a = selector(codebook_weight=codebook, manual_probs=recipe)
    b = selector(codebook_weight=2 * codebook, manual_probs=recipe)
    torch.testing.assert_close(b['style_feature'], 2 * a['style_feature'])
    assert all(not p.requires_grad for p in selector.proto_norm.parameters())
    old = build_flowmatch_controller_from_config(dict(use_style_code_selector=True,
        style_slot_indices=[0, 1, 2, 3], flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32),
        vqvae=controller().vqvae, num_styles=3)
    assert old.style_selector.prototype_normalization == 'layernorm'


def test_recipe_conditioning_regularizes_learned_selector_not_fixed_recipe():
    model = controller(); x, labels = loss_args()
    recipe = torch.eye(16)[torch.tensor([0, 3, 7, 9])]
    outputs = []
    gradients = []
    for manual in (None, recipe):
        model.zero_grad()
        torch.manual_seed(9)
        out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
            manual_style_probs=manual, flow_weight=0., style_hist_weight=.02, style_entropy_weight=.003)
        out['loss'].backward()
        outputs.append(out)
        gradients.append([p.grad.clone() for p in model.style_selector.to_logits.parameters()])
    for metric in ('style_hist_loss', 'style_entropy', 'style_top1_prob', 'style_prior_kl'):
        torch.testing.assert_close(outputs[0][metric], outputs[1][metric])
    assert outputs[1]['conditioning_entropy'] == 0
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_recipe_is_reused_through_unroll_and_steps_one_matches_direct_loss():
    model = controller().eval(); x, labels = loss_args()
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(3, 4, 8)
    recipe = torch.eye(16)[:4][None].expand(3, -1, -1)
    common = dict(recon_weight=.1, style_hist_weight=.02, style_entropy_weight=.003,
                  manual_style_probs=recipe, solver_steps=2)
    torch.manual_seed(12)
    direct = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
        prev_latent=latents[:, 0], target_latent=latents[:, 1], **common)
    torch.manual_seed(12)
    single = trainer.unrolled_flow_loss(model=model, x_window=x[:, :2], style_label=labels,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:2], recon_targets=x[:, 1:2, 84:], steps=1, **common)
    for key in direct:
        torch.testing.assert_close(single[key], direct[key])
    seen = []
    compute = model.compute_loss
    def spy(**kwargs):
        seen.append(kwargs['manual_style_probs'].clone())
        return compute(**kwargs)
    model.compute_loss = spy
    trainer.unrolled_flow_loss(model=model, x_window=x[:, :3], style_label=labels,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:3], recon_targets=x[:, 1:3, 84:], steps=2, **common)
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], rtol=0, atol=0)


def test_pool_reads_only_train_clips_and_histogram_matches_export(tmp_path):
    db, _, _ = small_training_fixture(tmp_path); model = controller()
    segments = json.loads((db / 'segments.json').read_text())
    indices = np.full((54, 4), 100, np.int64)
    for segment in segments[:3]:
        indices[segment['frame_start']:segment['frame_start'] + segment['num_frames']] = segment['label'] + 1
    (db / 'segments.json').write_text(json.dumps(segments))
    cache = SimpleNamespace(indices=indices)
    pool = TrainStyleRecipePool(db=db, model=model, cache=cache, mean=np.zeros(375), std=np.ones(375),
        frames=4, exemplars=5)
    recipes, fraction = pool.sample(torch.tensor([0, 1, 2]), probability=1.)
    assert fraction == 1.
    assert recipes.argmax(-1).tolist() == [[1] * 4, [2] * 4, [3] * 4]
    isolated, actual = pool.sample(torch.tensor([0, 1, 2]), probability=1., excluded_source_ids={0})
    assert actual == pytest.approx(2/3) and isolated[0].sum() == 0
    slots = _build_recipe_slots(indices=indices[:4], selected_slots=(0, 1, 2, 3), num_embeddings=16, topk=2)
    expected = sparse_code_histogram(indices[:4], selected_slots=(0, 1, 2, 3), num_embeddings=16, topk=2)
    for local, slot in enumerate(slots):
        np.testing.assert_allclose(expected[local, slot['codes']], slot['weights'])
    segments[2]['split'] = 'test'; (db / 'segments.json').write_text(json.dumps(segments))
    with pytest.raises(ValueError, match='lacks'):
        TrainStyleRecipePool(db=db, model=model, cache=cache, mean=np.zeros(375), std=np.ones(375), frames=4)


def test_cache_and_online_train_recipe_match_and_vq_stays_frozen(tmp_path):
    db, _, _ = small_training_fixture(tmp_path); model = controller().train()
    raw = np.fromfile(db / 'X.f32', dtype=np.float32).reshape(-1, 375)
    _, indices, _ = model.vqvae.encode(torch.from_numpy(raw))
    cache = SimpleNamespace(indices=indices.numpy())
    before = state_fingerprint(model.vqvae.state_dict())
    pools = [TrainStyleRecipePool(db=db, model=model, cache=c, mean=np.zeros(375), std=np.ones(375),
             frames=4, exemplars=1) for c in (cache, None)]
    results = []
    for pool in pools:
        torch.manual_seed(56)
        results.append(pool.sample(torch.arange(3), probability=1.)[0])
    torch.testing.assert_close(*results, rtol=0, atol=0)
    assert state_fingerprint(model.vqvae.state_dict()) == before


def style_cli_fixture(tmp_path):
    db, vq, norm = small_training_fixture(tmp_path)
    source = np.fromfile(db / 'X.f32', dtype='float32').reshape(-1, 375)
    np.concatenate((source, source[:24])).tofile(db / 'X.f32')
    labels = np.fromfile(db / 'Labels.i32', dtype='int32')
    np.concatenate((labels, labels[:24])).tofile(db / 'Labels.i32')
    segments = json.loads((db / 'segments.json').read_text())
    segments += [{**segments[i], 'frame_start': 54 + 8 * i, 'source_entry_id': 9 + i} for i in range(3)]
    write_meta(db, num_frames=78, num_styles=3, segments=segments)
    counts = np.random.default_rng(8).integers(1, 20, (3, 4, 16))
    prior = counts / counts.sum(-1, keepdims=True)
    path = tmp_path / 'prior.npz'
    save_style_code_prior(path, prior=prior, global_prior=prior.mean(0), counts=counts,
        slot_saliency=np.ones(4), selected_slots=np.arange(4), style_names=[f'style_{i}' for i in range(3)],
        fingerprints=artifact_fingerprints(db, vq, norm), split='train')
    def argv(out):
        result = training_argv(db, vq, norm, out, 2)
        result.remove('--no-use-style-code-selector')
        return result + ['--style-code-prior', str(path), '--style-num-slots', '4', '--style-selector-hidden-dim', '16',
                         '--recipe-frames', '4', '--recipe-exemplars', '1', '--recipe-prob', '1']
    return db, vq, norm, argv


def test_style_cli_recipe_entry_initialization_and_exact_resume(tmp_path, monkeypatch):
    _, vq, _, argv = style_cli_fixture(tmp_path)
    full = tmp_path / 'full'; resumed = tmp_path / 'resumed'
    monkeypatch.setattr(sys, 'argv', argv(full)); trainer.main()
    a = torch.load(full / 'latest.pt', weights_only=False)
    assert a['training_objective_version'] == 11  # donor-free objective family
    assert a['args']['selector_initialization']['method'] == 'prior_logits_lstsq'
    records = [json.loads(row) for row in (full / 'history.jsonl').read_text().splitlines()]
    assert any(r['train']['recipe_fraction'] > 0 for r in records)
    assert all(r['val']['recipe_fraction'] == 0 for r in records)
    save = trainer._save_checkpoint
    class Interrupted(Exception):
        pass
    def stop(**kwargs):
        save(**kwargs)
        if kwargs['path'].name == 'latest.pt' and kwargs['epoch'] == 1:
            raise Interrupted()
    monkeypatch.setattr(trainer, '_save_checkpoint', stop)
    monkeypatch.setattr(sys, 'argv', argv(resumed))
    with pytest.raises(Interrupted):
        trainer.main()
    monkeypatch.setattr(trainer, '_save_checkpoint', save)
    monkeypatch.setattr(sys, 'argv', argv(resumed) + ['--resume', str(resumed / 'latest.pt')]); trainer.main()
    b = torch.load(resumed / 'latest.pt', weights_only=False)
    assert a['args']['selector_initialization'] == b['args']['selector_initialization']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    original = torch.load(vq, weights_only=False)['model']
    bundled = {k[6:]: v for k, v in a['model'].items() if k.startswith('vqvae.')}
    assert state_fingerprint(original) == state_fingerprint(bundled)
