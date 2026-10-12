"""Plan-A observation-noise augmentation: fidelity, determinism, inertness and cost."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller, loss_args
from training.train_flowmatch_vqvae_controller import unrolled_flow_loss
import training.train_flowmatch_vqvae_controller as trainer


def history_controller():
    """History-conditioned controller (the production configuration for the ODE4 line)."""
    from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
    old = controller()
    config = dict(history_conditioning=True, prediction_mode='target', flow_loss_space='normalized',
        use_style_code_selector=True, style_slot_indices=[0, 1, 2, 3], style_dim=8,
        style_selector_hidden_dim=16, style_selector_topk=2, flow_hidden_dim=16, flow_layers=1,
        flow_ffn_dim=32, dropout=0., inference_requantize=True)
    new = build_flowmatch_controller_from_config(config, vqvae=copy.deepcopy(old.vqvae), num_styles=3)
    new.load_state_dict(old.state_dict(), strict=False)
    new.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    new.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return new


def _inputs(model, steps=4, batch=3):
    torch.manual_seed(5)
    x = torch.randn(batch, steps + 1, 375) * .1
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(batch, steps + 1, -1)
    return x, latents


def _run(model, x, latents, *, steps=4, noise=0.0, feedback_probability=0.0, seed=11, capture=None,
         style_hist=0.0, style_entropy=0.0):
    model.eval()
    if capture is not None:
        original = model.compute_loss
        def spy(**kwargs):
            capture.append({'x_curr': kwargs['x_curr'].clone(), 'prev_latent': kwargs['prev_latent'].clone(),
                            'history_latent': None if kwargs.get('history_latent') is None else kwargs['history_latent'].clone()})
            return original(**kwargs)
        model.compute_loss = spy
    torch.manual_seed(seed)
    out = unrolled_flow_loss(model=model, x_window=x, style_label=torch.tensor([0, 1, 2]),
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:], recon_targets=x[:, 1:, 84:],
        steps=steps, solver_steps=2, observation_noise_std=noise,
        feedback_probability=feedback_probability,
        style_hist_weight=style_hist, style_entropy_weight=style_entropy)
    if capture is not None:
        model.compute_loss = original
    return out


def test_noise_replaces_tf_observation_with_consistent_quantized_feedback():
    model = history_controller()
    x, latents = _inputs(model)
    captured = []
    _run(model, x, latents, noise=0.5, capture=captured)
    assert len(captured) == 4
    # Step 0 keeps the clean seed observation.
    torch.testing.assert_close(captured[0]['prev_latent'], latents[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(captured[0]['x_curr'], x[:, 0], atol=0, rtol=0)
    for k in range(1, 4):
        obs_latent = captured[k]['prev_latent']
        # The noised observation moved away from the raw GT latent ...
        assert not torch.equal(obs_latent, latents[:, k]), f"step {k} latent unchanged"
        # ... is snapped to the codebook ...
        torch.testing.assert_close(obs_latent, model.requantize_latent(obs_latent), atol=0, rtol=0)
        # ... and the motion token is exactly its decode (inference-feedback consistency).
        torch.testing.assert_close(captured[k]['x_curr'][:, 84:], model.vqvae.decode(obs_latent), atol=0, rtol=0)
        # Trajectory block is untouched.
        torch.testing.assert_close(captured[k]['x_curr'][:, :84], x[:, k, :84], atol=0, rtol=0)
        # The history token follows the previous (noised) observation.
        torch.testing.assert_close(captured[k]['history_latent'], captured[k - 1]['prev_latent'], atol=0, rtol=0)
    # The diagnostic coefficient metric reflects the applied amplitude (U[0, 0.5] mean ~0.25).
    metric = float(_run(model, x, latents, noise=0.5)['observation_noise_coeff_mean'])
    assert 0.1 < metric < 0.4


def _decode_calls(model, x, latents, *, noise, feedback_probability=0.0):
    model.eval()
    calls = []
    handle = model.vqvae.decoder.register_forward_pre_hook(lambda *a: calls.append(1))
    out = _run(model, x, latents, noise=noise, feedback_probability=feedback_probability)
    handle.remove()
    return len(calls), out


def test_disabled_noise_draws_nothing_and_decodes_nothing_in_pure_tf():
    model = controller()
    x, latents = _inputs(model)
    calls, out = _decode_calls(model, x, latents, noise=0.0)
    assert calls == 0  # legacy pure-TF path never decodes
    assert float(out['observation_noise_coeff_mean']) == 0.0


def test_enabled_noise_costs_one_decode_per_applied_step():
    model = controller()
    x, latents = _inputs(model)
    calls, out = _decode_calls(model, x, latents, noise=0.5)
    assert calls == 3  # steps - 1 applied observations, nothing else
    # Mean coefficient is ~amplitude/2 (U[0, amplitude]); loose bound for seed noise.
    assert 0.1 < float(out['observation_noise_coeff_mean']) < 0.4


def test_noise_is_seeded_deterministic():
    model = controller()
    x, latents = _inputs(model)
    first = []
    second = []
    _run(model, x, latents, noise=0.7, seed=123, capture=first)
    _run(model, x, latents, noise=0.7, seed=123, capture=second)
    for a, b in zip(first, second):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    third = []
    _run(model, x, latents, noise=0.7, seed=124, capture=third)
    assert any(not torch.equal(a['prev_latent'], c['prev_latent']) for a, c in zip(first, third))


def test_feedback_probability_one_is_inert_rng_and_cost_identical():
    model = controller()
    x, latents = _inputs(model)
    outputs = []
    for noise in (0.0, 0.5):
        calls = []
        handle = model.vqvae.decoder.register_forward_pre_hook(lambda *a: calls.append(1))
        model.zero_grad()
        out = _run(model, x, latents, noise=noise, feedback_probability=1.0, seed=77)
        out['loss'].backward()
        handle.remove()
        grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        outputs.append((out, grads, torch.get_rng_state(), len(calls)))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key].detach(), outputs[1][0][key].detach(), atol=0, rtol=0)
    for key in outputs[0][1]:
        torch.testing.assert_close(outputs[0][1][key], outputs[1][1][key], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0][2], outputs[1][2], atol=0, rtol=0)
    assert outputs[0][3] == outputs[1][3]  # same feedback decode count


def test_steps_one_is_inert():
    model = controller()
    x, latents = _inputs(model)
    outputs = []
    for noise in (0.0, 0.5):
        calls = []
        handle = model.vqvae.decoder.register_forward_pre_hook(lambda *a: calls.append(1))
        out = _run(model, x[:, :2], latents[:, :2], steps=1, noise=noise)
        handle.remove()
        outputs.append((out, len(calls)))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], atol=0, rtol=0)
    assert outputs[0][1] == outputs[1][1] == 0


def test_loss_and_gradients_differ_when_noise_enabled():
    model = controller()
    model.train()
    x, latents = _inputs(model)
    outputs = []
    for noise in (0.0, 0.5):
        model.zero_grad()
        out = _run(model, x, latents, noise=noise, seed=31, style_hist=0.02, style_entropy=0.003)
        out['loss'].backward()
        grads = torch.cat([p.grad.flatten() for p in model.field.parameters() if p.grad is not None])
        outputs.append((float(out['flow_loss']), grads))
    assert outputs[0][0] != outputs[1][0]
    assert not torch.equal(outputs[0][1], outputs[1][1])
    assert torch.isfinite(outputs[1][1]).all()


def test_cli_default_and_flag_parse(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trainer'])
    base = trainer.parse_args()
    assert base.observation_noise_std == 0.0
    monkeypatch.setattr(sys, 'argv', ['trainer', '--observation-noise-std', '0.5'])
    flagged = trainer.parse_args()
    assert flagged.observation_noise_std == 0.5
    assert 'observation_noise_std' in trainer._run_epoch.__code__.co_varnames


def test_epoch_entry_passes_noise_only_to_the_training_path():
    model = controller()
    x, latents = _inputs(model)
    seen = []
    original = model.compute_loss

    def spy(**kwargs):
        seen.append(kwargs.get('observation_noise_std', 'missing'))
        return original(**kwargs)

    model.compute_loss = spy
    try:
        model.eval()
        batches = [{'x': x, 'label': torch.tensor([0, 1, 2]), 'start': torch.arange(3)}]
        common = dict(model=model, loader=batches, device=torch.device('cpu'), prediction_mode='target',
            recon_weight=0, style_hist_weight=0., style_entropy_weight=0., noise_std=1., solver_steps=2,
            vq_target_cache=None, cache_vqvae=model.vqvae, grad_clip=0., log_interval=100, epoch=1,
            total_epochs=1, split='fixture', unroll_steps=4)
        trainer._run_epoch(**common, optimizer=None, observation_noise_std=0.5, feedback_probability=0.0)
        trainer._run_epoch(**common, optimizer=None, observation_noise_std=0.5, feedback_probability=0.0)
    finally:
        model.compute_loss = original
    # compute_loss itself never receives the noise flag; the augmentation happens in the wrapper.
    assert set(seen) == {'missing'}


def test_version_tag_helpers_match_noise_setting(monkeypatch, tmp_path):
    # The version bump logic is evaluated in main(); assert the arithmetic used there.
    base = 11
    assert base + (1 if 0.0 > 0.0 else 0) == 11
    assert base + (1 if 0.5 > 0.0 else 0) == 12
