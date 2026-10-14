"""Code-decision CE (--code-ce-weight): fidelity, gradient direction, freezing, wiring.

The 2026-10-14 code-fidelity revision adds a per-slot cross-entropy over nearest-code
decisions on top of the continuous flow MSE: logits = -||p - e_k||^2 / T with the GT
next latent's code as target, T auto = (median codebook NN spacing / 2)^2. Weight 0 must
stay bit-identical; the codebook must stay frozen; the gradient must pull the prediction
toward the target code.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller


def _inputs(model, batch=3):
    x = torch.randn(batch, 4, 375) * .1
    return x, torch.tensor([0, 1, 2][:batch])


def test_zero_weight_is_bit_identical_and_ce_is_inert():
    model = controller()
    x, labels = _inputs(model)
    torch.manual_seed(9)
    a = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    torch.manual_seed(9)
    b = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels, code_ce_weight=0.0)
    assert torch.equal(a['loss'], b['loss'])
    assert float(b['code_ce_loss']) == 0.0 and float(b['code_ce_match']) == 0.0
    # The key must exist even when disabled so `_metric_keys` aggregation works.
    assert 'code_ce_loss' in b and 'code_ce_match' in b


def test_prediction_equals_target_gives_full_match_and_near_zero_ce():
    model = controller()
    torch.manual_seed(3)
    z = torch.randn(5, model.latent_dim) * .5
    ce, match = model.code_decision_loss(z, z)
    assert float(match) == 1.0
    assert float(ce) < 0.2  # distance-to-own-centre logit 0 dominates
    far, far_match = model.code_decision_loss(z + 5.0, z)
    assert float(far) > float(ce)
    assert float(far_match) < 0.5


def test_gradient_pulls_prediction_toward_the_target_code():
    model = controller()
    torch.manual_seed(4)
    target = torch.randn(6, model.latent_dim) * .5
    pred = (target + 0.25).requires_grad_(True)
    ce, _ = model.code_decision_loss(pred, target)
    ce.backward()
    # Gradient descent applies pred <- pred - lr * grad, so sign(grad) == sign(pred - target)
    # is the direction that closes the gap.
    toward = (pred.grad.sign() == (pred.detach() - target).sign()).float().mean()
    assert float(toward) > 0.85


def test_auto_temperature_follows_codebook_spacing_and_can_be_overridden():
    model = controller()
    auto = model.code_ce_temperature()
    embed = model._shared_codebook().float()
    d = torch.cdist(embed, embed); d.fill_diagonal_(float('inf'))
    spacing = float(d.min(-1).values.median())
    assert abs(auto - max((spacing / 2) ** 2, 1e-12)) < 1e-9
    assert model.code_ce_temperature(0.25) == 0.25


def test_ce_training_keeps_vq_frozen_and_field_gradients_flow():
    model = controller()
    model.train()
    x, labels = _inputs(model)
    vq_before = {k: v.clone() for k, v in model.vqvae.state_dict().items()}
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels, code_ce_weight=1.0)
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.vqvae.parameters())
    assert all(torch.equal(vq_before[k], v) for k, v in model.vqvae.state_dict().items())


def test_shaped_errors_are_rejected():
    model = controller()
    z = torch.randn(4, model.latent_dim)
    with pytest.raises(Exception):
        model.code_decision_loss(z, torch.randn(3, model.latent_dim))  # batch mismatch is a
        # broadcast/CE error rather than silent truncation


def test_trainer_wiring_flag_resume_and_metric_keys():
    import training.train_flowmatch_vqvae_controller as trainer
    source = Path(trainer.__file__).read_text()
    assert '"--code-ce-weight"' in source and '"--code-ce-temperature"' in source
    assert "'code_ce_weight'" in source and "'code_ce_temperature'" in source
    keys = trainer._metric_keys('target')
    assert 'code_ce_loss' in keys and 'code_ce_match' in keys
    import argparse
    args = argparse.Namespace(code_ce_weight=0.01, recon_weight=0.0, root_recon_weight=0.0,
                              style_hist_weight=0.0, style_consistency_weight=0.0, style_entropy_weight=0.0,
                              style_prior_loss='dense')
    breakdown = trainer._weighted_loss_breakdown({'loss': 0.05, 'flow_loss': 0.04, 'code_ce_loss': 2.5,
                                                 'style_prior_target_entropy': 0.0, 'style_prior_kl': 0.0}, args)
    assert breakdown['code_ce']['weighted'] == pytest.approx(0.025)


def test_rollout_training_step_carries_the_ce_through_unrolled_loss():
    """The CE must be applied on the same prediction that feeds the next rollout step."""
    import training.train_flowmatch_vqvae_controller as trainer
    model = controller()
    model.train()
    x = torch.randn(3, 3, 375) * .1
    latents = model.frame_latent(x.reshape(-1, 375)).reshape(3, 3, -1)
    out = trainer.unrolled_flow_loss(
        model=model, x_window=x, style_label=torch.tensor([0, 1, 2]),
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:], recon_targets=x[:, 1:, 84:],
        steps=2, solver_steps=1, feedback_probability=1.0,
        code_ce_weight=1.0)
    assert 'code_ce_loss' in out and float(out['code_ce_loss']) > 0
    assert float(out['code_ce_match']) >= 0.0
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
