from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller, loss_args
from Utils.motion_eval import reconstruct_motion_batch
from training.train_flowmatch_vqvae_controller import unrolled_flow_loss, controller_collate
import training.train_flowmatch_vqvae_controller as trainer


@pytest.mark.parametrize('mode', ['target', 'velocity'])
def test_teacher_forced_without_motion_losses_skips_solver_decoder_and_cached_encoder(mode, monkeypatch):
    model = controller(mode)
    x, labels = loss_args()
    x = x[:, :2]
    latents = model.frame_latent(x.flatten(0, 1))
    original_vq = {k: v.clone() for k, v in model.vqvae.state_dict().items()}
    original_field = [p.detach().clone() for p in model.field.parameters()]
    calls = []
    hook = model.field.register_forward_hook(lambda *args: calls.append(1))

    def unused(*args, **kwargs):
        raise AssertionError('Teacher-forced latent-only training must skip this path')

    monkeypatch.setattr(model, '_integrate', unused)
    monkeypatch.setattr(model, 'frame_latent', unused)
    monkeypatch.setattr(model.vqvae, 'decode', unused)

    class Cache:
        def lookup_latents(self, ids, **kwargs):
            return latents[ids]

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    common = dict(model=model,
        loader=[dict(x=x, label=labels, start=torch.arange(3) * 2)],
        device=torch.device('cpu'), prediction_mode=mode, recon_weight=0.,
        root_recon_weight=0., style_hist_weight=.02, style_entropy_weight=.003,
        noise_std=1.,
        solver_steps=8,
        vq_target_cache=Cache(), cache_vqvae=model.vqvae, grad_clip=1.,
        log_interval=100, epoch=1, total_epochs=1, split='fixture',
        unroll_steps=1, microbatch_size=2, train_feedback_context=False)
    for current_optimizer in (optimizer, None):
        metrics = trainer._run_epoch(**common, optimizer=current_optimizer)
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics['recon_loss'] == metrics['root_recon_loss'] == metrics['style_consistency_loss'] == 0.
    hook.remove()
    assert len(calls) == 4  # Two microbatches each for train and validation; one field call per microbatch.
    assert any(not torch.equal(before, after) for before, after in zip(original_field, model.field.parameters()))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.style_selector.parameters())
    assert not model.vqvae.training and all(p.grad is None for p in model.vqvae.parameters())
    for key, tensor in model.vqvae.state_dict().items():
        torch.testing.assert_close(tensor, original_vq[key], rtol=0, atol=0)


def uncached_euler(self, *, z_prev, e_c, e_traj, e_s, num_steps, noise_std, z0=None,
                   history_delta=None, condition_tokens=None):
    z = self.sample_transport_noise_like(z_prev, noise_std) if z0 is None else z0
    for step in range(num_steps):
        t = z.new_full((z.size(0), 1), step / num_steps)
        velocity, _, _ = self._predict_velocity(z_t=z, z_prev=z_prev, t=t,
                                               e_c=e_c, e_traj=e_traj, e_s=e_s,history_delta=history_delta)
        z = z + velocity / num_steps
    return z


@pytest.mark.parametrize('mode', ['target', 'velocity'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cached_euler_preserves_loss_and_all_controller_gradients(mode, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable in this execution context')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = controller(mode).to(device).train()
    reference = copy.deepcopy(model)
    reference._integrate = MethodType(uncached_euler, reference)
    x, labels = loss_args()
    x, labels = x.to(device), labels.to(device)
    outputs = []
    for current in (model, reference):
        torch.manual_seed(53)
        out = current.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
            recon_weight=.1, root_recon_weight=.1, style_hist_weight=.02,
            style_entropy_weight=.003, solver_steps=8, return_prediction=True)
        out['loss'].backward()
        outputs.append(out)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=1e-5, atol=1e-6)
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
        else:
            assert a.grad is None and b.grad is None


def test_controller_collate_preserves_inputs_without_duplicate_feature_blocks():
    x, labels = loss_args()
    samples = [{'x': row, 'traj': row[:, :84], 'motion': row[:, 84:],
                'label': label, 'start': torch.tensor(i), 'segment_id': torch.tensor(i)}
               for i, (row, label) in enumerate(zip(x, labels))]
    expected = torch.utils.data.default_collate(samples)
    actual = controller_collate(samples)
    assert set(actual) == {'x', 'label', 'start', 'segment_id'}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual['x'].dtype == torch.float32


def test_euler_projects_fixed_conditions_once_and_readiness_restores_on_load():
    model = controller().eval()
    calls = []
    hooks = [module.register_forward_hook(lambda *args: calls.append(1)) for module in
             (model.field.motion_proj, model.field.traj_proj, model.field.style_proj)]
    x, labels = loss_args()
    model.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=8)
    assert len(calls) == 3
    for hook in hooks:
        hook.remove()
    state = copy.deepcopy(model.state_dict())
    model._latent_stats_initialized = False
    model.set_style_code_prior(None)
    model.load_state_dict(state)
    assert model._latent_stats_initialized and model._style_prior_initialized
    model.set_style_code_prior(None)
    assert not model._style_prior_initialized
    with pytest.raises(RuntimeError, match='prior'):
        model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels, style_hist_weight=.02)


def test_unroll_skips_only_unused_style_statistics():
    model = controller().eval()
    x, labels = loss_args()
    latents = model.frame_latent(x.flatten(0, 1)).reshape(3, 4, 8)
    flags = []
    original = model.compute_loss
    def capture(**kwargs):
        flags.append(kwargs.get('compute_style_metrics', True))
        return original(**kwargs)
    model.compute_loss = capture
    out = unrolled_flow_loss(model=model, x_window=x, style_label=labels, steps=3,
        prev_latent_0=latents[:, 0], target_latents=latents[:, 1:], recon_targets=x[:, 1:, 84:],
        recon_weight=.1, style_hist_weight=.02, style_entropy_weight=.003, solver_steps=2)
    assert flags == [True, False, False]
    torch.testing.assert_close(out['loss'], out['flow_loss'] + .1*out['recon_loss']
                               + .02*out['style_hist_loss'] + .003*out['style_entropy'])


class SharedStyleTableTests(unittest.TestCase):
    """Hoisting the selector table once per window is bit-identical and cuts table forwards."""

    def _run(self, *, reuse, seed=11):
        from test_controller_repair import controller as build
        model = build().train()
        torch.manual_seed(seed)
        window = torch.randn(8, 4, 375) * 0.1
        latents = model.frame_latent(window.reshape(-1, 375)).reshape(8, 4, -1)
        targets = latents[:, 1:].contiguous()
        recon = model.vqvae.decode(targets.reshape(-1, targets.size(-1)), from_indices=False).reshape(8, 3, -1)
        calls = []
        original = model.selector_output_all_styles
        def counting():
            calls.append(1)
            return original()
        model.selector_output_all_styles = counting
        torch.manual_seed(seed + 1)
        out = unrolled_flow_loss(
            model=model, x_window=window, style_label=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
            prev_latent_0=latents[:, 0], target_latents=targets, recon_targets=recon,
            steps=3, style_hist_weight=.02, style_entropy_weight=.003, reuse_style_table=reuse)
        out["loss"].backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return float(out["loss"]), grads, len(calls)

    def test_reuse_is_equivalent_and_reduces_calls(self):
        loss_reuse, grads_reuse, calls_reuse = self._run(reuse=True)
        loss_plain, grads_plain, calls_plain = self._run(reuse=False)
        # Forward loss is identical (same deterministic table); gradients differ only by
        # floating-point reassociation, because a shared table accumulates three uses through
        # one node while the per-step path accumulates through three separate nodes.
        self.assertAlmostEqual(loss_reuse, loss_plain, places=7)
        torch.testing.assert_close(grads_reuse, grads_plain, rtol=1e-6, atol=1e-8)
        self.assertEqual(calls_reuse, 1)
        self.assertEqual(calls_plain, 3)
