"""Per-layer style FiLM on the transformer generation query: identity, gradients, caching.

Covers the 2026-10-13 revision: ``--use-style-film`` adds a zero-initialized per-layer
(scale, shift) modulation of the generation query token, driven by e_s. A fresh run must be
mathematically (and bitwise) identical to the unmodulated field, the modulation must receive
FM gradients while the VQ stays frozen, coefficients must be computed once per Euler solve,
and the disabled path must stay exactly the legacy encoder call.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller


FILM_KEYS = dict(prediction_mode='target', flow_loss_space='normalized', use_style_code_selector=True,
                 style_slot_indices=[0, 1, 2, 3], style_dim=8, style_selector_hidden_dim=16,
                 style_selector_topk=2, flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32,
                 dropout=0., inference_requantize=True)


def film_controller(*, use_style_film=True, history_conditioning=False, seed=31):
    """Unmodulated fixture (test_controller_repair.controller) plus the same field with FiLM."""
    from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
    torch.manual_seed(seed)
    base = controller()
    film = build_flowmatch_controller_from_config(
        dict(FILM_KEYS, use_style_film=use_style_film, history_conditioning=history_conditioning),
        vqvae=copy.deepcopy(base.vqvae), num_styles=3)
    missing, unexpected = film.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    allowed_prefixes = ('field.style_film.',) + (('history_delta_proj.',) if history_conditioning else ())
    assert all(k.startswith(allowed_prefixes) for k in missing), missing
    film.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    film.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return base, film


def controller_pair(layers, seed=31):
    """Two controllers from one config, differing only in use_style_film, same seed."""
    from arch.basic_vqvae_375to291 import BasicVQVAE375to291
    from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config

    def build(film_on):
        torch.manual_seed(seed)
        vq = BasicVQVAE375to291(hidden_dim=16, decoder_hidden_dim=16, encoder_layers=2,
            decoder_layers=2, num_latents=4, code_dim=2, num_embeddings=16,
            kmeans_init=False, dead_code_threshold=0)
        model = build_flowmatch_controller_from_config(
            dict(FILM_KEYS, flow_layers=layers, use_style_film=film_on),
            vqvae=vq, num_styles=3)
        model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
        model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
        return model

    return build(False), build(True)


def _inputs(model, batch=3):
    x = torch.randn(batch, 4, 375) * .1
    return x, torch.tensor([0, 1, 2][:batch])


def test_zero_init_film_is_bitwise_identity_in_loss_field_and_gradients():
    base, film = film_controller()
    base.train(); film.train()
    x, labels = _inputs(film)
    # Only the final projection is zero-initialized; the identity property follows from it,
    # not from the first layer being zero.
    assert all((p == 0).all().item() for p in [film.field.style_film.proj[-1].weight,
                                               film.field.style_film.proj[-1].bias])
    with torch.no_grad():
        e_s = torch.randn(3, film.style_dim)
        scale, shift = film.field.prepare_film(e_s)
    assert (scale == 0).all() and (shift == 0).all()

    torch.manual_seed(5)
    a = base.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    torch.manual_seed(5)
    b = film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    assert torch.equal(a['loss'], b['loss'])
    a['loss'].backward(); b['loss'].backward()

    base_grads = {n: p.grad for n, p in base.field.named_parameters() if p.grad is not None}
    film_grads = {n: p.grad for n, p in film.field.named_parameters()
                  if p.grad is not None and not n.startswith('style_film.')}
    assert set(base_grads) == set(film_grads)
    assert all(torch.equal(base_grads[n], film_grads[n]) for n in base_grads)
    # AdaLN-Zero ordering: the zero-initialized output layer takes the first gradient step
    # (proj.0 is behind a zero weight and stays at zero gradient until then).
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in film.field.style_film.named_parameters() if n.startswith('proj.2'))
    from torch.optim import AdamW
    optimizer = AdamW([p for p in film.field.style_film.parameters()], lr=.01)
    optimizer.step()
    film.zero_grad()
    torch.manual_seed(5)
    film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in film.field.style_film.named_parameters() if n.startswith('proj.0'))


def test_nonzero_film_changes_predictions_and_keeps_vq_frozen():
    _, film = film_controller()
    film.train()
    x, labels = _inputs(film)
    torch.manual_seed(5)
    before = film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    with torch.no_grad():
        film.field.style_film.proj[-1].bias.fill_(0.1)
    torch.manual_seed(5)
    after = film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    assert not torch.isclose(before['loss'], after['loss'], atol=1e-7)
    after['loss'].backward()
    assert all(not p.requires_grad and p.grad is None for p in film.vqvae.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in film.field.style_film.parameters())


def test_scale_is_bounded_and_shift_is_free():
    _, film = film_controller()
    _, _, e_s, _ = film.encode_conditions(x_curr=torch.randn(4, 375) * .1,
                                          style_label=torch.tensor([0, 1, 2, 0]),
                                          return_style_details=True)
    with torch.no_grad():
        film.field.style_film.proj[-1].bias.fill_(100.)
        scale, shift = film.field.prepare_film(e_s)
        assert (scale.abs() <= 1.0).all()
        assert (scale.abs() > .99).all()  # saturated but inside the bound
        assert (shift.abs() > .99).all()  # shift is not squashed
        film.field.style_film.proj[-1].bias.zero_()
        scale, shift = film.field.prepare_film(e_s)
        assert (scale == 0).all() and (shift == 0).all()


def test_film_coefficients_are_cached_once_per_euler_solve():
    _, film = film_controller()
    film.eval()
    x, labels = _inputs(film)
    calls = {'n': 0}
    original = film.prepare_style_film

    def spy(e_s):
        calls['n'] += 1
        return original(e_s)

    film.prepare_style_film = spy
    film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels, solver_steps=8)
    assert calls['n'] == 1
    calls['n'] = 0
    z0 = torch.zeros(3, film.latent_dim)
    film.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=6, noise_std=0.0, z0=z0)
    assert calls['n'] == 1


def test_sampling_path_is_bitwise_identical_at_zero_init():
    base, film = film_controller()
    base.eval(); film.eval()
    x, labels = _inputs(film)
    z0 = torch.zeros(3, film.latent_dim)
    with torch.no_grad():
        a = base.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=4, noise_std=0.0, z0=z0)
        b = film.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=4, noise_std=0.0, z0=z0)
    assert torch.equal(a, b)


def test_explicit_and_auto_resolved_film_agree_and_validation_rejects_bad_shapes():
    _, film = film_controller()
    film.eval()
    x, labels = _inputs(film)
    _, _, e_s, _ = film.encode_conditions(x_curr=x[:, 0], style_label=labels, return_style_details=True)
    z_t = torch.zeros(3, film.latent_dim); z_prev = z_t.clone(); t = torch.full((3, 1), .4)
    e_c = torch.zeros(3, film.motion_dim); e_traj = torch.zeros(3, film.traj_dim)
    with torch.no_grad():
        scale, shift = film.field.prepare_film(e_s)
        explicit = film.field(z_t=z_t, z_prev=z_prev, t=t, content=e_c, traj=e_traj, style=e_s,
                              film=(scale, shift))
        automatic = film.field(z_t=z_t, z_prev=z_prev, t=t, content=e_c, traj=e_traj, style=e_s)
    assert torch.equal(explicit, automatic)
    with pytest.raises(ValueError, match="film coefficients"):
        film.field(z_t=z_t, z_prev=z_prev, t=t, content=e_c, traj=e_traj, style=e_s,
                   film=(scale[:2], shift))

    _, off = film_controller(use_style_film=False)
    with pytest.raises(ValueError, match="without StyleFiLM"):
        off.field(z_t=z_t, z_prev=z_prev, t=t, content=e_c, traj=e_traj, style=e_s,
                  film=(scale, shift))


def test_history_conditioning_combines_with_film():
    _, film = film_controller(history_conditioning=True)
    film.train()
    x = torch.randn(3, 4, 375) * .1
    labels = torch.tensor([0, 1, 2])
    torch.manual_seed(6)
    out = film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
                            history_latent=film.frame_latent(x[:, 0]))
    out['loss'].backward()
    film_params = [(n, p) for n, p in film.field.style_film.named_parameters()]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in film_params
               if n.startswith('proj.0') or n.startswith('proj.2'))


def test_film_rejected_on_skip_cat_mlp_and_disabled_path_is_untouched():
    from arch.flowmatch_vqvae_controller import ConditionalSkipCatMLPField
    with pytest.raises(ValueError, match="transformer field"):
        ConditionalSkipCatMLPField(latent_dim=8, motion_dim=291, traj_dim=84, style_dim=8,
                                   hidden_dim=16, time_embed_dim=8, num_layers=2,
                                   use_style_film=True)
    from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
    with pytest.raises(ValueError, match="transformer field"):
        build_flowmatch_controller_from_config(
            dict(FILM_KEYS, field_arch='skip_cat_mlp', use_style_film=True),
            vqvae=controller().vqvae, num_styles=3)


def test_disabled_run_has_no_film_parameters_and_legacy_forward_path():
    base, off = film_controller(use_style_film=False)
    assert off.field.style_film is None
    assert not any('style_film' in k for k in off.state_dict())
    assert off.prepare_style_film(torch.randn(2, off.style_dim)) is None
    off.train()
    x, labels = _inputs(off)
    torch.manual_seed(5)
    a = base.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    torch.manual_seed(5)
    b = off.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    assert torch.equal(a['loss'], b['loss'])


def test_multilayer_pair_same_seed_shared_weights_and_loss_identity():
    """With the FiLM module constructed last, a same-seed film/non-film pair shares all legacy
    weights bit-exactly and the zero-initialized film is loss-identical on a multi-layer field."""
    base, film = controller_pair(layers=3)
    assert len(film.field.encoder.layers) == 3
    sd_a, sd_b = base.state_dict(), film.state_dict()
    shared = [k for k in sd_a if k in sd_b and not k.startswith('field.style_film.')]
    assert shared and all(torch.equal(sd_a[k], sd_b[k]) for k in shared)
    base.train(); film.train()
    x, labels = _inputs(film)
    torch.manual_seed(5)
    a = base.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    torch.manual_seed(5)
    b = film.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
    assert torch.equal(a['loss'], b['loss'])
    # The coefficients are per layer.
    with torch.no_grad():
        scale, shift = film.field.prepare_film(torch.randn(2, film.style_dim))
    assert scale.shape == (2, 3, film.hidden_dim) and shift.shape == (2, 3, film.hidden_dim)


def test_trainer_flag_resume_field_and_init_allowlist_are_wired():
    """The trainer CLI flag, the resume comparison field and the init allowlist exist."""
    import training.train_flowmatch_vqvae_controller as trainer
    source = Path(trainer.__file__).read_text()
    assert '"--use-style-film"' in source
    assert '"use_style_film"' in source          # resume_fields entry
    assert "'use_style_film'" in source          # legacy default + config passthrough
    assert "field.style_film." in source         # init-ckpt missing-key allowance
