"""V1 physical-position second-order loss (posacc): analytic, gradient, freeze invariants.

The plan requires: constant-velocity ~0, matching GT acceleration == 0, oscillation and
time misalignment > 0, non-zero decoder gradients, E/Q/EMA bitwise unchanged under the
decoder-only freeze (verified through model.train(), not just requires_grad), seq<3 safe,
cm/index correctness (84-d trajectory offset), and FP32 behaviour.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from arch.basic_vqvae_375to291 import (
    POSITION_MOTION_END,
    POSITION_MOTION_START,
    BasicVQVAE375to291,
    motion_positions_cm,
    physical_position_acceleration_loss,
    second_difference,
)


def _build() -> BasicVQVAE375to291:
    torch.manual_seed(0)
    return BasicVQVAE375to291(
        input_dim=375, traj_dim=84, motion_dim=291, input_mode="traj_motion",
        hidden_dim=32, encoder_layers=2, decoder_hidden_dim=32, decoder_layers=2,
        num_latents=4, code_dim=4, num_embeddings=32, commitment_cost=0.25,
        use_rotation_trick=True, kmeans_init=False, dead_code_threshold=0,
    )


def _norm_375():
    mean = torch.zeros(375)
    std = torch.ones(375)
    return mean, std


def _make_motion(positions_m: torch.Tensor) -> torch.Tensor:
    """[T,24,3] positions in meters -> [B=1,T,291] normalized motion with zero stats."""
    t = positions_m.size(0)
    motion = torch.zeros(1, t, 291)
    motion[..., POSITION_MOTION_START:POSITION_MOTION_END] = positions_m.reshape(1, t, 72)
    return motion


def test_constant_velocity_second_difference_is_zero_at_function_level():
    mean, std = torch.zeros(375), torch.ones(375)
    t = 8
    step_m = 1.0 / 60.0 / 100.0            # 1 cm/s at 60 fps, in meters
    linear = (torch.arange(t, dtype=torch.float32) * step_m)[:, None, None].expand(t, 24, 3)
    target = _make_motion(linear)
    # prediction == target -> both sides have identical (zero) second difference
    smooth, absolute = physical_position_acceleration_loss(
        target.clone(), target, motion_mean=mean, motion_std=std)
    assert float(smooth.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(absolute.detach()) == pytest.approx(0.0, abs=1e-6)

    # a constant +1cm offset also has zero second difference on both sides
    offset = (linear + 0.01)
    smooth2, absolute2 = physical_position_acceleration_loss(
        _make_motion(offset), target, motion_mean=mean, motion_std=std)
    assert float(smooth2.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(absolute2.detach()) == pytest.approx(0.0, abs=1e-6)


def test_acceleration_mismatch_and_oscillation_are_positive_and_ordered():
    mean, std = torch.zeros(375), torch.ones(375)
    t = 6
    time = torch.arange(t, dtype=torch.float32)
    accel_cm = 3.0
    pos_cm = 0.5 * accel_cm * (time / 60.0) ** 2
    base = _make_motion((pos_cm / 100.0)[:, None, None].expand(t, 24, 3))
    match = physical_position_acceleration_loss(base.clone(), base, motion_mean=mean, motion_std=std)
    assert float(match[0].detach()) == pytest.approx(0.0, abs=1e-6)

    # time misalignment: roll by one frame -> genuine acceleration mismatch
    shifted = torch.roll(base, shifts=-1, dims=1)
    shifted[:, -1] = shifted[:, -2]
    misaligned = physical_position_acceleration_loss(shifted, base, motion_mean=mean, motion_std=std)
    assert float(misaligned[1].detach()) > 1e-3

    # alternating +/-1cm ripple has a large second difference -> strictly worse
    ripple_cm = torch.where(time % 2 == 0, 1.0, -1.0)
    ringing_pos = pos_cm + ripple_cm
    ringing = physical_position_acceleration_loss(
        _make_motion((ringing_pos / 100.0)[:, None, None].expand(t, 24, 3)), base,
        motion_mean=mean, motion_std=std)
    assert float(ringing[1].detach()) > float(misaligned[1].detach())


def test_absolute_cm_error_grows_monotonically_with_ripple_amplitude():
    mean, std = torch.zeros(375), torch.ones(375)
    t = 6
    time = torch.arange(t, dtype=torch.float32)
    base = _make_motion(torch.zeros(t, 24, 3))
    errors = []
    for amp in (0.0, 0.5, 1.0, 2.0):
        ripple_cm = amp * (time % 2)
        pred = _make_motion((ripple_cm / 100.0)[:, None, None].expand(t, 24, 3))
        _, absolute = physical_position_acceleration_loss(pred, base, motion_mean=mean, motion_std=std)
        errors.append(float(absolute.detach()))
    assert errors[0] == pytest.approx(0.0, abs=1e-6)
    assert errors == sorted(errors)
    assert errors[-1] > 0.0


def test_cm_conversion_uses_the_84_offset_and_meter_scale():
    norm = torch.zeros(1, 2, 291)
    norm[..., POSITION_MOTION_START] = 0.01  # 1 cm in meters
    mean375, std375 = torch.zeros(375), torch.ones(375)
    cm = motion_positions_cm(norm, mean375, std375)
    assert cm.shape == (1, 2, 24, 3)
    assert float(cm[0, 0, 0, 0]) == pytest.approx(1.0, rel=1e-6)
    # 291-wide and 72-wide stats must agree with the 375-wide form
    mean291, std291 = torch.zeros(291), torch.ones(291)
    cm291 = motion_positions_cm(norm, mean291, std291)
    torch.testing.assert_close(cm291, cm)
    mean72, std72 = torch.zeros(72), torch.ones(72)
    block = norm[..., POSITION_MOTION_START:POSITION_MOTION_END]
    cm72 = motion_positions_cm(norm, mean72, std72)
    torch.testing.assert_close(cm72, cm)
    # consistency: normalizing by a std then converting with that same std is identity
    normalized = norm[..., POSITION_MOTION_START:POSITION_MOTION_END] / 2.0
    renorm = torch.zeros_like(norm)
    renorm[..., POSITION_MOTION_START:POSITION_MOTION_END] = normalized
    scaled = motion_positions_cm(renorm, mean375, std375.clone().fill_(2.0))
    torch.testing.assert_close(scaled, cm)
    with pytest.raises(ValueError):
        motion_positions_cm(norm[..., :100], mean375, std375)


def test_second_difference_requires_three_steps():
    with pytest.raises(ValueError):
        second_difference(torch.zeros(1, 2, 3))
    values = torch.tensor([[[0.0], [1.0], [3.0], [6.0]]])  # increments 1,2,3 -> d2 = 1,1
    torch.testing.assert_close(second_difference(values), torch.tensor([[[1.0], [1.0]]]))


def test_seq_len_two_with_posacc_weight_raises():
    model = _build()
    x = torch.randn(2, 2, 375)
    with pytest.raises(ValueError, match="seq_len >= 3|at least 3 steps"):
        model.compute_loss(x, posacc_weight=1.0,
                           motion_mean=torch.zeros(375), motion_std=torch.ones(375))


def test_posacc_weight_zero_is_inert_and_requires_stats_when_positive():
    model = _build().eval()
    x = torch.randn(2, 4, 375)
    zero = model.compute_loss(x, posacc_weight=0.0)
    default = model.compute_loss(x)
    assert float(zero["loss"].detach()) == pytest.approx(float(default["loss"].detach()), rel=0, abs=0)
    assert float(zero["posacc_loss"]) == 0.0
    with pytest.raises(ValueError, match="motion_mean and motion_std"):
        model.compute_loss(x, posacc_weight=1.0)
    with pytest.raises(ValueError):
        model.compute_loss(x, posacc_weight=-1.0)
    with pytest.raises(ValueError):
        model.compute_loss(x, posacc_weight=1.0, posacc_beta_cm=0.0,
                           motion_mean=torch.zeros(375), motion_std=torch.ones(375))


def test_decoder_only_freeze_keeps_posacc_gradients_on_the_decoder_only():
    model = _build().train()
    model.set_representation_frozen(True)
    before = copy.deepcopy(model.state_dict())
    mean, std = _norm_375()
    x = torch.randn(3, 5, 375)
    out = model.compute_loss(x, posacc_weight=1.0, motion_mean=mean, motion_std=std)
    out["posacc_loss"].backward()
    decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
    assert decoder_grads and any(g.abs().sum() > 0 for g in decoder_grads)
    assert all(p.grad is None for p in model.encoder.parameters())
    # E/Q/EMA bitwise unchanged: forward ran in train mode with the frozen contract
    assert model.quantizer.training is False and model.encoder.training is False
    after = model.state_dict()
    for key, value in before.items():
        if key.startswith("encoder.") or key.startswith("quantizer."):
            torch.testing.assert_close(after[key], value, rtol=0, atol=0)


def test_train_mode_forward_still_updates_ema_without_freeze_and_posacc_reports_fp32():
    model = _build().train()
    mean, std = _norm_375()
    x = torch.randn(3, 5, 375)
    before = model.quantizer.embedding_weight.clone()
    out = model.compute_loss(x, posacc_weight=1.0, motion_mean=mean, motion_std=std)
    assert out["posacc_loss"].dtype == torch.float32
    assert float(out["posacc_abs_cm"]) >= 0.0
    assert not torch.equal(before, model.quantizer.embedding_weight)  # EMA path is live

    frozen = _build().train()
    frozen.set_representation_frozen(True)
    frozen.train()  # re-assert after a train() call
    before_frozen = frozen.quantizer.embedding_weight.clone()
    frozen.compute_loss(x, posacc_weight=1.0, motion_mean=mean, motion_std=std)
    torch.testing.assert_close(frozen.quantizer.embedding_weight, before_frozen, rtol=0, atol=0)


