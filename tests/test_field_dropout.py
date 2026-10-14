"""Field/selector dropout separation (--dropout vs --field-dropout).

The 2026-10-14 revision isolates the field dropout so a stochastic transformer can be
trained without perturbing the deterministic selector fast paths:
* ``dropout`` still configures the selector and remains the field fallback (legacy runs);
* ``field_dropout`` overrides only the field;
* ``StyleCodeSelector.dropout_prob`` is now actually set, so the batched style-table
  fast paths gate themselves off when the selector is stochastic.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_controller_repair import controller


BASE = dict(prediction_mode='target', flow_loss_space='normalized', use_style_code_selector=True,
            style_slot_indices=[0, 1, 2, 3], style_dim=8, style_selector_hidden_dim=16,
            style_selector_topk=2, flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32,
            inference_requantize=True)


def build(**overrides):
    from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
    model = build_flowmatch_controller_from_config(dict(BASE, **overrides),
                                                   vqvae=copy.deepcopy(controller().vqvae),
                                                   num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return model


def _dropouts(module):
    return [m.p for m in module.modules() if isinstance(m, torch.nn.Dropout)]


def test_legacy_single_knob_keeps_field_inheriting_selector_dropout():
    model = build(dropout=0.2)
    assert set(_dropouts(model.style_selector)) == {0.2}
    assert set(_dropouts(model.field)) == {0.2}
    assert model.field_dropout == 0.2
    assert model.style_selector.dropout_prob == 0.2


def test_field_dropout_override_isolates_the_transformer():
    model = build(dropout=0.0, field_dropout=0.2)
    assert set(_dropouts(model.style_selector)) == {0.0}
    assert set(_dropouts(model.field)) == {0.2}
    assert model.field_dropout == 0.2
    # Selector stays deterministic: its fast-path gate must remain open.
    assert model.style_selector.dropout_prob == 0.0


def test_selector_dropout_prob_gates_the_style_table_fast_paths():
    deterministic = build(dropout=0.0)
    stochastic = build(dropout=0.2)
    assert deterministic.style_selector.dropout_prob == 0.0
    assert stochastic.style_selector.dropout_prob == 0.2

    # With >= 2*num_styles rows the deterministic selector builds the batched style table;
    # a stochastic selector must fall back to the per-sample path instead.
    x = torch.randn(8, 4, 375) * .1
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    def table_calls(model):
        seen = {'n': 0}
        original = model.selector_output_all_styles

        def spy():
            seen['n'] += 1
            return original()

        model.selector_output_all_styles = spy
        model.eval()
        model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels)
        return seen['n']

    assert table_calls(deterministic) == 1
    assert table_calls(stochastic) == 0


@pytest.mark.parametrize('value', [-0.1, 1.0, 1.5, float('nan')])
def test_invalid_field_dropout_is_rejected(value):
    with pytest.raises(ValueError, match="field_dropout"):
        build(field_dropout=value)


def test_invalid_selector_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        build(dropout=1.0)


def test_trainer_flag_and_config_passthrough_are_wired():
    import training.train_flowmatch_vqvae_controller as trainer
    source = Path(trainer.__file__).read_text()
    assert '"--field-dropout"' in source
    assert '"field_dropout"' in source  # resume field + saved config
    assert "args.field_dropout = float(args.dropout)" in source  # fallback resolution
