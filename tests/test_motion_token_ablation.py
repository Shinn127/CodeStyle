"""Motion-token ablation: 5-token forward path, 6->5 state-dict remap, legacy compatibility."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arch.basic_vqvae_375to291 import BasicVQVAE375to291
from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config
from training.train_flowmatch_vqvae_controller import remap_token_state_dict
from test_controller_repair import controller


def build(*, use_motion_token):
    torch.manual_seed(31)
    vq = BasicVQVAE375to291(hidden_dim=16, decoder_hidden_dim=16, encoder_layers=2,
        decoder_layers=2, num_latents=4, code_dim=2, num_embeddings=16,
        kmeans_init=False, dead_code_threshold=0)
    model = build_flowmatch_controller_from_config(dict(prediction_mode='target',
        flow_loss_space='normalized', use_style_code_selector=True,
        style_slot_indices=[0, 1, 2, 3], style_dim=8, style_selector_hidden_dim=16,
        style_selector_topk=2, flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32,
        dropout=0., inference_requantize=True, use_motion_token=use_motion_token),
        vqvae=vq, num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return model


def test_default_config_keeps_six_tokens_and_motion_projection():
    model = build(use_motion_token=True)
    assert model.use_motion_token is True
    assert model.field.num_tokens == 6
    assert model.field.motion_proj is not None
    assert model.field.position_embeddings.shape[1] == 6
    # legacy config without the key resolves to the original 6-token layout
    legacy = copy.deepcopy(dict(use_style_code_selector=False))
    legacy_model = build_flowmatch_controller_from_config(
        dict(prediction_mode='target', flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32, style_dim=8),
        vqvae=model.vqvae, num_styles=3)
    assert legacy_model.field.num_tokens == 6


def test_five_token_model_forwards_and_backpropagates():
    model = build(use_motion_token=False).train()
    assert model.field.num_tokens == 5
    assert model.field.motion_proj is None
    assert model.field.position_embeddings.shape[1] == 5
    x = torch.randn(3, 4, 375) * .1
    labels = torch.tensor([0, 1, 2])
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
        flow_weight=1., solver_steps=2)
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.field.parameters())
    # sampled inference runs end to end and decodes motion
    model.eval()
    latent = model.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=4)
    motion, requantized = model.feedback_state(latent, requantize=True)
    assert latent.shape == (3, 8) and motion.shape == (3, 291) and requantized.shape == (3, 8)


def test_six_to_five_remap_drops_motion_proj_and_keeps_other_tokens_identical():
    six = build(use_motion_token=True)
    remapped = remap_token_state_dict(copy.deepcopy(six.state_dict()), drop_motion_token=True)
    assert not any(k.startswith('field.motion_proj.') for k in remapped)
    keep = torch.tensor([0, 1, 3, 4, 5])
    torch.testing.assert_close(
        remapped['field.position_embeddings'],
        six.state_dict()['field.position_embeddings'].index_select(1, keep), atol=0, rtol=0)
    five = build(use_motion_token=False)
    missing, unexpected = five.load_state_dict(remapped, strict=False)
    assert not missing and not unexpected
    # every non-motion tensor is bit-identical after the remap
    six_state, five_state = six.state_dict(), five.state_dict()
    for key, value in five_state.items():
        if key == 'field.position_embeddings':
            continue
        if key.startswith('field.motion_proj.'):
            continue
        torch.testing.assert_close(value, six_state[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='6-token'):
        remap_token_state_dict(copy.deepcopy(six.state_dict()), drop_motion_token=False)


def test_remap_rejects_five_token_source():
    five = build(use_motion_token=False)
    with pytest.raises(ValueError, match='position embeddings'):
        remap_token_state_dict(copy.deepcopy(five.state_dict()), drop_motion_token=True)


def test_eval_loader_builds_five_token_model_from_config():
    from Utils.motion_eval import _build_flowmatch_from_checkpoint
    import tempfile
    five = build(use_motion_token=False).eval()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vq_path = tmp / 'vq.pt'
        torch.save({'model': five.vqvae.state_dict(), 'args': dict(hidden_dim=16,
            decoder_hidden_dim=16, encoder_layers=2, decoder_layers=2, num_latents=4,
            code_dim=2, num_embeddings=16, kmeans_init=False, dead_code_threshold=0)}, vq_path)
        ckpt_path = tmp / 'flow.pt'
        torch.save({'model': five.state_dict(), 'args': {
            'prediction_mode': 'target', 'flow_loss_space': 'normalized',
            'use_style_code_selector': True, 'style_slot_indices': [0, 1, 2, 3],
            'style_dim': 8, 'style_selector_hidden_dim': 16, 'style_selector_topk': 2,
            'flow_hidden_dim': 16, 'flow_layers': 1, 'flow_ffn_dim': 32, 'dropout': 0.,
            'inference_requantize': True, 'use_motion_token': False,
            'style_selector_temperature': 1.0, 'style_proto_normalization': 'none',
            'vqvae_ckpt': str(vq_path),
        }}, ckpt_path)
        loaded = _build_flowmatch_from_checkpoint(
            ckpt_path=ckpt_path, num_styles=3, vqvae_ckpt_override=None)
        assert loaded.field.num_tokens == 5
        assert loaded.field.motion_proj is None


def build_skip_cat(*, use_motion_token=True):
    torch.manual_seed(31)
    vq = BasicVQVAE375to291(hidden_dim=16, decoder_hidden_dim=16, encoder_layers=2,
        decoder_layers=2, num_latents=4, code_dim=2, num_embeddings=16,
        kmeans_init=False, dead_code_threshold=0)
    model = build_flowmatch_controller_from_config(dict(prediction_mode='target',
        flow_loss_space='normalized', use_style_code_selector=True,
        style_slot_indices=[0, 1, 2, 3], style_dim=8, style_selector_hidden_dim=16,
        style_selector_topk=2, flow_hidden_dim=16, flow_layers=2, flow_ffn_dim=32,
        dropout=0., inference_requantize=True, field_arch='skip_cat_mlp',
        use_motion_token=use_motion_token), vqvae=vq, num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8), transport_std=torch.ones(8) * .5)
    model.set_style_code_prior(torch.ones(3, 4, 16) / 16)
    return model


def test_skip_cat_field_default_config_and_forward_backward():
    from arch import ConditionalSkipCatMLPField
    model = build_skip_cat().train()
    assert isinstance(model.field, ConditionalSkipCatMLPField)
    assert model.field_arch == 'skip_cat_mlp'
    assert not hasattr(model.field, 'position_embeddings')
    x = torch.randn(3, 4, 375) * .1
    labels = torch.tensor([0, 1, 2])
    out = model.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=labels,
        flow_weight=1., solver_steps=2)
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.field.parameters())
    # sampled inference + feedback decode end to end
    model.eval()
    latent = model.sample_latent(x_curr=x[:, 0], style_label=labels, num_steps=4)
    motion, requantized = model.feedback_state(latent, requantize=True)
    assert latent.shape == (3, 8) and motion.shape == (3, 291) and requantized.shape == (3, 8)


def test_skip_cat_conditions_cached_once_per_euler_solve():
    model = build_skip_cat().eval()
    calls = []
    hook = model.field.net.register_forward_hook(lambda *args: calls.append(1))
    x = torch.randn(2, 375) * .1
    model.sample_latent(x_curr=x, style_label=torch.tensor([0, 1]), num_steps=8)
    hook.remove()
    assert len(calls) == 8  # the MLP runs per Euler step; the flat condition is reused


def test_skip_cat_five_token_variant():
    model = build_skip_cat(use_motion_token=False).eval()
    assert model.field.use_motion_token is False
    x = torch.randn(2, 375) * .1
    latent = model.sample_latent(x_curr=x, style_label=torch.tensor([0, 1]), num_steps=2)
    assert latent.shape == (2, 8) and bool(torch.isfinite(latent).all())


def test_transformer_and_skip_cat_weights_do_not_cross_load():
    transformer = build(use_motion_token=True)
    skip = build_skip_cat()
    with pytest.raises(RuntimeError, match='Incompatible|size mismatch|Missing key'):
        skip.load_state_dict(transformer.state_dict(), strict=True)
