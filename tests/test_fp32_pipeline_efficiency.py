"""Cache invalidation, scalar-compatible RNG, and FP32 attention regressions."""
import copy
import math

import numpy as np
import pytest
import torch

from arch.flowmatch_vqvae_controller import HistogramCodeSelector
from Utils.style_code_prior import TrainStyleRecipePool
from test_controller_repair import controller, loss_args
from test_recipe_training import pool_fixture


def test_slot_identity_cache_updates_after_buffer_copy_and_load():
    model = controller()
    old = model.selected_style_slots
    model.style_slot_indices.copy_(torch.tensor([3,2,1,0]))
    assert model.selected_style_slots == tuple(reversed(old))
    state = copy.deepcopy(model.state_dict())
    state['style_slot_indices'] = torch.tensor(old)
    model.load_state_dict(state)
    assert model.selected_style_slots == old


def test_frozen_geometry_cache_refreshes_after_codebook_change():
    selector = HistogramCodeSelector(2,4)
    book = torch.randn(16,2); histogram = torch.rand(3,4,16)
    selector(histogram, book); previous = selector._geometry_cache
    selector(histogram, book)
    assert selector._geometry_cache is previous
    book[0].add_(2)
    selector(histogram, book)
    assert selector._geometry_cache is not previous
    e = (book-book.mean(0))/book.std(0,unbiased=False).clamp_min(1e-6)
    torch.testing.assert_close(selector._geometry_cache,torch.cat((e,e.square()),-1))


def test_prior_entropy_cache_and_kl_keep_distinct_clamp_semantics():
    model = controller()
    prior = torch.rand(3,4,16); prior[...,0] = 1e-10
    model.set_style_code_prior(prior)
    p = model.style_prior_probs
    torch.testing.assert_close(model._prior_entropy_by_style(),-(p*p.clamp_min(1e-8).log()).sum(-1).mean(-1))
    torch.testing.assert_close(model._prior_entropy_by_style(kl=True),-(p*p.clamp_min(1e-12).log()).sum(-1).mean(-1))
    previous = model._prior_entropy_by_style()
    model.set_style_code_prior(torch.ones_like(p))
    assert model._prior_entropy_by_style() is not previous


@pytest.mark.parametrize('bounds', [(3,64),(7,13),(64,),(1,)])
def test_batched_small_integer_draws_preserve_scalar_values_and_rng(bounds):
    torch.manual_seed(42)
    expected = [[torch.randint(n,()).item() for n in bounds] for _ in range(300)]
    state = torch.get_rng_state()
    torch.manual_seed(42)
    draws = torch.randint(math.lcm(*bounds),(300,len(bounds)))
    assert (draws % torch.tensor(bounds)).tolist() == expected
    assert torch.equal(torch.get_rng_state(),state)


def test_recipe_bank_batched_choices_equal_scalar_reference(tmp_path,monkeypatch):
    model,_,cache = pool_fixture(tmp_path)
    kwargs = dict(db=tmp_path,model=model,cache=cache,mean=np.zeros(375),std=np.ones(375),
                  frames=4,exemplar_choices=(1,3,5),bank_size=64)
    fast,reference = TrainStyleRecipePool(**kwargs),TrainStyleRecipePool(**kwargs)
    labels = torch.tensor([0,1,2]*4); exclusions = [{int(y)*6} for y in labels]
    torch.manual_seed(93)
    actual = fast.sample(labels,probability=1.,require_all=True,per_sample_exclusions=exclusions)[0]
    actual_state = torch.get_rng_state()
    original = torch.randint
    def scalar_draw(high,size,**kw):
        if len(size) == 2:
            return torch.stack([original(high,(),**kw) for _ in range(math.prod(size))]).reshape(size)
        return original(high,size,**kw)
    monkeypatch.setattr(torch,'randint',scalar_draw)
    torch.manual_seed(93)
    expected = reference.sample(labels,probability=1.,require_all=True,per_sample_exclusions=exclusions)[0]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.equal(torch.get_rng_state(),actual_state)


@pytest.mark.parametrize('mode',['target','velocity'])
def test_endpoint_supervision_uses_one_field_call_without_solver_decoder(mode,monkeypatch):
    model = controller(mode).train()
    x,labels = loss_args()
    latents = model.frame_latent(x[:,:2].flatten(0,1)).reshape(3,2,8)
    def forbidden(*a,**k):
        raise AssertionError('Endpoint retrieval must not integrate or decode')
    monkeypatch.setattr(model,'_integrate',forbidden)
    monkeypatch.setattr(model.vqvae,'decode',forbidden)
    calls = []
    hook = model.field.register_forward_hook(lambda *a:calls.append(1))
    out = model.compute_loss(x_curr=x[:,0],x_next=x[:,1],style_label=labels,
        prev_latent=latents[:,0],target_latent=latents[:,1],return_endpoint=True)
    out['endpoint_latent'].square().mean().backward()
    hook.remove()
    assert len(calls)==1 and out['endpoint_latent'].shape==(3,8)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.field.parameters())
    assert all(p.grad is None for p in model.vqvae.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.gpu
def test_six_token_fp32_math_attention_loss_gradient_and_rng():
    model = controller().cuda().train(); reference = copy.deepcopy(model)
    model.field.fp32_math_attention = True
    x,labels = loss_args(); x,labels = x.cuda(),labels.cuda()
    outputs = []
    for current in (reference,model):
        torch.manual_seed(42)
        out = current.compute_loss(x_curr=x[:,0],x_next=x[:,1],style_label=labels,style_hist_weight=.02)
        out['loss'].backward(); outputs.append((out,torch.cuda.get_rng_state()))
    torch.testing.assert_close(outputs[0][0]['loss'],outputs[1][0]['loss'],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(outputs[0][1],outputs[1][1],atol=0,rtol=0)
    for a,b in zip(reference.parameters(),model.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=2e-4)
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.gpu
def test_actual_cuda_entry_epoch_resume_matches_continuous(tmp_path,monkeypatch):
    # Exercise the real parser/optimizer/save/load path with disposable tiny data.
    import test_controller_repair as practical
    original_argv = practical.training_argv
    def cuda_argv(*args,**kwargs):
        return original_argv(*args,**kwargs)+['--device','cuda','--fp32-math-attention']
    monkeypatch.setattr(practical,'training_argv',cuda_argv)
    practical.test_actual_cli_resume_matches_continuous_epoch_training(tmp_path,monkeypatch)
    full = torch.load(tmp_path/'full/latest.pt',weights_only=False)
    resumed = torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    for a,b in zip(full['rng_state']['cuda'],resumed['rng_state']['cuda']):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert full['optimizer']['param_groups'] == resumed['optimizer']['param_groups']
    for key,state in full['optimizer']['state'].items():
        for name,value in state.items():
            other = resumed['optimizer']['state'][key][name]
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(value,other,atol=0,rtol=0)
            else:
                assert value == other
