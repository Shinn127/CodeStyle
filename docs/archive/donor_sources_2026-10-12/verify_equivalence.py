"""Real FP32 batches: before/after donor, recipe, RNG, losses and gradients."""
import copy
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_flowmatch_vqvae_controller as trainer
from dataset import build_train_val_test_dataloaders
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import load_norm_stats, configure_torch_runtime, set_seed
from Utils.style_code_prior import TrainStyleRecipePool
from Utils.style_donor_bank import StyleDonorBank
from Utils.vq_target_cache import VQTargetCache
from Utils.reproducibility import rng_state, restore_rng_state, state_fingerprint


def before(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent/'before'/filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--math-attention', action='store_true')
    options = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device('cuda')
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision='highest')
    set_seed(42)
    state = torch.load(ROOT/'checkpoints/clip_recipe_scratch150/latest.pt', map_location='cpu', weights_only=False)
    args = state['args']; mean, std = load_norm_stats(args['norm_stats'])
    vq = build_vqvae_from_checkpoint(args['vqvae_ckpt']).to(device)
    current = trainer.build_flowmatch_controller_from_config(args, vqvae=vq, num_styles=90).to(device)
    current.load_state_dict(state['model'])
    current.field.fp32_math_attention = options.math_attention
    original = before('arch.before_controller', 'flowmatch_vqvae_controller.py').build_flowmatch_controller_from_config(
        args, vqvae=copy.deepcopy(vq), num_styles=90).to(device)
    original.load_state_dict(state['model'])
    cache = VQTargetCache(args['vq_target_cache'])
    pool_args = dict(db=args['db'], cache=cache, mean=mean, std=std, frames=240,
                     exemplars=5, exemplar_choices=(1,3,5), bank_size=64)
    old_pool = before('before_pool', 'style_code_prior.py').TrainStyleRecipePool(model=original, **pool_args)
    new_pool = TrainStyleRecipePool(model=current, **pool_args)
    old_bank = before('before_bank', 'style_donor_bank.py').StyleDonorBank(args['db'], device=device)
    new_bank = StyleDonorBank(args['db'], device=device)
    bundle = build_train_val_test_dataloaders(db_path=args['db'], seq_len=2, stride=1,
        enforce_same_label=True, batch_size=512, num_workers=0, pin_memory=False, drop_last=True,
        normalize=True, norm_mean=mean, norm_std=std)
    bundle.train_loader.collate_fn = trainer.controller_collate
    trainer._seed_loader(bundle.train_loader, seed=42, epoch=5)
    maxima = dict(loss_max_abs=0., gradient_max_abs=0., gradient_l2_relative=0.)
    frozen = state_fingerprint(current.vqvae.state_dict())
    for number, batch in enumerate(bundle.train_loader):
        if number == 16:
            break
        x = batch['x'].to(device); labels = batch['label'].to(device)
        ids = batch['start'][:, None] + torch.arange(2)
        latents = cache.lookup_latents(ids.flatten(), vqvae=vq, device=device).reshape(512,2,128)
        source_ids = [bundle.train_dataset.segments[i]['source_entry_id'] for i in batch['segment_id'].tolist()]
        start_rng = rng_state(); outputs = []
        for model, bank, pool in ((original, old_bank, old_pool), (current, new_bank, new_pool)):
            restore_rng_state(start_rng); model.train(); model.zero_grad(set_to_none=True)
            gate = torch.rand(512, device=device)
            mask = gate < .2; selected = (gate >= .2) & (gate < .5)
            observed, previous, _, mixed, entries = bank.perturb_history(current=x[:,0],
                previous_latent=latents[:,0], labels=labels, segment_ids=batch['segment_id'],
                selected=selected, norm_mean=mean, norm_std=std, frame_latent_fn=model.frame_latent)
            mask |= selected & ~mixed
            histogram, _ = pool.sample(labels, probability=1., require_all=True,
                per_sample_exclusions=[{a,b} for a,b in zip(source_ids, entries)])
            out = model.compute_loss(x_curr=observed, x_next=x[:,1], style_label=None,
                prior_labels=labels, clip_histograms=histogram, clip_histograms_validated=True,
                prev_latent=previous, target_latent=latents[:,1], recon_weight=0.,
                style_hist_weight=.02, style_entropy_weight=.003, history_noise_mode='structured',
                history_noise_probability=0., history_mask=mask, flow_low_t_probability=.5,
                return_per_sample=True)
            out['loss'].backward()
            outputs.append((observed, previous, mixed, entries, histogram, out, rng_state(),
                {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}))
        a,b = outputs
        for i in (0,1,2,4):
            torch.testing.assert_close(a[i], b[i], atol=0, rtol=0)
        assert a[3] == b[3]
        torch.testing.assert_close(a[6]['torch'], b[6]['torch'], atol=0, rtol=0)
        for u,v in zip(a[6]['cuda'], b[6]['cuda']):
            torch.testing.assert_close(u,v,atol=0,rtol=0)
        for key in a[5]:
            torch.testing.assert_close(a[5][key],b[5][key],atol=3e-6,rtol=1e-5)
            maxima['loss_max_abs']=max(maxima['loss_max_abs'], float((a[5][key]-b[5][key]).detach().abs().max()))
        error, scale = [], []
        for key in a[7]:
            torch.testing.assert_close(a[7][key],b[7][key],atol=1e-5 if options.math_attention else 2e-6,
                                       rtol=2e-4 if options.math_attention else 1e-5)
            maxima['gradient_max_abs']=max(maxima['gradient_max_abs'], float((a[7][key]-b[7][key]).abs().max()))
            error.append((a[7][key]-b[7][key]).square().sum())
            scale.append(a[7][key].square().sum())
        relative = float((torch.stack(error).sum()/torch.stack(scale).sum().clamp_min(1e-20)).sqrt())
        assert relative < 5e-5
        maxima['gradient_l2_relative'] = max(maxima['gradient_l2_relative'], relative)
    assert state_fingerprint(current.vqvae.state_dict()) == frozen
    assert all(p.grad is None for p in current.vqvae.parameters())
    result = dict(real_batches=16, batch_size=512, fp32_math_attention=options.math_attention, donor_recipe_rng_exact=True,
                  frozen_vq_unchanged=True, **maxima)
    (Path(__file__).parent/('equivalence_math.json' if options.math_attention else 'equivalence.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
