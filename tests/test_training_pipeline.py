"""Encoding-only behavior and worker batching, independent of real data."""
import copy

import pytest
import torch
import numpy as np

from test_controller_repair import controller,loss_args
from test_recipe_training import pool_fixture
from test_clip_recipe_revision import clip_model
from Utils.reproducibility import rng_state, restore_rng_state, state_fingerprint
from Utils.style_code_prior import TrainStyleRecipePool, sparse_code_histogram, _full_code_profile
import Utils.style_code_prior as prior_module
from Utils.vq_target_cache import VQTargetCache, VQTargetCacheMeta, save_vq_target_cache_meta
from dataset import Motion375Dataset
from _memmap_fixtures import write_db
from training.train_flowmatch_vqvae_controller import configure_controller_loader, _run_epoch, _seed_loader


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('sequence',[False,True])
def test_vq_encode_matches_full_forward_without_running_decoder(device,sequence):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=controller().vqvae.to(device).eval();x,_=loss_args();x=x.to(device)
    x=x if sequence else x[:,0]
    with torch.no_grad():expected=model(x,return_details=True)
    def forbidden(*a):raise AssertionError('encode must never call the decoder module')
    hook=model.decoder.register_forward_pre_hook(forbidden)
    actual=model.encode(x);hook.remove()
    for value,key in zip(actual,('quantized_latent','indices','perplexity')):
        torch.testing.assert_close(value,expected[key],atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_host_recipe_labels_preserve_probabilities_and_rng(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model,_,cache=pool_fixture(tmp_path);model.to(device)
    args=dict(db=tmp_path,model=model,cache=cache,mean=torch.zeros(375).numpy(),
        std=torch.ones(375).numpy(),frames=4,exemplar_choices=(1,3,5),bank_size=64)
    cpu=torch.arange(3);labels=cpu.to(device)
    out=[]
    for pool,hint in ((TrainStyleRecipePool(**args),None),(TrainStyleRecipePool(**args),cpu)):
        torch.manual_seed(48)
        value,_=pool.sample(labels,probability=1.,require_all=True,
            per_sample_exclusions=[{0},{6},{12}],labels_cpu=hint)
        out.append((value,torch.get_rng_state(),torch.cuda.get_rng_state() if device=='cuda' else None))
    torch.testing.assert_close(out[0][0],out[1][0],atol=0,rtol=0)
    torch.testing.assert_close(out[0][1],out[1][1],atol=0,rtol=0)
    if device=='cuda':torch.testing.assert_close(out[0][2],out[1][2],atol=0,rtol=0)


@pytest.mark.parametrize('frames,size',[(7,16),(1200,2048)])
def test_full_sparse_profile_matches_dense_histogram_exactly(frames,size):
    codes=np.random.default_rng(42).integers(0,size,(frames,32),dtype=np.uint16)
    selected=[31,5,4,9,10,17,18,20]
    dense=sparse_code_histogram(codes,selected_slots=selected,num_embeddings=size,topk=size)
    si,code,weight=_full_code_profile(codes[:,selected].astype(np.int64),size)
    expected=np.nonzero(dense)
    np.testing.assert_array_equal(si,expected[0]);np.testing.assert_array_equal(code,expected[1])
    np.testing.assert_array_equal(weight,dense[expected])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sparse_recipe_bank_matches_dense_reference_and_rng(tmp_path,device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,_,cache=pool_fixture(tmp_path);model,_=clip_model();model.to(device)
    model.style_slot_indices.copy_(torch.tensor([3,0,2,1],device=device))
    cache.indices=np.random.default_rng(82).integers(0,16,(152,4),dtype=np.uint16)
    def dense_profile(indices,size):
        dense=sparse_code_histogram(indices,selected_slots=range(indices.shape[1]),num_embeddings=size,topk=size)
        si,code=np.nonzero(dense)
        return si,code,dense[si,code]
    outputs=[]
    for reference in (False,True):
        if reference:monkeypatch.setattr(prior_module,'_full_code_profile',dense_profile)
        pool=TrainStyleRecipePool(db=tmp_path,model=model,cache=cache,mean=np.zeros(375),std=np.ones(375),
            frames=4,exemplar_choices=(1,3,5),bank_size=8,bank_max_entries=1)
        host=torch.tensor([0,1,2]*4);labels=host.to(device);torch.manual_seed(73);recipes=[]
        for _ in range(4):
            recipes.append(pool.sample(labels,probability=1.,require_all=True,labels_cpu=host,
                per_sample_exclusions=[{int(y)*6} for y in host])[0])
        outputs.append((torch.stack(recipes),rng_state()))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=0,rtol=0)
    torch.testing.assert_close(outputs[0][1]['torch'],outputs[1][1]['torch'],atol=0,rtol=0)
    if device=='cuda':
        for left,right in zip(outputs[0][1]['cuda'],outputs[1][1]['cuda']):torch.testing.assert_close(left,right,atol=0,rtol=0)


@pytest.mark.parametrize('workers',[0,2])
@pytest.mark.parametrize('seq_len',[2,4])
def test_worker_batches_preserve_all_rows_and_cache_indices(tmp_path,workers,seq_len):
    db=tmp_path/'db';db.mkdir()
    write_db(db,num_frames=96,num_styles=3,segments=[dict(label=0,frame_start=0,num_frames=96,split='train')])
    folder=tmp_path/'cache';folder.mkdir()
    codes=(np.arange(96*4)%16).astype(np.uint16).reshape(96,4);codes.tofile(folder/'indices.u16')
    save_vq_target_cache_meta(folder,VQTargetCacheMeta(db=db,vqvae_ckpt=tmp_path/'vq',norm_stats=tmp_path/'norm',
        num_frames=96,feature_dim=375,num_codebooks=4,code_dim=2,latent_dim=8,num_embeddings=16,
        indices_dtype='uint16',indices_file='indices.u16',complete=True))
    cache=VQTargetCache(folder)
    data=Motion375Dataset(db,seq_len=seq_len,normalize=True,norm_mean=np.zeros(375,np.float32),norm_std=np.ones(375,np.float32))
    loader=torch.utils.data.DataLoader(data,batch_size=4,num_workers=workers)
    expected=[data[i] for i in range(4)]
    configure_controller_loader(loader,cache)
    actual=next(iter(loader))
    for key in ('x','label','start','segment_id'):
        torch.testing.assert_close(actual[key],torch.stack([sample[key] for sample in expected]),atol=0,rtol=0)
    torch.testing.assert_close(actual['vq_indices'],torch.from_numpy(codes[np.arange(4)[:,None]+np.arange(seq_len)].astype(np.int64)),atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('seq_len,steps',[(2,1),(4,1),(4,3)])
def test_cached_worker_epoch_preserves_metrics_gradients_weights_and_rng(tmp_path,device,seq_len,steps):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pool_fixture(tmp_path)
    base,_=clip_model();base.to(device)
    raw=np.fromfile(tmp_path/'X.f32',dtype=np.float32).reshape(-1,375)
    codes=base.vqvae.encode(torch.from_numpy(raw).to(device))[1].cpu().numpy().astype(np.uint16)
    folder=tmp_path/'cache';folder.mkdir();codes.tofile(folder/'indices.u16')
    save_vq_target_cache_meta(folder,VQTargetCacheMeta(db=tmp_path,vqvae_ckpt=tmp_path/'vq',norm_stats=tmp_path/'norm',
        num_frames=len(raw),feature_dim=375,num_codebooks=4,code_dim=2,latent_dim=8,num_embeddings=16,
        indices_dtype='uint16',indices_file='indices.u16',complete=True))
    cache=VQTargetCache(folder);before=state_fingerprint(base.vqvae.state_dict());outputs=[]
    torch.manual_seed(73);initial=rng_state()
    for fast in (False,True):
        model=copy.deepcopy(base)
        data=Motion375Dataset(tmp_path,seq_len=seq_len,normalize=True,norm_mean=np.zeros(375,np.float32),norm_std=np.ones(375,np.float32))
        loader=torch.utils.data.DataLoader(data,batch_size=4,num_workers=0,shuffle=True)
        configure_controller_loader(loader,cache,worker_batches=fast)
        pool=TrainStyleRecipePool(db=tmp_path,model=model,cache=cache,mean=np.zeros(375),std=np.ones(375),
            frames=4,exemplar_choices=(1,3,5),bank_size=8)
        if not fast:
            old_sample=pool.sample
            def without_host_hint(labels,**kwargs):
                kwargs.pop('labels_cpu',None)
                return old_sample(labels,**kwargs)
            pool.sample=without_host_hint
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        restore_rng_state(initial);_seed_loader(loader,seed=9,epoch=1)
        metrics=_run_epoch(model=model,loader=loader,device=torch.device(device),prediction_mode='target',
            recon_weight=0.,style_hist_weight=.02,style_entropy_weight=.003,noise_std=1.,

            solver_steps=2,optimizer=optimizer,vq_target_cache=cache,cache_vqvae=model.vqvae,
            grad_clip=1.,log_interval=10,epoch=1,total_epochs=1,split='fixture',max_batches=2,
            recipe_pool=pool,recipe_probability=1.,unroll_steps=steps,microbatch_size=3)
        outputs.append((model,metrics,rng_state()))
        assert state_fingerprint(model.vqvae.state_dict())==before
        assert all(p.grad is None for p in model.vqvae.parameters())
    a,b=outputs
    assert a[1]==b[1]
    for name,tensor in a[0].state_dict().items():torch.testing.assert_close(tensor,b[0].state_dict()[name],atol=0,rtol=0)
    for left,right in zip(a[0].parameters(),b[0].parameters()):
        if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=0,rtol=0)
    torch.testing.assert_close(a[2]['torch'],b[2]['torch'],atol=0,rtol=0)
    if device=='cuda':
        for left,right in zip(a[2]['cuda'],b[2]['cuda']):torch.testing.assert_close(left,right,atol=0,rtol=0)


