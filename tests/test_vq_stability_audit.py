"""Independent geometry/gradient/state checks for VQ stability auxiliaries."""
import copy
import json
from types import SimpleNamespace
import sys
import os
import subprocess
from pathlib import Path

import pytest
import torch

from arch.basic_vqvae_375to291 import codebook_boundary_distance, LibraryVectorQuantizer
import test_basic_vqvae as vq_fixtures
from test_controller_repair import controller, small_training_fixture
import training.train_basic_vqvae_375to291 as trainer


def test_second_closest_code_does_not_define_closest_boundary():
    book=torch.tensor([[0.,0.],[.2,0.],[0.,1.]],requires_grad=True)
    z=torch.tensor([[.09,.495]],requires_grad=True)
    assert torch.cdist(z,book).argsort(-1).tolist()==[[0,1,2]]
    old,_=codebook_boundary_distance(z,book,mode='nearest_code')
    radius,valid=codebook_boundary_distance(z,book)
    torch.testing.assert_close(old,torch.tensor([.01]),atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(radius,torch.tensor([.005]),atol=1e-7,rtol=1e-5)
    torch.relu(.02-radius).sum().backward()
    assert valid.all() and book.grad is None
    torch.testing.assert_close(z.grad,torch.tensor([[0.,1.]]))


def test_exact_margin_matches_direct_plane_oracle_values_and_gradients():
    torch.manual_seed(13);book=torch.randn(17,4);z=torch.randn(31,4,requires_grad=True)
    ref=z.detach().clone().requires_grad_();own=torch.cdist(ref.detach(),book).argmin(-1)
    distances=(ref[:,None]-book[None]).square().sum(-1)
    lengths=(book[None]-book[own,None]).norm(dim=-1)
    planes=(distances-distances.gather(1,own[:,None]))/(2*lengths.clamp_min(1e-9))
    planes=planes.masked_fill(lengths<1e-9,float('inf'))
    oracle=planes.min(-1).values
    actual,valid=codebook_boundary_distance(z,book,chunk_rows=7)
    assert valid.all()
    torch.testing.assert_close(actual,oracle,atol=1e-6,rtol=1e-5)
    torch.relu(2-actual).sum().backward();torch.relu(2-oracle).sum().backward()
    torch.testing.assert_close(z.grad,ref.grad,atol=1e-6,rtol=1e-5)


def test_duplicate_codes_and_single_code_have_defined_finite_behavior():
    z=torch.tensor([[.1,0.]],requires_grad=True)
    radius,valid=codebook_boundary_distance(z,torch.tensor([[0.,0.],[0.,0.],[1.,0.]]))
    assert valid.all();torch.testing.assert_close(radius,torch.tensor([.4]))
    for book in (torch.zeros(1,2),torch.zeros(3,2)):
        r,v=codebook_boundary_distance(z,book)
        assert not v.any() and torch.isfinite(r).all() and r.item()==0
        r.sum().backward();assert z.grad.abs().sum()==0


def test_raw_margin_scaling_is_not_improved_relative_stability():
    z=torch.tensor([[.09,.495]]);book=torch.tensor([[0.,0.],[.2,0.],[0.,1.]])
    a,_=codebook_boundary_distance(z,book);b,_=codebook_boundary_distance(z*3,book*3)
    torch.testing.assert_close(b,a*3,atol=1e-6,rtol=1e-5)
    eps=torch.tensor([[0.,.006]])
    assert torch.cdist(z+eps,book).argmin(-1).item()==torch.cdist((z+eps)*3,book*3).argmin(-1).item()


def test_code_geometry_and_ce_remain_fp32_inside_external_autocast():
    torch.manual_seed(42);z=torch.randn(133,4);book=torch.randn(97,4)
    ref=codebook_boundary_distance(z,book)[0]
    model=controller();p=torch.randn(3,model.latent_dim);target=torch.randn_like(p)
    ref_ce=model.code_decision_loss(p,target)[0]
    q=LibraryVectorQuantizer(97,4,1,kmeans_init=False).eval()
    ref_q=q.quantize_only(z)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        actual=codebook_boundary_distance(z,book)[0]
        actual_ce=model.code_decision_loss(p,target)[0]
        actual_q=q.quantize_only(z)
    torch.testing.assert_close(actual,ref,atol=0,rtol=0)
    torch.testing.assert_close(actual_ce,ref_ce,atol=0,rtol=0)
    torch.testing.assert_close(actual_q,ref_q,atol=0,rtol=0)
    assert actual.dtype==actual_ce.dtype==actual_q.dtype==torch.float32


def test_separate_head_fast_quantizer_preserves_batch_slot_order():
    from vector_quantize_pytorch import VectorQuantize
    q=LibraryVectorQuantizer(2,2,2,kmeans_init=False).eval()
    q.vq=VectorQuantize(dim=4,codebook_size=2,codebook_dim=2,heads=2,
        separate_codebook_per_head=True,kmeans_init=False,threshold_ema_dead_code=0).eval()
    with torch.no_grad():q.vq.codebook.copy_(torch.tensor([[[0.,0.],[10.,10.]],[[100.,100.],[110.,110.]]]))
    x=torch.tensor([[.1,.2,109.,110.],[9.9,10.1,100.1,100.2],[0.,0.,110.,110.]])
    expected=q(x).quantized
    torch.testing.assert_close(q.quantize_only(x,chunk_rows=1),expected,atol=0,rtol=0)


def test_auxiliary_gradients_are_isolated_and_ema_updates_only_once():
    model=vq_fixtures.BoundaryMarginAndRoundtripTests._build().eval();x=torch.randn(2,3,375)
    before=copy.deepcopy(model.state_dict())
    model.compute_loss(x,latent_margin_weight=1)['latent_margin_loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    model.zero_grad();inputs=[];decoded=[];calls=[]
    h1=model.encoder.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().clone()))
    h2=model.decoder.register_forward_hook(lambda m,a,o:decoded.append(o.detach().clone()))
    h3=model.quantizer.register_forward_hook(lambda *a:calls.append(1))
    out=model.compute_loss(x,roundtrip_weight=1)
    out['roundtrip_loss'].backward()
    for hook in (h1,h2,h3):hook.remove()
    assert len(calls)==1 and len(inputs)==2
    expected=torch.cat((x[...,:84],decoded[0].reshape(2,3,291)),-1).reshape(-1,375)
    torch.testing.assert_close(inputs[-1].reshape(-1,375),expected,atol=0,rtol=0)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.decoder.parameters())
    model.train();before=model.quantizer.embedding_weight.clone()
    model.compute_loss(x,latent_margin_weight=1,roundtrip_weight=1)
    assert not torch.equal(before,model.quantizer.embedding_weight)
    assert not list(model.quantizer.parameters())


def test_init_missing_ema_buffer_fails_and_k_change_reinitializes():
    old=vq_fixtures.BoundaryMarginAndRoundtripTests._build();checkpoint=dict(model=old.state_dict(),args={})
    same=vq_fixtures.BoundaryMarginAndRoundtripTests._build()
    assert not trainer.load_initial_vq_weights(same,checkpoint)
    for k,v in old.state_dict().items():torch.testing.assert_close(same.state_dict()[k],v,atol=0,rtol=0)
    corrupt=copy.deepcopy(checkpoint);corrupt['model'].pop('quantizer.vq._codebook.cluster_size')
    with pytest.raises(ValueError,match='missing'):
        trainer.load_initial_vq_weights(same,corrupt)
    smaller=vq_fixtures.BoundaryMarginAndRoundtripTests._build();smaller.quantizer=LibraryVectorQuantizer(8,4,4,kmeans_init=True)
    assert trainer.load_initial_vq_weights(smaller,checkpoint)
    assert not bool(smaller.quantizer.vq._codebook.initted)


def test_resume_rejects_silent_margin_geometry_change():
    a=SimpleNamespace(code_align_weight=0.,latent_margin_weight=1.,roundtrip_weight=1.,
        latent_margin_mode='nearest_boundary',latent_margin_target=.05)
    old={'args':dict(code_align_weight=0.,latent_margin_weight=1.,roundtrip_weight=1.)}
    with pytest.raises(ValueError,match='geometry'):trainer.validate_stability_resume(a,old)
    a.latent_margin_mode='nearest_code';trainer.validate_stability_resume(a,old)


def test_temperature_cache_is_reused_and_invalidates_after_table_mutation(monkeypatch):
    model=controller();calls=[];cdist=torch.cdist
    def count(*a,**kw):calls.append(1);return cdist(*a,**kw)
    monkeypatch.setattr(torch,'cdist',count)
    first=model.code_ce_temperature()
    for _ in range(6):assert model.code_ce_temperature()==first
    assert len(calls)==1
    with torch.no_grad():model.vqvae.quantizer.embedding_weight.mul_(2)
    assert model.code_ce_temperature()==pytest.approx(4*first)
    assert len(calls)==2
    with pytest.raises(ValueError):model.code_ce_temperature(float('nan'))


def test_temperature_ignores_coincident_code_aliases():
    model=controller();book=model.vqvae.quantizer.embedding_weight
    with torch.no_grad():
        book.zero_();book[len(book)//2:,0]=2
    assert model.code_ce_temperature()==pytest.approx(1.)
    with torch.no_grad():book.zero_()
    assert model.code_ce_temperature()==1.


def test_code_ce_is_on_fm_endpoint_not_independent_euler_output(monkeypatch):
    model=controller();x=torch.randn(2,2,375);seen=[];ce=model.code_decision_loss
    def capture(pred,target,**kw):seen.append(pred.detach());return ce(pred,target,**kw)
    monkeypatch.setattr(model,'code_decision_loss',capture)
    monkeypatch.setattr(model,'_integrate',lambda **kw:torch.full_like(kw['z_prev'],7.))
    out=model.compute_loss(x_curr=x[:,0],x_next=x[:,1],style_label=torch.tensor([0,1]),
        code_ce_weight=.01,return_rollout_latent=True)
    assert len(seen)==1
    torch.testing.assert_close(model.normalize_latent(out['pred_latent']),torch.full_like(seen[0],7.))
    assert not torch.allclose(seen[0],model.normalize_latent(out['pred_latent']))


def test_real_vq_cli_entry_saves_exact_geometry_metadata(tmp_path,monkeypatch):
    db,_,norm=small_training_fixture(tmp_path);out=tmp_path/'run'
    argv=['vq','--db',str(db),'--norm-stats',str(norm),'--out-dir',str(out),'--device','cpu',
        '--epochs','1','--seq-len','2','--batch-size','6','--num-workers','0','--no-compile',
        '--hidden-dim','16','--decoder-hidden-dim','16','--encoder-layers','2','--decoder-layers','2',
        '--num-latents','4','--code-dim','2','--num-embeddings','16','--eval-every','0',
        '--latent-margin-weight','1','--roundtrip-weight','1']
    monkeypatch.setattr(sys,'argv',argv);trainer.main()
    c=torch.load(out/'latest.pt',weights_only=False)
    assert c['args']['latent_margin_mode']=='nearest_boundary' and c['args']['vq_stability_version']==2
    assert json.load(open(out/'config.json'))['latent_margin_weight']==1
    with pytest.raises(ValueError,match='already contains'):
        trainer.main()


def test_training_wrapper_dry_run_never_starts_training(tmp_path):
    root=Path(trainer.__file__).resolve().parents[1]
    env=dict(os.environ,DRY_RUN='1',PYTHON_BIN=sys.executable,OUT=str(tmp_path/'not_created'))
    r=subprocess.run(['bash',str(root/'scripts/run_train_basic_vqvae_375to291.sh')],env=env,
        cwd=root,capture_output=True,text=True,check=True)
    assert '[Dry run]' in r.stdout and '--latent-margin-mode nearest_boundary' in r.stdout
    assert not (tmp_path/'not_created').exists()


def test_training_quantized_values_precede_the_ema_update():
    q=LibraryVectorQuantizer(8,2,2,kmeans_init=False,dead_code_threshold=0,ema_decay=.5).train()
    before=q.embedding_weight.clone()
    out=q(torch.randn(16,4)*.1)
    expected=before[out.indices].reshape(16,4)
    torch.testing.assert_close(out.quantized,expected,atol=1e-6,rtol=1e-5)
    assert not torch.allclose(q.lookup(out.indices),expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_exact_boundary_matches_cpu_value_and_gradient():
    torch.manual_seed(42);book=torch.randn(97,4);points=torch.randn(133,4)
    values=[];gradients=[]
    for device in ('cpu','cuda'):
        z=points.to(device).detach().clone().requires_grad_()
        d,valid=codebook_boundary_distance(z,book.to(device),chunk_rows=31)
        assert valid.all();torch.relu(1-d).mean().backward()
        values.append(d.detach().cpu());gradients.append(z.grad.cpu())
    torch.testing.assert_close(*values,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(*gradients,atol=2e-6,rtol=2e-5)
