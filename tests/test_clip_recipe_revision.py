import json
from pathlib import Path
import sys
import os
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config, FlowMatchVQVAEController
from arch.basic_vqvae_375to291 import BasicVQVAE375to291
from Utils.style_code_prior import (TrainStyleRecipePool, recipe_probabilities_from_indices,
    recipe_slots_from_probs, validate_recipe_metadata, sparse_code_histogram)
from Utils.reproducibility import state_fingerprint
from Utils.controller_validation import teacher_forced_selection_key, TeacherForcedStyleValidation
from test_controller_repair import controller, loss_args
from test_recipe_training import pool_fixture, recipe_cli
from test_style_refinement import style_cli_fixture
from _memmap_fixtures import write_db
import training.train_flowmatch_vqvae_controller as trainer


def clip_model():
    config = dict(conditioning_mode='clip_recipe', use_style_code_selector=True, style_slot_indices=[0,1,2,3],
        style_dim=8, style_selector_hidden_dim=16, style_selector_topk=2, style_proto_normalization='none',
        flow_hidden_dim=16, flow_layers=1, flow_ffn_dim=32, prediction_mode='target',
        flow_loss_space='normalized', dropout=0., inference_requantize=True)
    model = build_flowmatch_controller_from_config(config, vqvae=controller().vqvae, num_styles=3)
    model.set_latent_prior_stats(torch.zeros(8), torch.ones(8))
    model.set_style_code_prior(torch.ones(3,4,16) / 16)
    return model, config


def test_zero_residual_export_parity_and_manual_bypass(monkeypatch):
    model, _ = clip_model(); model.eval()
    indices = np.random.default_rng(42).integers(0,16,(80,4))
    histogram = torch.from_numpy(sparse_code_histogram(indices, selected_slots=range(4), num_embeddings=16, topk=16))
    expected = torch.from_numpy(sparse_code_histogram(indices, selected_slots=range(4), num_embeddings=16, topk=2))
    exported = torch.from_numpy(recipe_probabilities_from_indices(indices, model))
    torch.testing.assert_close(exported, expected, atol=1e-7, rtol=1e-6)
    # Check parity after the selector has actually changed, not only its identity initialization.
    with torch.no_grad(): model.histogram_selector.query[-1].weight.normal_(0,.3)
    exported = model.recipe_from_histogram(histogram)[0]
    x, _ = loss_args()
    a = model.encode_conditions(x_curr=x[:,0], style_label=None, clip_histograms=histogram)[2]
    def forbidden(*a, **k): raise AssertionError('Manual recipes cannot invoke histogram selection or labels')
    monkeypatch.setattr(model.histogram_selector, 'forward', forbidden)
    monkeypatch.setattr(model.style_selector.style_embedding, 'forward', forbidden)
    b = model.encode_conditions(x_curr=x[:,0], style_label=None, manual_style_probs=exported)[2]
    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    assert len(recipe_slots_from_probs(exported.detach().numpy(), model.selected_style_slots)) == 4
    with pytest.raises(ValueError, match='mutually exclusive'):
        model.encode_conditions(x_curr=x[:,0], style_label=None, clip_histograms=histogram, manual_style_probs=exported)


@pytest.mark.parametrize('weights', [(1.,0.), (0.,.02)])
def test_fm_and_prior_train_selector_without_decoder_or_label(weights, monkeypatch):
    model, _ = clip_model(); x, labels = loss_args()
    prev, target = model.frame_latent(x[:,0]), model.frame_latent(x[:,1])
    before = state_fingerprint(model.vqvae.state_dict())
    def forbidden(*a,**k): raise AssertionError('Unexpected decoder/Euler/label path')
    monkeypatch.setattr(model.vqvae, 'decode', forbidden)
    monkeypatch.setattr(model, '_integrate', forbidden)
    monkeypatch.setattr(model.style_selector.style_embedding, 'forward', forbidden)
    hist = torch.rand(3,4,16) + .01
    out = model.compute_loss(x_curr=x[:,0], x_next=x[:,1], style_label=None, prior_labels=labels,
        clip_histograms=hist, prev_latent=prev, target_latent=target, flow_weight=weights[0], style_hist_weight=weights[1])
    out['loss'].backward()
    assert model.histogram_selector.query[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vqvae.parameters())
    assert state_fingerprint(model.vqvae.state_dict()) == before


def test_full_histogram_bank_eviction_is_rng_and_value_invariant(tmp_path):
    _, _, cache = pool_fixture(tmp_path)
    model, _ = clip_model()
    pool = TrainStyleRecipePool(db=tmp_path, model=model, cache=cache, mean=np.zeros(375), std=np.ones(375),
        frames=4, exemplars=3, bank_size=8, bank_max_entries=1)
    torch.manual_seed(9)
    a = pool.sample(torch.arange(3), probability=1., require_all=True)[0]
    rng_a = torch.get_rng_state()
    assert len(pool.bank) == 1 and (a > 0).sum(-1).max() > model.style_selector.topk
    torch.manual_seed(9)
    b = pool.sample(torch.arange(3), probability=1., require_all=True)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(rng_a,torch.get_rng_state(),rtol=0,atol=0)


def test_validated_histogram_training_preserves_loss_gradients_and_rng():
    import copy
    model, _ = clip_model()
    reference = copy.deepcopy(model)
    x, labels = loss_args()
    histogram = torch.rand(3, 4, 16)
    histogram /= histogram.sum(-1, keepdim=True)
    latent = model.frame_latent(x[:, :2].flatten(0, 1)).reshape(3, 2, -1)
    outputs = []
    for current, validated in ((model, True), (reference, False)):
        torch.manual_seed(63)
        out = current.compute_loss(x_curr=x[:, 0], x_next=x[:, 1], style_label=None,
            clip_histograms=histogram, clip_histograms_validated=validated, prior_labels=labels,
            prev_latent=latent[:, 0], target_latent=latent[:, 1], style_hist_weight=.02,
            style_entropy_weight=.003)
        out['loss'].backward()
        outputs.append((out, torch.get_rng_state()))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    with pytest.raises(ValueError, match='non-negative'):
        model.encode_conditions(x_curr=x[:, 0], style_label=None, clip_histograms=-histogram)
    with pytest.raises(ValueError, match='all selected slots'):
        model.recipe_from_histogram(torch.zeros_like(histogram))


def bank_fixture(tmp_path):
    segments=[dict(label=i%2, source_entry_id=i, frame_start=i*10,num_frames=10,
        suffix='FW', split='train' if i<3 else 'val') for i in range(4)]
    x=write_db(tmp_path,num_frames=40,num_styles=2,segments=segments)
    x[:,84:87]=0
    x[:,87:159]=np.tile(np.arange(72,dtype=np.float32)/100,(40,1))
    x[:,303:375]=0
    x.tofile(tmp_path/'X.f32')
    return x


def test_selection_prefers_constraints_then_cross_and_rejects_nan():
    ref=dict(matched_accuracy=.95,cross_accuracy=.5,matched_root_ade_cm=4.,cross_root_ade_cm=8.)
    good={**ref,'cross_accuracy':.6}; bad={**good,'cross_root_ade_cm':20.}
    assert teacher_forced_selection_key(good,ref)<teacher_forced_selection_key(ref,ref)<teacher_forced_selection_key(bad,ref)
    with pytest.raises(ValueError):teacher_forced_selection_key({**ref,'cross_accuracy':float('nan')},ref)
    with pytest.raises(ValueError,match='final control'):
        validate_recipe_metadata({'probability_space':'histogram'},vq_fingerprint='a',selected_slots=[0])


def new_cli(argv,out):
    return recipe_cli(argv,out)+['--conditioning-mode','clip_recipe',
        '--style-hist-weight','.02','--style-entropy-weight','.003']


def test_new_actual_train_validation_and_epoch_resume(tmp_path,monkeypatch):
    _,vq,_,argv=style_cli_fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Training cannot decode or integrate')
    monkeypatch.setattr(BasicVQVAE375to291,'decode',forbidden)
    monkeypatch.setattr(FlowMatchVQVAEController,'_integrate',forbidden)
    full,resumed=tmp_path/'full',tmp_path/'resumed'
    monkeypatch.setattr(sys,'argv',new_cli(argv,full));trainer.main()
    save=trainer._save_checkpoint
    class Stop(Exception):pass
    def interrupt(**kwargs):
        save(**kwargs)
        if kwargs['path'].name=='latest.pt' and kwargs['epoch']==1:raise Stop()
    monkeypatch.setattr(trainer,'_save_checkpoint',interrupt)
    monkeypatch.setattr(sys,'argv',new_cli(argv,resumed))
    with pytest.raises(Stop):trainer.main()
    monkeypatch.setattr(trainer,'_save_checkpoint',save)
    monkeypatch.setattr(sys,'argv',new_cli(argv,resumed)+['--resume',str(resumed/'latest.pt')]);trainer.main()
    a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(resumed/'latest.pt',weights_only=False)
    assert a['training_objective_version']==11
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    torch.testing.assert_close(a['rng_state']['torch'],b['rng_state']['torch'],rtol=0,atol=0)
    rows=[json.loads(s) for s in (full/'history.jsonl').read_text().splitlines()]
    assert all(r['train']['style_hist_loss']>0 and r['val']['recipe_fraction']==1 for r in rows)
    assert all(r['train']['cross_samples']==0 for r in rows)  # donor-free: no cross-style targets
    original=torch.load(vq,weights_only=False)['model']
    assert state_fingerprint(original)==state_fingerprint({k[6:]:v for k,v in a['model'].items() if k.startswith('vqvae.')})


@pytest.mark.parametrize('eval_mode',['reconstruct','rollout'])
@pytest.mark.parametrize('conditioning',['label','clip_recipe'])
def test_real_tf_validator_only_uses_train_exemplars_val_sources_and_240_generated_frames(tmp_path,monkeypatch,eval_mode,conditioning):
    import Utils.controller_validation as validation
    import evaluation.evaluate_minor_revision_generalization as evaluator
    segments=[]
    for label in range(3):
        for i in range(7):
            segments.append(dict(label=label,source_entry_id=len(segments),frame_start=len(segments)*242,
                num_frames=242,split='train' if i<5 else 'val' if i==5 else 'test',suffix='FW'))
    write_db(tmp_path,num_frames=len(segments)*242,num_styles=3,segments=segments)
    class Classifier(torch.nn.Module):
        def forward(self,x):return x.new_zeros((len(x),8)),x.new_zeros((len(x),3))
    metadata=dict(window_size=240,fps=60,class_names=['style_0','style_1','style_2'],joint_set='ybot15')
    monkeypatch.setattr(validation,'build_fid_feature_extractor_from_checkpoint',lambda *a,**k:(Classifier(),metadata))
    classifier=tmp_path/'classifier.pt';classifier.write_bytes(b'test-classifier')
    original=evaluator._classifier_stats
    seen=[]
    def check(classifier,raw,*args,**kwargs):
        assert raw.shape[1]==240
        seen.append(len(raw))
        return original(classifier,raw,*args,**kwargs)
    monkeypatch.setattr(evaluator,'_classifier_stats',check)
    reconstruct=validation.reconstruct_motion_batch
    def check_tf(**kwargs):
        assert kwargs['flow_eval_mode']==eval_mode and kwargs['flow_requantize_z']
        assert (kwargs['style_label'] is None)==(conditioning=='clip_recipe')
        assert (kwargs['manual_style_probs'] is not None)==(conditioning=='clip_recipe')
        return reconstruct(**kwargs)
    monkeypatch.setattr(validation,'reconstruct_motion_batch',check_tf)
    model=controller() if conditioning=='label' else clip_model()[0]
    before=state_fingerprint(model.vqvae.state_dict())
    validator=TeacherForcedStyleValidation(model,db=tmp_path,mean=np.zeros(375),std=np.ones(375),
        classifier_path=classifier,clips=3,windows=1,out_dir=tmp_path,eval_mode=eval_mode)
    a=validator.evaluate(model)
    repeat = eval_mode=='rollout' and conditioning=='clip_recipe'
    if repeat:
        assert a==validator.evaluate(model)
    assert a['matched_cases']==3 and a['cross_cases']==6 and sum(seen)==(18 if repeat else 9)
    assert state_fingerprint(model.vqvae.state_dict())==before
    assert all(segments[c['source_entry_id']]['split']=='train' for r in validator.manifest['recipes'] for c in r['clips'])


def test_legacy_weight_initialization_and_tf_best_checkpoint_entry(tmp_path,monkeypatch):
    import Utils.controller_validation as validation
    _,_,_,argv=style_cli_fixture(tmp_path)
    old,new=tmp_path/'old',tmp_path/'new'
    monkeypatch.setattr(sys,'argv',recipe_cli(argv,old));trainer.main()
    calls=[]
    class Validator:
        manifest={}
        def __init__(self,*a,**k):pass
        def evaluate(self,model):
            model.eval();calls.append(1)
            return dict(matched_accuracy=.95,cross_accuracy=.3+.1*len(calls),matched_root_ade_cm=4.,cross_root_ade_cm=8.)
    monkeypatch.setattr(validation,'TeacherForcedStyleValidation',Validator)
    classifier=tmp_path/'classifier.pt';classifier.write_bytes(b'fixture')
    monkeypatch.setattr(sys,'argv',new_cli(argv,new)+['--init-ckpt',str(old/'latest.pt'),
        '--tf-val-every','1','--tf-val-classifier',str(classifier),'--tf-val-selection','relative',
        '--style-val-mode','reconstruct'])
    trainer.main()
    best=torch.load(new/'best.pt',weights_only=False);latest=torch.load(new/'latest.pt',weights_only=False)
    assert len(calls)==3 and best['epoch']==2 and best['selection']['protocol']=='teacher_forced_240f'
    assert best['selection']==latest['selection'] and best['selection']['constraints_passed']
    assert latest['args']['selector_initialization']['method']=='histogram_zero_residual'
    from Utils.motion_eval import load_motion_model
    restored=load_motion_model(model_name='flowmatch_target',ckpt_path=new/'best.pt',num_styles=3)
    assert restored.conditioning_mode=='clip_recipe'
    torch.testing.assert_close(restored.histogram_selector.query[-1].weight,best['model']['histogram_selector.query.2.weight'])
    import tools.build_zeroshot_style_recipe as builder
    exported=tmp_path/'exported.json'
    monkeypatch.setattr(sys,'argv',['builder','--db',str(tmp_path/'db'),'--flow-ckpt',str(new/'best.pt'),
        '--segment-index','0','--num-frames','4','--topk','4','--device','cpu','--out',str(exported)])
    builder.main()
    payload=json.loads(exported.read_text())
    assert payload['metadata']['recipe_source']=='learned_histogram_selector'
    assert payload['metadata']['selector_sha256']==state_fingerprint(restored.histogram_selector.state_dict())
    raw=np.fromfile(tmp_path/'db/X.f32',np.float32).reshape(-1,375)
    start=payload['metadata']['crop_frame_start']
    _,indices,_=restored.vqvae.encode(torch.from_numpy(raw[start:start+4]))
    expected=recipe_probabilities_from_indices(indices.numpy(),restored)
    actual=np.zeros_like(expected)
    for row,entry in enumerate(payload['slots']):actual[row,entry['codes']]=entry['weights']
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)


def test_bounded_runner_commands_and_safe_smoke_paths():
    root=Path(__file__).resolve().parents[1]
    for smoke in (False,True):
        result=subprocess.run(['bash',str(root/'scripts'/'run_train_clip_recipe_revision.sh')],
            env={**os.environ,'DRY_RUN':'1','SMOKE':'1' if smoke else '0'},check=True,capture_output=True,text=True)
        commands=[shlex.split(line)[4:] for line in result.stdout.splitlines() if line.startswith('[Run ')]
        assert len(commands)==1
        configs=[]
        for command in commands:
            previous=sys.argv
            try:
                sys.argv=['trainer']+command
                configs.append(trainer.parse_args())
            finally:sys.argv=previous
        assert sum(c.epochs for c in configs)==(1 if smoke else 150)
        assert [c.conditioning_mode for c in configs]==['clip_recipe']
        assert all(c.recon_weight==0 and c.rollout_val_every==0 and not c.tf32 for c in configs)
        assert all(('_smoke' in str(c.out_dir))==smoke for c in configs)
        c=configs[0]
        assert c.init_ckpt is None and c.resume is None and c.lr==5e-4 and c.lr_schedule_epochs==150
        assert c.unroll_steps == 1
        assert c.tf_val_selection=='absolute' and c.tf_val_reference is None
        assert c.latent_stats.parent==c.out_dir


def test_absolute_selection_has_fixed_limits_not_random_initial_reference():
    good=dict(matched_accuracy=.9,cross_accuracy=.8,matched_root_ade_cm=4.,cross_root_ade_cm=8.)
    assert teacher_forced_selection_key(good)[0]==0
    assert teacher_forced_selection_key({**good,'matched_accuracy':.84})[0]==1
    assert teacher_forced_selection_key({**good,'cross_root_ade_cm':10.1})[0]==1
    with pytest.raises(ValueError):teacher_forced_selection_key(good,max_cross_ade=0.)


def test_label_mode_accepts_style_validation_on_its_own_classifier(tmp_path,monkeypatch):
    """The style validator is conditioning-agnostic; label mode must pass the CLI gate too."""
    import Utils.controller_validation as validation
    _, vq, _, argv = style_cli_fixture(tmp_path)
    # Drop the recipe flags so the run takes the plain label path.
    raw = argv(tmp_path / 'label')
    base = []
    skip = False
    for item in raw:                    # drop the recipe triplet (--recipe-prob 1)
        if skip:
            skip = False
            continue
        if item == '--recipe-prob':
            skip = True
            continue
        base.append(item)
    calls = []
    class Validator:
        manifest = {}
        def __init__(self, *a, **k): pass
        def evaluate(self, model):
            model.eval(); calls.append(1)
            return dict(matched_accuracy=.9, cross_accuracy=.5, matched_root_ade_cm=4., cross_root_ade_cm=8.)
    monkeypatch.setattr(validation, 'TeacherForcedStyleValidation', Validator)
    classifier = tmp_path / 'classifier.pt'; classifier.write_bytes(b'fixture')
    out = tmp_path / 'label'          # argv already points --out-dir here
    monkeypatch.setattr(sys, 'argv', base + ['--tf-val-every', '1', '--tf-val-classifier', str(classifier)])
    # Label mode keeps the raw label selector, so the histogram/prior path must not run. The
    # validator itself is stubbed; the call count depends on the absolute-selection limits, so
    # only require that validation ran and was logged.
    trainer.main()
    records = [json.loads(row) for row in (out / 'history.jsonl').read_text().splitlines()]
    assert any('rollout_style_val' in row or 'tf_style_val' in row for row in records)
    assert len(calls) >= 1


def test_scratch_tf_entry_skips_initial_model_reference(tmp_path,monkeypatch):
    import Utils.controller_validation as validation
    _,_,_,argv=style_cli_fixture(tmp_path)
    calls=[]
    class Validator:
        manifest={}
        def __init__(self,*a,**k):pass
        def evaluate(self,model):
            model.eval();calls.append(1)
            return dict(matched_accuracy=.9,cross_accuracy=.5,matched_root_ade_cm=4.,cross_root_ade_cm=8.)
    monkeypatch.setattr(validation,'TeacherForcedStyleValidation',Validator)
    classifier=tmp_path/'classifier.pt';classifier.write_bytes(b'fixture')
    out=tmp_path/'scratch'
    monkeypatch.setattr(sys,'argv',new_cli(argv,out)+['--tf-val-every','1','--tf-val-classifier',str(classifier)])
    trainer.main()
    best=torch.load(out/'best.pt',weights_only=False)
    assert len(calls)==2 and best['args']['init_ckpt'] is None
    assert best['selection']['policy']=='absolute' and not (out/'tf_validation_reference.json').exists()
    assert best['selection']['protocol']=='rollout_240f'
