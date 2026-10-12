from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_unseen_style as unseen
from test_controller_repair import controller
from _memmap_fixtures import write_db
from Utils.motion_eval import reconstruct_motion_batch
from Utils.reproducibility import case_noise_plan
from Utils.style_code_prior import validate_recipe_metadata
from dataset import norm_stats_fingerprint
from simulate_realtime_flowmatch_control import _load_style_recipe
import evaluate_minor_revision_generalization as seen_eval
import build_zeroshot_style_recipe as recipe_builder


def test_rollout_uses_generated_history_not_future_gt_motion():
    model=controller().eval()
    x=torch.randn(2,6,375)
    probabilities=torch.ones(2,4,16)/16
    noise=case_noise_plan(model,[10,20],[0,1],6,seed=42,noise_std=1.)
    expected=unseen.generate_pure_recipe(model,x,probabilities,noise,mode='rollout',
        model_name='flowmatch_target',steps=2,microbatch=32)
    changed=x.clone();changed[:,1:,84:]=torch.randn_like(changed[:,1:,84:])*10
    actual=unseen.generate_pure_recipe(model,changed,probabilities,noise,mode='rollout',
        model_name='flowmatch_target',steps=2,microbatch=32)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_rollout_runner_locks_shared_protocol_and_separate_output(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(root/'run_evaluate_style_rollout.sh')],
        env={**os.environ,'DRY_RUN':'1','EVAL_UNSEEN':'1'},check=True,capture_output=True,text=True)
    seen_command,unseen_command=[shlex.split(line)[4:] for line in result.stdout.splitlines()]
    monkeypatch.setattr(sys,'argv',['seen']+seen_command);seen=seen_eval.parse_args()
    monkeypatch.setattr(sys,'argv',['unseen']+unseen_command);ood=unseen.parse_args()
    assert seen.flow_eval_mode==ood.flow_eval_mode=='rollout'
    assert seen.case_manifest==ood.source_manifest
    assert seen.num_sources==ood.num_sources==1000 and seen.targets_per_source==ood.targets_per_source==5
    assert seen.horizons==[240] and seen.recipe_window==240 and ood.ode_steps==seen.flow_ode_steps==8
    assert seen.out_dir!=ood.out_dir and seen.flow_ckpt==ood.flow_ckpt
    assert seen.flow_requantize_z and seen.flow_latent_clip


def test_independent_exemplar_reference_and_name_mapping():
    segments = []
    for label in range(2):
        for source in range(8):
            segments.append(dict(label=label,source_entry_id=10*label+source,
                frame_start=len(segments)*300,num_frames=300))
    # Explicit classifier order differs from DB order; generator does not contain either style.
    plans = unseen.plan_recipes(['oodA','oodB'],segments,seen_names=['seen'],
        classifier_names=['oodB','seen','oodA'],frames=240,exemplars=5,seed=42)
    assert [p['classifier_label'] for p in plans] == [2,0]
    for p in plans:
        example = {c['source_entry_id'] for c in p['clips'] if c['role']=='exemplar'}
        refs = {c['source_entry_id'] for c in p['clips'] if c['role']=='reference'}
        assert len(example)==5 and len(refs)==3 and not example & refs
    again = unseen.plan_recipes(['oodA','oodB'],segments,seen_names=['seen'],
        classifier_names=['oodB','seen','oodA'],frames=240,exemplars=5,seed=42)
    assert plans == again
    with pytest.raises(ValueError,match='separate reference'):
        unseen.plan_recipes(['oodA'],segments[:5],seen_names=['seen'],
            classifier_names=['oodA'],frames=240,exemplars=5,seed=42)


def test_sources_use_test_clips_only_and_reject_bad_manifest():
    segments = [dict(label=0,source_entry_id=i,frame_start=i*500,num_frames=500,
                     split='train' if i==0 else 'test') for i in range(4)]
    rows = unseen.plan_sources(['seen'],segments,frames=241,stride=30,count=5,
        target_count=2,recipe_count=3,seed=42)
    assert all(row['source_entry_id'] in (1,2,3) for row in rows)
    assert all(len(set(row['recipe_indices']))==2 for row in rows)
    manifest = {'horizons':{'240':rows}}
    inherited = unseen.plan_sources(['seen'],segments,frames=241,stride=30,count=5,
        target_count=2,recipe_count=3,seed=42,source_manifest=manifest)
    assert [r['source_start'] for r in inherited] == [r['source_start'] for r in rows]
    manifest['horizons']['240'][0]['source_start'] = 500
    with pytest.raises(ValueError,match='non-test'):
        unseen.plan_sources(['seen'],segments,frames=241,stride=30,count=5,
            target_count=2,recipe_count=3,seed=42,source_manifest=manifest)
    manifest['protocol_version'] = 2
    with pytest.raises(ValueError,match='generated-only'):
        unseen.plan_sources(['seen'],segments,frames=241,stride=30,count=5,
            target_count=2,recipe_count=3,seed=42,source_manifest=manifest)


def test_seen_and_unseen_protocols_select_identical_test_sources(tmp_path,monkeypatch):
    segments = []
    for label in range(3):
        for split_index,split in enumerate(('train','val','test')):
            segments.append(dict(label=label,source_entry_id=10*label+split_index,
                frame_start=len(segments)*300,num_frames=300,split=split))
    write_db(tmp_path,num_frames=len(segments)*300,num_styles=3,segments=segments)
    names = json.loads((tmp_path/'StyleNames.json').read_text())
    monkeypatch.setattr(sys,'argv',['evaluate','--classifier-ckpt','fixture','--db',str(tmp_path),
        '--num-sources','5','--num-workers','0'])
    args = seen_eval.parse_args()
    args.case_data = {}
    loader,_,_,_,_ = seen_eval._build_test_loader(args,horizon=240,
        mean=np.zeros(375,np.float32),std=np.ones(375,np.float32))
    starts = loader.dataset.dataset.sample_starts[loader.dataset.indices].tolist()
    planned = unseen.plan_sources(names,segments,frames=241,stride=30,count=5,
        target_count=2,recipe_count=3,seed=42)
    assert starts == [r['source_start'] for r in planned]


def test_recipe_builder_prefers_checkpoint_args_and_rejects_wrong_manifold(tmp_path,monkeypatch):
    from test_controller_repair import small_training_fixture
    from Utils.reproducibility import state_fingerprint
    db,vq,norm = small_training_fixture(tmp_path)
    model = controller().eval()
    flow = tmp_path/'flow.pt'
    args = dict(vqvae_ckpt=str(vq),norm_stats=str(norm),style_slot_indices=list(model.selected_style_slots),
        use_style_code_selector=True,fingerprints={'norm_fingerprint':norm_stats_fingerprint(np.zeros(375,np.float32),np.ones(375,np.float32))})
    torch.save({'model':model.state_dict(),'args':args},flow)
    (tmp_path/'config.json').write_text(json.dumps({'vqvae_ckpt':'stale-path'}))
    payload,_ = recipe_builder._resolve_flow_config_dict(flow_ckpt=flow,flow_config=None)
    assert payload['vqvae_ckpt'] == str(vq)
    output = tmp_path/'recipe.json'
    argv = ['build','--db',str(db),'--flow-ckpt',str(flow),'--out',str(output),
            '--num-frames','5','--device','cpu']
    monkeypatch.setattr(sys,'argv',argv); recipe_builder.main()
    saved = json.loads(output.read_text())
    assert saved['metadata']['norm_fingerprint'] == args['fingerprints']['norm_fingerprint']
    assert saved['metadata']['vq_state_sha256'] == state_fingerprint(model.vqvae.state_dict())
    wrong = tmp_path/'wrong.npz'; np.savez(wrong,mean=np.ones(375,np.float32),std=np.ones(375,np.float32))
    monkeypatch.setattr(sys,'argv',argv+['--norm-stats',str(wrong)])
    with pytest.raises(ValueError,match='normalization'): recipe_builder.main()
    monkeypatch.setattr(sys,'argv',argv+['--strength','.5'])
    with pytest.raises(ValueError,match='Pure recipes'): recipe_builder.main()


def test_vectorized_tf_pure_recipe_matches_existing_path_and_never_reads_embedding(monkeypatch):
    model = controller().eval()
    model.eval_noise_std = 1.; model.eval_solver_steps = 2
    x = torch.randn(2,5,375)*.05
    recipes = torch.eye(16)[:4][None].repeat(2,1,1)
    recipes[1] = recipes[1].roll(3,-1)
    noise = case_noise_plan(model,[0,10],[90,99],5,seed=42)
    def forbidden(*args, **kwargs): raise AssertionError('Pure recipe cannot access a style embedding')
    monkeypatch.setattr(model.style_selector.style_embedding,'forward',forbidden)
    expected = reconstruct_motion_batch(model_name='flowmatch_target',model=model,x=x,
        style_label=None,manual_style_probs=recipes,manual_style_strength=1.,
        flow_eval_mode='reconstruct',flow_ode_steps=2,noise_plan=noise)
    actual = unseen.generate_pure_recipe(model,x,recipes,noise,mode='reconstruct',
        model_name='flowmatch_target',steps=2,microbatch=3)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual[:,:, :84],x[:,:, :84],rtol=0,atol=0)
    torch.testing.assert_close(actual[:,0],x[:,0],rtol=0,atol=0)
    smaller = unseen.generate_pure_recipe(model,x,recipes,noise,mode='reconstruct',
        model_name='flowmatch_target',steps=2,microbatch=1)
    torch.testing.assert_close(smaller,actual,atol=1e-6,rtol=1e-5)


def test_wrong_normalization_is_rejected_without_breaking_old_recipe_metadata():
    metadata = dict(vq_state_sha256='vq',selected_slots=[1,2],norm_fingerprint='old')
    with pytest.raises(ValueError,match='normalization'):
        validate_recipe_metadata(metadata,vq_fingerprint='vq',selected_slots=[1,2],norm_fingerprint='new')
    metadata.pop('norm_fingerprint')
    validate_recipe_metadata(metadata,vq_fingerprint='vq',selected_slots=[1,2],norm_fingerprint='new')


def test_selector_probability_recipe_preserves_token_exactly_and_large_weights_still_normalize():
    model=controller().eval()
    selector=model.style_selector
    book=model.vqvae.quantizer.embedding_weight
    label=selector(codebook_weight=book,style_label=torch.arange(3))
    pure=selector(codebook_weight=book,manual_probs=label['style_probs'].clone(),batch_size=3)
    torch.testing.assert_close(pure['style_probs'],label['style_probs'],rtol=0,atol=0)
    torch.testing.assert_close(pure['style_feature'],label['style_feature'],rtol=0,atol=0)
    huge=torch.full((4,16),1e38)
    p,mask=selector._normalize_manual_probs(huge,3)
    assert torch.isfinite(p).all() and mask.all()
    torch.testing.assert_close(p.sum(-1),torch.ones(3,4))


def test_seen_canonical_recipe_uses_independent_train_clips_and_topk(tmp_path,monkeypatch):
    segments=[]
    for label in range(2):
        for index in range(6):
            segments.append(dict(label=label,source_entry_id=label*10+index,
                frame_start=len(segments)*245,num_frames=245,split='train',mirror=False))
        for split in ('val','test'):
            segments.append(dict(label=label,source_entry_id=label*10+6+(split=='test'),
                frame_start=len(segments)*245,num_frames=245,split=split,mirror=False))
    data=write_db(tmp_path,num_frames=len(segments)*245,num_styles=2,segments=segments)
    for segment in segments:
        if segment['split']!='train':
            data[segment['frame_start']:segment['frame_start']+245]=np.nan
    data.tofile(tmp_path/'X.f32')
    model=controller().eval()
    def encode(x):
        assert torch.isfinite(x).all(), 'Canonical recipe read poisoned validation/test data'
        indices=torch.arange(x.size(1))[None,:,None].expand(x.size(0),-1,4)%16
        return None,indices,None
    monkeypatch.setattr(model.vqvae,'encode',encode)
    names=json.loads((tmp_path/'StyleNames.json').read_text())
    probs,meta=seen_eval._build_target_style_recipe_probs(model=model,recipe_db=tmp_path,
        target_style_names=names,classifier_class_names=names,target_style_count=2,
        seq_len=240,stride=30,exemplars_per_style=5,batch_size=2,seed=42,
        norm_mean=np.zeros(375,np.float32),norm_std=np.ones(375,np.float32),device=torch.device('cpu'))
    assert meta['recipe_profile']=='canonical' and meta['representation']=='topk_sparse'
    assert ((probs>0).sum(-1)<=model.style_selector.topk).all()
    torch.testing.assert_close(probs.sum(-1),torch.ones(2,4))
    for record in meta['chosen_clips']:
        assert len({s['source_entry_id'] for s in record['clips']})==5
        assert all(segments[s['segment_id']]['split']=='train' for s in record['clips'])


def test_actual_small_entry_exports_pure_recipes_and_excludes_gt_context(tmp_path,monkeypatch):
    source_db = tmp_path/'ind'; source_db.mkdir()
    source_segments = [dict(label=i,source_entry_id=i,frame_start=i*245,num_frames=245,
        split='test',suffix='FW') for i in range(3)]
    write_db(source_db,num_frames=735,num_styles=3,segments=source_segments)
    recipe_db = tmp_path/'ood'; recipe_db.mkdir()
    recipe_segments = [dict(label=i//3,source_entry_id=i,frame_start=i*245,num_frames=245,
        suffix='FW') for i in range(6)]
    write_db(recipe_db,num_frames=1470,num_styles=2,segments=recipe_segments)
    (recipe_db/'StyleNames.json').write_text(json.dumps(['oodA','oodB']))
    norm = tmp_path/'norm.npz'; np.savez(norm,mean=np.zeros(375,np.float32),std=np.ones(375,np.float32))
    ckpt = tmp_path/'flow.pt'; ckpt.write_bytes(b'fixture')
    classifier_path = tmp_path/'classifier.pth'; classifier_path.write_bytes(b'fixture')
    model = controller().eval()
    model.eval_noise_std = 1.; model.eval_solver_steps = 2
    model.eval_fingerprints = {'norm_fingerprint':norm_stats_fingerprint(np.zeros(375,np.float32),np.ones(375,np.float32))}
    names = json.loads((source_db/'StyleNames.json').read_text())
    classifier_names = ['oodA',names[0],'oodB',names[1],names[2]]
    class Classifier(torch.nn.Module): pass
    monkeypatch.setattr(unseen,'_resolve_norm_stats_path',lambda *args:norm)
    monkeypatch.setattr(unseen,'_resolve_flow_model_name',lambda *args:'flowmatch_target')
    monkeypatch.setattr(unseen,'load_motion_model',lambda **kwargs:model)
    monkeypatch.setattr(unseen,'build_fid_feature_extractor_from_checkpoint',lambda *args,**kwargs:
        (Classifier(),dict(class_names=classifier_names,window_size=240,fps=60,joint_set='ybot15')))
    lengths = []
    def stats(classifier, clips, targets, **kwargs):
        lengths.extend([clip.shape[0] for clip in clips])
        return targets.copy(),np.ones(len(clips)),np.ones(len(clips))
    monkeypatch.setattr(unseen,'_classifier_stats',stats)
    def forbidden(*args, **kwargs): raise AssertionError('OOD evaluation read style embedding')
    monkeypatch.setattr(model.style_selector.style_embedding,'forward',forbidden)
    out = tmp_path/'out'
    command = ['evaluate','--db',str(source_db),'--recipe-db',str(recipe_db),'--flow-ckpt',str(ckpt),
        '--classifier-ckpt',str(classifier_path),'--out-dir',str(out),'--device','cpu',
        '--num-sources','2','--targets-per-source','2','--exemplars-per-style','2','--ode-steps','2']
    monkeypatch.setattr(sys,'argv',command+['--prepare-only']); unseen.main()
    assert (out/'case_manifest.json').is_file() and not (out/'summary.json').exists()
    monkeypatch.setattr(sys,'argv',command); unseen.main()
    result = json.loads((out/'summary.json').read_text())
    assert result['num_cases']==4 and result['flow_eval_mode']=='rollout'
    assert set(lengths)=={240}
    assert all(s['bootstrap_samples']==1000 for s in result['summaries'])
    for path in (out/'recipes').glob('*.json'):
        payload = json.loads(path.read_text())
        probabilities,strength,label,_ = _load_style_recipe(path,style_names=names,
            selected_slots=model.selected_style_slots,num_embeddings=16,fallback_style_label=None,
            vq_fingerprint=payload['metadata']['vq_state_sha256'],norm_fingerprint=model.eval_fingerprints['norm_fingerprint'])
        assert label is None and strength==1
        np.testing.assert_allclose(probabilities.sum(-1),1)
    with pytest.raises(ValueError,match='already has results'): unseen.main()
