from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _memmap_fixtures import write_db
from Utils.fid_feature_extractor import (MotionFeatureClassifier, YBOT15_TO_SELECTED24_INDICES,
    build_motion_feature_classifier_from_checkpoint)
import training.train_fid_feature_extractor_camdm as trainer
import evaluation.evaluate_minor_revision_generalization as evaluator


def test_default_classifier_is_native_60fps_four_seconds(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trainer'])
    args = trainer.parse_args()
    assert args.window_size == 240 and trainer.LOCAL_FPS == 60
    assert args.out_dir.name.endswith('60fps_240f_nomirror')
    assert evaluator._resolve_evaluation_horizons(None, 240) == [240]
    assert evaluator._resolve_evaluation_horizons([240, 480], 240) == [240, 480]
    with pytest.raises(ValueError, match='at least 240'):
        evaluator._resolve_evaluation_horizons([60], 240)


def test_native_windows_have_240_consecutive_frames_and_do_not_cross_clips(tmp_path):
    db = tmp_path/'db'; db.mkdir()
    raw = write_db(db, num_frames=500, num_styles=2, segments=[
        dict(frame_start=0, num_frames=250, label=0, split='train'),
        dict(frame_start=250, num_frames=250, label=1, split='train')])
    dataset = trainer.CamdmWindowDataset(db=db, window_size=240, clip_offset=10)
    assert dataset.starts.tolist() == [0, 10, 250, 260]
    for i, start in enumerate(dataset.starts):
        window, label = dataset[i]
        expected = raw[start:start+240,87:159].reshape(240,24,3)[:,YBOT15_TO_SELECTED24_INDICES].reshape(240,45)
        np.testing.assert_array_equal(window.numpy(), expected)
        assert window.dtype == torch.float32 and int(label) == i//2


def test_real_240_frame_classifier_training_save_load_and_output_guard(tmp_path, monkeypatch):
    db = tmp_path/'db'; db.mkdir()
    write_db(db, num_frames=500, num_styles=2, segments=[
        dict(frame_start=0, num_frames=250, label=0, split='train'),
        dict(frame_start=250, num_frames=250, label=1, split='train')])
    out = tmp_path/'classifier'
    monkeypatch.setattr(sys,'argv',['trainer','--db',str(db),'--out-dir',str(out),
        '--epochs','1','--batch-size','4','--num-workers','0','--device','cpu','--log-interval','0'])
    trainer.main()  # One optimizer update on a disposable fixture, including the 240-frame FC layer.
    model, metadata = build_motion_feature_classifier_from_checkpoint(out/'100style_position_classifier.pth')
    assert metadata['window_size'] == 240 and metadata['fps'] == 60 and metadata['window_seconds'] == 4
    assert metadata['temporal_stride'] == 1 and metadata['acc_split'] == 'train'
    assert metadata['precision'] == 'float32' and not metadata['tf32']
    assert model.fc1.in_features == 128*240
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.no_grad():
        features, logits = model(torch.randn(2,45,240))
    assert features.shape == (2,512) and logits.shape == (2,2)
    with pytest.raises(ValueError, match='window size 240'):
        model(torch.randn(2,45,60))
    with pytest.raises(ValueError, match='already contains'):
        trainer.main()


def test_legacy_classifier_without_window_metadata_still_loads_60_frames(tmp_path):
    original = MotionFeatureClassifier(in_feat=45,class_num=2,window_size=60).eval()
    path = tmp_path/'legacy.pth'
    torch.save(dict(state_dict=original.state_dict(),in_feat=45,class_names=['a','b']),path)
    restored, metadata = build_motion_feature_classifier_from_checkpoint(path)
    assert restored.window_size == metadata['window_size'] == 60
    x = torch.randn(2,45,60)
    with torch.no_grad():
        torch.testing.assert_close(restored(x)[1],original(x)[1],rtol=0,atol=0)


@pytest.mark.parametrize('window,legacy',[(240,False),(60,True)])
def test_generalization_entry_classifies_only_generated_240_frames_and_supports_legacy(tmp_path,monkeypatch,window,legacy):
    db=tmp_path/'db'; db.mkdir()
    segments=[]; start=0
    for split in ['train','test']:
        for label in range(2):
            segments.append(dict(frame_start=start,num_frames=300,label=label,split=split,
                source_entry_id=len(segments)))
            start+=300
    write_db(db,num_frames=start,num_styles=2,segments=segments)
    norm=tmp_path/'norm.npz'; np.savez(norm,mean=np.zeros(375,np.float32),std=np.ones(375,np.float32))
    flow=tmp_path/'flow.pt'; flow.write_bytes(b'fixture')
    classifier=tmp_path/'classifier.pth'; classifier.write_bytes(b'fixture')
    out=tmp_path/'evaluation'
    seen=[]; calls=[]
    class Classifier(torch.nn.Module):
        def forward(self,x):
            assert x.shape[1:]==(45,window)
            seen.append(x.detach().clone())
            return x.new_zeros((len(x),512)),x.new_zeros((len(x),2))
    class Flow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('latent_offset',torch.zeros(8))
            self.register_buffer('latent_scale',torch.ones(8))
            self.register_buffer('latent_transport_std',torch.ones(8))
            self.latent_dim=8; self.eval_noise_std=1.; self.eval_solver_steps=8
            self.inference_requantize=True; self.eval_fingerprints={}
    monkeypatch.setattr(evaluator,'_resolve_flow_model_name',lambda *args:'flowmatch_target')
    monkeypatch.setattr(evaluator,'load_motion_model',lambda **kwargs:Flow())
    monkeypatch.setattr(evaluator,'build_fid_feature_extractor_from_checkpoint',lambda *args,**kwargs:
        (Classifier(),dict(window_size=window,fps=60.,joint_set='ybot15',in_feat=45,
            class_num=2,class_names=['style_0','style_1'])))
    def reconstruct(**kwargs):
        expected=window if legacy else window+1
        assert kwargs['x'].shape[1]==expected
        assert kwargs['noise_plan'].shape[1:]==(expected-1,8)
        calls.append(expected)
        prediction=kwargs['x'].clone()
        prediction[:,0,87:159]=999.  # Must never reach the new classifier; legacy replay deliberately includes it.
        return prediction
    monkeypatch.setattr(evaluator,'reconstruct_motion_batch',reconstruct)
    argv=['eval','--db',str(db),'--flow-ckpt',str(flow),'--classifier-ckpt',str(classifier),
        '--norm-stats',str(norm),'--out-dir',str(out),'--num-sources','2','--targets-per-source','1',
        '--batch-size','2','--num-workers','0','--device','cpu','--flow-eval-mode','reconstruct']
    if legacy: argv+=['--include-gt-context']
    monkeypatch.setattr(sys,'argv',argv)
    evaluator.main()
    assert len(calls)==2 and len(seen)==6  # One classifier call per source for real/matched/cross.
    assert all(bool((x[:,:,0]==999).all())==legacy for x in seen[2:])
    report=json.loads((out/'minor_revision_generalization_summary.json').read_text())
    cfg=report['config']
    assert cfg['classifier_window']==window and cfg['horizons']==[window]
    assert cfg['classifier_window_seconds']==window/60.
    assert cfg['include_gt_context']==legacy and cfg['effective_flow_requantize_z']
    assert cfg['protocol_version']==(2 if legacy else 3)
    assert ('teacher_forced' in report['summaries'][1]['setting']) != legacy
    manifest=json.loads((out/'case_manifest.json').read_text())
    assert len(manifest['horizons'][str(window)])==2
