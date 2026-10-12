"""Bounded real-asset interface checks; no training or aggregate accuracy claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from Visualization.cli.interactive_label_control import parse_args,load_session,demo_controls
from Utils.controller_validation import TeacherForcedStyleValidation,motion_quality_selection_key
from Utils.reproducibility import isolated_rng,state_fingerprint
from evaluation.simulate_realtime_flowmatch_control import reconstruct_sequence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path(__file__).with_name('rollout_verification.json'))
    opt=parser.parse_args()
    args=parse_args(['--flow-ckpt',str(opt.checkpoint),'--headless','--max-frames','600','--smooth-plan'])
    session,info=load_session(args);before=state_fingerprint(session.model.vqvae.state_dict())
    for frames in (60,600):
        session.reset();latency=[];t0=time.perf_counter()
        for i in range(frames):
            started=time.perf_counter();session.step(*demo_controls(session),speed=1.2)
            latency.append((time.perf_counter()-started)*1000)
            if not np.isfinite(session.raw_motion).all():raise AssertionError('Nonfinite motion')
            if i%20==0:
                torch.testing.assert_close(session.motion,session.model.vqvae.decode(session.latent),atol=0,rtol=0)
        info[str(frames)+'f']=dict(seconds=time.perf_counter()-t0,latency_ms_mean=float(np.mean(latency[10:])),
            latency_ms_p95=float(np.percentile(latency[10:],95)),finite=True,feedback_matches_decoder=True)
    config=json.loads((opt.checkpoint.parent/'config.json').read_text())
    mean,std=session.mean,session.std
    # Exercise the separate legacy simulator TF entry with actual two-frame state.
    db=Path(config['db']);meta=json.loads((db/'meta.json').read_text())
    source=np.memmap(db/'X.f32',dtype=np.float32,mode='r',shape=(meta['num_frames'],375))
    raw=np.array(source[info['initial_frame']:info['initial_frame']+12]);x=(raw-mean)/std
    pred,_=reconstruct_sequence(model=session.model,base_model_type='vqvae',x_ref_norm=x,
        style_label=session.style,manual_style_probs=None,manual_style_strength=1.,ode_steps=session.steps,noise_std=1.,
        solver='euler',device=session.device,requantize_z=True)
    assert pred.shape==x.shape and np.isfinite(pred).all()
    info['simulator_tf_12f_finite']=True
    with isolated_rng(1044):
        validator=TeacherForcedStyleValidation(session.model,db=db,mean=mean,std=std,
            classifier_path=config['tf_val_classifier'],clips=2,windows=1,batch_size=4,
            quality_metrics=True,eval_mode='rollout')
        started=time.perf_counter();metrics=validator.evaluate(session.model)
    assert metrics['matched_cases']==2 and metrics['cross_cases']==4
    assert metrics['matched_quality_cases']==2 and metrics['cross_quality_cases']==4
    assert metrics['quality_protocol'].endswith('240f')
    for k in ('matched_accuracy','cross_accuracy','jitter_cm','cross_jitter_cm'):
        assert np.isfinite(metrics[k])
    info['validation_6case_smoke']=dict(seconds=time.perf_counter()-started,metrics=metrics,
        selection_key=motion_quality_selection_key(metrics),
        limitation='Tiny interface smoke on a 100-update checkpoint, not a training result or accuracy estimate.')
    assert state_fingerprint(session.model.vqvae.state_dict())==before
    info['vq_unchanged']=True;info['protocol']='bounded_history_phase_interface_smoke'
    opt.report.write_text(json.dumps(info,ensure_ascii=False,indent=2));print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
