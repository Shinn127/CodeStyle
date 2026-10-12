"""Bounded visual diagnostics; these are not paper evaluation metrics."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from Visualization.cli.interactive_label_control import parse_args,load_session,demo_controls
from Visualization.io.genoview_bvh import (_load_template,_template_local_transforms,_expand_selected_to_full,
    _convert_to_geno_basis,GENOVIEW_RESOURCES_DIR)
from Visualization.pose_postprocess import PosePostprocessor
from Utils.reproducibility import state_fingerprint

args=parse_args(['--headless','--demo','--max-frames','120'])
session,info=load_session(args);vq=state_fingerprint(session.model.vqvae.state_dict())
template=_load_template(GENOVIEW_RESOURCES_DIR/'Geno_bind.bvh')
parents=np.asarray(template['parents']);offsets,_=_template_local_transforms(template,0)
processors=[PosePostprocessor(template['names'],parents,offsets,foot_ik=flag) for flag in (False,True)]
records={'raw':[],'smooth_fk':[],'smooth_fk_ik':[]};locked=[];cost=[]
for frame in range(120):
    session.step(*demo_controls(session),speed=1.2)
    before_motion=session.raw_motion.copy();before_noise=session.generator.get_state()
    p,r=session.selected_pose();p,r=_convert_to_geno_basis(p,r,enabled=session.coordinate_convert)
    p,r=_expand_selected_to_full(p,r,full_names=template['names'],full_parents=parents,template_local_pos=offsets,
        template_local_rot=_template_local_transforms(template,0)[1])
    records['raw'].append(p[0].copy())
    for name,processor in zip(('smooth_fk','smooth_fk_ik'),processors):
        start=time.perf_counter();filtered,_=processor.process(p[0],r[0]);elapsed=(time.perf_counter()-start)*1000
        records[name].append(filtered)
        if processor.foot_ik:cost.append(elapsed)
    locked.append([(s['active'] and s['weight']>=.99,s['kind']) for s in processors[-1].contacts])
    np.testing.assert_array_equal(session.raw_motion,before_motion)
    assert torch.equal(session.generator.get_state(),before_noise)
report={}
for name,frames in records.items():
    frames=np.array(frames);support=[];count=0
    for i in range(1,len(frames)):
        for leg,ids in enumerate(processors[-1].legs):
            a,kind=locked[i][leg];b,previous_kind=locked[i-1][leg]
            if a and b and kind==previous_kind:
                joint=ids[2+kind];support.append(float(np.linalg.norm((frames[i,joint]-frames[i-1,joint])[[0,2]])*100));count+=1
    sole=np.stack([np.minimum(frames[:,foot,1]-.08,frames[:,toe,1]-.02) for _,_,foot,toe in processors[-1].legs])
    lengths=np.linalg.norm(frames[:,1:]-frames[:,parents[1:]],axis=-1)
    report[name]=dict(world_second_difference_cm=float(np.linalg.norm(np.diff(frames,n=2,axis=0),axis=-1).mean()*100),
        locked_support_slide_cm_per_frame=float(np.mean(support)) if count else None,locked_support_pairs=count,
        locked_support_pair_coverage=count/(2*(len(frames)-1)),
        sole_penetration_max_cm=float(np.maximum(-sole,0).max()*100),
        bone_length_max_error_cm=float(np.abs(lengths-np.linalg.norm(offsets[1:],axis=-1)).max()*100))
assert vq==state_fingerprint(session.model.vqvae.state_dict())
reference=json.loads((ROOT/'analysis/interactive_label/headless.json').read_text())
np.testing.assert_array_equal(session.planner.position,reference['root_position'])
report.update(protocol='render_diagnostics_same_120frame_demo_not_paper_metrics',fp32=True,seed=42,
    raw_controller_feedback_unchanged=True,vq_unchanged=True,postprocess_ms_mean=float(np.mean(cost)),
    limitations='Sole heights are joint proxies; raw and processed support slide use the same final-IK contact intervals.')
(Path(__file__).parent/'diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
