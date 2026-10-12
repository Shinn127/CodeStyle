"""Read-only style intervention: identical history/noise, six requested styles.

Three existing checkpoints, one validation source, 240 generated frames each.
This is a bounded diagnostic, not an aggregate evaluation or training experiment.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from Visualization.cli.interactive_label_control import parse_args,load_session
from Utils.motion_eval import reconstruct_motion_batch
from Utils.reproducibility import case_noise_plan,state_fingerprint
from Utils.controller_validation import classifier_label_mapping
from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint
from evaluation.evaluate_minor_revision_generalization import _classifier_stats


def main():
    out=Path(__file__).parent
    folder=ROOT/'checkpoints/flowmatch_label_history_ode4_100ep'
    names=json.loads((ROOT/'database_100sty_375_memmap_nomirror/StyleNames.json').read_text())
    styles=['Neutral','ArmsAboveHead','ArmsFolded','Robot','HighKnees','Old']
    labels=torch.tensor([names.index(n) for n in styles],device='cuda')
    rows=[]
    for file in ('epoch_0050.pt','best.pt','latest.pt'):
        session,info=load_session(parse_args(['--flow-ckpt',str(folder/file),'--headless','--max-frames','1']))
        model=session.model;before=state_fingerprint(model.vqvae.state_dict())
        config=json.loads((folder/'config.json').read_text());db=Path(config['db'])
        segments=json.loads((db/'segments.json').read_text());meta=json.loads((db/'meta.json').read_text())
        segment=next(s for s in segments if s['split']=='val' and s['label']==names.index('Neutral')
                     and s['suffix']=='FW' and s['num_frames']>=245)
        start=segment['frame_start']+2
        source=np.memmap(db/'X.f32',dtype=np.float32,mode='r',shape=(meta['num_frames'],375))
        raw=np.array(source[start-1:start+241]);x=torch.from_numpy((raw-session.mean)/session.std).cuda()
        with torch.no_grad():
            q=model.frame_latent(x[:3]);obs=x[1:2].repeat(len(styles),1)
            obs[:,84:]=model.vqvae.decode(q[1:2]).repeat(len(styles),1)
            features=model.encode_conditions(x_curr=obs,style_label=labels)[2]
            all_features=model.selector_output_all_styles()['style_feature']
            pair=torch.pdist(all_features)
            noise=case_noise_plan(model,[start],[names.index('Neutral')],241,seed=42).repeat(len(styles),1,1)
            h=q[0:1].repeat(len(styles),1);p=q[1:2].repeat(len(styles),1)
            continuous=model.sample_latent(x_curr=obs,style_label=labels,prev_latent=p,history_latent=h,
                                          z0=noise[:,0],num_steps=4)
            motion,z=model.feedback_state(continuous)
            book=z.reshape(len(styles),model.vqvae.num_latents,model.vqvae.code_dim)
            pose=motion[:,3:75]*session.std_t[87:159]+session.mean_t[87:159]
            pose_changes=(pose-pose[0]).reshape(len(styles),24,3).norm(dim=-1).mean(-1)*100
            step_changes=(model.normalize_latent(continuous)-model.normalize_latent(continuous[:1])).norm(dim=-1)
            full_x=x[1:].unsqueeze(0).repeat(len(styles),1,1)
            generated=reconstruct_motion_batch(model_name='flowmatch_target',model=model,x=full_x,style_label=labels,
                noise_plan=noise,flow_eval_mode='rollout',flow_ode_steps=4,flow_noise_std=1.,flow_requantize_z=True,flow_latent_clip=True)
            classifier,metadata=build_fid_feature_extractor_from_checkpoint(Path(config['tf_val_classifier']),map_location='cpu')
            classifier.cuda().eval().requires_grad_(False)
            mapping=classifier_label_mapping(names,metadata,device='cuda')
            gt=np.array(raw[1:])
            ae=np.array(gt,copy=True)
            ae[:,84:]=model.vqvae.decode(model.frame_latent(x[1:])).cpu().numpy()*session.std[84:]+session.mean[84:]
            references=np.stack((gt,ae))
            reference_predictions,_,_=_classifier_stats(classifier,references[:,1:],
                np.repeat(mapping[labels[:1]].cpu().numpy(),2),device=torch.device('cuda'),
                classifier_window=240,classifier_stride=240,classifier_batch_size=2,joint_set=metadata['joint_set'])
            motion_raw=generated.cpu().numpy()*session.std+session.mean
            predicted,_,_=_classifier_stats(classifier,motion_raw[:,1:],mapping[labels].cpu().numpy(),
                device=torch.device('cuda'),classifier_window=240,classifier_stride=240,classifier_batch_size=len(styles),joint_set=metadata['joint_set'])
            many_obs=obs[:1].repeat(32,1);many_h=h[:1].repeat(32,1);many_p=p[:1].repeat(32,1)
            many_noise=case_noise_plan(model,list(range(32)),[names.index('Neutral')]*32,2,seed=42)
            sampled=model.sample_latent(x_curr=many_obs,style_label=labels[:1].repeat(32),prev_latent=many_p,
                history_latent=many_h,z0=many_noise[:,0],num_steps=4)
            sampled_motion,_=model.feedback_state(sampled)
            sample_pose=(sampled_motion[:,3:75]*session.std_t[87:159]+session.mean_t[87:159]).reshape(32,24,3)
            noise_pose_std_cm=float(sample_pose.std(0,unbiased=False).norm(dim=-1).mean()*100)
        # Verify the interactive bridge itself reaches the requested feature.
        session.set_style(names.index('ArmsAboveHead'))
        for _ in range(16):session.step()
        reached=torch.allclose(session.current_feature,session.feature(names.index('ArmsAboveHead')),atol=0,rtol=0)
        assert reached and state_fingerprint(model.vqvae.state_dict())==before
        row=dict(checkpoint=file,validation_start=start,source_entry_id=segment['source_entry_id'],source_style='Neutral',
            requested_styles=styles,all_style_feature_pairwise_mean=float(pair.mean()),all_style_feature_pairwise_min=float(pair.min()),
            first_continuous_change_norm=step_changes.cpu().tolist(),first_pose_change_cm=pose_changes.cpu().tolist(),
            first_code_change_fraction=(book!=book[0:1]).any(-1).float().mean(-1).cpu().tolist(),
            rollout_classifier_predictions=[metadata['class_names'][int(i)] for i in predicted],
            gt_and_quantized_ae_classifier=[metadata['class_names'][int(i)] for i in reference_predictions],
            fixed_history_32_noise_pose_std_cm=noise_pose_std_cm,
            interactive_reaches_target_feature=reached)
        rows.append(row);print(json.dumps(row),flush=True)
        del model,session,classifier
    report=dict(protocol='one_validation_source_six_styles_shared_noise_240f_Euler4_FP32_seed42',rows=rows,
                limitation='One source/style probe; cannot estimate aggregate accuracy or establish training causality.')
    (out/'style_intervention.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
