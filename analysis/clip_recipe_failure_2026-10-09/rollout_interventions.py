"""Small frozen validation study, not training or an official benchmark."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Utils.motion_eval import load_motion_model
from Utils.runtime import load_norm_stats,configure_torch_runtime
from Utils.reproducibility import case_noise_plan,state_fingerprint,file_sha256
from Utils.vq_target_cache import VQTargetCache
from Utils.style_code_prior import sparse_code_histogram
from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint
from evaluation.evaluate_minor_revision_generalization import _classifier_stats,_root_ade_cm


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'rollout_interventions.json')
    p.add_argument('--checkpoints',nargs='+',choices=('clip_recipe_gt_history50','clip_recipe_scratch150'),
        default=('clip_recipe_gt_history50','clip_recipe_scratch150'))
    p.add_argument('--variants',nargs='+',choices=('baseline','cycle_projection','output_prior_0.25','output_prior_1.0'),
        default=('baseline','cycle_projection','output_prior_0.25','output_prior_1.0'))
    args=p.parse_args();device=torch.device(args.device)
    torch.set_num_threads(2);torch.manual_seed(42)
    configure_torch_runtime(device=device,tf32=False,cudnn_benchmark=False,matmul_precision='highest')
    mean,std=load_norm_stats(ROOT/'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep/norm_stats.npz')
    db=ROOT/'database_100sty_375_memmap';meta=json.loads((db/'meta.json').read_text())
    raw=np.memmap(db/'X.f32',mode='r',dtype=np.float32,shape=(meta['num_frames'],375))
    names=json.loads((db/'StyleNames.json').read_text())
    cases=json.loads((ROOT/'checkpoints/clip_recipe_gt_history50/tf_validation_cases.json').read_text())
    picks=np.random.default_rng(42).choice(len(cases['starts']),8,replace=False)
    starts=[cases['starts'][i] for i in picks];source_labels=[cases['source_labels'][i] for i in picks]
    targets=[names[i] for i in source_labels]+['BigSteps','March','Skip','Morris','Teapot','OnPhoneLeft','WildArms','Zombie']
    starts+=starts.copy()
    source_raw=np.stack([np.array(raw[s:s+241]) for s in starts]);x=torch.from_numpy((source_raw-mean)/std).to(device)
    classifier,cm=build_fid_feature_extractor_from_checkpoint(
        ROOT/'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth',map_location='cpu')
    classifier.to(device).eval().requires_grad_(False)
    target_classes=np.array([cm['class_names'].index(n) for n in targets])
    cache=VQTargetCache(ROOT/'cache/vq_targets_vq150ep_repair_v2')
    report=json.loads((ROOT/'eval_results/clip_recipe_practical_test/minor_revision_generalization_summary.json').read_text())
    indices={}
    for row in report['config']['recipe_meta']['chosen_clips']:
        name=names[row['target_label']]
        if name in targets:
            indices[name]=np.concatenate([np.asarray(cache.indices[c['frame_start']:c['frame_start']+240]) for c in row['clips']])
    udb=ROOT/'database_100sty_375_memmap_zeroshot';um=json.loads((udb/'meta.json').read_text())
    uraw=np.memmap(udb/'X.f32',mode='r',dtype=np.float32,shape=(um['num_frames'],375))
    plans=json.loads((ROOT/'eval_results/clip_recipe_practical_unseen/case_manifest.json').read_text())['recipes']
    results={}
    for checkpoint_name in args.checkpoints:
        path=ROOT/'checkpoints'/checkpoint_name/'best.pt'
        model=load_motion_model(model_name='flowmatch_target',ckpt_path=path,num_styles=90).to(device).eval()
        before=state_fingerprint(model.vqvae.state_dict())
        if 'WildArms' not in indices:
            for name in ('WildArms','Zombie'):
                plan=next(q for q in plans if q['style_name']==name)
                parts=[]
                for c in plan['clips']:
                    if c['role']=='exemplar':
                        sample=torch.from_numpy((np.array(uraw[c['start']:c['start']+240])-mean)/std).to(device)
                        parts.append(model.vqvae.encode(sample)[1].cpu().numpy())
                indices[name]=np.concatenate(parts)
        h=torch.from_numpy(np.stack([sparse_code_histogram(indices[name],selected_slots=model.selected_style_slots,
            num_embeddings=2048,topk=2048) for name in targets])).to(device)
        probs=model.recipe_from_histogram(h)
        style=model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=probs)[2]
        noise=case_noise_plan(model,starts,target_classes,241,seed=42,noise_std=1.)
        variants=args.variants if checkpoint_name=='clip_recipe_gt_history50' else ('baseline',)
        summaries={}
        for variant in variants:
            current=x[:,0];previous=model.frame_latent(current);motions=[];cycle_change=[]
            for step in range(240):
                latent=model.sample_latent(x_curr=current,style_label=None,prev_latent=previous,
                    style_feature=style,z0=noise[:,step],num_steps=8,noise_std=1.)
                motion,z=model.feedback_state(latent)
                next_traj=x[:,step+1,:84]
                if variant=='cycle_projection':
                    projected=model.frame_latent(torch.cat((next_traj,motion),-1))
                    cycle_change.append((projected.reshape(16,32,4)!=z.reshape(16,32,4)).any(-1).float().mean())
                    motion,z=model.feedback_state(projected)
                elif variant.startswith('output_prior_'):
                    strength=float(variant.rsplit('_',1)[1]);book=model.vqvae.quantizer.embedding_weight
                    clipped=model.clip_latent_to_data_bounds(latent)
                    controlled=clipped.reshape(16,32,4)[:,model.selected_style_slots]
                    distance=((controlled[:,:,None]-book[None,None]).square().sum(-1)
                        / book.var(0,unbiased=False).sum().clamp_min(1e-8))
                    # Full reference histogram is a soft output-code prior, never a GT target sequence.
                    prior=(.9*h+.1/2048).clamp_min(1e-8)
                    selected=(distance-strength*prior.log()).argmin(-1)
                    z=z.clone().reshape(16,32,4);z[:,model.selected_style_slots]=book[selected]
                    z=z.reshape(16,128);motion=model.vqvae.decode(z)
                current=torch.cat((next_traj,motion),-1);previous=z;motions.append(motion)
            generated=source_raw.copy()
            generated[:,1:,84:]=torch.stack(motions,1).cpu().numpy()*std[84:]+mean[84:]
            prediction,confidence,_=_classifier_stats(classifier,generated[:,1:],target_classes,device=device,
                classifier_window=240,classifier_stride=240,classifier_batch_size=16,joint_set=cm['joint_set'])
            groups={}
            for group,(a,b) in dict(matched=(0,8),dynamic_cross=(8,12),pose_cross=(12,14),ood=(14,16)).items():
                groups[group]=dict(correct=int((prediction[a:b]==target_classes[a:b]).sum()),n=b-a,
                    root_ade_cm=float(np.mean([_root_ade_cm(r,g,dt=1/60,context_frames=1) for r,g in zip(source_raw[a:b],generated[a:b])])),
                    position_jitter_cm=float(np.mean(np.linalg.norm(np.diff(generated[a:b,1:,87:159].reshape(b-a,240,24,3),n=3,axis=1),axis=-1))*100))
            summaries[variant]=dict(groups=groups,predictions=[cm['class_names'][i] for i in prediction],
                target_confidence=confidence.tolist(),cycle_changed_fraction=float(torch.stack(cycle_change).mean()) if cycle_change else None)
            print(checkpoint_name,variant,groups,flush=True)
        assert state_fingerprint(model.vqvae.state_dict())==before
        results[checkpoint_name]=dict(checkpoint_sha256=file_sha256(path),vq_unchanged=True,variants=summaries)
        del model
    args.out.write_text(json.dumps(dict(seed=42,fp32=True,mode='rollout',output_prior_clip_before_distance=True,
        sources=starts,target_names=targets,
        results=results,limitations='16 post-hoc validation cases; not aggregate benchmark estimates or OOD hyperparameter selection. All variants start from the same source GT and use the same Gaussian cases. Output-prior variants change final code selection and require richer recipe metadata; they are research diagnostics, not deployed methods. No optimizer updates or GT root overwrite.'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
