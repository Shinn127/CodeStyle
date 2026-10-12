"""Frozen-checkpoint diagnostics; no optimizer updates, rollout or benchmark replacement."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utils.motion_eval import load_motion_model
from Utils.runtime import load_norm_stats, configure_torch_runtime
from Utils.reproducibility import case_noise_plan, state_fingerprint, file_sha256
from Utils.style_code_prior import sparse_code_histogram, TrainStyleRecipePool
from Utils.style_donor_bank import retarget_motion, StyleDonorBank
from Utils.vq_target_cache import VQTargetCache
from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint
from evaluate_minor_revision_generalization import _classifier_stats, _root_ade_cm
from dataset import build_train_val_test_dataloaders
from train_flowmatch_vqvae_controller import controller_collate

OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--guidance-only', action='store_true')
    parser.add_argument('--history-only', action='store_true', help='Frozen input ablations only; skip training-gradient probes')
    parser.add_argument('--flow-ckpt', type=Path, default=ROOT/'checkpoints/clip_recipe_scratch150/best.pt')
    parser.add_argument('--seen-report', type=Path, default=ROOT/'eval_results/clip_recipe_revision_test/minor_revision_generalization_summary.json')
    parser.add_argument('--unseen-manifest', type=Path, default=ROOT/'eval_results/clip_recipe_revision_unseen/case_manifest.json')
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2); torch.manual_seed(42)
    configure_torch_runtime(device=device, tf32=False, cudnn_benchmark=False, matmul_precision='highest')
    ckpt = args.flow_ckpt
    model = load_motion_model(model_name='flowmatch_target', ckpt_path=ckpt, num_styles=90).to(device).eval()
    vq_before = state_fingerprint(model.vqvae.state_dict())
    mean, std = load_norm_stats(ROOT/'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep/norm_stats.npz')
    db = ROOT/'database_100sty_375_memmap'
    names = json.loads((db/'StyleNames.json').read_text())
    segments = json.loads((db/'segments.json').read_text())
    meta = json.loads((db/'meta.json').read_text())
    raw = np.memmap(db/'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
    cache = VQTargetCache(ROOT/'cache/vq_targets_vq150ep_repair_v2')
    classifier, cm = build_fid_feature_extractor_from_checkpoint(
        ROOT/'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth', map_location='cpu')
    classifier.to(device).eval().requires_grad_(False)
    assert cm['window_size'] == 240 and cm['fps'] == 60
    class_names = cm['class_names']
    report = json.loads(args.seen_report.read_text())
    histograms = {}
    for record in report['config']['recipe_meta']['chosen_clips']:
        codes = np.concatenate([np.asarray(cache.indices[c['frame_start']:c['frame_start']+240]) for c in record['clips']])
        histograms[names[record['target_label']]] = sparse_code_histogram(codes,
            selected_slots=model.selected_style_slots, num_embeddings=2048, topk=2048)
    unseen_db = ROOT/'database_100sty_375_memmap_zeroshot'
    um = json.loads((unseen_db/'meta.json').read_text())
    uraw = np.memmap(unseen_db/'X.f32', mode='r', dtype=np.float32, shape=(um['num_frames'],375))
    uplans = json.loads(args.unseen_manifest.read_text())['recipes']
    for plan in uplans:
        codes = []
        for clip in plan['clips']:
            if clip['role'] != 'exemplar': continue
            x = torch.from_numpy((np.array(uraw[clip['start']:clip['start']+240])-mean)/std).to(device)
            with torch.no_grad(): codes.append(model.vqvae.encode(x)[1].cpu().numpy())
        histograms[plan['style_name']] = sparse_code_histogram(np.concatenate(codes),
            selected_slots=model.selected_style_slots, num_embeddings=2048, topk=2048)
    all_hist = torch.from_numpy(np.stack(list(histograms.values()))).to(device)
    with torch.no_grad():
        learned, logits = model.recipe_from_histogram(all_hist, return_logits=True)
        idx = torch.argsort(all_hist, descending=True, stable=True)[..., :4]
        plain = torch.zeros_like(all_hist).scatter(-1, idx, all_hist.gather(-1,idx))
        plain /= plain.sum(-1, keepdim=True)
        representation = dict(histogram_mass_raw_topk=float((all_hist*(plain>0)).sum(-1).mean()),
            histogram_mass_learned_topk=float((all_hist*(learned>0)).sum(-1).mean()),
            mean_topk_overlap=float(((plain>0)&(learned>0)).sum(-1).float().mean()/4),
            residual_saturation_fraction=float(((logits-all_hist.clamp_min(1e-8).log()).abs()>3.8).float().mean()))
    keys = list(histograms)
    manifest = json.loads((ckpt.parent/'tf_validation_cases.json').read_text())
    picked = np.random.default_rng(42).choice(len(manifest['starts']),8,replace=False)
    starts = [manifest['starts'][i] for i in picked]
    source_labels = [manifest['source_labels'][i] for i in picked]
    target_names = [names[i] for i in source_labels]+['BigSteps','March','Skip','Morris','Teapot','OnPhoneLeft','WildArms','Zombie']
    starts += starts.copy()
    source_raw = np.stack([np.array(raw[s:s+241]) for s in starts])
    requested = np.array([class_names.index(n) for n in target_names])
    probabilities = learned[[keys.index(n) for n in target_names]]
    raw_probabilities = plain[[keys.index(n) for n in target_names]]
    x = torch.from_numpy((source_raw-mean)/std).to(device)
    current = x[:,:-1].reshape(-1,375)
    noises = case_noise_plan(model,starts,requested,241,seed=42,noise_std=1.).reshape(-1,128)
    oracle_raw=[];oracle_same_action=[]
    for i,name in enumerate(target_names):
        if i<8: oracle_raw.append(source_raw[i,:-1].copy());oracle_same_action.append(True);continue
        source_segment=next(s for s in segments if s['frame_start']<=starts[i]<s['frame_start']+s['num_frames'])
        if name in names:
            possible=[s for s in segments if s['style_name']==name and s['split']=='val' and not s['mirror'] and s['num_frames']>=240]
            matched=[s for s in possible if s['suffix']==source_segment['suffix']]
            segment=(matched or possible)[0]
            start=segment['frame_start']+(segment['num_frames']-240)//2
            oracle_raw.append(np.array(raw[start:start+240]));oracle_same_action.append(bool(matched))
        else:
            plan=next(p for p in uplans if p['style_name']==name)
            clip=next(c for c in plan['clips'] if c['role']=='reference')
            oracle_raw.append(np.array(uraw[clip['start']:clip['start']+240]));oracle_same_action.append(None)
    donor=torch.from_numpy(np.stack(oracle_raw)[:,:,84:]).to(device)
    donor=retarget_motion(torch.from_numpy(source_raw[:,:-1,84:87]).to(device),donor)
    hybrid=torch.cat((torch.from_numpy(source_raw[:,:-1,:84]).to(device),donor),-1)
    hybrid=((hybrid-torch.as_tensor(mean,device=device))/torch.as_tensor(std,device=device)).reshape(-1,375)
    summaries={}
    def summarize(name,motions):
        generated=source_raw.copy();generated[:,1:,84:]=motions.reshape(16,240,291).cpu().numpy()*std[84:]+mean[84:]
        pred,conf,_=_classifier_stats(classifier,generated[:,1:],requested,device=device,
            classifier_window=240,classifier_stride=240,classifier_batch_size=16,joint_set=cm['joint_set'])
        groups={}
        for group,(a,b) in dict(matched=(0,8),dynamic_cross=(8,12),pose_cross=(12,14),ood=(14,16)).items():
            groups[group]=dict(correct=int((pred[a:b]==requested[a:b]).sum()),n=b-a,
                root_ade_cm=float(np.mean([_root_ade_cm(r,g,dt=1/60,context_frames=1) for r,g in zip(source_raw[a:b],generated[a:b])])) )
        summaries[name]=dict(groups=groups,predicted=[class_names[int(p)] for p in pred],target_confidence=conf.tolist())
        print(name,groups,flush=True)
    def encode(rows):
        return torch.cat([model.frame_latent(rows[i:i+512]) for i in range(0,len(rows),512)])
    def sample(rows,prev,probs):
        feature=model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=probs)[2]
        feature=feature[:,None].expand(-1,240,-1).reshape(-1,64)
        return torch.cat([model.sample_latent(x_curr=rows[i:i+512],prev_latent=prev[i:i+512],
            style_label=None,style_feature=feature[i:i+512],z0=noises[i:i+512],num_steps=8,noise_std=1.)
            for i in range(0,len(rows),512)])
    def decode(z,quantize=True,clip=True):
        return torch.cat([model.feedback_state(z[i:i+512],requantize=quantize,latent_clip=clip)[0] for i in range(0,len(z),512)])
    if args.guidance_only:
        with torch.no_grad():
            previous=encode(current)
            source_probabilities=learned[[keys.index(names[label]) for label in source_labels+source_labels]]
            positive=model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=probabilities)[2]
            reference=model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=source_probabilities)[2]
            positive=positive[:,None].expand(-1,240,-1).reshape(-1,64)
            reference=reference[:,None].expand(-1,240,-1).reshape(-1,64)
            def guided(scale):
                latents=[]
                for begin in range(0,len(current),512):
                    rows=current[begin:begin+512]
                    trajectory,motion=model.split_traj_motion(rows)
                    prev=model.normalize_latent(previous[begin:begin+512])
                    z=model.normalize_latent(noises[begin:begin+512])
                    pos=positive[begin:begin+512];ref=reference[begin:begin+512]
                    pos_tokens=model.field.prepare_conditions(prev,motion,trajectory,pos)
                    ref_tokens=model.field.prepare_conditions(prev,motion,trajectory,ref)
                    for step in range(8):
                        t=z.new_full((len(rows),1),step/8)
                        vpos=model._predict_velocity(z_t=z,z_prev=prev,t=t,e_c=motion,e_traj=trajectory,
                            e_s=pos,condition_tokens=pos_tokens)[0]
                        vref=model._predict_velocity(z_t=z,z_prev=prev,t=t,e_c=motion,e_traj=trajectory,
                            e_s=ref,condition_tokens=ref_tokens)[0]
                        z=z+(vpos+(scale-1)*(vpos-vref))/8
                    latents.append(model.denormalize_latent(z))
                return torch.cat(latents)
            baseline=sample(current,previous,probabilities)
            torch.testing.assert_close(guided(1.),baseline,atol=0,rtol=0)
            summarize('guidance_1',decode(baseline))
            for scale in (2.,3.): summarize(f'guidance_{int(scale)}',decode(guided(scale)))
        assert state_fingerprint(model.vqvae.state_dict())==vq_before
        result=dict(checkpoint_sha256=file_sha256(ckpt),device=str(device),fp32=True,seed=42,
            sources=starts,target_names=target_names,summaries=summaries,scale1_exact=True,vq_unchanged=True,
            limitations='16 post-hoc validation cases, not a benchmark. Experimental contrast guidance uses an additional source-style reference recipe derived from known source labels and train exemplars. It is not standard unconditional CFG, and cannot replace the original target-recipe-only evaluation without reporting that extra input. No training or rollout.')
        (args.out_dir/'guidance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return
    with torch.no_grad():
        previous=encode(current);z=sample(current,previous,probabilities)
        summarize('source_history_quantized',decode(z))
        summarize('source_history_continuous',decode(z,False,True))
        if args.history_only:
            summarize('source_history_unclipped_continuous',decode(z,False,False))
            summarize('source_history_unclipped_quantized',decode(z,True,False))
        summarize('selector_bypass_raw_topk',decode(sample(current,previous,raw_probabilities)))
        h_previous=encode(hybrid)
        summarize('oracle_target_history_ae',model.vqvae.decode(h_previous))
        if args.history_only:
            summarize('oracle_target_history_clipped_ae',decode(h_previous))
        summarize('oracle_target_history',decode(sample(hybrid,h_previous,probabilities)))
        masked=current.clone();masked[:,87:]=0
        zero_previous=model.denormalize_latent(torch.zeros_like(previous))
        summarize('joint_history_mask',decode(sample(masked,zero_previous,probabilities)))
    if args.history_only:
        assert state_fingerprint(model.vqvae.state_dict())==vq_before
        args.out_dir.mkdir(parents=True,exist_ok=True)
        result=dict(checkpoint_path=str(ckpt.resolve()),checkpoint_sha256=file_sha256(ckpt),
            device=str(device),fp32=True,seed=42,sources=starts,target_names=target_names,
            oracle_same_action=oracle_same_action,representation=representation,history=summaries,vq_unchanged=True,
            limitations='16 post-hoc validation cases, not population estimates. Oracle target history supplies GT each frame and may change action/content. Raw TopK bypass changes the condition distribution; masking preserves root/trajectory but removes phase. No optimizer updates or rollout.')
        result['latent_bounds']={}
        for name,latent in (('predicted',z),('oracle_gt',h_previous)):
            outside=((latent<model.latent_data_min)|(latent>model.latent_data_max)).float().reshape(16,240,128)
            result['latent_bounds'][name]={group:float(outside[a:b].mean()) for group,(a,b) in
                dict(matched=(0,8),dynamic_cross=(8,12),pose_cross=(12,14),ood=(14,16)).items()}
        (args.out_dir/'probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print('Representation',representation,flush=True)
        return
    # Actual training-target gradient diagnostic, without an optimizer or updates.
    bundle=build_train_val_test_dataloaders(db_path=db,seq_len=2,stride=1,enforce_same_label=True,
        val_ratio=.1,test_ratio=.1,seed=42,batch_size=512,num_workers=0,pin_memory=False,drop_last=True,
        normalize=True,norm_mean=mean,norm_std=std)
    bundle.train_loader.collate_fn=controller_collate
    bank=StyleDonorBank(db,window=16,windows_per_clip=16,device=device)
    pool=TrainStyleRecipePool(db=db,model=model,cache=cache,mean=mean,std=std,
        frames=240,exemplar_choices=[1,3,5],bank_size=64)
    params=list(model.histogram_selector.parameters());gradient=[]
    model.train()
    def vector(gs):return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gs,params)])
    for number,batch in enumerate(bundle.train_loader):
        if number==3:break
        tx=batch['x'].to(device);labels=batch['label'].to(device)
        ids=batch['start'][:,None]+torch.arange(2)
        latent=cache.lookup_latents(ids.flatten(),vqvae=model.vqvae,device=device).reshape(512,2,128)
        labels,targets,recon,changed,entries,_=bank.targets(x_window=tx,labels=labels,latents=latent,
            starts=batch['start'],segment_ids=batch['segment_id'],norm_mean=mean,norm_std=std,
            frame_latent_fn=model.frame_latent,probability=.3)
        source=[segments[i]['source_entry_id'] for i in batch['segment_id'].tolist()]
        histogram,_=pool.sample(labels,probability=1.,per_sample_exclusions=[{a,b} for a,b in zip(source,entries)],require_all=True)
        loss=model.compute_loss(x_curr=tx[:,0],x_next=tx[:,1],style_label=None,prior_labels=labels,
            clip_histograms=histogram,clip_histograms_validated=True,prev_latent=latent[:,0],target_latent=targets[:,0],
            style_hist_weight=.02,style_entropy_weight=.003,history_noise_mode='structured',history_noise_probability=.3,
            prev_latent_noise_std=.3,prev_motion_noise_std=.2,history_root_noise_std=.05)
        gf=vector(torch.autograd.grad(loss['flow_loss'],params,retain_graph=True,allow_unused=True))
        gp=vector(torch.autograd.grad(.02*loss['style_hist_loss'],params,retain_graph=True,allow_unused=True))
        ge=vector(torch.autograd.grad(.003*loss['style_entropy'],params,allow_unused=True))
        gradient.append(dict(flow_norm=float(gf.norm()),weighted_prior_norm=float(gp.norm()),
            weighted_entropy_norm=float(ge.norm()),prior_over_flow=float(gp.norm()/gf.norm().clamp_min(1e-12)),
            flow_prior_cosine=float(torch.dot(gf,gp)/(gf.norm()*gp.norm()).clamp_min(1e-12)),actual_cross=float(changed.float().mean())))
    assert state_fingerprint(model.vqvae.state_dict())==vq_before
    result=dict(checkpoint_sha256=file_sha256(ckpt),device=str(device),fp32=True,seed=42,
        sources=starts,target_names=target_names,oracle_same_action=oracle_same_action,
        representation=representation,history=summaries,selector_gradient=gradient,vq_unchanged=True,
        limitations='16 post-hoc validation cases and 3 training-target gradient batches, not population estimates. Oracle history supplies target GT every frame; masking/raw-recipe ablations are out-of-training-distribution. No rollout or optimizer updates; no suggested ablation constitutes measured retraining improvement.')
    (args.out_dir/'probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('Gradient',gradient,flush=True);print('Representation',representation,flush=True)


if __name__=='__main__':main()
