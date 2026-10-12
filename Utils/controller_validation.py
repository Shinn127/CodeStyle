"""Fixed validation rollouts and joint checkpoint selection (never reads the test split)."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint, motion375_clip_to_classifier_input, infer_joint_set_from_in_feat
from Utils.motion_eval import reconstruct_motion_batch, compute_sequence_metric_sums, MotionEvalConfig
from Utils.motion375 import reconstruct_root_track_from_375
from Utils.reproducibility import case_noise_plan, isolated_rng, file_sha256, state_fingerprint
from Utils.vqvae_eval import select_eval_windows


def classifier_label_mapping(style_names, metadata, *, device='cpu'):
    names = metadata.get('class_names', [])
    if not names or len(set(names)) != len(names):
        raise ValueError('Classifier requires unique class_names for label mapping')
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in style_names if name not in lookup]
    if missing:
        raise ValueError(f'Classifier lacks generator styles: {missing}')
    return torch.tensor([lookup[name] for name in style_names], dtype=torch.long, device=device)


def selection_key(metrics):
    passed = metrics['matched_root_ade_cm'] <= 4.12 and metrics['cross_root_ade_cm'] <= 4.63
    score = (metrics['mpjpe_cm'] / 6.016 + metrics['jitter_cm'] / .348
             + metrics['foot_sliding_local_cm'] / .600
             + metrics['matched_root_ade_cm'] / 2.747 + metrics['cross_root_ade_cm'] / 3.088
             + 2 * (1 - metrics['matched_accuracy']) + 2 * (1 - metrics['cross_accuracy']))
    if not np.isfinite(score):
        return (1, float('inf'))
    return (0 if passed else 1, float(score))


def teacher_forced_selection_key(metrics, reference=None, root_tolerance=1.15,
        *, min_matched_accuracy=.85, max_matched_ade=5., max_cross_ade=10.):
    fields = ('matched_accuracy', 'cross_accuracy', 'matched_root_ade_cm', 'cross_root_ade_cm')
    if not all(np.isfinite(metrics[k]) for k in fields):
        raise ValueError('Non-finite teacher-forced validation metrics')
    if reference is None:
        limits = (max_matched_ade, max_cross_ade)
        floor = min_matched_accuracy
        if not 0 <= floor <= 1 or not all(np.isfinite(x) and x > 0 for x in limits):
            raise ValueError('Invalid absolute teacher-forced selection limits')
    else:
        if not all(np.isfinite(reference[k]) for k in fields):
            raise ValueError('Non-finite teacher-forced validation reference')
        floor = max(min_matched_accuracy, reference['matched_accuracy'] - .05)
        limits = tuple(root_tolerance * max(reference[k], 1e-6) for k in fields[2:])
    passed = metrics['matched_accuracy'] >= floor and all(metrics[k] <= limit for k, limit in zip(fields[2:], limits))
    ade = sum(metrics[k] / limit for k, limit in zip(fields[2:], limits))
    return (0 if passed else 1, -metrics['cross_accuracy'], -metrics['matched_accuracy'], float(ade))


def motion_quality_selection_key(metrics, *, min_matched=.85, min_cross=.80,
        max_matched_ade=5., max_cross_ade=10., max_jitter=.5, max_local_fs=.8, max_world_fs=1.):
    limits = (max_matched_ade,max_cross_ade,max_jitter,max_local_fs,max_world_fs,max_jitter)
    if not all(np.isfinite(v) and v>0 for v in limits): raise ValueError('Invalid quality limits')
    names = ('matched_root_ade_cm','cross_root_ade_cm','jitter_cm','foot_sliding_local_cm','foot_sliding_world_cm','cross_jitter_cm')
    values = [metrics.get(k) for k in names]
    usable = all(v is not None and np.isfinite(v) for v in values) and metrics.get('world_contact_count',0)>0
    if not usable: return (1,float('inf'))
    passed = (metrics['matched_accuracy']>=min_matched and metrics['cross_accuracy']>=min_cross
              and all(v<=limit for v,limit in zip(values,limits)))
    score = sum(v/limit for v,limit in zip(values,limits))+2*(1-metrics['matched_accuracy'])+2*(1-metrics['cross_accuracy'])
    return (0 if passed else 1,float(score))


class TeacherForcedStyleValidation:
    """Fixed style cases; historical class name retained, rollout is the default."""

    @torch.no_grad()
    def __init__(self, model, *, db, mean, std, classifier_path, seed=42, clips=90,
                 windows=3, out_dir=None, exemplars=5, eval_mode='rollout', batch_size=64, quality_metrics=False):
        if eval_mode not in ('rollout', 'reconstruct'):
            raise ValueError('Style validation mode must be rollout or reconstruct')
        self.eval_mode = eval_mode
        if batch_size < 1:
            raise ValueError('Style validation batch size must be positive')
        self.batch_size = batch_size
        self.quality_metrics = bool(quality_metrics)
        self.windows_per_clip = windows
        from Utils.style_code_prior import sparse_code_histogram
        self.device = next(model.field.parameters()).device
        self.mean, self.std, self.seed = np.asarray(mean, np.float32), np.asarray(std, np.float32), seed
        self.ode_steps = int(getattr(model, 'eval_solver_steps', 8))
        db = Path(db)
        meta = json.loads((db / 'meta.json').read_text())
        self.names = json.loads((db / 'StyleNames.json').read_text())
        self.source = np.memmap(db / 'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
        segments = json.loads((db / 'segments.json').read_text())
        self.classifier, self.classifier_meta = build_fid_feature_extractor_from_checkpoint(Path(classifier_path), map_location='cpu')
        if int(self.classifier_meta.get('window_size', 0)) != 240 or float(self.classifier_meta.get('fps', 0)) != 60:
            raise ValueError('TF style validation requires the 60fps / 240-frame classifier')
        self.classifier.to(self.device).eval().requires_grad_(False)
        self.mapping = classifier_label_mapping(self.names, self.classifier_meta, device=self.device)
        spans = select_eval_windows(db, split='val', clips=clips, window_frames=241,
            windows_per_clip=windows, min_frames=241, seed=seed)
        if len(spans) != clips * windows:
            raise ValueError('Not enough independent validation clips for the fixed TF protocol')
        self.starts = np.asarray([s for s, _ in spans], dtype=np.int64)
        if any(not any(s['split'] == 'val' and s['frame_start'] <= start
                and start + 241 <= s['frame_start'] + s['num_frames'] for s in segments) for start in self.starts):
            raise ValueError('TF validation sources must be validation clips')
        labels = np.memmap(db / 'Labels.i32', mode='r', dtype=np.int32, shape=(meta['num_frames'],))
        self.labels = np.array(labels[self.starts], dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.targets = np.stack([rng.choice(np.flatnonzero(np.arange(len(self.names)) != label), 2, replace=False) for label in self.labels])
        profiles, recipe_manifest, reference_indices = [], [], []
        model.eval()
        for label in range(len(self.names)):
            independent = {s['source_entry_id']: s for s in segments if s['split'] == 'train'
                and s['label'] == label and s['num_frames'] >= 240}
            choices = list(independent.values())
            if len(choices) < exemplars:
                raise ValueError('TF validation needs five independent train exemplars for every style')
            selected = rng.choice(len(choices), exemplars, replace=False)
            indices, records = [], []
            for i in selected:
                s = choices[int(i)]
                start = s['frame_start'] + int(rng.integers(s['num_frames'] - 240 + 1))
                x = torch.from_numpy((np.array(self.source[start:start + 240]) - self.mean) / self.std).to(self.device)
                _, code, _ = model.vqvae.encode(x)
                indices.append(code.cpu().numpy())
                records.append(dict(source_entry_id=s['source_entry_id'], start=start, frames=240))
                if getattr(model,'conditioning_version',0)==2:records[-1]['suffix']=s.get('suffix','')
            profiles.append(sparse_code_histogram(np.concatenate(indices), selected_slots=model.selected_style_slots,
                num_embeddings=model.vqvae.quantizer.num_embeddings, topk=model.vqvae.quantizer.num_embeddings))
            reference_indices.append(np.stack(indices))
            recipe_manifest.append(dict(target_label=label, clips=records))
        self.histograms = torch.from_numpy(np.stack(profiles)).to(self.device)
        self.reference_indices = np.stack(reference_indices) if getattr(model,'conditioning_version',0)==2 else None
        digest = hashlib.sha256()
        for start in self.starts:
            digest.update(np.asarray(self.source[start:start + 241]).tobytes())
        from dataset import norm_stats_fingerprint
        self.manifest = dict(protocol='rollout_60fps_240f' if eval_mode == 'rollout' else 'teacher_forced_60fps_240f', split='val', seed=seed,
            starts=self.starts.tolist(), source_labels=self.labels.tolist(), cross_targets=self.targets.tolist(),
            recipes=recipe_manifest, classifier_sha256=file_sha256(classifier_path), ode_steps=self.ode_steps,
            requantize=True, latent_clip=True, source_windows_sha256=digest.hexdigest(),
            histogram_sha256=hashlib.sha256(np.stack(profiles).tobytes()).hexdigest(),
            vq_state_sha256=state_fingerprint(model.vqvae.state_dict()), norm_fingerprint=norm_stats_fingerprint(self.mean, self.std))
        if self.reference_indices is not None:
            self.manifest.update(conditioning_version=2,reference_indices_sha256=hashlib.sha256(self.reference_indices.tobytes()).hexdigest(),
                                 style_guidance_scale=model.style_guidance_scale)
        if out_dir is not None:
            path = Path(out_dir) / ('rollout_style_validation_cases.json' if eval_mode == 'rollout' else 'tf_validation_cases.json')
            if path.exists() and json.loads(path.read_text()) != self.manifest:
                raise ValueError('Fixed teacher-forced validation manifest changed')
            path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2))

    @torch.no_grad()
    def evaluate(self, model):
        from evaluation.evaluate_minor_revision_generalization import _classifier_stats, _root_ade_cm
        model.eval()
        recipe_conditioning = model.conditioning_mode in ('recipe', 'clip_recipe')
        if getattr(model,'conditioning_version',0)==2:
            positions=np.linspace(0,239,model.reference_tokens).astype(np.int64)
            parts=[]
            for begin in range(0,len(self.reference_indices),8):
                codes=torch.from_numpy(self.reference_indices[begin:begin+8][:,:,positions].astype(np.int64)).to(self.device)
                latents=model.vqvae.quantizer.lookup(codes.reshape(-1,codes.shape[-1])).reshape(*codes.shape[:-1],model.latent_dim)
                times=torch.from_numpy(positions.astype(np.float32)/60.).to(self.device).expand(codes.shape[:-1])
                parts.append(model.probabilities_from_references(latents,times,torch.ones_like(times,dtype=torch.bool))[0])
            probs=torch.cat(parts)
        elif model.conditioning_mode == 'clip_recipe':
            probs = model.recipe_from_histogram(self.histograms)
        else:
            indices = torch.argsort(self.histograms, descending=True, stable=True)[..., :model.style_selector.topk]
            probs = torch.zeros_like(self.histograms).scatter(-1, indices, self.histograms.gather(-1, indices))
            probs /= probs.sum(-1, keepdim=True)
        results = {}
        quality = dict(jitter_sum=0.,jitter_count=0,local_sum=0.,local_count=0,world_sum=0.,world_count=0,possible=0,
                       cross_jitter_sum=0.,cross_jitter_count=0)
        cross_steps=[];matched_steps=[]
        for name in ('matched', 'cross'):
            starts = self.starts if name == 'matched' else np.repeat(self.starts, 2)
            labels = self.labels if name == 'matched' else self.targets.reshape(-1)
            correct, ades = [], []
            for begin in range(0, len(starts), self.batch_size):
                ids, targets = starts[begin:begin + self.batch_size], labels[begin:begin + self.batch_size]
                raw = np.stack([np.array(self.source[s:s + 241]) for s in ids])
                x = torch.from_numpy((raw - self.mean) / self.std).to(self.device)
                p = probs[torch.as_tensor(targets, device=self.device)]
                noise = case_noise_plan(model, ids, targets, 241, seed=self.seed, noise_std=1.)
                pred = reconstruct_motion_batch(model_name=f'flowmatch_{model.prediction_mode}', model=model,
                    x=x, style_label=None if recipe_conditioning else torch.as_tensor(targets, device=self.device),
                    manual_style_probs=p if recipe_conditioning else None, flow_eval_mode=self.eval_mode, noise_plan=noise,
                    flow_ode_steps=self.ode_steps, flow_noise_std=1., flow_requantize_z=True, flow_latent_clip=True)
                generated = pred.cpu().numpy() * self.std + self.mean
                target_classes = self.mapping[torch.as_tensor(targets, device=self.device)].cpu().numpy()
                prediction, _, _ = _classifier_stats(self.classifier, generated[:, 1:], target_classes,
                    device=self.device, classifier_window=240, classifier_stride=240, classifier_batch_size=self.batch_size,
                    joint_set=self.classifier_meta['joint_set'])
                correct.extend((prediction == target_classes).tolist())
                ades.extend(_root_ade_cm(ref, out, dt=1/60, context_frames=1) for ref, out in zip(raw, generated))
                if self.quality_metrics:
                    for index,(ref,out) in enumerate(zip(raw,generated)):
                        source_index = begin+index if name=='matched' else (begin+index)//2
                        if source_index % self.windows_per_clip: continue
                        results[name+'_quality_cases'] = results.get(name+'_quality_cases',0)+1
                        pos=out[1:,87:159].reshape(-1,24,3)*100
                        if name=='cross':
                            # A cross-style gait can have different contact timing;
                            # source GT contacts are not its foot-sliding reference.
                            jitter=np.linalg.norm(np.diff(pos,n=2,axis=0),axis=-1)
                            quality['cross_jitter_sum']+=float(jitter.sum());quality['cross_jitter_count']+=jitter.size
                            cross_steps.extend(np.linalg.norm(np.diff(pos,axis=0),axis=-1).mean(-1).tolist())
                            continue
                        sums = compute_sequence_metric_sums(reference_375=ref,prediction_375=out,cfg=MotionEvalConfig(metric_start_frame=1))
                        matched_steps.extend(np.linalg.norm(np.diff(pos,axis=0),axis=-1).mean(-1).tolist())
                        quality['jitter_sum'] += sums.jitter_sum_cm; quality['jitter_count'] += sums.jitter_count
                        quality['local_sum'] += sums.foot_sliding_sum_cm; quality['local_count'] += sums.foot_sliding_count
                        quality['world_sum'] += sums.foot_sliding_world_sum_cm; quality['world_count'] += sums.foot_sliding_world_count
                        quality['possible'] += sums.foot_contact_possible_count
            results[name + '_accuracy'] = float(np.mean(correct))
            results[name + '_root_ade_cm'] = float(np.mean(ades))
            results[name + '_cases'] = len(correct)
        results['evaluation_batch_size'] = self.batch_size
        if self.quality_metrics:
            results.update(jitter_cm=quality['jitter_sum']/max(1,quality['jitter_count']),
                foot_sliding_local_cm=quality['local_sum']/quality['local_count'] if quality['local_count'] else None,
                foot_sliding_world_cm=quality['world_sum']/quality['world_count'] if quality['world_count'] else None,
                world_contact_count=quality['world_count'],world_contact_coverage=quality['world_count']/max(1,quality['possible']),
                cross_jitter_cm=quality['cross_jitter_sum']/max(1,quality['cross_jitter_count']),
                cross_pose_step_p99_cm=float(np.percentile(cross_steps,99)) if cross_steps else None,
                cross_pose_step_max_cm=float(np.max(cross_steps)) if cross_steps else None,
                matched_pose_step_p99_cm=float(np.percentile(matched_steps,99)) if matched_steps else None,
                matched_pose_step_max_cm=float(np.max(matched_steps)) if matched_steps else None,
                quality_protocol='first_fixed_window_per_clip_matched_reference_contacts_cross_raw_jitter_240f')
        if getattr(model,'conditioning_version',0)==2:results['style_guidance_scale']=model.style_guidance_scale
        return results

    @torch.no_grad()
    def condition_diagnostics(self,model,*,cases=4):
        """Small fixed validation rollouts; never reads OOD or changes sampling inputs."""
        if getattr(model,'conditioning_version',0)!=2:raise ValueError('Condition diagnostics require V2')
        from evaluation.evaluate_minor_revision_generalization import _classifier_stats,_root_ade_cm
        model.eval();n=min(cases,len(self.starts));starts=self.starts[:n];targets=self.targets[:n,0]
        positions=np.linspace(0,239,model.reference_tokens).astype(np.int64)
        codes=torch.from_numpy(self.reference_indices[:,:,positions].astype(np.int64)).to(self.device)
        z=model.vqvae.quantizer.lookup(codes.reshape(-1,codes.shape[-1])).reshape(*codes.shape[:-1],model.latent_dim)
        stamp=torch.from_numpy(positions.astype(np.float32)/60.).to(self.device).expand(codes.shape[:-1])
        # Process one style at a time so the diagnostic export and clip paths share a batch shape.
        probabilities=torch.cat([model.probabilities_from_references(z[i:i+1],stamp[i:i+1],
            torch.ones_like(stamp[i:i+1],dtype=torch.bool))[0] for i in range(len(z))])
        wrong=[]
        for target in targets:
            actions={c['suffix'] for c in self.manifest['recipes'][int(target)]['clips']}
            candidates=[i for i,r in enumerate(self.manifest['recipes']) if i!=target
                and actions.intersection(c['suffix'] for c in r['clips'])]
            if not candidates:candidates=[i for i in range(len(probabilities)) if i!=target]
            wrong.append(candidates[0])
        raw=np.stack([np.array(self.source[s:s+241]) for s in starts])
        x=torch.from_numpy((raw-self.mean)/self.std).to(self.device)
        target_t=torch.as_tensor(targets,device=self.device);wrong_t=torch.as_tensor(wrong,device=self.device)
        correct_p=probabilities[target_t];wrong_p=probabilities[wrong_t]
        correct=model.condition_from_probabilities(correct_p)
        wrong_condition=model.condition_from_probabilities(wrong_p)
        exported=model.encode_conditions(x_curr=x[:,0],style_label=None,manual_style_probs=correct_p)[2]
        torch.testing.assert_close(correct,exported,atol=0,rtol=0)
        noise=case_noise_plan(model,starts,targets,241,seed=self.seed,noise_std=1.)
        results={};code_paths={};initial_latents={}
        for name,style in (('correct_reference',correct),('wrong_reference',wrong_condition),
                           ('null_style',model.null_style.detach()[None].expand(n,-1))):
            current=x[:,0];previous=model.frame_latent(current);motions=[];path=[]
            for step in range(240):
                latent=model.sample_latent(x_curr=current,style_label=None,style_feature=style,
                    prev_latent=previous,z0=noise[:,step],num_steps=8)
                if step==0:initial_latents[name]=latent
                motion,previous=model.feedback_state(latent);motions.append(motion);path.append(previous)
                current=torch.cat((x[:,step+1,:84],motion),-1)
            output=np.array(raw,copy=True)
            output[:,1:,84:]=torch.stack(motions,1).cpu().numpy()*self.std[84:]+self.mean[84:]
            target_classes=self.mapping[target_t].cpu().numpy()
            predictions,_,_=_classifier_stats(self.classifier,output[:,1:],target_classes,device=self.device,
                classifier_window=240,classifier_stride=240,classifier_batch_size=self.batch_size,
                joint_set=self.classifier_meta['joint_set'])
            results[name]=dict(original_target_accuracy=float(np.mean(predictions==target_classes)),
                requested_wrong_accuracy=float(np.mean(predictions==self.mapping[wrong_t].cpu().numpy())) if name=='wrong_reference' else None,
                root_ade_cm=float(np.mean([_root_ade_cm(a,b,dt=1/60,context_frames=1) for a,b in zip(raw,output)])))
            code_paths[name]=torch.stack(path,1).reshape(n,240,model.vqvae.num_latents,model.vqvae.code_dim)
        for name in ('wrong_reference','null_style'):
            results[name]['rollout_code_change_fraction']=float((code_paths[name]!=code_paths['correct_reference']).any(-1).float().mean())
            results[name]['first_continuous_latent_distance']=float((initial_latents[name]-initial_latents['correct_reference']).norm(dim=-1).mean())
            results[name]['first_quantized_code_change_fraction']=float((code_paths[name][:,0]!=code_paths['correct_reference'][:,0]).any(-1).float().mean())
        return dict(protocol='fixed_validation_condition_rollout_240f',cases=n,starts=starts.tolist(),targets=targets.tolist(),
            wrong_targets=wrong,noise_seed=self.seed,style_guidance_scale=model.style_guidance_scale,
            recipe_condition_identical=True,condition_distance=float((correct-wrong_condition).norm(dim=-1).mean()),
            variants=results,limitations='Small validation intervention, not an aggregate accuracy estimate or OOD selection.')


@torch.no_grad()
def evaluate_validation(model, *, db, mean, std, classifier_path, seed=42, batch_size=64):
    model.eval()
    device = next(model.field.parameters()).device
    db = Path(db)
    meta = json.loads((db / 'meta.json').read_text())
    names = json.loads((db / 'StyleNames.json').read_text())
    spans = select_eval_windows(db, split='val', clips=90, window_frames=60,
        windows_per_clip=3, min_frames=60, seed=seed)
    if len(spans) != 90 * 3:
        raise ValueError('Fixed validation protocol requires 90 clips with 3 windows each')
    starts = np.asarray([start for start, _ in spans], dtype=np.int64)
    source = np.memmap(db / 'X.f32', dtype='float32', mode='r', shape=(meta['num_frames'], 375))
    labels_source = np.memmap(db / 'Labels.i32', dtype='int32', mode='r', shape=(meta['num_frames'],))
    labels = np.asarray(labels_source[starts], dtype=np.int64)
    rng = np.random.default_rng(seed)
    targets = np.stack([rng.choice(np.flatnonzero(np.arange(len(names)) != label), 2, replace=False) for label in labels])
    with isolated_rng(seed + 501):
        classifier, classifier_meta = build_fid_feature_extractor_from_checkpoint(Path(classifier_path))
        classifier = classifier.to(device).eval()
        mapping = classifier_label_mapping(names, classifier_meta, device=device)
        joint_set = infer_joint_set_from_in_feat(int(classifier_meta['in_feat']))
        matched = []
        cross_ade = []
        matched_correct = []
        cross_correct = []
        for begin in range(0, len(starts), batch_size):
            ids = starts[begin:begin + batch_size]
            raw = np.stack([np.asarray(source[s:s + 60]) for s in ids])
            x = torch.from_numpy((raw - mean) / std).to(device)
            label = torch.as_tensor(labels[begin:begin + batch_size], device=device)
            noise = case_noise_plan(model, ids, label.cpu().tolist(), 60, seed=seed)
            pred = reconstruct_motion_batch(model_name=f'flowmatch_{model.prediction_mode}',
                model=model, x=x, style_label=label, noise_plan=noise)
            pred_raw = pred.cpu().numpy() * std + mean
            _, logits = classifier(motion375_clip_to_classifier_input(pred_raw, joint_set=joint_set).to(device))
            matched_correct.extend((logits.argmax(1) == mapping[label]).cpu().tolist())
            for ref, gen in zip(raw, pred_raw):
                matched.append(compute_sequence_metric_sums(reference_375=ref, prediction_375=gen, cfg=MotionEvalConfig()))
            requested = targets[begin:begin + batch_size].reshape(-1)
            cross_x = x.repeat_interleave(2, dim=0)
            target = torch.as_tensor(requested, device=device)
            cross_noise = case_noise_plan(model, np.repeat(ids, 2), requested, 60, seed=seed)
            generated = reconstruct_motion_batch(model_name=f'flowmatch_{model.prediction_mode}',
                model=model, x=cross_x, style_label=target, noise_plan=cross_noise)
            generated_raw = generated.cpu().numpy() * std + mean
            _, logits = classifier(motion375_clip_to_classifier_input(generated_raw, joint_set=joint_set).to(device))
            cross_correct.extend((logits.argmax(1) == mapping[target]).cpu().tolist())
            for ref, gen in zip(raw.repeat(2, axis=0), generated_raw):
                ref_pos, _ = reconstruct_root_track_from_375(ref)
                gen_pos, _ = reconstruct_root_track_from_375(gen)
                cross_ade.append(float(np.linalg.norm(ref_pos[:, [0, 2]] - gen_pos[:, [0, 2]], axis=-1).mean() * 100))
    def ratio(value, count):
        denominator = sum(getattr(row, count) for row in matched)
        return sum(getattr(row, value) for row in matched) / denominator if denominator else float('nan')
    result = {'mpjpe_cm': ratio('mpjpe_sum_cm', 'mpjpe_count'),
              'jitter_cm': ratio('jitter_sum_cm', 'jitter_count'),
              'foot_sliding_local_cm': ratio('foot_sliding_sum_cm', 'foot_sliding_count'),
              'matched_root_ade_cm': ratio('root_ade_sum_cm', 'root_ade_count'),
              'cross_root_ade_cm': float(np.mean(cross_ade)),
              'matched_accuracy': float(np.mean(matched_correct)),
              'cross_accuracy': float(np.mean(cross_correct)),
              'starts': starts.tolist(), 'cross_targets': targets.tolist(),
              'seed': seed, 'split': 'val', 'classifier_sha256': file_sha256(classifier_path)}
    key = selection_key(result)
    result.update(constraints_passed=key[0] == 0, joint_score=key[1])
    return result
