"""IND sources -> unseen style via pure VQ recipes; defaults to teacher-forced evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from Utils.fid_feature_extractor import build_fid_feature_extractor_from_checkpoint
from Utils.motion_eval import load_motion_model, reconstruct_motion_batch
from Utils.reproducibility import case_noise_plan, file_sha256, state_fingerprint
from Utils.runtime import configure_torch_runtime, load_norm_stats, resolve_device, validate_memmap_db_dir
from Utils.style_code_prior import sparse_code_histogram, recipe_probabilities_from_indices, recipe_slots_from_probs, recipe_conditioning_metadata
from dataset import norm_stats_fingerprint
from evaluate_minor_revision_generalization import (
    _classifier_stats, _make_summary, _resolve_flow_model_name, _resolve_norm_stats_path, _root_ade_cm,
)
from build_zeroshot_style_recipe import _build_recipe_slots

ROOT = Path(__file__).resolve().parent


def read_db(path):
    validate_memmap_db_dir(path)
    names = json.loads((path/'StyleNames.json').read_text())
    segments = json.loads((path/'segments.json').read_text())
    meta = json.loads((path/'meta.json').read_text())
    if len(set(names)) != len(names):
        raise ValueError('DB contains duplicate style names')
    x = np.memmap(path/'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
    for segment in segments:
        if ('source_entry_id' not in segment or segment['frame_start'] < 0
                or segment['num_frames'] < 1
                or segment['frame_start'] + segment['num_frames'] > len(x)
                or not 0 <= segment['label'] < len(names)):
            raise ValueError('Invalid segment boundaries, label, or missing original source identity')
    identity = {f: file_sha256(path/f) for f in ('meta.json','StyleNames.json','segments.json')}
    return names, segments, x, identity


def plan_recipes(names, segments, *, seen_names, classifier_names, frames, exemplars, seed):
    if min(frames, exemplars) < 1 or len(set(classifier_names)) != len(classifier_names):
        raise ValueError('Invalid recipe settings or classifier class names')
    unseen = [(label, name) for label, name in enumerate(names) if name not in seen_names]
    if not unseen:
        raise ValueError('Recipe DB has no styles unseen by the generator')
    plans = []
    for label, name in unseen:
        if name not in classifier_names:
            raise ValueError(f'Classifier does not recognize unseen style {name}')
        independent = {}
        for index, segment in enumerate(segments):
            if segment['label'] == label and segment['num_frames'] >= frames:
                independent.setdefault(segment['source_entry_id'], (index, segment))
        candidates = list(independent.values())
        if len(candidates) <= exemplars:
            raise ValueError(f'{name} needs {exemplars} independent exemplars and at least one separate reference clip')
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        chosen = set(rng.choice(len(candidates), exemplars, replace=False).tolist())
        items = []
        for i, (index, segment) in enumerate(candidates):
            offset = int(rng.integers(segment['num_frames'] - frames + 1))
            items.append(dict(segment_id=index, source_entry_id=segment['source_entry_id'],
                              start=segment['frame_start']+offset, frames=frames,
                              role='exemplar' if i in chosen else 'reference'))
        plans.append(dict(style_name=name, db_label=label,
                          classifier_label=classifier_names.index(name), clips=items))
    return plans


def plan_sources(names, segments, *, frames, stride, count, target_count, recipe_count, seed,
                 source_manifest=None):
    if min(stride, count, target_count) < 1 or target_count > recipe_count:
        raise ValueError('Invalid source/window count or targets-per-source exceeds available unseen styles')
    candidates = []
    for index, segment in enumerate(segments):
        if segment.get('split') == 'test':
            for start in range(segment['frame_start'], segment['frame_start']+segment['num_frames']-frames+1, stride):
                candidates.append(dict(source_start=start, source_entry_id=segment['source_entry_id'],
                    segment_id=index, source_label=segment['label'], source_style=names[segment['label']]))
    if not candidates:
        raise ValueError('No test sources with the requested length; explicit clip-level test split required')
    rng = np.random.default_rng(seed + frames - 1)
    if source_manifest is not None:
        recorded = source_manifest.get('horizons', {}).get(str(frames-1))
        if (not recorded or len(recorded) != count or source_manifest.get('include_gt_context', False)
                or source_manifest.get('protocol_version',3) != 3 or source_manifest.get('seed',seed) != seed):
            raise ValueError('Source manifest must contain the requested generated-only horizon and source count')
        lookup = {row['source_start']: row for row in candidates}
        chosen = []
        for row in recorded:
            current = lookup.get(row['source_start'])
            if current is None or any(current[k] != row[k] for k in ('source_entry_id','source_label')):
                raise ValueError('Source manifest identity differs or includes non-test sources')
            chosen.append(current)
    else:
        chosen = [candidates[i] for i in rng.choice(len(candidates), min(count,len(candidates)), replace=False)]
    for row in chosen:
        row['recipe_indices'] = rng.choice(recipe_count, target_count, replace=False).tolist()
        row['noise_seed'] = seed
    return chosen


@torch.no_grad()
def encode_recipe(model, source, clips, *, mean, std, chunk_rows):
    raw = np.concatenate([source[c['start']:c['start']+c['frames']] for c in clips if c['role']=='exemplar'])
    device = next(model.parameters()).device
    parts = []
    for begin in range(0, len(raw), chunk_rows):
        x = torch.from_numpy((raw[begin:begin+chunk_rows]-mean)/std).to(device)
        _, indices, _ = model.vqvae.encode(x)
        parts.append(indices.cpu().numpy())
    indices = np.concatenate(parts)
    probs = recipe_probabilities_from_indices(indices, model,
        clip_lengths=[c['frames'] for c in clips if c['role']=='exemplar'])
    slots = recipe_slots_from_probs(probs, model.selected_style_slots)
    return torch.from_numpy(probs).to(device), slots


@torch.no_grad()
def generate_pure_recipe(model, x, probs, noise, *, mode, model_name, steps, microbatch):
    if mode == 'rollout':
        return reconstruct_motion_batch(model_name=model_name, model=model, x=x,
            style_label=None, manual_style_probs=probs, manual_style_strength=1.,
            flow_eval_mode='rollout', flow_ode_steps=steps, noise_plan=noise)
    # Independent TF time points share a cached style feature; do not expand [B,M,K] over time.
    feature = model.encode_conditions(x_curr=x[:,0], style_label=None,
        manual_style_probs=probs, manual_style_strength=1.)[2]
    batch, frames, _ = x.shape
    current = x[:,:-1].reshape(-1,375)
    sources = noise.reshape(-1,model.latent_dim)
    features = feature[:,None].expand(-1,frames-1,-1).reshape(batch*(frames-1),-1)
    motions = []
    for begin in range(0,len(current),microbatch):
        clip = current[begin:begin+microbatch]
        previous = model.frame_latent(clip)
        latent = model.sample_latent(x_curr=clip, style_label=None, prev_latent=previous,
            style_feature=features[begin:begin+microbatch], z0=sources[begin:begin+microbatch],
            num_steps=steps, noise_std=model.eval_noise_std)
        motion, _ = model.feedback_state(latent)
        motions.append(motion)
    out = x.clone()
    out[:,1:,84:] = torch.cat(motions).reshape(batch,frames-1,291)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=ROOT/'database_100sty_375_memmap_nomirror')
    parser.add_argument('--recipe-db',type=Path,default=ROOT/'database_100sty_375_memmap_zeroshot')
    parser.add_argument('--flow-ckpt',type=Path,default=ROOT/'checkpoints/clip_recipe_v2/controller/best.pt')
    parser.add_argument('--classifier-ckpt',type=Path,default=ROOT/'checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'eval_results/unseen_rollout240_staged_noise')
    parser.add_argument('--source-manifest',type=Path,default=None)
    parser.add_argument('--flow-eval-mode',choices=['reconstruct','rollout'],default='rollout')
    parser.add_argument('--num-sources',type=int,default=1000)
    parser.add_argument('--targets-per-source',type=int,default=5)
    parser.add_argument('--exemplars-per-style',type=int,default=5)
    parser.add_argument('--stride',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--microbatch-size',type=int,default=512)
    parser.add_argument('--chunk-rows',type=int,default=512)
    parser.add_argument('--ode-steps',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--prepare-only',action='store_true',help='Save and validate fixed cases/exemplar split without encoding or generating motion')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if min(args.batch_size,args.microbatch_size,args.chunk_rows,args.ode_steps) < 1:
        raise ValueError('Batch/chunk sizes and ODE steps must be positive')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if (args.out_dir/'summary.json').exists():
        raise ValueError('Output already has results; choose a new out-dir')
    seen, segments, source, source_identity = read_db(args.db)
    unseen_names, unseen_segments, reference, recipe_identity = read_db(args.recipe_db)
    norm_path = _resolve_norm_stats_path(args.flow_ckpt,None)
    if norm_path is None: raise ValueError('Saved controller/VQ normalization is required')
    mean, std = load_norm_stats(norm_path)
    model_name = _resolve_flow_model_name(args.flow_ckpt,'auto')
    model = load_motion_model(model_name=model_name,ckpt_path=args.flow_ckpt,num_styles=len(seen)).eval()
    if model.num_styles != len(seen) or not model.use_style_code_selector:
        raise ValueError('Generator DB/style count must match a selector-enabled controller')
    if norm_stats_fingerprint(mean,std) != model.eval_fingerprints.get('norm_fingerprint'):
        raise ValueError('Normalization differs from the frozen controller/VQ manifold')
    classifier, metadata = build_fid_feature_extractor_from_checkpoint(args.classifier_ckpt,map_location='cpu')
    names = metadata.get('class_names',[])
    if any(name not in names for name in seen): raise ValueError('Classifier lacks generator style names')
    frames = int(metadata.get('window_size',0))
    fps = float(metadata.get('fps',0))
    if frames != 240 or not np.isclose(fps,60): raise ValueError('This protocol requires the 60fps/240-frame classifier')
    plans = plan_recipes(unseen_names,unseen_segments,seen_names=seen,classifier_names=names,
        frames=frames,exemplars=args.exemplars_per_style,seed=args.seed)
    inherited = json.loads(args.source_manifest.read_text()) if args.source_manifest else None
    if inherited is not None and inherited.get('split_sha256') != source_identity['segments.json']:
        raise ValueError('Seen-source manifest split fingerprint differs from this generator DB')
    sources = plan_sources(seen,segments,frames=frames+1,stride=args.stride,count=args.num_sources,
        target_count=args.targets_per_source,recipe_count=len(plans),seed=args.seed,source_manifest=inherited)
    # Hash the exact used windows, including X content. Unused rows do not influence this protocol.
    digest = hashlib.sha256()
    for row in sources: digest.update(np.ascontiguousarray(source[row['source_start']:row['source_start']+frames+1]).tobytes())
    for plan in plans:
        for c in plan['clips']: digest.update(np.ascontiguousarray(reference[c['start']:c['start']+c['frames']]).tobytes())
    manifest = dict(protocol_version=1,fps=fps,generated_frames=frames,include_gt_context=False,seed=args.seed,
        exemplars_per_style=args.exemplars_per_style,
        source_identity=source_identity,recipe_identity=recipe_identity,used_window_sha256=digest.hexdigest(),
        vq_state_sha256=state_fingerprint(model.vqvae.state_dict()),norm_fingerprint=norm_stats_fingerprint(mean,std),
        selected_slots=list(model.selected_style_slots),topk=model.style_selector.topk,
        classifier_sha256=file_sha256(args.classifier_ckpt),sources=sources,recipes=plans)
    path = args.out_dir/'case_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest: raise ValueError('Saved manifest differs in protocol, cases, exemplar split or data content')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'[Info] {len(sources)} test sources, {len(plans)} unseen styles; exemplar/reference source IDs are disjoint')
    if args.prepare_only:
        print('[Done] Manifest prepared; no recipe encoding, flow generation or rollout executed.'); return
    device = resolve_device(args.device)
    configure_torch_runtime(device=device,tf32=False,cudnn_benchmark=False,matmul_precision='highest')
    model = model.to(device); classifier = classifier.to(device).eval()
    recipe_dir = args.out_dir/'recipes'; recipe_dir.mkdir(exist_ok=True)
    probabilities = []
    rows = []
    def classify(raw,targets):
        return _classifier_stats(classifier,raw,np.asarray(targets),device=device,classifier_window=frames,
            classifier_stride=frames,classifier_batch_size=args.batch_size,joint_set=metadata['joint_set'])
    for i,plan in enumerate(tqdm(plans,desc='recipes and real/quantized-AE references')):
        probs, slots = encode_recipe(model,reference,plan['clips'],mean=mean,std=std,chunk_rows=args.chunk_rows)
        probabilities.append(probs)
        payload = dict(strength=1.,slots=slots,metadata={**manifest, **recipe_conditioning_metadata(model), 'sources':None,'recipes':None,
            'style_name':plan['style_name'],'exemplars':[c for c in plan['clips'] if c['role']=='exemplar']})
        (recipe_dir/f'style_{i:02d}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
        for c in plan['clips']:
            if c['role'] != 'reference': continue
            raw = np.array(reference[c['start']:c['start']+frames],copy=True)
            normalized = torch.from_numpy((raw-mean)/std).to(device)
            decoded = []
            for begin in range(0,frames,args.chunk_rows):
                decoded.append(model.vqvae.decode(model.frame_latent(normalized[begin:begin+args.chunk_rows])))
            ae = raw.copy(); ae[:,84:] = torch.cat(decoded).cpu().numpy()*std[84:]+mean[84:]
            for setting,clip in [('real_unseen_reference',raw),('quantized_ae_unseen_reference',ae)]:
                pred,confidence,maximum = classify(clip[None],[plan['classifier_label']])
                rows.append(dict(setting=setting,source_entry_id=c['source_entry_id'],source_start=c['start'],
                    target_style=plan['classifier_label'],target_name=plan['style_name'],pred_style=int(pred[0]),
                    target_conf=float(confidence[0]),pred_conf=float(maximum[0]),root_ade_cm=float('nan'),uses_recipe=False,recipe_strength=0.))
    probabilities = torch.stack(probabilities)
    cases = [(row,index) for row in sources for index in row['recipe_indices']]
    setting = f'unseen_style_{"teacher_forced" if args.flow_eval_mode=="reconstruct" else "rollout"}_h{frames}'
    for begin in tqdm(range(0,len(cases),args.batch_size),desc=setting):
        batch = cases[begin:begin+args.batch_size]
        raw = np.stack([source[row['source_start']:row['source_start']+frames+1] for row,_ in batch])
        x = torch.from_numpy((raw-mean)/std).to(device)
        recipe_ids = [index for _,index in batch]
        targets = [plans[index]['classifier_label'] for index in recipe_ids]
        starts = [row['source_start'] for row,_ in batch]
        noise = case_noise_plan(model,starts,targets,frames+1,seed=args.seed,noise_std=model.eval_noise_std)
        generated = generate_pure_recipe(model,x,probabilities[recipe_ids],noise,mode=args.flow_eval_mode,
            model_name=model_name,steps=args.ode_steps,microbatch=args.microbatch_size)
        generated = generated.cpu().numpy()*std+mean
        pred,confidence,maximum = classify(generated[:,1:],targets)
        for j,(row,index) in enumerate(batch):
            rows.append(dict(setting=setting,source_entry_id=row['source_entry_id'],source_start=row['source_start'],
                source_style=row['source_label'],source_classifier_label=names.index(row['source_style']),
                target_style=targets[j],target_name=plans[index]['style_name'],recipe_index=index,noise_seed=args.seed,
                pred_style=int(pred[j]),target_conf=float(confidence[j]),pred_conf=float(maximum[j]),
                root_ade_cm=_root_ade_cm(raw[j],generated[j],dt=1/fps,context_frames=1),uses_recipe=True,recipe_strength=1.))
    summaries = [_make_summary(name,[r for r in rows if r['setting']==name]) for name in dict.fromkeys(r['setting'] for r in rows)]
    generated_rows = [r for r in rows if r['setting']==setting]
    per_style = [_make_summary(p['style_name'],[r for r in generated_rows if r['target_style']==p['classifier_label']])
                 for p in plans if any(r['target_style']==p['classifier_label'] for r in generated_rows)]
    result = dict(protocol='unseen_pure_recipe_60fps_240f',flow_eval_mode=args.flow_eval_mode,
        generator_conditioning='style_label=None, strength=1, all slots, TopK from controller',
        flow_ckpt=str(args.flow_ckpt),flow_ckpt_sha256=file_sha256(args.flow_ckpt),
        classifier_ckpt=str(args.classifier_ckpt),classifier_acc_split=metadata.get('acc_split'),
        classifier_split_protocol=metadata.get('split_protocol'),ode_steps=args.ode_steps,
        requantize=model.inference_requantize,latent_clip=model.inference_latent_clip,
        num_sources=len(sources),num_cases=len(cases),seed=args.seed,summaries=summaries,per_target_style=per_style,
        macro_style_acc_pct=float(np.mean([s['target_style_acc_pct'] for s in per_style])),
        source_retention_pct=100*np.mean([r['pred_style']==r['source_classifier_label'] for r in generated_rows]))
    (args.out_dir/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (args.out_dir/'per_sample.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
    print(f'[Done] Results: {args.out_dir}')


if __name__=='__main__': main()
