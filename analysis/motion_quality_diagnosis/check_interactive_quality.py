"""Bounded read-only probes of the live controller and its actual render postprocessor.

These are single-case diagnostics, not style benchmark or paper foot-sliding metrics.
Outputs go into a separate directory; checkpoints and production sampling stay unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Visualization.cli.interactive_label_control import load_session
from Visualization.io.genoview_bvh import (
    _load_template, _template_local_transforms, _expand_selected_to_full,
    _convert_to_geno_basis, GENOVIEW_RESOURCES_DIR,
)
from Visualization.pose_postprocess import PosePostprocessor
from Utils.reproducibility import state_fingerprint, file_sha256
from analysis.motion_quality_diagnosis.record_control_video import control_for, analyze


def pose_metrics(positions):
    step = np.linalg.norm(np.diff(positions, axis=0), axis=-1).mean(-1) * 100
    return dict(second_difference_cm=float(np.linalg.norm(np.diff(positions, n=2, axis=0), axis=-1).mean()*100),
                step_p99_cm=float(np.percentile(step, 99)), step_max_cm=float(step.max()))


@torch.no_grad()
def run(args):
    ns = SimpleNamespace(db=ROOT/'database_100sty_375_memmap_nomirror', flow_ckpt=args.checkpoint,
        vqvae_ckpt=None, norm_stats=None, device=args.device, style='Neutral', source_style=None,
        seed=42, ode_steps=None, noise_std=args.noise_std, style_blend=.25, requantize=args.requantize,
        turn_speed=90.)
    session, info = load_session(ns)
    model = session.model
    fingerprint = state_fingerprint(model.vqvae.state_dict())
    template = _load_template(GENOVIEW_RESOURCES_DIR/'Geno_bind.bvh')
    offsets, rest_rot = _template_local_transforms(template, 0)
    processor = PosePostprocessor(template['names'], template['parents'], offsets)
    original_sample = model.sample_latent
    continuous = []
    previous_noise = None
    snapshot = None

    def capture(**kwargs):
        nonlocal previous_noise, snapshot
        if args.noise_rho:
            # Preserve N(0,I) marginal; only temporal correlation changes. Diagnostic only.
            epsilon = (kwargs['z0']-model.latent_offset)/(model.latent_scale*model.latent_transport_std*args.noise_std)
            if previous_noise is not None:
                epsilon = args.noise_rho*previous_noise + np.sqrt(1-args.noise_rho**2)*epsilon
            previous_noise = epsilon.clone()
            kwargs['z0'] = model.latent_offset+model.latent_scale*model.latent_transport_std*args.noise_std*epsilon
        if session.frames == 60:
            snapshot = {k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()}
        z = original_sample(**kwargs)
        continuous.append(model.vqvae.decode(model.clip_latent_to_data_bounds(z)).clone())
        return z

    model.sample_latent = capture
    selected, raw_motion, full, processed, contacts, weights = [], [], [], [], [], []
    for frame in range(300):
        session.step(*control_for(frame), speed=1.2)
        p, r = session.selected_pose()
        selected.append(p[0].copy()); raw_motion.append(session.raw_motion.copy())
        p, r = _convert_to_geno_basis(p, r, enabled=session.coordinate_convert)
        p, r = _expand_selected_to_full(p, r, full_names=template['names'], full_parents=template['parents'],
            template_local_pos=offsets, template_local_rot=rest_rot)
        full.append(p[0].copy())
        pp, _ = processor.process(p[0], r[0])
        processed.append(pp.copy())
        contacts.append([s['active'] for s in processor.contacts])
        weights.append([s['weight'] for s in processor.contacts])
    model.sample_latent = original_sample
    full = np.asarray(full); processed = np.asarray(processed)
    raw_motion = np.asarray(raw_motion); contacts = np.asarray(contacts); weights = np.asarray(weights)
    cont_motion = (torch.cat(continuous)*session.std_t[84:]+session.mean_t[84:]).cpu().numpy()
    # Same-input causal comparison: continuous and quantized endpoints under exactly
    # the same history, trajectory and noise. Neither variant changes the feedback here.
    counterfactual = {}
    generator = torch.Generator(device=session.device).manual_seed(987)
    epsilon = torch.randn((32, model.latent_dim), device=session.device, generator=generator)
    for steps in (2, 8):
        kwargs = {k: v.expand(32, *v.shape[1:]) if torch.is_tensor(v) else v for k, v in snapshot.items()}
        kwargs['num_steps'] = steps
        kwargs['z0'] = model.latent_offset+model.latent_scale*model.latent_transport_std*args.noise_std*epsilon
        z = original_sample(**kwargs)
        q_motion, q = model.feedback_state(z)
        c_motion = model.vqvae.decode(model.clip_latent_to_data_bounds(z))
        counterfactual[str(steps)] = dict(
            continuous_pose_std_cm=float((c_motion[:, 3:75]*session.std_t[87:159]).reshape(32,24,3).std(0,unbiased=False).norm(dim=-1).mean()*100),
            quantized_pose_std_cm=float((q_motion[:, 3:75]*session.std_t[87:159]).reshape(32,24,3).std(0,unbiased=False).norm(dim=-1).mean()*100),
            continuous_quantized_pose_gap_cm=float(((q_motion-c_motion)[:,3:75]*session.std_t[87:159]).reshape(32,24,3).norm(dim=-1).mean()*100),
            unique_quantized_states=int(torch.unique(q, dim=0).shape[0]))
    support = {}
    # Use identical height-only contact pairs from the processed output for both
    # paths; do not prefilter by speed, which would hide skating. Contact zero = None.
    for name, p in [('raw',full),('processed',processed)]:
        speeds, count = [], 0
        for leg in processor.legs:
            _, _, foot, toe = leg
            height = np.minimum(processed[:,foot,1]-.08, processed[:,toe,1]-.02)
            mask = (height[:-1] < .04) & (height[1:] < .04)
            speed = np.linalg.norm(np.diff(p[:,foot,[0,2]],axis=0),axis=-1)*60*100
            speeds.extend(speed[mask]); count += int(mask.sum())
        support[name] = dict(ground_proxy_speed_cm_s=float(np.mean(speeds)) if count else None,
            contact_pairs=count, coverage=count/(2*299))
    pos = raw_motion[:,3:75].reshape(300,24,3)
    cpos = cont_motion[:,3:75].reshape(300,24,3)
    report = dict(info=info, noise_rho=args.noise_rho, protocol='single_Neutral_keyboard_case_300f_seed42_not_benchmark',
        checkpoint_sha256=file_sha256(args.checkpoint), raw_local=pose_metrics(pos), continuous_local_same_feedback=pose_metrics(cpos),
        raw_render=pose_metrics(full), processed_render=pose_metrics(processed),
        glide=analyze(np.asarray(selected), name='raw')['summary'], support=support,
        ik_active_fraction=float(contacts.mean()), ik_fully_blended_fraction=float((weights>=.99).mean()),
        frozen_history_counterfactual=counterfactual, vq_unchanged=fingerprint==state_fingerprint(model.vqvae.state_dict()))
    assert report['vq_unchanged']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'trace.npz', raw_motion=raw_motion, continuous_motion=cont_motion,
        raw_render=full, processed_render=processed, contacts=contacts, weights=weights)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-rho', type=float, default=0.)
    parser.add_argument('--requantize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([args.noise_std,args.noise_rho]).all() or args.noise_std<=0 or not 0<=args.noise_rho<1:
        parser.error('Require finite noise_std > 0 and 0 <= noise_rho < 1')
    run(args)
