"""Read-only batch correspondence and target-Euler audit; never trains a model."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Visualization.cli.interactive_label_control import parse_args, load_session
from train_flowmatch_vqvae_controller import matched_root_targets
from Utils.reproducibility import state_fingerprint
from analysis.motion_quality_diagnosis.record_control_video import control_for


def audit_velocity_consistency():
    """CPU replay: compare each sequence's decoded velocities with its actual pose differences."""
    from Utils import quat
    from Utils.motion375 import reconstruct_selected_globals_from_375, reconstruct_root_track_from_375
    from analysis.motion_quality_diagnosis.record_control_video import gt_reference
    folder = ROOT/'analysis/interactive_quality_2026-10-12'
    report = {}
    for tag in ('gt', 'best_std1', 'best_std025', 'best_std1_rho0999'):
        if tag == 'gt':
            reference = gt_reference('Neutral', 300, 42)
            x = reference['raw']
        else:
            with np.load(folder/tag/'trace.npz') as trace:
                raw = trace['raw_motion']
            x = np.concatenate((np.zeros((len(raw),84),np.float32),raw),1)
        p, _ = reconstruct_selected_globals_from_375(x)
        _, root_rotation = reconstruct_root_track_from_375(x)
        velocity = quat.mul_vec(root_rotation[:,None], x[:,303:375].reshape(-1,24,3))
        actual = (p[2:]-p[:-2])*30
        error = np.linalg.norm(actual-velocity[1:-1],axis=-1)
        feet = [18,19,22,23]
        report[tag] = dict(joint_velocity_mismatch_m_s_mean=float(error.mean()),
            feet_velocity_mismatch_m_s_mean=float(error[:,feet].mean()),
            feet_predicted_speed_m_s_mean=float(np.linalg.norm(velocity[1:-1,feet],axis=-1).mean()),
            feet_actual_speed_m_s_mean=float(np.linalg.norm(actual[:,feet],axis=-1).mean()))
        if tag == 'gt':
            report[tag].update(source_entry_id=reference['segment'],suffix=reference['suffix'])
    return report


@torch.no_grad()
def main():
    output = ROOT/'analysis/interactive_quality_2026-10-12/no_loss_audit'
    output.mkdir(parents=True, exist_ok=True)
    args = parse_args(['--headless','--max-frames','61','--style','Neutral',
        '--flow-ckpt',str(ROOT/'checkpoints/flowmatch_label_6tok_curriculum100/best.pt')])
    session, info = load_session(args)
    model = session.model
    before = state_fingerprint(model.state_dict())
    original = model.sample_latent
    snapshot = {}
    def capture(**kwargs):
        if session.frames == 60:
            snapshot.update({k: v.clone() if torch.is_tensor(v) else v for k,v in kwargs.items()})
        return original(**kwargs)
    model.sample_latent = capture
    for frame in range(61):
        session.step(*control_for(frame), speed=1.2)
    model.sample_latent = original
    generator = torch.Generator(device=session.device).manual_seed(987)
    epsilon = torch.randn((32,model.latent_dim),device=session.device,generator=generator)
    kwargs = {k:v.expand(32,*v.shape[1:]) if torch.is_tensor(v) else v for k,v in snapshot.items()}
    kwargs['z0'] = model.latent_offset+model.latent_scale*model.latent_transport_std*epsilon
    predict = model._predict_velocity
    traces = []
    def track(**values):
        result = predict(**values)
        endpoint = model.vqvae.decode(model.denormalize_latent(result[2]))
        pose = (endpoint[:,3:75]*session.std_t[87:159]).reshape(32,24,3)
        traces.append(dict(t=float(values['t'][0]), endpoint_pose_std_cm=float(pose.std(0,unbiased=False).norm(dim=-1).mean()*100)))
        return result
    outputs = {}
    for steps in (2,4,8,16):
        traces.clear(); model._predict_velocity = track
        z = original(**dict(kwargs,num_steps=steps))
        model._predict_velocity = predict
        motion = model.vqvae.decode(z)
        pose = (motion[:,3:75]*session.std_t[87:159]).reshape(32,24,3)
        # Target-Euler's last update must equal f(z_(N-1), (N-1)/N, conditions).
        outputs[str(steps)] = dict(endpoint_pose_std_cm=float(pose.std(0,unbiased=False).norm(dim=-1).mean()*100),
                                  query_trace=list(traces))

    meta = json.loads((args.db/'meta.json').read_text())
    segments = [s for s in json.loads((args.db/'segments.json').read_text()) if s['split']=='train' and s['num_frames']>=5]
    source = np.memmap(args.db/'X.f32',mode='r',dtype=np.float32,shape=(meta['num_frames'],375))
    counts = np.array([s['num_frames']-4 for s in segments]); cumulative = counts.cumsum()
    rng = np.random.default_rng(42)
    window_ids = rng.integers(int(cumulative[-1]),size=1024)
    segment_ids = np.searchsorted(cumulative,window_ids,side='right')
    offsets = window_ids-np.r_[0,cumulative[:-1]][segment_ids]
    starts = np.array([segments[i]['frame_start'] for i in segment_ids])+offsets
    raw = torch.from_numpy(np.array(source[starts[:,None]+np.arange(5)],copy=True)).to(session.device)
    x = (raw-session.mean_t)/session.std_t
    labels = torch.tensor([segments[i]['label'] for i in segment_ids],device=session.device)
    suffixes = [segments[i]['suffix'] for i in segment_ids]
    latent = model.frame_latent(x.flatten(0,1)).reshape(1024,5,-1)
    torch.manual_seed(42)
    labels_out, _, recon, fraction, donors = matched_root_targets(x_window=x,labels=labels,latents=latent,
        suffixes=suffixes,norm_mean=session.mean,norm_std=session.std,frame_latent_fn=model.frame_latent,
        prob=.3,return_donors=True,style_count=model.num_styles)
    changed = labels_out!=labels
    pos = raw[...,87:159].reshape(1024,5,24,3)
    synthetic = recon*session.std_t[84:]+session.mean_t[84:]
    target_pos = synthetic[...,3:75].reshape(1024,4,24,3)
    gt_step = (pos[:,1]-pos[:,0]).norm(dim=-1).mean(-1)*100
    cross_step = (target_pos[:,0]-pos[:,0]).norm(dim=-1).mean(-1)*100
    feet = pos[..., [18,19,22,23], :]
    # Left/right difference cancels root translation. Its short-window change
    # is a phase-direction proxy, not a style-independent ground-truth phase label.
    lr = feet[...,:2,:].mean(-2)-feet[...,2:,:].mean(-2)
    phase_v = (lr[:,-1]-lr[:,1])*20
    donor_v = phase_v[donors]
    norm = phase_v.norm(dim=-1); donor_norm = donor_v.norm(dim=-1)
    cosine = (phase_v*donor_v).sum(-1)/(norm*donor_norm).clamp_min(1e-8)
    moving = changed & (norm>.1) & (donor_norm>.1)
    roots = raw[:,1:,84:87]
    dv = (roots[:,None,:,[0,2]]-roots[None,:,:,[0,2]]).norm(dim=-1)
    dy = (roots[:,None,:,1]-roots[None,:,:,1]).abs()
    limits = torch.maximum(roots[...,[0,2]].norm(dim=-1)*.2, roots.new_tensor(.25))
    suffix_map = {name:i for i,name in enumerate(dict.fromkeys(suffixes))}
    suffix_ids = torch.tensor([suffix_map[s] for s in suffixes],device=session.device)
    valid = ((dv<=limits[:,None]).all(-1) & (dy<=30).all(-1)
             & (suffix_ids[:,None]==suffix_ids[None]) & (labels[:,None]!=labels[None]))
    phase_cosine = phase_v@phase_v.T/(norm[:,None]*norm[None]).clamp_min(1e-8)
    direction_ok = (norm[:,None]<=.1) | (norm[None]<=.1) | (phase_cosine>=0)
    fixed_style = labels[None]==labels_out[:,None]
    donor_report = dict(samples=1024,changed=int(changed.sum()),actual_cross_fraction=float(fraction),
        matched_one_frame_pose_step_cm=float(gt_step[changed].mean()),
        cross_target_one_frame_pose_step_cm=float(cross_step[changed].mean()),
        phase_direction_eligible=int(moving.sum()),
        phase_direction_opposite_fraction=float((cosine[moving]<0).float().mean()) if moving.any() else None,
        same_target_style_has_nonopposite_candidate_fraction=float((valid & direction_ok & fixed_style).any(-1)[changed].float().mean()),
        gt_normalized_latent_step_std_mean=float(model.normalize_latent(latent).diff(dim=1).flatten(0,1).std(0,unbiased=False).mean()),
        note='Phase sign and whole-body step are diagnostic proxies, not proof of incorrect style transitions.')
    result = dict(info=info,solver_outputs=outputs,train_only_batch_donors=donor_report,
                  model_state_unchanged=before==state_fingerprint(model.state_dict()),seed=42,fp32=True)
    assert result['model_state_unchanged']
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
