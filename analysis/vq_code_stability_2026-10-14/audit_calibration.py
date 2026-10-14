"""Read-only A/B audit: norms, fixed teacher, GT derivatives, gradient scale, RNG.

No optimizer steps; CPU FP32, two threads. Reuses the original 12-clip seed42
probe selection (122 frames, hence 120 second differences). Writes audit JSON only.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import load_norm_stats
from Utils.reproducibility import state_fingerprint
from training.train_basic_vqvae_375to291 import build_frozen_teacher


def measure(a, b):
    error = a - b
    return {name: float(torch.linalg.vector_norm(torch.diff(error, n=n, dim=0), dim=-1).mean())
            for n, name in enumerate(('r', 'dr', 'd2r'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = {'baseline': 'vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror',
             'A': 'vqvae_decoder_calib_A', 'B': 'vqvae_decoder_calib_B'}
    paths = {k: ROOT / 'checkpoints' / n for k, n in names.items()}
    norms = {k: load_norm_stats(p / 'norm_stats.npz') for k, p in paths.items()}
    models = {k: build_vqvae_from_checkpoint(p / 'best.pt', freeze=False).to(dev).eval()
              for k, p in paths.items()}
    before = {k: state_fingerprint(m.state_dict()) for k, m in models.items()}
    rep = {k: state_fingerprint({n: v for n, v in m.state_dict().items()
                               if n.startswith(('encoder.', 'quantizer.'))}) for k, m in models.items()}
    mean, std = norms['baseline']
    norm_report = {k: dict(equal=all(np.array_equal(a, b) for a, b in zip(norms[k], norms['baseline'])),
                          mean_shift_in_baseline_std_max=float(np.max(np.abs(norms[k][0]-mean)/std)),
                          std_ratio_change_max=float(np.max(np.abs(norms[k][1]/std-1)))) for k in names}
    db = ROOT / 'database_100sty_375_memmap_nomirror'
    segs = json.loads((db / 'segments.json').read_text())
    meta = json.loads((db / 'meta.json').read_text())
    data = np.memmap(db / 'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
    pool = [s for s in segs if s.get('split') == 'val' and not s.get('mirror', False) and s['num_frames'] >= 122]
    rng = np.random.RandomState(42)
    picks = [pool[i] for i in rng.choice(len(pool), 12, replace=False)]
    results = []
    for s in picks:
        rows = np.array(data[s['frame_start']:s['frame_start']+122])
        x = torch.as_tensor((rows-mean)/std, device=dev)
        st, mt = torch.as_tensor(std[84:], device=dev), torch.as_tensor(mean[84:], device=dev)
        pos = lambda motion: ((motion*st+mt)[:, 3:75].reshape(-1, 24, 3)*100)
        with torch.no_grad():
            base = models['baseline']
            ze = base.encode_pre_vq(x)
            q, idx, _ = base.encode(x)
            fixed = base.decode(ze)
            gt = pos(x[:, 84:])
            record = dict(source_entry_id=s.get('source_entry_id'), start=s['frame_start'], frames=122, models={})
            for k, m in models.items():
                mq, mc = m.decode(q), m.decode(ze)
                native = torch.as_tensor((rows-norms[k][0])/norms[k][1], device=dev)
                nq, ni, _ = m.encode(native)
                native_out = m.decode(nq)
                ns, nm = (torch.as_tensor(a[84:], device=dev) for a in (norms[k][1], norms[k][0]))
                native_pos = ((native_out*ns+nm)[:, 3:75].reshape(-1, 24, 3)*100)
                record['models'][k] = dict(self_residual=measure(pos(mq), pos(mc)),
                    fixed_teacher_residual=measure(pos(mq), pos(fixed)), gt_error=measure(pos(mq), gt),
                    native_norm_gt_error=measure(native_pos, gt),
                    fixed_teacher_mse=float((mq-fixed).square().mean()),
                    norm_changed_slots=float((idx != ni).float().mean()),
                    norm_changed_frames=float((idx != ni).reshape(122, -1).any(-1).float().mean()))
            results.append(record)
    averages = {}
    for k in names:
        averages[k] = {}
        for key, value in results[0]['models'][k].items():
            averages[k][key] = ({n: float(np.mean([r['models'][k][key][n] for r in results])) for n in value}
                               if isinstance(value, dict) else float(np.mean([r['models'][k][key] for r in results])))
    # Real train data, four independent B32 x T4 batches; no optimizer updates.
    train = [s for s in segs if s['split'] == 'train' and s['num_frames'] >= 4]
    rng = np.random.default_rng(42)
    grads = {k: [] for k in names}
    for batch_id in range(4):
        rows = []
        for _ in range(32):
            s = train[int(rng.integers(len(train)))]; start = s['frame_start']+int(rng.integers(s['num_frames']-3))
            rows.append(np.array(data[start:start+4]))
        rows = np.stack(rows)
        # Actual A/B training normalization, for all models to compare identical inputs.
        x = torch.as_tensor((rows-norms['B'][0])/norms['B'][1], device=dev)
        with torch.no_grad():
            target = models['baseline'].decode(models['baseline'].encode_pre_vq(x))
        for k, m in models.items():
            m.set_representation_frozen(True)
            m.train()
            out = m.compute_loss(x, recon_weight=3, vel_weight=15, root_weight=3, vq_weight=0,
                                 teacher_align_weight=2, teacher_motion=target)
            ps = list(m.decoder.parameters())
            flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
            loss_gt = 3*out['recon_loss']+15*out['vel_loss']+3*out['root_loss']
            loss_teacher = 2*out['teacher_align_loss']
            gg = flat(torch.autograd.grad(loss_gt, ps, retain_graph=True))
            gt = flat(torch.autograd.grad(loss_teacher, ps))
            grads[k].append(dict(gt_norm=float(gg.norm()), teacher_norm=float(gt.norm()),
                                 ratio=float(gt.norm()/gg.norm()), cosine=float(torch.nn.functional.cosine_similarity(gg, gt, dim=0)),
                                 teacher_loss_share=float((loss_teacher/(loss_gt+loss_teacher)).detach())))
    # Teacher construction consumes global RNG, used by DataLoader's unseeded sampler.
    config = json.loads((paths['B'] / 'config.json').read_text())
    config['teacher_ckpt'] = Path(config['teacher_ckpt'])
    torch.manual_seed(42)
    rng_before = torch.get_rng_state().clone()
    build_frozen_teacher(SimpleNamespace(**config), device=torch.device('cpu'))
    rng_after = torch.get_rng_state().clone()
    torch.set_rng_state(rng_before); order_a = torch.randperm(100).tolist()
    torch.set_rng_state(rng_after); order_b = torch.randperm(100).tolist()
    # Paired source-cluster bootstrap describes case uncertainty only, not training seeds.
    paired = {}
    for candidate, reference in (('A', 'baseline'), ('B', 'baseline'), ('B', 'A')):
        for metric in ('fixed_teacher_residual', 'gt_error'):
            grouped = {}
            for r in results:
                grouped.setdefault(r['source_entry_id'], []).append(
                    r['models'][candidate][metric]['d2r']-r['models'][reference][metric]['d2r'])
            values = np.array([np.mean(v) for v in grouped.values()])
            rng = np.random.default_rng(42)
            boot = values[rng.integers(len(values), size=(1000,len(values)))].mean(-1)
            paired[f'{candidate}_minus_{reference}/{metric}/d2r'] = dict(
                mean=float(values.mean()), ci95=np.quantile(boot,[.025,.975]).tolist(), clusters=len(values))
    result = dict(device=str(dev), torch=torch.__version__, protocol='FP32 no optimizer updates; seed42; val12 prefixes122; gradients4x32x4',
        representation_identical=len(set(rep.values()))==1, norms=norm_report, averages=averages,
        gradients=grads, paired_d2_differences=paired, per_clip=results, teacher_construction_changes_rng=not torch.equal(rng_before,rng_after),
        example_sampler_prefix_A=order_a[:10], example_sampler_prefix_B=order_b[:10],
        all_model_states_unchanged=all(before[k]==state_fingerprint(m.state_dict()) for k, m in models.items()))
    out = Path(__file__).with_name('calibration_independent_audit.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_clip','gradients')}, indent=2))
    print('Gradient mean:', {k:{n:float(np.mean([g[n] for g in gs])) for n in gs[0]} for k,gs in grads.items()})
    print(out)


if __name__ == '__main__':
    main()
