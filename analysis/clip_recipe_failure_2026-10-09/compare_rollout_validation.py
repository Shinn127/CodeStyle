"""Frozen checkpoints on a shared validation manifest; no checkpoint search or training."""
from pathlib import Path
import json
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utils.controller_validation import TeacherForcedStyleValidation
from Utils.motion_eval import load_motion_model
from Utils.reproducibility import file_sha256, state_fingerprint
from Utils.runtime import configure_torch_runtime, load_norm_stats


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    configure_torch_runtime(device=torch.device('cuda'), tf32=False,
                            cudnn_benchmark=False, matmul_precision='highest')
    mean, std = load_norm_stats(ROOT / 'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep/norm_stats.npz')
    out_dir = Path(__file__).parent / 'shared_rollout_validation'
    out_dir.mkdir(exist_ok=True)
    validator = None
    results = {}
    vq_hash = None
    slots = None
    for name in ('clip_recipe_gt_history50', 'clip_recipe_scratch150', 'flowmatch_tf150_staged_noise'):
        checkpoint = ROOT / 'checkpoints' / name / 'best.pt'
        checkpoint_hash = file_sha256(checkpoint)
        model = load_motion_model(model_name='flowmatch_target', ckpt_path=checkpoint,
                                  num_styles=90).to('cuda').eval()
        before = state_fingerprint(model.vqvae.state_dict())
        if validator is None:
            validator = TeacherForcedStyleValidation(model, db=ROOT / 'database_100sty_375_memmap',
                mean=mean, std=std, classifier_path=ROOT / 'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth',
                seed=42, clips=90, windows=1, exemplars=5, out_dir=out_dir, eval_mode='rollout')
            vq_hash, slots = before, model.selected_style_slots
        assert before == vq_hash and model.selected_style_slots == slots
        started = time.monotonic()
        metrics = validator.evaluate(model)
        torch.cuda.synchronize()
        metrics.update(seconds=time.monotonic() - started, checkpoint_sha256=checkpoint_hash,
                       vq_unchanged=state_fingerprint(model.vqvae.state_dict()) == before,
                       checkpoint_unchanged=file_sha256(checkpoint) == checkpoint_hash)
        assert metrics['vq_unchanged'] and metrics['checkpoint_unchanged']
        results[name] = metrics
        print(name, json.dumps(metrics, ensure_ascii=False), flush=True)
        (out_dir / 'summary.json').write_text(json.dumps(dict(
            protocol='rollout_240f', seed=42, fp32=True, results=results,
            limitations='90 validation sources, one fixed window each, two cross targets. Shared manifest and references; no OOD selection or optimizer updates. Different training recipes and durations prevent attributing differences to a single change.'),
            ensure_ascii=False, indent=2))
        del model


if __name__ == '__main__':
    main()
