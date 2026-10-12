"""Fixed 90-clip validation rollout; compare batching with the earlier 8-case record."""
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Utils.controller_validation import TeacherForcedStyleValidation
from Utils.motion_eval import load_motion_model
from Utils.runtime import configure_torch_runtime, load_norm_stats
from Utils.reproducibility import state_fingerprint


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    configure_torch_runtime(device=torch.device('cuda'), tf32=False, cudnn_benchmark=False, matmul_precision='highest')
    model = load_motion_model(model_name='flowmatch_target', ckpt_path=ROOT/'checkpoints/clip_recipe_gt_history50/best.pt', num_styles=90).to('cuda').eval()
    before = state_fingerprint(model.vqvae.state_dict())
    mean, std = load_norm_stats(ROOT/'checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep/norm_stats.npz')
    validator = TeacherForcedStyleValidation(model, db=ROOT/'database_100sty_375_memmap', mean=mean, std=std,
        classifier_path=ROOT/'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth',
        clips=90, windows=1, batch_size=64, eval_mode='rollout')
    folder = ROOT/'analysis/clip_recipe_failure_2026-10-09/shared_rollout_validation'
    old_manifest = json.loads((folder/'rollout_style_validation_cases.json').read_text())
    assert validator.manifest == old_manifest
    old = json.loads((folder/'summary.json').read_text())['results']['clip_recipe_gt_history50']
    t0 = time.monotonic(); metrics = validator.evaluate(model); torch.cuda.synchronize()
    assert state_fingerprint(model.vqvae.state_dict()) == before
    result = dict(batch_size=64, seconds=time.monotonic()-t0, before_batch_size=8,
        before=old, after=metrics, identical_case_manifest=True, frozen_vq_unchanged=True)
    (Path(__file__).parent/'validation_batch64.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
