#!/usr/bin/env bash
set -euo pipefail

# One clip/recipe controller, from scratch. Practical branch: 50 epochs; original: 150.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/checkpoints/clip_recipe_scratch150}"
VQ_DIR="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
if [[ "${SMOKE:-0}" == 1 ]]; then RUN_ROOT="${RUN_ROOT}_smoke"; fi
if [[ $# != 0 || -n "${INITIAL_CKPT:-}" || -n "${INIT_CKPT:-}" ]]; then
  echo 'This entry trains from scratch. Use DRY_RUN=1, SMOKE=1, or RESUME=1 for this new run; initialization checkpoints are not accepted.' >&2
  exit 2
fi
training_epochs=150
if [[ "${PRACTICAL_HISTORY:-0}" == 1 ]]; then
  training_epochs=50
fi

cmd=("${PYTHON_BIN}" "${ROOT_DIR}/training/train_flowmatch_vqvae_controller.py"
  --db "${ROOT_DIR}/database_100sty_375_memmap_nomirror"
  --vqvae-ckpt "${VQ_DIR}/best.pt" --norm-stats "${VQ_DIR}/norm_stats.npz"
  --latent-stats "${RUN_ROOT}/latent_stats.npz"
  --vq-target-cache "${ROOT_DIR}/cache/vq_targets_vq150ep_nomirror_repair_v2"
  --style-code-prior "${ROOT_DIR}/cache/style_code_prior_vq150ep_nomirror_repair_v2.npz"
  --out-dir "$RUN_ROOT" --epochs "$training_epochs" --lr-schedule-epochs "$training_epochs"
  --conditioning-mode clip_recipe --recipe-prob 1 --recipe-frames 240 --recipe-exemplar-choices 1 3 5
  --recipe-bank-size 64 --recipe-bank-max-entries "${RECIPE_BANK_MAX_ENTRIES:-131072}"
  --batch-size 512 --microbatch-size "${MICROBATCH_SIZE:-512}" --seed 42
  --seq-len 2 --stride 1 --unroll-steps 1 --no-train-feedback-context
  --prediction-mode target --flow-loss-space normalized
  --flow-hidden-dim 256 --flow-layers 2 --flow-heads 4 --flow-ffn-dim 1024 --dropout 0
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1
  --style-proto-normalization none --style-selector-init random --style-prior-loss dense
  --style-hist-weight .02 --style-entropy-weight .003
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0
  --lr 5e-4 --min-lr 1e-5 --weight-decay 1e-4 --grad-clip 1
  --noise-std 1 --solver-steps 8 --inference-requantize --inference-latent-clip
  --matmul-precision highest --no-tf32 --rollout-val-every 0
  --worker-batches --fp32-math-attention --style-val-batch-size "${STYLE_VAL_BATCH_SIZE:-64}"
  --tf-val-every 5 --style-val-mode rollout --tf-val-clips 90 --tf-val-windows 3 --tf-val-selection absolute
  --tf-val-min-matched-accuracy .85 --tf-val-max-matched-ade 5 --tf-val-max-cross-ade 10
  --val-every 1 --save-every 5 --num-workers "${NUM_WORKERS:-8}" --prefetch-factor 4
  --device "${DEVICE:-cuda}")

if [[ "${PRACTICAL_HISTORY:-0}" == 1 ]]; then
  cmd+=(--flow-low-t-probability .5 --flow-low-t-max .2
    --stop-after-epoch "${STOP_AFTER_EPOCH:-0}")
fi

if [[ "${SMOKE:-0}" == 1 ]]; then
  cmd+=(--epochs 1 --max-train-batches 2 --max-val-batches 1 --tf-val-every 0 --num-workers 0)
fi
if [[ "${RESUME:-0}" == 1 ]]; then
  cmd+=(--resume "${RUN_ROOT}/latest.pt")
fi
printf '[Run scratch%s]' "$training_epochs"; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
"${cmd[@]}"
if [[ "${SMOKE:-0}" != 1 ]]; then
  "${PYTHON_BIN}" - "$RUN_ROOT" <<'SELECT_PY'
from pathlib import Path
import json
import sys
import torch
root = Path(sys.argv[1])
path = root/'best.pt'
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
selected = dict(checkpoint['selection'])          # already carries epoch/metrics/constraints_passed
selected['path'] = str(path.resolve())
selected['epoch'] = checkpoint['epoch']
latest = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
(root/'selected_checkpoint.json').write_text(json.dumps(dict(training_origin='from_scratch',
    training_complete=latest['epoch'] >= latest['args']['epochs'], completed_epochs=latest['epoch'],
    planned_epochs=latest['args']['epochs'], selected=selected), indent=2))
print('[Selected by validation]', selected['path'])
SELECT_PY
fi
