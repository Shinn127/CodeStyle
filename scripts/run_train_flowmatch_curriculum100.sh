#!/usr/bin/env bash
set -euo pipefail

# From-scratch 100-epoch TF -> rollout curriculum run:
#   ep   1-30 : pure multi-step teacher forcing (p=0). GT history for every window; the
#               unroll loop applies only the FM loss, no Euler/decoder integration.
#   ep  31-40 : per-window feedback share ramps 10% -> 100% (--feedback-ramp-epochs 10).
#   ep  41-100: full rollout training (p=1), unroll window 4.
# Velocity field: transformer, 5 inputs (no 291-d motion token) -- the direction validated by
# the 50-epoch attribution runs (skip_cat_mlp stalled; transformer is the stronger field).
# Rollout feedback uses 2 Euler steps for speed (--feedback-solver-steps 2); the FM loss is a
# single field evaluation and is not affected. History noise is removed from this codebase.
# Validation: 240-frame rollout style validation every 5 epochs (same protocol as prior runs).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
VQ_DIR="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
cmd=("${PYTHON_BIN}" "${ROOT_DIR}/training/train_flowmatch_vqvae_controller.py"
  --db "${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
  --vqvae-ckpt "${VQ_CKPT:-${VQ_DIR}/best.pt}"
  --norm-stats "${NORM_STATS:-${VQ_DIR}/norm_stats.npz}"
  --vq-target-cache "${VQ_TARGET_CACHE:-${ROOT_DIR}/cache/vq_targets_vq150ep_nomirror_repair_v2}"
  --style-code-prior "${STYLE_CODE_PRIOR:-${ROOT_DIR}/cache/style_code_prior_vq150ep_nomirror_repair_v2.npz}"
  --latent-stats "${LATENT_STATS:-${ROOT_DIR}/checkpoints/flowmatch_label_nomirror_150ep/latent_stats.npz}"
  --out-dir "${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_curriculum100}"
  --field-arch transformer --no-use-motion-token
  --epochs 100 --lr-schedule-epochs 100 --seed 42
  --seq-len 5 --stride 1 --unroll-steps 4
  --feedback-start-epoch "${FEEDBACK_START_EPOCH:-31}" --feedback-ramp-epochs "${FEEDBACK_RAMP_EPOCHS:-10}"
  --feedback-probability "${FEEDBACK_PROBABILITY:-1}" --feedback-solver-steps "${FEEDBACK_SOLVER_STEPS:-2}"
  --batch-size 512 --microbatch-size "${MICROBATCH_SIZE:-0}"
  --prediction-mode target --flow-loss-space normalized
  --lr "${LR:-5e-4}" --min-lr 1e-5 --weight-decay 1e-4 --grad-clip 1
  --flow-hidden-dim 256 --flow-layers 2 --flow-heads 4 --flow-ffn-dim 1024 --dropout 0
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1
  --style-selector-init prior --style-proto-normalization none --style-prior-loss dense
  --style-hist-weight .02 --style-entropy-weight .003 --recipe-prob 0
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0
  --noise-std 1 --solver-steps 8 --inference-requantize --inference-latent-clip
  --matmul-precision highest --no-tf32 --rollout-val-every 0
  --tf-val-every 5 --style-val-mode rollout --style-val-batch-size "${STYLE_VAL_BATCH_SIZE:-64}"
  --tf-val-clips 90 --tf-val-windows 3 --tf-val-selection absolute
  --tf-val-min-matched-accuracy .85 --tf-val-max-matched-ade 5 --tf-val-max-cross-ade 10
  --val-every 1 --save-every 5 --num-workers 8 --prefetch-factor 4)
if [[ -n "${DEVICE:-}" ]]; then cmd+=(--device "${DEVICE}"); fi
if [[ -n "${RESUME:-}" ]]; then cmd+=(--resume "${RESUME}"); fi
cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
