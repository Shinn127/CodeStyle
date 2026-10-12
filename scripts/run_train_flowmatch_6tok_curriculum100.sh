#!/usr/bin/env bash
set -euo pipefail

# From-scratch 100-epoch TF -> rollout curriculum run, motion-token restored (6 inputs).
#
# Decisions (each backed by this session's experiments; see analysis/controller_compare/
# and analysis/motion_quality_diagnosis/):
#   * field: transformer (skip_cat_mlp stalled in the 50-epoch attribution run)
#   * motion token RESTORED: the diagnostic control-video runs showed the no-motion-token models
#     collapse into a frozen-pose glide under low sampling noise (tr5 62% / ep86 33% of walk
#     frames) while the motion-token model stays at 1.7% -- leg-phase state cannot be inferred
#     from the 128-d quantized latent alone in long closed loops.
#   * TF -> rollout curriculum: ep 1-30 pure multi-step TF (p=0, no Euler/decode), ep 31-40
#     per-window feedback share 10% -> 100%, ep 41-100 full rollout (p=1), unroll window 4,
#     feedback 2 Euler steps (speed; the FM loss is a single field evaluation).
#   * batch 1024 + lr 7e-4 (sqrt-scaled from 5e-4): measured 1.45x wall speedup, GPU 79 -> 95%
#   * no motion-recon loss: diagnosis showed quantisation staircase was NOT the cause of the
#     display jitter (residuals/code-switches flat across models); the first recon attempt
#     diverged and the corrected dose is unvalidated, so keep the baseline clean.
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
  --out-dir "${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_6tok_curriculum100}"
  --field-arch transformer --use-motion-token
  --epochs 100 --lr-schedule-epochs 100 --seed 42
  --seq-len 5 --stride 1 --unroll-steps 4
  --feedback-start-epoch "${FEEDBACK_START_EPOCH:-31}" --feedback-ramp-epochs "${FEEDBACK_RAMP_EPOCHS:-10}"
  --feedback-probability "${FEEDBACK_PROBABILITY:-1}" --feedback-solver-steps "${FEEDBACK_SOLVER_STEPS:-2}"
  --batch-size 1024 --microbatch-size "${MICROBATCH_SIZE:-0}"
  --prediction-mode target --flow-loss-space normalized
  --lr "${LR:-7e-4}" --min-lr 1e-5 --weight-decay 1e-4 --grad-clip 1
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
