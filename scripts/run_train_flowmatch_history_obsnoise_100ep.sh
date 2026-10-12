#!/usr/bin/env bash
set -euo pipefail

# Plan-A: observation-noise augmentation on the teacher-forced branch (no Euler rollout).
#
# Rationale: ControlOperators (SIGGRAPH Asia 2025) trains single-step with input noise and
# needs no rollout for stable interaction. Our A/B measurements showed their no-quantization
# feedback loop is not the decisive difference; the missing input-noise augmentation is.
# See analysis/control_operators_study_2026-10-12/REPORT.md and OBSERVATION_NOISE_PLAN_A.md.
#
# Config: same 50 TF + 20 ramp + 30 rollout course as the ODE4 run, history conditioning and
# Euler4 everywhere; the TF epochs now see clip->quantize->decode(z_gt + U[0,A]*transport_std*eps)
# as the rolled observation. Set OBS_NOISE_STD=0 for the clean baseline A/B.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OUT="${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_history_obsnoise_100ep}"
export LR="${LR:-7e-4}"
export FEEDBACK_START_EPOCH="${FEEDBACK_START_EPOCH:-51}"
export FEEDBACK_RAMP_EPOCHS="${FEEDBACK_RAMP_EPOCHS:-20}"
export FEEDBACK_PROBABILITY="${FEEDBACK_PROBABILITY:-1}"
export FEEDBACK_SOLVER_STEPS="${FEEDBACK_SOLVER_STEPS:-4}"
export OBS_NOISE_STD="${OBS_NOISE_STD:-0.5}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
extra=()
if [[ -n "${INIT_CKPT:-}" ]]; then
  extra+=(--init-ckpt "${INIT_CKPT}")
fi
bash "$SCRIPT_DIR/run_train_flowmatch_6tok_curriculum100.sh" \
  --epochs 100 --lr-schedule-epochs 100 --seq-len 6 --history-conditioning \
  --train-feedback-context --feedback-batch-schedule batch \
  --observation-noise-std "${OBS_NOISE_STD}" \
  --gpu-data-cache-gb "${GPU_DATA_CACHE_GB:-9}" --cache-teacher-motion \
  --motion-quality-selection \
  --tf-val-every 5 --solver-steps 4 --feedback-solver-steps 4 \
  "${extra[@]}" "$@"
