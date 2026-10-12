#!/usr/bin/env bash
set -euo pipefail

# Donor-free, history-conditioned 100-epoch run with uniform Euler4 ODE.
#
# Curriculum: 50 epochs pure TF (p=0, no Euler/decode) -> 20 epochs ramp (p 0.05..1.00,
# per-batch deterministic gate) -> 30 epochs full rollout (p=1.0).
# Solver: --solver-steps 4 AND --feedback-solver-steps 4 everywhere, so training feedback,
# validation selection and the interactive window all integrate on the same 4-step Euler
# grid (the old Euler2-feedback vs Euler8-inference mismatch is gone).
#
# Donor/cross-style supervision was removed on 2026-10-12 (objective version 11); this run
# trains from scratch (no compatible init checkpoint survives) unless INIT_CKPT is set.
# See analysis/controller_compare/ODE4_100EP_PLAN.md for the full config and rationale.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OUT="${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_history_ode4_100ep}"
export LR="${LR:-7e-4}"
# 50 TF epochs, then a 20-epoch ramp, then 30 epochs of full rollout.
export FEEDBACK_START_EPOCH="${FEEDBACK_START_EPOCH:-51}"
export FEEDBACK_RAMP_EPOCHS="${FEEDBACK_RAMP_EPOCHS:-20}"
export FEEDBACK_PROBABILITY="${FEEDBACK_PROBABILITY:-1}"
export FEEDBACK_SOLVER_STEPS="${FEEDBACK_SOLVER_STEPS:-4}"
# Large immutable FP32 caches otherwise fragment the CUDA allocator over a full epoch.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
extra=()
if [[ -n "${INIT_CKPT:-}" ]]; then
  extra+=(--init-ckpt "${INIT_CKPT}")
fi
bash "$SCRIPT_DIR/run_train_flowmatch_6tok_curriculum100.sh" \
  --epochs 100 --lr-schedule-epochs 100 --seq-len 6 --history-conditioning \
  --train-feedback-context --feedback-batch-schedule batch \
  --gpu-data-cache-gb "${GPU_DATA_CACHE_GB:-9}" --cache-teacher-motion \
  --motion-quality-selection \
  --tf-val-every 5 --solver-steps 4 --feedback-solver-steps 4 \
  "${extra[@]}" "$@"
