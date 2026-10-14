#!/usr/bin/env bash
set -euo pipefail

# Plan-A observation-noise augmentation, pure teacher forcing for all 100 epochs.
#
# The 50 TF + 20 ramp + 30 rollout course is intentionally NOT used here: the current
# direction keeps the controller on the TF manifold and lets the observation noise
# (clip->quantize->decode(z_gt + U[0,A]*transport_std*eps)) provide the robustness that
# the rollout phase used to supply. Feedback probability stays 0 for the whole run, so
# training never runs Euler/decode except the noise re-renderings (3 per unroll window).
#
# Architecture (2026-10-13): 6-layer transformer field (was 2). The encoder activation
# is already GELU (nn.TransformerEncoderLayer(activation="gelu"), norm_first) -- GELU is
# the only activation in the label inference path and no switch is needed.
# The 2-layer results stay in checkpoints/flowmatch_label_history_obsnoise_tf100/;
# this entry now defaults to the 6-layer output dir. Reproduce the 2-layer run with
# FLOW_LAYERS=2 OUT=checkpoints/flowmatch_label_history_obsnoise_tf100.
#
# Batch: 2048 by default (sqrt-scaled lr 1e-3 from the 5e-4@512 base). Set BATCH_SIZE=1024
# LR=7e-4 to reproduce the earlier scale. For short pilots set EPOCHS=10 and keep the LR on
# the sqrt rule for the chosen batch (512 -> 5e-4); lr-schedule-epochs follows EPOCHS.
# See analysis/control_operators_study_2026-10-12/OBSERVATION_NOISE_PLAN_A.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OUT="${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_history_obsnoise_tf100_l6}"
export LR="${LR:-1e-3}"
export BATCH_SIZE="${BATCH_SIZE:-2048}"
export FLOW_LAYERS="${FLOW_LAYERS:-6}"
export EPOCHS="${EPOCHS:-100}"
# STYLE_FILM=1 enables the per-layer style FiLM on the transformer generation query
# (zero-initialized, so the run starts at the unmodulated field and trains the modulation).
export STYLE_FILM="${STYLE_FILM:-0}"
# DROPOUT applies to the style selector; unset FIELD_DROPOUT makes the field inherit it.
# Set FIELD_DROPOUT separately (e.g. DROPOUT=0 FIELD_DROPOUT=0.2) to train only the
# transformer with dropout while the selector stays deterministic (keeps its fast path).
export DROPOUT="${DROPOUT:-0}"
export FIELD_DROPOUT="${FIELD_DROPOUT:-}"
# Code-decision CE (code fidelity): weight + ramp-in epochs; 0 keeps the legacy loss.
export CODE_CE_W="${CODE_CE_W:-0}"
export CODE_CE_WARMUP="${CODE_CE_WARMUP:-0}"
# Pure teacher forcing: p=0 means no feedback window ever (no epoch enters the ramp).
export FEEDBACK_START_EPOCH=1
export FEEDBACK_RAMP_EPOCHS=0
export FEEDBACK_PROBABILITY=0
export OBS_NOISE_STD="${OBS_NOISE_STD:-0.5}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
extra=()
if [[ -n "${INIT_CKPT:-}" ]]; then
  extra+=(--init-ckpt "${INIT_CKPT}")
fi
if [[ "${STYLE_FILM}" == "1" ]]; then
  extra+=(--use-style-film)
fi
if [[ -n "${FIELD_DROPOUT}" ]]; then
  extra+=(--field-dropout "${FIELD_DROPOUT}")
fi
if [[ "${CODE_CE_W}" != "0" ]]; then
  extra+=(--code-ce-weight "${CODE_CE_W}" --code-ce-warmup-epochs "${CODE_CE_WARMUP}")
fi
bash "${ROOT_DIR}/scripts/run_train_flowmatch_6tok_curriculum100.sh" \
  --epochs "${EPOCHS}" --lr-schedule-epochs "${EPOCHS}" --seq-len 6 --history-conditioning \
  --train-feedback-context \
  --observation-noise-std "${OBS_NOISE_STD}" \
  --flow-layers "${FLOW_LAYERS}" \
  --dropout "${DROPOUT}" \
  --batch-size "${BATCH_SIZE}" \
  --gpu-data-cache-gb "${GPU_DATA_CACHE_GB:-8}" --cache-teacher-motion \
  --motion-quality-selection \
  --tf-val-every 5 --solver-steps 4 --feedback-solver-steps 4 \
  "${extra[@]}" "$@"
