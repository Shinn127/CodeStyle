#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export OUT="${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_history_tf10}"
export LR="${LR:-5e-5}"
# Pure teacher forcing: p=0 means no feedback window ever, so no Euler/decode during training.
export FEEDBACK_START_EPOCH=1
export FEEDBACK_RAMP_EPOCHS=0
export FEEDBACK_PROBABILITY=0
# Large immutable FP32 caches otherwise fragment the CUDA allocator over a full epoch.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
extra=()
if [[ -z "${RESUME:-}" && "${FROM_SCRATCH:-0}" != 1 ]]; then
  # The donor-era default init checkpoint was deleted by the 2026-10-12 cleanup.
  init_ckpt="${INIT_CKPT:-${ROOT_DIR}/checkpoints/flowmatch_label_6tok_curriculum100/best.pt}"
  if [[ -f "${init_ckpt}" ]]; then
    extra+=(--init-ckpt "${init_ckpt}")
  else
    echo "[Warn] init checkpoint not found (${init_ckpt}); training from scratch." >&2
    echo "[Warn] Set INIT_CKPT=<6-token transformer ckpt> to fine-tune, or FROM_SCRATCH=1 to silence." >&2
  fi
fi
bash "$SCRIPT_DIR/run_train_flowmatch_6tok_curriculum100.sh" \
  --epochs 10 --lr-schedule-epochs 10 --seq-len 6 --history-conditioning \
  --train-feedback-context \
  --gpu-data-cache-gb "${GPU_DATA_CACHE_GB:-9}" --cache-teacher-motion \
  --motion-quality-selection \
  --tf-val-every 5 --solver-steps 8 \
  "${extra[@]}" "$@"
