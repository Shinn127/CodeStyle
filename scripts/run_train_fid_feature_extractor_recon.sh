#!/usr/bin/env bash
set -euo pipefail

# Train the FID feature extractor with a *reconstruction* objective on the full 100STYLE set.
# Same data/protocol/architecture family as run_train_fid_feature_extractor_camdm.sh, but the
# head reconstructs the input window instead of classifying the style (no labels used).
#
#   ./NewCode/scripts/run_train_fid_feature_extractor_recon.sh
#
# Env overrides: DB, OUT, DEVICE, EPOCHS, BATCH_SIZE, CLIP_OFFSET, VEL_WEIGHT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_fid_feature_extractor_recon.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_fid_full100_nomirror}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/fid_feature_extractor_recon100_nomirror}"
DEVICE="${DEVICE:-cuda}"
SPLIT_BY="${SPLIT_BY:-temporal}"
VAL_SUFFIXES="${VAL_SUFFIXES:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DROPOUT="${DROPOUT:-0.3}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
CLIP_OFFSET="${CLIP_OFFSET:-10}"
VEL_WEIGHT="${VEL_WEIGHT:-0.0}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" torch tqdm

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --clip-offset "${CLIP_OFFSET}"
  --vel-weight "${VEL_WEIGHT}"
  --val-ratio "${VAL_RATIO}"
  --split-by "${SPLIT_BY}"
  --val-suffixes "${VAL_SUFFIXES}"
  --split-seed "${SPLIT_SEED}"
  --dropout "${DROPOUT}"
  --head-dropout "${HEAD_DROPOUT}"
)


cmd+=("$@")
"${cmd[@]}"
