#!/usr/bin/env bash
set -euo pipefail

# Train the FID feature extractor with CAMDM's recipe on the full 100STYLE database.
#
#   ./NewCode/scripts/run_train_fid_feature_extractor_camdm.sh
#
# Env overrides: DB, OUT, DEVICE, EPOCHS, BATCH_SIZE, WINDOW_SIZE, CLIP_OFFSET, SPLIT_BY, VAL_RATIO, DROPOUT
# Extra flags are forwarded to training/train_fid_feature_extractor_camdm.py.
#
# Data: all 100 styles of `database_100sty_375_memmap_fid_full100_nomirror` (90 IND + 10 OOD merged),
# every clip treated as training data (CAMDM's train_classifier.py has no split).
# 240 frames at native 60 fps, mirrored clips excluded, 45-dim input = 15 ybot joints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_fid_feature_extractor_camdm.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_fid_full100_nomirror}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
WINDOW_SIZE="${WINDOW_SIZE:-240}"
CLIP_OFFSET="${CLIP_OFFSET:-10}"
SPLIT_BY="${SPLIT_BY:-temporal}"
VAL_SUFFIXES="${VAL_SUFFIXES:-}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DROPOUT="${DROPOUT:-0.3}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.3}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" torch tqdm

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --window-size "${WINDOW_SIZE}"
  --clip-offset "${CLIP_OFFSET}"
  --val-ratio "${VAL_RATIO}"
  --split-by "${SPLIT_BY}"
  --val-suffixes "${VAL_SUFFIXES}"
  --split-seed "${SPLIT_SEED}"
  --dropout "${DROPOUT}"
  --head-dropout "${HEAD_DROPOUT}"
)

cmd+=("$@")
"${cmd[@]}"
