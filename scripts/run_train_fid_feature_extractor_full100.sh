#!/usr/bin/env bash
set -euo pipefail

# Train the FID feature extractor on the full 100STYLE style set.
#
# The classifier features are only meaningful when the extractor has seen every style:
# a 10-style extractor puts the other 90 styles in an out-of-distribution corner of its
# feature space. This wrapper points the (generic) trainer at the merged 100-style
# database produced by tools/merge_memmap_dbs.py.
#
#   DB=database_100sty_375_memmap_fid_full100_nomirror OUT=checkpoints/fid_feature_extractor_full100_v4 \
#   ./NewCode/scripts/run_train_fid_feature_extractor_full100.sh
#
# Env overrides: DB, OUT, DEVICE, EPOCHS, SEQ_LEN, BATCH_SIZE, STRIDE, LOAD_INTO_MEMORY
# Extra flags are forwarded to training/train_fid_feature_extractor_head10.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_fid_feature_extractor_head10.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_fid_full100_nomirror}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/fid_feature_extractor_full100_v4}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
SEQ_LEN="${SEQ_LEN:-60}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
STRIDE="${STRIDE:-10}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" torch tqdm

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --seq-len "${SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --stride "${STRIDE}"
)

ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory

cmd+=("$@")
"${cmd[@]}"
