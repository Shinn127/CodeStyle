#!/usr/bin/env bash
set -euo pipefail

# Usage and common recipes: NewCode/README.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/tools/precompute_style_code_prior.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
INPUT_MODE="${INPUT_MODE:-traj_motion}"
NETWORK_ARCH="${NETWORK_ARCH:-skip_cat_mlp}"
VQ_TARGET_CACHE="${VQ_TARGET_CACHE:-${ROOT_DIR}/cache/vq_targets_vq150ep_nomirror_repair_v2}"
PRIOR_NAME="style_code_prior_${VQ_TARGET_CACHE##*/vq_targets_}.npz"
OUT="${OUT:-${ROOT_DIR}/cache/${PRIOR_NAME}}"
SPLIT="${SPLIT:-train}"
STYLE_NUM_SLOTS="${STYLE_NUM_SLOTS:-8}"
SMOOTHING="${SMOOTHING:-1.0}"
CHUNK_ROWS="${CHUNK_ROWS:-32768}"

ru_require_file "${RUN_PY}" "style prior precompute script"
ru_require_python_modules "${PYTHON_BIN}" numpy

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --vq-target-cache "${VQ_TARGET_CACHE}"
  --out "${OUT}"
  --split "${SPLIT}"
  --style-num-slots "${STYLE_NUM_SLOTS}"
  --smoothing "${SMOOTHING}"
  --chunk-rows "${CHUNK_ROWS}"
)

cmd+=("$@")
ru_run "${cmd[@]}"
