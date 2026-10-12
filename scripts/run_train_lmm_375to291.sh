#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... OUT=... DEVICE=cuda AMP_DTYPE=bf16 \
#   LOAD_INTO_MEMORY=1 \
#   ./NewCode/scripts/run_train_lmm_375to291.sh
#
# For less common options, pass raw trainer flags directly:
#   ./NewCode/scripts/run_train_lmm_375to291.sh --epochs 80 --compile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_lmm_375to291.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/lmm_375to291}"
DEVICE="${DEVICE:-cuda}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --amp-dtype "${AMP_DTYPE}"
)

ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory

cmd+=("$@")
"${cmd[@]}"
