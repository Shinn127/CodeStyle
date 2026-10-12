#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... OUT=... DEVICE=cuda \
#   ./NewCode/scripts/run_evaluate_motion_375to291.sh
#
# Optional:
#   ./NewCode/scripts/run_evaluate_motion_375to291.sh --models basic_vqvae lmm mvae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PY="${ROOT_DIR}/evaluation/evaluate_motion_375to291.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
OUT="${OUT:-${ROOT_DIR}/eval_results/motion_375to291}"
DEVICE="${DEVICE:-cuda}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"

ru_require_file "${EVAL_PY}" "evaluation script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch scipy tqdm

cmd=(
  "${PYTHON_BIN}" "${EVAL_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
)

ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory

cmd+=("$@")
"${cmd[@]}"
