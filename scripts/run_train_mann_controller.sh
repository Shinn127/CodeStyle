#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... OUT=... DEVICE=cuda \
#   ./NewCode/scripts/run_train_mann_controller.sh --epochs 100 --lr 1e-4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_mann_controller.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/mann_controller}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
)

print_info "db" "${DB}"
print_info "out dir" "${OUT}"
print_info "device" "${DEVICE}"
print_info "epochs" "${EPOCHS}"
print_info "batch size" "${BATCH_SIZE}"

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
