#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... LMM_CKPT=... OUT=...
#   NORM_STATS=... LATENT_STATS=...
#   ./NewCode/scripts/run_train_diffusion_lmm_controller.sh --epochs 100 --lr 1e-4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_diffusion_lmm_controller.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
LMM_CKPT="${LMM_CKPT:-${ROOT_DIR}/checkpoints/lmm_375to291_head10/best.pt}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/diffusion_lmm_controller}"
NORM_STATS="${NORM_STATS:-}"
LATENT_STATS="${LATENT_STATS:-}"
DEVICE="${DEVICE:-cuda}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_file "${LMM_CKPT}" "LMM checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi
if [[ -n "${LATENT_STATS}" ]]; then
  ru_require_file "${LATENT_STATS}" "latent stats"
fi

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --lmm-ckpt "${LMM_CKPT}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
)

ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --latent-stats "${LATENT_STATS}"

print_info "db" "${DB}"
print_info "lmm ckpt" "${LMM_CKPT}"
print_info "out dir" "${OUT}"

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
