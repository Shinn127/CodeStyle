#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... OUT=... DEVICE=cuda NUM_FRAMES=600 CURVE_INTERVAL=50 FULL_TEST_SET=1 \
#   ./NewCode/scripts/run_evaluate_rollout_mpjpe_curve.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PY="${ROOT_DIR}/evaluation/evaluate_rollout_mpjpe_curve.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
OUT="${OUT:-${ROOT_DIR}/eval_results/rollout_mpjpe_curve}"
DEVICE="${DEVICE:-cuda}"
NUM_FRAMES="${NUM_FRAMES:-600}"
CURVE_INTERVAL="${CURVE_INTERVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
FULL_TEST_SET="${FULL_TEST_SET:-1}"
MPJPE_SCALE="${MPJPE_SCALE:-0.1}"
SEED="${SEED:-1234}"

CAMDM_CKPT="${CAMDM_CKPT:-}"
MANN_CKPT="${MANN_CKPT:-}"
FLOWMATCH_VELOCITY_CKPT="${FLOWMATCH_VELOCITY_CKPT:-}"
FLOWMATCH_LMM_VELOCITY_CKPT="${FLOWMATCH_LMM_VELOCITY_CKPT:-}"
FLOW_VQVAE_CKPT="${FLOW_VQVAE_CKPT:-}"
FLOW_LMM_CKPT="${FLOW_LMM_CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
SOURCE_STYLE="${SOURCE_STYLE:-}"
TARGET_STYLE="${TARGET_STYLE:-}"

ru_require_file "${EVAL_PY}" "evaluation script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch matplotlib scipy
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

cmd=(
  "${PYTHON_BIN}" "${EVAL_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --num-frames "${NUM_FRAMES}"
  --curve-interval "${CURVE_INTERVAL}"
  --mpjpe-scale "${MPJPE_SCALE}"
  --seed "${SEED}"
)
if ! ru_is_enabled "${FULL_TEST_SET}"; then
  cmd+=(--num-samples "${NUM_SAMPLES}")
fi

ru_append_if_nonempty cmd --camdm-ckpt "${CAMDM_CKPT}"
ru_append_if_nonempty cmd --mann-ckpt "${MANN_CKPT}"
ru_append_if_nonempty cmd --flowmatch-velocity-ckpt "${FLOWMATCH_VELOCITY_CKPT}"
ru_append_if_nonempty cmd --flowmatch-lmm-velocity-ckpt "${FLOWMATCH_LMM_VELOCITY_CKPT}"
ru_append_if_nonempty cmd --flow-vqvae-ckpt "${FLOW_VQVAE_CKPT}"
ru_append_if_nonempty cmd --flow-lmm-ckpt "${FLOW_LMM_CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --source-style "${SOURCE_STYLE}"
ru_append_if_nonempty cmd --target-style "${TARGET_STYLE}"
ru_append_if_enabled cmd "${FULL_TEST_SET}" --use-test-split

print_info "db" "${DB}"
print_info "out dir" "${OUT}"
print_info "device" "${DEVICE}"
print_info "num frames" "${NUM_FRAMES}"
print_info "curve interval" "${CURVE_INTERVAL}"
print_info "mpjpe scale" "${MPJPE_SCALE}"
print_info "full test set" "${FULL_TEST_SET}"
if ! ru_is_enabled "${FULL_TEST_SET}"; then
  print_info "num samples" "${NUM_SAMPLES}"
fi
print_info "seed" "${SEED}"

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
