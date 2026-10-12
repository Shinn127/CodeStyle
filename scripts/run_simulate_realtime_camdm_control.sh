#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... CAMDM_CKPT=... OUT=... PREFIX=... \
#   NORM_STATS=... NUM_FRAMES=300 MODE=rollout|reconstruct \
#   SEED=1234 SOURCE_STYLE= TARGET_STYLE= START_INDEX=... \
#   DEVICE=cuda BENCHMARK=1 \
#   VISUALIZE=1 VIS_BACKEND=genoview|quick|matplotlib GIF_PATH=... \
#   ./NewCode/scripts/run_simulate_realtime_camdm_control.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/evaluation/simulate_realtime_camdm_control.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
CAMDM_CKPT="${CAMDM_CKPT:-${ROOT_DIR}/checkpoints/camdm_controller/best.pt}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-${ROOT_DIR}/rollout_vis}"
PREFIX="${PREFIX:-}"

NUM_FRAMES="${NUM_FRAMES:-300}"
MODE="${MODE:-rollout}"
SEED="${SEED:-1234}"
SOURCE_STYLE="${SOURCE_STYLE:-}"
TARGET_STYLE="${TARGET_STYLE:-}"
START_INDEX="${START_INDEX:-}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
DEVICE="${DEVICE:-cuda}"
BENCHMARK="${BENCHMARK:-0}"

VISUALIZE="${VISUALIZE:-1}"
VIS_BACKEND="${VIS_BACKEND:-genoview}"
GIF_PATH="${GIF_PATH:-}"

ru_require_file "${RUN_PY}" "simulate script"
ru_require_file "${CAMDM_CKPT}" "CAMDM checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
if ru_is_enabled "${VISUALIZE}" && [[ "${VIS_BACKEND}" == "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" raylib
fi
if { ru_is_enabled "${VISUALIZE}" || [[ -n "${GIF_PATH}" ]]; } && [[ "${VIS_BACKEND}" != "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" matplotlib
fi
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

if [[ -z "${NORM_STATS}" ]]; then
  candidate_norm="${CAMDM_CKPT%/*}/norm_stats.npz"
  if [[ -f "${candidate_norm}" ]]; then
    NORM_STATS="${candidate_norm}"
  fi
fi
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi

case "${MODE}" in
  reconstruct|rollout)
    ;;
  *)
    ru_die "MODE must be one of: reconstruct, rollout (got ${MODE})"
    ;;
esac
case "${VIS_BACKEND}" in
  genoview|quick|matplotlib)
    ;;
  *)
    ru_die "VIS_BACKEND must be one of: genoview, quick, matplotlib (got ${VIS_BACKEND})"
    ;;
esac
if [[ -n "${GIF_PATH}" && "${VIS_BACKEND}" == "genoview" ]]; then
  ru_die "GIF_PATH is only supported when VIS_BACKEND is quick or matplotlib."
fi

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --camdm-ckpt "${CAMDM_CKPT}"
  --mode "${MODE}"
  --out-dir "${OUT}"
  --num-frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --vis-backend "${VIS_BACKEND}"
)

ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --prefix "${PREFIX}"
ru_append_if_nonempty cmd --source-style "${SOURCE_STYLE}"
ru_append_if_nonempty cmd --target-style "${TARGET_STYLE}"
ru_append_if_nonempty cmd --start-index "${START_INDEX}"
ru_append_if_nonempty cmd --sampling-steps "${SAMPLING_STEPS}"
ru_append_if_nonempty cmd --gif-path "${GIF_PATH}"
ru_append_if_enabled cmd "${BENCHMARK}" --benchmark
if ! ru_is_enabled "${VISUALIZE}"; then
  cmd+=(--no-visualize)
fi

print_info "db" "${DB}"
print_info "camdm ckpt" "${CAMDM_CKPT}"
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
else
  print_info "norm stats" "infer from checkpoint sibling"
fi
print_info "mode" "${MODE}"
print_info "out dir" "${OUT}"
print_info "seed" "${SEED}"
print_info "device" "${DEVICE}"
print_info "visualize" "${VISUALIZE}"
print_info "vis backend" "${VIS_BACKEND}"
print_info "benchmark" "${BENCHMARK}"
if [[ -n "${SAMPLING_STEPS}" ]]; then
  print_info "sampling steps" "${SAMPLING_STEPS}"
fi
if [[ -n "${GIF_PATH}" ]]; then
  print_info "gif path" "${GIF_PATH}"
fi

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
