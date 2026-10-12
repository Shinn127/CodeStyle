#!/usr/bin/env bash
set -euo pipefail

# Top-level reconstruction/visualization runner for baseline models.
#
# Key overrides:
#   MODEL=basic_vqvae|lmm|mvae \
#   DB=... CKPT=... NORM_STATS=... OUT=... PREFIX=... \
#   NUM_FRAMES=300 STYLE=Zombie START_INDEX=1234 SEED=1234 DEVICE=cuda \
#   VISUALIZE=1 VIS_BACKEND=genoview|quick|matplotlib GENO_GAP=2.4 GIF_PATH=... \
#   ./NewCode/scripts/run_simulate_realtime_baselines.sh --gif-path NewCode/reconstruction_vis/test.gif \
#   ./NewCode/scripts/run_simulate_realtime_baselines.sh
#
# For less common options, pass raw flags directly:
#   ./NewCode/scripts/run_simulate_realtime_baselines.sh --fps 30 --local-only
#   MODEL=mvae \
#   ./NewCode/scripts/run_simulate_realtime_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/evaluation/simulate_realtime_baselines.py"

MODEL="${MODEL:-basic_vqvae}"
DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
CKPT="${CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-${ROOT_DIR}/reconstruction_vis}"
PREFIX="${PREFIX:-}"
NUM_FRAMES="${NUM_FRAMES:-300}"
STYLE="${STYLE:-}"
START_INDEX="${START_INDEX:-}"
SEED="${SEED:-3907}"
DEVICE="${DEVICE:-cuda}"
VISUALIZE="${VISUALIZE:-1}"
VIS_BACKEND="${VIS_BACKEND:-genoview}"
GENO_GAP="${GENO_GAP:-2.4}"
GIF_PATH="${GIF_PATH:-}"

ru_require_file "${RUN_PY}" "unified simulate script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

if [[ -n "${MODE:-}" && "${MODE}" != "reconstruct" ]]; then
  ru_die "Baseline runner is reconstruct-only; MODE=${MODE} is unsupported."
fi
for arg in "$@"; do
  case "${arg}" in
    --mode|--mode=*|--noise-std|--noise-std=*)
      ru_die "run_simulate_realtime_baselines.sh is reconstruct-only; do not pass ${arg}."
      ;;
  esac
done

# Prefer explicit CKPT; otherwise auto-pick a sensible default checkpoint.
if [[ -z "${CKPT}" ]]; then
  case "${MODEL}" in
    basic_vqvae)
      candidate_basic="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048/best.pt"
      if [[ -f "${candidate_basic}" ]]; then
        CKPT="${candidate_basic}"
      fi
      ;;
    lmm)
      candidate_lmm="${ROOT_DIR}/checkpoints/lmm_375to291/best.pt"
      legacy_lmm="${ROOT_DIR}/checkpoints/lmm_ae_291to84plus32/best.pt"
      if [[ -f "${candidate_lmm}" ]]; then
        CKPT="${candidate_lmm}"
      elif [[ -f "${legacy_lmm}" ]]; then
        CKPT="${legacy_lmm}"
      fi
      ;;
    mvae)
      candidate_mvae="${ROOT_DIR}/checkpoints/mvae_375to291/best.pt"
      if [[ -f "${candidate_mvae}" ]]; then
        CKPT="${candidate_mvae}"
      fi
      ;;
  esac
fi
if [[ -n "${CKPT}" ]]; then
  ru_require_file "${CKPT}" "checkpoint"
fi
if [[ -z "${NORM_STATS}" && -n "${CKPT}" ]]; then
  candidate_norm="${CKPT%/*}/norm_stats.npz"
  if [[ -f "${candidate_norm}" ]]; then
    NORM_STATS="${candidate_norm}"
  fi
fi
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

case "${MODEL}" in
  basic_vqvae|lmm|mvae)
    ;;
  *)
    ru_die "MODEL must be one of: basic_vqvae, lmm, mvae (got ${MODEL})"
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
if ru_is_enabled "${VISUALIZE}" && [[ "${VIS_BACKEND}" == "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" raylib
fi
if { ru_is_enabled "${VISUALIZE}" || [[ -n "${GIF_PATH}" ]]; } && [[ "${VIS_BACKEND}" != "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" matplotlib
fi

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --model "${MODEL}"
  --db "${DB}"
  --num-frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --out-dir "${OUT}"
  --vis-backend "${VIS_BACKEND}"
  --geno-gap "${GENO_GAP}"
)

ru_append_if_nonempty cmd --ckpt "${CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --prefix "${PREFIX}"
ru_append_if_nonempty cmd --style "${STYLE}"
ru_append_if_nonempty cmd --start-index "${START_INDEX}"
ru_append_if_nonempty cmd --gif-path "${GIF_PATH}"

if ! ru_is_enabled "${VISUALIZE}"; then
  cmd+=(--no-visualize)
fi

print_info "model" "${MODEL}"
print_info "task" "reconstruct"
print_info "db" "${DB}"
if [[ -n "${CKPT}" ]]; then
  print_info "ckpt" "${CKPT}"
else
  print_info "ckpt" "resolve in python default"
fi
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
else
  print_info "norm stats" "infer from checkpoint sibling"
fi
print_info "out dir" "${OUT}"
if [[ -n "${PREFIX}" ]]; then
  print_info "prefix" "${PREFIX}"
else
  print_info "prefix" "auto from model/reconstruct/num-frames"
fi
print_info "num frames" "${NUM_FRAMES}"
if [[ -n "${STYLE}" ]]; then
  print_info "style" "${STYLE}"
fi
if [[ -n "${START_INDEX}" ]]; then
  print_info "start index" "${START_INDEX}"
fi
print_info "seed" "${SEED}"
print_info "device" "${DEVICE}"
print_info "visualize" "${VISUALIZE}"
print_info "vis backend" "${VIS_BACKEND}"
if [[ -n "${GIF_PATH}" ]]; then
  print_info "gif path" "${GIF_PATH}"
fi

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
