#!/usr/bin/env bash
set -euo pipefail

# Flow-matching controller realtime generation runner over the project base autoencoders.
#
# Key overrides:
#   DB=... FLOW_CKPT=... OUT=... PREFIX=... \
#   VQ_CKPT=... LMM_CKPT=... NORM_STATS=... \
#   NUM_FRAMES=300 MODE=rollout|reconstruct \
#   SEED=... SOURCE_STYLE= TARGET_STYLE= START_INDEX=... \
#   STYLE_RECIPE=... STYLE_STRENGTH=... \
#   SOLVER=euler ODE_STEPS=... DEVICE=cuda \
#   LATENT_CLIP=1 \
#   REQUANTIZE_Z=1 \
#   VISUALIZE=1 VIS_BACKEND=genoview|quick|matplotlib GIF_PATH=... \
#   BENCHMARK=1 \
#   ./NewCode/scripts/run_simulate_realtime_flowmatch_control.sh
#
# For less common options, pass raw flags directly:
#   ./NewCode/scripts/run_simulate_realtime_flowmatch_control.sh --fps 30 --local-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/evaluation/simulate_realtime_flowmatch_control.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/checkpoints/flowmatch_label_nomirror_150ep/best.pt}"
if [[ ! -f "${FLOW_CKPT}" ]]; then
  echo "Flow controller checkpoint not found: ${FLOW_CKPT}" >&2
  echo "Old controller checkpoints were removed by the 2026-10-12 cleanup; train one first" >&2
  echo "(e.g. bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh) and pass FLOW_CKPT=<out>/best.pt." >&2
  exit 2
fi
VQ_CKPT="${VQ_CKPT:-${VQVAE_CKPT:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt}}"
LMM_CKPT="${LMM_CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-${ROOT_DIR}/rollout_vis}"
PREFIX="${PREFIX:-}"

NUM_FRAMES="${NUM_FRAMES:-300}"
MODE="${MODE:-rollout}"
SEED="${SEED:-}"
SOURCE_STYLE="${SOURCE_STYLE:-}"
TARGET_STYLE="${TARGET_STYLE:-}"
START_INDEX="${START_INDEX:-}"
STYLE_RECIPE="${STYLE_RECIPE:-}"
STYLE_STRENGTH="${STYLE_STRENGTH:-}"

SOLVER="${SOLVER:-}"
ODE_STEPS="${ODE_STEPS:-}"
DEVICE="${DEVICE:-cuda}"
LATENT_CLIP="${LATENT_CLIP:-1}"
REQUANTIZE_Z="${REQUANTIZE_Z:-0}"
BENCHMARK="${BENCHMARK:-0}"

VISUALIZE="${VISUALIZE:-1}"
VIS_BACKEND="${VIS_BACKEND:-genoview}"
GIF_PATH="${GIF_PATH:-}"

ru_require_file "${RUN_PY}" "simulate script"
ru_require_file "${FLOW_CKPT}" "flow checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
if ru_is_enabled "${VISUALIZE}" && [[ "${VIS_BACKEND}" == "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" raylib
fi
if { ru_is_enabled "${VISUALIZE}" || [[ -n "${GIF_PATH}" ]]; } && [[ "${VIS_BACKEND}" != "genoview" ]]; then
  ru_require_python_modules "${PYTHON_BIN}" matplotlib
fi
[[ -e "${DB}" ]] || ru_die "db path not found: ${DB}"

if [[ -n "${VQ_CKPT}" ]]; then
  ru_require_file "${VQ_CKPT}" "VQ checkpoint"
fi
if [[ -n "${LMM_CKPT}" ]]; then
  ru_require_file "${LMM_CKPT}" "LMM checkpoint"
fi
if [[ -n "${NORM_STATS}" ]]; then
  ru_require_file "${NORM_STATS}" "norm stats"
fi
if [[ -n "${SOLVER}" ]]; then
  case "${SOLVER}" in
    euler)
      ;;
    *)
      ru_die "SOLVER must be: euler (got ${SOLVER})"
      ;;
  esac
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
  --flow-ckpt "${FLOW_CKPT}"
  --mode "${MODE}"
  --out-dir "${OUT}"
  --num-frames "${NUM_FRAMES}"
  --device "${DEVICE}"
  --vis-backend "${VIS_BACKEND}"
)

ru_append_if_nonempty cmd --prefix "${PREFIX}"
ru_append_if_nonempty cmd --seed "${SEED}"
ru_append_if_nonempty cmd --vqvae-ckpt "${VQ_CKPT}"
ru_append_if_nonempty cmd --lmm-ckpt "${LMM_CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --source-style "${SOURCE_STYLE}"
ru_append_if_nonempty cmd --target-style "${TARGET_STYLE}"
ru_append_if_nonempty cmd --style-recipe "${STYLE_RECIPE}"
ru_append_if_nonempty cmd --style-strength "${STYLE_STRENGTH}"
ru_append_if_nonempty cmd --start-index "${START_INDEX}"
ru_append_if_nonempty cmd --solver "${SOLVER}"
ru_append_if_nonempty cmd --ode-steps "${ODE_STEPS}"
ru_append_if_nonempty cmd --gif-path "${GIF_PATH}"
if ! ru_is_enabled "${LATENT_CLIP}"; then
  cmd+=(--no-latent-clip)
fi
ru_append_if_enabled cmd "${REQUANTIZE_Z}" --requantize-z
ru_append_if_enabled cmd "${BENCHMARK}" --benchmark
if ! ru_is_enabled "${VISUALIZE}"; then
  cmd+=(--no-visualize)
fi

print_info "db" "${DB}"
print_info "flow ckpt" "${FLOW_CKPT}"
if [[ -n "${VQ_CKPT}" ]]; then
  print_info "vq ckpt" "${VQ_CKPT}"
fi
if [[ -n "${LMM_CKPT}" ]]; then
  print_info "lmm ckpt" "${LMM_CKPT}"
fi
if [[ -z "${VQ_CKPT}" && -z "${LMM_CKPT}" ]]; then
  print_info "base ckpt" "infer from flow checkpoint args"
fi
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
else
  print_info "norm stats" "infer from flow checkpoint / VQ model"
fi
print_info "mode" "${MODE}"
print_info "out dir" "${OUT}"
if [[ -n "${PREFIX}" ]]; then
  print_info "prefix" "${PREFIX}"
else
  print_info "prefix" "auto from mode/num-frames"
fi
print_info "visualize" "${VISUALIZE}"
print_info "vis backend" "${VIS_BACKEND}"
if [[ -n "${GIF_PATH}" ]]; then
  print_info "gif path" "${GIF_PATH}"
fi
print_info "vis mode" "left=reference | right=prediction"
print_info "requantize z" "${REQUANTIZE_Z}"
print_info "benchmark" "${BENCHMARK}"
if [[ -n "${SEED}" ]]; then
  print_info "seed" "${SEED}"
else
  print_info "seed" "random each run"
fi
if [[ -n "${STYLE_RECIPE}" ]]; then
  print_info "style recipe" "${STYLE_RECIPE}"
fi
if [[ -n "${STYLE_STRENGTH}" ]]; then
  print_info "style strength" "${STYLE_STRENGTH}"
fi
if [[ -n "${SOLVER}" ]]; then
  print_info "solver" "${SOLVER}"
else
  print_info "solver" "minimal default: euler"
fi
if [[ -n "${ODE_STEPS}" ]]; then
  print_info "ode steps" "${ODE_STEPS}"
else
  print_info "ode steps" "inherit from flow checkpoint"
fi
print_info "latent clip" "${LATENT_CLIP}"

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
