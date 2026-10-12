#!/usr/bin/env bash
set -euo pipefail

# Zero-shot exemplar -> style recipe helper for VQ+FlowMatch.
#
# Common overrides:
#   STYLE=Zombie \
#   FLOW_CKPT=... DB=... OUT=... \
#   NUM_FRAMES=240 PICK=longest|first|random SEGMENT_INDEX=... \
#   TOPK=4 STRENGTH=1.0 BASE_STYLE=ArmsFolded DEVICE=cuda \
#   ./NewCode/scripts/run_build_zeroshot_style_recipe.sh
#
# Less common options can be passed through directly:
#   ./NewCode/scripts/run_build_zeroshot_style_recipe.sh --frame-stride 2 --crop random

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_PY="${ROOT_DIR}/tools/build_zeroshot_style_recipe.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_zeroshot}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/checkpoints/clip_recipe_v2/controller/best.pt}"
FLOW_CONFIG="${FLOW_CONFIG:-}"
STYLE_CODE_PRIOR="${STYLE_CODE_PRIOR:-}"
VQ_CKPT="${VQ_CKPT:-}"
NORM_STATS="${NORM_STATS:-}"
OUT="${OUT:-}"

STYLE="${STYLE:-}"
SEGMENT_INDEX="${SEGMENT_INDEX:-}"
PICK="${PICK:-longest}"
NUM_FRAMES="${NUM_FRAMES:-240}"
CROP="${CROP:-center}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
TOPK="${TOPK:-4}"
STRENGTH="${STRENGTH:-1.0}"
BASE_STYLE="${BASE_STYLE:-}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

ru_require_file "${BUILD_PY}" "recipe builder"
ru_require_file "${FLOW_CKPT}" "flow checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -d "${DB}" ]] || ru_die "db directory not found: ${DB}"

case "${PICK}" in
  longest|first|random)
    ;;
  *)
    ru_die "PICK must be one of: longest, first, random (got ${PICK})"
    ;;
esac

case "${CROP}" in
  start|center|random)
    ;;
  *)
    ru_die "CROP must be one of: start, center, random (got ${CROP})"
    ;;
esac

cmd=(
  "${PYTHON_BIN}" "${BUILD_PY}"
  --db "${DB}"
  --flow-ckpt "${FLOW_CKPT}"
  --pick "${PICK}"
  --num-frames "${NUM_FRAMES}"
  --crop "${CROP}"
  --frame-stride "${FRAME_STRIDE}"
  --topk "${TOPK}"
  --strength "${STRENGTH}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

ru_append_if_nonempty cmd --flow-config "${FLOW_CONFIG}"
ru_append_if_nonempty cmd --style-code-prior "${STYLE_CODE_PRIOR}"
ru_append_if_nonempty cmd --vqvae-ckpt "${VQ_CKPT}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --out "${OUT}"
ru_append_if_nonempty cmd --style "${STYLE}"
ru_append_if_nonempty cmd --segment-index "${SEGMENT_INDEX}"
ru_append_if_nonempty cmd --base-style "${BASE_STYLE}"

print_info "db" "${DB}"
print_info "flow ckpt" "${FLOW_CKPT}"
if [[ -n "${FLOW_CONFIG}" ]]; then
  print_info "flow config" "${FLOW_CONFIG}"
fi
if [[ -n "${STYLE_CODE_PRIOR}" ]]; then
  print_info "style prior" "${STYLE_CODE_PRIOR}"
fi
if [[ -n "${STYLE}" ]]; then
  print_info "style" "${STYLE}"
else
  print_info "style" "all styles eligible"
fi
if [[ -n "${SEGMENT_INDEX}" ]]; then
  print_info "segment index" "${SEGMENT_INDEX}"
else
  print_info "pick" "${PICK}"
fi
print_info "num frames" "${NUM_FRAMES}"
print_info "crop" "${CROP}"
print_info "frame stride" "${FRAME_STRIDE}"
print_info "topk" "${TOPK}"
print_info "strength" "${STRENGTH}"
if [[ -n "${BASE_STYLE}" ]]; then
  print_info "base style" "${BASE_STYLE}"
fi
print_info "device" "${DEVICE}"
print_info "seed" "${SEED}"
if [[ -n "${OUT}" ]]; then
  print_info "out" "${OUT}"
else
  print_info "out" "auto under NewCode/cache/zeroshot_style_recipes"
fi

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
