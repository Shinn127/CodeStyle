#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... FLOW_CKPT=... CLASSIFIER_CKPT=... OUT=... DEVICE=cuda \
#   ./NewCode/scripts/run_evaluate_minor_revision_generalization.sh
#
# Example:
#   ./NewCode/scripts/run_evaluate_minor_revision_generalization.sh \
#     --horizons 240 --flow-eval-mode rollout \
#     --num-sources 1000 \
#     --targets-per-source 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PY="${ROOT_DIR}/evaluation/evaluate_minor_revision_generalization.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/checkpoints/clip_recipe_v2/controller/best.pt}"
# Native-60fps, 240-frame unmirrored classifier. Train it before using this entry.
# CLASSIFIER_CKPT can select other models; horizons then follow their metadata.
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${ROOT_DIR}/checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth}"
OUT="${OUT:-${ROOT_DIR}/eval_results/minor_revision_generalization_60fps_240f}"
DEVICE="${DEVICE:-cuda}"
FLOW_EVAL_MODE="${FLOW_EVAL_MODE:-rollout}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"

ru_require_file "${EVAL_PY}" "minor revision evaluation script"
ru_require_file "${FLOW_CKPT}" "FlowMatch checkpoint"
ru_require_file "${CLASSIFIER_CKPT}" "style classifier checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch tqdm

cmd=(
  "${PYTHON_BIN}" "${EVAL_PY}"
  --db "${DB}"
  --flow-ckpt "${FLOW_CKPT}"
  --classifier-ckpt "${CLASSIFIER_CKPT}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --flow-eval-mode "${FLOW_EVAL_MODE}"
)

ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory

cmd+=("$@")
"${cmd[@]}"
