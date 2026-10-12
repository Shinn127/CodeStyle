#!/usr/bin/env bash
set -euo pipefail
# Seen (in-distribution) style rollout evaluation for a label controller.
# Unseen/OOD evaluation is out of scope for the label model: it has no condition channel
# for a style it never trained on, so the recipe path is not exercised here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/checkpoints/flowmatch_label_nomirror_150ep/best.pt}"
if [[ ! -f "${FLOW_CKPT}" ]]; then
  echo "Flow controller checkpoint not found: ${FLOW_CKPT}" >&2
  echo "Old controller checkpoints were removed by the 2026-10-12 cleanup; pass FLOW_CKPT=<trained>/best.pt." >&2
  exit 2
fi
DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/eval_results/label_nomirror_rollout}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${ROOT_DIR}/checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth}"
cmd=("${PYTHON_BIN}" "${ROOT_DIR}/evaluation/evaluate_minor_revision_generalization.py"
  --db "$DB" --flow-ckpt "$FLOW_CKPT" --classifier-ckpt "$CLASSIFIER_CKPT"
  --flow-eval-mode rollout --horizons 240 --source-split test
  --num-sources 1000 --targets-per-source 5 --seed 42 --flow-noise-std 1
  --flow-requantize-z --flow-latent-clip
  --out-dir "${OUT_ROOT}/seen" --device "${DEVICE:-cuda}")
printf '[Seen rollout]'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ -f "${OUT_ROOT}/seen/minor_revision_generalization_summary.json" ]]; then
  echo "Results already exist: ${OUT_ROOT}/seen; choose a new OUT_ROOT." >&2; exit 2
fi
"${cmd[@]}"
