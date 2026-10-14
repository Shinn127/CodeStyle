#!/usr/bin/env bash
set -euo pipefail

# C2a: pure-TF observation-noise robustness A/B (docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md §4 C2).
#
#   ARM=C2-A025|C2-A050|C2-A075 bash scripts/run_train_c2_obsnoise_ab.sh
#
# Three arms differ ONLY in --observation-noise-std. All start from the same trained 100ep
# controller weights (--init-ckpt), pure teacher forcing (p=0, no rollout training), 10 epochs.
# DRY_RUN=1 prints the expanded command. Training requires explicit user authorization.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARM="${ARM:-}"
case "${ARM}" in
  C2-A025) A="0.25";;
  C2-A050) A="0.50";;
  C2-A075) A="0.75";;
  *) echo "[Error] set ARM=C2-A025|C2-A050|C2-A075" >&2; exit 2;;
esac

SRC_RUN="${ROOT_DIR}/checkpoints/flowmatch_label_history_obsnoise_tf100"
INIT_CKPT="${INIT_CKPT:-${SRC_RUN}/best.pt}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/c2_obsnoise_A${A//./}_10ep}"
LOG="${LOG:-${ROOT_DIR}/analysis/vq_controller_recovery/C2/arms/${ARM}.log}"
[[ -f "${INIT_CKPT}" ]] || { echo "[Error] missing init controller: ${INIT_CKPT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG}")"
echo "[C2a] arm=${ARM} obs_noise_std=${A} init=${INIT_CKPT} out=${OUT}"

# Frozen C2a protocol: the existing obs-noise augmentation (objective v12) is the only knob.
OBS_NOISE_STD="${A}" INIT_CKPT="${INIT_CKPT}" OUT="${OUT}" \
EPOCHS=10 BATCH_SIZE=512 LR=5e-4 FLOW_LAYERS=2 STYLE_FILM=0 DROPOUT=0 \
CODE_CE_W=0 CODE_CE_WARMUP=0 \
  bash "${ROOT_DIR}/scripts/run_train_flowmatch_history_obsnoise_tf100.sh" \
  --seq-len 6 --history-conditioning --train-feedback-context \
  --feedback-probability 0 --feedback-ramp-epochs 0 \
  --seed 42 --save-every 5 --val-every 1 \
  2>&1 | tee "${LOG}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}" "${PYTHON_BIN}" \
    "${ROOT_DIR}/analysis/vq_controller_recovery/C2/c2_finalize_manifest.py" \
    --run-dir "${OUT}" --arm "${ARM}" --obs-noise-std "${A}"
fi
