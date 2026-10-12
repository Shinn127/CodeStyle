#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/checkpoints/flowmatch_label_nomirror_150ep/best.pt}"
cmd=("$PYTHON_BIN" "${ROOT_DIR}/Visualization/cli/interactive_label_control.py"
  --db "${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
  --flow-ckpt "${FLOW_CKPT}"
  --device "${DEVICE:-cuda}" --seed "${SEED:-42}")
# Only pass --ode-steps when explicitly requested; otherwise the session inherits
# solver_steps from the checkpoint (4 for the ODE4 runs, 8 for older checkpoints).
if [[ -n "${ODE_STEPS:-}" ]]; then cmd+=(--ode-steps "${ODE_STEPS}"); fi
if [[ -n "${STYLE:-}" ]]; then cmd+=(--style "$STYLE"); fi
if [[ -n "${VQ_CKPT:-}" ]]; then cmd+=(--vqvae-ckpt "$VQ_CKPT"); fi
if [[ -n "${NORM_STATS:-}" ]]; then cmd+=(--norm-stats "$NORM_STATS"); fi
if [[ -n "${NOISE_STD:-}" ]]; then cmd+=(--noise-std "$NOISE_STD"); fi
if [[ "${REQUANTIZE:-1}" == "0" ]]; then cmd+=(--no-requantize); fi
cmd+=("$@")
printf '[Run]'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ ! -f "${FLOW_CKPT}" ]]; then
  echo "Flow controller checkpoint not found: ${FLOW_CKPT}" >&2
  echo "Old controller checkpoints were removed by the 2026-10-12 cleanup; train one first" >&2
  echo "(e.g. bash scripts/run_train_flowmatch_history_tf10.sh) and pass FLOW_CKPT=<out>/best.pt." >&2
  exit 2
fi
exec "${cmd[@]}"
