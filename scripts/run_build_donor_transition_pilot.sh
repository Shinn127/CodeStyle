#!/usr/bin/env bash
set -euo pipefail

# Bounded entry point for the offline donor-transition pilot (see
# docs/DONOR_AGENT_EXECUTION_GUIDE.md).  Runs from any cwd.
#
#   scripts/run_build_donor_transition_pilot.sh --dry-run
#   scripts/run_build_donor_transition_pilot.sh                 # full pilot (defaults)
#   MAX_PAIRS=4 MAX_ACCEPTED=4 MAX_VIDEOS=2 \
#     scripts/run_build_donor_transition_pilot.sh               # bounded validation run
#
# OUT=<dir> selects the run directory; an existing non-empty directory is refused.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/tools/build_donor_transition_dataset.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
VQ_CKPT="${VQ_CKPT:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt}"
NORM_STATS="${NORM_STATS:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
MAX_PAIRS="${MAX_PAIRS:-128}"
MAX_ACCEPTED="${MAX_ACCEPTED:-32}"
MAX_VIDEOS="${MAX_VIDEOS:-12}"
OUT="${OUT:-}"
CHUNK_ROWS="${CHUNK_ROWS:-2048}"
MAX_GPU_SECONDS="${MAX_GPU_SECONDS:-120}"
MAX_TOTAL_SECONDS="${MAX_TOTAL_SECONDS:-1800}"

ru_require_file "${RUN_PY}" "pilot builder"
ru_require_python_modules "${PYTHON_BIN}" numpy torch scipy matplotlib

if [[ -z "${OUT}" ]]; then
  RUN_ID="$(date +%Y%m%d-%H%M%S)_seed${SEED}"
  OUT="${ROOT_DIR}/analysis/donor_transition_pilot_v1/${RUN_ID}"
fi

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --vq-ckpt "${VQ_CKPT}"
  --norm-stats "${NORM_STATS}"
  --out "${OUT}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --max-pairs "${MAX_PAIRS}"
  --max-accepted "${MAX_ACCEPTED}"
  --max-videos "${MAX_VIDEOS}"
  --chunk-rows "${CHUNK_ROWS}"
  --max-gpu-seconds "${MAX_GPU_SECONDS}"
  --max-total-seconds "${MAX_TOTAL_SECONDS}"
)

cmd+=("$@")
ru_run "${cmd[@]}"
