#!/usr/bin/env bash
set -euo pipefail

# Usage and common recipes: NewCode/README.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${ROOT_DIR}/tools/precompute_vq_targets.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
INPUT_MODE="${INPUT_MODE:-traj_motion}"
NETWORK_ARCH="${NETWORK_ARCH:-skip_cat_mlp}"
VQ_CKPT="${VQ_CKPT:-${VQVAE_CKPT:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt}}"
NORM_STATS="${NORM_STATS:-}"
CACHE_NAME="vq_targets_$(basename "$(dirname "${VQ_CKPT}")")"
if [[ "$(dirname "${VQ_CKPT}")" == "${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror" ]]; then CACHE_NAME="vq_targets_vq150ep_nomirror_repair_v2"; fi
OUT="${OUT:-${ROOT_DIR}/cache/${CACHE_NAME}}"
DEVICE="${DEVICE:-cuda}"
CHUNK_ROWS="${CHUNK_ROWS:-8192}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PIN_MEMORY="${PIN_MEMORY:-1}"

ru_require_file "${RUN_PY}" "precompute script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

cmd=(
  "${PYTHON_BIN}" "${RUN_PY}"
  --db "${DB}"
  --input-mode "${INPUT_MODE}"
  --network-arch "${NETWORK_ARCH}"
  --vqvae-ckpt "${VQ_CKPT}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --chunk-rows "${CHUNK_ROWS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
)

ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_bool_flag cmd "${PIN_MEMORY}" --pin-memory --no-pin-memory

cmd+=("$@")
ru_run "${cmd[@]}"
