#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/checkpoints/clip_recipe_v2}"
if [[ "${SMOKE:-0}" == 1 ]]; then RUN_ROOT="${RUN_ROOT}_smoke"; fi
if [[ $# != 0 || -n "${INIT_CKPT:-}" || -n "${INITIAL_CKPT:-}" ]]; then
  echo 'V2 trains from scratch. Configure RUN_ROOT, SMOKE, DRY_RUN or same-run RESUME.' >&2
  exit 2
fi
common=(--device "${DEVICE:-cuda}" --seed 42)
metric=("$PYTHON_BIN" "${ROOT_DIR}/training/train_motion_style_metric.py" "${common[@]}"
  --out-dir "$RUN_ROOT/style_metric" --updates 1000 --held-styles 10)
controller=("$PYTHON_BIN" "${ROOT_DIR}/training/train_clip_recipe_v2.py" "${common[@]}"
  --out-dir "$RUN_ROOT/controller" --style-metric-ckpt "$RUN_ROOT/style_metric/best.pt"
  --epochs 50 --flow-hidden-dim 256 --flow-layers 4 --flow-ffn-dim 1024
  --style-guidance-scale 1.5 --style-dropout .1 --microbatch "${MICROBATCH_SIZE:-512}"
  --workers "${NUM_WORKERS:-8}" --val-every 5 --val-batch-size "${STYLE_VAL_BATCH_SIZE:-64}"
  --diagnostic-epoch "${DIAGNOSTIC_EPOCH:-10}")
if [[ -n "${GPU_DATA_CACHE_GB:-}" ]]; then controller+=(--gpu-data-cache-gb "$GPU_DATA_CACHE_GB"); fi
if [[ -n "${CUDA_FIELD_GRAPH:-}" ]]; then
  case "$CUDA_FIELD_GRAPH" in
    1) controller+=(--cuda-field-graph) ;;
    0) controller+=(--no-cuda-field-graph) ;;
    *) echo 'CUDA_FIELD_GRAPH must be 0 or 1.' >&2; exit 2 ;;
  esac
fi
if [[ "${SMOKE:-0}" == 1 ]]; then
  metric+=(--updates 2 --val-every 2)
  controller+=(--epochs 1 --updates-per-epoch 2 --workers 0 --val-every 0
    --aux-warmup-epochs 0 --metric-min-retrieval 0)
fi
if [[ "${RESUME:-0}" == 1 ]]; then controller+=(--resume "$RUN_ROOT/controller/latest.pt"); fi
printf '[Metric]'; printf ' %q' "${metric[@]}"; printf '\n'
printf '[Controller]'; printf ' %q' "${controller[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
if [[ ! -f "$RUN_ROOT/style_metric/best.pt" ]]; then "${metric[@]}"; fi
"${controller[@]}"
