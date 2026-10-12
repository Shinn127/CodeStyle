#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PRACTICAL_HISTORY=1
export RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/checkpoints/clip_recipe_gt_history50}"
# Complete the bounded 50-epoch validation run; optional early stop stays explicit.
export STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-0}"
exec bash "${SCRIPT_DIR}/run_train_clip_recipe_revision.sh" "$@"
