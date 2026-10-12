#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for flow-matching controller training.
#
# Common overrides:
#   DB=... VQ_CKPT=... OUT=...
#   PREDICTION_MODE=velocity|target
#   VQ_TARGET_CACHE=... NORM_STATS=... LATENT_STATS=...
#   STYLE_CODE_PRIOR=... USE_STYLE_CODE_SELECTOR=1 STYLE_NUM_SLOTS=8
#   RECON_WEIGHT=0 (default; optional reconstruction requires >0)
#   DEVICE=auto|cuda|cpu
#   ./NewCode/scripts/run_train_flowmatch_vqvae_controller.sh --epochs 100 --lr 1e-4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_flowmatch_vqvae_controller.py"
PRECOMPUTE_PY="${ROOT_DIR}/tools/precompute_vq_targets.py"
PRECOMPUTE_STYLE_PRIOR_PY="${ROOT_DIR}/tools/precompute_style_code_prior.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
VQ_CKPT="${VQ_CKPT:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/flowmatch_tf50_fp32_norecon}"
DEVICE="${DEVICE:-}"
PREDICTION_MODE="${PREDICTION_MODE:-target}"
VQ_TARGET_CACHE="${VQ_TARGET_CACHE:-}"
NORM_STATS="${NORM_STATS:-}"
LATENT_STATS="${LATENT_STATS:-}"
STYLE_CODE_PRIOR="${STYLE_CODE_PRIOR:-}"
STYLE_NUM_SLOTS="${STYLE_NUM_SLOTS:-8}"
# dense | js | sparse_eps | legacy (see Utils/style_code_prior.py)
STYLE_PRIOR_LOSS="${STYLE_PRIOR_LOSS:-dense}"
STYLE_HIST_WEIGHT="${STYLE_HIST_WEIGHT:-}"
STYLE_ENTROPY_WEIGHT="${STYLE_ENTROPY_WEIGHT:-}"
RECON_WEIGHT="${RECON_WEIGHT:-0}"
RESUME="${RESUME:-}"
USE_VQ_TARGET_CACHE="${USE_VQ_TARGET_CACHE:-auto}"
USE_STYLE_CODE_SELECTOR="${USE_STYLE_CODE_SELECTOR:-1}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_file "${VQ_CKPT}" "VQ checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -d "${DB}" ]] || ru_die "db directory not found: ${DB}"

# The unmirrored 150-epoch VQ uses the short artifact names shared by every entry.
NOMIRROR_VQ_DIR_NAME="vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"

default_vq_target_cache() {
  local vqvae_ckpt="$1"
  local ckpt_dir_name
  ckpt_dir_name="$(basename "$(dirname "${vqvae_ckpt}")")"
  if [[ "${ckpt_dir_name}" == "${NOMIRROR_VQ_DIR_NAME}" ]]; then
    printf '%s' "${ROOT_DIR}/cache/vq_targets_vq150ep_nomirror_repair_v2"
  else
    printf '%s' "${ROOT_DIR}/cache/vq_targets_${ckpt_dir_name}_repair_v2"
  fi
}

default_style_code_prior() {
  local vqvae_ckpt="$1"
  local ckpt_dir_name
  ckpt_dir_name="$(basename "$(dirname "${vqvae_ckpt}")")"
  if [[ "${ckpt_dir_name}" == "${NOMIRROR_VQ_DIR_NAME}" ]]; then
    printf '%s' "${ROOT_DIR}/cache/style_code_prior_vq150ep_nomirror_repair_v2.npz"
  else
    printf '%s' "${ROOT_DIR}/cache/style_code_prior_${ckpt_dir_name}_repair_v2.npz"
  fi
}

looks_like_vq_target_cache() {
  local cache_dir="$1"
  [[ -d "${cache_dir}" ]] || return 1
  [[ -f "${cache_dir}/meta.json" ]] || return 1
  [[ -f "${cache_dir}/indices.u16" || -f "${cache_dir}/indices.i32" ]] || return 1
}

if [[ -z "${VQ_TARGET_CACHE}" && "${USE_VQ_TARGET_CACHE,,}" != "0" && "${USE_VQ_TARGET_CACHE,,}" != "false" ]]; then
  VQ_TARGET_CACHE="$(default_vq_target_cache "${VQ_CKPT}")"
fi
if ru_is_enabled "${USE_STYLE_CODE_SELECTOR}" && [[ -z "${STYLE_CODE_PRIOR}" ]]; then
  STYLE_CODE_PRIOR="$(default_style_code_prior "${VQ_CKPT}")"
fi

if [[ -n "${VQ_TARGET_CACHE}" ]]; then
  if looks_like_vq_target_cache "${VQ_TARGET_CACHE}"; then
    print_info "vq target cache" "${VQ_TARGET_CACHE} (reuse existing)"
  else
    ru_require_file "${PRECOMPUTE_PY}" "precompute script"
    print_info "vq target cache" "${VQ_TARGET_CACHE} (missing, precomputing)"
    precompute_cmd=(
      "${PYTHON_BIN}" "${PRECOMPUTE_PY}"
      --db "${DB}"
      --vqvae-ckpt "${VQ_CKPT}"
      --out-dir "${VQ_TARGET_CACHE}"
    )
    ru_append_if_nonempty precompute_cmd --norm-stats "${NORM_STATS}"
    ru_append_if_nonempty precompute_cmd --device "${DEVICE}"
    printf '[Run]'
    printf ' %q' "${precompute_cmd[@]}"
    printf '\n'
    "${precompute_cmd[@]}"
  fi
fi

if ru_is_enabled "${USE_STYLE_CODE_SELECTOR}"; then
  [[ -n "${VQ_TARGET_CACHE}" ]] || ru_die "style selector training requires VQ_TARGET_CACHE to be enabled"
  if [[ -f "${STYLE_CODE_PRIOR}" ]]; then
    print_info "style code prior" "${STYLE_CODE_PRIOR} (reuse existing)"
  else
    ru_require_file "${PRECOMPUTE_STYLE_PRIOR_PY}" "style prior precompute script"
    print_info "style code prior" "${STYLE_CODE_PRIOR} (missing, precomputing)"
    precompute_style_cmd=(
      "${PYTHON_BIN}" "${PRECOMPUTE_STYLE_PRIOR_PY}"
      --db "${DB}"
      --vq-target-cache "${VQ_TARGET_CACHE}"
      --out "${STYLE_CODE_PRIOR}"
      --style-num-slots "${STYLE_NUM_SLOTS}"
    )
    printf '[Run]'
    printf ' %q' "${precompute_style_cmd[@]}"
    printf '\n'
    "${precompute_style_cmd[@]}"
  fi
fi

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --vqvae-ckpt "${VQ_CKPT}"
  --out-dir "${OUT}"
  --style-prior-loss "${STYLE_PRIOR_LOSS}"
  --recon-weight "${RECON_WEIGHT}"
)

ru_append_if_nonempty cmd --style-hist-weight "${STYLE_HIST_WEIGHT}"
ru_append_if_nonempty cmd --style-entropy-weight "${STYLE_ENTROPY_WEIGHT}"

ru_append_if_nonempty cmd --device "${DEVICE}"
ru_append_if_nonempty cmd --prediction-mode "${PREDICTION_MODE}"
ru_append_if_nonempty cmd --vq-target-cache "${VQ_TARGET_CACHE}"
ru_append_if_nonempty cmd --norm-stats "${NORM_STATS}"
ru_append_if_nonempty cmd --latent-stats "${LATENT_STATS}"
if ! ru_is_enabled "${USE_STYLE_CODE_SELECTOR}"; then
  cmd+=(--no-use-style-code-selector)
fi
if ru_is_enabled "${USE_STYLE_CODE_SELECTOR}"; then
  ru_append_if_nonempty cmd --style-code-prior "${STYLE_CODE_PRIOR}"
  ru_append_if_nonempty cmd --style-num-slots "${STYLE_NUM_SLOTS}"
fi
ru_append_if_nonempty cmd --resume "${RESUME}"

print_info "db" "${DB}"
print_info "vq ckpt" "${VQ_CKPT}"
print_info "out dir" "${OUT}"
print_info "recon weight" "${RECON_WEIGHT}"
if [[ -n "${PREDICTION_MODE}" ]]; then
  print_info "prediction mode" "${PREDICTION_MODE}"
fi
if [[ -n "${VQ_TARGET_CACHE}" ]]; then
  print_info "vq target cache" "${VQ_TARGET_CACHE}"
fi
if [[ -n "${NORM_STATS}" ]]; then
  print_info "norm stats" "${NORM_STATS}"
fi
if [[ -n "${LATENT_STATS}" ]]; then
  print_info "latent stats" "${LATENT_STATS}"
fi
if ru_is_enabled "${USE_STYLE_CODE_SELECTOR}"; then
  print_info "style selector" "${USE_STYLE_CODE_SELECTOR}"
  print_info "style code prior" "${STYLE_CODE_PRIOR}"
  print_info "style num slots" "${STYLE_NUM_SLOTS}"
else
  print_info "style selector" "${USE_STYLE_CODE_SELECTOR}"
fi
if [[ -n "${RESUME}" ]]; then
  print_info "resume" "${RESUME}"
fi

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
