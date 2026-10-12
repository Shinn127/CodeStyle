#!/usr/bin/env bash
set -euo pipefail

# Key overrides:
#   DB=... VQ_CKPT=... OUT=...
#   VQ_TARGET_CACHE=... NORM_STATS=... LATENT_STATS=...
#   STYLE_CODE_PRIOR=... USE_STYLE_CODE_SELECTOR=1 STYLE_NUM_SLOTS=8
#   DEVICE=auto|cuda|cpu
#   ./NewCode/scripts/run_train_diffusion_vqvae_controller.sh --epochs 100 --lr 1e-4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

print_info() {
  printf '[Info] %-15s: %s\n' "$1" "$2"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_diffusion_vqvae_controller.py"
PRECOMPUTE_PY="${ROOT_DIR}/tools/precompute_vq_targets.py"
PRECOMPUTE_STYLE_PRIOR_PY="${ROOT_DIR}/tools/precompute_style_code_prior.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
VQ_CKPT="${VQ_CKPT:-${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048/best.pt}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/diffusion_vqvae_controller}"
DEVICE="${DEVICE:-}"
VQ_TARGET_CACHE="${VQ_TARGET_CACHE:-}"
NORM_STATS="${NORM_STATS:-}"
LATENT_STATS="${LATENT_STATS:-}"
STYLE_CODE_PRIOR="${STYLE_CODE_PRIOR:-}"
STYLE_NUM_SLOTS="${STYLE_NUM_SLOTS:-8}"
RESUME="${RESUME:-}"
USE_VQ_TARGET_CACHE="${USE_VQ_TARGET_CACHE:-auto}"
USE_STYLE_CODE_SELECTOR="${USE_STYLE_CODE_SELECTOR:-1}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_file "${VQ_CKPT}" "VQ checkpoint"
ru_require_python_modules "${PYTHON_BIN}" numpy torch
[[ -d "${DB}" ]] || ru_die "db directory not found: ${DB}"

default_vq_target_cache() {
  local vqvae_ckpt="$1"
  local ckpt_dir_name
  ckpt_dir_name="$(basename "$(dirname "${vqvae_ckpt}")")"
  printf '%s' "${ROOT_DIR}/cache/vq_targets_${ckpt_dir_name}"
}

default_style_code_prior() {
  local vqvae_ckpt="$1"
  local ckpt_dir_name
  ckpt_dir_name="$(basename "$(dirname "${vqvae_ckpt}")")"
  printf '%s' "${ROOT_DIR}/cache/style_code_prior_${ckpt_dir_name}.npz"
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
)

ru_append_if_nonempty cmd --device "${DEVICE}"
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

cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
