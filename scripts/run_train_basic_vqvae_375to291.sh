#!/usr/bin/env bash
set -euo pipefail

# Usage and common recipes: NewCode/README.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/Utils/run_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${ROOT_DIR}/training/train_basic_vqvae_375to291.py"

DB="${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
SPLIT_MODE="${SPLIT_MODE:-auto}"
NORM_STATS="${NORM_STATS:-}"
INPUT_MODE="${INPUT_MODE:-traj_motion}"
NETWORK_ARCH="${NETWORK_ARCH:-skip_cat_mlp}"
ENCODER_ARCH="${ENCODER_ARCH:-${NETWORK_ARCH}}"
DECODER_ARCH="${DECODER_ARCH:-${NETWORK_ARCH}}"
ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
DECODER_LAYERS="${DECODER_LAYERS:-6}"
if [[ "${ENCODER_ARCH}" == "${DECODER_ARCH}" ]]; then
  ARCH_TAG="${ENCODER_ARCH}"
else
  ARCH_TAG="${ENCODER_ARCH}_to_${DECODER_ARCH}"
fi
OUT="${OUT:-${ROOT_DIR}/checkpoints/vqvae_${INPUT_MODE}_${ARCH_TAG}_375to291_32_4_2048_nomirror}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEQ_LEN="${SEQ_LEN:-2}"
STRIDE="${STRIDE:-1}"
# fp32 by default: measured as fast as bf16 at this model size, and bf16 changes the
# codebook assignment (see DATA_PIPELINE.md).
AMP_DTYPE="${AMP_DTYPE:-fp32}"
TF32="${TF32:-0}"
MATMUL_PRECISION="${MATMUL_PRECISION:-highest}"
LOAD_INTO_MEMORY="${LOAD_INTO_MEMORY:-0}"
# torch.compile currently crashes in this environment (inductor "duplicate template name"),
# and measured slower than eager here anyway.
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-max-autotune}"
# Paper MotionVQ weights.
RECON_WEIGHT="${RECON_WEIGHT:-3.0}"
VQ_WEIGHT="${VQ_WEIGHT:-1.0}"
VEL_WEIGHT="${VEL_WEIGHT:-15.0}"
# Second-order (acceleration) term, requires SEQ_LEN >= 3. 0 = paper recipe.
ACC_WEIGHT="${ACC_WEIGHT:-0.0}"
LATENT_L1_WEIGHT="${LATENT_L1_WEIGHT:-0.01}"
LATENT_VEL_WEIGHT="${LATENT_VEL_WEIGHT:-0.1}"
# Extra L1 on the 3 root-motion channels. Paper has no such term; 3.0 is the current setting.
ROOT_WEIGHT="${ROOT_WEIGHT:-3.0}"
# Continuous/discrete code alignment: the decoder must reproduce the *quantized* path from
# the continuous latent, so the two code paths stay interchangeable. 0 restores the paper recipe.
CODE_ALIGN_WEIGHT="${CODE_ALIGN_WEIGHT:-2.0}"
# Code-stability auxiliaries (2026-10-14): keep encoder outputs off the Voronoi boundary
# (margin hinge) and pull the re-encoded decoder output back to its code (roundtrip). The
# measured failure mode is boundary survival, so these two are the direct levers.
LATENT_MARGIN_WEIGHT="${LATENT_MARGIN_WEIGHT:-0.0}"
LATENT_MARGIN_TARGET="${LATENT_MARGIN_TARGET:-0.05}"
LATENT_MARGIN_MODE="${LATENT_MARGIN_MODE:-nearest_boundary}"
ROUNDTRIP_WEIGHT="${ROUNDTRIP_WEIGHT:-0.0}"
# Warm start (encoder/decoder only); codebook re-initialises when the size changes.
INIT_CKPT="${INIT_CKPT:-}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-2048}"
# Decoder-only calibration: freeze encoder+quantizer (EMA included), train only the decoder.
# TEACHER_CKPT + TEACHER_ALIGN_WEIGHT add the frozen continuous-teacher alignment
# ||D(q) - sg(D0(E0(x)))||^2 (see docs/VQ_DECODE_CONTINUITY_PLAN.md).
DECODER_ONLY="${DECODER_ONLY:-0}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
TEACHER_ALIGN_WEIGHT="${TEACHER_ALIGN_WEIGHT:-0.0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
# Primary metric: reported in the logs and stored as best_metric. It is always part of the
# gate below, but never decides alone.
EVAL_METRIC="${EVAL_METRIC:-mpjpe}"
# best.pt is only replaced when ALL of these beat their value at the current best.pt, so
# saved checkpoints are better across the board rather than better on one metric alone.
# 'none' falls back to the plain single --eval-metric rule.
BEST_METRICS="${BEST_METRICS:-mpjpe,jitter,foot_sliding,root_ade}"
# Required improvement per metric, in cm. 0 = strict; negative tolerates that much regression.
BEST_MIN_DELTA="${BEST_MIN_DELTA:-0.0}"
EVAL_CLIPS="${EVAL_CLIPS:-90}"
EVAL_WINDOW_FRAMES="${EVAL_WINDOW_FRAMES:-60}"
USE_ROTATION_TRICK="${USE_ROTATION_TRICK:-1}"
KMEANS_INIT="${KMEANS_INIT:-1}"
KMEANS_ITERS="${KMEANS_ITERS:-10}"
# EMA assignments per batch below which a code is replaced (library quantizer only).
# NOT a step count: with batch 512 x 32 heads over 2048 codes the per-code mean is ~8.
DEAD_CODE_THRESHOLD="${DEAD_CODE_THRESHOLD:-2}"

ru_require_file "${TRAIN_PY}" "train script"
ru_require_python_modules "${PYTHON_BIN}" numpy torch

cmd=(
  "${PYTHON_BIN}" "${TRAIN_PY}"
  --db "${DB}"
  --out-dir "${OUT}"
  --device "${DEVICE}"
  --split-mode "${SPLIT_MODE}"
  --input-mode "${INPUT_MODE}"
  --encoder-arch "${ENCODER_ARCH}"
  --decoder-arch "${DECODER_ARCH}"
  --encoder-layers "${ENCODER_LAYERS}"
  --decoder-layers "${DECODER_LAYERS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --stride "${STRIDE}"
  --amp-dtype "${AMP_DTYPE}"
  --matmul-precision "${MATMUL_PRECISION}"
  --compile-mode "${COMPILE_MODE}"
  --recon-weight "${RECON_WEIGHT}"
  --vq-weight "${VQ_WEIGHT}"
  --vel-weight "${VEL_WEIGHT}"
  --acc-weight "${ACC_WEIGHT}"
  --latent-l1-weight "${LATENT_L1_WEIGHT}"
  --latent-vel-weight "${LATENT_VEL_WEIGHT}"
  --root-weight "${ROOT_WEIGHT}"
  --code-align-weight "${CODE_ALIGN_WEIGHT}"
  --latent-margin-weight "${LATENT_MARGIN_WEIGHT}"
  --latent-margin-target "${LATENT_MARGIN_TARGET}"
  --latent-margin-mode "${LATENT_MARGIN_MODE}"
  --roundtrip-weight "${ROUNDTRIP_WEIGHT}"
  --num-embeddings "${NUM_EMBEDDINGS}"
  --teacher-align-weight "${TEACHER_ALIGN_WEIGHT}"
  --eval-every "${EVAL_EVERY}"
  --eval-metric "${EVAL_METRIC}"
  --best-metrics "${BEST_METRICS}"
  --best-min-delta "${BEST_MIN_DELTA}"
  --eval-clips "${EVAL_CLIPS}"
  --eval-window-frames "${EVAL_WINDOW_FRAMES}"
  --dead-code-threshold "${DEAD_CODE_THRESHOLD}"
  --kmeans-iters "${KMEANS_ITERS}"
)

ru_append_bool_flag cmd "${TF32}" --tf32 --no-tf32
ru_append_if_enabled cmd "${LOAD_INTO_MEMORY}" --load-into-memory
# Path-valued options must use an explicit non-empty test: ru_append_if_enabled treats the
# value as a boolean literal ("1"/"true"/...), so a path was silently dropped before.
if [[ -n "${NORM_STATS}" ]]; then cmd+=(--norm-stats "${NORM_STATS}"); fi
if [[ -n "${INIT_CKPT}" ]]; then cmd+=(--init-ckpt "${INIT_CKPT}"); fi
if [[ -n "${TEACHER_CKPT}" ]]; then cmd+=(--teacher-ckpt "${TEACHER_CKPT}"); fi
ru_append_bool_flag cmd "${DECODER_ONLY}" --decoder-only --no-decoder-only
ru_append_bool_flag cmd "${COMPILE}" --compile --no-compile
ru_append_bool_flag cmd "${USE_ROTATION_TRICK}" --use-rotation-trick --no-use-rotation-trick
ru_append_bool_flag cmd "${KMEANS_INIT}" --kmeans-init --no-kmeans-init

cmd+=("$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[Dry run]'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi
ru_run "${cmd[@]}"
