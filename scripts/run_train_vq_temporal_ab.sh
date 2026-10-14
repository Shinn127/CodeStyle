#!/usr/bin/env bash
set -euo pipefail

# VQ temporal-continuity paired pilot (docs/VQ_SOURCE_AGENT_EXECUTION_PLAN.md S3).
#
#   ARM=B0 bash scripts/run_train_vq_temporal_ab.sh     # control: acc_weight=0
#   ARM=B1 bash scripts/run_train_vq_temporal_ab.sh     # candidate: acc_weight=4.9842 (S2)
#
# Both arms cold-start from the SAME ep90 VQ state (--init-ckpt is an init, not a resume),
# update encoder + EMA codebook + decoder, and share everything except ACC_WEIGHT.
# DRY_RUN=1 prints the expanded command without starting anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARM="${ARM:-}"
case "${ARM}" in
  B0) ACC_WEIGHT_PINNED="0" ;;
  B1) ACC_WEIGHT_PINNED="4.9842" ;;   # S2 legacy-geometry gap midpoint (see COMMON.md)
  *) echo "[Error] set ARM=B0 or ARM=B1" >&2; exit 2 ;;
esac
if [[ -n "${ACC_WEIGHT:-}" && "${ACC_WEIGHT}" != "${ACC_WEIGHT_PINNED}" ]]; then
  echo "[Error] ACC_WEIGHT=${ACC_WEIGHT} conflicts with ARM=${ARM} pin ${ACC_WEIGHT_PINNED}" >&2
  exit 2
fi

SRC_RUN="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
INIT_CKPT="${INIT_CKPT:-${SRC_RUN}/best.pt}"
NORM_STATS="${NORM_STATS:-${SRC_RUN}/norm_stats.npz}"   # explicit: never recompute (contract)
OUT="${OUT:-${ROOT_DIR}/checkpoints/vq_temporal_${ARM}_ep90seq8_10ep}"
LOG="${LOG:-${ROOT_DIR}/analysis/vq_temporal_source/S3/${ARM}.log}"

for path in "${INIT_CKPT}" "${NORM_STATS}"; do
  [[ -f "${path}" ]] || { echo "[Error] missing required file: ${path}" >&2; exit 2; }
done

mkdir -p "$(dirname "${LOG}")"
echo "[AB] arm=${ARM} acc_weight=${ACC_WEIGHT_PINNED} out=${OUT} log=${LOG}"

# Pinned protocol (COMMON.md section 6). Every value below is part of the frozen contract.
SEQ_LEN=8 STRIDE=1 BATCH_SIZE=512 EPOCHS=10 \
EVAL_EVERY=1 SAVE_EVERY=5 EVAL_METRIC=mpjpe \
BEST_METRICS=mpjpe,jitter,foot_sliding,root_ade BEST_MIN_DELTA=0 \
LOADER_SEED=4242 \
ACC_WEIGHT="${ACC_WEIGHT_PINNED}" \
RECON_WEIGHT=3.0 VQ_WEIGHT=1.0 VEL_WEIGHT=15.0 ROOT_WEIGHT=3.0 \
CODE_ALIGN_WEIGHT=2.0 LATENT_L1_WEIGHT=0.01 LATENT_VEL_WEIGHT=0.1 \
LATENT_MARGIN_WEIGHT=0.0 ROUNDTRIP_WEIGHT=0.0 \
DECODER_ONLY=0 TEACHER_CKPT="" TEACHER_ALIGN_WEIGHT=0.0 \
INIT_CKPT="${INIT_CKPT}" NORM_STATS="${NORM_STATS}" \
AMP_DTYPE=fp32 TF32=0 MATMUL_PRECISION=highest COMPILE=0 \
DEVICE="${DEVICE:-cuda}" \
OUT="${OUT}" \
  bash "${ROOT_DIR}/scripts/run_train_basic_vqvae_375to291.sh" \
  --lr 1e-4 --min-lr 1e-5 --seed 42 --save-every 5 \
  2>&1 | tee "${LOG}"

# Post-run evidence: manifest.json with hashes, per-epoch updates, and the final state
# fingerprint. The probe and S4 gates read these files, not the terminal.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}" "${PYTHON_BIN}" \
    "${ROOT_DIR}/analysis/vq_temporal_source/s3_finalize_manifest.py" \
    --run-dir "${OUT}" --arm "${ARM}" --log "${LOG}"
fi
