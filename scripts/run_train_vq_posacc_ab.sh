#!/usr/bin/env bash
set -euo pipefail

# V1 paired pilot: decoder-only physical-position second-order loss (docs ... §5 V1).
#
#   ARM=V1-B0 bash scripts/run_train_vq_posacc_ab.sh              # control: posacc=0
#   ARM=V1-B1 bash scripts/run_train_vq_posacc_ab.sh              # candidate: calibrated lambda
#
# Both arms cold-start from the SAME ep90 VQ state, freeze E/Q/EMA, and update the decoder
# only. The single variable is POSACC_WEIGHT; the protocol is frozen in V1/PROTOCOL.md.
# DRY_RUN=1 prints the expanded command. Training starts only with explicit authorization.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARM="${ARM:-}"
case "${ARM}" in
  V1-B0) POSACC_PINNED="0";;
  V1-B1)
    # Calibrated in analysis/vq_controller_recovery/V1/gradient_calibration.json
    # (decoder ratio 0.105, cos 0.057, window [1.058,2.505] -> geometric centre).
    POSACC_PINNED="${POSACC_WEIGHT:-1.6278}";;
  *) echo "[Error] set ARM=V1-B0 or ARM=V1-B1" >&2; exit 2;;
esac
SRC_RUN="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
INIT_CKPT="${INIT_CKPT:-${SRC_RUN}/best.pt}"
NORM_STATS="${NORM_STATS:-${SRC_RUN}/norm_stats.npz}"
OUT="${OUT:-${ROOT_DIR}/checkpoints/vq_posacc_${ARM}_ep90seq8_5ep}"
LOG="${LOG:-${ROOT_DIR}/analysis/vq_controller_recovery/V1/arms/${ARM}.log}"
POSACC_BETA_CM="${POSACC_BETA_CM:-0.5}"

for path in "${INIT_CKPT}" "${NORM_STATS}"; do
  [[ -f "${path}" ]] || { echo "[Error] missing required file: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${LOG}")"
echo "[V1-AB] arm=${ARM} posacc_weight=${POSACC_PINNED} beta=${POSACC_BETA_CM}cm out=${OUT}"

# Frozen V1 protocol: decoder-only, seq 8, 5 epochs, shared loader order.
DECODER_ONLY=1 INIT_CKPT="${INIT_CKPT}" NORM_STATS="${NORM_STATS}" \
SEQ_LEN=8 STRIDE=1 BATCH_SIZE=512 EPOCHS=5 EVAL_EVERY=1 SAVE_EVERY=5 EVAL_METRIC=mpjpe \
BEST_METRICS=mpjpe,jitter,foot_sliding,root_ade BEST_MIN_DELTA=0 LOADER_SEED=4242 \
ACC_WEIGHT=0 POSACC_WEIGHT="${POSACC_PINNED}" POSACC_BETA_CM="${POSACC_BETA_CM}" \
RECON_WEIGHT=3.0 VQ_WEIGHT=1.0 VEL_WEIGHT=15.0 ROOT_WEIGHT=3.0 \
CODE_ALIGN_WEIGHT=2.0 LATENT_L1_WEIGHT=0.01 LATENT_VEL_WEIGHT=0.1 \
LATENT_MARGIN_WEIGHT=0.0 ROUNDTRIP_WEIGHT=0.0 TEACHER_CKPT="" TEACHER_ALIGN_WEIGHT=0.0 \
AMP_DTYPE=fp32 TF32=0 MATMUL_PRECISION=highest COMPILE=0 DEVICE="${DEVICE:-cuda}" \
OUT="${OUT}" \
  bash "${ROOT_DIR}/scripts/run_train_basic_vqvae_375to291.sh" \
  --lr 1e-4 --min-lr 1e-5 --seed 42 --save-every 5 2>&1 | tee "${LOG}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}" "${PYTHON_BIN}" \
    "${ROOT_DIR}/analysis/vq_controller_recovery/V1/v1_finalize_manifest.py" \
    --run-dir "${OUT}" --arm "${ARM}" --posacc-weight "${POSACC_PINNED}"
fi
