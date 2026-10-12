#!/usr/bin/env bash
set -euo pipefail

# Attribution arm B: transformer field, no motion token.
#   * --field-arch transformer   (the original 5-token self-attention field; matches
#                                 flowmatch_label_nomirror_150ep's architecture minus motion)
#   * --no-use-motion-token      (inputs: z_t, z_prev, traj, style, time; no 291-d motion path)
# Everything else follows run_train_flowmatch_tf150_staged_noise.sh (the label-150 recipe) and
# run_train_flowmatch_skipcat_notoken.sh, so the two 5-token runs differ ONLY in --field-arch:
#   A) skip_cat_mlp + no motion token -> checkpoints/flowmatch_label_skipcat_notoken
#   B) transformer + no motion token  -> checkpoints/flowmatch_label_transformer_notoken (this run)
# Together with the existing 6-token transformer run they isolate architecture vs input.
#
# Cosine horizon stays at 150; STOP_AFTER_EPOCH=50 previews the clean-history phase. Remove
# STOP_AFTER_EPOCH (or set 0) and resume the same run to continue to the full budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shinn/miniforge3/envs/mcc/bin/python}"
VQ_DIR="${ROOT_DIR}/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
cmd=("${PYTHON_BIN}" "${ROOT_DIR}/training/train_flowmatch_vqvae_controller.py"
  --db "${DB:-${ROOT_DIR}/database_100sty_375_memmap_nomirror}"
  --vqvae-ckpt "${VQ_CKPT:-${VQ_DIR}/best.pt}"
  --norm-stats "${NORM_STATS:-${VQ_DIR}/norm_stats.npz}"
  --vq-target-cache "${VQ_TARGET_CACHE:-${ROOT_DIR}/cache/vq_targets_vq150ep_nomirror_repair_v2}"
  --style-code-prior "${STYLE_CODE_PRIOR:-${ROOT_DIR}/cache/style_code_prior_vq150ep_nomirror_repair_v2.npz}"
  --out-dir "${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_transformer_notoken}"
  --field-arch transformer --no-use-motion-token
  --epochs 150 --lr-schedule-epochs 150 --seed 42
  --seq-len 2 --stride 1 --unroll-steps 1 --no-train-feedback-context
  --batch-size 512 --microbatch-size "${MICROBATCH_SIZE:-0}"
  --prediction-mode target --flow-loss-space normalized
  --lr 5e-4 --min-lr 1e-5 --weight-decay 1e-4 --grad-clip 1
  --flow-hidden-dim 256 --flow-layers 2 --flow-heads 4 --flow-ffn-dim 1024 --dropout 0
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1
  --style-selector-init prior --style-proto-normalization none --style-prior-loss dense
  --style-hist-weight .02 --style-entropy-weight .003 --recipe-prob 0
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0
  --noise-std 1 --solver-steps 8 --inference-requantize --inference-latent-clip
  --matmul-precision highest --no-tf32 --rollout-val-every 0
  --tf-val-every 5 --style-val-mode rollout --style-val-batch-size "${STYLE_VAL_BATCH_SIZE:-64}"
  --tf-val-clips 90 --tf-val-windows 3 --tf-val-selection absolute
  --tf-val-min-matched-accuracy .85 --tf-val-max-matched-ade 5 --tf-val-max-cross-ade 10
  --val-every 1 --save-every 5 --num-workers 8 --prefetch-factor 4)
if [[ "${STOP_AFTER_EPOCH:-0}" != "0" ]]; then cmd+=(--stop-after-epoch "${STOP_AFTER_EPOCH}"); fi
if [[ -n "${DEVICE:-}" ]]; then cmd+=(--device "${DEVICE}"); fi
if [[ -n "${INIT_CKPT:-}" ]]; then cmd+=(--init-ckpt "${INIT_CKPT}"); fi
if [[ -n "${RESUME:-}" ]]; then cmd+=(--resume "${RESUME}"); fi
cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
