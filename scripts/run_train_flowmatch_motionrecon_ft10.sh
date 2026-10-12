#!/usr/bin/env bash
set -euo pipefail

# Motion-space fine-tune of the curriculum100 controller (epoch 86), revised dose after the
# first attempt (weight 1.0, all steps, no anchor) diverged at ~epoch 4:
#   * --recon-weight 0.25 --recon-anchor-weight 0.1 --recon-inference-path: supervises
#     decode(requantize(clip(z_hat))) on the FINAL step only (full Euler8, the inference
#     configuration) with a commitment anchor keeping the predicted latent near its code
#   * lr 5e-5 (vs 1.4e-4): gentle fine-tune so the visual gradient cannot dominate the FM term
#   * continues full-rollout training (p=1 from epoch 1: feedback-start-epoch 1, ramp 0)
#   * bs1024: effective batch 1024, unroll 4, feedback solver 2, shared style table, fast
#     quantize, no-grad feedback (all A/B-tested / bit-identical)
# 10 epochs, cosine 5e-5 -> 1e-5; validation every 5 epochs (240-frame rollout protocol).
#
# NOTE (2026-10-12 checkpoint cleanup): the original 5-token curriculum checkpoint
# (checkpoints/flowmatch_label_curriculum100) was deleted. Set INIT_CKPT explicitly, e.g.
#   INIT_CKPT=checkpoints/<your_run>/best.pt FROM_SCRATCH-independent
# and remember the 5-token remap only applies to 6-token transformer sources.

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
  --latent-stats "${LATENT_STATS:-${ROOT_DIR}/checkpoints/flowmatch_label_nomirror_150ep/latent_stats.npz}"
  ${INIT_CKPT:+--init-ckpt "${INIT_CKPT}"}
  --out-dir "${OUT:-${ROOT_DIR}/checkpoints/flowmatch_label_motionrecon_ft10}"
  --field-arch transformer --no-use-motion-token
  --epochs "${EPOCHS:-10}" --lr-schedule-epochs "${EPOCHS:-10}" --seed 42
  --seq-len 5 --stride 1 --unroll-steps 4
  --feedback-start-epoch 1 --feedback-ramp-epochs 0
  --feedback-probability 1 --feedback-solver-steps 2
  --recon-weight "${RECON_WEIGHT:-0.25}" --recon-anchor-weight "${RECON_ANCHOR_WEIGHT:-0.1}" --recon-inference-path
  --batch-size "${BATCH_SIZE:-1024}" --microbatch-size "${MICROBATCH_SIZE:-0}"
  --prediction-mode target --flow-loss-space normalized
  --lr "${LR:-5e-5}" --min-lr 1e-5 --weight-decay 1e-4 --grad-clip 1
  --flow-hidden-dim 256 --flow-layers 2 --flow-heads 4 --flow-ffn-dim 1024 --dropout 0
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1
  --style-selector-init prior --style-proto-normalization none --style-prior-loss dense
  --style-hist-weight .02 --style-entropy-weight .003 --recipe-prob 0
  --root-recon-weight 0 --style-consistency-weight 0
  --noise-std 1 --solver-steps 8 --inference-requantize --inference-latent-clip
  --matmul-precision highest --no-tf32 --rollout-val-every 0
  --tf-val-every 5 --style-val-mode rollout --style-val-batch-size "${STYLE_VAL_BATCH_SIZE:-64}"
  --tf-val-clips 90 --tf-val-windows 3 --tf-val-selection absolute
  --tf-val-min-matched-accuracy .85 --tf-val-max-matched-ade 5 --tf-val-max-cross-ade 10
  --val-every 1 --save-every 5 --num-workers 8 --prefetch-factor 4)
if [[ -n "${DEVICE:-}" ]]; then cmd+=(--device "${DEVICE}"); fi
if [[ -n "${RESUME:-}" ]]; then cmd+=(--resume "${RESUME}"); fi
cmd+=("$@")
printf '[Run]'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
