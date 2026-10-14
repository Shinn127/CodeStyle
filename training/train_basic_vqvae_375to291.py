"""
Train a single-branch ControlOperators-style VQ-VAE on motion375to291 data.

Core design:
  - input_mode=traj_motion: [traj, motion] -> encoder -> shared VQ -> decoder -> motion
  - input_mode=motion_only: motion -> encoder -> shared VQ -> decoder -> motion
  - ControlOperators plain, residual-skip, and concatenative-skip MLP variants
  - shared EMA codebook with K-means initialization, dead-code reset, and rotation trick
  - training on short windows (default seq_len=2) with motion/velocity/latent regularizers

Usage example:
  python NewCode/train_basic_vqvae_375to291.py \
    --db NewCode/database_100sty_375_memmap_nomirror \
    --out-dir NewCode/checkpoints/vqvae_traj_motion_375to291 \
    --seq-len 2 --batch-size 512 --epochs 80

The database holds the 90 in-distribution styles split at clip level into
train/val/test; the 10 held-out styles live in `database_100sty_375_memmap_zeroshot`.
`--split-mode` defaults to `auto`, which uses those offline splits, so two overlapping
windows from one clip can never reach different splits.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import contextlib
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from Utils.runtime import (
    build_adamw,
    configure_torch_runtime,
    estimate_path_size_bytes,
    load_checkpoint,
    resolve_amp_dtype,
    resolve_device,
    set_seed,
    to_float,
    require_unmirrored_db,
)
from Utils.vqvae_eval import (
    VQVAEEvalConfig,
    check_best_gate,
    eval_score,
    evaluate_vqvae,
    resolve_best_metrics,
    select_eval_windows,
)
from dataset import load_motion375_metadata
from Utils.training import (
    CommonDataDefaults,
    CommonLoggingDefaults,
    CommonRuntimeDefaults,
    CommonTrainDefaults,
    add_bool_arg,
    add_common_data_args,
    add_common_io_args,
    add_common_logging_args,
    add_common_runtime_args,
    add_common_train_args,
    build_grad_scaler,
    canonicalize_path_args,
    finalize_epoch_outputs,
    maybe_compile_model,
    prepare_data_loaders_from_args,
    resume_training_state,
    validate_common_data_args,
    write_config_json,
)
from arch import BasicVQVAE375to291, normalize_control_network_arch


METRIC_KEYS = (
    "loss",
    "recon_loss",
    "vq_loss",
    "vel_loss",
    "acc_loss",
    "root_loss",
    "latent_l1_loss",
    "latent_vel_loss",
    "codebook_loss",
    "commitment_loss",
    "code_align_loss",
    "latent_margin_loss",
    "boundary_distance",
    "roundtrip_loss",
    "roundtrip_drift",
    "latent_code_gap",
    "perplexity",
    "dead_codes_reset",
)


def epoch_pass(
    *,
    model: BasicVQVAE375to291,
    loader,
    device: torch.device,
    recon_type: str,
    temporal_loss_type: Optional[str],
    recon_weight: float,
    vq_weight: float,
    vel_weight: float,
    acc_weight: float = 0.0,
    latent_l1_weight: float,
    latent_vel_weight: float,
    root_weight: float = 0.0,
    code_align_weight: float = 0.0,
    latent_margin_weight: float = 0.0,
    latent_margin_target: float = 0.05,
    roundtrip_weight: float = 0.0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 0.0,
    grad_accum_steps: int = 1,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    log_interval: int = 100,
    epoch: int = 0,
    total_epochs: int = 0,
    split: str = "train",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    num_steps = len(loader)
    metric_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in METRIC_KEYS}
    n_samples = 0
    n_updates = 0
    start_time = time.time()

    # Built once instead of per step: this loop runs ~11k times per epoch and constructing
    # the context objects inside it was pure overhead.
    autocast_ctx = (
        torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda" and amp_dtype is not None,
        )
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    # Eval passes never call backward, so letting autograd record their graph is waste.
    grad_ctx = contextlib.nullcontext() if is_train else torch.no_grad()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True).float()
        target_motion = batch["motion"].to(device, non_blocking=True).float()
        batch_size = x.shape[0]

        if is_train and ((step - 1) % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)

        remainder = num_steps % grad_accum_steps
        is_last_incomplete_group = remainder != 0 and step > (num_steps - remainder)
        denom = float(remainder if is_last_incomplete_group else grad_accum_steps)
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        with grad_ctx, autocast_ctx:
            loss_dict = model.compute_loss(
                x=x,
                target_motion=target_motion,
                recon_type=recon_type,
                temporal_loss_type=temporal_loss_type,
                recon_weight=recon_weight,
                vq_weight=vq_weight,
                vel_weight=vel_weight,
                acc_weight=acc_weight,
                root_weight=root_weight,
                latent_l1_weight=latent_l1_weight,
                latent_vel_weight=latent_vel_weight,
                code_align_weight=code_align_weight,
                latent_margin_weight=latent_margin_weight,
                latent_margin_target=latent_margin_target,
                roundtrip_weight=roundtrip_weight,
            )
            loss = loss_dict["loss"] / denom

        if is_train:
            if scaler is not None and amp_enabled and device.type == "cuda":
                scaler.scale(loss).backward()
                if is_update_step:
                    if grad_clip > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    n_updates += 1
            else:
                loss.backward()
                if is_update_step:
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    n_updates += 1

        metric_vec = torch.stack(
            [loss_dict[key].detach().to(torch.float32) for key in METRIC_KEYS]
        )
        for idx, key in enumerate(METRIC_KEYS):
            metric_sums[key] = metric_sums[key] + metric_vec[idx] * float(batch_size)
        n_samples += batch_size

        if is_train and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            samples_per_sec = n_samples / max(elapsed, 1e-6)
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={to_float(loss_dict['loss']):.6f} "
                f"recon={to_float(loss_dict['recon_loss']):.6f} "
                f"vq={to_float(loss_dict['vq_loss']):.6f} "
                f"vel={to_float(loss_dict['vel_loss']):.6f} "
                f"acc={to_float(loss_dict['acc_loss']):.6f} "
                f"root={to_float(loss_dict['root_loss']):.6f} "
                f"z_l1={to_float(loss_dict['latent_l1_loss']):.6f} "
            f"align={to_float(loss_dict['code_align_loss']):.6f} "
                f"z_gap={to_float(loss_dict['latent_code_gap']):.6f} "
                f"bd={to_float(loss_dict['boundary_distance']):.6f} "
                f"margin={to_float(loss_dict['latent_margin_loss']):.6f} "
                f"rt={to_float(loss_dict['roundtrip_loss']):.6f} "
                f"rt_drift={to_float(loss_dict['roundtrip_drift']):.6f} "
                f"z_vel={to_float(loss_dict['latent_vel_loss']):.6f} "
                f"ppl={to_float(loss_dict['perplexity']):.3f} "
                f"dead_reset={to_float(loss_dict['dead_codes_reset']):.0f} "
                f"updates={n_updates} "
                f"{samples_per_sec:.1f} samples/s "
                f"time={elapsed:.1f}s"
            )

    if n_samples == 0:
        return {**{key: float("nan") for key in METRIC_KEYS}, "num_samples": 0.0, "num_updates": 0.0}

    mean_metrics = {k: to_float(v / float(n_samples)) for k, v in metric_sums.items()}
    return {
        **mean_metrics,
        "num_samples": float(n_samples),
        "num_updates": float(n_updates),
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Train an input-mode VQ-VAE on motion375to291 memmap DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_io_args(parser, script_dir=script_dir, out_dir_name="vqvae_MODE_375to291_32_4_2048")
    add_common_data_args(
        parser,
        defaults=CommonDataDefaults(
            seq_len=2,
            stride=1,
            batch_size=512,
            max_windows_for_stats=0,
        ),
    )
    add_common_train_args(
        parser,
        defaults=CommonTrainDefaults(
            epochs=50,
            lr=2e-4,
            min_lr=1e-5,
            grad_accum_steps=1,
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run the representation-layer metric evaluation every N epochs (0 disables it)",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        choices=["mpjpe", "local_mpjpe", "jitter", "mpjpe_jitter", "loss", "none"],
        default="mpjpe",
        help=(
            "Primary scalar reported in the logs and stored as `best_metric`; lower is better. "
            "It does not select the checkpoint on its own any more: --best-metrics does "
            "(and is always extended with this metric, so best.pt and best_metric stay in sync). "
            "`none` disables the representation evaluation entirely"
        ),
    )
    parser.add_argument(
        "--best-metrics",
        type=str,
        default="mpjpe,jitter,foot_sliding,root_ade",
        help=(
            "Comma-separated metrics that must ALL beat their value at the last saved best.pt "
            "before it is replaced; lower is better except codebook_usage. Known names: "
            "mpjpe, local_mpjpe, jitter, foot_sliding, root_ade, heading_error, "
            "velocity_dir_error, codebook_usage. 'none' falls back to the single --eval-metric "
            "rule"
        ),
    )
    parser.add_argument(
        "--best-min-delta",
        type=float,
        default=0.0,
        help=(
            "Required improvement per metric for the best-checkpoint gate, in the metric's own "
            "unit (cm). 0 demands a strict improvement; negative values tolerate that much "
            "regression, which is useful when eval noise freezes best.pt"
        ),
    )
    parser.add_argument("--eval-clips", type=int, default=90, help="Validation source clips scored per evaluation")
    parser.add_argument(
        "--eval-window-frames",
        type=int,
        default=60,
        help=(
            "Length of each evaluated sequence. Keep it near 60: the shared metric integrates "
            "the reconstructed root velocity, so longer windows report accumulated drift rather "
            "than reconstruction quality"
        ),
    )
    parser.add_argument(
        "--eval-windows-per-clip",
        type=int,
        default=1,
        help="Windows drawn from random offsets inside each selected clip",
    )
    parser.add_argument("--eval-min-frames", type=int, default=60, help="Skip clips shorter than this")
    parser.add_argument("--eval-clip-seed", type=int, default=1234, help="Seeds which clips form the fixed eval set")
    parser.add_argument("--eval-batch-windows", type=int, default=512)
    add_common_runtime_args(
        parser,
        defaults=CommonRuntimeDefaults(),
    )

    parser.add_argument("--input-dim", type=int, default=375)
    parser.add_argument("--traj-dim", type=int, default=84)
    parser.add_argument("--motion-dim", type=int, default=291)
    parser.add_argument("--input-mode", type=str, choices=["traj_motion", "motion_only"], default="traj_motion")
    parser.add_argument(
        "--encoder-arch",
        type=str,
        choices=["mlp", "skip_mlp", "skip_cat_mlp"],
        default="skip_cat_mlp",
    )
    parser.add_argument(
        "--decoder-arch",
        type=str,
        choices=["mlp", "skip_mlp", "skip_cat_mlp"],
        default="skip_cat_mlp",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--num-latents", type=int, default=32)
    parser.add_argument("--code-dim", type=int, default=4)
    parser.add_argument("--num-embeddings", type=int, default=2048)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    add_bool_arg(parser, name="use-rotation-trick", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-eps", type=float, default=1e-5)
    add_bool_arg(parser, name="kmeans-init", default=True)
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument(
        "--dead-code-threshold",
        type=int,
        default=2,
        help=(
            "EMA cluster-size threshold "
            "below which a code is replaced. Units are assignments per batch, NOT steps: "
            "with batch 512 x 32 heads over 2048 codes the per-code mean is ~8, so 2 is the "
            "upstream default and values far above ~8 replace the whole codebook each step"
        ),
    )

    parser.add_argument("--recon-type", type=str, choices=["l1", "mse", "smooth_l1"], default="l1")
    parser.add_argument(
        "--temporal-loss-type",
        type=str,
        choices=["l1", "mse", "smooth_l1"],
        default="l1",
    )
    parser.add_argument("--recon-weight", type=float, default=3.0)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--code-align-weight", type=float, default=0.0,
        help=("Weight of the continuous-vs-discrete decode consistency term: the decoder must "
              "reproduce the quantized path from the continuous latent (see compute_loss); "
              "0 keeps the paper recipe."))
    parser.add_argument("--latent-margin-weight", type=float, default=0.0,
        help=("Weight of a hinge that pushes each encoder output away from its Voronoi "
              "boundary (the measured codec failure mode: median exact boundary distance 0.011 "
              "vs code spacing 0.154, so any equal-norm disturbance flips the code). Trains the "
              "encoder only; the EMA codebook has no gradient."))
    parser.add_argument("--latent-margin-target", type=float, default=0.05,
        help="Target boundary distance (raw latent units) for --latent-margin-weight.")
    parser.add_argument("--roundtrip-weight", type=float, default=0.0,
        help=("Weight of the re-encode consistency term: decode the prediction, re-encode it "
              "(with the GT trajectory block) and pull the result back to the code it came "
              "from. Directly minimizes out-of-cell drift under self-feedback."))
    parser.add_argument("--vel-weight", type=float, default=15.0)
    parser.add_argument(
        "--acc-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of a second-order (acceleration) term |d2 pred - d2 target| on the motion "
            "block. The first-order vel term still leaves per-frame output noise, which is what "
            "jitter / foot sliding measure; this term targets that band. Needs --seq-len >= 3. "
            "0 keeps the paper recipe; measured d2 error is 0.013 (autoencoder path) vs 0.055 "
            "(quantized) in normalized units, so 3-5 is a useful range"
        ),
    )
    parser.add_argument("--latent-l1-weight", type=float, default=0.01)
    parser.add_argument("--latent-vel-weight", type=float, default=0.1)
    parser.add_argument(
        "--root-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of an extra L1 term on the 3 root-motion channels (planar velocity, yaw "
            "rate, vertical velocity). Inside the 291-dim reconstruction those channels only "
            "get 3/291 of the gradient, but their error is the one that integrates into "
            "trajectory drift. 0 keeps the paper's recipe; 3-10 is a useful range"
        ),
    )

    parser.add_argument("--init-ckpt", type=Path, default=None,
        help=("Warm start: initialise encoder/decoder from this checkpoint while starting a "
              "fresh run (fresh optimizer/scheduler, epoch 1). When the codebook size differs "
              "from the checkpoint's, quantizer buffers are dropped and re-initialised by "
              "k-means on the first batch."))
    add_common_logging_args(
        parser,
        defaults=CommonLoggingDefaults(val_every=1, log_interval=500),
    )
    parser.set_defaults(compile=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.seq_len < 1:
        raise ValueError(f"--seq-len must be >= 1, got {args.seq_len}")
    if args.input_dim != args.traj_dim + args.motion_dim:
        raise ValueError(
            f"--input-dim must equal --traj-dim + --motion-dim, got "
            f"{args.input_dim} != {args.traj_dim} + {args.motion_dim}"
        )
    if args.hidden_dim < 1:
        raise ValueError(f"--hidden-dim must be >= 1, got {args.hidden_dim}")
    if args.decoder_hidden_dim < 1:
        raise ValueError(f"--decoder-hidden-dim must be >= 1, got {args.decoder_hidden_dim}")
    if args.encoder_layers < 1:
        raise ValueError(f"--encoder-layers must be >= 1, got {args.encoder_layers}")
    if args.decoder_layers < 1:
        raise ValueError(f"--decoder-layers must be >= 1, got {args.decoder_layers}")
    if args.num_latents < 1:
        raise ValueError(f"--num-latents must be >= 1, got {args.num_latents}")
    if args.code_dim < 1:
        raise ValueError(f"--code-dim must be >= 1, got {args.code_dim}")
    if args.num_embeddings < 1:
        raise ValueError(f"--num-embeddings must be >= 1, got {args.num_embeddings}")
    if args.commitment_cost < 0.0:
        raise ValueError(f"--commitment-cost must be >= 0, got {args.commitment_cost}")
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError(f"--ema-decay must be in [0,1), got {args.ema_decay}")
    if args.ema_eps <= 0.0:
        raise ValueError(f"--ema-eps must be > 0, got {args.ema_eps}")
    if args.kmeans_iters < 1:
        raise ValueError(f"--kmeans-iters must be >= 1, got {args.kmeans_iters}")
    if args.recon_weight < 0.0:
        raise ValueError(f"--recon-weight must be >= 0, got {args.recon_weight}")
    if args.vq_weight < 0.0:
        raise ValueError(f"--vq-weight must be >= 0, got {args.vq_weight}")
    if args.vel_weight < 0.0:
        raise ValueError(f"--vel-weight must be >= 0, got {args.vel_weight}")
    if args.acc_weight < 0.0:
        raise ValueError(f"--acc-weight must be >= 0, got {args.acc_weight}")
    if args.acc_weight > 0.0 and args.seq_len < 3:
        raise ValueError(
            f"--acc-weight > 0 needs --seq-len >= 3 to form a second difference, got seq_len={args.seq_len}"
        )
    if args.latent_l1_weight < 0.0:
        raise ValueError(f"--latent-l1-weight must be >= 0, got {args.latent_l1_weight}")
    if args.latent_vel_weight < 0.0:
        raise ValueError(f"--latent-vel-weight must be >= 0, got {args.latent_vel_weight}")
    args.input_mode = BasicVQVAE375to291.normalize_input_mode(args.input_mode)
    args.encoder_arch = normalize_control_network_arch(args.encoder_arch)
    args.decoder_arch = normalize_control_network_arch(args.decoder_arch)


def build_model(args: argparse.Namespace, device: torch.device) -> BasicVQVAE375to291:
    model = BasicVQVAE375to291(
        input_dim=args.input_dim,
        traj_dim=args.traj_dim,
        motion_dim=args.motion_dim,
        input_mode=args.input_mode,
        encoder_arch=args.encoder_arch,
        decoder_arch=args.decoder_arch,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_layers=args.decoder_layers,
        num_latents=args.num_latents,
        code_dim=args.code_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        use_rotation_trick=args.use_rotation_trick,
        ema_decay=args.ema_decay,
        ema_eps=args.ema_eps,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        dead_code_threshold=args.dead_code_threshold,
    ).to(device)
    return maybe_compile_model(model=model, args=args)


def build_optimizer(model: BasicVQVAE375to291, args: argparse.Namespace, device: torch.device):
    return build_adamw(
        model.parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=bool(args.fused_adamw and device.type == "cuda"),
    )


def main() -> None:
    args = parse_args()
    # Fail on a bad --best-metrics name before spending time on data preparation.
    best_metric_names = resolve_best_metrics(args.best_metrics, primary=args.eval_metric)
    script_dir = Path(__file__).resolve().parents[1]
    default_out_dir = script_dir / "checkpoints" / "vqvae_MODE_375to291_32_4_2048"
    if args.out_dir == default_out_dir:
        arch_tag = (
            args.encoder_arch
            if args.encoder_arch == args.decoder_arch
            else f"{args.encoder_arch}_to_{args.decoder_arch}"
        )
        args.out_dir = script_dir / "checkpoints" / f"vqvae_{args.input_mode}_{arch_tag}_375to291_32_4_2048"
    canonicalize_path_args(args)
    require_unmirrored_db(args.db)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device=device)
    amp_enabled = bool(args.amp and device.type == "cuda" and amp_dtype is not None)

    args.model_name = (
        f"BasicVQVAE375to291[{args.input_mode};"
        f"{args.encoder_arch}->{args.decoder_arch}]"
    )
    args.model_type = "input_mode_vqvae_375to291"

    print(f"[Info] model           : {args.model_name}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out_dir         : {args.out_dir}")
    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {amp_enabled} ({args.amp_dtype})")
    print(f"[Info] input mode      : {args.input_mode}")
    print(f"[Info] encoder arch    : {args.encoder_arch}")
    print(f"[Info] decoder arch    : {args.decoder_arch}")
    print(
        f"[Info] backend         : tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"[Info] seq_len         : {args.seq_len}")
    print(
        f"[Info] loss weights    : recon={args.recon_weight} vq={args.vq_weight} "
        f"vel={args.vel_weight} acc={args.acc_weight} root={args.root_weight} "
        f"z_l1={args.latent_l1_weight} z_vel={args.latent_vel_weight}"
    )
    print(
        f"\n[Info] codebook        : num_latents={args.num_latents} code_dim={args.code_dim} "
        f"num_embeddings={args.num_embeddings} ema=True"
    )
    print(f"[Info] rot trick       : {args.use_rotation_trick}")
    print(f"[Info] k-means init    : {args.kmeans_init} (iters={args.kmeans_iters})")
    print(
        f"[Info] dead code reset : threshold={args.dead_code_threshold} "
        f"(EMA assignments per batch)"
    )
    print(f"[Info] grad accum      : {args.grad_accum_steps}")
    print(f"[Info] split ratio     : val={args.val_ratio} test={args.test_ratio}")
    print(
        f"[Info] data fraction   : train={args.train_fraction} "
        f"val={args.val_fraction} test={args.test_fraction}"
    )
    print(f"[Info] val every       : {args.val_every}")
    db_size_gb = estimate_path_size_bytes(args.db) / (1024.0 ** 3) if args.db.exists() else float("nan")
    print(f"[Info] db size         : {db_size_gb:.2f} GB")
    if not args.load_into_memory and np.isfinite(db_size_gb):
        print(
            "[Tip] if RAM is sufficient, add --load-into-memory for faster training throughput "
            f"(needs about {db_size_gb + 2.0:.1f} GB memory for X + overhead)."
        )

    write_config_json(out_dir=args.out_dir, args=args)

    prepared = prepare_data_loaders_from_args(
        args=args,
        test_loader_note="prepared (reserved for later evaluation)",
    )
    train_loader = prepared.train_loader
    val_loader = prepared.val_loader

    model = build_model(args, device=device)
    if args.init_ckpt is not None:
        # Warm start: load encoder/decoder weights only, keep a fresh optimizer/scheduler and
        # start at epoch 1. When the codebook size changed the quantizer buffers are dropped
        # so the library re-initialises the new table with k-means on the first batch.
        initial = load_checkpoint(args.init_ckpt)
        initial_args = initial.get("args") or {}
        state = dict(initial["model"])
        drop_quantizer = int(initial_args.get("num_embeddings", -1)) != int(args.num_embeddings)
        if drop_quantizer:
            state = {k: v for k, v in state.items() if not k.startswith("quantizer.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("quantizer.")]
        if unexpected or real_missing:
            raise ValueError(
                f"init-ckpt incompatible: missing={real_missing}, unexpected={unexpected}")
        print(
            f"[Info] init from       : {args.init_ckpt} "
            f"({'encoder/decoder only, codebook re-initialised' if drop_quantizer else 'full state'})")
    optimizer, use_fused_adamw = build_optimizer(model=model, args=args, device=device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = build_grad_scaler(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    print(f"[Info] optimizer       : AdamW(fused={use_fused_adamw})")

    history_path = args.out_dir / "history.jsonl"
    best_metric = float("inf")
    # Values of the metrics at the checkpoint currently stored as best.pt. `best.pt` is only
    # replaced when every gated metric beats its entry here, so the accepted checkpoints form
    # a sequence that is better across the board instead of better on one metric alone.
    best_metric_values: dict[str, float] | None = None
    best_epoch: int | None = None
    gate_reject_streak = 0

    # The representation-layer evaluation set is fixed for the whole run, so "best" is
    # comparable across epochs. It reads the same val split the loader uses.
    eval_config = None
    if args.eval_every > 0 and args.eval_metric != "none":
        db_meta = load_motion375_metadata(args.db)
        eval_config = VQVAEEvalConfig(
            clips=args.eval_clips,
            window_frames=args.eval_window_frames,
            windows_per_clip=args.eval_windows_per_clip,
            min_frames_per_clip=args.eval_min_frames,
            split="val",
            seed=args.eval_clip_seed,
            batch_windows=args.eval_batch_windows,
        )
        preview = select_eval_windows(
            args.db,
            split="val",
            clips=args.eval_clips,
            window_frames=args.eval_window_frames,
            windows_per_clip=args.eval_windows_per_clip,
            min_frames=args.eval_min_frames,
            seed=args.eval_clip_seed,
        )
        print(
            f"[Info] eval            : every {args.eval_every} epochs, primary metric={args.eval_metric}, "
            f"{len(preview)} fixed val windows of {args.eval_window_frames} frames "
            f"({sum(frames for _, frames in preview):,} frames total)"
        )
        print(
            f"[Info] best gate       : save best.pt only when ALL of "
            f"{', '.join(best_metric_names)} improve (min_delta={args.best_min_delta})"
        )
    else:
        db_meta = None
        print("[Info] eval            : disabled (best checkpoint follows the validation loss)")

    start_epoch, best_metric = resume_training_state(
        resume_path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        reset_optimizer=args.reset_optimizer,
        best_metric=best_metric,
        strict=True,
    )

    # A resumed run that switches metric has an incomparable best_metric baseline: the
    # stored value belongs to the previous metric's scale. The same holds for the gate
    # reference, so both are reset together.
    if args.resume is not None and Path(args.resume).exists():
        previous = load_checkpoint(Path(args.resume))
        previous_args = previous.get("args") or {}
        previous_metric = previous_args.get("eval_metric")
        if previous_metric is not None and previous_metric != args.eval_metric:
            print(
                f"[Warn] eval metric changed on resume ({previous_metric} -> {args.eval_metric}); "
                f"resetting the best-checkpoint baseline"
            )
            best_metric = float("inf")
            best_metric_values = None
        previous_names = previous.get("best_metrics")
        stored_values = previous.get("best_metric_values")
        if best_metric_names and previous_names is not None:
            if tuple(str(name) for name in previous_names) != tuple(best_metric_names):
                print(
                    f"[Warn] best-metric gate changed on resume ({list(previous_names)} -> "
                    f"{list(best_metric_names)}); resetting the gate reference"
                )
                best_metric_values = None
            elif isinstance(stored_values, dict) and stored_values:
                best_metric_values = {str(k): float(v) for k, v in stored_values.items()}
                reference_text = ", ".join(
                    f"{name}={best_metric_values[name]:.4f}"
                    for name in best_metric_names
                    if name in best_metric_values
                )
                print(f"[Info] best gate       : restored reference from resume ({reference_text})")

    if start_epoch > args.epochs:
        print(
            f"[Warn] start_epoch ({start_epoch}) > epochs ({args.epochs}). "
            "Nothing to train. Increase --epochs or use another checkpoint."
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={lr_now:.8f}")

        train_metrics = epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            recon_type=args.recon_type,
            temporal_loss_type=args.temporal_loss_type,
            recon_weight=args.recon_weight,
            vq_weight=args.vq_weight,
            vel_weight=args.vel_weight,
            acc_weight=args.acc_weight,
            latent_l1_weight=args.latent_l1_weight,
            latent_vel_weight=args.latent_vel_weight,
            code_align_weight=args.code_align_weight,
            latent_margin_weight=args.latent_margin_weight,
            latent_margin_target=args.latent_margin_target,
            roundtrip_weight=args.roundtrip_weight,
            root_weight=args.root_weight,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )

        if val_loader is None:
            val_metrics = None
            metric_for_best = train_metrics["loss"]
        else:
            if (epoch % args.val_every) == 0:
                with torch.no_grad():
                    val_metrics = epoch_pass(
                        model=model,
                        loader=val_loader,
                        device=device,
                        recon_type=args.recon_type,
                        temporal_loss_type=args.temporal_loss_type,
                        recon_weight=args.recon_weight,
                        vq_weight=args.vq_weight,
                        vel_weight=args.vel_weight,
                        acc_weight=args.acc_weight,
                        latent_l1_weight=args.latent_l1_weight,
                        latent_vel_weight=args.latent_vel_weight,
                        code_align_weight=args.code_align_weight,
                        latent_margin_weight=args.latent_margin_weight,
                        latent_margin_target=args.latent_margin_target,
                        roundtrip_weight=args.roundtrip_weight,
                        root_weight=args.root_weight,
                        optimizer=None,
                        scaler=None,
                        grad_clip=0.0,
                        grad_accum_steps=1,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        log_interval=args.log_interval,
                        epoch=epoch,
                        total_epochs=args.epochs,
                        split="val",
                    )
                metric_for_best = val_metrics["loss"]
            else:
                val_metrics = None
                metric_for_best = None

        # Representation-layer metrics (MPJPE / Jitter / F.S. from Utils.motion_eval) on a
        # fixed set of validation clips. This is what selects the best checkpoint.
        eval_metrics = None
        gate = None
        if eval_config is not None and (epoch % args.eval_every) == 0:
            eval_metrics = evaluate_vqvae(
                model,
                db_path=args.db,
                num_frames_total=int(db_meta.num_frames),
                norm_mean=prepared.bundle.norm_mean,
                norm_std=prepared.bundle.norm_std,
                device=device,
                config=eval_config,
                num_embeddings=int(args.num_embeddings),
                traj_dim=int(args.traj_dim),
            )
            print(
                f"[Eval  {epoch}/{args.epochs}] "
                f"MPJPE={eval_metrics['mpjpe_cm']:.4f} cm  "
                f"poseMPJPE={eval_metrics['local_mpjpe_cm']:.4f} cm  "
                f"Jitter={eval_metrics['jitter_cm']:.4f} cm  "
                f"F.S.={eval_metrics['foot_sliding_cm']:.4f} cm  "
                f"rootADE={eval_metrics['root_ade_cm']:.4f} cm  "
                f"codebook_usage={100 * eval_metrics['codebook_usage']:.2f}%  "
                f"clips={int(eval_metrics['eval_clips'])}  "
                f"frames={int(eval_metrics['eval_frames']):,}"
            )
            metric_for_best = eval_score(eval_metrics, args.eval_metric)
            if best_metric_names:
                gate = check_best_gate(
                    eval_metrics,
                    best_metric_names,
                    reference=best_metric_values,
                    min_delta=args.best_min_delta,
                )
                if gate.accept:
                    print(f"[Gate  {epoch}/{args.epochs}] accept: {gate.describe()}")
                    gate_reject_streak = 0
                else:
                    metric_for_best = None
                    gate_reject_streak += 1
                    print(
                        f"[Gate  {epoch}/{args.epochs}] reject, best.pt kept: {gate.describe()}"
                    )
                    if gate_reject_streak >= 3:
                        print(
                            f"[Warn] best.pt has not been replaced for {gate_reject_streak} "
                            f"consecutive evaluations (blocked by {', '.join(gate.blockers)}). "
                            f"Relax the gate with --best-min-delta -0.02, drop noisy metrics from "
                            f"--best-metrics, or fall back to the periodic epoch_*.pt checkpoints "
                            f"(best.pt itself is unchanged)"
                        )
        elif eval_config is not None and args.eval_metric not in {"none", "loss"}:
            # Do not let the (noisier, differently scaled) validation loss overwrite the
            # best checkpoint between metric evaluations.
            metric_for_best = None

        scheduler.step()

        epoch_sec = time.time() - epoch_t0
        message = (
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"train_vq={train_metrics['vq_loss']:.6f} "
            f"train_vel={train_metrics['vel_loss']:.6f} "
            f"train_acc={train_metrics['acc_loss']:.6f} "
            f"train_root={train_metrics['root_loss']:.6f} "
            f"train_z_l1={train_metrics['latent_l1_loss']:.6f} "
            f"train_z_vel={train_metrics['latent_vel_loss']:.6f} "
            f"train_ppl={train_metrics['perplexity']:.3f}"
        )
        if val_metrics is not None:
            message += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"val_vq={val_metrics['vq_loss']:.6f} "
                f"val_vel={val_metrics['vel_loss']:.6f} "
                f"val_acc={val_metrics['acc_loss']:.6f} "
                f"val_z_l1={val_metrics['latent_l1_loss']:.6f} "
                f"val_z_vel={val_metrics['latent_vel_loss']:.6f} "
                f"val_ppl={val_metrics['perplexity']:.3f}"
            )
        if eval_metrics is not None:
            message += (
                f" | eval_mpjpe={eval_metrics['mpjpe_cm']:.4f} "
                f"eval_pose_mpjpe={eval_metrics['local_mpjpe_cm']:.4f} "
                f"eval_jitter={eval_metrics['jitter_cm']:.4f} "
                f"eval_fs={eval_metrics['foot_sliding_cm']:.4f} "
                f"eval_root_ade={eval_metrics['root_ade_cm']:.4f} "
                f"eval_cb_usage={100 * eval_metrics['codebook_usage']:.2f}%"
            )
            if metric_for_best is not None:
                message += f" | best_by={args.eval_metric}({metric_for_best:.4f})"
            if gate is not None:
                message += f" | gate={'accept' if gate.accept else 'reject'}"
        message += f" | time={epoch_sec:.1f}s"
        print(message)

        previous_best_metric = best_metric
        # The gate reference moves only together with best.pt itself, so a rejected epoch can
        # never become the yardstick for later epochs. `metric_for_best < best_metric` is the
        # same condition finalize_epoch_outputs uses to rewrite best.pt; the gate always
        # contains the primary metric, so an accepted gate implies it holds.
        accepted_values = (
            dict(gate.values)
            if (
                gate is not None
                and gate.accept
                and metric_for_best is not None
                and metric_for_best < best_metric
            )
            else None
        )
        reference_after_save = accepted_values if accepted_values is not None else best_metric_values
        best_metric = finalize_epoch_outputs(
            history_path=history_path,
            record={
                "epoch": epoch,
                "lr": lr_now,
                "train": train_metrics,
                "val": val_metrics,
                "eval": eval_metrics,
                "best_metric_name": args.eval_metric,
                "best_metric_value": metric_for_best,
                "best_gate": None
                if gate is None
                else {
                    "metrics": list(best_metric_names),
                    "min_delta": args.best_min_delta,
                    "accept": bool(gate.accept),
                    "reason": gate.reason,
                    "values": dict(gate.values),
                    "reference": dict(gate.reference),
                    "blockers": list(gate.blockers),
                },
                "best_metric_values": None
                if reference_after_save is None
                else dict(reference_after_save),
                "time_sec": epoch_sec,
            },
            out_dir=args.out_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            metric_for_best=metric_for_best,
            save_every=args.save_every,
            args=args,
            extra={
                "best_metrics": list(best_metric_names),
                "best_metric_values": None
                if reference_after_save is None
                else dict(reference_after_save),
            },
        )
        if accepted_values is not None and best_metric < previous_best_metric:
            best_metric_values = accepted_values
            best_epoch = epoch
        elif accepted_values is not None:
            print(
                "[Warn] gate accepted but best.pt was not rewritten; keeping the previous "
                "gate reference"
            )

    print("\n[Done] Training complete.")
    print(f"[Done] Best metric ({args.eval_metric}): {best_metric:.6f}")
    if best_metric_values:
        print(
            "[Done] Best metrics: "
            + "  ".join(f"{name}={best_metric_values[name]:.4f}" for name in best_metric_names if name in best_metric_values)
            + (f"  (best.pt from epoch {best_epoch})" if best_epoch is not None else "")
        )
    print(f"[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
