"""
Training script for the NewCode CAMDM stylized motion controller.

The controller predicts the clean next-frame motion state from a noisy target
motion sample, conditioned on:
  - the target next-frame 84-d trajectory plan
  - the current 291-d motion state
  - the target style label
  - the diffusion timestep
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

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from Utils.runtime import (
    build_adamw,
    configure_torch_runtime,
    estimate_path_size_bytes,
    resolve_amp_dtype,
    resolve_device,
    sanitize_state_dict_keys,
    set_seed,
    to_float,
    require_unmirrored_db,
)
from Utils.training import (
    CommonDataDefaults,
    CommonLoggingDefaults,
    CommonRuntimeDefaults,
    CommonTrainDefaults,
    add_common_data_args,
    add_common_io_args,
    add_common_logging_args,
    add_common_runtime_args,
    add_common_train_args,
    build_grad_scaler,
    canonicalize_path_args,
    finalize_epoch_outputs,
    load_best_checkpoint_for_eval,
    maybe_compile_model,
    prepare_data_loaders_from_args,
    resume_training_state,
    validate_common_data_args,
    write_config_json,
)
from arch import CAMDMController


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CAMDM stylized motion controller on motion375 memmap data")
    add_common_io_args(parser, script_dir=SCRIPT_DIR, out_dir_name="camdm_controller")
    add_common_data_args(
        parser,
        defaults=CommonDataDefaults(
            seq_len=2,
            stride=1,
            batch_size=512,
            max_windows_for_stats=0,
        ),
    )
    add_common_train_args(parser, defaults=CommonTrainDefaults(epochs=50, lr=2e-4, min_lr=1e-5))
    add_common_runtime_args(parser, defaults=CommonRuntimeDefaults())
    add_common_logging_args(parser, defaults=CommonLoggingDefaults())

    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--loss-type", type=str, choices=["mse", "l1", "smooth_l1"], default="mse")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    return parser.parse_args()


def _flatten_adjacent_pairs(
    x: torch.Tensor,
    style_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3 or x.size(1) < 2:
        raise ValueError(f"Expected x shape [B,T,375] with T>=2, got {tuple(x.shape)}")
    batch, steps, dims = x.shape
    if dims != 375:
        raise ValueError(f"Expected x last dim 375, got {dims}")
    if style_label.numel() != batch:
        raise ValueError("Batch metadata shape mismatch")

    x_curr = x[:, :-1].reshape(-1, dims)
    x_next = x[:, 1:].reshape(-1, dims)
    style_bt = style_label[:, None].expand(batch, steps - 1).reshape(-1)
    return x_curr, x_next, style_bt


def _motion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    smooth_l1_beta: float,
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    if loss_type == "l1":
        return F.l1_loss(prediction, target)
    return F.smooth_l1_loss(prediction, target, beta=float(smooth_l1_beta))


def epoch_pass(
    *,
    model: CAMDMController,
    loader,
    device: torch.device,
    loss_type: str,
    smooth_l1_beta: float,
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

    metric_sums = {
        "loss": torch.zeros((), device=device, dtype=torch.float32),
        "motion_mse": torch.zeros((), device=device, dtype=torch.float32),
        "motion_mae": torch.zeros((), device=device, dtype=torch.float32),
    }
    total_pairs = 0
    start_time = time.time()
    num_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device=device, non_blocking=True).float()
        style_label = batch["label"].to(device=device, non_blocking=True).long()
        x_curr, x_next, style_bt = _flatten_adjacent_pairs(x, style_label)
        traj_next = x_next[:, : model.traj_dim]
        target_motion = x_next[:, model.traj_dim :]

        if is_train and ((step - 1) % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)

        remainder = num_steps % grad_accum_steps
        is_last_incomplete_group = remainder != 0 and step > (num_steps - remainder)
        denom = float(remainder if is_last_incomplete_group else grad_accum_steps)
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        timesteps = torch.randint(0, model.diffusion_steps, (target_motion.size(0),), device=device)
        noise = torch.randn_like(target_motion)
        x_noisy = model.q_sample(target_motion, timesteps, noise)

        autocast_ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled and device.type == "cuda" and amp_dtype is not None,
            )
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            pred_motion = model(
                x_noisy=x_noisy,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_bt,
                timesteps=timesteps,
            )
            loss = _motion_loss(
                pred_motion,
                target_motion,
                loss_type=loss_type,
                smooth_l1_beta=smooth_l1_beta,
            )

        if is_train:
            scaled_loss = loss / denom
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if is_update_step:
                if grad_clip > 0.0:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        pair_count = int(pred_motion.size(0))
        total_pairs += pair_count
        metric_sums["loss"] += loss.detach() * pair_count
        metric_sums["motion_mse"] += F.mse_loss(pred_motion.detach(), target_motion.detach()) * pair_count
        metric_sums["motion_mae"] += torch.mean(torch.abs(pred_motion.detach() - target_motion.detach())) * pair_count

        if log_interval > 0 and (step % log_interval == 0 or step == num_steps):
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[{split}] epoch {epoch:03d}/{total_epochs:03d} "
                f"step {step:05d}/{num_steps:05d} "
                f"loss={to_float(loss):.6f} "
                f"pairs/s={total_pairs / elapsed:.1f}"
            )

    if total_pairs == 0:
        raise RuntimeError(f"{split} loader produced zero adjacent pairs")

    elapsed = max(time.time() - start_time, 1e-6)
    return {
        "loss": to_float(metric_sums["loss"] / total_pairs),
        "motion_mse": to_float(metric_sums["motion_mse"] / total_pairs),
        "motion_mae": to_float(metric_sums["motion_mae"] / total_pairs),
        "pairs_per_sec": float(total_pairs / elapsed),
    }


def main() -> None:
    args = parse_args()
    canonicalize_path_args(args)
    require_unmirrored_db(args.db)
    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.seq_len < 2:
        raise ValueError(f"--seq-len must be >= 2 for CAMDM next-frame supervision, got {args.seq_len}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    print(f"[Info] db              : {args.db}")
    print(f"[Info] out dir         : {args.out_dir}")
    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {args.amp} (dtype={amp_dtype})")
    print(f"[Info] db size         : {estimate_path_size_bytes(args.db) / (1024 ** 3):.3f} GiB")

    prepared = prepare_data_loaders_from_args(
        args=args,
        test_loader_note="best checkpoint will be reloaded for final test evaluation",
    )
    num_styles = len(prepared.bundle.train_dataset.style_names)

    model = CAMDMController(
        num_styles=num_styles,
        traj_dim=84,
        motion_dim=291,
        latent_dim=args.latent_dim,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        activation=args.activation,
        diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
    ).to(device)

    args.model_type = "camdm_375to291_controller"
    args.num_styles = int(num_styles)
    args.traj_dim = 84
    args.motion_dim = 291
    if prepared.stats_path is not None:
        args.norm_stats = str(prepared.stats_path)

    model = maybe_compile_model(model=model, args=args)

    optimizer, fused_enabled = build_adamw(
        model.parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=args.fused_adamw and device.type == "cuda",
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = build_grad_scaler(device=device, amp_enabled=args.amp, amp_dtype=amp_dtype)

    print(f"[Info] num styles      : {num_styles}")
    print(f"[Info] diffusion steps : {args.diffusion_steps}")
    print(f"[Info] optimizer       : AdamW (fused={fused_enabled})")
    print(f"[Info] model params    : {sum(p.numel() for p in model.parameters()):,}")

    config_path = write_config_json(out_dir=args.out_dir, args=args)
    print(f"[Info] config          : {config_path}")

    start_epoch, best_metric = resume_training_state(
        resume_path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        reset_optimizer=args.reset_optimizer,
        best_metric=float("inf"),
        state_dict_transform=sanitize_state_dict_keys,
    )

    history_path = args.out_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = epoch_pass(
            model=model,
            loader=prepared.train_loader,
            device=device,
            loss_type=args.loss_type,
            smooth_l1_beta=args.smooth_l1_beta,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            amp_enabled=args.amp,
            amp_dtype=amp_dtype,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )

        val_metrics = None
        if prepared.val_loader is not None and ((epoch % args.val_every) == 0 or epoch == args.epochs):
            with torch.no_grad():
                val_metrics = epoch_pass(
                    model=model,
                    loader=prepared.val_loader,
                    device=device,
                    loss_type=args.loss_type,
                    smooth_l1_beta=args.smooth_l1_beta,
                    optimizer=None,
                    scaler=None,
                    grad_clip=0.0,
                    grad_accum_steps=1,
                    amp_enabled=args.amp,
                    amp_dtype=amp_dtype,
                    log_interval=max(args.log_interval, 0),
                    epoch=epoch,
                    total_epochs=args.epochs,
                    split="val",
                )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_motion_mse": train_metrics["motion_mse"],
            "train_motion_mae": train_metrics["motion_mae"],
            "train_pairs_per_sec": train_metrics["pairs_per_sec"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_motion_mse": val_metrics["motion_mse"],
                    "val_motion_mae": val_metrics["motion_mae"],
                    "val_pairs_per_sec": val_metrics["pairs_per_sec"],
                }
            )

        metric_for_best = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        best_metric = finalize_epoch_outputs(
            history_path=history_path,
            record=record,
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
        )

        summary = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mae={train_metrics['motion_mae']:.6f}"
        )
        if val_metrics is not None:
            summary += (
                f" val_loss={val_metrics['loss']:.6f}"
                f" val_mae={val_metrics['motion_mae']:.6f}"
            )
        print(summary)

    if prepared.test_loader is not None:
        load_best_checkpoint_for_eval(
            model=model,
            ckpt_path=args.out_dir / "best.pt",
            state_dict_transform=sanitize_state_dict_keys,
        )
        with torch.no_grad():
            test_metrics = epoch_pass(
                model=model,
                loader=prepared.test_loader,
                device=device,
                loss_type=args.loss_type,
                smooth_l1_beta=args.smooth_l1_beta,
                optimizer=None,
                scaler=None,
                grad_clip=0.0,
                grad_accum_steps=1,
                amp_enabled=args.amp,
                amp_dtype=amp_dtype,
                log_interval=0,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        print(
            "[Test] "
            f"loss={test_metrics['loss']:.6f} "
            f"motion_mse={test_metrics['motion_mse']:.6f} "
            f"motion_mae={test_metrics['motion_mae']:.6f}"
        )


if __name__ == "__main__":
    main()
