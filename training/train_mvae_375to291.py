"""
Training script for MVAE375to291.

Adapted to this project's dataset format:
  375 = traj(84) + motion(291)

The MVAE predicts future motion from:
  - autoregressive condition frames of full 375-d features
  - future trajectory condition (84-d)
  - stochastic latent z

Usage example:
  python NewCode/train_mvae_375to291.py \
    --db NewCode/database_100sty_375_memmap_nomirror \
    --out-dir NewCode/checkpoints/mvae_375to291 \
    --seq-len 32 --batch-size 512 --epochs 50
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
    resolve_amp_dtype,
    resolve_device,
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
    append_jsonl,
    canonicalize_path_args,
    load_best_checkpoint_for_eval,
    prepare_data_loaders_from_args,
    resume_training_state,
    save_training_checkpoint,
    validate_common_data_args,
    write_config_json,
)
from arch import MVAE375to291


def compute_student_prob(
    epoch: int,
    *,
    teacher_epochs: int,
    ramp_epochs: int,
) -> float:
    if teacher_epochs < 0:
        raise ValueError(f"teacher_epochs must be >= 0, got {teacher_epochs}")
    if ramp_epochs < 0:
        raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")
    if epoch <= teacher_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    if epoch <= (teacher_epochs + ramp_epochs):
        return float(epoch - teacher_epochs) / float(ramp_epochs)
    return 1.0


def _weighted_future_recon_loss(
    *,
    model: MVAE375to291,
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    recon_type: str,
    future_weights: torch.Tensor,
) -> torch.Tensor:
    # pred/target: [B, F, D]
    elem = model._loss_by_type(pred_motion, target_motion, recon_type)
    per_future = elem.mean(dim=(0, 2))
    return torch.sum(per_future * future_weights)


def epoch_pass(
    *,
    model: MVAE375to291,
    loader,
    device: torch.device,
    recon_type: str,
    recon_weight: float,
    kl_weight: float,
    rollout_steps: int,
    student_prob: float,
    future_weights: torch.Tensor,
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
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
    if not (0.0 <= student_prob <= 1.0):
        raise ValueError(f"student_prob must be in [0,1], got {student_prob}")

    num_steps = len(loader)
    metric_keys = [
        "loss",
        "recon_loss",
        "kl_loss",
        "avg_rollout_steps",
    ]
    metric_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in metric_keys}
    n_samples = 0
    n_updates = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True).float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected batch x rank-3 [B,T,D], got {tuple(x.shape)}")
        if x.size(-1) != model.input_dim:
            raise ValueError(
                f"Expected x last dim {model.input_dim}, got {x.size(-1)}"
            )

        bs, seq_len, _ = x.shape
        max_rollout = seq_len - model.num_condition_frames - model.num_future_predictions + 1
        if max_rollout < 1:
            raise ValueError(
                "Sequence is too short for configured MVAE windows: "
                f"seq_len={seq_len}, condition={model.num_condition_frames}, "
                f"future={model.num_future_predictions}"
            )
        n_rollout = min(rollout_steps, max_rollout)

        if is_train and ((step - 1) % grad_accum_steps == 0):
            optimizer.zero_grad(set_to_none=True)

        remainder = num_steps % grad_accum_steps
        is_last_incomplete_group = remainder != 0 and step > (num_steps - remainder)
        denom = float(remainder if is_last_incomplete_group else grad_accum_steps)
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

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
            history = x[:, : model.num_condition_frames, :].clone()

            batch_recon_sum = x.new_zeros(())
            batch_kl_sum = x.new_zeros(())
            batch_total_sum = x.new_zeros(())

            for offset in range(n_rollout):
                begin = model.num_condition_frames + offset
                end = begin + model.num_future_predictions

                target_x = x[:, begin:end, :]
                target_traj = target_x[..., : model.traj_dim]
                target_motion = target_x[..., model.traj_dim :]

                pred_motion, mu, logvar = model(
                    x_future=target_x,
                    future_traj=target_traj,
                    condition=history[:, : model.num_condition_frames, :],
                )
                if pred_motion.dim() == 2:
                    pred_motion = pred_motion.unsqueeze(1)

                recon_loss = _weighted_future_recon_loss(
                    model=model,
                    pred_motion=pred_motion,
                    target_motion=target_motion,
                    recon_type=recon_type,
                    future_weights=future_weights,
                )
                kl_loss = model.kl_loss(mu, logvar)
                total_loss = recon_weight * recon_loss + kl_weight * kl_loss

                batch_recon_sum = batch_recon_sum + recon_loss
                batch_kl_sum = batch_kl_sum + kl_loss
                batch_total_sum = batch_total_sum + total_loss

                use_student = bool(torch.rand((), device=x.device) < student_prob)
                if use_student:
                    next_frame = torch.cat(
                        (target_traj[:, 0, :], pred_motion[:, 0, :]),
                        dim=-1,
                    )
                else:
                    next_frame = target_x[:, 0, :]

                history = history.roll(shifts=1, dims=1)
                history[:, 0, :].copy_(next_frame.detach())

            loss_value = batch_total_sum / float(n_rollout)
            recon_value = batch_recon_sum / float(n_rollout)
            kl_value = batch_kl_sum / float(n_rollout)
            loss = loss_value / denom

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

        metric_sums["loss"] = metric_sums["loss"] + loss_value.detach().to(torch.float32) * float(bs)
        metric_sums["recon_loss"] = (
            metric_sums["recon_loss"] + recon_value.detach().to(torch.float32) * float(bs)
        )
        metric_sums["kl_loss"] = metric_sums["kl_loss"] + kl_value.detach().to(torch.float32) * float(bs)
        metric_sums["avg_rollout_steps"] = (
            metric_sums["avg_rollout_steps"] + torch.tensor(float(n_rollout), device=device) * float(bs)
        )
        n_samples += bs

        if is_train and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            samples_per_sec = n_samples / max(elapsed, 1e-6)
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={to_float(loss_value):.6f} "
                f"recon={to_float(recon_value):.6f} "
                f"kl={to_float(kl_value):.6f} "
                f"sp={student_prob:.3f} "
                f"updates={n_updates} "
                f"{samples_per_sec:.1f} samples/s "
                f"time={elapsed:.1f}s"
            )

    if n_samples == 0:
        return {
            "loss": float("nan"),
            "recon_loss": float("nan"),
            "kl_loss": float("nan"),
            "avg_rollout_steps": float("nan"),
            "student_prob": float(student_prob),
            "num_samples": 0.0,
            "num_updates": 0.0,
        }

    mean_metrics = {k: to_float(v / float(n_samples)) for k, v in metric_sums.items()}
    return {
        "loss": mean_metrics["loss"],
        "recon_loss": mean_metrics["recon_loss"],
        "kl_loss": mean_metrics["kl_loss"],
        "avg_rollout_steps": mean_metrics["avg_rollout_steps"],
        "student_prob": float(student_prob),
        "num_samples": float(n_samples),
        "num_updates": float(n_updates),
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train MVAE375to291 on 100STYLE memmap DB")
    add_common_io_args(parser, script_dir=script_dir, out_dir_name="mvae_375to291")
    add_common_data_args(
        parser,
        defaults=CommonDataDefaults(
            seq_len=8,
            stride=4,
            batch_size=512,
            max_windows_for_stats=0,
        ),
    )
    add_common_train_args(
        parser,
        defaults=CommonTrainDefaults(min_lr=1e-5),
    )
    add_common_runtime_args(
        parser,
        defaults=CommonRuntimeDefaults(),
    )

    parser.add_argument("--input-dim", type=int, default=375)
    parser.add_argument("--traj-dim", type=int, default=84)
    parser.add_argument("--motion-dim", type=int, default=291)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--num-condition-frames", type=int, default=1)
    parser.add_argument("--num-future-predictions", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=6)

    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--ramp-epochs", type=int, default=10)
    parser.add_argument("--val-student-prob", type=float, default=1.0)

    parser.add_argument("--recon-type", type=str, choices=["l1", "mse", "smooth_l1"], default="mse")
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.2)

    add_common_logging_args(
        parser,
        defaults=CommonLoggingDefaults(val_every=5),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonicalize_path_args(args)
    require_unmirrored_db(args.db)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    amp_enabled = bool(args.amp and device.type == "cuda" and amp_dtype is not None)

    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.num_condition_frames < 1:
        raise ValueError(f"--num-condition-frames must be >= 1, got {args.num_condition_frames}")
    if args.num_future_predictions < 1:
        raise ValueError(
            f"--num-future-predictions must be >= 1, got {args.num_future_predictions}"
        )
    min_seq_for_one_rollout = args.num_condition_frames + args.num_future_predictions
    if args.seq_len < min_seq_for_one_rollout:
        raise ValueError(
            f"--seq-len ({args.seq_len}) must be >= "
            f"num_condition_frames + num_future_predictions ({min_seq_for_one_rollout})"
        )
    if args.rollout_steps < 1:
        raise ValueError(f"--rollout-steps must be >= 1, got {args.rollout_steps}")
    if args.teacher_epochs < 0:
        raise ValueError(f"--teacher-epochs must be >= 0, got {args.teacher_epochs}")
    if args.ramp_epochs < 0:
        raise ValueError(f"--ramp-epochs must be >= 0, got {args.ramp_epochs}")
    if (args.teacher_epochs + args.ramp_epochs) > args.epochs:
        raise ValueError(
            f"--teacher-epochs + --ramp-epochs must be <= --epochs, got "
            f"{args.teacher_epochs} + {args.ramp_epochs} > {args.epochs}"
        )
    if not (0.0 <= args.val_student_prob <= 1.0):
        raise ValueError(
            f"--val-student-prob must be in [0,1], got {args.val_student_prob}"
        )

    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {amp_enabled} ({args.amp_dtype})")
    print(
        f"[Info] backend         : tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out_dir         : {args.out_dir}")
    db_size_gb = estimate_path_size_bytes(args.db) / (1024.0 ** 3) if args.db.exists() else float("nan")
    print(f"[Info] db size         : {db_size_gb:.2f} GB")
    print(
        f"[Info] model           : cond={args.num_condition_frames} "
        f"future={args.num_future_predictions} latent={args.latent_dim} experts={args.num_experts}"
    )
    print(
        f"[Info] schedule        : teacher={args.teacher_epochs} "
        f"ramp={args.ramp_epochs} rollout={args.rollout_steps}"
    )
    print(
        f"[Info] losses          : recon={args.recon_weight} "
        f"kl={args.kl_weight} recon_type={args.recon_type}"
    )
    print(f"[Info] grad accum      : {args.grad_accum_steps}")
    print(f"[Info] split ratio     : val={args.val_ratio} test={args.test_ratio}")
    print(
        f"[Info] data fraction   : train={args.train_fraction} "
        f"val={args.val_fraction} test={args.test_fraction}"
    )
    print(f"[Info] val every       : {args.val_every}")
    print(
        f"[Info] dataloader      : workers={args.num_workers} "
        f"persistent={args.persistent_workers} prefetch={args.prefetch_factor}"
    )
    print(f"[Info] stats chunk     : {args.stats_windows_chunk_size}")
    if not args.load_into_memory and np.isfinite(db_size_gb):
        print(
            "[Tip] if RAM is sufficient, add --load-into-memory for faster training throughput "
            f"(needs about {db_size_gb + 2.0:.1f} GB memory for X + overhead)."
        )

    write_config_json(out_dir=args.out_dir, args=args)

    prepared = prepare_data_loaders_from_args(
        args=args,
        test_loader_note="prepared (will evaluate after training)",
    )
    train_loader = prepared.train_loader
    val_loader = prepared.val_loader
    test_loader = prepared.test_loader

    model = MVAE375to291(
        input_dim=args.input_dim,
        traj_dim=args.traj_dim,
        motion_dim=args.motion_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        num_condition_frames=args.num_condition_frames,
        num_future_predictions=args.num_future_predictions,
        num_experts=args.num_experts,
    ).to(device)

    if args.compile:
        if not hasattr(torch, "compile"):
            print("[Warn] torch.compile is not available in this torch build; skipping --compile.")
        else:
            model = torch.compile(
                model,
                mode=args.compile_mode,
                fullgraph=bool(args.compile_fullgraph),
            )
            print(
                f"[Info] compile         : enabled (mode={args.compile_mode}, fullgraph={args.compile_fullgraph})"
            )

    optimizer, use_fused_adamw = build_adamw(
        model.parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=bool(args.fused_adamw and device.type == "cuda"),
    )
    print(f"[Info] optimizer       : AdamW(fused={use_fused_adamw})")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16 and device.type == "cuda")
    scaler = None
    if device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            # PyTorch >= 2.4 preferred API.
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            # Backward compatibility for older torch builds.
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_metric = float("inf")
    history_path = args.out_dir / "history.jsonl"
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

    if start_epoch > args.epochs:
        print(
            f"[Warn] start_epoch ({start_epoch}) > epochs ({args.epochs}). "
            "Nothing to train. Increase --epochs or use another checkpoint."
        )
        return

    future_weights = torch.ones(
        (args.num_future_predictions,),
        device=device,
        dtype=torch.float32,
    ).div_(float(args.num_future_predictions))

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        student_prob = compute_student_prob(
            epoch,
            teacher_epochs=args.teacher_epochs,
            ramp_epochs=args.ramp_epochs,
        )
        print(
            f"\n[Epoch {epoch}/{args.epochs}] lr={lr_now:.8f} "
            f"student_prob={student_prob:.3f}"
        )

        train_metrics = epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            recon_type=args.recon_type,
            recon_weight=args.recon_weight,
            kl_weight=args.kl_weight,
            rollout_steps=args.rollout_steps,
            student_prob=student_prob,
            future_weights=future_weights,
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
                        recon_weight=args.recon_weight,
                        kl_weight=args.kl_weight,
                        rollout_steps=args.rollout_steps,
                        student_prob=args.val_student_prob,
                        future_weights=future_weights,
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

        scheduler.step()

        epoch_sec = time.time() - epoch_t0
        msg = (
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"train_kl={train_metrics['kl_loss']:.6f} "
            f"train_roll={train_metrics['avg_rollout_steps']:.2f} "
            f"train_sp={train_metrics['student_prob']:.3f}"
        )
        if val_metrics is not None:
            msg += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"val_kl={val_metrics['kl_loss']:.6f} "
                f"val_roll={val_metrics['avg_rollout_steps']:.2f} "
                f"val_sp={val_metrics['student_prob']:.3f}"
            )
        msg += f" | time={epoch_sec:.1f}s"
        print(msg)

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
            "time_sec": epoch_sec,
        }
        append_jsonl(path=history_path, record=record)

        save_training_checkpoint(
            ckpt_path=args.out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            args=args,
        )

        if metric_for_best is not None and metric_for_best < best_metric:
            best_metric = metric_for_best
            save_training_checkpoint(
                ckpt_path=args.out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )
            print(f"[Info] saved best checkpoint (metric={best_metric:.6f})")

        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_training_checkpoint(
                ckpt_path=args.out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )

    if test_loader is not None:
        best_ckpt_path = args.out_dir / "best.pt"
        load_best_checkpoint_for_eval(
            model=model,
            ckpt_path=best_ckpt_path,
            allow_incompatible=True,
        )

        with torch.no_grad():
            test_metrics = epoch_pass(
                model=model,
                loader=test_loader,
                device=device,
                recon_type=args.recon_type,
                recon_weight=args.recon_weight,
                kl_weight=args.kl_weight,
                rollout_steps=args.rollout_steps,
                student_prob=args.val_student_prob,
                future_weights=future_weights,
                optimizer=None,
                scaler=None,
                grad_clip=0.0,
                grad_accum_steps=1,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                log_interval=args.log_interval,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        print(
            "[Test] "
            f"loss={test_metrics['loss']:.6f} "
            f"recon={test_metrics['recon_loss']:.6f} "
            f"kl={test_metrics['kl_loss']:.6f} "
            f"roll={test_metrics['avg_rollout_steps']:.2f} "
            f"sp={test_metrics['student_prob']:.3f}"
        )
        append_jsonl(
            path=history_path,
            record={"epoch": "test", "split": "test", "metrics": test_metrics},
        )

    print("\n[Done] Training complete.")
    print(f"[Done] Best metric: {best_metric:.6f}")
    print(f"[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
