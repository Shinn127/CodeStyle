"""
Training script for LMM375to291:
  input(375) = [traj(84), motion(291)]
  motion(291) -> compressor -> latent(32)
  [traj(84), latent(32)] -> decompressor -> motion(291)

Motion-Matching-main alignment applied here:
  - training samples are fixed to contiguous 2-frame windows
  - network hidden size / latent size are fixed to the local train_decompressor.py setup

Usage example:
  python NewCode/train_lmm_375to291.py \
    --db NewCode/database_100sty_375_memmap_nomirror \
    --out-dir NewCode/checkpoints/lmm_375to291 \
    --seq-len 2 --stride 1 --batch-size 512 --epochs 50
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
from arch import LMM375to291


LMM_WINDOW = 2
LMM_STRIDE = 1
LMM_LATENT_DIM = 32
LMM_HIDDEN_DIM = 512


def epoch_pass(
    *,
    model: LMM375to291,
    loader,
    device: torch.device,
    recon_type: str,
    recon_weight: float,
    pos_weight: float,
    rot_weight: float,
    vel_weight: float,
    root_vel_weight: float,
    root_ang_weight: float,
    pos_delta_weight: float,
    rot_delta_weight: float,
    latent_l1_weight: float,
    latent_l2_weight: float,
    latent_vel_weight: float,
    dt: float,
    motion_mean: Optional[torch.Tensor] = None,
    motion_std: Optional[torch.Tensor] = None,
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
    metric_keys = [
        "loss",
        "recon_loss",
        "pos_loss",
        "rot_loss",
        "vel_loss",
        "root_vel_loss",
        "root_ang_loss",
        "pos_delta_loss",
        "rot_delta_loss",
        "latent_l1_loss",
        "latent_l2_loss",
        "latent_vel_loss",
    ]
    metric_sums = {k: torch.zeros((), device=device, dtype=torch.float32) for k in metric_keys}
    n_samples = 0
    n_updates = 0
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        motion = batch["motion"].to(device, non_blocking=True).float()
        traj = batch["traj"].to(device, non_blocking=True).float()
        bs = motion.shape[0]

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
            loss_dict = model.compute_loss(
                motion=motion,
                traj=traj,
                recon_type=recon_type,
                recon_weight=recon_weight,
                pos_weight=pos_weight,
                rot_weight=rot_weight,
                vel_weight=vel_weight,
                root_vel_weight=root_vel_weight,
                root_ang_weight=root_ang_weight,
                pos_delta_weight=pos_delta_weight,
                rot_delta_weight=rot_delta_weight,
                latent_l1_weight=latent_l1_weight,
                latent_l2_weight=latent_l2_weight,
                latent_vel_weight=latent_vel_weight,
                dt=dt,
                motion_mean=motion_mean,
                motion_std=motion_std,
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

        for k in metric_keys:
            metric_sums[k] = metric_sums[k] + loss_dict[k].detach().to(torch.float32) * float(bs)
        n_samples += bs

        if is_train and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            samples_per_sec = n_samples / max(elapsed, 1e-6)
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={to_float(loss_dict['loss']):.6f} "
                f"recon={to_float(loss_dict['recon_loss']):.6f} "
                f"pos={to_float(loss_dict['pos_loss']):.6f} "
                f"rot={to_float(loss_dict['rot_loss']):.6f} "
                f"vel={to_float(loss_dict['vel_loss']):.6f} "
                f"rvel={to_float(loss_dict['root_vel_loss']):.6f} "
                f"rang={to_float(loss_dict['root_ang_loss']):.6f} "
                f"pdel={to_float(loss_dict['pos_delta_loss']):.6f} "
                f"rdel={to_float(loss_dict['rot_delta_loss']):.6f} "
                f"l1={to_float(loss_dict['latent_l1_loss']):.6f} "
                f"l2={to_float(loss_dict['latent_l2_loss']):.6f} "
                f"lvel={to_float(loss_dict['latent_vel_loss']):.6f} "
                f"updates={n_updates} "
                f"{samples_per_sec:.1f} samples/s "
                f"time={elapsed:.1f}s"
            )

    if n_samples == 0:
        return {
            "loss": float("nan"),
            "recon_loss": float("nan"),
            "pos_loss": float("nan"),
            "rot_loss": float("nan"),
            "vel_loss": float("nan"),
            "root_vel_loss": float("nan"),
            "root_ang_loss": float("nan"),
            "pos_delta_loss": float("nan"),
            "rot_delta_loss": float("nan"),
            "latent_l1_loss": float("nan"),
            "latent_l2_loss": float("nan"),
            "latent_vel_loss": float("nan"),
            "num_samples": 0.0,
            "num_updates": 0.0,
        }

    mean_metrics = {k: to_float(v / float(n_samples)) for k, v in metric_sums.items()}
    return {
        "loss": mean_metrics["loss"],
        "recon_loss": mean_metrics["recon_loss"],
        "pos_loss": mean_metrics["pos_loss"],
        "rot_loss": mean_metrics["rot_loss"],
        "vel_loss": mean_metrics["vel_loss"],
        "root_vel_loss": mean_metrics["root_vel_loss"],
        "root_ang_loss": mean_metrics["root_ang_loss"],
        "pos_delta_loss": mean_metrics["pos_delta_loss"],
        "rot_delta_loss": mean_metrics["rot_delta_loss"],
        "latent_l1_loss": mean_metrics["latent_l1_loss"],
        "latent_l2_loss": mean_metrics["latent_l2_loss"],
        "latent_vel_loss": mean_metrics["latent_vel_loss"],
        "num_samples": float(n_samples),
        "num_updates": float(n_updates),
    }


def validate_lmm_alignment_args(args: argparse.Namespace) -> None:
    if int(args.seq_len) != LMM_WINDOW:
        raise ValueError(
            f"LMM375to291 training must use seq_len={LMM_WINDOW} to match "
            "Motion-Matching-main/resources/train_decompressor.py"
        )
    if int(args.stride) != LMM_STRIDE:
        raise ValueError(
            f"LMM375to291 training must use stride={LMM_STRIDE} to match "
            "Motion-Matching-main/resources/train_decompressor.py"
        )
    if int(args.latent_dim) != LMM_LATENT_DIM:
        raise ValueError(
            f"LMM375to291 latent_dim must be {LMM_LATENT_DIM} to match "
            "Motion-Matching-main/resources/train_decompressor.py"
        )
    if int(args.hidden_dim) != LMM_HIDDEN_DIM:
        raise ValueError(
            f"LMM375to291 hidden_dim must be {LMM_HIDDEN_DIM} to match "
            "Motion-Matching-main/resources/train_decompressor.py"
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train LMM375to291 on 100STYLE memmap DB")
    add_common_io_args(parser, script_dir=script_dir, out_dir_name="lmm_375to291")
    add_common_data_args(
        parser,
        defaults=CommonDataDefaults(
            seq_len=LMM_WINDOW,
            stride=LMM_STRIDE,
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

    parser.add_argument("--motion-dim", type=int, default=291)
    parser.add_argument("--traj-dim", type=int, default=84)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=LMM_LATENT_DIM,
        help="Fixed to 32 to match Motion-Matching-main/resources/train_decompressor.py",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=LMM_HIDDEN_DIM,
        help="Fixed to 512 to match Motion-Matching-main/resources/train_decompressor.py",
    )

    parser.add_argument("--recon-type", type=str, choices=["l1", "mse", "smooth_l1"], default="l1")
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=15.0)
    parser.add_argument("--rot-weight", type=float, default=5.0)
    parser.add_argument("--vel-weight", type=float, default=2.0)
    parser.add_argument("--root-vel-weight", type=float, default=2.0)
    parser.add_argument("--root-ang-weight", type=float, default=2.0)
    parser.add_argument("--pos-delta-weight", type=float, default=2.0)
    parser.add_argument("--rot-delta-weight", type=float, default=0.75)
    parser.add_argument("--latent-l1-weight", type=float, default=0.1)
    parser.add_argument("--latent-l2-weight", type=float, default=0.1)
    parser.add_argument("--latent-vel-weight", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)

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
    args.model_name = "LMM375to291"
    args.model_type = "lmm_375to291"
    validate_lmm_alignment_args(args)
    validate_common_data_args(
        grad_accum_steps=args.grad_accum_steps,
        val_every=args.val_every,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"[Info] model           : {args.model_name}")
    print(f"[Info] device          : {device}")
    print(f"[Info] amp             : {amp_enabled} ({args.amp_dtype})")
    print(
        f"[Info] backend         : tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out_dir         : {args.out_dir}")
    print(f"[Info] sample window   : seq_len={args.seq_len} stride={args.stride}")
    db_size_gb = estimate_path_size_bytes(args.db) / (1024.0 ** 3) if args.db.exists() else float("nan")
    print(f"[Info] db size         : {db_size_gb:.2f} GB")
    print(
        f"[Info] losses          : recon={args.recon_weight} pos={args.pos_weight} rot={args.rot_weight} "
        f"vel={args.vel_weight} root_vel={args.root_vel_weight} root_ang={args.root_ang_weight} "
        f"pos_delta={args.pos_delta_weight} rot_delta={args.rot_delta_weight} "
        f"latent_l1={args.latent_l1_weight} latent_l2={args.latent_l2_weight} latent_vel={args.latent_vel_weight}"
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
    motion_mean = None
    motion_std = None
    if prepared.bundle.norm_mean is not None and prepared.bundle.norm_std is not None:
        motion_mean = torch.as_tensor(
            prepared.bundle.norm_mean[args.traj_dim :],
            device=device,
            dtype=torch.float32,
        )
        motion_std = torch.as_tensor(
            prepared.bundle.norm_std[args.traj_dim :],
            device=device,
            dtype=torch.float32,
        )

    model = LMM375to291(
        motion_dim=args.motion_dim,
        traj_dim=args.traj_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
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

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={lr_now:.8f}")

        train_metrics = epoch_pass(
            model=model,
            loader=train_loader,
            device=device,
            recon_type=args.recon_type,
            recon_weight=args.recon_weight,
            pos_weight=args.pos_weight,
            rot_weight=args.rot_weight,
            vel_weight=args.vel_weight,
            root_vel_weight=args.root_vel_weight,
            root_ang_weight=args.root_ang_weight,
            pos_delta_weight=args.pos_delta_weight,
            rot_delta_weight=args.rot_delta_weight,
            latent_l1_weight=args.latent_l1_weight,
            latent_l2_weight=args.latent_l2_weight,
            latent_vel_weight=args.latent_vel_weight,
            dt=args.dt,
            motion_mean=motion_mean,
            motion_std=motion_std,
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
                        pos_weight=args.pos_weight,
                        rot_weight=args.rot_weight,
                        vel_weight=args.vel_weight,
                        root_vel_weight=args.root_vel_weight,
                        root_ang_weight=args.root_ang_weight,
                        pos_delta_weight=args.pos_delta_weight,
                        rot_delta_weight=args.rot_delta_weight,
                        latent_l1_weight=args.latent_l1_weight,
                        latent_l2_weight=args.latent_l2_weight,
                        latent_vel_weight=args.latent_vel_weight,
                        dt=args.dt,
                        motion_mean=motion_mean,
                        motion_std=motion_std,
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
            f"train_pos={train_metrics['pos_loss']:.6f} "
            f"train_rot={train_metrics['rot_loss']:.6f} "
            f"train_vel={train_metrics['vel_loss']:.6f} "
            f"train_rvel={train_metrics['root_vel_loss']:.6f} "
            f"train_rang={train_metrics['root_ang_loss']:.6f} "
            f"train_pdel={train_metrics['pos_delta_loss']:.6f} "
            f"train_rdel={train_metrics['rot_delta_loss']:.6f} "
            f"train_l1={train_metrics['latent_l1_loss']:.6f} "
            f"train_l2={train_metrics['latent_l2_loss']:.6f} "
            f"train_lvel={train_metrics['latent_vel_loss']:.6f}"
        )
        if val_metrics is not None:
            msg += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"val_pos={val_metrics['pos_loss']:.6f} "
                f"val_rot={val_metrics['rot_loss']:.6f} "
                f"val_vel={val_metrics['vel_loss']:.6f} "
                f"val_rvel={val_metrics['root_vel_loss']:.6f} "
                f"val_rang={val_metrics['root_ang_loss']:.6f} "
                f"val_pdel={val_metrics['pos_delta_loss']:.6f} "
                f"val_rdel={val_metrics['rot_delta_loss']:.6f} "
                f"val_l1={val_metrics['latent_l1_loss']:.6f} "
                f"val_l2={val_metrics['latent_l2_loss']:.6f} "
                f"val_lvel={val_metrics['latent_vel_loss']:.6f}"
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
                pos_weight=args.pos_weight,
                rot_weight=args.rot_weight,
                vel_weight=args.vel_weight,
                root_vel_weight=args.root_vel_weight,
                root_ang_weight=args.root_ang_weight,
                pos_delta_weight=args.pos_delta_weight,
                rot_delta_weight=args.rot_delta_weight,
                latent_l1_weight=args.latent_l1_weight,
                latent_l2_weight=args.latent_l2_weight,
                latent_vel_weight=args.latent_vel_weight,
                dt=args.dt,
                motion_mean=motion_mean,
                motion_std=motion_std,
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
            f"pos={test_metrics['pos_loss']:.6f} "
            f"rot={test_metrics['rot_loss']:.6f} "
            f"vel={test_metrics['vel_loss']:.6f} "
            f"rvel={test_metrics['root_vel_loss']:.6f} "
            f"rang={test_metrics['root_ang_loss']:.6f} "
            f"pdel={test_metrics['pos_delta_loss']:.6f} "
            f"rdel={test_metrics['rot_delta_loss']:.6f} "
            f"l1={test_metrics['latent_l1_loss']:.6f} "
            f"l2={test_metrics['latent_l2_loss']:.6f} "
            f"lvel={test_metrics['latent_vel_loss']:.6f}"
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
