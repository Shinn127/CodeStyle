"""
Training script for a latent diffusion VQ-VAE controller that predicts x0.

The controller reuses the same conditional latent Transformer structure as the
flow-matching VQ controller, but changes the training target to denoised latent
prediction and uses DDIM-style skipped-step sampling at inference.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from arch import DiffusionVQVAEController, build_diffusion_vqvae_controller_from_config
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import (
    configure_torch_runtime,
    load_checkpoint,
    load_norm_stats,
    resolve_device,
    sanitize_state_dict_keys,
    save_norm_stats_if_present,
    set_seed,
    require_unmirrored_db,
)
from Utils.style_code_prior import load_style_code_prior
from Utils.vq_target_cache import VQTargetCache
from dataset import build_train_val_test_dataloaders
from training.train_flowmatch_vqvae_controller import (
    _append_jsonl,
    _build_latent_stats_payload,
    _estimate_raw_latent_stats,
    _flatten_adjacent_pairs,
    _load_latent_stats,
    _move_optimizer_state_to_device,
    _save_json,
    _save_latent_stats,
    tqdm,
)


def _metric_keys() -> list[str]:
    return [
        "loss",
        "diffusion_loss",
        "recon_loss",
        "style_hist_loss",
        "style_entropy",
        "style_top1_prob",
        "latent_target_norm",
        "noisy_latent_norm",
        "pred_x0_norm",
    ]


def _run_epoch(
    *,
    model: DiffusionVQVAEController,
    loader,
    device: torch.device,
    recon_weight: float,
    diffusion_weight: float,
    style_hist_weight: float,
    style_entropy_weight: float,
    prev_latent_noise_std: float,
    prev_motion_noise_std: float,
    prev_latent_dropout: float,
    prev_motion_dropout: float,
    sample_steps: int,
    sample_noise_std: float,
    optimizer: AdamW | None,
    vq_target_cache: VQTargetCache | None,
    cache_vqvae,
    grad_clip: float,
    log_interval: int,
    epoch: int,
    total_epochs: int,
    split: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    keys = _metric_keys()
    metric_sums = {key: 0.0 for key in keys}
    num_samples = 0
    start_time = time.time()
    total_steps = len(loader)
    desc = f"{split} {epoch}/{total_epochs}"
    log_interval = max(int(log_interval), 1)

    with tqdm(total=total_steps, desc=desc, unit="batch", dynamic_ncols=True) as progress:
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True).float()
            if x.dim() != 3 or x.size(1) < 2:
                raise ValueError(f"Expected x shape [B,T,375] with T>=2, got {tuple(x.shape)}")
            style = batch["label"].to(device, non_blocking=True).long()
            start = batch["start"].to(device, non_blocking=True).long()
            x_curr, x_next, style_bt, curr_ids, next_ids = _flatten_adjacent_pairs(x, style, start)
            prev_latent = None
            target_latent = None
            if vq_target_cache is not None:
                prev_latent = vq_target_cache.lookup_latents(curr_ids, vqvae=cache_vqvae, device=device)
                target_latent = vq_target_cache.lookup_latents(next_ids, vqvae=cache_vqvae, device=device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            loss_dict = model.compute_loss(
                x_curr=x_curr,
                x_next=x_next,
                style_label=style_bt,
                prev_latent=prev_latent,
                target_latent=target_latent,
                diffusion_weight=diffusion_weight,
                recon_weight=recon_weight,
                style_hist_weight=style_hist_weight,
                style_entropy_weight=style_entropy_weight,
                prev_latent_noise_std=prev_latent_noise_std,
                prev_motion_noise_std=prev_motion_noise_std,
                prev_latent_dropout=prev_latent_dropout,
                prev_motion_dropout=prev_motion_dropout,
                sample_steps=sample_steps,
                sample_noise_std=sample_noise_std,
            )
            loss_value = loss_dict["loss"]

            if is_train:
                loss_value.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_pairs = int(x_curr.size(0))
            num_samples += batch_pairs
            for key in keys:
                metric_sums[key] += float(loss_dict[key].detach().cpu()) * batch_pairs

            progress.update(1)
            if step % log_interval == 0 or step == total_steps:
                elapsed = time.time() - start_time
                samples_per_sec = num_samples / max(elapsed, 1e-6)
                progress.set_postfix(
                    loss=f"{float(loss_value.detach().cpu()):.4f}",
                    diff=f"{float(loss_dict['diffusion_loss'].detach().cpu()):.4f}",
                    recon=f"{float(loss_dict['recon_loss'].detach().cpu()):.4f}",
                    style=f"{float(loss_dict['style_hist_loss'].detach().cpu()):.4f}",
                    rate=f"{samples_per_sec:.1f}/s",
                )

    metrics = {key: metric_sums[key] / max(num_samples, 1) for key in keys}
    metrics["num_samples"] = float(num_samples)
    return metrics


def _save_checkpoint(
    *,
    path: Path,
    model: DiffusionVQVAEController,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Latent diffusion VQ-VAE controller trainer")
    parser.add_argument("--db", type=Path, default=script_dir / "database_100sty_375_memmap_nomirror")
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048" / "best.pt",
    )
    parser.add_argument("--vq-target-cache", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--latent-stats", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=script_dir / "checkpoints" / "diffusion_vqvae_controller")
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-into-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-same-label", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-styles", type=int, default=0)
    parser.add_argument("--use-style-code-selector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--style-code-prior", type=Path, default=None)
    parser.add_argument("--style-num-slots", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--style-selector-hidden-dim", type=int, default=128)
    parser.add_argument("--style-selector-topk", type=int, default=4)
    parser.add_argument("--style-selector-temperature", type=float, default=1.0)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--flow-heads", type=int, default=4)
    parser.add_argument("--flow-ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target-time-eps", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--beta-schedule", type=str, choices=["linear", "cosine"], default="cosine")

    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--style-hist-weight", type=float, default=0.2)
    parser.add_argument("--style-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--prev-latent-noise-std", type=float, default=0.3)
    parser.add_argument("--prev-motion-noise-std", type=float, default=0.3)
    parser.add_argument("--prev-latent-dropout", type=float, default=0.0)
    parser.add_argument("--prev-motion-dropout", type=float, default=0.0)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--sample-noise-std", type=float, default=1.0)
    parser.add_argument("--latent-stats-chunk-rows", type=int, default=65536)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.vqvae_ckpt = args.vqvae_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.resume = None if args.resume is None else args.resume.expanduser().resolve()
    args.vq_target_cache = None if args.vq_target_cache is None else args.vq_target_cache.expanduser().resolve()
    args.norm_stats = None if args.norm_stats is None else args.norm_stats.expanduser().resolve()
    args.latent_stats = (args.out_dir / "latent_stats.npz") if args.latent_stats is None else args.latent_stats.expanduser().resolve()
    args.style_code_prior = None if args.style_code_prior is None else args.style_code_prior.expanduser().resolve()
    args.model_type = "diffusion_vqvae_controller"
    args.base_model_type = "vqvae"

    require_unmirrored_db(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ checkpoint not found: {args.vqvae_ckpt}")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.sample_steps < 1:
        raise ValueError("--sample-steps must be >= 1")
    if args.diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be >= 1")
    if args.sample_steps > args.diffusion_steps:
        raise ValueError("--sample-steps must be <= --diffusion-steps")
    if args.style_num_slots < 1:
        raise ValueError("--style-num-slots must be >= 1")
    if args.style_selector_topk < 1:
        raise ValueError("--style-selector-topk must be >= 1")
    if args.style_selector_temperature <= 0.0:
        raise ValueError("--style-selector-temperature must be > 0")
    if not args.normalize:
        raise ValueError("Latent diffusion training expects normalized inputs.")
    if args.use_style_code_selector and args.style_code_prior is None:
        raise ValueError("--style-code-prior is required when --use-style-code-selector is enabled.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    vq_target_cache = None
    if args.vq_target_cache is not None:
        vq_target_cache = VQTargetCache(args.vq_target_cache)
        if args.norm_stats is None:
            args.norm_stats = vq_target_cache.meta.norm_stats
    fixed_norm_mean = None
    fixed_norm_std = None
    if args.norm_stats is not None:
        fixed_norm_mean, fixed_norm_std = load_norm_stats(args.norm_stats)

    bundle = build_train_val_test_dataloaders(
        db_path=args.db,
        seq_len=args.seq_len,
        stride=args.stride,
        enforce_same_label=args.enforce_same_label,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        load_into_memory=args.load_into_memory,
        normalize=args.normalize,
        norm_mean=fixed_norm_mean,
        norm_std=fixed_norm_std,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    norm_stats_path = save_norm_stats_if_present(
        out_dir=args.out_dir,
        norm_mean=bundle.norm_mean,
        norm_std=bundle.norm_std,
    )
    if args.norm_stats is None:
        args.norm_stats = norm_stats_path

    style_names = getattr(bundle.train_dataset, "style_names", None)
    if args.num_styles > 0:
        num_styles = int(args.num_styles)
    elif style_names:
        num_styles = len(style_names)
    else:
        labels = getattr(bundle.train_dataset, "labels", None)
        num_styles = int(np.max(labels)) + 1

    print(f"[Info] device          : {device}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out dir         : {args.out_dir}")
    print(f"[Info] vq ckpt         : {args.vqvae_ckpt}")
    print(f"[Info] diffusion steps : {args.diffusion_steps}")
    print(f"[Info] sample steps    : {args.sample_steps}")
    print(f"[Info] num styles      : {num_styles}")
    if args.norm_stats is not None:
        print(f"[Info] norm stats      : {args.norm_stats}")
    if args.vq_target_cache is not None:
        print(f"[Info] vq target cache : {args.vq_target_cache}")

    vq_model = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)
    if vq_target_cache is not None:
        vq_target_cache.validate(
            db=args.db,
            vqvae_ckpt=args.vqvae_ckpt,
            norm_stats=args.norm_stats,
            vqvae=vq_model,
        )

    style_prior_payload = None
    if args.use_style_code_selector:
        style_prior_payload = load_style_code_prior(args.style_code_prior)
        style_prior = style_prior_payload["prior"]
        if style_prior.shape[0] != num_styles:
            raise ValueError(
                f"style code prior num_styles mismatch: prior={style_prior.shape[0]}, dataset={num_styles}"
            )
        selected_slots = [int(slot) for slot in np.asarray(style_prior_payload["selected_slots"], dtype=np.int64).tolist()]
        if args.style_num_slots != len(selected_slots):
            print(
                f"[Warn] style_num_slots={args.style_num_slots} overridden by prior selected_slots={len(selected_slots)}"
            )
        args.style_num_slots = len(selected_slots)
        args.style_slot_indices = selected_slots
        print(f"[Info] style prior     : {args.style_code_prior}")
        print(f"[Info] style slots     : {selected_slots}")
    else:
        args.style_slot_indices = []

    model = build_diffusion_vqvae_controller_from_config(
        vars(args),
        vqvae=vq_model,
        num_styles=num_styles,
    ).to(device)
    if style_prior_payload is not None:
        model.set_style_code_prior(style_prior_payload["prior"])
    if args.latent_stats.exists():
        latent_stats = _load_latent_stats(args.latent_stats, latent_dim=model.latent_dim)
    else:
        raw_mean, raw_std, raw_min, raw_max = _estimate_raw_latent_stats(
            db_path=args.db,
            vqvae=vq_model,
            device=device,
            norm_mean=bundle.norm_mean,
            norm_std=bundle.norm_std,
            vq_target_cache=vq_target_cache,
            chunk_rows=args.latent_stats_chunk_rows,
        )
        latent_stats = _build_latent_stats_payload(
            raw_mean=raw_mean,
            raw_std=raw_std,
            raw_min=raw_min,
            raw_max=raw_max,
        )
        _save_latent_stats(args.latent_stats, latent_stats)
    model.set_latent_prior_stats(
        latent_stats["offset"],
        latent_stats["scale"],
        transport_std=latent_stats["transport_std"],
        data_min=latent_stats["data_min"],
        data_max=latent_stats["data_max"],
    )
    print(f"[Info] latent stats    : {args.latent_stats}")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in diffusion controller.")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch = 1
    best_metric = float("inf")
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[Info] resume         : {args.resume} (epoch {start_epoch})")

    _save_json(args.out_dir / "config.json", vars(args))
    history_path = args.out_dir / "history.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.8f}")
        train_metrics = _run_epoch(
            model=model,
            loader=bundle.train_loader,
            device=device,
            recon_weight=args.recon_weight,
            diffusion_weight=args.diffusion_weight,
            style_hist_weight=args.style_hist_weight,
            style_entropy_weight=args.style_entropy_weight,
            prev_latent_noise_std=args.prev_latent_noise_std,
            prev_motion_noise_std=args.prev_motion_noise_std,
            prev_latent_dropout=args.prev_latent_dropout,
            prev_motion_dropout=args.prev_motion_dropout,
            sample_steps=args.sample_steps,
            sample_noise_std=args.sample_noise_std,
            optimizer=optimizer,
            vq_target_cache=vq_target_cache,
            cache_vqvae=vq_model,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )

        val_metrics = None
        metric_for_best = train_metrics["loss"]
        if bundle.val_loader is not None and epoch % args.val_every == 0:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model=model,
                    loader=bundle.val_loader,
                    device=device,
                    recon_weight=args.recon_weight,
                    diffusion_weight=args.diffusion_weight,
                    style_hist_weight=args.style_hist_weight,
                    style_entropy_weight=args.style_entropy_weight,
                    prev_latent_noise_std=0.0,
                    prev_motion_noise_std=0.0,
                    prev_latent_dropout=0.0,
                    prev_motion_dropout=0.0,
                    sample_steps=args.sample_steps,
                    sample_noise_std=args.sample_noise_std,
                    optimizer=None,
                    vq_target_cache=vq_target_cache,
                    cache_vqvae=vq_model,
                    grad_clip=0.0,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    split="val",
                )
            metric_for_best = val_metrics["loss"]

        scheduler.step()
        epoch_sec = time.time() - epoch_t0
        _append_jsonl(
            history_path,
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "time_sec": epoch_sec,
            },
        )
        print(
            f"[Epoch {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"train_diff={train_metrics['diffusion_loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f}"
            + (
                ""
                if val_metrics is None
                else f" | val_loss={val_metrics['loss']:.6f} val_diff={val_metrics['diffusion_loss']:.6f} val_recon={val_metrics['recon_loss']:.6f}"
            )
            + f" | time={epoch_sec:.1f}s"
        )

        _save_checkpoint(
            path=args.out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            args=args,
        )
        if metric_for_best <= best_metric:
            best_metric = metric_for_best
            _save_checkpoint(
                path=args.out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )
        if args.save_every > 0 and (epoch % args.save_every == 0):
            _save_checkpoint(
                path=args.out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )

    if bundle.test_loader is not None and (args.out_dir / "best.pt").exists():
        best_ckpt = load_checkpoint(args.out_dir / "best.pt", map_location="cpu")
        model.load_state_dict(sanitize_state_dict_keys(best_ckpt["model"]), strict=True)
        with torch.no_grad():
            test_metrics = _run_epoch(
                model=model,
                loader=bundle.test_loader,
                device=device,
                recon_weight=args.recon_weight,
                diffusion_weight=args.diffusion_weight,
                style_hist_weight=args.style_hist_weight,
                style_entropy_weight=args.style_entropy_weight,
                prev_latent_noise_std=0.0,
                prev_motion_noise_std=0.0,
                prev_latent_dropout=0.0,
                prev_motion_dropout=0.0,
                sample_steps=args.sample_steps,
                sample_noise_std=args.sample_noise_std,
                optimizer=None,
                vq_target_cache=vq_target_cache,
                cache_vqvae=vq_model,
                grad_clip=0.0,
                log_interval=args.log_interval,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        print(
            f"[Test] loss={test_metrics['loss']:.6f} "
            f"diff={test_metrics['diffusion_loss']:.6f} "
            f"recon={test_metrics['recon_loss']:.6f}"
        )
        _append_jsonl(history_path, {"epoch": "test", "split": "test", "metrics": test_metrics})

    print(f"\n[Done] best metric: {best_metric:.6f}")
    print(f"[Done] outputs    : {args.out_dir}")


if __name__ == "__main__":
    main()
