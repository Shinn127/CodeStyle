"""Throughput benchmark: separates data-pipeline cost from model cost.

Run: python bench_train_throughput.py [--num-workers 8] [--batch-size 512] [--batches 60]
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time

import torch

from dataset import build_train_val_test_dataloaders
from Utils.runtime import configure_torch_runtime, resolve_amp_dtype, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="database_100sty_375_memmap_nomirror")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--load-into-memory", action="store_true")
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--amp-dtype", default="bf16", choices=["auto", "fp16", "bf16", "none"])
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--matmul-precision", default="high", choices=["highest", "high", "medium"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device, tf32=args.tf32, cudnn_benchmark=True, matmul_precision=args.matmul_precision
    )

    bundle = build_train_val_test_dataloaders(
        db_path=args.db,
        seq_len=2,
        stride=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        normalize=True,
        split_mode="auto",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
    )
    loader = bundle.train_loader
    print(f"[Info] batches/epoch   : {len(loader)}  (batch={args.batch_size}, workers={args.num_workers})")
    print(f"[Info] split           : {bundle.split_mode} / {bundle.split_source}")
    print(f"[Info] amp             : {args.amp_dtype}  tf32={args.tf32}")

    # ---- data pipeline only -------------------------------------------------
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)
    start = time.time()
    seen = 0
    for _ in range(args.batches):
        batch = next(iterator)
        seen += batch["x"].shape[0]
    data_only = seen / (time.time() - start)
    print(f"[Data ] {data_only:10.1f} samples/s   ({1000 * args.batches / (time.time() - start):.2f} ms/batch)")

    # ---- model step ---------------------------------------------------------
    from arch.basic_vqvae_375to291 import BasicVQVAE375to291
    from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: F401  (import check only)

    model = BasicVQVAE375to291(
        input_dim=375,
        traj_dim=84,
        motion_dim=291,
        input_mode="traj_motion",
        encoder_arch="skip_cat_mlp",
        decoder_arch="skip_cat_mlp",
        hidden_dim=512,
        encoder_layers=6,
        decoder_hidden_dim=512,
        decoder_layers=6,
        num_latents=32,
        code_dim=4,
        num_embeddings=2048,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=(device.type == "cuda"))

    batch = next(iter(loader))
    x = batch["x"].to(device, non_blocking=True).float()
    target = batch["motion"].to(device, non_blocking=True).float()
    amp_dtype = None if args.amp_dtype == "none" else resolve_amp_dtype(args.amp_dtype, device=device)

    def one_step() -> float:
        import contextlib

        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
            if device.type == "cuda" and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with ctx:
            loss = model.compute_loss(x=x, target_motion=target, recon_weight=1.0, vq_weight=1.0, vel_weight=3.0)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    for _ in range(10):
        one_step()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.batches):
        one_step()
    torch.cuda.synchronize()
    model_only = args.batches * args.batch_size / (time.time() - start)
    print(f"[Model] {model_only:10.1f} samples/s   (cached batch, {1000 * (time.time() - start) / args.batches:.2f} ms/step)")

    # ---- combined ------------------------------------------------------------
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)
    torch.cuda.synchronize()
    start = time.time()
    seen = 0
    for _ in range(args.batches):
        batch = next(iterator)
        x = batch["x"].to(device, non_blocking=True).float()
        target = batch["motion"].to(device, non_blocking=True).float()
        import contextlib

        ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
            if device.type == "cuda" and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with ctx:
            loss = model.compute_loss(x=x, target_motion=target, recon_weight=1.0, vq_weight=1.0, vel_weight=3.0)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        seen += batch["x"].shape[0]
    torch.cuda.synchronize()
    combined = seen / (time.time() - start)
    print(f"[Total] {combined:10.1f} samples/s   <- end-to-end, bottleneck = "
          f"{'data' if data_only < model_only else 'model'}")
    print(
        f"[Note ] GPU util ceiling from data alone: {100 * min(1.0, model_only / data_only):.0f}%"
        f" | model alone is {model_only / data_only:.2f}x the data rate"
    )


if __name__ == "__main__":
    main()
