#!/usr/bin/env python
"""Train the FID feature extractor with a *reconstruction* objective on the full 100STYLE set.

This is the "motion reconstruction objective" flavour of `train_fid_feature_extractor_camdm.py`:
same data (all 100 styles, every clip, no split), same conv trunk and same 512-dim feature width
as CAMDM's classifier, same optimisation schedule -- but the head reconstructs the input window
instead of predicting a style, so no style labels are used at all.

    x [B, 45, 60] -> trunk (conv 64/128 + residuals) -> FC 1024 -> FC 512  (FID feature)
                                                   -> decoder -> x_hat [B, 45, 60]

Loss: reconstruction error on the 45-dim local positions, plus an optional temporal-difference
term (`--vel-weight`) which keeps the code sensitive to motion dynamics rather than to a static
pose. The FID feature is the 512-dim bottleneck activation.

Why one would train this instead of the classifier:
  * no style labels needed, and the objective is the one the manuscript text describes;
  * the code is optimised to *preserve* motion content rather than to separate styles.
Trade-offs to keep in mind:
  * FID in an autoencoder code is not comparable with CAMDM's reported FID (their features are
    trained to discriminate styles), so cross-paper numbers cannot be mixed;
  * if the extractor is a strong near-lossless autoencoder, FID becomes dominated by tiny
    residuals; if the bottleneck is too tight, by what the bottleneck happens to keep. The
    reconstruction error reported at the end of training is the honest indicator of which
    regime you are in.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.fid_feature_extractor import JOINT_SET_YBOT15, MotionReconFeatureExtractor
from Utils.runtime import resolve_device, set_seed, require_unmirrored_db
from training.train_fid_feature_extractor_camdm import (
    CAMDM_BATCH_SIZE,
    CAMDM_CLIP_OFFSET,
    CAMDM_LR,
    CAMDM_NUM_WORKERS,
    CAMDM_WINDOW_SIZE,
    CamdmWindowDataset,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
RECON_TYPES = ("l1", "mse", "smooth_l1")


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    recon_type: str,
    vel_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction error on [B, C, T] windows, optionally plus a temporal-difference term."""
    if recon_type == "l1":
        base = F.l1_loss(recon, target)
    elif recon_type == "mse":
        base = F.mse_loss(recon, target)
    elif recon_type == "smooth_l1":
        base = F.smooth_l1_loss(recon, target)
    else:
        raise ValueError(f"Unknown recon_type={recon_type!r}; expected one of {RECON_TYPES}")
    vel = recon.new_zeros(())
    if vel_weight > 0.0 and recon.size(-1) >= 2:
        vel = F.l1_loss(recon[..., 1:] - recon[..., :-1], target[..., 1:] - target[..., :-1])
    return base + float(vel_weight) * vel, base, vel


def run_epoch(
    *,
    model: MotionReconFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    recon_type: str,
    vel_weight: float,
    log_interval: int,
    training: bool = True,
) -> tuple[float, float, float]:
    model.train(training)
    total = 0
    sum_total = 0.0
    sum_base = 0.0
    sum_vel = 0.0
    tag = "train" if training else "val"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
      for step, (inputs, _labels) in enumerate(tqdm(loader, desc=f"{tag} {epoch}/{total_epochs}", leave=False), start=1):
        x = inputs.to(device=device, non_blocking=True).permute(0, 2, 1).float()
        if training:
            assert optimizer is not None, "training requires an optimizer"
            optimizer.zero_grad()
        _, recon = model(x)
        loss, base, vel = reconstruction_loss(recon, x, recon_type=recon_type, vel_weight=vel_weight)
        if training:
            loss.backward()
            optimizer.step()
        batch = int(x.size(0))
        total += batch
        sum_total += float(loss.detach()) * batch
        sum_base += float(base.detach()) * batch
        sum_vel += float(vel.detach()) * batch
        if training and log_interval > 0 and step % log_interval == 0:
            print(
                f"[train] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={float(loss.detach()):.6f} recon={float(base.detach()):.6f}"
            )
    denom = max(total, 1)
    return sum_total / denom, sum_base / denom, sum_vel / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reconstruction-objective FID feature extractor")
    parser.add_argument("--db", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_fid_full100_nomirror")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "checkpoints" / "fid_feature_extractor_recon100_nomirror")
    parser.add_argument("--window-size", type=int, default=CAMDM_WINDOW_SIZE)
    parser.add_argument("--clip-offset", type=int, default=CAMDM_CLIP_OFFSET)
    parser.add_argument("--batch-size", type=int, default=CAMDM_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=CAMDM_LR)
    parser.add_argument("--num-workers", type=int, default=CAMDM_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--feat-dim", type=int, default=512, help="bottleneck width = FID feature dim (CAMDM uses 512)")
    parser.add_argument("--recon-type", type=str, choices=RECON_TYPES, default="l1")
    parser.add_argument(
        "--vel-weight",
        type=float,
        default=0.0,
        help="weight of an L1 temporal-difference term on top of the reconstruction loss",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Clip-level per-style validation fraction used for best-checkpoint selection",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-by", choices=("temporal", "clip", "suffix"), default="temporal",
        help=(
            "temporal: hold out the tail of every clip (all locomotion types stay in both "
            "splits, guard band prevents window overlap); clip: hold out whole clips "
            "(unseen clips, action-imbalanced across styles); suffix: hold out whole "
            "suffixes for every style (unseen clips, identical action for all styles)"
        ),
    )
    parser.add_argument(
        "--val-suffixes", type=str, default="",
        help="comma-separated suffixes held out when --split-by suffix (e.g. SR,SW)",
    )
    parser.add_argument(
        "--temporal-guard-frames", type=int, default=0,
        help="frames skipped between the temporal train and val regions; 0 uses window_size",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout on the conv features (CAMDM uses 0.5)")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="Extra dropout before the FID feature")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = Path(args.db).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    require_unmirrored_db(args.db)
    val_suffixes = tuple(part.strip() for part in str(args.val_suffixes).split(",") if part.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset = CamdmWindowDataset(
        db=args.db,
        window_size=args.window_size,
        clip_offset=args.clip_offset,
        split='train' if args.val_ratio > 0 else 'all',
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
        val_suffixes=val_suffixes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_dataset = val_loader = None
    val_split_warning = None
    if args.val_ratio > 0:
        val_dataset = CamdmWindowDataset(
            db=args.db, window_size=args.window_size, clip_offset=args.clip_offset,
            split='val', val_ratio=args.val_ratio, split_seed=args.split_seed,
            split_by=args.split_by, val_suffixes=val_suffixes,
        )
        if len(val_dataset) < 2:
            # Short clips cannot host the temporal guard band; fall back to the no-split
            # protocol with a loud warning instead of failing the run.
            print(f"[Warn] validation split is empty ({len(val_dataset)} windows): the database's "
                  "clips are too short for the requested split; best selection is disabled")
            val_dataset = val_loader = None
            val_split_warning = 'empty_val_split'
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=(device.type == "cuda"),
            )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, pin_memory=(device.type == "cuda"),
        )
    model = MotionReconFeatureExtractor(
        in_feat=dataset.in_feat,
        window_size=args.window_size,
        hidden_dim=int(args.hidden_dim),
        feat_dim=int(args.feat_dim),
        dropout=args.dropout,
        head_dropout=args.head_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Info] db            : {args.db}")
    print(f"[Info] out_dir       : {args.out_dir}")
    print(f"[Info] device        : {device}")
    if val_loader is None:
        print(f"[Info] windows       : {len(dataset)} (all 100 styles, no split)")
    else:
        print(f"[Info] split_by      : {args.split_by} "
              + (f"val_suffixes={list(val_suffixes)}" if args.split_by == 'suffix' else ""))
        print(f"[Info] split         : train={len(dataset)} windows "
              f"({dataset.clip_counts['train']} clips) / val={len(val_dataset)} windows "
              f"({val_dataset.clip_counts['val']} clips); val_ratio={args.val_ratio} seed={args.split_seed}")
    print(f"[Info] dropout       : conv={args.dropout} head={args.head_dropout}")
    print(f"[Info] objective     : reconstruction ({args.recon_type}, vel_weight={args.vel_weight})")
    print(f"[Info] feature       : {args.feat_dim}-dim bottleneck (input {dataset.in_feat}-dim ybot local positions)")
    print(f"[Info] optimizer     : Adam(lr={args.lr})")

    history_path = args.out_dir / "history.jsonl"
    history = history_path.open("a", encoding="utf-8")
    last = (float("nan"), float("nan"), float("nan"))
    best: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        last = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            recon_type=args.recon_type,
            vel_weight=float(args.vel_weight),
            log_interval=args.log_interval,
        )
        val_record = None
        if val_loader is not None:
            val_last = run_epoch(
                model=model, loader=val_loader, device=device, optimizer=None,
                epoch=epoch, total_epochs=args.epochs, recon_type=args.recon_type,
                vel_weight=float(args.vel_weight), log_interval=0, training=False,
            )
            val_record = {"loss": val_last[0], "recon_loss": val_last[1], "vel_loss": val_last[2],
                          "windows": len(val_dataset)}
            # Selection: lowest validation reconstruction error (ties -> earlier epoch).
            if best is None or val_record["recon_loss"] < best["val_recon"]:
                best = {"epoch": epoch, "val_recon": val_record["recon_loss"], "val_loss": val_record["loss"]}
                torch.save(
                    {
                        "state_dict": {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()},
                        "epoch": epoch, "val_loss": val_record["loss"], "val_recon_loss": val_record["recon_loss"],
                        "in_feat": int(dataset.in_feat), "feat_dim": int(args.feat_dim),
                        "hidden_dim": int(args.hidden_dim), "window_size": int(args.window_size),
                        "joint_set": JOINT_SET_YBOT15, "objective": "reconstruction",
                        "dropout": float(args.dropout), "head_dropout": float(args.head_dropout),
                        "acc_split": "val",
                    },
                    args.out_dir / "best.pt",
                )
        record = {
            "epoch": epoch,
            "loss": last[0],
            "recon_loss": last[1],
            "vel_loss": last[2],
            "val": val_record,
            "lr": args.lr,
            "windows": len(dataset),
            "time_sec": time.time() - t0,
        }
        history.write(json.dumps(record) + "\n")
        history.flush()
        suffix = "" if val_record is None else f" val_recon={val_record['recon_loss']:.6f}"
        print(f"[Epoch {epoch}] loss={last[0]:.6f} recon={last[1]:.6f} vel={last[2]:.6f}{suffix} ({record['time_sec']:.1f}s)")
    history.close()

    final_state = None
    if best is not None:
        final_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        selected = torch.load(args.out_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        print(f"[Select] best epoch {best['epoch']} val_recon={best['val_recon']:.6f}")

    payload = {
        "state_dict": model.state_dict(),
        "objective": "reconstruction",
        "in_feat": int(dataset.in_feat),
        "feat_dim": int(args.feat_dim),
        "hidden_dim": int(args.hidden_dim),
        "window_size": int(args.window_size),
        "joint_set": JOINT_SET_YBOT15,
        "source": "train_fid_feature_extractor_recon.py",
        "recon_type": str(args.recon_type),
        "vel_weight": float(args.vel_weight),
        "recon_loss": float(last[1]),
        "dropout": float(args.dropout), "head_dropout": float(args.head_dropout),
        "val_ratio": float(args.val_ratio), "split_seed": int(args.split_seed),
        "split_by": str(args.split_by), "val_suffixes": list(val_suffixes),
        "temporal_guard_frames": int(args.temporal_guard_frames or args.window_size),
        "train_windows": len(dataset),
        "val_windows": len(val_dataset) if val_dataset is not None else 0,
        "best_epoch": None if best is None else int(best["epoch"]),
        "best_val_recon_loss": None if best is None else float(best["val_recon"]),
        "selection_rule": "min_val_recon_loss" if best is not None else "no_val_split",
        "val_split_warning": val_split_warning,
        "acc_split": "val" if best is not None else "train",
        "split_protocol": (f"extractor_split_by_{args.split_by}" if best is not None else "all_clips"),
        "db": str(args.db),
        "clip_offset": int(args.clip_offset),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "windows": len(dataset),
        "temporal_stride": 1,
        "class_names": [],
        "class_num": 0,
    }
    final_path = args.out_dir / "100style_position_recon.pth"
    torch.save(payload, final_path)
    torch.save(payload, args.out_dir / "latest.pt")
    if final_state is not None:
        final_payload = dict(payload)
        final_payload["state_dict"] = final_state
        torch.save(final_payload, args.out_dir / "final.pt")
        print(f"[Done] canonical file = best by val (epoch {best['epoch']}, val_recon={best['val_recon']:.6f})")
    print(f"[Done] final {args.recon_type} reconstruction error={last[1]:.6f} saved {final_path}")


if __name__ == "__main__":
    main()
