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
from torch.utils.data import Dataset
from tqdm import tqdm

from Utils.fid_feature_extractor import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_WINDOW_SIZE,
    MotionFeatureClassifier,
)
from Utils.runtime import build_adamw, configure_torch_runtime, resolve_device, set_seed, require_unmirrored_db
from Utils.training import (
    CommonLoggingDefaults,
    CommonRuntimeDefaults,
    CommonTrainDefaults,
    add_common_io_args,
    add_common_logging_args,
    add_common_runtime_args,
    add_common_train_args,
    append_jsonl,
    canonicalize_path_args,
    resume_training_state,
)
from dataset import Motion375Dataset, build_train_val_test_datasets


SCRIPT_DIR = Path(__file__).resolve().parents[1]


class MotionClipStyleDataset(Dataset):
    def __init__(self, base: Motion375Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        x = item["x"]
        if x.dim() != 2 or x.size(-1) != 375:
            raise ValueError(f"Expected base clip shape [T,375], got {tuple(x.shape)}")
        local_pos = x[:, 84 + 3 : 84 + 3 + DEFAULT_FEATURE_DIM]
        label = item["label"]
        return local_pos, label


def _build_dataset_splits(
    *,
    db: Path,
    seq_len: int,
    stride: int,
    load_into_memory: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[MotionClipStyleDataset, MotionClipStyleDataset | None, MotionClipStyleDataset | None]:
    train_ds, val_ds, test_ds = build_train_val_test_datasets(
        db_path=db,
        seq_len=seq_len,
        stride=stride,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        load_into_memory=load_into_memory,
        normalize=False,
    )
    return MotionClipStyleDataset(train_ds), MotionClipStyleDataset(val_ds) if val_ds is not None else None, MotionClipStyleDataset(test_ds) if test_ds is not None else None


def _run_epoch(
    *,
    model: MotionFeatureClassifier,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_interval: int = 100,
    epoch: int = 0,
    total_epochs: int = 0,
    split: str = "train",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    loss_sum = 0.0
    correct = 0
    count = 0
    t0 = time.time()

    for step, (local_pos, label) in enumerate(tqdm(loader, desc=f"{split}:{epoch}/{total_epochs}", leave=False), start=1):
        x = local_pos.to(device=device, non_blocking=True).float().permute(0, 2, 1)
        y = label.to(device=device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        feats, logits = model(x)
        loss = F.cross_entropy(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.detach().cpu()) * int(y.size(0))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        count += int(y.size(0))

        if is_train and (step % log_interval == 0):
            print(
                f"[{split}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={float(loss.detach().cpu()):.4f}"
            )

    elapsed = time.time() - t0
    return {
        "loss": loss_sum / max(count, 1),
        "acc": 100.0 * correct / max(count, 1),
        "num_samples": float(count),
        "time_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train motion FID feature extractor on 100STYLE head10")
    add_common_io_args(parser, script_dir=script_dir, out_dir_name="fid_feature_extractor_head10", include_db=False)
    parser.add_argument("--db", type=Path, default=script_dir / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    add_common_train_args(
        parser,
        defaults=CommonTrainDefaults(
            epochs=50,
            lr=1e-3,
            min_lr=1e-3,
            weight_decay=1e-4,
            beta1=0.9,
            beta2=0.999,
            grad_accum_steps=1,
            grad_clip=1.0,
        ),
    )
    add_common_runtime_args(parser, defaults=CommonRuntimeDefaults())
    add_common_logging_args(parser, defaults=CommonLoggingDefaults(save_every=10, val_every=1, log_interval=50))
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

    train_ds, val_ds, test_ds = _build_dataset_splits(
        db=args.db,
        seq_len=args.seq_len,
        stride=args.stride,
        load_into_memory=args.load_into_memory,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    with open(args.db / "StyleNames.json", "r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    class_num = len(class_names)
    in_feat = DEFAULT_FEATURE_DIM

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = (
        torch.utils.data.DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
        if val_ds is not None
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
        if test_ds is not None
        else None
    )

    print(f"[Info] db              : {args.db}")
    print(f"[Info] out_dir         : {args.out_dir}")
    print(f"[Info] classes         : {class_num}")
    print(f"[Info] feature dim     : {in_feat}")
    print(f"[Info] window size     : {args.seq_len}")
    print(f"[Info] train samples   : {len(train_ds)}")
    print(f"[Info] val samples     : {len(val_ds) if val_ds is not None else 0}")
    print(f"[Info] test samples    : {len(test_ds) if test_ds is not None else 0}")

    model = MotionFeatureClassifier(in_feat=in_feat, class_num=class_num, window_size=args.seq_len).to(device)
    optimizer, use_fused_adamw = build_adamw(
        model.parameters(),
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        use_fused=bool(args.fused_adamw and device.type == "cuda"),
    )
    print(f"[Info] optimizer       : AdamW(fused={use_fused_adamw})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    best_metric = float("inf")
    history_path = args.out_dir / "history.jsonl"

    start_epoch, best_metric = resume_training_state(
        resume_path=args.resume,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        reset_optimizer=args.reset_optimizer,
        best_metric=best_metric,
        strict=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )
        val_metrics = None
        metric_for_best = train_metrics["loss"]
        if val_loader is not None and (epoch % args.val_every) == 0:
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    split="val",
                )
            metric_for_best = val_metrics["loss"]
        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(path=history_path, record=record)

        payload_args = vars(args).copy()
        payload_args.update(
            {
                "class_names": class_names,
                "class_num": class_num,
                "in_feat": in_feat,
                "window_size": args.seq_len,
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "class_num": class_num,
                "in_feat": in_feat,
                "window_size": args.seq_len,
                "best_metric": best_metric,
                "args": payload_args,
            },
            args.out_dir / "latest.pt",
        )
        if metric_for_best < best_metric:
            best_metric = metric_for_best
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "class_num": class_num,
                    "in_feat": in_feat,
                    "window_size": args.seq_len,
                    "best_metric": best_metric,
                    "args": payload_args,
                },
                args.out_dir / "best.pt",
            )
            print(f"[Info] saved best checkpoint (metric={best_metric:.6f})")
        if args.save_every > 0 and (epoch % args.save_every == 0):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "class_num": class_num,
                    "in_feat": in_feat,
                    "window_size": args.seq_len,
                    "best_metric": best_metric,
                    "args": payload_args,
                },
                args.out_dir / f"epoch_{epoch:04d}.pt",
            )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['acc']:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
            + (
                f" | val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['acc']:.2f}"
                if val_metrics is not None
                else ""
            )
        )

    if test_loader is not None:
        with torch.no_grad():
            test_metrics = _run_epoch(
                model=model,
                loader=test_loader,
                device=device,
                optimizer=None,
                log_interval=args.log_interval,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        print(f"[Test] loss={test_metrics['loss']:.6f} acc={test_metrics['acc']:.2f}")
        append_jsonl(path=history_path, record={"epoch": "test", "split": "test", "metrics": test_metrics})

    print(f"[Done] Best metric: {best_metric:.6f}")
    print(f"[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
