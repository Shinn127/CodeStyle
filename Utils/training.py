from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import SPLIT_MODES, DataBundle, build_train_val_test_dataloaders
from Utils.runtime import (
    build_dataloader,
    load_checkpoint,
    save_norm_stats_if_present,
    subset_dataset_by_fraction,
)


@dataclass(frozen=True)
class CommonDataDefaults:
    seq_len: int
    stride: int
    batch_size: int
    max_windows_for_stats: int
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 8
    prefetch_factor: int = 4
    norm_eps: float = 1e-8
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    test_fraction: float = 1.0
    subset_seed: int = 123


@dataclass(frozen=True)
class CommonTrainDefaults:
    epochs: int = 50
    lr: float = 2e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    grad_accum_steps: int = 2
    grad_clip: float = 1.0


@dataclass(frozen=True)
class CommonRuntimeDefaults:
    amp: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    fused_adamw: bool = True
    matmul_precision: str = "high"
    compile_mode: str = "reduce-overhead"
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class CommonLoggingDefaults:
    save_every: int = 10
    val_every: int = 5
    log_interval: int = 100


@dataclass(frozen=True)
class PreparedDataLoaders:
    bundle: DataBundle
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    stats_path: Optional[Path]


def add_bool_arg(
    parser: Any,
    *,
    name: str,
    default: bool,
    help_enable: str | None = None,
    help_disable: str | None = None,
) -> None:
    dest = name.replace("-", "_")
    parser.set_defaults(**{dest: bool(default)})
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help_enable)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_disable)


def add_common_io_args(
    parser: Any,
    *,
    script_dir: Path,
    out_dir_name: str,
    include_db: bool = True,
) -> None:
    if include_db:
        parser.add_argument(
            "--db",
            type=Path,
            default=script_dir / "database_100sty_375_memmap_nomirror",
            help="Path to memmap directory",
        )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "checkpoints" / out_dir_name,
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--reset-optimizer", action="store_true", help="Ignore optimizer/scheduler states on resume")


def add_common_data_args(
    parser: Any,
    *,
    defaults: CommonDataDefaults,
) -> None:
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--stride", type=int, default=defaults.stride)
    add_bool_arg(parser, name="enforce-same-label", default=True)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    add_bool_arg(parser, name="persistent-workers", default=True)
    parser.add_argument("--prefetch-factor", type=int, default=defaults.prefetch_factor)
    add_bool_arg(parser, name="pin-memory", default=True)
    parser.add_argument("--load-into-memory", action="store_true")
    add_bool_arg(parser, name="normalize", default=True)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=list(SPLIT_MODES),
        default="auto",
        help=(
            "auto: offline splits from segments.json if present, else leak-free clip-level "
            "splitting; precomputed: require offline splits; clip: force clip-level; "
            "none: single unsplit dataset"
        ),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="norm_stats.npz to load when it exists, otherwise to write after computing from the train split",
    )
    parser.add_argument("--norm-eps", type=float, default=defaults.norm_eps)
    parser.add_argument(
        "--max-windows-for-stats",
        type=int,
        default=defaults.max_windows_for_stats,
        help="0 means use all train windows to estimate normalization stats",
    )
    parser.add_argument("--stats-windows-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=defaults.train_fraction,
        help="Use only this fraction of train data for quick tests",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=defaults.val_fraction,
        help="Use only this fraction of val data",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=defaults.test_fraction,
        help="Use only this fraction of test data",
    )
    parser.add_argument("--subset-seed", type=int, default=defaults.subset_seed)


def add_common_train_args(
    parser: Any,
    *,
    defaults: CommonTrainDefaults,
) -> None:
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)


def add_common_runtime_args(
    parser: Any,
    *,
    defaults: CommonRuntimeDefaults,
) -> None:
    add_bool_arg(parser, name="amp", default=defaults.amp)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help=(
            "Autocast compute dtype on CUDA; 'fp32' disables autocast for a single-precision "
            "run (combine with --no-tf32 --matmul-precision highest for true fp32)"
        ),
    )
    add_bool_arg(parser, name="tf32", default=defaults.tf32)
    add_bool_arg(parser, name="cudnn-benchmark", default=defaults.cudnn_benchmark)
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["highest", "high", "medium"],
        default=defaults.matmul_precision,
    )
    add_bool_arg(parser, name="fused-adamw", default=defaults.fused_adamw)
    add_bool_arg(parser, name="compile", default=False, help_enable="Use torch.compile for the model")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default=defaults.compile_mode,
    )
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device, help="auto/cuda/cpu")


def add_common_logging_args(
    parser: Any,
    *,
    defaults: CommonLoggingDefaults,
) -> None:
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--val-every", type=int, default=defaults.val_every, help="Run validation every N epochs")
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)


def canonicalize_path_args(
    args: argparse.Namespace,
    *,
    extra_fields: Iterable[str] = (),
) -> None:
    for field in ("db", "out_dir", "resume", *tuple(extra_fields)):
        if not hasattr(args, field):
            continue
        value = getattr(args, field)
        if value is None or not isinstance(value, Path):
            continue
        setattr(args, field, value.expanduser().resolve())


def validate_common_data_args(
    *,
    grad_accum_steps: int,
    val_every: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    if grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {grad_accum_steps}")
    if val_every < 1:
        raise ValueError(f"--val-every must be >= 1, got {val_every}")
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"--train-fraction must be in (0,1], got {train_fraction}")
    if not (0.0 < val_fraction <= 1.0):
        raise ValueError(f"--val-fraction must be in (0,1], got {val_fraction}")
    if not (0.0 < test_fraction <= 1.0):
        raise ValueError(f"--test-fraction must be in (0,1], got {test_fraction}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"--val-ratio must be in [0,1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"--test-ratio must be in [0,1), got {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"--val-ratio + --test-ratio must be < 1, got {val_ratio + test_ratio}")


def write_config_json(*, out_dir: Path, args: Any) -> Path:
    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    return config_path


def append_jsonl(*, path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_compile_model(*, model: torch.nn.Module, args: Any) -> torch.nn.Module:
    if not getattr(args, "compile", False):
        return model
    if not hasattr(torch, "compile"):
        print("[Warn] torch.compile is not available in this torch build; skipping --compile.")
        return model

    compiled_model = torch.compile(
        model,
        mode=args.compile_mode,
        fullgraph=bool(args.compile_fullgraph),
    )
    print(
        f"[Info] compile         : enabled (mode={args.compile_mode}, fullgraph={args.compile_fullgraph})"
    )
    return compiled_model


def build_grad_scaler(
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> Optional[torch.cuda.amp.GradScaler]:
    if device.type != "cuda":
        return None
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def prepare_data_loaders_from_args(
    *,
    args: Any,
    test_loader_note: str,
    norm_mean: Any = None,
    norm_std: Any = None,
) -> PreparedDataLoaders:
    bundle: DataBundle = build_train_val_test_dataloaders(
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
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=args.norm_eps,
        max_windows_for_stats=args.max_windows_for_stats,
        stats_windows_chunk_size=args.stats_windows_chunk_size,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        split_mode=getattr(args, "split_mode", "auto"),
        norm_stats_path=getattr(args, "norm_stats", None),
    )

    train_dataset = subset_dataset_by_fraction(bundle.train_dataset, args.train_fraction, args.subset_seed)
    val_dataset = subset_dataset_by_fraction(bundle.val_dataset, args.val_fraction, args.subset_seed + 1)
    test_dataset = subset_dataset_by_fraction(bundle.test_dataset, args.test_fraction, args.subset_seed + 2)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    print(f"[Info] train samples   : {len(bundle.train_dataset)} -> {len(train_dataset)}")
    if bundle.val_dataset is None:
        print("[Info] val samples     : 0 (disabled)")
    else:
        print(f"[Info] val samples     : {len(bundle.val_dataset)} -> {len(val_dataset)}")
    if bundle.test_dataset is None:
        print("[Info] test samples    : 0 (disabled)")
    else:
        print(f"[Info] test samples    : {len(bundle.test_dataset)} -> {len(test_dataset)}")
    if test_loader is not None and test_loader_note:
        print(f"[Info] test loader     : {test_loader_note}")
    print(f"[Info] normalize       : {args.normalize}")
    print(f"[Info] split mode      : {bundle.split_mode} ({bundle.split_source or 'n/a'})")
    if bundle.norm_fingerprint:
        print(f"[Info] norm fingerprint: {bundle.norm_fingerprint}")

    stats_path = save_norm_stats_if_present(
        out_dir=args.out_dir,
        norm_mean=bundle.norm_mean,
        norm_std=bundle.norm_std,
    )
    if stats_path is not None:
        print(f"[Info] norm stats      : {stats_path}")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check --db / --seq-len / --stride / --enforce-same-label settings."
        )

    return PreparedDataLoaders(
        bundle=bundle,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        stats_path=stats_path,
    )


def save_training_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    args: Any,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)


def finalize_epoch_outputs(
    *,
    history_path: Path,
    record: dict[str, Any],
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    metric_for_best: Optional[float],
    save_every: int,
    args: Any,
    extra: Optional[dict[str, Any]] = None,
) -> float:
    append_jsonl(path=history_path, record=record)

    save_training_checkpoint(
        ckpt_path=out_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_metric=best_metric,
        args=args,
        extra=extra,
    )

    updated_best_metric = best_metric
    if metric_for_best is not None and metric_for_best < updated_best_metric:
        updated_best_metric = metric_for_best
        save_training_checkpoint(
            ckpt_path=out_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=updated_best_metric,
            args=args,
            extra=extra,
        )
        print(f"[Info] saved best checkpoint (metric={updated_best_metric:.6f})")

    if save_every > 0 and (epoch % save_every == 0):
        save_training_checkpoint(
            ckpt_path=out_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=updated_best_metric,
            args=args,
            extra=extra,
        )

    return updated_best_metric


def resume_training_state(
    *,
    resume_path: Optional[Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    reset_optimizer: bool,
    best_metric: float,
    state_dict_transform: Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = None,
    strict: bool = True,
) -> tuple[int, float]:
    start_epoch = 1
    if resume_path is None:
        return start_epoch, best_metric
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    ckpt = load_checkpoint(resume_path, map_location="cpu")
    model_state = ckpt["model"]
    if state_dict_transform is not None:
        model_state = state_dict_transform(model_state)
    model.load_state_dict(model_state, strict=strict)

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", best_metric))

    if not reset_optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    print(f"[Info] resumed from    : {resume_path} (start_epoch={start_epoch})")
    return start_epoch, best_metric


def load_best_checkpoint_for_eval(
    *,
    model: torch.nn.Module,
    ckpt_path: Path,
    state_dict_transform: Optional[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] = None,
    strict: bool = True,
    allow_incompatible: bool = False,
) -> bool:
    if not ckpt_path.exists():
        print("[Warn] best.pt not found, evaluating test with current model weights.")
        return False

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model_state = ckpt["model"]
    if state_dict_transform is not None:
        model_state = state_dict_transform(model_state)

    try:
        model.load_state_dict(model_state, strict=strict)
    except RuntimeError as exc:
        if not allow_incompatible:
            raise
        print(
            "[Warn] best checkpoint state_dict is incompatible with current model "
            f"(likely from older architecture): {exc}"
        )
        print("[Warn] evaluating test with current in-memory model weights.")
        return False

    print(f"[Info] loaded best checkpoint for test: {ckpt_path}")
    return True


__all__ = [
    "CommonDataDefaults",
    "CommonLoggingDefaults",
    "CommonRuntimeDefaults",
    "CommonTrainDefaults",
    "PreparedDataLoaders",
    "add_bool_arg",
    "add_common_data_args",
    "add_common_io_args",
    "add_common_logging_args",
    "add_common_runtime_args",
    "add_common_train_args",
    "append_jsonl",
    "build_grad_scaler",
    "canonicalize_path_args",
    "finalize_epoch_outputs",
    "load_best_checkpoint_for_eval",
    "maybe_compile_model",
    "prepare_data_loaders_from_args",
    "resume_training_state",
    "save_training_checkpoint",
    "validate_common_data_args",
    "write_config_json",
]
