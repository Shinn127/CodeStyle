"""
Minimal training script for the flow-matching LMM controller.

Core pipeline:
  1) load normalized 375-dim frame windows from the memmap DB
  2) convert each window into adjacent frame pairs
  3) freeze the LMM autoencoder and train a flow field in latent space
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, desc: str = "", unit: str = "it", dynamic_ncols: bool = False) -> None:
            self.total = max(int(total), 0)
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.count = 0
            self.postfix = ""
            self._closed = False

        def __enter__(self):
            self._render()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def set_postfix(self, refresh: bool = True, **kwargs: object) -> None:
            self.postfix = " ".join(f"{key}={value}" for key, value in kwargs.items())
            if refresh:
                self._render()

        def update(self, n: int = 1) -> None:
            self.count = min(self.total, self.count + int(n)) if self.total else self.count + int(n)
            self._render()

        def close(self) -> None:
            if self._closed:
                return
            self._render(final=True)
            self._closed = True

        def _render(self, final: bool = False) -> None:
            width = 32
            ratio = (self.count / self.total) if self.total else 1.0
            filled = int(width * ratio)
            bar = "#" * filled + "." * (width - filled)
            suffix = f" {self.postfix}" if self.postfix else ""
            line = f"\r{self.desc} [{bar}] {self.count}/{self.total} {self.unit}{suffix}"
            sys.stderr.write(line)
            if final:
                sys.stderr.write("\n")
            sys.stderr.flush()

from arch import FlowMatchLMMController, build_flowmatch_lmm_controller_from_config
from dataset import build_train_val_test_dataloaders
from Utils.model_loading import build_lmm_from_checkpoint
from Utils.runtime import (
    MEMMAP_META_FILE,
    MEMMAP_X_FILE,
    configure_torch_runtime,
    load_checkpoint,
    load_norm_stats,
    resolve_device,
    sanitize_state_dict_keys,
    save_norm_stats_if_present,
    set_seed,
    require_unmirrored_db,
)


def _flatten_adjacent_pairs(
    x: torch.Tensor,
    style_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or x.size(1) < 2:
        raise ValueError(f"Expected x shape [B,T,D] with T>=2, got {tuple(x.shape)}")
    batch, steps, dims = x.shape
    if style_label.numel() != batch:
        raise ValueError("Batch metadata shape mismatch")

    x_curr = x[:, :-1].reshape(-1, dims)
    x_next = x[:, 1:].reshape(-1, dims)
    style_bt = style_label[:, None].expand(batch, steps - 1).reshape(-1)
    return x_curr, x_next, style_bt


def _metric_keys(prediction_mode: str) -> list[str]:
    base = [
        "loss",
        "flow_loss",
        "recon_loss",
        "latent_target_norm",
        "velocity_pred_norm",
        "velocity_target_norm",
    ]
    if prediction_mode == "target":
        base.extend(["target_pred_norm", "target_error", "target_weight_mean"])
    return base


def _save_json(path: Path, payload: dict) -> None:
    serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in payload.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_latent_stats(
    stats_path: Path,
    *,
    latent_dim: int,
) -> dict[str, np.ndarray | str]:
    data = np.load(stats_path, allow_pickle=True)
    required = {"offset", "scale", "transport_std", "data_min", "data_max"}
    if not required.issubset(data.files):
        raise ValueError(
            f"Latent stats file {stats_path} must contain {sorted(required)} for CO-style flow matching"
        )
    offset = np.asarray(data["offset"], dtype=np.float32).reshape(-1)
    scale = np.asarray(data["scale"], dtype=np.float32).reshape(-1)
    transport_std = np.asarray(data["transport_std"], dtype=np.float32).reshape(-1)
    data_min = np.asarray(data["data_min"], dtype=np.float32).reshape(-1)
    data_max = np.asarray(data["data_max"], dtype=np.float32).reshape(-1)
    if (
        offset.shape != (latent_dim,)
        or scale.shape != (latent_dim,)
        or transport_std.shape != (latent_dim,)
        or data_min.shape != (latent_dim,)
        or data_max.shape != (latent_dim,)
    ):
        raise ValueError(f"Invalid latent stats shape in {stats_path}")
    return {
        "mode": "co",
        "offset": offset,
        "scale": scale,
        "transport_std": transport_std,
        "data_min": data_min,
        "data_max": data_max,
    }


def _save_latent_stats(stats_path: Path, stats: dict[str, np.ndarray | str]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        mode=np.asarray(str(stats["mode"])),
        offset=np.asarray(stats["offset"], dtype=np.float32),
        scale=np.asarray(stats["scale"], dtype=np.float32),
        transport_std=np.asarray(stats["transport_std"], dtype=np.float32),
        data_min=np.asarray(stats["data_min"], dtype=np.float32),
        data_max=np.asarray(stats["data_max"], dtype=np.float32),
    )


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


@torch.no_grad()
def _estimate_raw_latent_stats(
    *,
    db_path: Path,
    lmm,
    device: torch.device,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    with (db_path / MEMMAP_META_FILE).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    total_frames = int(meta["num_frames"])
    x_source = np.memmap(db_path / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(total_frames, 375))

    current_chunk_rows = int(chunk_rows)
    while True:
        try:
            sum_latent = torch.zeros(lmm.latent_dim, dtype=torch.float64, device=device)
            sumsq_latent = torch.zeros(lmm.latent_dim, dtype=torch.float64, device=device)
            min_latent = torch.full((lmm.latent_dim,), float("inf"), dtype=torch.float64, device=device)
            max_latent = torch.full((lmm.latent_dim,), -float("inf"), dtype=torch.float64, device=device)
            for start in range(0, total_frames, current_chunk_rows):
                end = min(start + current_chunk_rows, total_frames)
                x_chunk = np.asarray(x_source[start:end], dtype=np.float32)
                x_chunk = (x_chunk - norm_mean[None]) / norm_std[None]
                motion_t = torch.from_numpy(x_chunk[:, lmm.traj_dim :]).to(device=device)
                latents = lmm.encode(motion_t).to(torch.float64)
                sum_latent += latents.sum(dim=0)
                sumsq_latent += latents.square().sum(dim=0)
                min_latent = torch.minimum(min_latent, latents.amin(dim=0))
                max_latent = torch.maximum(max_latent, latents.amax(dim=0))
            count = float(total_frames)
            mean = sum_latent / count
            var = (sumsq_latent / count) - mean.square()
            std = torch.sqrt(var.clamp_min(1e-6))
            return (
                mean.cpu().numpy().astype(np.float32),
                std.cpu().numpy().astype(np.float32),
                min_latent.cpu().numpy().astype(np.float32),
                max_latent.cpu().numpy().astype(np.float32),
            )
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc) or current_chunk_rows <= 1:
                raise
            next_chunk_rows = max(current_chunk_rows // 2, 1)
            print(
                f"[Warn] CUDA OOM while estimating latent stats with chunk_rows={current_chunk_rows}; "
                f"retrying with {next_chunk_rows}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_chunk_rows = next_chunk_rows


def _build_latent_stats_payload(
    *,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
) -> dict[str, np.ndarray | str]:
    offset = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
    raw_std = np.asarray(raw_std, dtype=np.float32).reshape(-1).clip(min=1e-6)
    raw_min = np.asarray(raw_min, dtype=np.float32).reshape(-1)
    raw_max = np.asarray(raw_max, dtype=np.float32).reshape(-1)
    global_scale = float(np.clip(raw_std.mean(), a_min=1e-6, a_max=None))
    scale = np.full_like(raw_std, global_scale, dtype=np.float32)
    transport_std = (raw_std / scale).astype(np.float32, copy=False)
    data_min = ((raw_min - offset) / scale).astype(np.float32, copy=False)
    data_max = ((raw_max - offset) / scale).astype(np.float32, copy=False)
    return {
        "mode": "co",
        "offset": offset,
        "scale": scale.astype(np.float32, copy=False),
        "transport_std": transport_std.astype(np.float32, copy=False),
        "data_min": data_min,
        "data_max": data_max,
    }


def _run_epoch(
    *,
    model: FlowMatchLMMController,
    loader,
    device: torch.device,
    prediction_mode: str,
    recon_weight: float,
    noise_std: float,
    prev_latent_noise_std: float,
    prev_motion_noise_std: float,
    prev_latent_dropout: float,
    prev_motion_dropout: float,
    solver_steps: int,
    optimizer: AdamW | None,
    grad_clip: float,
    log_interval: int,
    epoch: int,
    total_epochs: int,
    split: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    keys = _metric_keys(prediction_mode)
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
            x_curr, x_next, style_bt = _flatten_adjacent_pairs(x, style)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            loss_dict = model.compute_loss(
                x_curr=x_curr,
                x_next=x_next,
                style_label=style_bt,
                recon_weight=recon_weight,
                noise_std=noise_std,
                prev_latent_noise_std=prev_latent_noise_std,
                prev_motion_noise_std=prev_motion_noise_std,
                prev_latent_dropout=prev_latent_dropout,
                prev_motion_dropout=prev_motion_dropout,
                solver_steps=solver_steps,
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
                    flow=f"{float(loss_dict['flow_loss'].detach().cpu()):.4f}",
                    recon=f"{float(loss_dict['recon_loss'].detach().cpu()):.4f}",
                    rate=f"{samples_per_sec:.1f}/s",
                )

    metrics = {key: metric_sums[key] / max(num_samples, 1) for key in keys}
    metrics["num_samples"] = float(num_samples)
    return metrics


def _save_checkpoint(
    *,
    path: Path,
    model: FlowMatchLMMController,
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


def _move_optimizer_state_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Minimal flow-matching LMM controller trainer")
    parser.add_argument("--db", type=Path, default=script_dir / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--lmm-ckpt", type=Path, default=script_dir / "checkpoints" / "lmm_375to291" / "best.pt")
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--latent-stats", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=script_dir / "checkpoints" / "flowmatch_lmm_controller")
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--flow-heads", type=int, default=4)
    parser.add_argument("--flow-ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--prediction-mode", type=str, choices=["velocity", "target"], default="velocity")
    parser.add_argument("--target-time-eps", type=float, default=1e-3)

    parser.add_argument("--recon-weight", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--prev-latent-noise-std", type=float, default=0.3)
    parser.add_argument("--prev-motion-noise-std", type=float, default=0.3)
    parser.add_argument("--prev-latent-dropout", type=float, default=0.3)
    parser.add_argument("--prev-motion-dropout", type=float, default=0.3)
    parser.add_argument("--solver-steps", type=int, default=8)
    parser.add_argument("--latent-stats-chunk-rows", type=int, default=65536)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.lmm_ckpt = args.lmm_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.resume = None if args.resume is None else args.resume.expanduser().resolve()
    args.norm_stats = None if args.norm_stats is None else args.norm_stats.expanduser().resolve()
    args.latent_stats = (args.out_dir / "latent_stats.npz") if args.latent_stats is None else args.latent_stats.expanduser().resolve()
    args.model_type = "flowmatch_lmm_controller"
    args.base_model_type = "lmm"

    require_unmirrored_db(args.db)
    if not args.lmm_ckpt.exists():
        raise FileNotFoundError(f"LMM checkpoint not found: {args.lmm_ckpt}")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.solver_steps < 1:
        raise ValueError("--solver-steps must be >= 1")
    if not args.normalize:
        raise ValueError("Minimal flowmatch training expects normalized inputs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

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
    print(f"[Info] lmm ckpt        : {args.lmm_ckpt}")
    print(f"[Info] prediction mode : {args.prediction_mode}")
    print("[Info] latent norm     : co")
    print(f"[Info] num styles      : {num_styles}")
    if args.norm_stats is not None:
        print(f"[Info] norm stats      : {args.norm_stats}")

    lmm_model = build_lmm_from_checkpoint(args.lmm_ckpt).to(device)
    model = build_flowmatch_lmm_controller_from_config(
        vars(args),
        lmm=lmm_model,
        num_styles=num_styles,
    ).to(device)
    if args.latent_stats.exists():
        latent_stats = _load_latent_stats(
            args.latent_stats,
            latent_dim=model.latent_dim,
        )
    else:
        raw_mean, raw_std, raw_min, raw_max = _estimate_raw_latent_stats(
            db_path=args.db,
            lmm=lmm_model,
            device=device,
            norm_mean=bundle.norm_mean,
            norm_std=bundle.norm_std,
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
        raise RuntimeError("No trainable parameters found in flow controller.")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch = 1
    best_metric = float("inf")
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        resume_ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(sanitize_state_dict_keys(resume_ckpt["model"]), strict=True)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_metric = float(resume_ckpt.get("best_metric", best_metric))
        print(f"[Info] resume         : {args.resume} (start_epoch={start_epoch})")

    history_path = args.out_dir / "history.jsonl"
    _save_json(args.out_dir / "config.json", vars(args))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=bundle.train_loader,
            device=device,
            prediction_mode=args.prediction_mode,
            recon_weight=args.recon_weight,
            noise_std=args.noise_std,
            prev_latent_noise_std=args.prev_latent_noise_std,
            prev_motion_noise_std=args.prev_motion_noise_std,
            prev_latent_dropout=args.prev_latent_dropout,
            prev_motion_dropout=args.prev_motion_dropout,
            solver_steps=args.solver_steps,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            split="train",
        )
        scheduler.step()

        val_metrics = None
        if bundle.val_loader is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model=model,
                    loader=bundle.val_loader,
                    device=device,
                    prediction_mode=args.prediction_mode,
                    recon_weight=args.recon_weight,
                    noise_std=args.noise_std,
                    prev_latent_noise_std=0.0,
                    prev_motion_noise_std=0.0,
                    prev_latent_dropout=0.0,
                    prev_motion_dropout=0.0,
                    solver_steps=args.solver_steps,
                    optimizer=None,
                    grad_clip=0.0,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    split="val",
                )

        metric_for_best = train_metrics["loss"] if val_metrics is None else val_metrics["loss"]
        history_row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "best_metric": min(best_metric, metric_for_best),
        }
        _append_jsonl(history_path, history_row)

        last_ckpt_path = args.out_dir / "last.pt"
        _save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=min(best_metric, metric_for_best),
            args=args,
        )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_checkpoint(
                path=args.out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=min(best_metric, metric_for_best),
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

        train_desc = (
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_flow={train_metrics['flow_loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f}"
        )
        if val_metrics is not None:
            train_desc += (
                f" | val_loss={val_metrics['loss']:.6f} "
                f"val_flow={val_metrics['flow_loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f}"
            )
        train_desc += f" | best={best_metric:.6f}"
        print(train_desc)

    if bundle.test_loader is not None:
        with torch.no_grad():
            best_ckpt = load_checkpoint(args.out_dir / "best.pt", map_location="cpu")
            model.load_state_dict(sanitize_state_dict_keys(best_ckpt["model"]), strict=True)
            test_metrics = _run_epoch(
                model=model,
                loader=bundle.test_loader,
                device=device,
                prediction_mode=args.prediction_mode,
                recon_weight=args.recon_weight,
                noise_std=args.noise_std,
                prev_latent_noise_std=0.0,
                prev_motion_noise_std=0.0,
                prev_latent_dropout=0.0,
                prev_motion_dropout=0.0,
                solver_steps=args.solver_steps,
                optimizer=None,
                grad_clip=0.0,
                log_interval=args.log_interval,
                epoch=args.epochs,
                total_epochs=args.epochs,
                split="test",
            )
        _save_json(args.out_dir / "test_metrics.json", test_metrics)
        print(
            f"[Test] loss={test_metrics['loss']:.6f} "
            f"flow={test_metrics['flow_loss']:.6f} "
            f"recon={test_metrics['recon_loss']:.6f}"
        )

    print(f"[Done] best checkpoint : {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
