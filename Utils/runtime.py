from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_amp_dtype(
    amp_dtype_arg: str,
    device: torch.device,
) -> Optional[torch.dtype]:
    """Map an --amp-dtype choice to an autocast dtype, or None for full fp32.

    `none`/`fp32` disables autocast entirely, which is what reproduces a true
    single-precision run. Note that fp32 matmuls still run in TF32 while
    `torch.backends.cuda.matmul.allow_tf32` is True, so a真 fp32 run also needs
    tf32=False and matmul_precision="highest".
    """
    key = amp_dtype_arg.lower().strip()
    if key in {"none", "fp32", "float32"}:
        return None
    if device.type != "cuda":
        return None
    if key == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype_arg}")


def configure_torch_runtime(
    *,
    device: torch.device,
    tf32: bool,
    cudnn_benchmark: bool,
    matmul_precision: str,
) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.set_float32_matmul_precision(matmul_precision)


def estimate_path_size_bytes(path: Path) -> float:
    if path.is_file():
        return float(path.stat().st_size)
    if path.is_dir():
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        return float(total)
    return float("nan")


def to_float(v: torch.Tensor) -> float:
    return float(v.detach().cpu().item())


def validate_memmap_db_dir(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"--db not found: {db_path}")
    if not db_path.is_dir():
        raise ValueError(f"--db must be a memmap directory, got file path: {db_path}")
    required = [MEMMAP_META_FILE, MEMMAP_X_FILE, MEMMAP_LABELS_FILE]
    missing = [name for name in required if not (db_path / name).exists()]
    if missing:
        raise ValueError(f"Invalid memmap directory: {db_path}. Missing files: {missing}")


def require_unmirrored_db(db_path: Path) -> None:
    """Validate a memmap database and refuse mirrored segments.

    Mirrored augmentation is not used for training: the generated mirror segments distort
    the mesh, so every training/precompute entry must consume an unmirrored database
    (`filter_memmap_db.py`, or `tools/generate_database_100sty.py` without `--mirror`).
    """
    db_path = Path(db_path)
    validate_memmap_db_dir(db_path)
    meta_path = db_path / MEMMAP_META_FILE
    segments_path = db_path / "segments.json"
    if not segments_path.exists():
        return
    try:
        meta_layout = json.loads(meta_path.read_text(encoding="utf-8")).get("mirror_layout")
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    mirrored = sum(1 for segment in segments if segment.get("mirror"))
    if mirrored:
        raise ValueError(
            f"database {db_path} is not unmirrored: {mirrored} mirrored segment(s) "
            f"(mirror_layout={meta_layout!r}). Training data must exclude mirrored clips."
        )


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_state_dict_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def sanitize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = strip_state_dict_prefix_if_present(state_dict, "_orig_mod.")
    state_dict = strip_state_dict_prefix_if_present(state_dict, "module.")
    return state_dict


def load_state_dict_with_allowed_missing(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    allowed_missing: Iterable[str] = (),
    allowed_unexpected: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
    allowed_missing_set = set(str(key) for key in allowed_missing)
    allowed_unexpected_set = set(str(key) for key in allowed_unexpected)
    real_missing = [key for key in missing_keys if key not in allowed_missing_set]
    real_unexpected = [key for key in unexpected_keys if key not in allowed_unexpected_set]
    if real_missing or real_unexpected:
        raise RuntimeError(
            "Incompatible state_dict after applying compatibility allowlist. "
            f"missing={real_missing}, unexpected={real_unexpected}"
        )
    return list(missing_keys), list(unexpected_keys)


def subset_dataset_by_fraction(
    dataset: Optional[Dataset],
    fraction: float,
    seed: int,
) -> Optional[Dataset]:
    if dataset is None or fraction >= 1.0:
        return dataset
    num_samples = len(dataset)
    if num_samples == 0:
        return dataset
    keep = max(1, int(round(num_samples * fraction)))
    if keep >= num_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_samples, size=keep, replace=False)
    return Subset(dataset, indices.tolist())


def build_dataloader(
    dataset: Optional[Dataset],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    generator: Optional[torch.Generator] = None,
) -> Optional[DataLoader]:
    if dataset is None:
        return None
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if generator is not None:
        # Explicit per-loader generator: sampling order (and worker seeding) no longer
        # depends on the global torch RNG, so unrelated setup work cannot perturb it.
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    use_fused: bool,
) -> tuple[AdamW, bool]:
    params = list(parameters)
    try:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            fused=bool(use_fused),
        )
        return optimizer, bool(use_fused)
    except TypeError:
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        return optimizer, False


def save_norm_stats(
    path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> Path:
    """Write z-score statistics to an .npz with keys `mean`/`std`, validating shape first."""
    stats_path = Path(path).expanduser().resolve()
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_array.shape != (375,) or std_array.shape != (375,):
        raise ValueError(
            f"norm stats must have shape (375,), got mean={mean_array.shape}, std={std_array.shape}"
        )
    if np.any(std_array <= 0):
        raise ValueError("norm std must be > 0 for all dims")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stats_path, mean=mean_array, std=std_array)
    return stats_path


def save_norm_stats_if_present(
    *,
    out_dir: Path,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
) -> Optional[Path]:
    if norm_mean is None or norm_std is None:
        return None
    return save_norm_stats(Path(out_dir) / "norm_stats.npz", norm_mean, norm_std)


def load_norm_stats(norm_stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm stats not found: {norm_stats_path}")
    data = np.load(norm_stats_path)
    if "mean" not in data or "std" not in data:
        raise KeyError(f"{norm_stats_path} must contain keys: mean, std")
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["std"], dtype=np.float32).reshape(-1)
    if mean.shape != (375,) or std.shape != (375,):
        raise ValueError(f"norm stats must have shape (375,), got mean={mean.shape}, std={std.shape}")
    if np.any(std <= 0):
        raise ValueError("norm std must be > 0 for all dims")
    return mean, std


__all__ = [
    "MEMMAP_LABELS_FILE",
    "MEMMAP_META_FILE",
    "MEMMAP_STYLES_FILE",
    "MEMMAP_X_FILE",
    "build_adamw",
    "build_dataloader",
    "configure_torch_runtime",
    "estimate_path_size_bytes",
    "load_checkpoint",
    "load_state_dict_with_allowed_missing",
    "load_norm_stats",
    "resolve_amp_dtype",
    "resolve_device",
    "sanitize_state_dict_keys",
    "save_norm_stats",
    "save_norm_stats_if_present",
    "set_seed",
    "strip_state_dict_prefix_if_present",
    "subset_dataset_by_fraction",
    "to_float",
    "validate_memmap_db_dir",
]
