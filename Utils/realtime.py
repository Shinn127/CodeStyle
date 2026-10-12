from __future__ import annotations

import difflib
import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from Utils.runtime import (
    MEMMAP_LABELS_FILE,
    MEMMAP_META_FILE,
    MEMMAP_STYLES_FILE,
    MEMMAP_X_FILE,
)


def parse_style_selector(selector: Optional[str], style_names: Sequence[str]) -> Optional[int]:
    if selector is None:
        return None
    text = selector.strip()
    if text == "":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        label = int(text)
        if not (0 <= label < len(style_names)):
            raise ValueError(f"Style label out of range: {label}, valid [0, {len(style_names)-1}]")
        return label
    if text in style_names:
        return style_names.index(text)
    lower_map = {name.lower(): idx for idx, name in enumerate(style_names)}
    key = text.lower()
    if key in lower_map:
        return lower_map[key]

    contains_matches = [name for name in style_names if key in name.lower()]
    close_matches = difflib.get_close_matches(text, style_names, n=5, cutoff=0.5)
    suggestions = contains_matches[:5]
    for name in close_matches:
        if name not in suggestions:
            suggestions.append(name)

    message = f"Style '{selector}' not found in style names"
    if suggestions:
        message += f". Did you mean: {', '.join(suggestions)}"
    else:
        message += f". Available styles include: {', '.join(style_names[:10])}, ..."
    raise ValueError(message)


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if labels.size == 0:
        return runs
    start = 0
    current = int(labels[0])
    for idx in range(1, labels.shape[0]):
        value = int(labels[idx])
        if value != current:
            runs.append((start, idx, current))
            start = idx
            current = value
    runs.append((start, labels.shape[0], current))
    return runs


def choose_segment(
    labels: np.ndarray,
    num_frames: int,
    seed: int,
    target_label: Optional[int] = None,
    start_index: Optional[int] = None,
) -> tuple[int, int, int, tuple[int, int, int]]:
    runs = contiguous_runs(labels)
    if start_index is not None:
        start = int(start_index)
        end = start + num_frames
        if start < 0 or end > labels.shape[0]:
            raise ValueError(
                f"--start-index out of range: [{start}, {end}) with total frames {labels.shape[0]}"
            )
        label0 = int(labels[start])
        if np.any(labels[start:end] != label0):
            raise ValueError(
                f"Segment [{start},{end}) crosses style label boundary. Use a different --start-index."
            )
        if target_label is not None and label0 != target_label:
            raise ValueError(
                f"--start-index segment label {label0} does not match requested style {target_label}"
            )
        return start, end, label0, (start, end, label0)

    rng = random.Random(seed)
    candidates: list[tuple[int, int, int]] = []
    for run_start, run_end, label in runs:
        if run_end - run_start < num_frames:
            continue
        if target_label is not None and label != target_label:
            continue
        candidates.append((run_start, run_end, label))
    if not candidates:
        max_len = max((run_end - run_start for run_start, run_end, _ in runs), default=0)
        raise RuntimeError(
            f"No contiguous run can provide {num_frames} frames for style {target_label}. Max run length={max_len}"
        )
    run_start, run_end, label = rng.choice(candidates)
    start = rng.randint(run_start, run_end - num_frames)
    end = start + num_frames
    return start, end, label, (run_start, run_end, label)


def load_db_arrays(db_path: Path) -> tuple[np.memmap, np.ndarray, list[str]]:
    if not db_path.is_dir():
        raise ValueError(f"Expected --db to be a memmap directory, got: {db_path}")

    meta_path = db_path / MEMMAP_META_FILE
    x_path = db_path / MEMMAP_X_FILE
    labels_path = db_path / MEMMAP_LABELS_FILE
    if not (meta_path.exists() and x_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"Memmap DB must include {MEMMAP_META_FILE}, {MEMMAP_X_FILE}, {MEMMAP_LABELS_FILE}"
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    num_frames = int(meta.get("num_frames", -1))
    feature_dim = int(meta.get("feature_dim", -1))
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames in {meta_path}: {num_frames}")
    if feature_dim != 375:
        raise ValueError(f"Invalid feature_dim in {meta_path}: {feature_dim}, expected 375")

    x_source = np.memmap(x_path, dtype=np.float32, mode="r", shape=(num_frames, 375))
    labels_mm = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(num_frames,))
    labels = np.asarray(labels_mm, dtype=np.int64)
    styles_path = db_path / MEMMAP_STYLES_FILE
    if styles_path.exists():
        with styles_path.open("r", encoding="utf-8") as handle:
            style_names = list(json.load(handle))
    else:
        max_label = int(labels.max()) if labels.size else -1
        style_names = [f"label_{i}" for i in range(max_label + 1)]
    return x_source, labels, style_names


def resolve_db_coordinate_convert(db_path: Path) -> bool:
    db_path = db_path.expanduser().resolve()
    if not db_path.is_dir():
        return False
    meta_path = db_path / MEMMAP_META_FILE
    if not meta_path.exists():
        return False
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return bool(meta.get("coordinate_convert", False))


def prepare_output_dir(out_dir: Path) -> Path:
    resolved = out_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_motion_pair_npz(
    *,
    out_dir: Path,
    prefix: str,
    primary_suffix: str,
    primary_x: np.ndarray,
    reference_x: np.ndarray,
    primary_label: int,
    reference_label: int,
    style_names: Sequence[str],
    coordinate_convert: bool | None = None,
) -> tuple[Path, Path]:
    out_dir = prepare_output_dir(out_dir)
    primary_path = out_dir / f"{prefix}.{primary_suffix}.npz"
    reference_path = out_dir / f"{prefix}.reference.npz"
    style_names_np = np.asarray(style_names, dtype=np.str_)
    coord_value = np.asarray(False if coordinate_convert is None else bool(coordinate_convert))

    np.savez_compressed(
        primary_path,
        X=np.asarray(primary_x, dtype=np.float32),
        Labels=np.full((primary_x.shape[0],), primary_label, dtype=np.int32),
        StyleNames=style_names_np,
        coordinate_convert=coord_value,
    )
    np.savez_compressed(
        reference_path,
        X=np.asarray(reference_x, dtype=np.float32),
        Labels=np.full((reference_x.shape[0],), reference_label, dtype=np.int32),
        StyleNames=style_names_np,
        coordinate_convert=coord_value,
    )
    return primary_path, reference_path


def write_generated_reference_npz(
    *,
    out_dir: Path,
    prefix: str,
    generated_x: np.ndarray,
    reference_x: np.ndarray,
    generated_label: int,
    reference_label: int,
    style_names: Sequence[str],
    coordinate_convert: bool | None = None,
) -> tuple[Path, Path]:
    return _write_motion_pair_npz(
        out_dir=out_dir,
        prefix=prefix,
        primary_suffix="generated",
        primary_x=generated_x,
        reference_x=reference_x,
        primary_label=generated_label,
        reference_label=reference_label,
        style_names=style_names,
        coordinate_convert=coordinate_convert,
    )


def write_reconstructed_reference_npz(
    *,
    out_dir: Path,
    prefix: str,
    reconstructed_x: np.ndarray,
    reference_x: np.ndarray,
    reconstructed_label: int,
    reference_label: int,
    style_names: Sequence[str],
    coordinate_convert: bool | None = None,
) -> tuple[Path, Path]:
    return _write_motion_pair_npz(
        out_dir=out_dir,
        prefix=prefix,
        primary_suffix="reconstructed",
        primary_x=reconstructed_x,
        reference_x=reference_x,
        primary_label=reconstructed_label,
        reference_label=reference_label,
        style_names=style_names,
        coordinate_convert=coordinate_convert,
    )


def write_meta_json(
    *,
    out_dir: Path,
    prefix: str,
    meta: dict,
) -> Path:
    out_dir = prepare_output_dir(out_dir)
    meta_path = out_dir / f"{prefix}.meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return meta_path


__all__ = [
    "choose_segment",
    "contiguous_runs",
    "load_db_arrays",
    "parse_style_selector",
    "prepare_output_dir",
    "resolve_db_coordinate_convert",
    "write_generated_reference_npz",
    "write_reconstructed_reference_npz",
    "write_meta_json",
]
