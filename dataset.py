"""
Dataset/DataLoader helpers for 100STYLE 375-dim memmap databases.

Required memmap layout:
  - meta.json
  - X.f32
  - Labels.i32
Optional:
  - StyleNames.json
  - segments.json

`load_motion375_metadata()` exposes validated, typed access to core metadata.

Normalization:
  - optional z-score normalization on 375 dims
  - stats are estimated from TRAIN sample windows and shared with val/test
  - pass `norm_stats_path` to persist them, so a resumed or re-evaluated run reuses
    exactly the same statistics instead of silently recomputing them

Splits (`split_mode`):
  - "auto"        precomputed splits if `segments.json` carries them, otherwise
                  leak-free clip-level splitting (zero-shot databases get no split)
  - "precomputed" require offline split metadata; every segment must carry a split
  - "clip"        force clip-level splitting (adjacent windows never straddle splits)
  - "none"        a single train dataset over every window (no val/test)

Splits are always assigned to whole clips, never to individual overlapping windows, so no
window in the validation or test split can overlap a window in the training split. Training
databases are unmirrored by policy (mirrored segments were found to distort the mesh), so a
source clip normally contributes exactly one clip per split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Utils.runtime import load_norm_stats as _load_norm_stats_file
from Utils.runtime import save_norm_stats as _save_norm_stats_file


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"
MEMMAP_SEGMENTS_FILE = "segments.json"
MEMMAP_FORMAT = "motion375_memmap"
MEMMAP_SCHEMA_VERSION = 4

SPLIT_MODES = ("auto", "precomputed", "clip", "none")
DATASET_ROLE_ZERO_SHOT = "zero_shot"


@dataclass(frozen=True, slots=True)
class Motion375Metadata:
    db_path: Path
    format: str
    schema_version: int
    num_frames: int
    feature_dim: int
    num_styles: int | None
    num_segments: int | None
    dataset_role: str
    split_strategy: str | None
    raw: dict[str, object]


def load_motion375_metadata(memmap_dir: str | Path) -> Motion375Metadata:
    memmap_dir = Path(memmap_dir).expanduser().resolve()
    meta_path = memmap_dir / MEMMAP_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Memmap meta not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path} must contain a JSON object")

    schema_version = int(meta.get("schema_version", 0))
    if schema_version != MEMMAP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version} in {meta_path}; this codebase "
            f"reads schema v{MEMMAP_SCHEMA_VERSION} only. Rebuild the database with "
            f"generate_database_100sty.py."
        )
    if meta.get("format") != MEMMAP_FORMAT:
        raise ValueError(
            f"Unsupported memmap format in {meta_path}: {meta.get('format')!r}"
        )
    expected_files = {
        "features": (MEMMAP_X_FILE, "float32"),
        "labels": (MEMMAP_LABELS_FILE, "int32"),
        "style_names": (MEMMAP_STYLES_FILE, None),
        "segments": (MEMMAP_SEGMENTS_FILE, None),
    }
    files = meta.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"schema v{schema_version} metadata requires a files object")
    for key, (expected_path, expected_dtype) in expected_files.items():
        descriptor = files.get(key)
        if not isinstance(descriptor, dict) or descriptor.get("path") != expected_path:
            raise ValueError(
                f"Invalid files.{key}.path in {meta_path}: expected {expected_path!r}"
            )
        if expected_dtype is not None and descriptor.get("dtype") != expected_dtype:
            raise ValueError(
                f"Invalid files.{key}.dtype in {meta_path}: expected {expected_dtype!r}"
            )

    feature_layout = meta.get("feature_layout")
    expected_layout = {
        "trajectory": {"offset": 0, "dim": 84},
        "motion": {"offset": 84, "dim": 291},
    }
    if feature_layout != expected_layout:
        raise ValueError(
            f"Invalid feature_layout in {meta_path}: expected {expected_layout}"
        )
    # v4 records the segment counts and split summary
    # variant contributed; a database without them is either truncated or hand-edited.
    for key in ("segment_counts",):
        if key not in meta:
            raise ValueError(f"schema v{schema_version} metadata requires {key!r} in {meta_path}")
    split_meta = meta.get("split")
    split_strategy = (
        str(split_meta.get("strategy"))
        if isinstance(split_meta, dict) and split_meta.get("strategy") is not None
        else None
    )
    return Motion375Metadata(
        db_path=memmap_dir,
        format=str(meta.get("format", "")),
        schema_version=schema_version,
        num_frames=int(meta.get("num_frames", -1)),
        feature_dim=int(meta.get("feature_dim", -1)),
        num_styles=int(meta["num_styles"]) if "num_styles" in meta else None,
        num_segments=int(meta["num_segments"]) if "num_segments" in meta else None,
        dataset_role=str(meta.get("dataset_role", "")),
        split_strategy=split_strategy,
        raw=meta,
    )


def _load_memmap_segments(memmap_dir: Path, *, num_frames: int) -> list[dict[str, object]] | None:
    segments_path = memmap_dir / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        return None

    with segments_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{segments_path} must contain a JSON list")

    segments: list[dict[str, object]] = []
    for idx, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid segment record at index {idx}: expected object, got {type(raw)}")
        if "frame_start" not in raw or "num_frames" not in raw or "label" not in raw:
            raise KeyError(f"Segment record {idx} must contain frame_start, num_frames, label")

        frame_start = int(raw["frame_start"])
        num_segment_frames = int(raw["num_frames"])
        label = int(raw["label"])
        if frame_start < 0:
            raise ValueError(f"Segment record {idx} has negative frame_start: {frame_start}")
        if num_segment_frames < 0:
            raise ValueError(f"Segment record {idx} has negative num_frames: {num_segment_frames}")
        if frame_start + num_segment_frames > num_frames:
            raise ValueError(
                f"Segment record {idx} exceeds DB bounds: start={frame_start} len={num_segment_frames} "
                f"num_frames={num_frames}"
            )

        record = {
            "frame_start": frame_start,
            "num_frames": num_segment_frames,
            "label": label,
        }
        for key in (
            "split",
            "style_name",
            "source_entry_id",
            "source_path",
            "suffix",

            "source_start",
            "source_stop",
            "source_global_label",
        ):
            if key in raw:
                record[key] = raw[key]
        segments.append(record)

    segments.sort(key=lambda item: (int(item["frame_start"]), int(item["num_frames"])))
    prev_end = 0
    for idx, record in enumerate(segments):
        frame_start = int(record["frame_start"])
        num_segment_frames = int(record["num_frames"])
        if idx > 0 and frame_start < prev_end:
            raise ValueError(f"Overlapping segments detected near frame {frame_start} in {segments_path}")
        prev_end = frame_start + num_segment_frames
    return segments


def _build_sample_starts(
    labels: np.ndarray,
    seq_len: int,
    stride: int,
    enforce_same_label: bool,
) -> np.ndarray:
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    n = int(labels.shape[0])
    if n < seq_len:
        return np.empty((0,), dtype=np.int64)

    if seq_len == 1:
        return np.arange(0, n, stride, dtype=np.int64)

    if not enforce_same_label:
        return np.arange(0, n - seq_len + 1, stride, dtype=np.int64)

    boundaries = np.flatnonzero(labels[1:] != labels[:-1]).astype(np.int64) + 1
    run_starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    run_ends = np.concatenate((boundaries, np.array([n], dtype=np.int64)))
    parts = [
        np.arange(begin, end - seq_len + 1, stride, dtype=np.int64)
        for begin, end in zip(run_starts.tolist(), run_ends.tolist())
        if (end - begin) >= seq_len
    ]
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def _build_sample_starts_from_segments(
    segments: list[dict[str, object]],
    *,
    seq_len: int,
    stride: int,
) -> np.ndarray:
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    parts: list[np.ndarray] = []
    for segment in segments:
        frame_start = int(segment["frame_start"])
        num_frames = int(segment["num_frames"])
        if num_frames < seq_len:
            continue
        parts.append(
            np.arange(
                frame_start,
                frame_start + num_frames - seq_len + 1,
                stride,
                dtype=np.int64,
            )
        )
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def _validate_segments_against_labels(
    segments: list[dict[str, object]],
    labels: np.ndarray,
) -> None:
    """Each segment must stay inside a single style run and declare its true label.

    Windows are built per segment, so a segment that crossed a style boundary would
    silently produce samples whose `label` disagrees with most of their frames.
    """
    if not segments:
        return
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]).astype(np.int64) + 1
    for idx, segment in enumerate(segments):
        frame_start = int(segment["frame_start"])
        num_frames = int(segment["num_frames"])
        if num_frames <= 0:
            continue
        declared = int(segment["label"])
        actual = int(labels[frame_start])
        if actual != declared:
            raise ValueError(
                f"Segment {idx} starts at frame {frame_start} and declares label {declared}, "
                f"but Labels.i32 holds {actual}"
            )
        runs_spanned = np.searchsorted(boundaries, frame_start + num_frames - 1, side="right") - np.searchsorted(
            boundaries, frame_start, side="right"
        )
        if runs_spanned != 0:
            raise ValueError(
                f"Segment {idx} (frame_start={frame_start}, num_frames={num_frames}) spans a style "
                f"boundary in Labels.i32; windows built from it would mix styles"
            )


def _assert_splits_disjoint(
    split_segments: dict[str, list[dict[str, object]]],
) -> None:
    """Precomputed splits must not share frames, otherwise evaluation leaks.

    Overlapping segments inside one split are already rejected by
    `_load_memmap_segments`; this checks the cross-split case that metadata can get wrong.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    for name, segments in split_segments.items():
        spans = [(int(s["frame_start"]), int(s["frame_start"]) + int(s["num_frames"])) for s in segments]
        ranges[name] = sorted(spans)
    names = sorted(ranges)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_spans, right_spans = ranges[left], ranges[right]
            j = 0
            for start, end in left_spans:
                while j < len(right_spans) and right_spans[j][1] <= start:
                    j += 1
                if j < len(right_spans) and right_spans[j][0] < end:
                    raise ValueError(
                        f"Offline splits {left!r} and {right!r} share frames around "
                        f"{max(start, right_spans[j][0])}; refusing to build a leaking split"
                    )


@dataclass(frozen=True)
class ClipGroups:
    """Window -> clip grouping used to assign splits without leaking overlapping windows."""

    window_group: np.ndarray
    group_label: np.ndarray
    group_size: np.ndarray
    source: str


def _resolve_clip_groups(
    *,
    labels: np.ndarray,
    sample_starts: np.ndarray,
    segments: list[dict[str, object]] | None,
) -> ClipGroups:
    """Group sample windows by their source clip.

    Several clips can come from one source BVH file, so grouping by `source_entry_id`
    keeps every clip of a file in a single split - exactly what the database generator
    does when it assigns offline splits.
    """
    starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return ClipGroups(
            window_group=np.empty((0,), dtype=np.int64),
            group_label=np.empty((0,), dtype=np.int64),
            group_size=np.empty((0,), dtype=np.int64),
            source="empty",
        )

    if segments:
        frame_starts = np.array([int(s["frame_start"]) for s in segments], dtype=np.int64)
        seg_index = np.searchsorted(frame_starts, starts, side="right") - 1
        seg_index = np.clip(seg_index, 0, len(segments) - 1)

        has_source_id = all("source_entry_id" in segment for segment in segments)
        seg_keys = np.array(
            [
                int(segment["source_entry_id"]) if has_source_id else idx
                for idx, segment in enumerate(segments)
            ],
            dtype=np.int64,
        )
        keys = seg_keys[seg_index]
        uniq, inverse = np.unique(keys, return_inverse=True)
        window_group = inverse.astype(np.int64)
        seg_labels = np.array([int(s["label"]) for s in segments], dtype=np.int64)
        # Every window of a group comes from one source clip, hence one style.
        group_label = np.zeros((uniq.size,), dtype=np.int64)
        group_label[window_group] = seg_labels[seg_index]
        group_size = np.bincount(window_group, minlength=uniq.size).astype(np.int64)
        return ClipGroups(
            window_group=window_group,
            group_label=group_label,
            group_size=group_size,
            source="segments:source_entry_id" if has_source_id else "segments:index",
        )

    # No segment metadata: fall back to contiguous same-label runs, which is the finest
    # clip granularity recoverable from Labels.i32 alone.
    runs = np.flatnonzero(labels[1:] != labels[:-1]).astype(np.int64) + 1
    run_starts = np.concatenate((np.array([0], dtype=np.int64), runs))
    window_group = (np.searchsorted(run_starts, starts, side="right") - 1).astype(np.int64)
    group_size = np.bincount(window_group, minlength=run_starts.size).astype(np.int64)
    group_label = np.asarray(labels[run_starts], dtype=np.int64)
    return ClipGroups(
        window_group=window_group,
        group_label=group_label,
        group_size=group_size,
        source="labels:contiguous_runs",
    )


def _assign_clip_splits(
    *,
    group_label: np.ndarray,
    group_size: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> np.ndarray:
    """Assign each clip group to train/val/test, stratified per style and frame-balanced.

    Legacy fallback for databases without recorded splits: a frame-count greedy balanced
    assignment. Production databases carry the generator's *suffix-balanced* offline split in
    `segments.json` (`split_mode='auto'`/`'precomputed'` uses it), which holds out whole
    locomotion types and keeps val/test action-balanced; the greedy balance used here is
    length-driven and biases short running clips into val/test, so prefer precomputed splits.
    """
    num_groups = int(group_label.shape[0])
    codes = np.zeros((num_groups,), dtype=np.int8)  # 0=train, 1=val, 2=test
    if num_groups == 0:
        return codes

    for style_index, style in enumerate(np.unique(group_label).tolist()):
        member_ids = np.flatnonzero(group_label == style)
        n_val, n_test = _resolve_split_counts(int(member_ids.size), val_ratio, test_ratio)
        if n_val == 0 and n_test == 0:
            continue
        remaining = {
            "train": int(member_ids.size) - n_val - n_test,
            "val": n_val,
            "test": n_test,
        }
        sizes = group_size[member_ids].astype(np.float64)
        total = float(sizes.sum())
        targets = {
            "train": total * (1.0 - val_ratio - test_ratio),
            "val": total * val_ratio if n_val else 0.0,
            "test": total * test_ratio if n_test else 0.0,
        }
        assigned = {"train": 0.0, "val": 0.0, "test": 0.0}
        priority = {"val": 2, "test": 1, "train": 0}
        code_of = {"train": 0, "val": 1, "test": 2}

        order = np.random.default_rng(seed + style_index * 1009).permutation(member_ids.size)
        ordered = member_ids[order]
        ordered = ordered[np.argsort(-sizes[order], kind="stable")]

        for group_id in ordered.tolist():
            choices = [name for name in ("val", "test", "train") if remaining[name] > 0]
            best = max(
                choices,
                key=lambda name: (
                    targets[name] - assigned[name],
                    priority[name],
                    remaining[name],
                    -assigned[name],
                ),
            )
            codes[group_id] = code_of[best]
            remaining[best] -= 1
            assigned[best] += float(group_size[group_id])
    return codes


def _split_sample_starts_by_clip(
    *,
    labels: np.ndarray,
    sample_starts: np.ndarray,
    segments: list[dict[str, object]] | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], ClipGroups]:
    groups = _resolve_clip_groups(labels=labels, sample_starts=sample_starts, segments=segments)
    starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return starts.copy(), None, None, groups

    codes = _assign_clip_splits(
        group_label=groups.group_label,
        group_size=groups.group_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    window_codes = codes[groups.window_group]
    train_starts = starts[window_codes == 0]
    val_starts = starts[window_codes == 1]
    test_starts = starts[window_codes == 2]
    return (
        train_starts,
        val_starts if val_starts.size else None,
        test_starts if test_starts.size else None,
        groups,
    )


def norm_stats_fingerprint(mean: np.ndarray, std: np.ndarray) -> str:
    """Short stable digest of normalization stats, for logging and mismatch checks."""
    digest = hashlib.sha256()
    for array in (np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)):
        rounded = np.round(array, 6)
        digest.update(np.ascontiguousarray(rounded).tobytes())
    return digest.hexdigest()[:16]


def _validate_norm_stats(norm_mean: np.ndarray, norm_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(norm_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(norm_std, dtype=np.float32).reshape(-1)
    if mean.shape != (375,):
        raise ValueError(f"norm_mean must have shape (375,), got {mean.shape}")
    if std.shape != (375,):
        raise ValueError(f"norm_std must have shape (375,), got {std.shape}")
    if np.any(std <= 0):
        raise ValueError("norm_std must be positive for all dims")
    return mean, std


def _compute_zscore_stats_from_starts(
    sample_starts: np.ndarray,
    seq_len: int,
    norm_eps: float,
    max_windows_for_stats: int = 0,
    seed: int = 42,
    x_source=None,
    db_path: Optional[Path] = None,
    windows_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    del db_path

    starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        raise ValueError("Cannot compute normalization stats from empty sample_starts")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if norm_eps <= 0:
        raise ValueError(f"norm_eps must be > 0, got {norm_eps}")
    if windows_chunk_size < 1:
        raise ValueError(f"windows_chunk_size must be >= 1, got {windows_chunk_size}")
    if x_source is None:
        raise ValueError("x_source is required for memmap normalization-stat computation")

    if max_windows_for_stats > 0 and starts.size > max_windows_for_stats:
        rng = np.random.default_rng(seed)
        keep = rng.choice(starts.size, size=max_windows_for_stats, replace=False)
        starts = starts[keep]
    starts = np.sort(starts)

    num_frames = int(x_source.shape[0])
    frame_weight_delta = np.zeros((num_frames + 1,), dtype=np.int64)
    np.add.at(frame_weight_delta, starts, 1)
    np.add.at(frame_weight_delta, starts + seq_len, -1)
    np.cumsum(frame_weight_delta[:-1], out=frame_weight_delta[:-1])
    frame_weights = frame_weight_delta[:-1]
    weighted_frame_ids = np.flatnonzero(frame_weights)

    sum_x = np.zeros((375,), dtype=np.float64)
    sum_x2 = np.zeros((375,), dtype=np.float64)
    count = int(frame_weights.sum(dtype=np.int64))
    for offset in range(0, weighted_frame_ids.size, windows_chunk_size):
        frame_ids = weighted_frame_ids[offset : offset + windows_chunk_size]
        block = np.asarray(x_source[frame_ids], dtype=np.float64).reshape(-1, 375)
        weights = frame_weights[frame_ids].astype(np.float64, copy=False)
        sum_x += weights @ block
        np.square(block, out=block)
        sum_x2 += weights @ block

    if count == 0:
        raise ValueError("No samples found while computing normalization stats")

    mean = sum_x / count
    var = sum_x2 / count - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, norm_eps)
    return mean.astype(np.float32), std.astype(np.float32)


class Motion375Dataset(Dataset):
    """
    Read 375-dim features from memmap and emit frame/sequence samples.

    Returned item fields:
      - x:          [375] or [T, 375]
      - traj:       [84]  or [T, 84]
      - motion:     [291] or [T, 291]
      - label:      scalar int64 style id
      - start:      scalar int64 start frame index in original database
      - segment_id: scalar int64 index into segments.json, or -1 when unavailable
    """

    def __init__(
        self,
        db_path: str | Path,
        seq_len: int = 1,
        stride: int = 1,
        enforce_same_label: bool = True,
        sample_starts: Optional[np.ndarray] = None,
        load_into_memory: bool = False,
        normalize: bool = False,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        norm_eps: float = 1e-8,
        max_windows_for_stats: int = 0,
        stats_seed: int = 42,
        _shared_dataset: Optional["Motion375Dataset"] = None,
    ):

        super().__init__()
        self.db_path = Path(db_path).expanduser().resolve()
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.enforce_same_label = bool(enforce_same_label)
        self.load_into_memory = bool(load_into_memory)
        self.normalize = bool(normalize)
        self.norm_eps = float(norm_eps)
        self.max_windows_for_stats = int(max_windows_for_stats)
        self.stats_seed = int(stats_seed)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path not found: {self.db_path}")
        if not self.db_path.is_dir():
            raise ValueError(f"Unsupported db path: {self.db_path}. Expected memmap directory.")

        if _shared_dataset is not None:
            if _shared_dataset.db_path != self.db_path:
                raise ValueError("_shared_dataset must refer to the same database path")
            if _shared_dataset.load_into_memory != self.load_into_memory:
                raise ValueError("_shared_dataset load_into_memory setting must match")
            self.num_frames = _shared_dataset.num_frames
            self.labels = _shared_dataset.labels
            self.style_names = _shared_dataset.style_names
            self.segments = _shared_dataset.segments
            self.metadata = _shared_dataset.metadata
            self.dataset_role = _shared_dataset.dataset_role
            self.has_precomputed_splits = _shared_dataset.has_precomputed_splits
            self._x_mem = _shared_dataset._x_mem
        else:
            metadata = load_motion375_metadata(self.db_path)
            meta = metadata.raw
            num_frames = metadata.num_frames
            feature_dim = metadata.feature_dim
            if num_frames < 0:
                raise ValueError(
                    f"Invalid num_frames in {self.db_path / MEMMAP_META_FILE}: {num_frames}"
                )
            if feature_dim != 375:
                raise ValueError(
                    f"Expected feature_dim=375 in {self.db_path / MEMMAP_META_FILE}, got {feature_dim}"
                )

            x_path = self.db_path / MEMMAP_X_FILE
            labels_path = self.db_path / MEMMAP_LABELS_FILE
            if not x_path.exists() or not labels_path.exists():
                raise FileNotFoundError(
                    f"Memmap files missing under {self.db_path}: "
                    f"need {MEMMAP_X_FILE} and {MEMMAP_LABELS_FILE}"
                )
            expected_x_bytes = num_frames * feature_dim * np.dtype(np.float32).itemsize
            expected_labels_bytes = num_frames * np.dtype(np.int32).itemsize
            if x_path.stat().st_size != expected_x_bytes:
                raise ValueError(
                    f"Invalid {x_path}: expected {expected_x_bytes} bytes, "
                    f"got {x_path.stat().st_size}"
                )
            if labels_path.stat().st_size != expected_labels_bytes:
                raise ValueError(
                    f"Invalid {labels_path}: expected {expected_labels_bytes} bytes, "
                    f"got {labels_path.stat().st_size}"
                )
            if metadata.schema_version >= MEMMAP_SCHEMA_VERSION:
                files = meta["files"]
                if files["features"].get("shape") != [num_frames, feature_dim]:
                    raise ValueError("meta files.features.shape does not match num_frames/feature_dim")
                if files["labels"].get("shape") != [num_frames]:
                    raise ValueError("meta files.labels.shape does not match num_frames")

            self.num_frames = num_frames
            labels_mm = np.memmap(labels_path, dtype=np.int32, mode="r", shape=(self.num_frames,))
            # int32 keeps each DataLoader worker's resident copy at 4 bytes/frame; the
            # emitted `label` tensor is still cast to torch.long.
            self.labels = np.array(labels_mm, dtype=np.int32, copy=True)

            styles_path = self.db_path / MEMMAP_STYLES_FILE
            if styles_path.exists():
                with styles_path.open("r", encoding="utf-8") as f:
                    self.style_names = list(json.load(f))
            else:
                max_label = int(self.labels.max()) if self.labels.size else -1
                self.style_names = [f"label_{i}" for i in range(max_label + 1)]
                print(
                    f"[Warn] {MEMMAP_STYLES_FILE} missing in {self.db_path}; "
                    f"style names fall back to label_0..label_{max_label}, so any mapping "
                    f"from style names to labels is unverified"
                )

            self.metadata = metadata
            self.dataset_role = metadata.dataset_role
            self.segments = _load_memmap_segments(self.db_path, num_frames=self.num_frames)
            _validate_segments_against_labels(self.segments or [], self.labels)
            if metadata.num_styles != len(self.style_names):
                raise ValueError("meta num_styles does not match StyleNames.json")
            expected_segments = len(self.segments) if self.segments is not None else 0
            if metadata.num_segments != expected_segments:
                raise ValueError("meta num_segments does not match segments.json")
            self.has_precomputed_splits = bool(
                self.segments
                and any(
                    str(segment.get("split", "")) in {"train", "val", "test"}
                    for segment in self.segments
                )
            )

            if self.load_into_memory:
                x_mm = np.memmap(
                    x_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(self.num_frames, 375),
                )
                self._x_mem = np.asarray(x_mm, dtype=np.float32).copy()
            else:
                self._x_mem = None

        self._segment_frame_starts: Optional[np.ndarray] = None
        self._segment_num_frames: Optional[np.ndarray] = None
        if self.segments:
            self._segment_frame_starts = np.array(
                [int(s["frame_start"]) for s in self.segments], dtype=np.int64
            )
            self._segment_num_frames = np.array(
                [int(s["num_frames"]) for s in self.segments], dtype=np.int64
            )

        if sample_starts is None:
            if self.segments is not None:
                # Windows are built inside each segment, so every sample already stays
                # within one style; `enforce_same_label` needs no further work here.
                self.sample_starts = _build_sample_starts_from_segments(
                    self.segments,
                    seq_len=self.seq_len,
                    stride=self.stride,
                )
            else:
                self.sample_starts = _build_sample_starts(
                    labels=self.labels,
                    seq_len=self.seq_len,
                    stride=self.stride,
                    enforce_same_label=self.enforce_same_label,
                )
        else:
            starts = np.asarray(sample_starts, dtype=np.int64).reshape(-1)
            if starts.size:
                if starts.min() < 0:
                    raise ValueError("sample_starts contains negative index")
                max_start = self.num_frames - self.seq_len
                if starts.max() > max_start:
                    raise ValueError(
                        f"sample_starts out of range: max start is {max_start}, got {int(starts.max())}"
                    )
                self._assert_starts_inside_segments(starts)
            self.sample_starts = starts

        self.segment_of_start = self._resolve_segment_index(self.sample_starts)

        self._x_ds = None
        # Set by build_train_val_test_datasets to record where this split came from.
        self.split_source = ""

        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        if self.normalize:
            if (norm_mean is None) ^ (norm_std is None):
                raise ValueError("norm_mean and norm_std must be both provided or both omitted")
            if norm_mean is None:
                self._ensure_open()
                stats_source = self._x_mem if self._x_mem is not None else self._x_ds
                mean, std = _compute_zscore_stats_from_starts(
                    sample_starts=self.sample_starts,
                    seq_len=self.seq_len,
                    norm_eps=self.norm_eps,
                    max_windows_for_stats=self.max_windows_for_stats,
                    seed=self.stats_seed,
                    x_source=stats_source,
                )
            else:
                mean, std = _validate_norm_stats(norm_mean, norm_std)
            self.norm_mean = mean
            self.norm_std = std

    def _resolve_segment_index(self, starts: np.ndarray) -> Optional[np.ndarray]:
        if self._segment_frame_starts is None or starts.size == 0:
            return None
        idx = np.searchsorted(self._segment_frame_starts, starts, side="right") - 1
        return np.clip(idx, 0, self._segment_frame_starts.size - 1).astype(np.int64)

    def _assert_starts_inside_segments(self, starts: np.ndarray) -> None:
        """Caller-supplied windows must not spill across a segment boundary."""
        idx = self._resolve_segment_index(starts)
        if idx is None or self._segment_frame_starts is None or self._segment_num_frames is None:
            return
        segment_ends = self._segment_frame_starts[idx] + self._segment_num_frames[idx]
        outside = starts < self._segment_frame_starts[idx]
        overflowing = (starts + self.seq_len) > segment_ends
        if np.any(outside | overflowing):
            bad = int(np.flatnonzero(outside | overflowing)[0])
            raise ValueError(
                f"sample_starts[{bad}]={int(starts[bad])} with seq_len={self.seq_len} crosses the "
                f"segment boundary at [{int(self._segment_frame_starts[idx[bad]])}, "
                f"{int(segment_ends[bad])})"
            )

    def segment_ids_for(self, indices: np.ndarray | list[int]) -> Optional[np.ndarray]:
        """Segment index per dataset row, or None when the database has no segments.json."""
        if self.segment_of_start is None:
            return None
        return self.segment_of_start[np.asarray(indices, dtype=np.int64)]

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_x_ds"] = None
        state['_controller_index_map'] = None
        return state

    def __del__(self):
        self._x_ds = None

    def _ensure_open(self):
        if self._x_mem is not None:
            return
        if self._x_ds is None:
            self._x_ds = self._open_memmap()

    def _open_memmap(self):
        x_path = self.db_path / MEMMAP_X_FILE
        return np.memmap(x_path, dtype=np.float32, mode="r", shape=(self.num_frames, 375))

    def _read_x(self, start: int, end: int) -> np.ndarray:
        if self._x_mem is not None:
            x = self._x_mem[start:end]
        else:
            self._ensure_open()
            x = self._x_ds[start:end]
        return np.asarray(x, dtype=np.float32)

    def _read_x_batch(self, starts: np.ndarray) -> np.ndarray:
        self._ensure_open()
        source = self._x_mem if self._x_mem is not None else self._x_ds
        frame_ids = starts[:, None] + np.arange(self.seq_len, dtype=np.int64)[None, :]
        return np.asarray(source[frame_ids], dtype=np.float32)

    def __getitem__(self, idx: int):
        start = int(self.sample_starts[idx])
        end = start + self.seq_len
        x = np.array(self._read_x(start, end), dtype=np.float32, copy=True)
        if self.seq_len == 1:
            x = x[0]

        if self.normalize:
            np.subtract(x, self.norm_mean, out=x)
            np.divide(x, self.norm_std, out=x)

        label = int(self.labels[start])
        x_t = torch.from_numpy(x)
        traj_t = x_t[..., :84]
        motion_t = x_t[..., 84:]

        return {
            "x": x_t,
            "traj": traj_t,
            "motion": motion_t,
            "label": torch.tensor(label, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
            "segment_id": torch.tensor(
                -1 if self.segment_of_start is None else int(self.segment_of_start[idx]),
                dtype=torch.long,
            ),
        }

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        if not indices:
            return []
        starts = np.asarray(self.sample_starts[np.asarray(indices, dtype=np.int64)], dtype=np.int64)
        x = self._read_x_batch(starts)
        if self.normalize:
            np.subtract(x, self.norm_mean, out=x)
            np.divide(x, self.norm_std, out=x)
        if self.seq_len == 1:
            x = x[:, 0]

        x_t = torch.from_numpy(x)
        labels_t = torch.from_numpy(np.asarray(self.labels[starts], dtype=np.int64))
        starts_t = torch.from_numpy(starts)
        if self.segment_of_start is None:
            segments_t = torch.full((len(indices),), -1, dtype=torch.long)
        else:
            segments_t = torch.from_numpy(
                np.asarray(self.segment_of_start[np.asarray(indices, dtype=np.int64)], dtype=np.int64)
            )
        if getattr(self, '_controller_batches', False):
            batch = dict(x=x_t, label=labels_t, start=starts_t, segment_id=segments_t)
            cache = getattr(self, '_controller_cache_spec', None)
            if cache is not None:
                if getattr(self, '_controller_index_map', None) is None:
                    path, shape, dtype = cache
                    self._controller_index_map = np.memmap(path, mode='r', dtype=dtype, shape=shape)
                ids = starts[:, None] + np.arange(self.seq_len)
                batch['vq_indices'] = torch.from_numpy(np.array(self._controller_index_map[ids], dtype=np.int64))
            return batch
        return [
            {
                "x": x_t[item_idx],
                "traj": x_t[item_idx, ..., :84],
                "motion": x_t[item_idx, ..., 84:],
                "label": labels_t[item_idx],
                "start": starts_t[item_idx],
                "segment_id": segments_t[item_idx],
            }
            for item_idx in range(len(indices))
        ]

    def enable_controller_batches(self, cache=None, *, pin_batches=False):
        """Only controller loaders opt in; workers assemble one tensor batch and cache rows.

        Pinning stays in the main process: calling CUDA allocation APIs in a forked worker
        after the parent has initialised CUDA raises `CUDA error: initialization error`.
        """
        self._controller_batches = True
        self._controller_pin_batches = bool(pin_batches)
        if cache is not None and cache.indices.shape[0] != self.num_frames:
            raise ValueError('Controller worker cache frame count differs from the database')
        self._controller_index_map = None
        self._controller_cache_spec = (str(cache.indices_path), tuple(cache.indices.shape),
                                       cache.indices.dtype) if cache is not None else None

    def get_norm_stats(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.norm_mean is None or self.norm_std is None:
            return None, None
        return self.norm_mean.copy(), self.norm_std.copy()


@dataclass
class DataBundle:
    train_dataset: Motion375Dataset
    val_dataset: Optional[Motion375Dataset]
    test_dataset: Optional[Motion375Dataset]
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    norm_mean: Optional[np.ndarray]
    norm_std: Optional[np.ndarray]
    split_mode: str = "auto"
    split_source: str = ""
    norm_fingerprint: str = ""
    norm_stats_path: Optional[Path] = None


def _validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")


def _resolve_split_counts(
    num_samples: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    if num_samples < 1:
        return 0, 0

    n_val = int(round(num_samples * val_ratio)) if val_ratio > 0.0 else 0
    n_test = int(round(num_samples * test_ratio)) if test_ratio > 0.0 else 0

    if val_ratio > 0.0:
        n_val = max(1, n_val)
    if test_ratio > 0.0:
        n_test = max(1, n_test)

    max_heldout = max(num_samples - 1, 0)
    while (n_val + n_test) > max_heldout:
        if n_test > 0 and (n_val == 0 or test_ratio <= val_ratio):
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def _precomputed_split_segments(
    segments: list[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
    if not segments:
        return None

    split_map = {"train": [], "val": [], "test": []}
    saw_precomputed = False
    for segment in segments:
        split = segment.get("split")
        if split in split_map:
            saw_precomputed = True
            split_map[str(split)].append(segment)
    if not saw_precomputed:
        return None
    return split_map["train"], split_map["val"], split_map["test"]


def _build_split_dataset(
    *,
    db_path: str | Path,
    seq_len: int,
    stride: int,
    enforce_same_label: bool,
    sample_starts: Optional[np.ndarray],
    load_into_memory: bool,
    normalize: bool,
    norm_mean: Optional[np.ndarray],
    norm_std: Optional[np.ndarray],
    norm_eps: float,
    max_windows_for_stats: int,
    seed: int,
    shared_dataset: Motion375Dataset,
) -> Optional[Motion375Dataset]:
    if sample_starts is None:
        return None
    return Motion375Dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=sample_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_seed=seed,
        _shared_dataset=shared_dataset,
    )


def _build_loader_for_split(
    dataset: Optional[Motion375Dataset],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Optional[DataLoader]:
    if dataset is None:
        return None
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def _resolve_split_mode(requested: str, base: Motion375Dataset) -> str:
    if requested != "auto":
        return requested
    if base.has_precomputed_splits:
        return "precomputed"
    if base.dataset_role == DATASET_ROLE_ZERO_SHOT:
        return "none"
    return "clip"


def _count_unsplit_segments(segments: list[dict[str, object]] | None) -> int:
    if not segments:
        return 0
    return sum(
        1 for segment in segments if str(segment.get("split", "")) not in {"train", "val", "test"}
    )


def build_train_val_test_datasets(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    split_mode: str = "auto",
    norm_stats_path: Optional[str | Path] = None,
) -> tuple[Motion375Dataset, Optional[Motion375Dataset], Optional[Motion375Dataset]]:
    """Build train/val/test datasets that never share overlapping windows.

    `split_mode="auto"` uses the offline clip-level splits recorded in `segments.json`
    when present, falls back to clip-level splitting computed here, and gives held-out
    (zero-shot) databases a single unsplit dataset. See the module docstring for the
    other modes.
    """
    _validate_split_ratios(val_ratio=val_ratio, test_ratio=test_ratio)
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")

    base = Motion375Dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=None,
        load_into_memory=load_into_memory,
        normalize=False,
    )

    resolved_norm_mean = None if norm_mean is None else np.asarray(norm_mean, dtype=np.float32)
    resolved_norm_std = None if norm_std is None else np.asarray(norm_std, dtype=np.float32)
    if (resolved_norm_mean is None) ^ (resolved_norm_std is None):
        raise ValueError("norm_mean and norm_std must be both provided or both omitted")

    if len(base) == 0:
        if normalize:
            raise RuntimeError("Cannot compute normalization stats: dataset has zero samples")
        return base, None, None

    resolved_mode = _resolve_split_mode(split_mode, base)
    split_source = ""

    if resolved_mode == "precomputed":
        precomputed = _precomputed_split_segments(base.segments)
        if precomputed is None:
            raise RuntimeError(
                f"split_mode={split_mode!r} requires offline train/val/test metadata in "
                f"{base.db_path / MEMMAP_SEGMENTS_FILE}, but none was found"
            )
        train_segments, val_segments, test_segments = precomputed
        _assert_splits_disjoint(
            {"train": train_segments, "val": val_segments, "test": test_segments}
        )
        unsplit = _count_unsplit_segments(base.segments)
        if unsplit:
            raise RuntimeError(
                f"{unsplit} of {len(base.segments or [])} segments in "
                f"{base.db_path / MEMMAP_SEGMENTS_FILE} carry no split. The in-distribution "
                f"database always assigns train/val/test to every segment; rebuild it with "
                f"generate_database_100sty.py."
            )
        train_starts = _build_sample_starts_from_segments(train_segments, seq_len=seq_len, stride=stride)
        val_starts = _build_sample_starts_from_segments(val_segments, seq_len=seq_len, stride=stride)
        test_starts = _build_sample_starts_from_segments(test_segments, seq_len=seq_len, stride=stride)
        val_starts = val_starts if val_starts.size > 0 else None
        test_starts = test_starts if test_starts.size > 0 else None
        split_source = f"segments.json#{base.metadata.split_strategy or 'unspecified'}"
        print(
            f"[Info] split_mode={resolved_mode} using {base.db_path / MEMMAP_SEGMENTS_FILE} "
            f"(strategy={base.metadata.split_strategy}); val_ratio/test_ratio/seed are ignored"
        )
    elif resolved_mode == "clip":
        train_starts, val_starts, test_starts, groups = _split_sample_starts_by_clip(
            labels=base.labels,
            sample_starts=base.sample_starts,
            segments=base.segments,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split_source = f"clip-level({groups.source},seed={seed})"
        print(
            f"[Info] split_mode=clip grouped windows by {groups.source}: "
            f"{groups.group_label.shape[0]} clips over {int(groups.group_label.max()) + 1 if groups.group_label.size else 0} "
            f"styles; no window is shared between splits"
        )
    elif resolved_mode == "none":
        train_starts = np.asarray(base.sample_starts, dtype=np.int64).reshape(-1)
        val_starts = None
        test_starts = None
        split_source = "single-split(all windows)"
        print(
            f"[Info] split_mode='none' (dataset_role={base.dataset_role or 'unknown'}): all "
            f"{train_starts.size} windows form one dataset, no val/test split is produced"
        )
    else:  # pragma: no cover - SPLIT_MODES is validated above
        raise ValueError(f"Unhandled split_mode: {resolved_mode!r}")

    if train_starts.size == 0:
        raise RuntimeError("Training split is empty after applying sample-window construction")

    requested_stats_path = (
        Path(norm_stats_path).expanduser().resolve() if norm_stats_path is not None else None
    )
    if normalize:
        if resolved_norm_mean is not None:
            resolved_norm_mean, resolved_norm_std = _validate_norm_stats(
                resolved_norm_mean, resolved_norm_std
            )
            print("[Info] normalization stats supplied by the caller")
        elif requested_stats_path is not None and requested_stats_path.exists():
            resolved_norm_mean, resolved_norm_std = _load_norm_stats_file(requested_stats_path)
            print(f"[Info] normalization stats loaded from {requested_stats_path}")
        else:
            base._ensure_open()
            stats_source = base._x_mem if base._x_mem is not None else base._x_ds
            resolved_norm_mean, resolved_norm_std = _compute_zscore_stats_from_starts(
                sample_starts=train_starts,
                seq_len=seq_len,
                norm_eps=norm_eps,
                max_windows_for_stats=max_windows_for_stats,
                seed=seed,
                x_source=stats_source,
                windows_chunk_size=stats_windows_chunk_size,
            )
            if requested_stats_path is not None:
                _save_norm_stats_file(requested_stats_path, resolved_norm_mean, resolved_norm_std)
                print(f"[Info] normalization stats written to {requested_stats_path}")
        print(
            f"[Info] normalization over {train_starts.size} train windows: "
            f"fingerprint={norm_stats_fingerprint(resolved_norm_mean, resolved_norm_std)}"
        )

    train_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=train_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        shared_dataset=base,
    )
    if train_dataset is None:
        raise RuntimeError("train split is unexpectedly empty")

    val_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=val_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        shared_dataset=base,
    )
    test_dataset = _build_split_dataset(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        sample_starts=test_starts,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=resolved_norm_mean,
        norm_std=resolved_norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        seed=seed,
        shared_dataset=base,
    )

    train_dataset.split_source = split_source
    if val_dataset is not None:
        val_dataset.split_source = split_source
    if test_dataset is not None:
        test_dataset.split_source = split_source

    base._x_ds = None
    return train_dataset, val_dataset, test_dataset


def build_train_val_datasets(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    split_mode: str = "auto",
    norm_stats_path: Optional[str | Path] = None,
) -> tuple[Motion375Dataset, Optional[Motion375Dataset]]:
    train_dataset, val_dataset, _ = build_train_val_test_datasets(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        split_mode=split_mode,
        norm_stats_path=norm_stats_path,
    )
    return train_dataset, val_dataset


def build_train_val_test_dataloaders(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    split_mode: str = "auto",
    norm_stats_path: Optional[str | Path] = None,
) -> DataBundle:
    if num_workers > 0 and prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1 when num_workers > 0, got {prefetch_factor}")

    train_dataset, val_dataset, test_dataset = build_train_val_test_datasets(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        split_mode=split_mode,
        norm_stats_path=norm_stats_path,
    )
    norm_mean, norm_std = train_dataset.get_norm_stats()

    train_loader = _build_loader_for_split(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _build_loader_for_split(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _build_loader_for_split(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    if train_loader is None:
        raise RuntimeError("train loader is unexpectedly None")

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        norm_mean=norm_mean,
        norm_std=norm_std,
        split_mode=split_mode,
        split_source=train_dataset.split_source,
        norm_fingerprint=(
            norm_stats_fingerprint(norm_mean, norm_std)
            if norm_mean is not None and norm_std is not None
            else ""
        ),
        norm_stats_path=(
            Path(norm_stats_path).expanduser().resolve() if norm_stats_path is not None else None
        ),
    )


def build_train_val_dataloaders(
    db_path: str | Path,
    seq_len: int = 1,
    stride: int = 1,
    enforce_same_label: bool = True,
    val_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    load_into_memory: bool = False,
    normalize: bool = True,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    norm_eps: float = 1e-8,
    max_windows_for_stats: int = 0,
    stats_windows_chunk_size: int = 4096,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    split_mode: str = "auto",
    norm_stats_path: Optional[str | Path] = None,
) -> DataBundle:
    return build_train_val_test_dataloaders(
        db_path=db_path,
        seq_len=seq_len,
        stride=stride,
        enforce_same_label=enforce_same_label,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        load_into_memory=load_into_memory,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        norm_eps=norm_eps,
        max_windows_for_stats=max_windows_for_stats,
        stats_windows_chunk_size=stats_windows_chunk_size,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        split_mode=split_mode,
        norm_stats_path=norm_stats_path,
    )


__all__ = [
    "Motion375Dataset",
    "Motion375Metadata",
    "DataBundle",
    "ClipGroups",
    "SPLIT_MODES",
    "load_motion375_metadata",
    "norm_stats_fingerprint",
    "build_train_val_test_datasets",
    "build_train_val_test_dataloaders",
    "build_train_val_datasets",
    "build_train_val_dataloaders",
]
