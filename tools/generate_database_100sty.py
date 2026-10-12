"""
Generate memmap-only 100STYLE databases with 375-dim trajectory+motion features.

Outputs (both the in-distribution and the zero-shot database):
  - meta.json
  - X.f32
  - Labels.i32
  - StyleNames.json
  - segments.json

Metadata schema v4 keeps core dimensions/counts at the top level and groups
file descriptors, feature layout, generation settings, split policy, and
related database paths into dedicated objects. Compared with v3 it additionally
records `context_stride`, the derived segment counts, and an input fingerprint
so a database can be traced back to the exact Frame_Cuts.csv and settings that
produced it.

Offline split policy:
  - the last K styles are peeled out as a zero-shot database
  - the remaining styles are split into train/val/test at clip level
  - split assignment is stratified per style and holds out whole locomotion types
    (suffixes), rotating by style so val/test cover every locomotion type about equally
  - every clip of one source BVH file stays in a single split

Mirrored augmentation was removed: the mirrored segments distorted the mesh, so
generation is unmirrored only.

Feature layout (fixed, validated downstream as well):
  - x[:, 0:84]    trajectory context: 7 blocks x `window // context_stride` samples
  - x[:, 84:375]  motion state of the PREVIOUS frame: root planar velocity/yaw rate
                  (3), selected local positions (72), local rotation frame columns
                  (72 + 72) and local velocities (72)

Data root resolution order when `--data-root` is not given:
  1. $MOTION375_DATA_ROOT
  2. <NewCode>/data/{100style,100sty}
  3. /home/shinn/桌面/Projects/data/{100style,100sty}
  4. /home/shinn/桌面/{multimedia/,}revision/data/{100style,100sty}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, TypeVar

import numpy as np
import scipy.signal as signal

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

        def set_description(self, desc: str) -> None:
            self.desc = desc
            self._render()

        def set_postfix(self, refresh: bool = True, **kwargs: object) -> None:
            self.postfix = " ".join(f"{key}={value}" for key, value in kwargs.items())
            if refresh:
                self._render()

        def update(self, n: int = 1) -> None:
            self.count = min(self.total, self.count + int(n)) if self.total else self.count + int(n)
            self._render()

        def write(self, message: str) -> None:
            sys.stderr.write(f"\n{message}\n")
            sys.stderr.flush()
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


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT / "Code", SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import Utils.bvh as bvh  # noqa: E402
from Utils import quat  # noqa: E402


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_LABELS_FILE = "Labels.i32"
MEMMAP_STYLES_FILE = "StyleNames.json"
MEMMAP_SEGMENTS_FILE = "segments.json"
FEATURE_DIM = 375
TRAJECTORY_DIM = 84
MOTION_DIM = 291
MEMMAP_SCHEMA_VERSION = 4
SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]
PREFERRED_SUFFIXES = ["BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"]
CONTEXT_STRIDE = 10
TRAJECTORY_BLOCKS = 7
SAVGOL_WINDOW = 61
SAVGOL_POLYORDER = 3
DATA_ROOT_SEARCH_BASES = ("/home/shinn/桌面/Projects/data",)
DATA_ROOT_DIR_NAMES = ("100style", "100sty")

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    label: int
    style_name: str
    split: str | None
    source_entry_id: int
    source_path: Path
    suffix: str
    source_start: int
    source_stop: int
    source_global_label: int


@dataclass(frozen=True, slots=True)
class EntryJob:
    path: Path
    style_name: str
    suffix: str
    source_entry_id: int
    source_global_label: int
    source_start: int
    source_stop: int
    label: int | None
    dataset_role: Literal["in_distribution", "zero_shot", "skip"]
    split: str | None
    coordinate_convert: bool
    window: int
    context_stride: int = CONTEXT_STRIDE


@dataclass(frozen=True, slots=True)
class EntryResult:
    job: EntryJob
    status: Literal["ok", "short", "skipped"]
    variants: tuple[np.ndarray, ...] = ()
    clip_len: int = 0


class MemmapWriter:
    def __init__(self, out_dir: Path, style_names: list[str], overwrite: bool) -> None:
        self.out_dir = out_dir
        self.style_names = list(style_names)
        self.num_frames = 0
        self.segments: list[dict[str, object]] = []
        self.meta_path = out_dir / MEMMAP_META_FILE
        self.x_path = out_dir / MEMMAP_X_FILE
        self.labels_path = out_dir / MEMMAP_LABELS_FILE
        self.styles_path = out_dir / MEMMAP_STYLES_FILE
        self.segments_path = out_dir / MEMMAP_SEGMENTS_FILE
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for path in (self.meta_path, self.x_path, self.labels_path, self.styles_path, self.segments_path):
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace.")
            if path.exists() and overwrite:
                path.unlink()

        self._x_file = self.x_path.open("wb")
        self._labels_file = self.labels_path.open("wb")

    def write(
        self,
        *,
        features: np.ndarray,
        segment: SegmentSpec,
    ) -> int:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected features shape (N, {FEATURE_DIM}), got {features.shape}"
            )
        if not len(features):
            return 0

        frame_start = self.num_frames
        features.tofile(self._x_file)
        np.full(len(features), int(segment.label), dtype=np.int32).tofile(self._labels_file)
        record = {
            "frame_start": int(frame_start),
            "num_frames": int(len(features)),
            "label": int(segment.label),
            "style_name": segment.style_name,
            "source_entry_id": int(segment.source_entry_id),
            "source_path": str(segment.source_path),
            "suffix": segment.suffix,
            "source_start": int(segment.source_start),
            "source_stop": int(segment.source_stop),
            "source_global_label": int(segment.source_global_label),
        }
        if segment.split is not None:
            record["split"] = segment.split
        self.segments.append(record)
        self.num_frames += int(len(features))
        return int(len(features))

    def split_summary(self) -> dict[str, dict[str, int]]:
        summary: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"frames": 0, "segments": 0})
        for segment in self.segments:
            split = str(segment.get("split", "all"))
            summary[split]["frames"] += int(segment["num_frames"])
            summary[split]["segments"] += 1
        return dict(sorted(summary.items()))

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        temp_path.replace(path)

    def _discard_outputs(self) -> None:
        for path in (self.meta_path, self.x_path, self.labels_path, self.styles_path, self.segments_path):
            if path.exists():
                path.unlink()

    def finalize(
        self,
        *,
        dataset_role: Literal["in_distribution", "zero_shot"],
        generation: dict[str, object],
        split: dict[str, object] | None = None,
        relations: dict[str, object] | None = None,
        allow_empty: bool = False,
    ) -> bool:
        self._x_file.close()
        self._labels_file.close()
        if not self.num_frames:
            self._discard_outputs()
            if not allow_empty:
                raise RuntimeError(
                    f"No samples were written to {self.out_dir}; removed the empty memmap files"
                )
            return False

        split_meta = dict(split or {})
        split_meta["summary"] = self.split_summary()
        meta: dict[str, object] = {
            "format": "motion375_memmap",
            "schema_version": MEMMAP_SCHEMA_VERSION,
            "dataset_role": dataset_role,
            "num_frames": int(self.num_frames),
            "feature_dim": FEATURE_DIM,
            "num_styles": len(self.style_names),
            "num_segments": len(self.segments),
            "segment_counts": {"total": len(self.segments)},
            "files": {
                "features": {
                    "path": MEMMAP_X_FILE,
                    "dtype": "float32",
                    "shape": [int(self.num_frames), FEATURE_DIM],
                },
                "labels": {
                    "path": MEMMAP_LABELS_FILE,
                    "dtype": "int32",
                    "shape": [int(self.num_frames)],
                },
                "style_names": {"path": MEMMAP_STYLES_FILE, "count": len(self.style_names)},
                "segments": {"path": MEMMAP_SEGMENTS_FILE, "count": len(self.segments)},
            },
            "feature_layout": {
                "trajectory": {"offset": 0, "dim": TRAJECTORY_DIM},
                "motion": {"offset": TRAJECTORY_DIM, "dim": MOTION_DIM},
            },
            "split": split_meta,
            "generation": generation,
            "relations": dict(relations or {}),
        }
        self._write_json(self.meta_path, meta)
        self._write_json(self.styles_path, self.style_names)
        self._write_json(self.segments_path, self.segments)
        return True


def _normalize_last_axis(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def _resolve_savgol_window(num_frames: int) -> int:
    """Largest odd Savitzky-Golay window that fits the clip and exceeds the polyorder.

    The extraction filter used to be hardcoded to 61 frames, which silently broke
    for any `--window` small enough that a clip could be shorter than 61 frames.
    """
    limit = min(SAVGOL_WINDOW, int(num_frames))
    if limit % 2 == 0:
        limit -= 1
    if limit <= SAVGOL_POLYORDER:
        raise ValueError(
            f"clip of {num_frames} frames is too short for a Savitzky-Golay window "
            f"(need > {SAVGOL_POLYORDER} frames)"
        )
    return limit


def trajectory_samples(window: int, context_stride: int) -> int:
    """Number of context samples per trajectory block, validating the fixed 84-dim layout.

    Every downstream reader (dataset, motion375 decoder, all trained checkpoints) assumes
    `x[:, :84]` is exactly 7 blocks of context samples, so any window/stride combination
    that changes that width is rejected here rather than at extraction time.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if context_stride < 1:
        raise ValueError(f"context_stride must be >= 1, got {context_stride}")
    if window % context_stride != 0:
        raise ValueError(
            f"window ({window}) must be a multiple of context_stride ({context_stride}) so that "
            f"the trajectory block stays at {TRAJECTORY_DIM} dims"
        )
    num_samples = len(range(-window, window, context_stride))
    if num_samples * TRAJECTORY_BLOCKS != TRAJECTORY_DIM:
        required_samples = TRAJECTORY_DIM // TRAJECTORY_BLOCKS
        # `range(-w, w, s)` yields 2 * w / s samples, so the required window is half the span.
        suggested_window = required_samples * context_stride // 2
        raise ValueError(
            f"window={window} with context_stride={context_stride} yields "
            f"{num_samples} context samples per block, i.e. {num_samples * TRAJECTORY_BLOCKS} "
            f"trajectory dims, but the memmap format requires exactly {TRAJECTORY_DIM} "
            f"({required_samples} samples per block). Use window={suggested_window} "
            f"with context_stride={context_stride}."
        )
    return num_samples


def prepare_motion(
    bvh_data: dict[str, object],
    *,
    start: int,
    stop: int,
    coordinate_convert: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(bvh_data["positions"][start:stop], dtype=np.float32).copy()
    rotations = quat.from_euler(np.radians(bvh_data["rotations"][start:stop]), order=bvh_data["order"])
    if coordinate_convert:
        positions[..., 0] *= -1
        xform = quat.to_xform(rotations)
        xform[..., 0, 1:] *= -1
        xform[..., 1:, 0] *= -1
        rotations = quat.from_xform(xform)
    positions *= 0.01
    return bvh_data["parents"], quat.normalize(quat.unroll(rotations)), positions


def extract_features_375(
    rotations: np.ndarray,
    positions: np.ndarray,
    parents: np.ndarray,
    selected: list[int],
    window: int,
    context_stride: int = CONTEXT_STRIDE,
) -> np.ndarray:
    if len(positions) < 2 * window + 2:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    num_traj_samples = trajectory_samples(window, context_stride)
    del num_traj_samples  # validated only; blocks are concatenated below in a fixed order
    savgol_window = _resolve_savgol_window(len(positions))

    velocities = np.empty_like(positions)
    velocities[1:-1] = 0.5 * (positions[2:] - positions[1:-1]) * 60.0 + 0.5 * (positions[1:-1] - positions[:-2]) * 60.0
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0
        + 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0
    )
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    global_rot, global_pos, global_vel, global_ang_vel = quat.fk_vel(
        rotations, positions, velocities, angular_velocities, parents
    )
    direction = np.array([1, 0, 1], dtype=np.float32) * quat.mul_vec(
        global_rot[:, 0:1], np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    direction = _normalize_last_axis(
        signal.savgol_filter(
            _normalize_last_axis(direction),
            savgol_window,
            SAVGOL_POLYORDER,
            axis=0,
            mode="interp",
        )
    )
    root_rot = quat.normalize(quat.between(np.array([0, 0, 1], dtype=np.float32), direction))
    root_vel = np.array([1, 0, 1], dtype=np.float32) * quat.inv_mul_vec(root_rot, global_vel[:, 0:1])
    root_ang_vel = np.degrees(quat.inv_mul_vec(root_rot, global_ang_vel[:, 0:1]))

    local_pos = global_pos.copy()
    local_pos[:, :, 0] -= local_pos[:, 0:1, 0]
    local_pos[:, :, 2] -= local_pos[:, 0:1, 2]
    local_pos = quat.inv_mul_vec(root_rot, local_pos)
    local_rot = quat.inv_mul(root_rot, global_rot)
    local_xy = quat.to_xform_xy(local_rot)
    local_vel = quat.inv_mul_vec(root_rot, global_vel)

    frame_ids = np.arange(window, len(positions) - window - 1, dtype=np.int64)
    context_offsets = np.arange(-window, window, context_stride, dtype=np.int64)
    context_ids = frame_ids[:, None] + context_offsets[None, :]
    root_rot_batch = root_rot[frame_ids, None]
    root_origin = global_pos[frame_ids, None, 0:1]
    planar_mask = np.array([1, 0, 1], dtype=np.float32)

    root_pos_ctx = planar_mask * quat.inv_mul_vec(
        root_rot_batch,
        global_pos[context_ids, 0:1] - root_origin,
    )
    root_dir_ctx = quat.inv_mul_vec(root_rot_batch, direction[context_ids])
    root_vel_ctx = quat.inv_mul_vec(root_rot_batch, global_vel[context_ids, 0:1])
    speed = np.sqrt(np.sum(np.square(root_vel_ctx), axis=-1))
    previous_ids = frame_ids - 1
    num_output_frames = frame_ids.size
    selected_local_xy = np.take(local_xy[previous_ids], selected, axis=1)

    x = np.concatenate(
        [
            root_pos_ctx[..., 0].reshape(num_output_frames, -1),
            root_pos_ctx[..., 2].reshape(num_output_frames, -1),
            root_dir_ctx[..., 0].reshape(num_output_frames, -1),
            root_dir_ctx[..., 2].reshape(num_output_frames, -1),
            root_vel_ctx[..., 0].reshape(num_output_frames, -1),
            root_vel_ctx[..., 2].reshape(num_output_frames, -1),
            speed.reshape(num_output_frames, -1),
            root_vel[previous_ids, :, 0].reshape(num_output_frames, -1),
            root_ang_vel[previous_ids, :, 1].reshape(num_output_frames, -1),
            root_vel[previous_ids, :, 2].reshape(num_output_frames, -1),
            local_pos[previous_ids][:, selected].reshape(num_output_frames, -1),
            selected_local_xy[..., 0].reshape(num_output_frames, -1),
            selected_local_xy[..., 1].reshape(num_output_frames, -1),
            local_vel[previous_ids][:, selected].reshape(num_output_frames, -1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if x.size and x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} feature dims, got {x.shape[1]}")
    return x.reshape(-1, FEATURE_DIM)


def _bounded_ordered_map(
    function: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    num_workers: int,
    max_pending: int,
) -> Iterator[_R]:
    if num_workers <= 1:
        for item in items:
            yield function(item)
        return

    pending_limit = max(int(max_pending), int(num_workers))
    item_iter = iter(items)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pending = deque()
        for _ in range(pending_limit):
            try:
                pending.append(executor.submit(function, next(item_iter)))
            except StopIteration:
                break

        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(item_iter)))
            except StopIteration:
                pass


def _process_entry_job(job: EntryJob) -> EntryResult:
    if job.dataset_role == "skip":
        return EntryResult(job=job, status="skipped")

    bvh_data = bvh.load_zeroeggs(str(job.path))
    clip_len = max(min(job.source_stop, len(bvh_data["positions"])) - job.source_start, 0)
    min_clip_len = 2 * job.window + 2
    if clip_len < min_clip_len:
        return EntryResult(job=job, status="short", clip_len=clip_len)

    parents, rotations, positions = prepare_motion(
        bvh_data,
        start=job.source_start,
        stop=job.source_stop,
        coordinate_convert=job.coordinate_convert,
    )
    features = extract_features_375(
        rotations,
        positions,
        parents,
        SELECTED_JOINTS,
        job.window,
        job.context_stride,
    )
    return EntryResult(job=job, status="ok", variants=(features,))


def parse_csv_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.upper() in {"N/A", "NA", "NAN"}:
        return None
    return int(float(value))


def resolve_bvh_path(data_root: Path, style_name: str, suffix: str) -> Path | None:
    for path in (data_root / style_name / f"{style_name}_{suffix}.bvh", data_root / f"{style_name}_{suffix}.bvh"):
        if path.exists():
            return path
    return None


def progress_write(message: str, progress: object | None = None) -> None:
    if progress is not None and hasattr(progress, "write"):
        progress.write(message)
        return
    print(message)


def load_entries(
    frame_cuts_csv: Path,
    data_root: Path,
    suffixes: list[str] | None,
    window: int,
    style_limit: int = 0,
):
    with frame_cuts_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    if "STYLE_NAME" not in columns:
        raise ValueError(f"STYLE_NAME column not found in {frame_cuts_csv}")

    found_suffixes: list[str] = []
    for col in columns:
        if col.endswith("_START") and f"{col[:-6]}_STOP" in columns and col[:-6] not in found_suffixes:
            found_suffixes.append(col[:-6])
    if suffixes is None:
        suffixes = [suffix for suffix in PREFERRED_SUFFIXES if suffix in found_suffixes]
        suffixes += sorted(suffix for suffix in found_suffixes if suffix not in PREFERRED_SUFFIXES)

    all_style_names, seen_styles = [], set()
    for row in rows:
        style_name = (row.get("STYLE_NAME") or "").strip()
        if style_name and style_name not in seen_styles:
            seen_styles.add(style_name)
            all_style_names.append(style_name)
    if style_limit < 0:
        raise ValueError(f"style_limit must be >= 0, got {style_limit}")
    if style_limit > len(all_style_names):
        raise ValueError(
            f"style_limit ({style_limit}) exceeds number of styles discovered in CSV ({len(all_style_names)})"
        )
    style_names = all_style_names[:style_limit] if style_limit > 0 else list(all_style_names)
    selected_style_set = set(style_names)
    style_to_global_label = {style_name: idx for idx, style_name in enumerate(all_style_names)}

    entries, missing = [], []
    for row in rows:
        style_name = (row.get("STYLE_NAME") or "").strip()
        if not style_name:
            continue
        if style_name not in selected_style_set:
            continue
        for suffix in suffixes:
            start = parse_csv_int(row.get(f"{suffix}_START"))
            stop = parse_csv_int(row.get(f"{suffix}_STOP"))
            if start is None or stop is None:
                continue
            bvh_path = resolve_bvh_path(data_root, style_name, suffix)
            if bvh_path is None:
                missing.append(f"{style_name}_{suffix}.bvh")
                continue
            entries.append(
                {
                    "entry_id": len(entries),
                    "path": bvh_path,
                    "style": style_name,
                    "global_label": style_to_global_label[style_name],
                    "suffix": suffix,
                    "start": start,
                    "stop": stop,
                    "estimated_rows": max(stop - start - (2 * window + 1), 0),
                }
            )
    return entries, style_names, all_style_names, missing, suffixes


def validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}")


def resolve_split_counts(num_entries: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if num_entries < 1:
        return 0, 0
    n_val = max(1, int(round(num_entries * val_ratio))) if val_ratio > 0.0 else 0
    n_test = max(1, int(round(num_entries * test_ratio))) if test_ratio > 0.0 else 0
    while n_val + n_test > max(num_entries - 1, 0):
        if n_test > 0 and (n_val == 0 or test_ratio <= val_ratio):
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def assign_style_splits(
    entries: list[dict[str, object]],
    *,
    val_ratio: float,
    test_ratio: float,
    rotation_index: int,
    suffix_order: list[str] | tuple[str, ...] = PREFERRED_SUFFIXES,
) -> dict[int, str]:
    """Suffix-balanced per-style split: hold out whole locomotion types, rotating by style.

    Every clip of a style is a different locomotion type (suffix), and clip length correlates
    with that type (FW/BW/TR1 ~7.5k frames, BR/FR ~4k, SR 2.4k, ID 0.7k). The previous
    frame-count greedy assignment therefore pushed the *short* running clips into val/test
    systematically: val/test ended up ~90% BR/FR/SR and every style's training set lost its
    FR/SR clips. Rotating the held-out suffixes by the style's position keeps val and test
    covering every suffix about equally across styles while each style still trains on all
    but `n_val + n_test` of its locomotion types.
    """
    if not entries:
        return {}
    n_val, n_test = resolve_split_counts(len(entries), val_ratio, test_ratio)
    assignment: dict[int, str] = {}
    if n_val + n_test == 0:
        for entry in entries:
            assignment[int(entry["entry_id"])] = "train"
        return assignment

    order = {str(name): index for index, name in enumerate(suffix_order)}
    ordered = sorted(
        entries,
        key=lambda entry: (order.get(str(entry.get("suffix", "")), len(order)), int(entry["entry_id"])),
    )
    count = len(ordered)
    offset = int(rotation_index) % count
    val_ids = {int(ordered[(offset + step) % count]["entry_id"]) for step in range(n_val)}
    test_ids = {
        int(ordered[(offset + n_val + step) % count]["entry_id"]) for step in range(n_test)
    }
    for entry in entries:
        entry_id = int(entry["entry_id"])
        assignment[entry_id] = "val" if entry_id in val_ids else "test" if entry_id in test_ids else "train"
    return assignment


def assign_entry_splits(
    entries: list[dict[str, object]],
    *,
    style_names: list[str],
    zero_shot_style_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[int, str], list[str], list[str]]:
    if not 0 <= zero_shot_style_count < len(style_names):
        raise ValueError(
            f"zero_shot_style_count must be >= 0 and smaller than number of styles ({len(style_names)}), "
            f"got {zero_shot_style_count}"
        )

    zero_shot_style_names = style_names[-zero_shot_style_count:] if zero_shot_style_count else []
    id_style_names = style_names[:-zero_shot_style_count] if zero_shot_style_count else list(style_names)
    zero_shot_style_set = set(zero_shot_style_names)
    entries_by_style: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        if str(entry["style"]) not in zero_shot_style_set and int(entry["estimated_rows"]) > 0:
            entries_by_style[str(entry["style"])].append(entry)

    assignments: dict[int, str] = {}
    for style_index, style_name in enumerate(id_style_names):
        assignments.update(
            assign_style_splits(
                entries_by_style.get(style_name, []),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                rotation_index=style_index,
                suffix_order=PREFERRED_SUFFIXES,
            )
        )
    return assignments, id_style_names, zero_shot_style_names


def data_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("MOTION375_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    for name in DATA_ROOT_DIR_NAMES:
        candidates.append(SCRIPT_DIR / "data" / name)
    for base in DATA_ROOT_SEARCH_BASES:
        for name in DATA_ROOT_DIR_NAMES:
            candidates.append(Path(base).expanduser() / name)
    return candidates


def resolve_default_data_root() -> Path | None:
    for candidate in data_root_candidates():
        if (candidate / "Frame_Cuts.csv").exists():
            return candidate
    return None


def summarize_split_coverage(
    assignment: dict[int, str],
    entries: list[dict[str, object]],
    style_names: list[str],
) -> tuple[dict[str, dict[str, int]], list[str], list[str]]:
    """Per-style clip counts per split, plus styles that ended up missing a split.

    The dataset protocol is: the last K styles become the OOD/zero-shot database, and
    every remaining (IND) style must appear in train, val and test. A style with only
    one or two usable clips cannot satisfy that, so those are reported separately
    instead of being treated as a bug.
    """
    counts: dict[str, dict[str, int]] = {
        name: {"train": 0, "val": 0, "test": 0, "frames": 0} for name in style_names
    }
    for entry in entries:
        style = str(entry["style"])
        if style not in counts:
            continue
        counts[style]["frames"] += max(int(entry["estimated_rows"]), 0)
        split = assignment.get(int(entry["entry_id"]))
        if split in ("train", "val", "test"):
            counts[style][split] += 1

    under_covered: list[str] = []
    impossible: list[str] = []
    for name in style_names:
        style_counts = counts[name]
        if all(style_counts[split] > 0 for split in ("train", "val", "test")):
            continue
        usable = sum(style_counts[split] for split in ("train", "val", "test"))
        (under_covered if usable >= 3 else impossible).append(name)
    return counts, under_covered, impossible


def coverage_summary(
    counts: dict[str, dict[str, int]],
    style_names: list[str],
) -> dict[str, object]:
    covered = [
        name for name in style_names if all(counts[name][split] > 0 for split in ("train", "val", "test"))
    ]
    return {
        "styles": len(style_names),
        "styles_in_all_splits": len(covered),
        "min_clips_per_split": {
            split: min((counts[name][split] for name in style_names), default=0)
            for split in ("train", "val", "test")
        },
        "max_clips_per_split": {
            split: max((counts[name][split] for name in style_names), default=0)
            for split in ("train", "val", "test")
        },
        "per_style_clips": {name: counts[name] for name in style_names},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate memmap-only 100STYLE 375-dim databases")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Directory holding Frame_Cuts.csv and the BVH files. "
            "Default: $MOTION375_DATA_ROOT, then <NewCode>/data/100sty, then known locations."
        ),
    )
    parser.add_argument("--frame-cuts", type=Path, default=None, help="Default: <data-root>/Frame_Cuts.csv")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--zero-shot-output", type=Path, default=None, help="Default: <output>_zeroshot")
    parser.add_argument("--zero-shot-style-count", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument(
        "--coordinate-convert",
        dest="coordinate_convert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable ZeroEGGS coordinate-system conversion before feature extraction",
    )
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument(
        "--context-stride",
        type=int,
        default=CONTEXT_STRIDE,
        help=f"Frame step between trajectory context samples; window must be a multiple of it (default {CONTEXT_STRIDE})",
    )
    parser.add_argument("--max-entries", type=int, default=0, help="Only process first N valid entries (0 means all)")
    parser.add_argument(
        "--style-limit",
        type=int,
        default=0,
        help="Only keep the first N styles in CSV order (0 means all styles)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel BVH/feature-extraction processes; 1 disables multiprocessing",
    )
    parser.add_argument(
        "--max-pending-jobs",
        type=int,
        default=0,
        help="Maximum submitted jobs/results held in memory; 0 uses 2 * num_workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs, splits and counts, print a summary, then exit without writing anything",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fingerprint_inputs(frame_cuts: Path) -> str:
    digest = hashlib.sha256()
    with frame_cuts.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.val_ratio, args.test_ratio)

    if args.data_root is None:
        resolved_root = resolve_default_data_root()
        if resolved_root is None:
            tried = ", ".join(str(candidate) for candidate in data_root_candidates())
            raise FileNotFoundError(
                "Could not locate a 100STYLE data root holding Frame_Cuts.csv. "
                f"Pass --data-root or set MOTION375_DATA_ROOT. Tried: {tried}"
            )
        args.data_root = resolved_root
    data_root = args.data_root.expanduser().resolve()
    frame_cuts = args.frame_cuts.expanduser().resolve() if args.frame_cuts else data_root / "Frame_Cuts.csv"
    output = args.output.expanduser().resolve()
    zero_shot_output = (
        args.zero_shot_output.expanduser().resolve()
        if args.zero_shot_output is not None
        else output.parent / f"{output.name}_zeroshot"
    )
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not frame_cuts.exists():
        raise FileNotFoundError(f"Frame_Cuts.csv not found: {frame_cuts}")
    if args.window < 1:
        raise ValueError(f"window must be >= 1, got {args.window}")
    if args.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {args.num_workers}")
    if args.max_pending_jobs < 0:
        raise ValueError(
            f"max_pending_jobs must be >= 0, got {args.max_pending_jobs}"
        )
    if args.zero_shot_style_count < 0:
        raise ValueError(f"zero_shot_style_count must be >= 0, got {args.zero_shot_style_count}")
    if output == zero_shot_output:
        raise ValueError("--zero-shot-output must be different from --output")

    # Fail before spending an hour extracting features: the trajectory block size and the
    # Savitzky-Golay filter both depend on `window`, and both used to be assumed valid.
    num_traj_samples = trajectory_samples(args.window, args.context_stride)
    min_clip_len = 2 * args.window + 2
    savgol_window = _resolve_savgol_window(min_clip_len)

    suffixes = [suffix.strip() for suffix in args.suffixes.split(",") if suffix.strip()] or None
    entries, style_names, all_style_names, missing_files, used_suffixes = load_entries(
        frame_cuts,
        data_root,
        suffixes,
        args.window,
        style_limit=args.style_limit,
    )
    if args.max_entries > 0:
        entries = entries[: args.max_entries]

    split_assignment, id_style_names, zero_shot_style_names = assign_entry_splits(
        entries,
        style_names=style_names,
        zero_shot_style_count=args.zero_shot_style_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    zero_shot_style_set = set(zero_shot_style_names)
    id_style_to_label = {style_name: idx for idx, style_name in enumerate(id_style_names)}
    zero_shot_style_to_label = {style_name: idx for idx, style_name in enumerate(zero_shot_style_names)}

    coverage_counts, under_covered_styles, impossible_coverage_styles = summarize_split_coverage(
        split_assignment, entries, id_style_names
    )
    if impossible_coverage_styles:
        print(
            f"[Warn] {len(impossible_coverage_styles)} IND style(s) have fewer than 3 usable clips and "
            f"therefore cannot appear in all three splits: {impossible_coverage_styles}"
        )
    if under_covered_styles:
        raise RuntimeError(
            f"{len(under_covered_styles)} IND style(s) have >= 3 usable clips but are missing from at "
            f"least one split: {under_covered_styles}. Every IND style must appear in train, val and "
            f"test; check assign_style_splits()."
        )
    coverage_meta = coverage_summary(coverage_counts, id_style_names)

    for label, value in (
        ("Data root", data_root),
        ("Frame cuts", frame_cuts),
        ("In-dist output", output),
        ("Zero-shot output", zero_shot_output),
        ("Styles total", len(all_style_names)),
        ("Styles selected", len(style_names)),
        ("In-dist styles", len(id_style_names)),
        ("Zero-shot styles", len(zero_shot_style_names)),
        (
            "Style coverage",
            f"{coverage_meta['styles_in_all_splits']}/{coverage_meta['styles']} IND styles present in "
            f"train+val+test (clips per split min-max: "
            f"train {coverage_meta['min_clips_per_split']['train']}-{coverage_meta['max_clips_per_split']['train']}, "
            f"val {coverage_meta['min_clips_per_split']['val']}-{coverage_meta['max_clips_per_split']['val']}, "
            f"test {coverage_meta['min_clips_per_split']['test']}-{coverage_meta['max_clips_per_split']['test']})",
        ),
        ("Selected head", style_names),
        ("Zero-shot tail", zero_shot_style_names),
        ("Suffixes", used_suffixes),
        ("Valid entries", len(entries)),
        ("Split ratio", f"val={args.val_ratio} test={args.test_ratio}"),
        ("Window / stride", f"{args.window} / {args.context_stride}"),
        ("Traj samples", f"{num_traj_samples} per block x {TRAJECTORY_BLOCKS} = {TRAJECTORY_DIM}"),
        ("Min clip len", min_clip_len),
        ("SavGol window", savgol_window),
        ("Coord convert", args.coordinate_convert),
        ("Workers", args.num_workers),
    ):
        print(f"[Info] {label:<17}: {value}")
    if missing_files:
        print(f"[Warn] Missing BVH files referenced by CSV: {len(missing_files)}")
        print("[Warn] First 10 missing:", missing_files[:10])

    if args.dry_run:
        print("[DryRun] No files were written. Summary of what would be produced:")
        for tag, names, variants_count in (
            ("in-dist", id_style_names, 1),
            ("zero-shot", zero_shot_style_names, 1),
        ):
            style_set = set(names)
            subset = [entry for entry in entries if str(entry["style"]) in style_set]
            rows = sum(max(int(entry["estimated_rows"]), 0) for entry in subset)
            print(
                f"[DryRun] {tag:<9}: styles={len(names)} entries={len(subset)} "
                f"estimated_frames={rows * variants_count} (x{variants_count} variants)"
            )
        return

    id_writer = MemmapWriter(output, id_style_names, args.overwrite)
    zero_writer = MemmapWriter(zero_shot_output, zero_shot_style_names, args.overwrite) if zero_shot_style_names else None
    max_pending_jobs = args.max_pending_jobs or (2 * args.num_workers)
    skipped_short = skipped_empty = 0

    def total_written_rows() -> int:
        return id_writer.num_frames + (zero_writer.num_frames if zero_writer else 0)

    jobs: list[EntryJob] = []
    for entry in entries:
        style_name = str(entry["style"])
        entry_id = int(entry["entry_id"])
        is_zero_shot = style_name in zero_shot_style_set
        split = None if is_zero_shot else split_assignment.get(entry_id)
        if is_zero_shot:
            dataset_role: Literal["in_distribution", "zero_shot", "skip"] = "zero_shot"
            label = zero_shot_style_to_label[style_name]
        elif split is None:
            dataset_role = "skip"
            label = None
        else:
            dataset_role = "in_distribution"
            label = id_style_to_label[style_name]
        jobs.append(
            EntryJob(
                path=Path(entry["path"]),
                style_name=style_name,
                suffix=str(entry["suffix"]),
                source_entry_id=entry_id,
                source_global_label=int(entry["global_label"]),
                source_start=int(entry["start"]),
                source_stop=int(entry["stop"]),
                label=label,
                dataset_role=dataset_role,
                split=split,
                coordinate_convert=bool(args.coordinate_convert),
                window=int(args.window),
                context_stride=int(args.context_stride),
            )
        )

    results = _bounded_ordered_map(
        _process_entry_job,
        jobs,
        num_workers=args.num_workers,
        max_pending=max_pending_jobs,
    )
    with tqdm(total=len(jobs), desc="Processing entries", unit="entry", dynamic_ncols=True) as progress:
        for idx, result in enumerate(results, 1):
            job = result.job
            path = job.path
            is_zero_shot = job.dataset_role == "zero_shot"
            progress.set_description(f"Processing {idx}/{len(jobs)}")
            progress.set_postfix(
                current=f"{job.style_name}:{job.suffix}",
                target="zero-shot" if is_zero_shot else (job.split or "skip"),
                rows=total_written_rows(),
                short=skipped_short,
                empty=skipped_empty,
                refresh=False,
            )

            try:
                if result.status == "skipped":
                    continue
                if result.status == "short":
                    skipped_short += 1
                    progress_write(
                        f"[Skip] {path.name}: clip len={result.clip_len}, requires >= {min_clip_len}",
                        progress,
                    )
                    continue
                if result.status != "ok":
                    raise ValueError(f"Unknown worker result status: {result.status!r}")
                writer = zero_writer if is_zero_shot else id_writer
                if writer is None:
                    continue
                if job.label is None:
                    raise RuntimeError("Non-skipped entry is missing a target label")
                for features in result.variants:
                    if not len(features):
                        skipped_empty += 1
                        progress_write(f"[Skip] {path.name}: no samples generated", progress)
                        continue
                    writer.write(
                        features=features,
                        segment=SegmentSpec(
                            label=job.label,
                            style_name=job.style_name,
                            split=job.split,
                            source_entry_id=job.source_entry_id,
                            source_path=job.path,
                            suffix=job.suffix,
                            source_start=job.source_start,
                            source_stop=job.source_stop,
                            source_global_label=job.source_global_label,
                        ),
                    )
            finally:
                progress.set_postfix(
                    current=f"{job.style_name}:{job.suffix}",
                    target="zero-shot" if is_zero_shot else (job.split or "skip"),
                    rows=total_written_rows(),
                    short=skipped_short,
                    empty=skipped_empty,
                    refresh=False,
                )
                progress.update(1)

    generation_meta = {
        "tool": "generate_database_100sty.py",
        "schema_version": MEMMAP_SCHEMA_VERSION,
        "frame_cuts_sha256": _fingerprint_inputs(frame_cuts),
        "inputs": {
            "frame_cuts": str(frame_cuts),
            "data_root": str(data_root),
        },
        "feature_extraction": {
            "window": int(args.window),
            "context_stride": int(args.context_stride),
            "trajectory_samples": int(num_traj_samples),
            "trajectory_blocks": int(TRAJECTORY_BLOCKS),
            "savgol_window": int(savgol_window),
            "savgol_polyorder": int(SAVGOL_POLYORDER),
            "min_clip_len": int(min_clip_len),
            "motion_source_offset": -1,
            "coordinate_convert": bool(args.coordinate_convert),
            "selected_joint_indices": SELECTED_JOINTS,
        },
        "selection": {
            "style_limit": int(args.style_limit),
            "max_entries": int(args.max_entries),
            "suffixes": used_suffixes,
            "selected_style_names": style_names,
        },
        "parallelism": {
            "num_workers": int(args.num_workers),
            "max_pending_jobs": int(max_pending_jobs),
            "ordered_output": True,
        },
        "augmentation": {
        },
    }
    id_summary = id_writer.split_summary()
    zero_summary = zero_writer.split_summary() if zero_writer else {}

    declared_splits = ["train", "val", "test"]
    empty_splits = [name for name in declared_splits if id_summary.get(name, {}).get("segments", 0) == 0]
    if empty_splits:
        raise RuntimeError(
            f"In-distribution split(s) {empty_splits} ended up empty; adjust --val-ratio/--test-ratio "
            f"or --split-seed. Actual summary: {id_summary}"
        )

    id_writer.finalize(
        dataset_role="in_distribution",
        generation=generation_meta,
        split={
            "strategy": "clip_level_stratified_by_style_and_frame_count",
            "seed": int(args.split_seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "style_partition": "csv_order: first (N - K) styles IND, last K styles OOD",
            "held_out_style_count": len(zero_shot_style_names),
            "held_out_style_names": zero_shot_style_names,
            "coverage": coverage_meta,
        },
        relations={
            "zero_shot_database": os.path.relpath(zero_shot_output, start=output)
        }
        if zero_writer
        else {},
    )
    zero_written = (
        zero_writer.finalize(
            dataset_role="zero_shot",
            generation=generation_meta,
            split={"strategy": "held_out_styles"},
            relations={
                "in_distribution_database": os.path.relpath(output, start=zero_shot_output)
            },
            allow_empty=True,
        )
        if zero_writer
        else False
    )

    print(f"[Info] Skipped short clips : {skipped_short}")
    print(f"[Info] Empty feature clips : {skipped_empty}")
    print(f"[Done] In-dist memmap      : {output}")
    print(f"[Done] Zero-shot memmap    : {zero_shot_output if zero_written else 'skipped (no held-out frames written)'}")
    print(f"[Done] In-dist summary     : {id_summary}")
    print(f"[Done] Zero-shot summary   : {zero_summary}")


if __name__ == "__main__":
    main()
