"""Shared fixtures that write valid schema-v4 memmap databases for tests.

Mirrors what `MemmapWriter.finalize()` emits, so the loader's validation is exercised the
same way a real database would exercise it. Keeping this in one place stops the two test
modules from drifting apart as the schema evolves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_DIM = 375
TRAJ_DIM = 84
MOTION_DIM = 291


def write_meta(
    root: Path,
    *,
    num_frames: int,
    num_styles: int,
    segments: list[dict[str, object]],
    dataset_role: str = "in_distribution",
) -> None:
    summary: dict[str, dict[str, int]] = {}
    for segment in segments:
        split = str(segment.get("split", "all"))
        entry = summary.setdefault(split, {"frames": 0, "segments": 0})
        entry["frames"] += int(segment["num_frames"])
        entry["segments"] += 1

    (root / "meta.json").write_text(
        json.dumps(
            {
                "format": "motion375_memmap",
                "schema_version": 4,
                "dataset_role": dataset_role,
                "num_frames": num_frames,
                "feature_dim": FEATURE_DIM,
                "num_styles": num_styles,
                "num_segments": len(segments),
                "segment_counts": {"total": len(segments)},
                "files": {
                    "features": {"path": "X.f32", "dtype": "float32", "shape": [num_frames, FEATURE_DIM]},
                    "labels": {"path": "Labels.i32", "dtype": "int32", "shape": [num_frames]},
                    "style_names": {"path": "StyleNames.json", "count": num_styles},
                    "segments": {"path": "segments.json", "count": len(segments)},
                },
                "feature_layout": {
                    "trajectory": {"offset": 0, "dim": TRAJ_DIM},
                    "motion": {"offset": TRAJ_DIM, "dim": MOTION_DIM},
                },
                "split": {"strategy": "test_fixture", "summary": summary},
                "generation": {"tool": "tests/_memmap_fixtures.py"},
                "relations": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "segments.json").write_text(json.dumps(segments), encoding="utf-8")


def write_db(
    root: Path,
    *,
    num_frames: int,
    num_styles: int,
    segments: list[dict[str, object]],
    dataset_role: str = "in_distribution",
    seed: int = 7,
) -> np.ndarray:
    """Write X.f32 / Labels.i32 / StyleNames.json / segments.json / meta.json (schema v4)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(num_frames, FEATURE_DIM)).astype(np.float32)
    labels = np.zeros((num_frames,), dtype=np.int32)
    for segment in segments:
        start = int(segment["frame_start"])
        length = int(segment["num_frames"])
        labels[start : start + length] = int(segment["label"])

    x.tofile(root / "X.f32")
    labels.tofile(root / "Labels.i32")
    (root / "StyleNames.json").write_text(
        json.dumps([f"style_{idx}" for idx in range(num_styles)]), encoding="utf-8"
    )
    write_meta(
        root,
        num_frames=num_frames,
        num_styles=num_styles,
        segments=segments,
        dataset_role=dataset_role,
    )
    return x
