#!/usr/bin/env python
"""Merge two schema-v4 memmap databases into one, remapping labels to a global style order.

Written for the FID feature extractor: the classifier has to be trained on the full
100STYLE style set, but the revision pipeline keeps the 90 in-distribution styles
(`database_100sty_375_memmap_nomirror`) and the 10 held-out styles
(`database_100sty_375_memmap_zeroshot`) in separate databases.

The merge is a byte-exact concatenation of the two feature/label streams, so the frames
stay identical to the sources (no re-extraction from BVH). Labels are rewritten through
`source_global_label`, which both databases already record, so style `k` of the merged
database is the same style as global label `k` of the original 100STYLE ordering.

Split policy: the primary database's per-segment `split` values are preserved; every
secondary segment is assigned `--secondary-split` (default `train`), so a classifier
trained on the merged database sees all styles while the primary val/test clips remain
held out for monitoring.
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]

MEMMAP_FORMAT = "motion375_memmap"
MEMMAP_SCHEMA_VERSION = 4
FEATURE_DIM = 375
COPY_CHUNK_FRAMES = 1 << 20  # 1M frames x 375 x 4 B = 1.5 GB per chunk
VALID_SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceDB:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        if int(self.meta.get("schema_version", 0)) != MEMMAP_SCHEMA_VERSION:
            raise ValueError(f"{self.root}: schema_version must be {MEMMAP_SCHEMA_VERSION}")
        if self.meta.get("format") != MEMMAP_FORMAT:
            raise ValueError(f"{self.root}: format must be {MEMMAP_FORMAT!r}")
        if int(self.meta.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError(f"{self.root}: feature_dim must be {FEATURE_DIM}")
        self.num_frames = int(self.meta["num_frames"])
        self.style_names = [str(name) for name in json.loads((self.root / "StyleNames.json").read_text(encoding="utf-8"))]
        self.segments = json.loads((self.root / "segments.json").read_text(encoding="utf-8"))
        self.labels = np.memmap(self.root / "Labels.i32", dtype=np.int32, mode="r")
        self.features = np.memmap(self.root / "X.f32", dtype=np.float32, mode="r").reshape(self.num_frames, FEATURE_DIM)
        if len(self.labels) != self.num_frames:
            raise ValueError(f"{self.root}: Labels.i32 length does not match meta.num_frames")
        covered = sum(int(segment["num_frames"]) for segment in self.segments)
        if covered != self.num_frames:
            raise ValueError(
                f"{self.root}: segments cover {covered} frames but meta.num_frames is {self.num_frames}"
            )

    def global_label_of(self, local_label: int) -> int:
        self._validate_global_labels()
        return self._label_map[local_label]

    def _validate_global_labels(self) -> None:
        if getattr(self, "_label_map", None) is not None:
            return
        from collections import defaultdict

        by_local: dict[int, set[int]] = defaultdict(set)
        for segment in self.segments:
            by_local[int(segment["label"])].add(int(segment["source_global_label"]))
        label_map: dict[int, int] = {}
        for local_label, candidates in by_local.items():
            if len(candidates) != 1:
                raise ValueError(
                    f"{self.root}: local label {local_label} maps to multiple source_global_labels {sorted(candidates)}"
                )
            label_map[local_label] = candidates.pop()
        if sorted(label_map) != list(range(len(label_map))):
            raise ValueError(f"{self.root}: local labels are not contiguous 0..N-1")
        self._label_map = label_map

    def frames_of_label(self, local_label: int) -> int:
        return int((self.labels == int(local_label)).sum())


def _copy_stream(
    *,
    source: SourceDB,
    x_out,
    labels_out,
    label_map: np.ndarray,
    frame_offset: int,
    segments_out: list[dict[str, object]],
    segment_frame_shift: int,
    secondary_split: str | None,
    default_split_from_source: bool,
) -> None:
    """Append one source database to the open output streams, remapping labels."""
    x_handle = (source.root / "X.f32").open("rb")
    label_handle = (source.root / "Labels.i32").open("rb")
    try:
        remaining = source.num_frames
        while remaining > 0:
            count = min(COPY_CHUNK_FRAMES, remaining)
            features = np.fromfile(x_handle, dtype=np.float32, count=count * FEATURE_DIM)
            labels = np.fromfile(label_handle, dtype=np.int32, count=count)
            if features.size != count * FEATURE_DIM or labels.size != count:
                raise RuntimeError(f"{source.root}: short read while copying frames")
            features.tofile(x_out)
            label_map[labels].astype(np.int32, copy=False).tofile(labels_out)
            remaining -= count
    finally:
        x_handle.close()
        label_handle.close()

    for segment in source.segments:
        record = dict(segment)
        record["frame_start"] = int(segment["frame_start"]) + int(segment_frame_shift)
        record["label"] = int(label_map[int(segment["label"])])
        if default_split_from_source:
            split = str(segment.get("split", "")).strip()
            if split not in VALID_SPLITS:
                split = secondary_split
        else:
            split = secondary_split
        if split is not None:
            record["split"] = split
        segments_out.append(record)


def _split_summary(segments: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for segment in segments:
        key = str(segment.get("split", "all"))
        entry = summary.setdefault(key, {"frames": 0, "segments": 0})
        entry["frames"] += int(segment["num_frames"])
        entry["segments"] += 1
    return dict(sorted(summary.items()))


def _verify(
    *,
    out_dir: Path,
    sources: list[SourceDB],
    style_names: list[str],
    label_maps: list[np.ndarray],
    verify_windows: int,
    seed: int,
) -> None:
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    num_frames = int(meta["num_frames"])
    written_x = np.memmap(out_dir / "X.f32", dtype=np.float32, mode="r").reshape(num_frames, FEATURE_DIM)
    written_labels = np.memmap(out_dir / "Labels.i32", dtype=np.int32, mode="r")
    out_styles = json.loads((out_dir / "StyleNames.json").read_text(encoding="utf-8"))
    out_segments = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))

    if out_styles != style_names:
        raise AssertionError("StyleNames.json does not match the requested global style order")
    if len(out_segments) != sum(len(source.segments) for source in sources):
        raise AssertionError("segment count mismatch")
    if sum(int(segment["num_frames"]) for segment in out_segments) != num_frames:
        raise AssertionError("segments do not cover every frame exactly once")
    expected_labels = np.concatenate(
        [
            label_maps[idx][np.asarray(source.labels, dtype=np.int64)]
            for idx, source in enumerate(sources)
        ]
    )
    if not np.array_equal(np.asarray(written_labels), expected_labels.astype(np.int32)):
        raise AssertionError("written labels are not the remapped concatenation of the sources")

    rng = np.random.default_rng(seed)
    offset = 0
    for source, label_map in zip(sources, label_maps, strict=True):
        for _ in range(max(verify_windows, 0)):
            start = int(rng.integers(0, max(source.num_frames - 2, 1)))
            if not np.array_equal(np.asarray(written_x[offset + start]), np.asarray(source.features[start])):
                raise AssertionError(f"feature mismatch against {source.root} at frame {start}")
            if int(written_labels[offset + start]) != int(label_map[int(source.labels[start])]):
                raise AssertionError(f"label mismatch against {source.root} at frame {start}")
        offset += source.num_frames
    print(f"[Verify] {verify_windows} sampled windows per source match byte-for-byte")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two schema-v4 motion375 memmap databases")
    parser.add_argument("--primary", type=Path, required=True, help="database whose style order and splits are kept")
    parser.add_argument("--secondary", type=Path, required=True, help="database appended after the primary")
    parser.add_argument("--out", type=Path, required=True, help="output memmap database directory")
    parser.add_argument(
        "--secondary-split",
        type=str,
        default="train",
        choices=["train", "val", "test", "keep"],
        help="split assigned to secondary segments ('keep' preserves the secondary's own values)",
    )
    parser.add_argument("--verify-windows", type=int, default=64, help="random frames compared against the sources after writing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print the merge plan without writing")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = SourceDB(args.primary)
    secondary = SourceDB(args.secondary)

    overlap = set(primary.style_names) & set(secondary.style_names)
    if overlap:
        raise ValueError(f"the two databases share style names: {sorted(overlap)}")

    # Global style order comes from source_global_label, which both databases record.
    global_labels: dict[str, int] = {}
    for source in (primary, secondary):
        source._validate_global_labels()
        for local_label, global_label in source._label_map.items():
            name = source.style_names[local_label]
            previous = global_labels.setdefault(name, global_label)
            if previous != global_label:
                raise ValueError(f"style {name!r} has conflicting global labels {previous} vs {global_label}")
    style_order = sorted(global_labels.items(), key=lambda item: item[1])
    if [label for _, label in style_order] != list(range(len(style_order))):
        raise ValueError("global labels are not contiguous 0..N-1 across the two databases")
    style_names = [name for name, _ in style_order]
    name_to_label = {name: label for name, label in style_order}

    label_maps: list[np.ndarray] = []
    for source in (primary, secondary):
        mapping = np.full(len(source.style_names), -1, dtype=np.int64)
        for local_label, local_name in enumerate(source.style_names):
            mapping[local_label] = name_to_label[local_name]
        if (mapping < 0).any():
            raise ValueError(f"{source.root}: some styles are missing from the global order")
        label_maps.append(mapping)

    secondary_split = None if args.secondary_split == "keep" else str(args.secondary_split)

    print(f"[Plan] primary   : {primary.root} ({len(primary.style_names)} styles, {primary.num_frames} frames, {len(primary.segments)} segments)")
    print(f"[Plan] secondary : {secondary.root} ({len(secondary.style_names)} styles, {secondary.num_frames} frames, {len(secondary.segments)} segments)")
    print(f"[Plan] merged    : {len(style_names)} styles, {primary.num_frames + secondary.num_frames} frames")
    print(f"[Plan] split     : primary segments keep their split, secondary segments -> {args.secondary_split!r}")
    if args.dry_run:
        print("[Plan] dry-run: nothing written")
        return

    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=False)

    segments_out: list[dict[str, object]] = []
    with (args.out / "X.f32").open("wb") as x_out, (args.out / "Labels.i32").open("wb") as labels_out:
        _copy_stream(
            source=primary,
            x_out=x_out,
            labels_out=labels_out,
            label_map=label_maps[0],
            frame_offset=0,
            segments_out=segments_out,
            segment_frame_shift=0,
            secondary_split=None,
            default_split_from_source=True,
        )
        _copy_stream(
            source=secondary,
            x_out=x_out,
            labels_out=labels_out,
            label_map=label_maps[1],
            frame_offset=primary.num_frames,
            segments_out=segments_out,
            segment_frame_shift=primary.num_frames,
            secondary_split=secondary_split,
            default_split_from_source=secondary_split is None,
        )

    num_frames = primary.num_frames + secondary.num_frames
    split_summary = _split_summary(segments_out)
    (args.out / "StyleNames.json").write_text(
        json.dumps(style_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "segments.json").write_text(
        json.dumps(segments_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    meta: dict[str, object] = {
        "format": MEMMAP_FORMAT,
        "schema_version": MEMMAP_SCHEMA_VERSION,
        "dataset_role": "fid_classifier",
        "num_frames": int(num_frames),
        "feature_dim": FEATURE_DIM,
        "num_styles": len(style_names),
        "num_segments": len(segments_out),
        "segment_counts": {
            "total": len(segments_out),
        },
        "files": {
            "features": {"path": "X.f32", "dtype": "float32", "shape": [int(num_frames), FEATURE_DIM]},
            "labels": {"path": "Labels.i32", "dtype": "int32", "shape": [int(num_frames)]},
            "style_names": {"path": "StyleNames.json", "count": len(style_names)},
            "segments": {"path": "segments.json", "count": len(segments_out)},
        },
        "feature_layout": {
            "trajectory": {"offset": 0, "dim": 84},
            "motion": {"offset": 84, "dim": 291},
        },
        "split": {
            "strategy": "merged_from_sources",
            "primary": str(primary.root),
            "secondary": str(secondary.root),
            "secondary_split": args.secondary_split,
            "summary": split_summary,
        },
        "generation": {
            "tool": "merge_memmap_dbs.py",
            "schema_version": MEMMAP_SCHEMA_VERSION,
            "derived_from": [
                {
                    "role": "primary",
                    "path": str(source.root),
                    "num_frames": source.num_frames,
                    "num_segments": len(source.segments),
                    "num_styles": len(source.style_names),
                    "meta_sha256": _sha256(source.root / "meta.json"),
                    "generation": source.meta.get("generation"),
                }
                for source in (primary, secondary)
            ],
            "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "frames copied verbatim from the sources; only labels/splits/offsets were rewritten",
        },
        "relations": {},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _verify(
        out_dir=args.out,
        sources=[primary, secondary],
        style_names=style_names,
        label_maps=label_maps,
        verify_windows=int(args.verify_windows),
        seed=int(args.seed),
    )
    size_gb = (args.out / "X.f32").stat().st_size / (1 << 30)
    print(f"[Done] {args.out}: {num_frames} frames, {len(segments_out)} segments, {len(style_names)} styles, X.f32={size_gb:.2f} GiB")
    print(f"[Done] split summary: {json.dumps(split_summary, ensure_ascii=False)}")
    print("[Done] label histogram (global label: frames):")
    labels = np.memmap(args.out / "Labels.i32", dtype=np.int32, mode="r")
    counts = np.bincount(np.asarray(labels), minlength=len(style_names))
    for idx, name in enumerate(style_names):
        print(f"    {idx:3d} {name:18s} {int(counts[idx]):9d} frames")


if __name__ == "__main__":
    main()
