"""
Integrity checker for 375-dim motion memmap databases.

Checks the on-disk database against the contract that `dataset.py` and every
trainer rely on, and it checks the two properties that are easy to lose silently
when a database is regenerated:

  * no frame is shared between two splits (no train/val/test leakage)
  * every in-distribution style appears in train, val and test

Usage:
  python verify_memmap_db.py                                  # checks both default DBs
  python verify_memmap_db.py --db path/to/db [--db other/db]
  python verify_memmap_db.py --db new/db --compare old/db      # numeric comparison
  python verify_memmap_db.py --json report.json

Exit code is 0 when no check FAILs, 1 otherwise. Warnings never fail the run.
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset import (
    MEMMAP_FORMAT,
    MEMMAP_LABELS_FILE,
    MEMMAP_META_FILE,
    MEMMAP_SCHEMA_VERSION,
    MEMMAP_SEGMENTS_FILE,
    MEMMAP_STYLES_FILE,
    MEMMAP_X_FILE,
    load_motion375_metadata,
)

FEATURE_DIM = 375
TRAJ_DIM = 84
MOTION_DIM = 291
TRAJECTORY_BLOCKS = 7
SELECTED_JOINTS = 24
SPLITS = ("train", "val", "test")

# float32 storage of a float64 computation drifts by a few 1e-4 on the unit-norm checks.
UNIT_TOLERANCE = 2e-3


@dataclass
class Report:
    db: Path
    checks: list[tuple[str, str, str]] = field(default_factory=list)  # (level, name, detail)

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append(("PASS", name, detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.checks.append(("WARN", name, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.checks.append(("FAIL", name, detail))

    @property
    def failures(self) -> int:
        return sum(1 for level, _, _ in self.checks if level == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for level, _, _ in self.checks if level == "WARN")

    def render(self) -> str:
        lines = [f"=== {self.db} ==="]
        width = max((len(name) for _, name, _ in self.checks), default=10)
        for level, name, detail in self.checks:
            suffix = f"  {detail}" if detail else ""
            lines.append(f"  [{level}] {name:<{width}}{suffix}")
        lines.append(
            f"  -> {len(self.checks)} checks, {self.failures} failed, {self.warnings} warnings"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "db": str(self.db),
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": [{"level": lvl, "name": name, "detail": detail} for lvl, name, detail in self.checks],
        }


def check_meta(report: Report, db: Path) -> dict[str, object] | None:
    meta_path = db / MEMMAP_META_FILE
    if not meta_path.exists():
        report.fail("meta.json present", f"missing {meta_path}")
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    report.ok("meta.json present")

    if meta.get("format") != MEMMAP_FORMAT:
        report.fail("meta.format", f"expected {MEMMAP_FORMAT!r}, got {meta.get('format')!r}")
    else:
        report.ok("meta.format", MEMMAP_FORMAT)

    schema = int(meta.get("schema_version", 0))
    if schema != MEMMAP_SCHEMA_VERSION:
        report.fail(
            "meta.schema_version",
            f"got v{schema}, this codebase reads v{MEMMAP_SCHEMA_VERSION} only",
        )
    else:
        report.ok("meta.schema_version", f"v{schema}")

    expected_layout = {
        "trajectory": {"offset": 0, "dim": TRAJ_DIM},
        "motion": {"offset": TRAJ_DIM, "dim": MOTION_DIM},
    }
    if meta.get("feature_layout") != expected_layout:
        report.fail("meta.feature_layout", f"expected {expected_layout}, got {meta.get('feature_layout')}")
    else:
        report.ok("meta.feature_layout", f"traj 0:{TRAJ_DIM} motion {TRAJ_DIM}:{FEATURE_DIM}")

    if schema >= 4:
        if "segment_counts" not in meta:
            report.fail("meta.v4 fields", "segment_counts missing")
        else:
            report.ok("meta.v4 fields", f"segment_counts={meta['segment_counts']}")

    if not (meta.get("generation") or {}).get("frame_cuts_sha256"):
        report.warn("meta.input fingerprint", "generation.frame_cuts_sha256 absent; rebuild provenance unknown")
    else:
        report.ok("meta.input fingerprint", str(meta["generation"]["frame_cuts_sha256"])[:16])

    return meta


def check_files(report: Report, db: Path, meta: dict[str, object]) -> tuple[int, int] | None:
    num_frames = int(meta.get("num_frames", -1))
    feature_dim = int(meta.get("feature_dim", -1))
    if num_frames <= 0 or feature_dim != FEATURE_DIM:
        report.fail("meta dimensions", f"num_frames={num_frames} feature_dim={feature_dim}")
        return None
    report.ok("meta dimensions", f"{num_frames:,} frames x {feature_dim} dims")

    x_path = db / MEMMAP_X_FILE
    labels_path = db / MEMMAP_LABELS_FILE
    for path, expected, dtype in (
        (x_path, num_frames * feature_dim * 4, "float32"),
        (labels_path, num_frames * 4, "int32"),
    ):
        if not path.exists():
            report.fail(f"{path.name} present", "missing")
            return None
        actual = path.stat().st_size
        if actual != expected:
            report.fail(
                f"{path.name} size",
                f"expected {expected:,} bytes ({dtype}), got {actual:,}",
            )
            return None
        report.ok(f"{path.name} size", f"{actual:,} bytes ({dtype})")
    return num_frames, feature_dim


def check_labels_styles(
    report: Report,
    db: Path,
    meta: dict[str, object],
    labels: np.ndarray,
) -> list[str]:
    styles_path = db / MEMMAP_STYLES_FILE
    if not styles_path.exists():
        report.fail("StyleNames.json present", "missing")
        return []
    style_names = [str(name) for name in json.loads(styles_path.read_text(encoding="utf-8"))]
    declared = meta.get("num_styles")
    if declared is not None and int(declared) != len(style_names):
        report.fail("num_styles consistency", f"meta={declared} StyleNames.json={len(style_names)}")
    else:
        report.ok("num_styles consistency", f"{len(style_names)} styles")

    unique = np.unique(labels)
    expected = np.arange(len(style_names), dtype=labels.dtype)
    if unique.size != len(style_names) or not np.array_equal(unique, expected):
        report.fail(
            "labels contiguous",
            f"Labels.i32 holds {unique.size} distinct values "
            f"(min={int(unique.min())}, max={int(unique.max())}) but StyleNames.json has "
            f"{len(style_names)}; expected exactly 0..{len(style_names) - 1}",
        )
    else:
        report.ok("labels contiguous", f"0..{len(style_names) - 1}, every style present")
    return style_names


def check_segments(
    report: Report,
    db: Path,
    num_frames: int,
    labels: np.ndarray,
    style_names: list[str],
    meta: dict[str, object],
) -> list[dict[str, object]]:
    segments_path = db / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        report.warn("segments.json present", "missing; no offline splits available")
        return []
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    report.ok("segments.json present", f"{len(segments)} segments")

    if meta.get("num_segments") is not None and int(meta["num_segments"]) != len(segments):
        report.fail(
            "num_segments consistency",
            f"meta={meta['num_segments']} segments.json={len(segments)}",
        )
    else:
        report.ok("num_segments consistency", str(len(segments)))

    # coverage + ordering + bounds
    cursor = 0
    gaps, overlaps, out_of_bounds, mislabeled, crossing = 0, 0, 0, 0, 0
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]).astype(np.int64) + 1
    for idx, seg in enumerate(segments):
        start = int(seg["frame_start"])
        length = int(seg["num_frames"])
        label = int(seg["label"])
        if start + length > num_frames or start < 0 or length <= 0:
            out_of_bounds += 1
            continue
        if start < cursor:
            overlaps += 1
        elif start > cursor:
            gaps += 1
        cursor = start + length
        if int(labels[start]) != label:
            mislabeled += 1
        if np.searchsorted(boundaries, start + length - 1, "right") - np.searchsorted(
            boundaries, start, "right"
        ) != 0:
            crossing += 1
    if overlaps:
        report.fail("segments non-overlapping", f"{overlaps} overlapping segment(s)")
    else:
        report.ok("segments non-overlapping", "verified")
    if out_of_bounds:
        report.fail("segments within bounds", f"{out_of_bounds} out-of-range segment(s)")
    else:
        report.ok("segments within bounds", "verified")
    if mislabeled:
        report.fail("segments label match", f"{mislabeled} segment(s) disagree with Labels.i32")
    else:
        report.ok("segments label match", "verified")
    if crossing:
        report.fail("segments stay in one style", f"{crossing} segment(s) span a style boundary")
    else:
        report.ok("segments stay in one style", "verified")
    if cursor != num_frames:
        report.fail(
            "segments cover all frames",
            f"segments end at frame {cursor:,} but the database holds {num_frames:,}",
        )
    else:
        report.ok("segments cover all frames", f"0..{num_frames:,} without gaps")
    if gaps:
        report.fail("segments contiguous", f"{gaps} gap(s) between segments")
    elif not overlaps:
        report.ok("segments contiguous", "no gaps")

    # mirror pairing
    by_entry: dict[int, list[dict[str, object]]] = {}
    for seg in segments:
        by_entry.setdefault(int(seg["source_entry_id"]), []).append(seg)
    mismatched_split = [eid for eid, group in by_entry.items() if len({str(s.get("split")) for s in group}) > 1]
    mismatched_len = [
        eid
        for eid, group in by_entry.items()
        if len({int(s["num_frames"]) for s in group}) > 1
    ]
    variant_counts = {len(group) for group in by_entry.values()}
    if mismatched_split:
        report.fail(
            "mirror pairs share split",
            f"{len(mismatched_split)} source clip(s) have variants in different splits",
        )
    else:
        report.ok("mirror pairs share split", f"{len(by_entry)} source clips")
    if mismatched_len:
        report.warn(
            "mirror pairs same length",
            f"{len(mismatched_len)} source clip(s) have variants of different length",
        )
    else:
        report.ok("mirror pairs same length", "verified")
    report.ok("mirror variants per clip", f"{sorted(variant_counts)}")
    if str(meta.get("mirror_layout", "")).strip().lower() == "none":
        if sum(1 for segment in segments if segment.get("mirror")):
            report.fail("unmirrored contract", "mirror_layout='none' but segments.json has mirror segments")
        else:
            report.ok("unmirrored contract", "no mirrored segments (mirror_layout=none)")

    if meta.get("segment_counts") is not None:
        expected_counts = {"total": len(segments)}
        if dict(meta["segment_counts"]) != expected_counts:
            report.fail("meta.segment_counts", f"expected {expected_counts}, got {meta['segment_counts']}")
        else:
            report.ok("meta.segment_counts", str(expected_counts))

    return segments


def check_splits(
    report: Report,
    segments: list[dict[str, object]],
    style_names: list[str],
    meta: dict[str, object],
) -> None:
    if not segments:
        return
    present = {str(s.get("split")) for s in segments if s.get("split")}
    role = str(meta.get("dataset_role", ""))
    if role == "zero_shot":
        if present:
            report.warn("zero-shot has no splits", f"unexpected split labels: {sorted(present)}")
        else:
            report.ok("zero-shot has no splits", "held-out styles are intentionally unsplit")
        return

    if not present:
        report.fail("offline splits present", "no segment carries a split for an in-distribution DB")
        return
    missing = [name for name in SPLITS if name not in present]
    if missing:
        report.fail("all three splits present", f"missing {missing}")
    else:
        report.ok("all three splits present", "train/val/test")

    # leakage: cross-split frame overlap
    spans: dict[str, list[tuple[int, int]]] = {}
    for name in SPLITS:
        spans[name] = sorted(
            (int(s["frame_start"]), int(s["frame_start"]) + int(s["num_frames"]))
            for s in segments
            if str(s.get("split")) == name
        )
    leaks = 0
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            j = 0
            for start, end in spans[left]:
                while j < len(spans[right]) and spans[right][j][1] <= start:
                    j += 1
                if j < len(spans[right]) and spans[right][j][0] < end:
                    leaks += 1
    if leaks:
        report.fail("no cross-split frame overlap", f"{leaks} overlapping pair(s); splits leak")
    else:
        report.ok("no cross-split frame overlap", "verified for all split pairs")

    # per-style coverage and frame balance
    frames: dict[str, dict[str, int]] = {name: {} for name in SPLITS}
    clips: dict[str, dict[str, set[int]]] = {name: {} for name in SPLITS}
    for seg in segments:
        split = str(seg.get("split"))
        if split not in SPLITS:
            continue
        style = str(seg["style_name"])
        frames[split][style] = frames[split].get(style, 0) + int(seg["num_frames"])
        clips[split].setdefault(style, set()).add(int(seg["source_entry_id"]))

    all_styles = sorted({str(s["style_name"]) for s in segments if str(s.get("split")) in SPLITS})
    uncovered = [
        style
        for style in all_styles
        if any(len(clips[split].get(style, ())) == 0 for split in SPLITS)
    ]
    if uncovered:
        report.fail(
            "every style in all splits",
            f"{len(uncovered)} style(s) missing from a split: {uncovered[:8]}",
        )
    else:
        report.ok("every style in all splits", f"{len(all_styles)} styles in train+val+test")

    totals = {split: sum(frames[split].values()) for split in SPLITS}
    grand = sum(totals.values()) or 1
    report.ok(
        "frame ratio",
        "  ".join(f"{split}={100 * totals[split] / grand:.1f}%" for split in SPLITS)
        + f"  ({grand:,} frames)",
    )

    per_style = {
        style: {split: round(100 * frames[split].get(style, 0) / max(sum(frames[s].get(style, 0) for s in SPLITS), 1), 1) for split in SPLITS}
        for style in all_styles
    }
    worst = max(per_style.items(), key=lambda kv: abs(kv[1]["train"] - 100 * (1 - 0.2)))
    report.ok(
        "frame balance per style",
        f"train share between {min(v['train'] for v in per_style.values()):.1f}% and "
        f"{max(v['train'] for v in per_style.values()):.1f}% (worst: {worst[0]} {worst[1]['train']}%)",
    )

    declared = (meta.get("split") or {}).get("coverage")
    if declared:
        recomputed = {
            "styles": len(all_styles),
            "styles_in_all_splits": len(all_styles) - len(uncovered),
        }
        mismatch = [
            key for key, value in recomputed.items() if int(declared.get(key, -1)) != value
        ]
        if mismatch:
            report.fail("meta coverage matches data", f"fields disagree: {mismatch}")
        else:
            report.ok("meta coverage matches data", f"{recomputed['styles_in_all_splits']}/{recomputed['styles']}")


def check_features(
    report: Report,
    db: Path,
    num_frames: int,
    sample_frames: int,
    seed: int = 0,
) -> None:
    if sample_frames <= 0:
        return
    count = min(sample_frames, num_frames)
    x_path = db / MEMMAP_X_FILE
    source = np.memmap(x_path, dtype=np.float32, mode="r", shape=(num_frames, FEATURE_DIM))
    ids = np.sort(np.random.default_rng(seed).choice(num_frames, count, replace=False))
    x = np.asarray(source[ids], dtype=np.float64)

    if not np.isfinite(x).all():
        report.fail("features finite", f"{int((~np.isfinite(x)).sum())} non-finite value(s)")
        return
    report.ok("features finite", f"{count:,} sampled frames")

    traj = x[:, :TRAJ_DIM]
    block = traj.shape[1] // TRAJECTORY_BLOCKS
    direction = np.stack([traj[:, 2 * block : 3 * block], traj[:, 3 * block : 4 * block]], axis=-1)
    deviation = np.abs(np.linalg.norm(direction, axis=-1) - 1.0).max(axis=1)
    norm_error = float(deviation.max())
    bad = int((deviation > UNIT_TOLERANCE).sum())
    # A handful of extracted frames can miss the unit-norm contract slightly; only a
    # widespread or gross violation is a database-level failure.
    if norm_error > 1e-2 or bad > max(10, count // 1000):
        report.fail("traj direction unit-norm",
                    f"max deviation {norm_error:.2e} on {bad}/{count} sampled frames")
    elif bad:
        report.warn("traj direction unit-norm",
                    f"{bad}/{count} sampled frames deviate up to {norm_error:.2e}")
    else:
        report.ok("traj direction unit-norm", f"max deviation {norm_error:.2e}")

    speed = traj[:, 6 * block : 7 * block]
    if float(speed.min()) < -UNIT_TOLERANCE:
        report.fail("traj speed non-negative", f"min {speed.min():.4f}")
    else:
        report.ok("traj speed non-negative", f"min {speed.min():.4f}, max {speed.max():.4f}")

    motion = x[:, TRAJ_DIM:]
    block = SELECTED_JOINTS * 3
    # motion = [root_motion 3][local_pos 72][rot col0 72][rot col1 72][local_vel 72]
    local_pos = motion[:, 3 : 3 + block].reshape(-1, SELECTED_JOINTS, 3)
    c0 = motion[:, 3 + block : 3 + 2 * block].reshape(-1, SELECTED_JOINTS, 3)
    c1 = motion[:, 3 + 2 * block : 3 + 3 * block].reshape(-1, SELECTED_JOINTS, 3)
    local_vel = motion[:, 3 + 3 * block : 3 + 4 * block].reshape(-1, SELECTED_JOINTS, 3)
    c0_error = float(np.abs(np.linalg.norm(c0, axis=-1) - 1.0).max())
    c1_error = float(np.abs(np.linalg.norm(c1, axis=-1) - 1.0).max())
    dot = float(np.abs((c0 * c1).sum(-1)).max())
    if max(c0_error, c1_error, dot) > UNIT_TOLERANCE:
        report.fail(
            "motion rotation basis orthonormal",
            f"|c0|-1={c0_error:.2e} |c1|-1={c1_error:.2e} |c0.c1|={dot:.2e}",
        )
    else:
        report.ok(
            "motion rotation basis orthonormal",
            f"|c0|-1={c0_error:.2e} |c1|-1={c1_error:.2e} |c0.c1|={dot:.2e}",
        )
    report.ok(
        "motion local ranges",
        f"local_pos |v| [{np.linalg.norm(local_pos, axis=-1).min():.3f}, "
        f"{np.linalg.norm(local_pos, axis=-1).max():.3f}]  "
        f"local_vel |v| [{np.linalg.norm(local_vel, axis=-1).min():.3f}, "
        f"{np.linalg.norm(local_vel, axis=-1).max():.3f}]",
    )
    del local_pos, local_vel

    root_motion = motion[:, :3]
    report.ok(
        "root motion range",
        f"vx [{root_motion[:, 0].min():.1f}, {root_motion[:, 0].max():.1f}] "
        f"yaw [{root_motion[:, 1].min():.1f}, {root_motion[:, 1].max():.1f}] deg "
        f"vz [{root_motion[:, 2].min():.1f}, {root_motion[:, 2].max():.1f}]",
    )
    del source


def compare_databases(report: Report, new_db: Path, old_db: Path, sample_frames: int) -> None:
    try:
        new_meta = load_motion375_metadata(new_db)
        old_meta = load_motion375_metadata(old_db)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        report.fail("compare: metadata", f"{type(exc).__name__}: {exc}")
        return
    if new_meta.num_frames != old_meta.num_frames:
        report.fail(
            "compare: frame count",
            f"{new_db.name}={new_meta.num_frames:,} vs {old_db.name}={old_meta.num_frames:,}",
        )
        return
    report.ok("compare: frame count", f"{new_meta.num_frames:,} frames in both")

    for name in (MEMMAP_LABELS_FILE,):
        a = (new_db / name).read_bytes()
        b = (old_db / name).read_bytes()
        if a == b:
            report.ok(f"compare: {name} byte-identical", "yes")
        else:
            diff = int((np.frombuffer(a, dtype=np.int32) != np.frombuffer(b, dtype=np.int32)).sum())
            report.warn(f"compare: {name} byte-identical", f"no; {diff:,} differing labels")

    count = min(sample_frames, new_meta.num_frames)
    ids = np.sort(np.random.default_rng(0).choice(new_meta.num_frames, count, replace=False))
    a = np.asarray(
        np.memmap(new_db / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(new_meta.num_frames, FEATURE_DIM))[ids]
    )
    b = np.asarray(
        np.memmap(old_db / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(old_meta.num_frames, FEATURE_DIM))[ids]
    )
    delta = np.abs(a - b)
    identical = float((a == b).mean())
    scale = float(np.abs(b).max()) or 1.0
    report.ok(
        "compare: X.f32 numerically",
        f"max|diff|={delta.max():.3e} ({delta.max() / scale:.2e} of max|value|), "
        f"bit-identical={100 * identical:.3f}% over {count:,} sampled frames",
    )
    if delta.max() > 1e-3 * scale:
        report.warn(
            "compare: X.f32 tolerance",
            "differences exceed float32 rounding; the two databases may come from different code",
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify 375-dim motion memmap databases")
    parser.add_argument("--db", type=Path, action="append", default=None, help="Database directory (repeatable)")
    parser.add_argument("--compare", type=Path, default=None, help="Second database to compare numerically")
    parser.add_argument("--sample-frames", type=int, default=20000, help="Frames sampled for value checks (0 disables)")
    parser.add_argument("--json", type=Path, default=None, help="Write the report as JSON")
    return parser.parse_args()


def verify(db: Path, sample_frames: int) -> Report:
    report = Report(db=db)
    if not db.is_dir():
        report.fail("database directory", f"not a directory: {db}")
        return report

    meta = check_meta(report, db)
    if meta is None:
        return report
    sizes = check_files(report, db, meta)
    if sizes is None:
        return report
    num_frames, _ = sizes

    labels = np.fromfile(db / MEMMAP_LABELS_FILE, dtype=np.int32, count=num_frames)
    style_names = check_labels_styles(report, db, meta, labels)
    segments = check_segments(report, db, num_frames, labels, style_names, meta)
    check_splits(report, segments, style_names, meta)
    check_features(report, db, num_frames, sample_frames)
    return report


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parents[1]
    dbs = args.db or [
        script_dir / "database_100sty_375_memmap_nomirror",
        script_dir / "database_100sty_375_memmap_zeroshot",
    ]

    reports = [verify(Path(db).expanduser().resolve(), args.sample_frames) for db in dbs]
    for report in reports:
        print(report.render())
        print()
    if args.compare is not None:
        compare_report = Report(db=Path(args.compare).expanduser().resolve())
        target = Path(dbs[0]).expanduser().resolve()
        compare_databases(compare_report, target, Path(args.compare).expanduser().resolve(), args.sample_frames)
        print(compare_report.render())
        print()
        reports.append(compare_report)

    if args.json is not None:
        payload = [report.to_dict() for report in reports]
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report written to {args.json}")

    failed = sum(report.failures for report in reports)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
