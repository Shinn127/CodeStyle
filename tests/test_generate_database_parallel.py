from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from tools.generate_database_100sty import (
    EntryJob,
    MemmapWriter,
    SELECTED_JOINTS,
    SegmentSpec,
    _bounded_ordered_map,
    _process_entry_job,
    bvh,
    extract_features_375,
)
from dataset import load_motion375_metadata


class SuffixBalancedSplitTests(unittest.TestCase):
    """The per-style split holds out whole locomotion types, rotated so val/test stay balanced."""

    def _assign(self, styles: int = 90, suffixes=("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1")):
        import collections
        from tools.generate_database_100sty import assign_style_splits
        assignments = {}
        for style_index in range(styles):
            entries = [
                {"entry_id": style_index * 100 + k, "suffix": suffix, "estimated_rows": 1000 + (7 - k) * 800}
                for k, suffix in enumerate(suffixes)
            ]
            assignments.update(
                assign_style_splits(entries, val_ratio=0.1, test_ratio=0.1, rotation_index=style_index)
            )
        return assignments

    def test_val_and_test_cover_every_locomotion_type(self) -> None:
        import collections
        assignments = self._assign()
        suffixes = ("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1")
        val = collections.Counter(); test = collections.Counter()
        for entry_id, split in assignments.items():
            suffix = suffixes[entry_id % 100]
            if split == "val":
                val[suffix] += 1
            elif split == "test":
                test[suffix] += 1
        self.assertEqual(sum(val.values()), 90)
        self.assertEqual(sum(test.values()), 90)
        # Balanced: no locomotion type may dominate a split (the old frame-count greedy gave
        # val 46% BR + 42% FR and test 92% running types).
        for counts in (val, test):
            self.assertGreaterEqual(min(counts.values()), 5)
            self.assertLessEqual(max(counts.values()), 20)
            for suffix in suffixes:
                self.assertIn(suffix, counts)

    def test_every_style_keeps_most_actions_in_train_and_never_shares_val_test(self) -> None:
        import collections
        assignments = self._assign()
        by_style = collections.defaultdict(dict)
        for entry_id, split in assignments.items():
            by_style[entry_id // 100][entry_id % 100] = split
        for style_index, per_clip in by_style.items():
            splits = collections.Counter(per_clip.values())
            self.assertEqual(splits["val"], 1)
            self.assertEqual(splits["test"], 1)
            self.assertEqual(splits["train"], 6)
            val_clip = next(clip for clip, split in per_clip.items() if split == "val")
            test_clip = next(clip for clip, split in per_clip.items() if split == "test")
            self.assertNotEqual(val_clip, test_clip)

    def test_styles_rotate_so_the_held_out_type_changes(self) -> None:
        import collections
        assignments = self._assign()
        val_clips = {entry_id // 100: entry_id % 100 for entry_id, split in assignments.items() if split == "val"}
        self.assertEqual(len(set(val_clips.values())), 8)  # all eight offsets occur


class GenerateDatabaseParallelTests(unittest.TestCase):
    def test_bounded_parallel_map_preserves_input_order(self) -> None:
        values = [-5, 3, -2, 0, 9, -1]
        serial = list(_bounded_ordered_map(abs, values, num_workers=1, max_pending=1))
        parallel = list(_bounded_ordered_map(abs, values, num_workers=2, max_pending=3))
        self.assertEqual(serial, [5, 3, 2, 0, 9, 1])
        self.assertEqual(parallel, serial)

    def test_vectorized_feature_extraction_shape(self) -> None:
        num_frames = 130
        num_joints = 72
        rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
        rotations[..., 0] = 1.0
        positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
        positions[:, 0, 2] = np.arange(num_frames, dtype=np.float32) * 0.01
        parents = np.arange(num_joints, dtype=np.int64) - 1

        features = extract_features_375(
            rotations,
            positions,
            parents,
            SELECTED_JOINTS,
            window=60,
        )

        self.assertEqual(tuple(features.shape), (9, 375))
        self.assertTrue(bool(np.isfinite(features).all()))

    def test_entry_worker_extracts_features(self) -> None:
        num_frames = 130
        num_joints = 72
        bvh_data = {
            "positions": np.zeros((num_frames, num_joints, 3), dtype=np.float32),
            "rotations": np.zeros((num_frames, num_joints, 3), dtype=np.float32),
            "parents": np.arange(num_joints, dtype=np.int64) - 1,
            "names": [f"joint_{idx}" for idx in range(num_joints)],
            "order": "zyx",
        }
        job = EntryJob(
            path=Path("/tmp/fake.bvh"),
            style_name="test",
            suffix="ID",
            source_entry_id=1,
            source_global_label=0,
            source_start=0,
            source_stop=num_frames,
            label=0,
            dataset_role="in_distribution",
            split="train",
            coordinate_convert=False,
            window=60,
        )

        with mock.patch.object(bvh, "load_zeroeggs", return_value=bvh_data):
            result = _process_entry_job(job)

        self.assertEqual(result.status, "ok")
        variants = result.variants
        self.assertEqual(len(variants), 1)
        self.assertEqual(tuple(variants[0].shape), (9, 375))

    def test_writer_emits_structured_metadata_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "db"
            writer = MemmapWriter(out_dir, ["test"], overwrite=False)
            writer.write(
                features=np.zeros((2, 375), dtype=np.float32),
                segment=SegmentSpec(
                    label=0,
                    style_name="test",
                    split="train",
                    source_entry_id=1,
                    source_path=Path("/tmp/fake.bvh"),
                    suffix="ID",
                    source_start=0,
                    source_stop=123,
                    source_global_label=0,
                ),
            )
            writer.finalize(
                dataset_role="in_distribution",
                generation={"tool": "test"},
                split={"strategy": "test"},
            )

            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["format"], "motion375_memmap")
            self.assertEqual(meta["schema_version"], 4)
            self.assertEqual(meta["files"]["features"]["shape"], [2, 375])
            self.assertEqual(meta["feature_layout"]["trajectory"], {"offset": 0, "dim": 84})
            self.assertEqual(meta["feature_layout"]["motion"], {"offset": 84, "dim": 291})
            self.assertEqual(meta["split"]["summary"]["train"], {"frames": 2, "segments": 1})
            self.assertEqual(meta["segment_counts"], {"total": 1})
            loaded = load_motion375_metadata(out_dir)
            self.assertEqual(loaded.num_frames, 2)
            self.assertEqual(loaded.num_styles, 1)
            self.assertEqual(loaded.num_segments, 1)
            self.assertEqual(loaded.dataset_role, "in_distribution")


if __name__ == "__main__":
    unittest.main()
