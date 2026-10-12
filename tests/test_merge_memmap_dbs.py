from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from tests._memmap_fixtures import write_db


MERGE_SCRIPT = NEWCODE_DIR / "tools" / "merge_memmap_dbs.py"


def _segments(entries: list[tuple[int, int, str, int, str]]) -> list[dict[str, object]]:
    """[(frame_start, num_frames, style, global_label, split)] -> segment records."""
    out = []
    for frame_start, num_frames, style, global_label, split in entries:
        out.append(
            {
                "frame_start": int(frame_start),
                "num_frames": int(num_frames),
                "label": int(style),
                "style_name": f"style_{style}",
                "source_entry_id": int(style),
                "source_path": f"/tmp/{style}.bvh",
                "suffix": "BR",
                "source_start": 0,
                "source_stop": int(num_frames) - 1,
                "source_global_label": int(global_label),
                "split": split,
            }
        )
    return out


def _rename_styles(db: Path, names: list[str]) -> None:
    """The fixture names styles `style_<local index>`; the merge needs globally unique names."""
    (db / "StyleNames.json").write_text(json.dumps(names), encoding="utf-8")
    segments = json.loads((db / "segments.json").read_text())
    for segment in segments:
        segment["style_name"] = names[int(segment["label"])]
    (db / "segments.json").write_text(json.dumps(segments), encoding="utf-8")


class MergeMemmapDBsTests(unittest.TestCase):
    def _write_sources(self, root: Path) -> tuple[Path, Path]:
        primary = root / "primary"
        secondary = root / "secondary"
        primary.mkdir()
        secondary.mkdir()
        # primary: styles 0 and 1 (global labels 0,1)
        primary_x = write_db(
            primary,
            num_frames=400,
            num_styles=2,
            segments=_segments(
                [
                    (0, 100, 0, 0, "train"),
                    (100, 100, 0, 0, "train"),
                    (200, 100, 1, 1, "val"),
                    (300, 100, 1, 1, "test"),
                ]
            ),
            seed=11,
        )
        # secondary: style 2 (global label 2), no split recorded
        secondary_segments = _segments([(0, 80, 0, 2, "train"), (80, 120, 0, 2, "train")])
        for segment in secondary_segments:
            segment.pop("split")
        secondary_x = write_db(
            secondary,
            num_frames=200,
            num_styles=1,
            segments=secondary_segments,
            dataset_role="zero_shot",
            seed=22,
        )
        _rename_styles(secondary, ["style_2"])
        return primary_x, secondary_x

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MERGE_SCRIPT), *args],
            cwd=str(NEWCODE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )

    def test_merge_remaps_labels_offsets_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_x, secondary_x = self._write_sources(root)
            out = root / "merged"
            self._run("--primary", str(root / "primary"), "--secondary", str(root / "secondary"), "--out", str(out))

            x = np.fromfile(out / "X.f32", dtype=np.float32).reshape(-1, 375)
            labels = np.fromfile(out / "Labels.i32", dtype=np.int32)
            segments = json.loads((out / "segments.json").read_text())
            meta = json.loads((out / "meta.json").read_text())
            styles = json.loads((out / "StyleNames.json").read_text())

            self.assertEqual(meta["num_frames"], 600)
            self.assertEqual(meta["num_styles"], 3)
            self.assertEqual(styles, ["style_0", "style_1", "style_2"])
            self.assertEqual(x.shape, (600, 375))
            np.testing.assert_allclose(x[:400], primary_x, rtol=0, atol=0)
            np.testing.assert_allclose(x[400:], secondary_x, rtol=0, atol=0)

            # primary labels are untouched, secondary labels are lifted to the global order
            np.testing.assert_array_equal(labels[:400], np.repeat([0, 0, 1, 1], 100))
            np.testing.assert_array_equal(labels[400:], np.full(200, 2))

            # secondary frames/splits are shifted and defaulted
            self.assertEqual([s["frame_start"] for s in segments], [0, 100, 200, 300, 400, 480])
            self.assertEqual([s["split"] for s in segments], ["train", "train", "val", "test", "train", "train"])
            self.assertEqual([s["label"] for s in segments], [0, 0, 1, 1, 2, 2])
            self.assertEqual(meta["split"]["summary"]["train"]["frames"], 400)
            self.assertEqual(meta["split"]["summary"]["test"]["segments"], 1)
            self.assertEqual(meta["segment_counts"], {"total": 6})
            self.assertEqual(meta["generation"]["tool"], "merge_memmap_dbs.py")
            self.assertEqual(len(meta["generation"]["derived_from"]), 2)

    def test_merged_database_loads_with_the_dataset_loader(self) -> None:
        from dataset import build_train_val_test_datasets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sources(root)
            out = root / "merged"
            self._run("--primary", str(root / "primary"), "--secondary", str(root / "secondary"), "--out", str(out))
            train, val, test = build_train_val_test_datasets(
                db_path=out, seq_len=20, stride=10, normalize=False, val_ratio=0.1, test_ratio=0.1, seed=0
            )
            self.assertGreater(len(train), 0)
            self.assertGreater(len(val), 0)
            self.assertGreater(len(test), 0)
            self.assertEqual(int(test.labels[int(test.sample_starts[0])]), 1)

    def test_overlapping_style_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sources(root)
            duplicate = root / "duplicate"
            duplicate.mkdir()
            write_db(
                duplicate,
                num_frames=100,
                num_styles=1,
                segments=_segments([(0, 100, 0, 5, "train")]),
                seed=33,
            )
            _rename_styles(duplicate, ["style_0"])
            # style_0 already exists in the primary database
            with self.assertRaises(subprocess.CalledProcessError):
                self._run(
                    "--primary",
                    str(root / "primary"),
                    "--secondary",
                    str(duplicate),
                    "--out",
                    str(root / "clash"),
                )

    def test_existing_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_sources(root)
            out = root / "merged"
            out.mkdir()
            with self.assertRaises(subprocess.CalledProcessError):
                self._run(
                    "--primary", str(root / "primary"), "--secondary", str(root / "secondary"), "--out", str(out)
                )
            self._run(
                "--primary",
                str(root / "primary"),
                "--secondary",
                str(root / "secondary"),
                "--out",
                str(out),
                "--overwrite",
            )
            self.assertTrue((out / "meta.json").exists())


if __name__ == "__main__":
    unittest.main()
