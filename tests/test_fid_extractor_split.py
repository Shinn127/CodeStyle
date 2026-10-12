from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

NEWCODE_DIR = Path(__file__).resolve().parents[1]

def _script_path(name: str) -> Path:
    for directory in ("training", "evaluation", "tools"):
        candidate = NEWCODE_DIR / directory / name
        if candidate.exists():
            return candidate
    return NEWCODE_DIR / name

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from tests._memmap_fixtures import write_db  # noqa: E402
from training.train_fid_feature_extractor_camdm import (  # noqa: E402
    CamdmWindowDataset,
    plan_clip_split,
    plan_suffix_split,
    plan_temporal_windows,
)
from Utils.fid_feature_extractor import (  # noqa: E402
    MotionFeatureClassifier,
    MotionReconFeatureExtractor,
)


def _segments_per_style(num_styles: int, clips_per_style: int, frames: int = 300) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for label in range(num_styles):
        for index in range(clips_per_style):
            segments.append(
                {
                    "label": label,
                    "style_name": f"style_{label}",
                    "source_entry_id": label * 100 + index,
                    "source_path": f"/tmp/{label}_{index}.bvh",
                    "suffix": "BR",
                    "frame_start": len(segments) * frames,
                    "num_frames": frames,
                    "source_start": 0,
                    "source_stop": frames - 1,
                    "source_global_label": label,
                    "split": "train",
                }
            )
    return segments


class ClipSplitTests(unittest.TestCase):
    def test_split_is_clip_level_per_style_and_disjoint(self) -> None:
        segments = _segments_per_style(num_styles=4, clips_per_style=5)
        train, val = plan_clip_split(segments, val_ratio=0.2, seed=7, num_styles=4)
        self.assertEqual(len(train) + len(val), len(segments))
        train_ids = {int(s["source_entry_id"]) for s in train}
        val_ids = {int(s["source_entry_id"]) for s in val}
        self.assertFalse(train_ids & val_ids)
        for label in range(4):
            self.assertEqual(sum(1 for s in val if int(s["label"]) == label), 1)
            self.assertGreaterEqual(sum(1 for s in train if int(s["label"]) == label), 1)
        # Deterministic for the same seed, different for another seed.
        again = plan_clip_split(segments, val_ratio=0.2, seed=7, num_styles=4)
        self.assertEqual({id(s) for s in again[1]}, val_ids and {id(s) for s in val})
        other = plan_clip_split(segments, val_ratio=0.4, seed=7, num_styles=4)
        self.assertGreater(len(other[1]), len(val))

    def test_zero_ratio_and_single_clip_styles_stay_in_train(self) -> None:
        segments = _segments_per_style(num_styles=2, clips_per_style=1)
        train, val = plan_clip_split(segments, val_ratio=0.5, seed=0, num_styles=2)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(train), 2)
        train, val = plan_clip_split(segments, val_ratio=0.0, seed=0, num_styles=2)
        self.assertEqual((len(train), len(val)), (2, 0))

    def test_dataset_respects_split_and_windows_never_straddle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "db"
            db.mkdir()
            segments = _segments_per_style(num_styles=3, clips_per_style=4, frames=100)
            write_db(db, num_frames=len(segments) * 100, num_styles=3, segments=segments)
            train = CamdmWindowDataset(db=db, window_size=60, clip_offset=10, split="train",
                                       val_ratio=0.25, split_seed=1, split_by="clip")
            val = CamdmWindowDataset(db=db, window_size=60, clip_offset=10, split="val",
                                     val_ratio=0.25, split_seed=1, split_by="clip")
            self.assertEqual(train.clip_counts["train"], 9)
            self.assertEqual(val.clip_counts["val"], 3)
            self.assertEqual(len(train) + 0, 9 * len(range(0, 100 - 60 + 1, 10)))
            # No window start can appear in both splits (clips are whole units).
            self.assertFalse(set(train.starts.tolist()) & set(val.starts.tolist()))
            # 1 val clip per style, 5 windows per 100-frame clip (60-frame window, offset 10)
            self.assertEqual(val.style_frequency().tolist(), [5, 5, 5])


class TemporalSplitTests(unittest.TestCase):
    """Every (style, suffix) contributes to both splits with a non-overlapping guard band."""

    def _segment(self, start: int, frames: int) -> dict[str, object]:
        return {"frame_start": start, "num_frames": frames, "label": 0, "suffix": "FW", "source_entry_id": start}

    def test_every_clip_contributes_to_both_splits_without_overlap(self) -> None:
        segment = self._segment(0, 1200)
        train, val = plan_temporal_windows(segment, window_size=240, clip_offset=10, val_ratio=0.1)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        guard = 240
        self.assertLessEqual(max(train) + 240, min(val) - 0)  # no window overlaps the val region
        self.assertGreaterEqual(min(val), 1200 - max(240, round(1200 * 0.1)))
        # Windows are aligned to clip_offset and stay inside the clip.
        self.assertTrue(all(start % 10 == 0 for start in train + val))
        self.assertTrue(all(start + 240 <= 1200 for start in train + val))

    def test_short_clips_fall_back_to_train_only(self) -> None:
        segment = self._segment(0, 300)
        train, val = plan_temporal_windows(segment, window_size=240, clip_offset=10, val_ratio=0.1)
        self.assertGreater(len(train), 0)
        self.assertEqual(val, [])

    def test_suffix_split_holds_out_whole_suffixes_for_every_style(self) -> None:
        segments = []
        for label in range(3):
            for suffix in ("FW", "SR", "ID"):
                segments.append({"frame_start": len(segments) * 100, "num_frames": 100,
                                 "label": label, "suffix": suffix, "source_entry_id": label * 10 + len(segments)})
        train, val = plan_suffix_split(segments, val_suffixes=("SR",))
        self.assertEqual(len(val), 3)
        self.assertTrue(all(segment["suffix"] == "SR" for segment in val))
        self.assertEqual(len(train), 6)
        with self.assertRaises(ValueError):
            plan_suffix_split(segments, val_suffixes=())


class DropoutTests(unittest.TestCase):
    def test_classifier_dropout_is_configurable_and_active_only_in_train(self) -> None:
        torch.manual_seed(0)
        model = MotionFeatureClassifier(in_feat=45, class_num=3, window_size=8, dropout=0.3, head_dropout=0.3)
        self.assertAlmostEqual(model.dropout_prob, 0.3)
        self.assertAlmostEqual(model.head_dropout_prob, 0.3)
        x = torch.randn(4, 45, 8)
        model.train()
        first = model.get_feats(x)
        second = model.get_feats(x)
        self.assertFalse(torch.allclose(first, second))
        model.eval()
        torch.testing.assert_close(model.get_feats(x), model.get_feats(x))
        with self.assertRaises(ValueError):
            MotionFeatureClassifier(in_feat=45, class_num=3, window_size=8, dropout=1.0)
        with self.assertRaises(ValueError):
            MotionFeatureClassifier(in_feat=45, class_num=3, window_size=8, head_dropout=-0.1)

    def test_recon_extractor_dropout_matches_the_same_contract(self) -> None:
        torch.manual_seed(0)
        model = MotionReconFeatureExtractor(in_feat=45, window_size=8, dropout=0.3, head_dropout=0.3)
        self.assertAlmostEqual(model.dropout_prob, 0.3)
        self.assertAlmostEqual(model.head_dropout_prob, 0.3)
        x = torch.randn(4, 45, 8)
        model.train()
        self.assertFalse(torch.allclose(model.encode(x), model.encode(x)))
        model.eval()
        torch.testing.assert_close(model.encode(x), model.encode(x))


class ExtractorValidationSplitRunTests(unittest.TestCase):
    """A short real run must produce val metrics and a val-selected canonical checkpoint."""

    def _run(self, script: str, out: Path, extra: list[str]) -> None:
        subprocess.run(
            [
                sys.executable,
                str(_script_path(script)),
                "--db", str(out.parent / "db"),
                "--out-dir", str(out),
                "--epochs", "2",
                "--batch-size", "8",
                "--num-workers", "0",
                "--device", "cpu",
                "--window-size", "60",
                "--clip-offset", "10",
                "--log-interval", "0",
                "--val-ratio", "0.25",
                "--dropout", "0.3",
                "--head-dropout", "0.3",
                *extra,
            ],
            cwd=str(NEWCODE_DIR),
            check=True,
            capture_output=True,
            text=True,
        )

    def _fixture(self, root: Path, num_styles: int = 3, clips: int = 4) -> Path:
        db = root / "db"
        db.mkdir()
        # 400-frame clips: long enough for the temporal holdout (2 x window + guard).
        segments = _segments_per_style(num_styles=num_styles, clips_per_style=clips, frames=400)
        write_db(db, num_frames=len(segments) * 400, num_styles=num_styles, segments=segments)
        return db

    def test_classifier_records_val_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            out = root / "out"
            self._run("train_fid_feature_extractor_camdm.py", out, [])
            payload = torch.load(out / "100style_position_classifier.pth", map_location="cpu", weights_only=False)
            self.assertEqual(payload["acc_split"], "val")
            self.assertEqual(payload["split_protocol"], "extractor_split_by_temporal")
            self.assertEqual(payload["val_windows"] > 0, True)
            self.assertEqual(payload["selection_rule"], "max_val_acc_then_min_val_loss")
            self.assertIsInstance(payload["best_epoch"], int)
            self.assertIsNone(payload["val_split_warning"])
            self.assertAlmostEqual(payload["dropout"], 0.3)
            self.assertAlmostEqual(payload["head_dropout"], 0.3)
            self.assertTrue((out / "best.pt").exists())
            self.assertTrue((out / "final.pt").exists())
            records = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertIsNotNone(records[-1]["val"])
            self.assertGreater(records[-1]["val"]["windows"], 0)
            # The canonical file is the val-selected best.pt.
            best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(max(payload["best_epoch"], 1), best["epoch"])
            for key, value in best["state_dict"].items():
                torch.testing.assert_close(value, payload["state_dict"][key])

    def test_recon_extractor_records_val_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            out = root / "out"
            self._run("train_fid_feature_extractor_recon.py", out, ["--window-size", "60"])
            payload = torch.load(out / "100style_position_recon.pth", map_location="cpu", weights_only=False)
            self.assertEqual(payload["acc_split"], "val")
            self.assertEqual(payload["selection_rule"], "min_val_recon_loss")
            self.assertIsInstance(payload["best_epoch"], int)
            self.assertGreater(payload["val_windows"], 0)
            self.assertIsNone(payload["val_split_warning"])
            self.assertTrue((out / "best.pt").exists())
            records = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
            self.assertIsNotNone(records[-1]["val"])
            self.assertGreater(records[-1]["val"]["recon_loss"], 0.0)


class EmptyValidationFallbackTests(unittest.TestCase):
    def test_too_short_clips_record_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "db"
            db.mkdir()
            # 100-frame clips cannot host 2 x window + guard, so the temporal split is empty.
            segments = _segments_per_style(num_styles=2, clips_per_style=3, frames=100)
            write_db(db, num_frames=len(segments) * 100, num_styles=2, segments=segments)
            out = root / "out"
            subprocess.run(
                [
                    sys.executable, str(NEWCODE_DIR / "training" / "train_fid_feature_extractor_camdm.py"),
                    "--db", str(db), "--out-dir", str(out), "--epochs", "1",
                    "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
                    "--window-size", "60", "--clip-offset", "10", "--log-interval", "0",
                ],
                cwd=str(NEWCODE_DIR), check=True, capture_output=True, text=True,
            )
            payload = torch.load(out / "100style_position_classifier.pth", map_location="cpu", weights_only=False)
            self.assertEqual(payload["val_split_warning"], "empty_val_split")
            self.assertEqual(payload["selection_rule"], "no_val_split")
            self.assertEqual(payload["val_windows"], 0)
            self.assertFalse((out / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
