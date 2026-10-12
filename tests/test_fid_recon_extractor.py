from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
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

from Utils.fid_feature_extractor import (
    JOINT_SET_SELECTED24,
    JOINT_SET_YBOT15,
    MotionFeatureClassifier,
    MotionReconFeatureExtractor,
    build_fid_feature_extractor_from_checkpoint,
    infer_joint_set_from_in_feat,
)
from tests._memmap_fixtures import write_db
from tests.test_fid_joint_sets import _segments


class ReconExtractorModuleTests(unittest.TestCase):
    def test_shapes_and_feature_semantics(self) -> None:
        model = MotionReconFeatureExtractor(in_feat=45, window_size=60).eval()
        x = torch.randn(4, 45, 60)
        with torch.no_grad():
            feats, recon = model(x)
            self.assertEqual(tuple(feats.shape), (4, 512))
            self.assertEqual(tuple(recon.shape), tuple(x.shape))
            self.assertTrue(torch.equal(model.get_feats(x), model.encode(x)))
            self.assertTrue(torch.equal(model.decode(feats), recon))

    def test_input_validation(self) -> None:
        model = MotionReconFeatureExtractor(in_feat=45, window_size=60).eval()
        with self.assertRaises(ValueError):
            model.encode(torch.randn(2, 45, 30))
        with self.assertRaises(ValueError):
            model.encode(torch.randn(2, 72, 60))
        with self.assertRaises(ValueError):
            model.encode(torch.randn(2, 45))

    def test_reconstruction_error_decreases(self) -> None:
        torch.manual_seed(0)
        model = MotionReconFeatureExtractor(in_feat=45, window_size=24)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(16, 45, 24) * 0.5
        with torch.no_grad():
            _, first = model(x)
            first_loss = float(torch.nn.functional.l1_loss(first, x))
        for _ in range(12):
            optimizer.zero_grad()
            _, recon = model(x)
            loss = torch.nn.functional.l1_loss(recon, x)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            _, last = model(x)
            last_loss = float(torch.nn.functional.l1_loss(last, x))
        self.assertLess(last_loss, first_loss * 0.9)


class ExtractorDispatchTests(unittest.TestCase):
    def _write(self, path: Path, payload: dict) -> None:
        torch.save(payload, path)

    def test_classification_checkpoint_keeps_using_the_classifier(self) -> None:
        model = MotionFeatureClassifier(in_feat=45, class_num=4, window_size=60)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "cls.pt"
            self._write(
                ckpt,
                {
                    "state_dict": model.state_dict(),
                    "in_feat": 45,
                    "class_names": ["a", "b", "c", "d"],
                    "acc": 99.0,
                },
            )
            loaded, meta = build_fid_feature_extractor_from_checkpoint(ckpt)
            self.assertIsInstance(loaded, MotionFeatureClassifier)
            self.assertEqual(meta["joint_set"], JOINT_SET_YBOT15)
            self.assertEqual(meta["class_num"], 4)

    def test_reconstruction_checkpoint_loads_without_class_names(self) -> None:
        model = MotionReconFeatureExtractor(in_feat=45, window_size=60)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "recon.pt"
            self._write(
                ckpt,
                {
                    "state_dict": model.state_dict(),
                    "objective": "reconstruction",
                    "in_feat": 45,
                    "window_size": 60,
                    "feat_dim": 512,
                    "hidden_dim": 1024,
                    "recon_loss": 0.006,
                },
            )
            loaded, meta = build_fid_feature_extractor_from_checkpoint(ckpt)
            self.assertIsInstance(loaded, MotionReconFeatureExtractor)
            self.assertEqual(meta["class_names"], [])
            self.assertEqual(meta["class_num"], 0)
            self.assertEqual(meta["joint_set"], infer_joint_set_from_in_feat(45))
            # the evaluation path only needs get_feats
            feats = loaded.get_feats(torch.randn(2, 45, 60))
            self.assertEqual(tuple(feats.shape), (2, 512))

    def test_joint_set_override_is_validated_for_both_flavours(self) -> None:
        model = MotionReconFeatureExtractor(in_feat=45, window_size=60)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "recon.pt"
            self._write(
                ckpt,
                {
                    "state_dict": model.state_dict(),
                    "objective": "reconstruction",
                    "in_feat": 45,
                    "window_size": 60,
                },
            )
            _, meta = build_fid_feature_extractor_from_checkpoint(ckpt, joint_set=JOINT_SET_YBOT15)
            self.assertEqual(meta["joint_set"], JOINT_SET_YBOT15)
            with self.assertRaises(ValueError):
                build_fid_feature_extractor_from_checkpoint(ckpt, joint_set=JOINT_SET_SELECTED24)

    def test_unknown_objective_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "weird.pt"
            self._write(ckpt, {"state_dict": {}, "objective": "contrastive", "in_feat": 45})
            with self.assertRaises(ValueError):
                build_fid_feature_extractor_from_checkpoint(ckpt)


class ReconTrainerTests(unittest.TestCase):
    def test_two_epoch_run_saves_a_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "db"
            db.mkdir()
            write_db(db, num_frames=400, num_styles=2, segments=_segments(200), seed=9)
            out = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(NEWCODE_DIR / "training" / "train_fid_feature_extractor_recon.py"),
                    "--db",
                    str(db),
                    "--out-dir",
                    str(out),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--window-size",
                    "60",
                    "--clip-offset",
                    "10",
                    "--log-interval",
                    "0",
                ],
                cwd=str(NEWCODE_DIR),
                check=True,
                capture_output=True,
                text=True,
            )
            ckpt_path = out / "100style_position_recon.pth"
            self.assertTrue(ckpt_path.exists())
            model, meta = build_fid_feature_extractor_from_checkpoint(ckpt_path)
            self.assertIsInstance(model, MotionReconFeatureExtractor)
            self.assertEqual(meta["objective"], "reconstruction")
            self.assertEqual(meta["in_feat"], 45)
            self.assertEqual(meta["joint_set"], JOINT_SET_YBOT15)
            history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
            self.assertEqual([record["epoch"] for record in history], [1, 2])
            self.assertTrue(np.isfinite(history[-1]["recon_loss"]))
            # no style labels are used: the dataset still yields labels, the loss ignores them
            self.assertEqual(meta["class_num"], 0)


if __name__ == "__main__":
    unittest.main()
