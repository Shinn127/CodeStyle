from __future__ import annotations

import json
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
    YBOT15_JOINT_NAMES,
    YBOT15_TO_SELECTED24_INDICES,
    MotionFeatureClassifier,
    build_motion_feature_classifier_from_checkpoint,
    infer_joint_set_from_in_feat,
    joint_count_for,
    motion375_clip_to_classifier_input,
)
from tests._memmap_fixtures import write_db


def _segments(n_frames: int = 200) -> list[dict[str, object]]:
    out = []
    start = 0
    for label in (0, 1):
        out.append(
            {
                "frame_start": start,
                "num_frames": n_frames,
                "label": label,
                "style_name": f"style_{label}",
                "source_entry_id": label,
                "source_path": f"/tmp/{label}.bvh",
                "suffix": "BR",
                "mirror": False,
                "source_start": 0,
                "source_stop": n_frames - 1,
                "source_global_label": label,
                "split": "train",
            }
        )
        start += n_frames
    return out


class JointSetMappingTests(unittest.TestCase):
    def test_ybot_mapping_matches_camdm_joint_order(self) -> None:
        from Utils.motion375 import SELECTED_24_NAMES

        self.assertEqual(len(YBOT15_JOINT_NAMES), 15)
        self.assertEqual(
            [SELECTED_24_NAMES[idx] for idx in YBOT15_TO_SELECTED24_INDICES],
            YBOT15_JOINT_NAMES,
        )
        # CAMDM's ybot_skeleton order, verbatim
        self.assertEqual(
            YBOT15_JOINT_NAMES,
            [
                "Hips",
                "Neck",
                "Head",
                "RightShoulder",
                "RightArm",
                "RightForeArm",
                "LeftShoulder",
                "LeftArm",
                "LeftForeArm",
                "RightUpLeg",
                "RightLeg",
                "RightFoot",
                "LeftUpLeg",
                "LeftLeg",
                "LeftFoot",
            ],
        )

    def test_ybot_input_is_a_channel_subset_of_the_24_joint_input(self) -> None:
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.standard_normal((3, 60, 375)).astype(np.float32))
        full = motion375_clip_to_classifier_input(x, joint_set=JOINT_SET_SELECTED24)
        small = motion375_clip_to_classifier_input(x, joint_set=JOINT_SET_YBOT15)
        channels = [3 * joint + axis for joint in YBOT15_TO_SELECTED24_INDICES for axis in range(3)]
        self.assertEqual(tuple(full.shape), (3, 72, 60))
        self.assertEqual(tuple(small.shape), (3, 45, 60))
        self.assertTrue(torch.equal(small, full[:, channels, :]))

    def test_joint_set_helpers_reject_unknown_inputs(self) -> None:
        self.assertEqual(infer_joint_set_from_in_feat(45), JOINT_SET_YBOT15)
        self.assertEqual(infer_joint_set_from_in_feat(72), JOINT_SET_SELECTED24)
        with self.assertRaises(ValueError):
            infer_joint_set_from_in_feat(48)
        with self.assertRaises(ValueError):
            joint_count_for("nope")


class ClassifierCheckpointJointSetTests(unittest.TestCase):
    def _write_ckpt(self, path: Path, *, in_feat: int, joint_set: str | None) -> None:
        model = MotionFeatureClassifier(in_feat=in_feat, class_num=4, window_size=60)
        payload = {
            "state_dict": model.state_dict(),
            "in_feat": in_feat,
            "acc": 99.0,
            "class_names": [f"s{idx}" for idx in range(4)],
        }
        if joint_set is not None:
            payload["joint_set"] = joint_set
        torch.save(payload, path)

    def test_joint_set_is_inferred_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for in_feat, expected in ((45, JOINT_SET_YBOT15), (72, JOINT_SET_SELECTED24)):
                ckpt = Path(tmp) / f"ckpt_{in_feat}.pt"
                self._write_ckpt(ckpt, in_feat=in_feat, joint_set=None)
                _, meta = build_motion_feature_classifier_from_checkpoint(ckpt)
                self.assertEqual(meta["joint_set"], expected)

    def test_recorded_joint_set_is_used_and_cross_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.pt"
            self._write_ckpt(good, in_feat=45, joint_set=JOINT_SET_YBOT15)
            _, meta = build_motion_feature_classifier_from_checkpoint(good)
            self.assertEqual(meta["joint_set"], JOINT_SET_YBOT15)

            bad = Path(tmp) / "bad.pt"
            self._write_ckpt(bad, in_feat=45, joint_set=JOINT_SET_SELECTED24)
            with self.assertRaises(ValueError):
                build_motion_feature_classifier_from_checkpoint(bad)

    def test_explicit_joint_set_must_match_the_input_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt.pt"
            self._write_ckpt(ckpt, in_feat=72, joint_set=None)
            _, meta = build_motion_feature_classifier_from_checkpoint(ckpt, joint_set=JOINT_SET_SELECTED24)
            self.assertEqual(meta["joint_set"], JOINT_SET_SELECTED24)
            with self.assertRaises(ValueError):
                build_motion_feature_classifier_from_checkpoint(ckpt, joint_set=JOINT_SET_YBOT15)


class CamdmTrainerTests(unittest.TestCase):
    """The CAMDM-recipe trainer must read 45-dim ybot inputs and save a loadable checkpoint."""

    def test_dataset_and_short_training_run(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "db"
            db.mkdir()
            x = write_db(db, num_frames=400, num_styles=2, segments=_segments(200), seed=5)
            out = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(NEWCODE_DIR / "training" / "train_fid_feature_extractor_camdm.py"),
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
                    # This test covers the 45-dim / joint-set contract on the all-clips
                    # protocol; the split protocols have their own tests.
                    "--val-ratio",
                    "0",
                ],
                cwd=str(NEWCODE_DIR),
                check=True,
                capture_output=True,
                text=True,
            )
            ckpt_path = out / "100style_position_classifier.pth"
            self.assertTrue(ckpt_path.exists())
            model, meta = build_motion_feature_classifier_from_checkpoint(ckpt_path)
            self.assertEqual(meta["in_feat"], 45)
            self.assertEqual(meta["joint_set"], JOINT_SET_YBOT15)
            self.assertEqual(meta["window_size"], 60)
            self.assertEqual(meta["class_names"], ["style_0", "style_1"])

            # the saved windows must be CAMDM's 15 joints read from the 24-joint block
            from training.train_fid_feature_extractor_camdm import CamdmWindowDataset

            dataset = CamdmWindowDataset(db=db, window_size=60, clip_offset=10)
            window, label = dataset[0]
            self.assertEqual(tuple(window.shape), (60, 45))
            channels = [3 * joint + axis for joint in YBOT15_TO_SELECTED24_INDICES for axis in range(3)]
            expected = np.asarray(x[0:60, 87:159], dtype=np.float32).reshape(60, 24, 3)
            expected = expected.reshape(60, 72)[:, channels]
            np.testing.assert_allclose(window.numpy(), expected, rtol=0, atol=0)
            self.assertEqual(int(label), 0)

            # every window is training data: no split files are consulted
            self.assertEqual(len(dataset), 2 * len(range(0, 200 - 60 + 1, 10)))

            history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
            self.assertEqual([record["epoch"] for record in history], [1, 2])
            self.assertEqual(history[-1]["windows"], len(dataset))


if __name__ == "__main__":
    unittest.main()
