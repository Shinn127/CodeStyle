from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.fid_feature_extractor import YBOT15_TO_SELECTED24_INDICES, MotionFeatureClassifier
from training.train_flowmatch_vqvae_controller import style_consistency_loss


class StyleConsistencyLossTests(unittest.TestCase):
    def test_loss_is_finite_and_reaches_the_decoded_motion(self) -> None:
        torch.manual_seed(0)
        classifier = MotionFeatureClassifier(in_feat=45, class_num=4, window_size=1)
        for parameter in classifier.parameters():
            parameter.requires_grad_(False)
        pred_motion = torch.randn(6, 291, requires_grad=True)
        style = torch.tensor([0, 1, 2, 3, 0, 1])
        mean = torch.zeros(375)
        std = torch.ones(375)
        loss = style_consistency_loss(
            pred_motion=pred_motion,
            style_label=style,
            classifier=classifier,
            motion_mean=mean,
            motion_std=std,
            joint_indices=YBOT15_TO_SELECTED24_INDICES,
            window_size=1,
        )
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(pred_motion.grad)
        self.assertTrue(bool((pred_motion.grad != 0).any()))
        # the frozen classifier receives no gradient
        self.assertTrue(all(parameter.grad is None for parameter in classifier.parameters()))

    def test_window_size_must_be_one(self) -> None:
        classifier = MotionFeatureClassifier(in_feat=45, class_num=4, window_size=1)
        with self.assertRaises(ValueError):
            style_consistency_loss(
                pred_motion=torch.randn(2, 291),
                style_label=torch.tensor([0, 1]),
                classifier=classifier,
                motion_mean=torch.zeros(375),
                motion_std=torch.ones(375),
                joint_indices=YBOT15_TO_SELECTED24_INDICES,
                window_size=60,
            )


if __name__ == "__main__":
    unittest.main()
