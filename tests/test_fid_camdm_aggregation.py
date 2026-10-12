from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.motion_eval import (
    camdm_clip_fid,
    camdm_style_fid,
    frechet_distance,
    frechet_distance_from_stats,
)


class FrechetFromStatsTests(unittest.TestCase):
    def test_stats_path_matches_the_feature_path(self) -> None:
        rng = np.random.default_rng(0)
        real = rng.normal(size=(40, 8))
        fake = rng.normal(loc=0.3, size=(50, 8))
        expected = frechet_distance(real, fake)
        got = frechet_distance_from_stats(
            real.mean(axis=0),
            np.cov(real, rowvar=False),
            fake.mean(axis=0),
            np.cov(fake, rowvar=False),
        )
        self.assertAlmostEqual(got, expected, places=10)

    def test_shape_mismatches_raise(self) -> None:
        with self.assertRaises(ValueError):
            frechet_distance_from_stats(np.zeros(4), np.eye(3), np.zeros(4), np.eye(3))
        with self.assertRaises(ValueError):
            frechet_distance_from_stats(np.zeros(4), np.eye(4), np.zeros(3), np.eye(3))


class CamdmAggregationTests(unittest.TestCase):
    def _data(self):
        rng = np.random.default_rng(1)
        labels = np.repeat([0, 1], 60)
        clips = np.repeat([0, 1, 2, 3], 30)
        real = rng.normal(size=(120, 12)).astype(np.float64)
        return real, labels, clips

    def test_identical_predictions_give_zero_for_the_style_variant(self) -> None:
        real, labels, _ = self._data()
        value, per_style, counts = camdm_style_fid(real, real.copy(), labels)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(set(per_style), {0, 1})
        self.assertEqual(counts[0], 60)

    def test_style_variant_equals_mean_of_per_group_fid_over_feature_dim(self) -> None:
        real, labels, _ = self._data()
        fake = real + 0.25
        value, _, _ = camdm_style_fid(real, fake, labels)
        expected = np.mean(
            [frechet_distance(real[labels == label], fake[labels == label]) / real.shape[1] for label in (0, 1)]
        )
        self.assertAlmostEqual(value, expected, places=10)

    def test_clip_variant_matches_the_style_floor_when_a_group_has_one_clip(self) -> None:
        real, labels, clips = self._data()
        # one clip per label -> the clip Gaussian is the style Gaussian, so the reference floor is 0
        clips_single = labels.copy()
        floor, used = camdm_clip_fid(real, real.copy(), labels, clips_single)
        self.assertAlmostEqual(floor, 0.0, places=12)
        self.assertEqual(used, 2)

        value, used_multi = camdm_clip_fid(real, real.copy(), labels, clips)
        self.assertEqual(used_multi, 4)
        self.assertGreater(value, 0.0)  # clip Gaussians differ from the pooled style Gaussian

    def test_groups_with_too_few_samples_are_skipped(self) -> None:
        real = np.zeros((5, 4))
        labels = np.array([0, 0, 0, 0, 1])
        value, per_style, counts = camdm_style_fid(real, real, labels, min_samples=2)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertEqual(set(per_style), {0})
        self.assertNotIn(1, counts)
        # clip 0 has 4 samples (kept), clip 1 only 1 (skipped)
        value_clip, used = camdm_clip_fid(real, real, labels, np.array([0, 0, 0, 0, 1]))
        self.assertEqual(used, 1)
        self.assertAlmostEqual(value_clip, 0.0, places=12)
        empty, used_none = camdm_clip_fid(real[:1], real[:1], labels[:1], np.array([0]))
        self.assertEqual(used_none, 0)
        self.assertTrue(np.isnan(empty))

    def test_length_mismatch_raises(self) -> None:
        real, labels, clips = self._data()
        with self.assertRaises(ValueError):
            camdm_style_fid(real, real, labels[:-1])
        with self.assertRaises(ValueError):
            camdm_clip_fid(real, real, labels, clips[:-1])


if __name__ == "__main__":
    unittest.main()
