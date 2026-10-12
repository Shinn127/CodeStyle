from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from _memmap_fixtures import write_db
from dataset import (
    Motion375Dataset,
    _build_sample_starts,
    _compute_zscore_stats_from_starts,
    build_train_val_test_datasets,
)


def _write_test_db(root: Path, *, num_frames: int = 96) -> np.ndarray:
    return write_db(
        root,
        num_frames=num_frames,
        num_styles=3,
        segments=[
            {"frame_start": 0, "num_frames": 32, "label": 0, "split": "train"},
            {"frame_start": 32, "num_frames": 32, "label": 1, "split": "val"},
            {"frame_start": 64, "num_frames": 32, "label": 2, "split": "test"},
        ],
    )


class DatasetThroughputTests(unittest.TestCase):
    def test_vectorized_sample_starts_match_expected_runs(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
        starts = _build_sample_starts(labels, seq_len=2, stride=2, enforce_same_label=True)
        np.testing.assert_array_equal(starts, np.array([0, 3, 5], dtype=np.int64))

    def test_weighted_stats_match_explicit_window_materialization(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=(40, 375)).astype(np.float32)
        starts = np.array([0, 1, 4, 9, 17, 30], dtype=np.int64)
        seq_len = 5
        expected = np.concatenate([x[start : start + seq_len] for start in starts], axis=0)

        mean, std = _compute_zscore_stats_from_starts(
            sample_starts=starts,
            seq_len=seq_len,
            norm_eps=1e-8,
            x_source=x,
            windows_chunk_size=3,
        )

        np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-5, atol=1e-6)

    def test_batched_getitems_matches_scalar_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir)
            _write_test_db(db_path)
            dataset = Motion375Dataset(
                db_path,
                seq_len=4,
                normalize=True,
                norm_mean=np.zeros(375, dtype=np.float32),
                norm_std=np.full(375, 2.0, dtype=np.float32),
            )
            indices = [0, 3, 8, 12]
            batched = dataset.__getitems__(indices)

            for batch_item, index in zip(batched, indices):
                scalar_item = dataset[index]
                for key in ("x", "traj", "motion", "label", "start"):
                    torch.testing.assert_close(batch_item[key], scalar_item[key])

    def test_splits_share_loaded_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir)
            _write_test_db(db_path)
            train, val, test = build_train_val_test_datasets(
                db_path,
                seq_len=2,
                load_into_memory=True,
                normalize=False,
            )

            self.assertIsNotNone(val)
            self.assertIsNotNone(test)
            self.assertIs(train._x_mem, val._x_mem)
            self.assertIs(train._x_mem, test._x_mem)
            self.assertIs(train.labels, val.labels)
            self.assertIs(train.labels, test.labels)


if __name__ == "__main__":
    unittest.main()
