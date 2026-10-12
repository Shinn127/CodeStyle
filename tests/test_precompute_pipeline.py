from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from _memmap_fixtures import write_db
from arch import BasicVQVAE375to291
from tools.precompute_style_code_prior import main as precompute_prior_main
from tools.precompute_vq_targets import main as precompute_targets_main


class PrecomputePipelineTests(unittest.TestCase):
    def test_chunked_vq_targets_and_style_prior(self) -> None:
        torch.manual_seed(23)
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "db"
            cache_path = root / "vq_cache"
            prior_path = root / "prior.npz"
            ckpt_path = root / "vq.pt"
            stats_path = root / "norm_stats.npz"
            db_path.mkdir()

            num_frames = 64
            x = write_db(
                db_path,
                num_frames=num_frames,
                num_styles=2,
                segments=[
                    {"frame_start": 0, "num_frames": 32, "label": 0, "split": "train"},
                    {"frame_start": 32, "num_frames": 32, "label": 1, "split": "train"},
                ],
            )
            np.savez(stats_path, mean=np.zeros(375, dtype=np.float32), std=np.ones(375, dtype=np.float32))

            model = BasicVQVAE375to291(
                hidden_dim=16,
                decoder_hidden_dim=16,
                encoder_layers=2,
                decoder_layers=2,
                num_latents=4,
                code_dim=2,
                num_embeddings=16,
                kmeans_init=True,
                kmeans_iters=2,
            )
            model.train()
            model(torch.from_numpy(x[:16]))
            checkpoint_args = {
                "model_type": "input_mode_vqvae_375to291",
                "input_dim": 375,
                "traj_dim": 84,
                "motion_dim": 291,
                "input_mode": "traj_motion",
                "encoder_arch": "mlp",
                "decoder_arch": "mlp",
                "hidden_dim": 16,
                "decoder_hidden_dim": 16,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "num_latents": 4,
                "code_dim": 2,
                "num_embeddings": 16,
                "use_rotation_trick": True,
                "kmeans_init": True,
                "kmeans_iters": 2,
            }
            torch.save({"model": model.state_dict(), "args": checkpoint_args}, ckpt_path)

            with mock.patch.object(
                sys,
                "argv",
                [
                    "precompute_vq_targets.py",
                    "--db",
                    str(db_path),
                    "--vqvae-ckpt",
                    str(ckpt_path),
                    "--norm-stats",
                    str(stats_path),
                    "--out-dir",
                    str(cache_path),
                    "--chunk-rows",
                    "17",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                ],
            ):
                precompute_targets_main()

            with mock.patch.object(
                sys,
                "argv",
                [
                    "precompute_style_code_prior.py",
                    "--db",
                    str(db_path),
                    "--vq-target-cache",
                    str(cache_path),
                    "--out",
                    str(prior_path),
                    "--split",
                    "train",
                    "--style-num-slots",
                    "2",
                    "--chunk-rows",
                    "13",
                ],
            ):
                precompute_prior_main()

            with np.load(prior_path, allow_pickle=True) as prior_data:
                counts = prior_data["counts"]
                prior = prior_data["prior"]
                global_prior = prior_data["global_prior"]
                saliency = prior_data["slot_saliency"]
                selected = prior_data["selected_slots"]

                # counts: one code per slot per frame (Eq. 1)
                self.assertEqual(tuple(counts.shape), (2, 4, 16))
                self.assertEqual(int(counts.sum()), num_frames * 4)

                # priors are proper distributions (Eq. 1)
                np.testing.assert_allclose(prior.sum(axis=-1), 1.0, atol=1e-5)
                np.testing.assert_allclose(global_prior.sum(axis=-1), 1.0, atol=1e-5)
                self.assertGreaterEqual(float(prior.min()), 0.0)

                # selected slots are the top-M saliency slots, ascending and unique (Eq. 2)
                self.assertEqual(selected.tolist(), sorted(set(selected.tolist())))
                self.assertEqual(len(selected), 2)
                order = np.argsort(-saliency, kind="stable")[: len(selected)]
                self.assertEqual(set(order.tolist()), set(selected.tolist()))

                # provenance for the codebook the prior was estimated from
                self.assertEqual(str(prior_data["vqvae_ckpt"]), str(ckpt_path))
                self.assertEqual(str(prior_data["db"]), str(db_path))
                self.assertEqual(int(prior_data["num_codebooks"]), 4)


if __name__ == "__main__":
    unittest.main()
