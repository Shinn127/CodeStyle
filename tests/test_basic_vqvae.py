from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.model_loading import build_vqvae_from_checkpoint
from arch.basic_vqvae_375to291 import (
    BasicVQVAE375to291,
    LibraryVectorQuantizer,
)


class RotationTrickTests(unittest.TestCase):
    """The quantizer delegates the STE to vector_quantize_pytorch's rotation trick.

    Verified behaviour: the gradient is the upstream gradient rotated by the
    z-to-q rotation and scaled by |q| / |z|; with the trick disabled it is the plain
    straight-through gradient.
    """

    @staticmethod
    def _gradient(*, use_rotation_trick: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantizer = LibraryVectorQuantizer(
            num_embeddings=2,
            code_dim=2,
            num_codebooks=1,
            use_rotation_trick=use_rotation_trick,
            kmeans_init=False,
        )
        with torch.no_grad():
            quantizer.vq.codebook.copy_(torch.tensor([[2.0, -1.0], [0.0, 0.0]]))
        z = torch.tensor([[1.0, 2.0]], requires_grad=True)
        quantized = quantizer(z).quantized
        quantized.sum().backward()
        return z.detach(), quantized.detach(), z.grad.detach()

    def test_rotation_trick_preserves_scale_and_rotates_direction(self) -> None:
        z, quantized, grad = self._gradient(use_rotation_trick=True)
        upstream = torch.ones_like(z)
        self.assertTrue(torch.isfinite(grad).all())
        # |grad| == (|q| / |z|) * |upstream|
        expected_scale = float(quantized.norm() / z.norm())
        torch.testing.assert_close(
            grad.norm(), upstream.norm() * expected_scale, rtol=1e-4, atol=1e-5
        )
        # A rotation, not the identity map.
        self.assertFalse(torch.allclose(grad, upstream, atol=1e-3))

    def test_disabling_rotation_trick_falls_back_to_plain_straight_through(self) -> None:
        _, _, grad = self._gradient(use_rotation_trick=False)
        torch.testing.assert_close(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-6)


class EmaCodebookTests(unittest.TestCase):
    def test_ema_updates_stay_finite_and_bounded(self) -> None:
        """The library's update divides by a Laplace-smoothed cluster size.

        That makes every code move a little on every step, including unselected ones, so
        the invariant worth asserting is that repeated updates stay finite and stay within
        a sane multiple of the data scale instead of diverging.
        """
        torch.manual_seed(7)
        quantizer = LibraryVectorQuantizer(
            num_embeddings=64,
            code_dim=4,
            num_codebooks=2,
            ema_decay=0.99,
            ema_eps=1e-5,
            kmeans_init=False,
        )
        quantizer.train()
        data_scale = 3.0
        max_norm = 0.0
        for _ in range(50):
            quantizer(torch.randn(64, 8) * data_scale)
            codebook = quantizer.vq.codebook.detach()
            self.assertTrue(bool(torch.isfinite(codebook).all()))
            max_norm = max(max_norm, float(codebook.norm(dim=1).max()))

        # Generous bound: a healthy EMA codebook tracks the data, it does not explode.
        self.assertLess(max_norm, 50.0 * data_scale)

    def test_ema_moves_the_codebook_towards_the_data(self) -> None:
        torch.manual_seed(11)
        quantizer = LibraryVectorQuantizer(
            num_embeddings=16,
            code_dim=4,
            num_codebooks=1,
            ema_decay=0.9,
            kmeans_init=False,
        )
        target = torch.full((64, 4), 4.0)
        before = float(torch.cdist(quantizer.vq.codebook.detach(), target).min(dim=1).values.mean())
        quantizer.train()
        for _ in range(20):
            quantizer(target)
        after = float(torch.cdist(quantizer.vq.codebook.detach(), target).min(dim=1).values.mean())
        self.assertLess(after, before)


class CodebookMaintenanceTests(unittest.TestCase):
    def test_kmeans_initializes_codebook_from_first_training_batch(self) -> None:
        torch.manual_seed(13)
        quantizer = LibraryVectorQuantizer(
            num_embeddings=4,
            code_dim=2,
            num_codebooks=1,
            kmeans_init=True,
            kmeans_iters=5,
        )
        samples = torch.tensor(
            [
                [-5.1, -5.0],
                [-4.9, -5.0],
                [5.0, 5.1],
                [5.0, 4.9],
                [-5.0, 5.1],
                [-5.0, 4.9],
                [5.1, -5.0],
                [4.9, -5.0],
            ]
        )

        before = quantizer.vq.codebook.detach().clone()
        self.assertFalse(bool(quantizer.vq._codebook.initted))

        quantizer.train()
        quantizer(samples)

        # Initialisation must have run on the first training batch and snapped the codebook
        # onto the data instead of leaving the random values in place.
        self.assertTrue(bool(quantizer.vq._codebook.initted))
        actual = quantizer.vq.codebook.detach()
        self.assertFalse(torch.allclose(actual, before, atol=1e-3))
        # The library's k-means can land in a local minimum on adversarial data, so assert
        # the robust property: the initialised codebook sits far closer to the data than the
        # random initialisation did.
        after_mean = float(torch.cdist(samples, actual).min(dim=1).values.mean())
        before_mean = float(torch.cdist(samples, before).min(dim=1).values.mean())
        self.assertLess(after_mean, 0.5 * before_mean)

    def test_dead_codes_are_replaced_and_counted(self) -> None:
        torch.manual_seed(17)
        # A threshold above the per-code usage of this batch forces replacement, which is
        # also what the adapter counts for the `dead_codes_reset` metric.
        quantizer = LibraryVectorQuantizer(
            num_embeddings=4,
            code_dim=2,
            num_codebooks=1,
            ema_decay=0.99,
            kmeans_init=False,
            dead_code_threshold=64,
        )
        with torch.no_grad():
            quantizer.vq.codebook.copy_(
                torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
            )
        before = quantizer.vq.codebook.detach().clone()

        samples = torch.zeros(8, 2)
        quantizer.train()
        result = quantizer(samples)

        self.assertGreater(float(result.dead_codes_reset), 0.0)
        self.assertFalse(torch.equal(before, quantizer.vq.codebook.detach()))

    def test_zero_threshold_leaves_the_codebook_untouched(self) -> None:
        torch.manual_seed(19)
        quantizer = LibraryVectorQuantizer(
            num_embeddings=4,
            code_dim=2,
            num_codebooks=1,
            ema_decay=0.99,
            kmeans_init=False,
            dead_code_threshold=0,
        )
        quantizer.train()
        result = quantizer(torch.zeros(8, 2))
        self.assertEqual(float(result.dead_codes_reset), 0.0)


class ControlNetworkTests(unittest.TestCase):
    def test_all_network_architectures_preserve_vq_shapes(self) -> None:
        x = torch.randn(2, 3, 375)
        for architecture in ("mlp", "skip_mlp", "skip_cat_mlp"):
            with self.subTest(architecture=architecture):
                model = BasicVQVAE375to291(
                    hidden_dim=32,
                    decoder_hidden_dim=32,
                    encoder_layers=3,
                    decoder_layers=3,
                    num_latents=4,
                    code_dim=2,
                    num_embeddings=16,
                    encoder_arch=architecture,
                    decoder_arch=architecture,
                )
                details = model(x, return_details=True)
                self.assertEqual(tuple(details["motion_hat"].shape), (2, 3, 291))
                self.assertEqual(tuple(details["quantized_latent"].shape), (2, 3, 8))
                self.assertEqual(tuple(details["indices"].shape), (2, 3, 4))

    def test_checkpoint_loader_restores_new_and_legacy_architecture(self) -> None:
        for architecture, include_arch_fields in (("skip_mlp", True), ("mlp", False)):
            with self.subTest(architecture=architecture):
                model = BasicVQVAE375to291(
                    hidden_dim=16,
                    decoder_hidden_dim=16,
                    encoder_layers=2,
                    decoder_layers=2,
                    num_latents=2,
                    code_dim=2,
                    num_embeddings=8,
                    encoder_arch=architecture,
                    decoder_arch=architecture,
                )
                args = {
                    "model_type": "input_mode_vqvae_375to291",
                    "input_dim": 375,
                    "traj_dim": 84,
                    "motion_dim": 291,
                    "input_mode": "traj_motion",
                    "hidden_dim": 16,
                    "decoder_hidden_dim": 16,
                    "encoder_layers": 2,
                    "decoder_layers": 2,
                    "num_latents": 2,
                    "code_dim": 2,
                    "num_embeddings": 8,
                    "use_rotation_trick": True,
                    "kmeans_init": True,
                    "kmeans_iters": 10,
                }
                if include_arch_fields:
                    args.update(
                        {
                            "encoder_arch": architecture,
                            "decoder_arch": architecture,
                        }
                    )
                model(torch.randn(2, 1, 375))
                with tempfile.TemporaryDirectory() as tmp_dir:
                    checkpoint_path = Path(tmp_dir) / "model.pt"
                    torch.save({"model": model.state_dict(), "args": args}, checkpoint_path)
                    restored = build_vqvae_from_checkpoint(checkpoint_path)
                self.assertEqual(restored.encoder_arch, architecture)
                self.assertEqual(restored.decoder_arch, architecture)
                # Behavioural check that covers both quantizer implementations: the restored
                # model must assign exactly the same codes as the one that was saved.
                probe = torch.randn(2, 1, 375)
                with torch.no_grad():
                    _, expected, _ = model.encode(probe)
                    _, actual, _ = restored.encode(probe)
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
