from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from arch.basic_vqvae_375to291 import BasicVQVAE375to291


def build_model() -> BasicVQVAE375to291:
    torch.manual_seed(0)
    return BasicVQVAE375to291(
        input_dim=375,
        traj_dim=84,
        motion_dim=291,
        input_mode="traj_motion",
        encoder_arch="mlp",
        decoder_arch="mlp",
        hidden_dim=64,
        encoder_layers=3,
        decoder_hidden_dim=64,
        decoder_layers=3,
        num_latents=8,
        code_dim=4,
        num_embeddings=64,
        commitment_cost=0.25,
        use_rotation_trick=True,
    )


class CodeAlignmentTests(unittest.TestCase):
    """Continuous/discrete alignment: latent gap logging and the decode-consistency term."""

    def test_disabled_by_default_and_gap_always_logged(self):
        model = build_model().eval()
        x = torch.randn(2, 4, 375)
        with_zero = model.compute_loss(x, code_align_weight=0.0)
        default = model.compute_loss(x)
        self.assertEqual(float(with_zero["code_align_loss"]), 0.0)
        self.assertAlmostEqual(float(with_zero["loss"]), float(default["loss"]), places=6)
        gap = float(with_zero["latent_code_gap"])
        details = model.forward(x, return_details=True)
        expected_gap = float((details["latent"] - details["quantized_latent"]).square().mean())
        self.assertGreaterEqual(gap, 0.0)
        self.assertAlmostEqual(gap, expected_gap, places=6)

    def test_continuous_decode_is_opt_in(self):
        # eval(): the EMA codebook only updates in training mode, so two forwards with the
        # same input are comparable here.
        model = build_model().eval()
        x = torch.randn(2, 3, 375)
        details = model.forward(x, return_details=True)
        self.assertIsNone(details["motion_hat_continuous"])
        with_continuous = model.forward(x, return_details=True, continuous_decode=True)
        self.assertIsNotNone(with_continuous["motion_hat_continuous"])
        # The quantized decode must not change when the extra pass is requested.
        torch.testing.assert_close(details["motion_hat"], with_continuous["motion_hat"])

    def test_alignment_value_matches_external_mse_and_enters_total(self):
        model = build_model().eval()
        x = torch.randn(2, 5, 375)
        details = model.forward(x, return_details=True, continuous_decode=True)
        expected = float((details["motion_hat_continuous"] - details["motion_hat"].detach()).square().mean())
        aligned = model.compute_loss(x, recon_weight=0.0, vq_weight=0.0, code_align_weight=1.0)
        zero = model.compute_loss(x, recon_weight=0.0, vq_weight=0.0, code_align_weight=0.0)
        self.assertAlmostEqual(float(aligned["code_align_loss"]), expected, places=6)
        self.assertGreater(float(aligned["code_align_loss"]), 0.0)
        self.assertAlmostEqual(
            float(aligned["loss"]) - float(zero["loss"]),
            float(aligned["code_align_loss"]),
            places=5,
        )

    def test_alignment_term_trains_encoder_and_decoder(self):
        model = build_model()
        x = torch.randn(4, 4, 375)
        loss = model.compute_loss(x, recon_weight=0.0, vq_weight=0.0, code_align_weight=1.0)["code_align_loss"]
        model.zero_grad()
        loss.backward()
        encoder = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        decoder = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
        self.assertGreater(encoder, 0.0)
        self.assertGreater(decoder, 0.0)
        # The EMA codebook is a buffer, not a parameter: it must never receive gradients.
        codebook_grads = [p for p in model.quantizer.parameters() if p.grad is not None]
        self.assertEqual(codebook_grads, [])

    def test_negative_weight_is_rejected(self):
        model = build_model().eval()
        with self.assertRaises(ValueError):
            model.compute_loss(torch.randn(1, 2, 375), code_align_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
