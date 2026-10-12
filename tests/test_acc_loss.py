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


class AccelerationLossTests(unittest.TestCase):
    """Second-order term: |d2 pred - d2 target| on the motion block."""

    def test_vanishes_when_prediction_matches_target(self) -> None:
        model = build_model()
        target = torch.randn(2, 5, 291)
        zero_pred = target
        # Use target_motion == x and a monkeypatched forward so pred == target exactly.
        original = model.forward

        def fake_forward(x, return_details=False, *, continuous_decode=False):  # type: ignore[no-untyped-def]
            details = original(x, return_details=True)
            details["motion_hat"] = target
            return details if return_details else (details["motion_hat"], details["vq_loss"], details["perplexity"])

        model.forward = fake_forward  # type: ignore[assignment]
        out = model.compute_loss(torch.randn(2, 5, 375), target_motion=target, acc_weight=1.0)
        self.assertAlmostEqual(float(out["acc_loss"]), 0.0, places=6)
        self.assertGreater(float(out["loss"]), 0.0)  # other terms still active

    def test_penalises_per_frame_noise(self) -> None:
        model = build_model()
        target = torch.zeros(2, 6, 291)
        original = model.forward
        preds = {}

        def make_forward(pred: torch.Tensor):  # type: ignore[no-untyped-def]
            def fake(x, return_details=False, *, continuous_decode=False):  # type: ignore[no-untyped-def]
                details = original(x, return_details=True)
                details["motion_hat"] = pred
                return details if return_details else (details["motion_hat"], details["vq_loss"], details["perplexity"])

            return fake

        smooth = torch.zeros(2, 6, 291)
        noisy = torch.zeros(2, 6, 291)
        noisy[:, 1::2] = 0.5  # alternating -> large second difference

        model.forward = make_forward(smooth)  # type: ignore[assignment]
        smooth_loss = float(model.compute_loss(torch.randn(2, 6, 375), target_motion=target, acc_weight=1.0)["acc_loss"])
        model.forward = make_forward(noisy)  # type: ignore[assignment]
        noisy_loss = float(model.compute_loss(torch.randn(2, 6, 375), target_motion=target, acc_weight=1.0)["acc_loss"])
        self.assertAlmostEqual(smooth_loss, 0.0, places=6)
        self.assertGreater(noisy_loss, 0.5)
        preds["noisy"] = noisy_loss

    def test_zero_weight_matches_paper_recipe(self) -> None:
        model = build_model()
        model.eval()  # EMA codebook updates in train mode would change the second call
        torch.manual_seed(1)
        x = torch.randn(2, 5, 375)
        with_zero = model.compute_loss(x, acc_weight=0.0)
        self.assertAlmostEqual(float(with_zero["acc_loss"]), 0.0, places=7)
        # acc_weight=0 must not change the reported total
        no_arg = model.compute_loss(x)
        self.assertAlmostEqual(float(with_zero["loss"]), float(no_arg["loss"]), places=6)

    def test_short_window_is_a_noop(self) -> None:
        """seq_len < 3 cannot form a second difference: the term stays zero, no crash."""
        model = build_model()
        x = torch.randn(2, 2, 375)
        out = model.compute_loss(x, acc_weight=5.0)
        self.assertAlmostEqual(float(out["acc_loss"]), 0.0, places=7)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_type_selection(self) -> None:
        model = build_model()
        x = torch.randn(2, 4, 375)
        l1 = float(model.compute_loss(x, acc_weight=1.0, temporal_loss_type="l1")["acc_loss"].detach())
        mse = float(model.compute_loss(x, acc_weight=1.0, temporal_loss_type="mse")["acc_loss"].detach())
        self.assertGreater(l1, 0.0)
        self.assertGreater(mse, 0.0)
        self.assertNotAlmostEqual(l1, mse, places=4)


class AccWeightCliGuardTests(unittest.TestCase):
    def test_seq_len_guard(self) -> None:
        """--acc-weight > 0 with --seq-len < 3 is rejected before training starts."""
        import sys as _sys

        import training.train_basic_vqvae_375to291 as train

        base = ["prog", "--db", "x", "--out-dir", "y", "--acc-weight", "3.0"]
        saved = _sys.argv
        try:
            _sys.argv = base + ["--seq-len", "2"]
            with self.assertRaises(ValueError):
                train.validate_args(train.parse_args())
            _sys.argv = base + ["--seq-len", "3"]
            train.validate_args(train.parse_args())  # must not raise
        finally:
            _sys.argv = saved


if __name__ == "__main__":
    unittest.main()
