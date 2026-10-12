from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.style_code_prior import (  # noqa: E402
    STYLE_PRIOR_LOSS_TYPES,
    load_style_code_prior,
    save_style_code_prior,
    style_code_entropy,
    style_prior_divergence,
)


def make_prior(batch: int = 2, slots: int = 3, codes: int = 16) -> torch.Tensor:
    logits = torch.randn(batch, slots, codes, generator=torch.Generator().manual_seed(0))
    return torch.softmax(logits, dim=-1)


def sparsify(probs: torch.Tensor, k: int = 4) -> torch.Tensor:
    values, indices = torch.topk(probs, k, dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, indices, values)
    return out / out.sum(dim=-1, keepdim=True)


class DensePriorLossTests(unittest.TestCase):
    """The default mode supervises the pre-TopK distribution, where KL = 0 is attainable."""

    def test_zero_only_at_the_prior(self) -> None:
        prior = make_prior()
        logits = torch.log(prior)  # softmax(log(prior)) == prior
        loss = style_prior_divergence(prior, sparsify(prior), logits=logits, loss_type="dense")
        expected = style_code_entropy(prior)  # CE(prior, prior) == H(prior)
        self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_never_dominated_by_a_clamp_constant(self) -> None:
        """The dense loss stays on the CE scale; the legacy one blows up on a diffuse prior."""
        prior = make_prior()
        sparse = sparsify(prior)
        dense = float(
            style_prior_divergence(prior, sparse, logits=torch.log(prior), loss_type="dense")
        )
        legacy = float(style_prior_divergence(prior, sparse, loss_type="legacy"))
        self.assertLess(dense, 5.0)
        self.assertGreater(legacy, dense)
        # the legacy value decomposes exactly into the in-support CE plus the clamp constant
        inside = -(prior * torch.log(sparse.clamp_min(1e-8))).sum(dim=-1).mean()
        self.assertAlmostEqual(legacy, float(inside), places=5)
        outside = float((prior - sparse).clamp_min(0).sum(dim=-1).mean())
        self.assertGreater(-np.log(1e-8) * outside, 1.0)  # the clamp term dominates
        self.assertGreater(legacy, -np.log(1e-8) * outside * 0.5)

    def test_requires_logits(self) -> None:
        prior = make_prior()
        with self.assertRaises(ValueError):
            style_prior_divergence(prior, sparsify(prior), loss_type="dense")

    def test_shape_and_temperature_validation(self) -> None:
        prior = make_prior()
        sparse = sparsify(prior)
        with self.assertRaises(ValueError):
            style_prior_divergence(prior, sparse, logits=torch.log(prior), temperature=0.0)
        with self.assertRaises(ValueError):
            style_prior_divergence(prior, sparse[:, :, :1], logits=torch.log(prior))
        with self.assertRaises(ValueError):
            style_prior_divergence(prior, sparse, logits=torch.log(prior), loss_type="nope")


class AlternativePriorLossTests(unittest.TestCase):
    def test_js_is_non_negative_and_bounded(self) -> None:
        prior = make_prior()
        sparse = sparsify(prior)
        value = float(style_prior_divergence(prior, sparse, loss_type="js"))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, float(np.log(2.0)) + 1e-6)
        self.assertAlmostEqual(float(style_prior_divergence(prior, prior, loss_type="js")), 0.0, places=6)

    def test_sparse_eps_is_a_softer_patch_than_legacy(self) -> None:
        prior = make_prior()
        sparse = sparsify(prior)
        eps = float(style_prior_divergence(prior, sparse, loss_type="sparse_eps", eps=1e-3))
        legacy = float(style_prior_divergence(prior, sparse, loss_type="legacy"))
        self.assertLess(eps, legacy)
        with self.assertRaises(ValueError):
            style_prior_divergence(prior, sparse, loss_type="sparse_eps", eps=0.0)

    def test_legacy_matches_the_original_expression_bit_for_bit(self) -> None:
        prior = make_prior()
        sparse = sparsify(prior)
        expected = -(prior * torch.log(sparse.clamp_min(1e-8))).sum(dim=-1).mean()
        got = style_prior_divergence(prior, sparse, loss_type="legacy")
        self.assertTrue(torch.equal(expected, got))

    def test_all_modes_are_registered(self) -> None:
        self.assertEqual(set(STYLE_PRIOR_LOSS_TYPES), {"dense", "js", "sparse_eps", "legacy"})


class EntropyTermTests(unittest.TestCase):
    """Eq. 7: L_ent = H(pi) is positive and minimized by sharpening (the code was already right)."""

    def test_is_entropy_not_minus_entropy(self) -> None:
        peaky = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]])
        flat = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
        self.assertGreater(float(style_code_entropy(peaky)), 0.0)
        self.assertLess(float(style_code_entropy(peaky)), float(style_code_entropy(flat)))
        self.assertAlmostEqual(float(style_code_entropy(flat)), float(np.log(4.0)), places=6)

    def test_zeros_contribute_nothing(self) -> None:
        sparse = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
        self.assertAlmostEqual(float(style_code_entropy(sparse)), float(np.log(2.0)), places=6)


class PriorProvenanceTests(unittest.TestCase):
    def _payload(self, with_provenance: bool) -> dict:
        rng = np.random.default_rng(0)
        prior = rng.random((2, 3, 5)).astype(np.float32)
        prior /= prior.sum(axis=-1, keepdims=True)
        kwargs = {}
        if with_provenance:
            kwargs = {"vqvae_ckpt": "/tmp/vqvae_a/best.pt", "db": "/tmp/db_a"}
        return dict(
            prior=prior,
            global_prior=prior.mean(axis=0),
            slot_saliency=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            selected_slots=np.array([0, 2], dtype=np.int64),
            style_names=["a", "b"],
            counts=np.ones((2, 3, 5), dtype=np.int64),
            **kwargs,
        )

    def test_round_trip_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            save_style_code_prior(path, **self._payload(with_provenance=True))
            loaded = load_style_code_prior(path)
            self.assertEqual(loaded["vqvae_ckpt"], "/tmp/vqvae_a/best.pt")
            self.assertEqual(loaded["db"], "/tmp/db_a")
            self.assertEqual(loaded["num_codebooks"], 3)
            np.testing.assert_allclose(loaded["prior"], self._payload(True)["prior"])

    def test_old_files_without_provenance_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            payload = self._payload(with_provenance=False)
            payload.pop("counts")
            np.savez_compressed(path, **payload, split=np.asarray("train"), smoothing=np.asarray(1.0))
            loaded = load_style_code_prior(path)
            self.assertIsNone(loaded["vqvae_ckpt"])
            self.assertIsNone(loaded["db"])
            self.assertIsNone(loaded["counts"])

    def test_invalid_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.npz"
            payload = self._payload(with_provenance=True)
            payload["selected_slots"] = np.array([0, 7], dtype=np.int64)  # out of range
            save_style_code_prior(path, **payload)
            with self.assertRaises(ValueError):
                load_style_code_prior(path)


class ControllerPriorLossTests(unittest.TestCase):
    """End-to-end: the controller's L_prior drops from the clamp-dominated scale to the CE scale."""

    @staticmethod
    def _build(loss_type: str):
        from arch.basic_vqvae_375to291 import BasicVQVAE375to291
        from arch.flowmatch_vqvae_controller import build_flowmatch_controller_from_config

        torch.manual_seed(0)
        vqvae = BasicVQVAE375to291(
            input_dim=375,
            traj_dim=84,
            motion_dim=291,
            input_mode="traj_motion",
            encoder_arch="mlp",
            decoder_arch="mlp",
            hidden_dim=32,
            encoder_layers=3,
            decoder_hidden_dim=32,
            decoder_layers=3,
            num_latents=4,
            code_dim=4,
            num_embeddings=32,
        )
        controller = build_flowmatch_controller_from_config(
            {
                "use_style_code_selector": True,
                "style_slot_indices": [0, 1],
                "style_prior_loss": loss_type,
                "flow_hidden_dim": 32,
                "flow_layers": 1,
                "flow_heads": 2,
                "flow_ffn_dim": 32,
            },
            vqvae=vqvae,
            num_styles=2,
        )
        return controller

    @staticmethod
    def _diffuse_prior(controller) -> torch.Tensor:
        prior = torch.full_like(controller.style_prior_probs, 1.0 / controller.style_prior_probs.size(-1))
        prior = prior + 0.01 * torch.rand_like(prior)
        return prior / prior.sum(dim=-1, keepdim=True)

    def test_dense_scale_vs_legacy_scale(self) -> None:
        for loss_type in ("dense", "legacy"):
            controller = self._build(loss_type)
            prior = self._diffuse_prior(controller)
            controller.set_style_code_prior(prior)
            details = controller.style_selector(
                codebook_weight=controller.vqvae.quantizer.embedding_weight.detach(),
                style_label=torch.tensor([0, 1]),
            )
            hist, entropy, top1 = controller._compute_style_regularizers(
                style_label=torch.tensor([0, 1]),
                style_probs=details["style_probs"],
                style_logits=details["style_logits"] if details["style_logits_are_dense"] else None,
            )
            hist_value = float(hist.detach())
            self.assertTrue(0.0 <= entropy.item() <= float(np.log(details["style_probs"].size(-1))) + 1e-6)
            self.assertTrue(0.0 < top1.item() <= 1.0)
            if loss_type == "dense":
                self.assertLess(hist_value, 12.0, "dense prior loss must stay on the CE scale")
            else:
                self.assertGreater(hist_value, 15.0, "legacy prior loss is clamp-dominated")

    def test_invalid_loss_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._build("nope")

    def test_controller_training_step_runs_end_to_end(self) -> None:
        """Regression: the controller's condition-encoding path used the legacy quantizer API.

        `encode_conditions` (and therefore the whole controller training loop) used to call
        `quantizer.embedding.weight`, which does not exist on the vector_quantize_pytorch adapter
        -- every downstream controller run crashed. Exercise the full path here.
        """
        controller = self._build("dense")
        prior = self._diffuse_prior(controller)
        controller.set_style_code_prior(prior)
        latent_dim = controller.latent_dim
        controller.set_latent_prior_stats(
            torch.zeros(latent_dim),
            torch.ones(latent_dim),
            transport_std=torch.ones(latent_dim),
        )
        controller.train()
        batch = 3
        out = controller.compute_loss(
            x_curr=torch.randn(batch, 375),
            x_next=torch.randn(batch, 375),
            style_label=torch.tensor([0, 1, 0]),
            flow_weight=1.0,
            recon_weight=0.1,
            style_hist_weight=0.2,
            style_entropy_weight=1e-3,
        )
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(float(out["style_hist_loss"].detach()), 0.0)
        self.assertLess(float(out["style_hist_loss"].detach()), 12.0)


if __name__ == "__main__":
    unittest.main()


class StylePriorGateTests(unittest.TestCase):
    """Warn-only gates added for the style-prior training runs (never change training numerics)."""

    def _payload(self, prior, slots):
        return {"prior": prior, "selected_slots": np.asarray(slots, dtype=np.int64)}

    def test_entropy_floor_matches_the_analytic_value(self) -> None:
        from training.train_flowmatch_vqvae_controller import _style_prior_entropy_floor

        # uniform prior over 2048 codes -> H = log(2048); only the selected slots count
        uniform = np.full((3, 4, 2048), 1.0 / 2048.0, dtype=np.float32)
        floor = _style_prior_entropy_floor(self._payload(uniform, [0, 2]))
        self.assertAlmostEqual(floor, float(np.log(2048)), places=5)

        # one-hot prior -> H = 0
        onehot = np.zeros((2, 3, 64), dtype=np.float32)
        onehot[..., 0] = 1.0
        self.assertAlmostEqual(_style_prior_entropy_floor(self._payload(onehot, [1])), 0.0, places=6)

    def test_entropy_floor_uses_only_the_selected_slots(self) -> None:
        from training.train_flowmatch_vqvae_controller import _style_prior_entropy_floor

        peaky = np.zeros((1, 2, 32), dtype=np.float32)
        peaky[0, 0, :] = 1.0 / 32.0  # flat slot
        peaky[0, 1, 0] = 1.0        # peaky slot
        flat_only = _style_prior_entropy_floor(self._payload(peaky, [0]))
        peaky_only = _style_prior_entropy_floor(self._payload(peaky, [1]))
        self.assertAlmostEqual(flat_only, float(np.log(32)), places=5)
        self.assertAlmostEqual(peaky_only, 0.0, places=6)

    def test_entropy_floor_degrades_gracefully(self) -> None:
        from training.train_flowmatch_vqvae_controller import _style_prior_entropy_floor

        self.assertTrue(np.isnan(_style_prior_entropy_floor(None)))
        self.assertTrue(np.isnan(_style_prior_entropy_floor({})))
        self.assertTrue(np.isnan(_style_prior_entropy_floor({"prior": np.zeros((2, 3))})))
