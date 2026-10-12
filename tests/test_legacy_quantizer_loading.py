from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from arch.basic_vqvae_375to291 import BasicVQVAE375to291
from Utils.model_loading import build_vqvae_from_checkpoint


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


def legacy_state_dict(model: BasicVQVAE375to291) -> dict[str, torch.Tensor]:
    """Rewrite the library-VQ keys into the pre-library layout used by the 2026-era runs."""
    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.startswith("quantizer.vq._codebook."):
            continue
        if key.startswith("quantizer."):
            continue
        out[key] = value
    latent_dim = int(model.num_latents) * int(model.code_dim)
    num_embeddings = int(model.quantizer.num_embeddings)
    out["quantizer.embedding.weight"] = torch.randn(num_embeddings, latent_dim)
    out["quantizer.ema_cluster_size"] = torch.rand(num_embeddings)
    out["quantizer.ema_embed"] = torch.randn(num_embeddings, latent_dim)
    return out


def write_ckpt(path: Path, model: BasicVQVAE375to291, state: dict[str, torch.Tensor]) -> None:
    torch.save(
        {
            "epoch": 49,
            "model": state,
            "args": {
                "model_type": "input_mode_vqvae_375to291",
                "num_latents": int(model.num_latents),
                "code_dim": int(model.code_dim),
                "num_embeddings": int(model.quantizer.num_embeddings),
                "hidden_dim": 64,
                "encoder_layers": 3,
                "decoder_hidden_dim": 64,
                "decoder_layers": 3,
                "input_mode": "traj_motion",
                "encoder_arch": "mlp",
                "decoder_arch": "mlp",
            },
        },
        path,
    )


class LegacyQuantizerLoadingTests(unittest.TestCase):
    def test_legacy_keys_are_rejected_without_the_flag(self) -> None:
        model = build_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "legacy.pt"
            write_ckpt(ckpt, model, legacy_state_dict(model))
            with self.assertRaises(RuntimeError) as ctx:
                build_vqvae_from_checkpoint(ckpt)
            self.assertIn("--allow-legacy-quantizer", str(ctx.exception))

    def test_legacy_keys_load_with_the_flag_and_keep_the_autoencoder(self) -> None:
        model = build_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "legacy.pt"
            write_ckpt(ckpt, model, legacy_state_dict(model))
            with self.assertWarns(RuntimeWarning):
                loaded = build_vqvae_from_checkpoint(ckpt, allow_legacy_quantizer=True)
            reference = model.state_dict()
            for key, value in loaded.state_dict().items():
                if key.startswith("quantizer."):
                    continue
                self.assertTrue(torch.equal(value, reference[key]), msg=key)

    def test_legacy_path_still_rejects_non_quantizer_mismatches(self) -> None:
        model = build_model()
        state = legacy_state_dict(model)
        key = next(iter(state))
        state[key] = torch.zeros_like(state[key])
        state["decoder.net.0.weight"] = torch.zeros(3, 3)  # wrong shape
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "legacy_broken.pt"
            write_ckpt(ckpt, model, state)
            with self.assertRaises(RuntimeError):
                build_vqvae_from_checkpoint(ckpt, allow_legacy_quantizer=True)

    def test_modern_checkpoints_ignore_the_flag(self) -> None:
        model = build_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "modern.pt"
            write_ckpt(ckpt, model, model.state_dict())
            loaded = build_vqvae_from_checkpoint(ckpt, allow_legacy_quantizer=True)
            for key, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), msg=key)


if __name__ == "__main__":
    unittest.main()
