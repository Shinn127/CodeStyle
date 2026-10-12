from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import torch

from arch import (
    BasicVQVAE375to291,
    LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD,
    CAMDMController,
    DiffusionLMMController,
    DiffusionVQVAEController,
    LMM375to291,
    MANNController,
    MVAE375to291,
    build_camdm_controller_from_config,
    build_diffusion_lmm_controller_from_config,
    build_diffusion_vqvae_controller_from_config,
    build_mann_controller_from_config,
)
from Utils.runtime import load_checkpoint, load_state_dict_with_allowed_missing, sanitize_state_dict_keys


LEGACY_QUANTIZER_KEY = "quantizer.embedding.weight"
LIBRARY_QUANTIZER_PREFIX = "quantizer.vq."


def _uses_legacy_quantizer(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Detect checkpoints trained before the codebook moved to the library VQ.

    Those checkpoints store the quantizer as `quantizer.embedding.weight` /
    `quantizer.ema_*`; the current module expects `quantizer.vq._codebook.*`.
    """
    return LEGACY_QUANTIZER_KEY in state_dict and not any(
        key.startswith(LIBRARY_QUANTIZER_PREFIX) for key in state_dict
    )


def _load_vqvae_state_dict(
    model: BasicVQVAE375to291,
    state_dict: Mapping[str, torch.Tensor],
    *,
    ckpt_path: Path,
    allow_legacy_quantizer: bool,
) -> None:
    if not _uses_legacy_quantizer(state_dict):
        model.load_state_dict(state_dict, strict=True)
        return
    ckpt_path = Path(ckpt_path)
    if not allow_legacy_quantizer:
        raise RuntimeError(
            f"{ckpt_path} stores its codebook with the pre-library quantizer keys "
            f"({LEGACY_QUANTIZER_KEY!r}, 'quantizer.ema_*'). The autoencoder evaluation path "
            "(`encode_pre_vq` -> `decode(z_e)`) never touches the quantizer, so such checkpoints "
            "can still be evaluated by dropping those keys: pass --allow-legacy-quantizer. "
            "The quantized path (codebook indices) is NOT available for them."
        )
    dropped = [key for key in state_dict if key.startswith("quantizer.")]
    model.legacy_quantizer_dropped = True
    remaining = {key: value for key, value in state_dict.items() if not key.startswith("quantizer.")}
    missing_quantizer = [key for key in model.state_dict() if key.startswith("quantizer.")]
    load_state_dict_with_allowed_missing(
        model,
        remaining,
        allowed_missing=missing_quantizer,
        allowed_unexpected=(),
    )
    warnings.warn(
        f"loaded {ckpt_path.name} with {len(dropped)} legacy quantizer tensors dropped; "
        "the codebook stays at its initialisation and only the autoencoder path is meaningful",
        RuntimeWarning,
        stacklevel=2,
    )


def build_vqvae_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    freeze: bool = True,
    allow_legacy_quantizer: bool = False,
) -> BasicVQVAE375to291:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model_type = str(ckpt_args.get("model_type", "input_mode_vqvae_375to291")).strip().lower()
    encoder_arch = str(ckpt_args.get("encoder_arch", ckpt_args.get("network_arch", "mlp")))
    decoder_arch = str(ckpt_args.get("decoder_arch", ckpt_args.get("network_arch", "mlp")))
    if model_type not in {
        "input_mode_vqvae_375to291",
        "inputmodevqvae375to291",
        "ablation_vqvae_375to291",
        "vqvae_input_mode_375to291",
    }:
        raise ValueError(
            f"Unsupported VQ model_type={model_type!r} in checkpoint: {ckpt_path}. "
            "This codebase now expects the input-mode VQ-VAE."
        )

    model = BasicVQVAE375to291(
        input_dim=int(ckpt_args.get("input_dim", 375)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        input_mode=str(ckpt_args.get("input_mode", "traj_motion")),
        encoder_arch=encoder_arch,
        decoder_arch=decoder_arch,
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
        encoder_layers=int(ckpt_args.get("encoder_layers", 2)),
        decoder_hidden_dim=int(ckpt_args.get("decoder_hidden_dim", 512)),
        decoder_layers=int(ckpt_args.get("decoder_layers", 2)),
        num_latents=int(ckpt_args.get("num_latents", 16)),
        code_dim=int(ckpt_args.get("code_dim", 8)),
        num_embeddings=int(ckpt_args.get("num_embeddings", 1024)),
        commitment_cost=float(ckpt_args.get("commitment_cost", 0.25)),
        use_rotation_trick=bool(ckpt_args.get("use_rotation_trick", True)),
        ema_decay=float(ckpt_args.get("ema_decay", 0.99)),
        ema_eps=float(ckpt_args.get("ema_eps", 1e-5)),
        kmeans_init=bool(ckpt_args.get("kmeans_init", True)),
        kmeans_iters=int(ckpt_args.get("kmeans_iters", 10)),
        dead_code_threshold=int(
            ckpt_args.get("dead_code_threshold", LIBRARY_DEFAULT_DEAD_CODE_THRESHOLD)
        ),
    )
    _load_vqvae_state_dict(
        model,
        sanitize_state_dict_keys(ckpt["model"]),
        ckpt_path=ckpt_path,
        allow_legacy_quantizer=bool(allow_legacy_quantizer),
    )
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def build_lmm_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
) -> LMM375to291:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model_type = str(ckpt_args.get("model_type", "lmm_375to291")).strip().lower()
    if model_type not in {
        "lmm",
        "lmm_375to291",
        "lmm375to291",
        "lmm_compressor_decompressor_ae",
        "lmmcompressordecompressorae",
    }:
        raise ValueError(
            f"Unsupported LMM model_type={model_type!r} in checkpoint: {ckpt_path}"
        )

    model = LMM375to291(
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        latent_dim=int(ckpt_args.get("latent_dim", 32)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_mvae_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
) -> MVAE375to291:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model = MVAE375to291(
        input_dim=int(ckpt_args.get("input_dim", 375)),
        traj_dim=int(ckpt_args.get("traj_dim", 84)),
        motion_dim=int(ckpt_args.get("motion_dim", 291)),
        latent_dim=int(ckpt_args.get("latent_dim", 32)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        gate_hidden_dim=int(ckpt_args.get("gate_hidden_dim", 64)),
        num_condition_frames=int(ckpt_args.get("num_condition_frames", 1)),
        num_future_predictions=int(ckpt_args.get("num_future_predictions", 1)),
        num_experts=int(ckpt_args.get("num_experts", 6)),
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_mann_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    num_styles: int | None = None,
) -> MANNController:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model_type = str(ckpt_args.get("model_type", "mann_375to291_controller")).strip().lower()
    if model_type not in {
        "mann",
        "mann_375to291",
        "mann_375to291_controller",
        "mann375to291controller",
    }:
        raise ValueError(
            f"Unsupported MANN model_type={model_type!r} in checkpoint: {ckpt_path}"
        )

    model = build_mann_controller_from_config(ckpt_args, num_styles=num_styles)
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_camdm_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    num_styles: int | None = None,
) -> CAMDMController:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}

    model_type = str(ckpt_args.get("model_type", "camdm_375to291_controller")).strip().lower()
    if model_type not in {
        "camdm",
        "camdm_375to291",
        "camdm_375to291_controller",
        "camdm375to291controller",
    }:
        raise ValueError(
            f"Unsupported CAMDM model_type={model_type!r} in checkpoint: {ckpt_path}"
        )

    model = build_camdm_controller_from_config(ckpt_args, num_styles=num_styles)
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_diffusion_vqvae_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    num_styles: int,
    vqvae_ckpt: Path | None = None,
) -> DiffusionVQVAEController:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}
    if vqvae_ckpt is None:
        raw = ckpt_args.get("vqvae_ckpt", None)
        if raw is None:
            raise ValueError(f"Checkpoint missing args['vqvae_ckpt']: {ckpt_path}")
        vqvae_ckpt = Path(raw)
        if not vqvae_ckpt.is_absolute():
            vqvae_ckpt = (ckpt_path.parent / vqvae_ckpt).resolve()
    else:
        vqvae_ckpt = Path(vqvae_ckpt).expanduser().resolve()
    vqvae = build_vqvae_from_checkpoint(vqvae_ckpt)
    model = build_diffusion_vqvae_controller_from_config(ckpt_args, vqvae=vqvae, num_styles=int(num_styles))
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


def build_diffusion_lmm_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    num_styles: int,
    lmm_ckpt: Path | None = None,
) -> DiffusionLMMController:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    ckpt_args = ckpt.get("args", {}) or {}
    if lmm_ckpt is None:
        raw = ckpt_args.get("lmm_ckpt", None)
        if raw is None:
            raise ValueError(f"Checkpoint missing args['lmm_ckpt']: {ckpt_path}")
        lmm_ckpt = Path(raw)
        if not lmm_ckpt.is_absolute():
            lmm_ckpt = (ckpt_path.parent / lmm_ckpt).resolve()
    else:
        lmm_ckpt = Path(lmm_ckpt).expanduser().resolve()
    lmm = build_lmm_from_checkpoint(lmm_ckpt)
    model = build_diffusion_lmm_controller_from_config(ckpt_args, lmm=lmm, num_styles=int(num_styles))
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    return model


__all__ = [
    "build_camdm_from_checkpoint",
    "build_diffusion_lmm_from_checkpoint",
    "build_diffusion_vqvae_from_checkpoint",
    "build_lmm_from_checkpoint",
    "build_mann_from_checkpoint",
    "build_mvae_from_checkpoint",
    "build_vqvae_from_checkpoint",
]
