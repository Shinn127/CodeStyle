from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import linalg

from arch import (
    build_diffusion_lmm_controller_from_config,
    build_diffusion_vqvae_controller_from_config,
    build_flowmatch_controller_from_config,
    build_flowmatch_lmm_controller_from_config,
    resolve_flowmatch_prediction_mode,
)
from Utils import quat
from Utils.model_loading import (
    build_camdm_from_checkpoint,
    build_lmm_from_checkpoint,
    build_mann_from_checkpoint,
    build_mvae_from_checkpoint,
    build_vqvae_from_checkpoint,
)
from Utils.motion375 import SELECTED_24_NAMES, reconstruct_root_track_from_375, reconstruct_selected_globals_from_375
from Utils.runtime import load_checkpoint, load_state_dict_with_allowed_missing, sanitize_state_dict_keys
from Utils.reproducibility import state_fingerprint


MotionModelName = Literal[
    "basic_vqvae",
    "camdm",
    "diffusion_lmm_x0",
    "diffusion_vqvae_x0",
    "lmm",
    "mann",
    "mvae",
    "flowmatch_velocity",
    "flowmatch_target",
    "flowmatch_lmm_velocity",
    "flowmatch_lmm_target",
]


FLOWMATCH_VQ_COMPAT_MISSING_KEYS = (
    "style_slot_indices",
    "style_prior_probs",
    "style_prior_ready",
)


TOE_JOINT_NAMES = ("LeftToeBase", "RightToeBase")
TOE_JOINT_INDICES = tuple(
    idx for idx, name in enumerate(SELECTED_24_NAMES) if name in TOE_JOINT_NAMES
)


@dataclass(frozen=True)
class MotionEvalConfig:
    dt: float = 1.0 / 60.0
    foot_contact_height_cm: float = 20.0
    foot_contact_speed_cm_s: float = 20.0
    velocity_dir_min_speed_cm_s: float = 1.0
    metric_start_frame: int = 0
    metric_end_frame: int = 0
    fid_eps: float = 1e-6


@dataclass(frozen=True)
class SequenceMetricSums:
    mpjpe_sum_cm: float
    mpjpe_count: int
    root_ade_sum_cm: float
    root_ade_count: int
    heading_error_sum_deg: float
    heading_error_count: int
    velocity_dir_error_sum_deg: float
    velocity_dir_error_count: int
    foot_sliding_sum_cm: float
    foot_sliding_count: int
    jitter_sum_cm: float
    jitter_count: int
    fid_ref_frames: np.ndarray
    fid_pred_frames: np.ndarray
    foot_sliding_world_sum_cm: float = 0.0
    foot_sliding_world_count: int = 0
    foot_contact_possible_count: int = 0


def _resolve_saved_path(path_like: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _infer_num_styles_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    fallback: int,
) -> int:
    sanitized = sanitize_state_dict_keys(state_dict)
    candidate_keys = (
        "style_encoder.embedding.weight",
        "style_selector.style_embedding.weight",
        "embed_style.weight",
        "style_prior_probs",
    )
    for key in candidate_keys:
        value = sanitized.get(key, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        inferred = int(shape[0])
        if inferred > 0:
            return inferred
    return int(fallback)


def _build_flowmatch_from_checkpoint(
    *,
    ckpt_path: Path,
    num_styles: int,
    expected_prediction_mode: Literal["velocity", "target"] | None = None,
    vqvae_ckpt_override: Path | None = None,
) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    flow_args = ckpt.get("args", {}) or {}
    prediction_mode = resolve_flowmatch_prediction_mode(str(flow_args.get("prediction_mode", "velocity")))
    if expected_prediction_mode is not None and prediction_mode != expected_prediction_mode:
        raise ValueError(
            f"Flow checkpoint prediction_mode mismatch: expected {expected_prediction_mode!r}, "
            f"got {prediction_mode!r} from {ckpt_path}"
        )

    if vqvae_ckpt_override is not None:
        vqvae_ckpt = Path(vqvae_ckpt_override).expanduser().resolve()
    else:
        vqvae_ckpt_raw = flow_args.get("vqvae_ckpt", None)
        if vqvae_ckpt_raw is None:
            raise ValueError(
                f"Flow checkpoint {ckpt_path} is missing args['vqvae_ckpt']; "
                "pass --flow-vqvae-ckpt explicitly for evaluation."
            )
        vqvae_ckpt = _resolve_saved_path(vqvae_ckpt_raw, base_dir=ckpt_path.parent)
    if not vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ checkpoint not found for flow model: {vqvae_ckpt}")

    vqvae = build_vqvae_from_checkpoint(vqvae_ckpt)
    bundled_vq = {k[6:]: v for k, v in sanitize_state_dict_keys(ckpt["model"]).items() if k.startswith("vqvae.")}
    if bundled_vq and state_fingerprint(vqvae.state_dict()) != state_fingerprint(bundled_vq):
        raise ValueError("External VQ weights differ from the controller bundled VQ; latent manifold override is invalid")
    model_num_styles = int(flow_args.get("num_styles", 0))
    if model_num_styles <= 0:
        model_num_styles = _infer_num_styles_from_state_dict(ckpt["model"], fallback=int(num_styles))
    model = build_flowmatch_controller_from_config(
        flow_args,
        vqvae=vqvae,
        num_styles=model_num_styles,
    )
    load_state_dict_with_allowed_missing(
        model,
        sanitize_state_dict_keys(ckpt["model"]),
        allowed_missing=FLOWMATCH_VQ_COMPAT_MISSING_KEYS,
    )
    model.eval()
    model.eval_prediction_mode = prediction_mode
    model.eval_solver_steps = int(flow_args.get("solver_steps", 8))
    model.eval_noise_std = float(flow_args.get("noise_std", 1.0))
    model.eval_vqvae_ckpt = str(vqvae_ckpt)
    model.eval_fingerprints = flow_args.get("fingerprints", {})
    return model


def _build_flowmatch_lmm_from_checkpoint(
    *,
    ckpt_path: Path,
    num_styles: int,
    expected_prediction_mode: Literal["velocity", "target"] | None = None,
    lmm_ckpt_override: Path | None = None,
) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    flow_args = ckpt.get("args", {}) or {}
    prediction_mode = resolve_flowmatch_prediction_mode(str(flow_args.get("prediction_mode", "velocity")))
    if expected_prediction_mode is not None and prediction_mode != expected_prediction_mode:
        raise ValueError(
            f"Flow checkpoint prediction_mode mismatch: expected {expected_prediction_mode!r}, "
            f"got {prediction_mode!r} from {ckpt_path}"
        )

    if lmm_ckpt_override is not None:
        lmm_ckpt = Path(lmm_ckpt_override).expanduser().resolve()
    else:
        lmm_ckpt_raw = flow_args.get("lmm_ckpt", None)
        if lmm_ckpt_raw is None:
            raise ValueError(
                f"Flow checkpoint {ckpt_path} is missing args['lmm_ckpt']; "
                "pass --flow-lmm-ckpt explicitly for evaluation."
            )
        lmm_ckpt = _resolve_saved_path(lmm_ckpt_raw, base_dir=ckpt_path.parent)
    if not lmm_ckpt.exists():
        raise FileNotFoundError(f"LMM checkpoint not found for flow model: {lmm_ckpt}")

    lmm = build_lmm_from_checkpoint(lmm_ckpt)
    model_num_styles = int(flow_args.get("num_styles", 0))
    if model_num_styles <= 0:
        model_num_styles = _infer_num_styles_from_state_dict(ckpt["model"], fallback=int(num_styles))
    model = build_flowmatch_lmm_controller_from_config(
        flow_args,
        lmm=lmm,
        num_styles=model_num_styles,
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    model.eval_prediction_mode = prediction_mode
    model.eval_solver_steps = int(flow_args.get("solver_steps", 8))
    model.eval_noise_std = float(flow_args.get("noise_std", 1.0))
    model.eval_lmm_ckpt = str(lmm_ckpt)
    return model


def _build_diffusion_vqvae_from_checkpoint(
    *,
    ckpt_path: Path,
    num_styles: int,
    vqvae_ckpt_override: Path | None = None,
) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    diff_args = ckpt.get("args", {}) or {}

    if vqvae_ckpt_override is not None:
        vqvae_ckpt = Path(vqvae_ckpt_override).expanduser().resolve()
    else:
        vqvae_ckpt_raw = diff_args.get("vqvae_ckpt", None)
        if vqvae_ckpt_raw is None:
            raise ValueError(
                f"Diffusion checkpoint {ckpt_path} is missing args['vqvae_ckpt']; "
                "pass --flow-vqvae-ckpt explicitly for evaluation."
            )
        vqvae_ckpt = _resolve_saved_path(vqvae_ckpt_raw, base_dir=ckpt_path.parent)
    if not vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ checkpoint not found for diffusion model: {vqvae_ckpt}")

    vqvae = build_vqvae_from_checkpoint(vqvae_ckpt)
    model_num_styles = int(diff_args.get("num_styles", 0))
    if model_num_styles <= 0:
        model_num_styles = _infer_num_styles_from_state_dict(ckpt["model"], fallback=int(num_styles))
    model = build_diffusion_vqvae_controller_from_config(
        diff_args,
        vqvae=vqvae,
        num_styles=model_num_styles,
    )
    load_state_dict_with_allowed_missing(
        model,
        sanitize_state_dict_keys(ckpt["model"]),
        allowed_missing=FLOWMATCH_VQ_COMPAT_MISSING_KEYS,
    )
    model.eval()
    model.eval_solver_steps = int(diff_args.get("sample_steps", diff_args.get("diffusion_steps", 16)))
    model.eval_noise_std = float(diff_args.get("sample_noise_std", 1.0))
    model.eval_vqvae_ckpt = str(vqvae_ckpt)
    return model


def _build_diffusion_lmm_from_checkpoint(
    *,
    ckpt_path: Path,
    num_styles: int,
    lmm_ckpt_override: Path | None = None,
) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {ckpt_path}")
    diff_args = ckpt.get("args", {}) or {}

    if lmm_ckpt_override is not None:
        lmm_ckpt = Path(lmm_ckpt_override).expanduser().resolve()
    else:
        lmm_ckpt_raw = diff_args.get("lmm_ckpt", None)
        if lmm_ckpt_raw is None:
            raise ValueError(
                f"Diffusion checkpoint {ckpt_path} is missing args['lmm_ckpt']; "
                "pass --flow-lmm-ckpt explicitly for evaluation."
            )
        lmm_ckpt = _resolve_saved_path(lmm_ckpt_raw, base_dir=ckpt_path.parent)
    if not lmm_ckpt.exists():
        raise FileNotFoundError(f"LMM checkpoint not found for diffusion model: {lmm_ckpt}")

    lmm = build_lmm_from_checkpoint(lmm_ckpt)
    model_num_styles = int(diff_args.get("num_styles", 0))
    if model_num_styles <= 0:
        model_num_styles = _infer_num_styles_from_state_dict(ckpt["model"], fallback=int(num_styles))
    model = build_diffusion_lmm_controller_from_config(
        diff_args,
        lmm=lmm,
        num_styles=model_num_styles,
    )
    model.load_state_dict(sanitize_state_dict_keys(ckpt["model"]), strict=True)
    model.eval()
    model.eval_solver_steps = int(diff_args.get("sample_steps", diff_args.get("diffusion_steps", 16)))
    model.eval_noise_std = float(diff_args.get("sample_noise_std", 1.0))
    model.eval_lmm_ckpt = str(lmm_ckpt)
    return model


def load_motion_model(
    model_name: MotionModelName,
    ckpt_path: Path,
    *,
    num_styles: int | None = None,
    flow_vqvae_ckpt: Path | None = None,
    flow_lmm_ckpt: Path | None = None,
    allow_legacy_quantizer: bool = False,
) -> torch.nn.Module:
    if model_name == "basic_vqvae":
        return build_vqvae_from_checkpoint(
            ckpt_path,
            allow_legacy_quantizer=bool(allow_legacy_quantizer),
        )
    if model_name == "camdm":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for CAMDM evaluation")
        return build_camdm_from_checkpoint(ckpt_path, num_styles=int(num_styles))
    if model_name == "lmm":
        return build_lmm_from_checkpoint(ckpt_path)
    if model_name == "diffusion_vqvae_x0":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for diffusion evaluation")
        return _build_diffusion_vqvae_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            vqvae_ckpt_override=flow_vqvae_ckpt,
        )
    if model_name == "diffusion_lmm_x0":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for diffusion evaluation")
        return _build_diffusion_lmm_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            lmm_ckpt_override=flow_lmm_ckpt,
        )
    if model_name == "mann":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for MANN evaluation")
        return build_mann_from_checkpoint(ckpt_path, num_styles=int(num_styles))
    if model_name == "mvae":
        return build_mvae_from_checkpoint(ckpt_path)
    if model_name == "flowmatch_velocity":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for flowmatch evaluation")
        return _build_flowmatch_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            expected_prediction_mode="velocity",
            vqvae_ckpt_override=flow_vqvae_ckpt,
        )
    if model_name == "flowmatch_target":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for flowmatch evaluation")
        return _build_flowmatch_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            expected_prediction_mode="target",
            vqvae_ckpt_override=flow_vqvae_ckpt,
        )
    if model_name == "flowmatch_lmm_velocity":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for flowmatch evaluation")
        return _build_flowmatch_lmm_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            expected_prediction_mode="velocity",
            lmm_ckpt_override=flow_lmm_ckpt,
        )
    if model_name == "flowmatch_lmm_target":
        if num_styles is None or int(num_styles) <= 0:
            raise ValueError("num_styles must be provided for flowmatch evaluation")
        return _build_flowmatch_lmm_from_checkpoint(
            ckpt_path=ckpt_path,
            num_styles=int(num_styles),
            expected_prediction_mode="target",
            lmm_ckpt_override=flow_lmm_ckpt,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


@torch.no_grad()
def reconstruct_motion_batch(
    *,
    model_name: MotionModelName,
    model: torch.nn.Module,
    x: torch.Tensor,
    style_label: torch.Tensor | None = None,
    manual_style_probs: torch.Tensor | None = None,
    manual_style_strength: float = 1.0,
    flow_eval_mode: Literal["rollout", "reconstruct"] = "rollout",
    flow_solver: Literal["euler"] = "euler",
    flow_ode_steps: int | None = None,
    flow_noise_std: float | None = None,
    flow_requantize_z: bool | None = None,
    flow_latent_clip: bool | None = None,
    vq_eval_path: str = "pre_vq",
    noise_plan: torch.Tensor | None = None,
    legacy_inference_order: bool = False,
) -> torch.Tensor:
    """
    Reconstruct a normalized [B, T, 375] batch.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected x rank 2 or 3, got {tuple(x.shape)}")
    if x.size(-1) != 375:
        raise ValueError(f"Expected x last dim == 375, got {x.size(-1)}")

    if model_name == "basic_vqvae":
        if vq_eval_path not in ("pre_vq", "quantized"):
            raise ValueError("vq_eval_path must be pre_vq or quantized")
        if vq_eval_path == 'quantized' and getattr(model, 'legacy_quantizer_dropped', False):
            raise ValueError('Legacy quantizer was dropped; only --vq-eval-path pre_vq is available')
        z = model.encode_pre_vq(x) if vq_eval_path == "pre_vq" else model.encode(x)[0]
        motion_hat = model.decode(z, from_indices=False)
        if motion_hat.dim() == 2:
            motion_hat = motion_hat.unsqueeze(1)
        return torch.cat((x[..., :84], motion_hat), dim=-1)

    if model_name == "lmm":
        motion_hat = model(motion=x[..., 84:], traj=x[..., :84])
        if motion_hat.dim() == 2:
            motion_hat = motion_hat.unsqueeze(1)
        return torch.cat((x[..., :84], motion_hat), dim=-1)

    if model_name == "camdm":
        if style_label is None:
            raise ValueError("style_label is required for CAMDM evaluation")
        return _reconstruct_camdm_sequence(model, x, style_label)

    if model_name == "mann":
        if style_label is None:
            raise ValueError("style_label is required for MANN evaluation")
        return _reconstruct_mann_sequence(model, x, style_label)

    if model_name == "mvae":
        return _reconstruct_mvae_teacher_forced(model, x)

    if model_name in {"flowmatch_velocity", "flowmatch_target"}:
        if style_label is None and manual_style_probs is None:
            raise ValueError("style_label or manual_style_probs is required for flowmatch evaluation")
        return _reconstruct_flowmatch_sequence(
            model=model,
            model_name=model_name,
            x=x,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            eval_mode=flow_eval_mode,
            solver=flow_solver,
            ode_steps=flow_ode_steps,
            noise_std=flow_noise_std,
            requantize_z=flow_requantize_z,
            latent_clip=flow_latent_clip,
            noise_plan=noise_plan, legacy_inference_order=legacy_inference_order,
        )
    if model_name in {"flowmatch_lmm_velocity", "flowmatch_lmm_target"}:
        if style_label is None and manual_style_probs is None:
            raise ValueError("style_label or manual_style_probs is required for flowmatch evaluation")
        if manual_style_probs is not None:
            raise ValueError("manual_style_probs is only supported for VQ-based flowmatch models")
        return _reconstruct_flowmatch_sequence(
            model=model,
            model_name=model_name,
            x=x,
            style_label=style_label,
            manual_style_probs=None,
            manual_style_strength=manual_style_strength,
            eval_mode=flow_eval_mode,
            solver=flow_solver,
            ode_steps=flow_ode_steps,
            noise_std=flow_noise_std,
            requantize_z=flow_requantize_z,
            latent_clip=flow_latent_clip,
        )
    if model_name in {"diffusion_vqvae_x0", "diffusion_lmm_x0"}:
        if style_label is None:
            raise ValueError("style_label is required for diffusion evaluation")
        return _reconstruct_diffusion_sequence(
            model=model,
            model_name=model_name,
            x=x,
            style_label=style_label,
            eval_mode=flow_eval_mode,
            inference_steps=flow_ode_steps,
            noise_std=flow_noise_std,
            requantize_z=flow_requantize_z,
            latent_clip=flow_latent_clip,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def _reconstruct_camdm_sequence(
    model: torch.nn.Module,
    x: torch.Tensor,
    style_label: torch.Tensor,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"CAMDM reconstruction expects rank-3 input, got {tuple(x.shape)}")
    _, steps, dims = x.shape
    if dims != 375:
        raise ValueError(f"CAMDM reconstruction expects last dim 375, got {dims}")
    if steps <= 1:
        return x.clone()

    seq = x.clone()
    diffusion_steps = int(getattr(model, "diffusion_steps", 32))
    for step_idx in range(1, steps):
        x_curr = seq[:, step_idx - 1, :]
        traj_next = x[:, step_idx, :84]
        seq[:, step_idx, :] = model.predict_next_frame(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            num_steps=diffusion_steps,
        )
    return seq


def _reconstruct_mann_sequence(
    model: torch.nn.Module,
    x: torch.Tensor,
    style_label: torch.Tensor,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"MANN reconstruction expects rank-3 input, got {tuple(x.shape)}")
    batch, steps, dims = x.shape
    if dims != 375:
        raise ValueError(f"MANN reconstruction expects last dim 375, got {dims}")
    if steps <= 1:
        return x.clone()

    seq = x.clone()
    for step_idx in range(1, steps):
        x_curr = seq[:, step_idx - 1, :]
        traj_next = x[:, step_idx, :84]
        seq[:, step_idx, :] = model.predict_next_frame(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
        )
    return seq


def _predict_flowmatch_step(
    *,
    model_name: MotionModelName,
    model: torch.nn.Module,
    x_curr: torch.Tensor,
    traj_next: torch.Tensor,
    style_label: torch.Tensor,
    manual_style_probs: torch.Tensor | None,
    manual_style_strength: float,
    prev_latent: torch.Tensor | None,
    solver: Literal["euler"],
    ode_steps: int,
    noise_std: float,
    requantize_z: bool,
    latent_clip: bool,
    z0: torch.Tensor | None = None,
    legacy_inference_order: bool = False,
    style_feature: torch.Tensor | None = None,
    history_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name in {"flowmatch_velocity", "flowmatch_target"}:
        z_hat = model.sample_latent(
            x_curr=x_curr,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            prev_latent=prev_latent,
            num_steps=ode_steps,
            noise_std=noise_std,
            solver=solver, z0=z0, style_feature=style_feature,
            **({"history_latent": history_latent} if getattr(model, "history_conditioning", False) else {}),
        )
        motion_hat, z_hat = model.feedback_state(z_hat, requantize=requantize_z,
            latent_clip=latent_clip, legacy_order=legacy_inference_order)
        return torch.cat((traj_next, motion_hat), dim=-1), z_hat

    if model_name in {"flowmatch_lmm_velocity", "flowmatch_lmm_target"}:
        if requantize_z:
            raise ValueError("flow_requantize_z is only supported for VQ-based flowmatch models")
        if manual_style_probs is not None:
            raise ValueError("manual_style_probs is only supported for VQ-based flowmatch models")
        motion_hat, z_hat = model.predict_next_motion(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            prev_latent=prev_latent,
            num_steps=ode_steps,
            noise_std=noise_std,
            solver=solver,
        )
        if latent_clip:
            z_hat = model.clip_latent_to_data_bounds(z_hat)
            motion_hat = model.decode_motion(traj=traj_next, latent=z_hat)
        return torch.cat((traj_next, motion_hat), dim=-1), z_hat

    raise ValueError(f"Unsupported flowmatch model name: {model_name}")


def _reconstruct_flowmatch_sequence(
    *,
    model: torch.nn.Module,
    model_name: MotionModelName,
    x: torch.Tensor,
    style_label: torch.Tensor | None,
    manual_style_probs: torch.Tensor | None,
    manual_style_strength: float,
    eval_mode: Literal["rollout", "reconstruct"],
    solver: Literal["euler"],
    ode_steps: int | None,
    noise_std: float | None,
    requantize_z: bool,
    latent_clip: bool,
    noise_plan: torch.Tensor | None = None,
    legacy_inference_order: bool = False,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3 for flowmatch evaluation, got {tuple(x.shape)}")
    if x.size(-1) != 375:
        raise ValueError(f"Expected x last dim == 375, got {x.size(-1)}")
    if x.size(1) < 2:
        return x.clone()
    if eval_mode not in {"rollout", "reconstruct"}:
        raise ValueError(f"Unsupported flow_eval_mode: {eval_mode}")
    if solver != "euler":
        raise ValueError(f"Unsupported flow solver: {solver}")

    batch_size, seq_len, _ = x.shape
    if style_label is not None:
        style_label = style_label.to(device=x.device, dtype=torch.long)
        if style_label.shape not in ((batch_size,), (batch_size, seq_len)):
            raise ValueError("Style labels must be [B] or [B,T] for scheduled style changes")
    requantize_z = getattr(model, "inference_requantize", False) if requantize_z is None else requantize_z
    latent_clip = getattr(model, "inference_latent_clip", True) if latent_clip is None else latent_clip
    if noise_plan is not None:
        if noise_plan.shape != (batch_size, seq_len - 1, model.latent_dim):
            raise ValueError("noise_plan must contain raw Gaussian sources [B,T-1,latent_dim]")
        if not torch.isfinite(noise_plan).all():
            raise ValueError("noise_plan must be finite")
        noise_plan = noise_plan.to(device=x.device, dtype=x.dtype)
    if manual_style_probs is not None:
        manual_style_probs = manual_style_probs.to(device=x.device, dtype=x.dtype)
        if manual_style_probs.dim() == 2:
            pass
        elif manual_style_probs.dim() == 3:
            if manual_style_probs.size(0) not in (1, batch_size):
                raise ValueError(
                    f"Expected manual_style_probs batch size {batch_size}, got {manual_style_probs.size(0)}"
                )
        else:
            raise ValueError(
                "manual_style_probs must have shape [num_slots,num_embeddings] or "
                f"[B,num_slots,num_embeddings], got {tuple(manual_style_probs.shape)}"
            )
    resolved_ode_steps = int(getattr(model, "eval_solver_steps", 8) if ode_steps is None else ode_steps)
    # Fixed label/recipe within this call: style tokens depend on neither motion nor time.
    # Scheduled [B,T] labels and train-mode dropout retain their per-frame behavior.
    style_feature = None
    if (model_name in {'flowmatch_velocity', 'flowmatch_target'} and not model.training
            and (style_label is None or style_label.ndim == 1)):
        style_feature = model.encode_conditions(x_curr=x[:, 0], style_label=style_label,
            manual_style_probs=manual_style_probs, manual_style_strength=manual_style_strength)[2]
    resolved_noise_std = float(getattr(model, "eval_noise_std", 1.0) if noise_std is None else noise_std)
    if resolved_ode_steps < 1:
        raise ValueError(f"flow ode_steps must be >= 1, got {resolved_ode_steps}")
    if resolved_noise_std <= 0.0:
        raise ValueError(f"flow noise_std must be > 0, got {resolved_noise_std}")

    teacher_latents = None
    teacher_history = None
    if getattr(model, 'history_conditioning', False) and eval_mode == 'reconstruct':
        teacher_latents = model.frame_latent(x[:, :-1].reshape(-1, 375)).reshape(batch_size, seq_len-1, -1)
        teacher_history = torch.cat((teacher_latents[:, :1], teacher_latents[:, :-1]), 1)
        x = x.clone()
        x[:, :-1, 84:] = model.vqvae.decode(teacher_latents.flatten(0, 1)).reshape(batch_size, seq_len-1, 291)
    if eval_mode == "reconstruct":
        if (style_label is None and manual_style_probs is not None and style_feature is not None
                and noise_plan is not None and not legacy_inference_order):
            # TF frames are independent. Reuse each recipe feature without tiling its K-way
            # probabilities over time, and retain the exact per-case Gaussian sources.
            micro = int(getattr(model, 'eval_microbatch_size', 512))
            if micro < 1:
                raise ValueError('Flow evaluation microbatch size must be positive')
            current = x[:, :-1].reshape(-1, 375)
            features = style_feature[:, None].expand(-1, seq_len - 1, -1).reshape(len(current), -1)
            noises = noise_plan.reshape(-1, model.latent_dim)
            motions = []
            for begin in range(0, len(current), micro):
                rows = current[begin:begin + micro]
                latent = model.sample_latent(x_curr=rows, style_label=None,
                    prev_latent=model.frame_latent(rows) if teacher_latents is None else teacher_latents.reshape(-1, model.latent_dim)[begin:begin+micro], style_feature=features[begin:begin + micro],
                    **({"history_latent": teacher_history.reshape(-1, model.latent_dim)[begin:begin+micro]} if teacher_history is not None else {}),
                    z0=noises[begin:begin + micro], num_steps=resolved_ode_steps,
                    noise_std=resolved_noise_std, solver=solver)
                motion, _ = model.feedback_state(latent, requantize=requantize_z, latent_clip=latent_clip)
                motions.append(motion)
            out = x.clone()
            out[:, 1:, 84:] = torch.cat(motions).reshape(batch_size, seq_len - 1, 291)
            return out
        out = x.clone()
        for frame_idx in range(1, seq_len):
            x_curr = x[:, frame_idx - 1, :]
            traj_next = x[:, frame_idx, :84]
            x_next, _ = _predict_flowmatch_step(
                model_name=model_name,
                model=model,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_label[:, frame_idx] if style_label is not None and style_label.ndim == 2 else style_label,
                manual_style_probs=manual_style_probs,
                manual_style_strength=manual_style_strength,
                prev_latent=None if teacher_latents is None else teacher_latents[:, frame_idx-1],
                history_latent=None if teacher_history is None else teacher_history[:, frame_idx-1],
                solver=solver,
                ode_steps=resolved_ode_steps,
                noise_std=resolved_noise_std,
                requantize_z=requantize_z,
                latent_clip=latent_clip,
                z0=None if noise_plan is None else noise_plan[:, frame_idx - 1],
                legacy_inference_order=legacy_inference_order,
                style_feature=style_feature,
            )
            out[:, frame_idx, :] = x_next
        return out

    out = torch.empty_like(x)
    out[:, 0, :] = x[:, 0, :]
    x_curr = out[:, 0, :]
    prev_latent = model.frame_latent(x_curr)
    history_latent = prev_latent
    if getattr(model, "history_conditioning", False):
        out[:, 0, 84:] = model.vqvae.decode(prev_latent)
    for frame_idx in range(1, seq_len):
        traj_next = x[:, frame_idx, :84]
        x_next, z_next = _predict_flowmatch_step(
            model_name=model_name,
            model=model,
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label[:, frame_idx] if style_label is not None and style_label.ndim == 2 else style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            prev_latent=prev_latent,
            history_latent=history_latent,
            solver=solver,
            ode_steps=resolved_ode_steps,
            noise_std=resolved_noise_std,
            requantize_z=requantize_z,
            latent_clip=latent_clip,
            z0=None if noise_plan is None else noise_plan[:, frame_idx - 1],
            legacy_inference_order=legacy_inference_order,
            style_feature=style_feature,
        )
        out[:, frame_idx, :] = x_next
        x_curr = x_next
        history_latent = prev_latent
        prev_latent = z_next
    return out


def _predict_diffusion_step(
    *,
    model_name: MotionModelName,
    model: torch.nn.Module,
    x_curr: torch.Tensor,
    traj_next: torch.Tensor,
    style_label: torch.Tensor,
    prev_latent: torch.Tensor | None,
    inference_steps: int,
    noise_std: float,
    requantize_z: bool,
    latent_clip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name == "diffusion_vqvae_x0":
        motion_hat, z_hat = model.predict_next_motion(
            x_curr=x_curr,
            style_label=style_label,
            prev_latent=prev_latent,
            num_steps=inference_steps,
            noise_std=noise_std,
            latent_clip=latent_clip and not requantize_z,
        )
        if requantize_z:
            z_hat = model.requantize_latent(z_hat)
            motion_hat = model.vqvae.decode(z_hat, from_indices=False)
        return torch.cat((traj_next, motion_hat), dim=-1), z_hat

    if model_name == "diffusion_lmm_x0":
        if requantize_z:
            raise ValueError("flow_requantize_z is only supported for VQ-based latent models")
        motion_hat, z_hat = model.predict_next_motion(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            prev_latent=prev_latent,
            num_steps=inference_steps,
            noise_std=noise_std,
            latent_clip=latent_clip,
        )
        return torch.cat((traj_next, motion_hat), dim=-1), z_hat

    raise ValueError(f"Unsupported diffusion model name: {model_name}")


def _reconstruct_diffusion_sequence(
    *,
    model: torch.nn.Module,
    model_name: MotionModelName,
    x: torch.Tensor,
    style_label: torch.Tensor,
    eval_mode: Literal["rollout", "reconstruct"],
    inference_steps: int | None,
    noise_std: float | None,
    requantize_z: bool,
    latent_clip: bool,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3 for diffusion evaluation, got {tuple(x.shape)}")
    if x.size(-1) != 375:
        raise ValueError(f"Expected x last dim == 375, got {x.size(-1)}")
    if x.size(1) < 2:
        return x.clone()
    if eval_mode not in {"rollout", "reconstruct"}:
        raise ValueError(f"Unsupported flow_eval_mode: {eval_mode}")

    batch_size, seq_len, _ = x.shape
    style_label = style_label.to(device=x.device, dtype=torch.long).reshape(-1)
    if style_label.numel() != batch_size:
        raise ValueError(
            f"Expected style_label batch size {batch_size}, got {style_label.numel()}"
        )
    resolved_steps = int(getattr(model, "eval_solver_steps", 16) if inference_steps is None else inference_steps)
    resolved_noise_std = float(getattr(model, "eval_noise_std", 1.0) if noise_std is None else noise_std)
    if resolved_steps < 1:
        raise ValueError(f"inference_steps must be >= 1, got {resolved_steps}")
    if resolved_noise_std <= 0.0:
        raise ValueError(f"noise_std must be > 0, got {resolved_noise_std}")

    if eval_mode == "reconstruct":
        out = x.clone()
        for frame_idx in range(1, seq_len):
            x_curr = x[:, frame_idx - 1, :]
            traj_next = x[:, frame_idx, :84]
            x_next, _ = _predict_diffusion_step(
                model_name=model_name,
                model=model,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_label,
                prev_latent=None,
                inference_steps=resolved_steps,
                noise_std=resolved_noise_std,
                requantize_z=requantize_z,
                latent_clip=latent_clip,
            )
            out[:, frame_idx, :] = x_next
        return out

    out = torch.empty_like(x)
    out[:, 0, :] = x[:, 0, :]
    x_curr = out[:, 0, :]
    prev_latent = model.frame_latent(x_curr)
    for frame_idx in range(1, seq_len):
        traj_next = x[:, frame_idx, :84]
        x_next, z_next = _predict_diffusion_step(
            model_name=model_name,
            model=model,
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            prev_latent=prev_latent,
            inference_steps=resolved_steps,
            noise_std=resolved_noise_std,
            requantize_z=requantize_z,
            latent_clip=latent_clip,
        )
        out[:, frame_idx, :] = x_next
        x_curr = x_next
        prev_latent = z_next
    return out


def _reconstruct_mvae_teacher_forced(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected x rank 3, got {tuple(x.shape)}")
    if x.size(-1) != 375:
        raise ValueError(f"Expected x last dim == 375, got {x.size(-1)}")

    num_condition_frames = int(model.num_condition_frames)
    num_future_predictions = int(model.num_future_predictions)
    seq_len = int(x.size(1))
    if seq_len < (num_condition_frames + num_future_predictions):
        raise ValueError(
            "Sequence is too short for MVAE evaluation: "
            f"seq_len={seq_len}, condition={num_condition_frames}, future={num_future_predictions}"
        )

    out = x.clone()
    for begin in range(num_condition_frames, seq_len - num_future_predictions + 1, num_future_predictions):
        cond = x[:, begin - num_condition_frames : begin, :]
        future = x[:, begin : begin + num_future_predictions, :]
        future_traj = future[..., :84]
        mu, _ = model.encode(x_future=future, condition=cond)
        motion_hat = model.sample(z=mu, future_traj=future_traj, condition=cond)
        if motion_hat.dim() == 2:
            motion_hat = motion_hat.unsqueeze(1)
        out[:, begin : begin + num_future_predictions, 84:] = motion_hat

    return out


def _resolve_metric_slice(seq_len: int, cfg: MotionEvalConfig) -> slice:
    start = max(0, int(cfg.metric_start_frame))
    if cfg.metric_end_frame > 0:
        stop = min(seq_len, int(cfg.metric_end_frame))
    else:
        stop = seq_len
    if start >= stop:
        raise ValueError(f"Empty metric slice after clipping: start={start}, stop={stop}, seq_len={seq_len}")
    return slice(start, stop)


def _sequence_to_numpy(seq: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(seq, torch.Tensor):
        return seq.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(seq, dtype=np.float32)


def denormalize_motion_batch(
    x: np.ndarray | torch.Tensor,
    *,
    mean: np.ndarray | torch.Tensor | None,
    std: np.ndarray | torch.Tensor | None,
) -> np.ndarray:
    x_np = _sequence_to_numpy(x)
    if (mean is None) != (std is None):
        raise ValueError("mean and std must both be provided or both omitted")
    if mean is None:
        return x_np
    mean_np = _sequence_to_numpy(mean)
    std_np = _sequence_to_numpy(std)
    if mean_np.shape != (375,) or std_np.shape != (375,):
        raise ValueError(f"Expected mean/std shape (375,), got {mean_np.shape} and {std_np.shape}")
    return x_np * std_np + mean_np


def smooth_motion_block(
    sequence_375: torch.Tensor | np.ndarray,
    taps: int,
) -> torch.Tensor | np.ndarray:
    """Binomial temporal filter over the 291-dim motion block of a [..., T, 375] sequence.

    The decoder emits one code combination per frame, so the decoded motion carries
    per-frame code-switch noise that shows up as jitter and foot sliding. A short
    binomial kernel ([1,2,1]/4 for 3 taps, [1,4,6,4,1]/16 for 5) suppresses that band
    without touching the trajectory channels, which stay exactly as given.

    `taps=1` returns the input unchanged.
    """
    taps = int(taps)
    if taps <= 1:
        return sequence_375
    if taps % 2 == 0:
        raise ValueError(f"taps must be odd, got {taps}")
    kernel = np.array([math.comb(taps - 1, i) for i in range(taps)], dtype=np.float64)
    kernel /= kernel.sum()
    is_torch = isinstance(sequence_375, torch.Tensor)
    if is_torch:
        dtype = sequence_375.dtype
        seq = sequence_375.detach().to(dtype=torch.float64)
        motion = seq[..., 84:]
        steps, channels = motion.shape[-2], motion.shape[-1]
        flat = motion.reshape(-1, steps, channels)
        pad = taps // 2
        padded = torch.nn.functional.pad(flat.transpose(-2, -1), (pad, pad), mode="replicate")
        weight = torch.tensor(kernel, dtype=torch.float64, device=seq.device).view(1, 1, -1)
        weight = weight.expand(channels, 1, taps)
        smoothed = torch.nn.functional.conv1d(padded, weight, groups=channels)
        smoothed = smoothed.transpose(-2, -1).reshape(*motion.shape[:-2], steps, channels)
        return torch.cat((seq[..., :84], smoothed.to(dtype)), dim=-1).to(dtype)
    seq = np.asarray(sequence_375, dtype=np.float64)
    motion = seq[..., 84:]
    pad = taps // 2
    padded = np.pad(motion, [(0, 0)] * (motion.ndim - 2) + [(pad, pad), (0, 0)], mode="edge")
    smoothed = np.zeros_like(motion)
    for offset, weight in enumerate(kernel):
        smoothed += weight * padded[..., offset : offset + motion.shape[-2], :]
    out = np.concatenate((seq[..., :84], smoothed), axis=-1)
    return out.astype(np.asarray(sequence_375).dtype, copy=False)


def extract_motion_frames_for_fid(seq_375: np.ndarray, frame_slice: slice) -> np.ndarray:
    seq = np.asarray(seq_375[frame_slice], dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 375:
        raise ValueError(f"Expected sequence shape (T, 375), got {seq.shape}")
    motion = seq[:, 84:]
    return np.asarray(motion, dtype=np.float32)


def _toe_contact_mask_from_reference(
    reference_local_pos: np.ndarray,
    *,
    dt: float,
    contact_height_cm: float,
    contact_speed_cm_s: float,
) -> np.ndarray:
    if reference_local_pos.shape[0] < 2:
        return np.zeros((0, 0), dtype=bool)
    if len(TOE_JOINT_INDICES) == 0:
        return np.zeros((reference_local_pos.shape[0] - 1, 0), dtype=bool)

    toe_pos = reference_local_pos[:, TOE_JOINT_INDICES]
    toe_height_cm = toe_pos[..., 1] * 100.0
    toe_speed_cm_s = np.linalg.norm(np.diff(toe_pos, axis=0), axis=-1) * 100.0 / float(dt)
    return (toe_height_cm[:-1] <= float(contact_height_cm)) & (toe_speed_cm_s <= float(contact_speed_cm_s))


def _foot_sliding_cm(
    prediction_local_pos: np.ndarray,
    *,
    contact_mask: np.ndarray,
) -> tuple[float, int]:
    if prediction_local_pos.shape[0] < 2:
        return 0.0, 0
    if len(TOE_JOINT_INDICES) == 0:
        return 0.0, 0

    toe_pos = prediction_local_pos[:, TOE_JOINT_INDICES]
    toe_disp_cm = np.linalg.norm(np.diff(toe_pos, axis=0), axis=-1) * 100.0
    if contact_mask.shape != toe_disp_cm.shape:
        raise ValueError(f"contact_mask shape mismatch: expected {toe_disp_cm.shape}, got {contact_mask.shape}")
    if not np.any(contact_mask):
        return 0.0, 0
    values = toe_disp_cm[contact_mask]
    return float(values.sum()), int(values.size)


def _jitteriness_cm(selected_local_pos: np.ndarray) -> tuple[float, int]:
    if selected_local_pos.shape[0] < 3:
        return 0.0, 0
    diff2 = selected_local_pos[2:] - 2.0 * selected_local_pos[1:-1] + selected_local_pos[:-2]
    values = np.linalg.norm(diff2, axis=-1) * 100.0
    return float(values.sum()), int(values.size)


def _safe_normalize_2d(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-8, None)


def _angular_error_deg_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = _safe_normalize_2d(a)
    b_n = _safe_normalize_2d(b)
    cos_sim = np.sum(a_n * b_n, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sim)).astype(np.float32)


def _root_guidance_metrics(
    ref_eval: np.ndarray,
    pred_eval: np.ndarray,
    *,
    dt: float,
    min_speed_cm_s: float,
) -> tuple[float, int, float, int, float, int]:
    ref_root_pos, ref_root_rot = reconstruct_root_track_from_375(
        ref_eval,
        dt=dt,
        root_wy_in_degrees=True,
    )
    pred_root_pos, pred_root_rot = reconstruct_root_track_from_375(
        pred_eval,
        dt=dt,
        root_wy_in_degrees=True,
    )

    root_ade_cm = np.linalg.norm(pred_root_pos[:, [0, 2]] - ref_root_pos[:, [0, 2]], axis=-1) * 100.0

    forward_local = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    ref_heading = quat.mul_vec(ref_root_rot, np.repeat(forward_local, ref_root_rot.shape[0], axis=0))[:, [0, 2]]
    pred_heading = quat.mul_vec(pred_root_rot, np.repeat(forward_local, pred_root_rot.shape[0], axis=0))[:, [0, 2]]
    heading_error_deg = _angular_error_deg_2d(pred_heading, ref_heading)

    ref_vel = np.diff(ref_root_pos[:, [0, 2]], axis=0) / float(dt)
    pred_vel = np.diff(pred_root_pos[:, [0, 2]], axis=0) / float(dt)
    ref_speed_cm_s = np.linalg.norm(ref_vel, axis=-1) * 100.0
    pred_speed_cm_s = np.linalg.norm(pred_vel, axis=-1) * 100.0
    valid_vel = (ref_speed_cm_s >= float(min_speed_cm_s)) & (pred_speed_cm_s >= float(min_speed_cm_s))
    if np.any(valid_vel):
        velocity_dir_error_deg = _angular_error_deg_2d(pred_vel[valid_vel], ref_vel[valid_vel])
        vel_sum = float(velocity_dir_error_deg.sum())
        vel_count = int(velocity_dir_error_deg.size)
    else:
        vel_sum = 0.0
        vel_count = 0

    return (
        float(root_ade_cm.sum()),
        int(root_ade_cm.size),
        float(heading_error_deg.sum()),
        int(heading_error_deg.size),
        vel_sum,
        vel_count,
    )


def compute_sequence_metric_sums(
    *,
    reference_375: np.ndarray,
    prediction_375: np.ndarray,
    cfg: MotionEvalConfig,
    motion_mean: np.ndarray | torch.Tensor | None = None,
    motion_std: np.ndarray | torch.Tensor | None = None,
) -> SequenceMetricSums:
    ref = denormalize_motion_batch(reference_375, mean=motion_mean, std=motion_std)
    pred = denormalize_motion_batch(prediction_375, mean=motion_mean, std=motion_std)

    ref_slice = _resolve_metric_slice(ref.shape[0], cfg)
    pred_slice = _resolve_metric_slice(pred.shape[0], cfg)
    if (ref_slice.start, ref_slice.stop) != (pred_slice.start, pred_slice.stop):
        raise ValueError("Reference/prediction metric slices do not match")

    ref_eval = ref[ref_slice]
    pred_eval = pred[pred_slice]

    ref_global_pos, _ = reconstruct_selected_globals_from_375(
        ref_eval,
        dt=cfg.dt,
        root_wy_in_degrees=True,
        use_global=True,
    )
    pred_global_pos, _ = reconstruct_selected_globals_from_375(
        pred_eval,
        dt=cfg.dt,
        root_wy_in_degrees=True,
        use_global=True,
    )

    mpjpe_values = np.linalg.norm(pred_global_pos - ref_global_pos, axis=-1) * 100.0
    mpjpe_sum = float(mpjpe_values.sum())
    mpjpe_count = int(mpjpe_values.size)
    (
        root_ade_sum,
        root_ade_count,
        heading_error_sum,
        heading_error_count,
        velocity_dir_error_sum,
        velocity_dir_error_count,
    ) = _root_guidance_metrics(
        ref_eval,
        pred_eval,
        dt=cfg.dt,
        min_speed_cm_s=cfg.velocity_dir_min_speed_cm_s,
    )

    ref_local_pos, _ = reconstruct_selected_globals_from_375(
        ref_eval,
        dt=cfg.dt,
        root_wy_in_degrees=True,
        use_global=False,
    )
    pred_local_pos, _ = reconstruct_selected_globals_from_375(
        pred_eval,
        dt=cfg.dt,
        root_wy_in_degrees=True,
        use_global=False,
    )

    toe_contact_mask = _toe_contact_mask_from_reference(
        ref_local_pos,
        dt=cfg.dt,
        contact_height_cm=cfg.foot_contact_height_cm,
        contact_speed_cm_s=cfg.foot_contact_speed_cm_s,
    )
    foot_sum, foot_count = _foot_sliding_cm(
        pred_local_pos,
        contact_mask=toe_contact_mask,
    )
    world_contact = _toe_contact_mask_from_reference(ref_global_pos, dt=cfg.dt,
        contact_height_cm=cfg.foot_contact_height_cm, contact_speed_cm_s=cfg.foot_contact_speed_cm_s)
    world_sum, world_count = _foot_sliding_cm(pred_global_pos, contact_mask=world_contact)
    jitter_sum, jitter_count = _jitteriness_cm(pred_local_pos)

    ref_motion = extract_motion_frames_for_fid(ref, ref_slice)
    pred_motion = extract_motion_frames_for_fid(pred, pred_slice)

    return SequenceMetricSums(
        mpjpe_sum_cm=mpjpe_sum,
        mpjpe_count=mpjpe_count,
        root_ade_sum_cm=root_ade_sum,
        root_ade_count=root_ade_count,
        heading_error_sum_deg=heading_error_sum,
        heading_error_count=heading_error_count,
        velocity_dir_error_sum_deg=velocity_dir_error_sum,
        velocity_dir_error_count=velocity_dir_error_count,
        foot_sliding_sum_cm=foot_sum,
        foot_sliding_count=foot_count,
        jitter_sum_cm=jitter_sum,
        jitter_count=jitter_count,
        fid_ref_frames=ref_motion,
        fid_pred_frames=pred_motion,
        foot_sliding_world_sum_cm=world_sum, foot_sliding_world_count=world_count,
        foot_contact_possible_count=int(world_contact.size),
    )


def frechet_distance_from_stats(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    mu_fake: np.ndarray,
    sigma_fake: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance from precomputed means/covariances (used by the CAMDM-style grouping)."""
    mu1 = np.asarray(mu_real, dtype=np.float64).ravel()
    mu2 = np.asarray(mu_fake, dtype=np.float64).ravel()
    sigma1 = np.asarray(sigma_real, dtype=np.float64)
    sigma2 = np.asarray(sigma_fake, dtype=np.float64)
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.ndim == 0:
        sigma1 = np.asarray([[float(sigma1)]], dtype=np.float64)
    if sigma2.ndim == 0:
        sigma2 = np.asarray([[float(sigma2)]], dtype=np.float64)
    if sigma1.shape != sigma2.shape or sigma1.shape[0] != sigma1.shape[1]:
        raise ValueError(f"covariance shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    if sigma1.shape[0] != mu1.shape[0]:
        raise ValueError(f"covariance {sigma1.shape} does not match mean dim {mu1.shape[0]}")

    sigma1 = sigma1 + np.eye(sigma1.shape[0], dtype=np.float64) * float(eps)
    sigma2 = sigma2 + np.eye(sigma2.shape[0], dtype=np.float64) * float(eps)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.real(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)))


def frechet_distance(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError(f"FID features must be rank-2, got {real.shape} and {fake.shape}")
    if real.shape[1] != fake.shape[1]:
        raise ValueError(f"FID feature dim mismatch: {real.shape[1]} vs {fake.shape[1]}")
    if real.shape[0] < 2 or fake.shape[0] < 2:
        raise ValueError("FID requires at least 2 samples in both sets")

    return frechet_distance_from_stats(
        real.mean(axis=0),
        np.cov(real, rowvar=False),
        fake.mean(axis=0),
        np.cov(fake, rowvar=False),
        eps=eps,
    )


def camdm_style_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    group_labels: np.ndarray,
    *,
    eps: float = 1e-6,
    min_samples: int = 2,
) -> tuple[float, dict[int, float], dict[int, int]]:
    """CAMDM's per-style FID aggregation.

    CAMDM (`Evaluation/eval_motion_FID.py`) compares a generated motion against the ground-truth
    distribution *of its own style*, using features from non-overlapping 60-frame clips, and then
    divides the Fréchet distance by the feature dimension before averaging over files.

    Here the same statistic is computed per style group: the group's real and predicted features
    are pooled into one Gaussian each, the Fréchet distance between them is divided by the feature
    dimension, and the per-style values are averaged.

    Returns (mean_over_styles, {style: fid/dim}, {style: n_samples}).
    """
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    labels = np.asarray(group_labels)
    if real.shape != fake.shape:
        raise ValueError(f"feature shape mismatch: {real.shape} vs {fake.shape}")
    if labels.shape[0] != real.shape[0]:
        raise ValueError(f"group_labels length {labels.shape[0]} does not match {real.shape[0]} features")
    dim = real.shape[1]

    per_style: dict[int, float] = {}
    counts: dict[int, int] = {}
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < int(min_samples):
            continue
        per_style[int(label)] = frechet_distance(real[mask], fake[mask], eps=eps) / dim
        counts[int(label)] = int(mask.sum())
    if not per_style:
        return float("nan"), {}, {}
    return float(np.mean(list(per_style.values()))), per_style, counts


def camdm_clip_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    group_labels: np.ndarray,
    clip_ids: np.ndarray,
    *,
    eps: float = 1e-6,
    min_samples: int = 2,
) -> tuple[float, int]:
    """CAMDM's exp1 structure: one Gaussian per generated clip vs the Gaussian of its style.

    CAMDM computes the ground-truth Gaussian once per style (pooled over that style's files) and
    compares each generated file's Gaussian against it, dividing by the feature dimension.
    Here every evaluated clip takes the role of a generated file; clips with fewer than
    `min_samples` windows are skipped.

    Returns (mean_over_clips, num_clips_used).
    """
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
    labels = np.asarray(group_labels)
    clips = np.asarray(clip_ids)
    if not (real.shape == fake.shape and labels.shape[0] == real.shape[0] == clips.shape[0]):
        raise ValueError("features, group_labels and clip_ids must share their sample dimension")
    dim = real.shape[1]

    style_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < int(min_samples):
            continue
        style_stats[int(label)] = (real[mask].mean(axis=0), np.cov(real[mask], rowvar=False))

    values: list[float] = []
    for clip in np.unique(clips):
        mask = clips == clip
        if int(mask.sum()) < int(min_samples):
            continue
        label = int(labels[mask][0])
        stats = style_stats.get(label)
        if stats is None:
            continue
        fid = frechet_distance_from_stats(
            stats[0],
            stats[1],
            fake[mask].mean(axis=0),
            np.cov(fake[mask], rowvar=False),
            eps=eps,
        )
        values.append(fid / dim)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


__all__ = [
    "TOE_JOINT_INDICES",
    "TOE_JOINT_NAMES",
    "MotionEvalConfig",
    "MotionModelName",
    "SequenceMetricSums",
    "camdm_clip_fid",
    "camdm_style_fid",
    "compute_sequence_metric_sums",
    "denormalize_motion_batch",
    "extract_motion_frames_for_fid",
    "frechet_distance",
    "frechet_distance_from_stats",
    "load_motion_model",
    "reconstruct_motion_batch",
    "smooth_motion_block",
]
