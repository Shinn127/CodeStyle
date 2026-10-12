"""
Minimal training script for the flow-matching VQ controller.

Core pipeline:
  1) load normalized 375-dim frame windows from the memmap DB
  2) convert each window into adjacent frame pairs
  3) freeze the basic VQ-VAE and train a flow field in latent space
  4) optionally reuse precomputed VQ target cache for faster training
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import default_collate

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, desc: str = "", unit: str = "it", dynamic_ncols: bool = False) -> None:
            self.total = max(int(total), 0)
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.count = 0
            self.postfix = ""
            self._closed = False

        def __enter__(self):
            self._render()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def set_postfix(self, refresh: bool = True, **kwargs: object) -> None:
            self.postfix = " ".join(f"{key}={value}" for key, value in kwargs.items())
            if refresh:
                self._render()

        def update(self, n: int = 1) -> None:
            self.count = min(self.total, self.count + int(n)) if self.total else self.count + int(n)
            self._render()

        def close(self) -> None:
            if self._closed:
                return
            self._render(final=True)
            self._closed = True

        def _render(self, final: bool = False) -> None:
            width = 32
            ratio = (self.count / self.total) if self.total else 1.0
            filled = int(width * ratio)
            bar = "#" * filled + "." * (width - filled)
            suffix = f" {self.postfix}" if self.postfix else ""
            line = f"\r{self.desc} [{bar}] {self.count}/{self.total} {self.unit}{suffix}"
            sys.stderr.write(line)
            if final:
                sys.stderr.write("\n")
            sys.stderr.flush()

from arch import (
    FlowMatchVQVAEController,
    build_flowmatch_controller_from_config,
)
from dataset import build_train_val_test_dataloaders, Motion375Dataset
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.runtime import (
    MEMMAP_META_FILE,
    MEMMAP_X_FILE,
    configure_torch_runtime,
    load_checkpoint,
    load_state_dict_with_allowed_missing,
    load_norm_stats,
    resolve_device,
    sanitize_state_dict_keys,
    save_norm_stats_if_present,
    set_seed,
    require_unmirrored_db,
)

FLOWMATCH_VQ_COMPAT_MISSING_KEYS = (
    "style_slot_indices",
    "style_prior_probs",
    "style_prior_ready",
)
from Utils.style_code_prior import load_style_code_prior, TrainStyleRecipePool
from Utils.vq_target_cache import VQTargetCache
from Utils.reproducibility import (
    IMPLEMENTATION_VERSION, artifact_fingerprints, isolated_rng, rng_state,
    restore_rng_state, state_fingerprint, validate_fingerprints, file_sha256,
)


def _flatten_adjacent_pairs(
    x: torch.Tensor,
    style_label: torch.Tensor,
    start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or x.size(1) < 2:
        raise ValueError(f"Expected x shape [B,T,D] with T>=2, got {tuple(x.shape)}")
    batch, steps, dims = x.shape
    if style_label.numel() != batch or start.numel() != batch:
        raise ValueError("Batch metadata shape mismatch")

    x_curr = x[:, :-1].reshape(-1, dims)
    x_next = x[:, 1:].reshape(-1, dims)
    style_bt = style_label[:, None].expand(batch, steps - 1).reshape(-1)
    offsets = torch.arange(1, steps, device=start.device, dtype=start.dtype)
    curr_ids = (start[:, None] + offsets[None] - 1).reshape(-1)
    next_ids = (start[:, None] + offsets[None]).reshape(-1)
    return x_curr, x_next, style_bt, curr_ids, next_ids


def remap_token_state_dict(state_dict, *, drop_motion_token):
    """Adapt a controller state dict from the 6-token to the 5-token layout.

    Tokens are ordered [z_t, z_prev, motion, traj, style, time], so dropping the motion
    token removes `field.motion_proj.*` and position-embedding row 2. The reverse
    direction (5->6) has no meaningful mapping and raises.
    """
    if not drop_motion_token:
        raise ValueError('Only 6-token -> 5-token remapping is supported')
    remapped = {k: v for k, v in state_dict.items() if not k.startswith('field.motion_proj.')}
    key = 'field.position_embeddings'
    if key not in remapped:
        raise KeyError(f'Expected {key} in the initialization state dict')
    positions = remapped[key]
    if positions.shape[1] != 6:
        raise ValueError(f'Expected 6 position embeddings to drop one, got shape {tuple(positions.shape)}')
    keep = torch.tensor([0, 1, 3, 4, 5], device=positions.device)
    remapped[key] = positions.index_select(1, keep)
    return remapped


def _noised_gt_observation(model, z_gt, amplitude, *, requantize, latent_clip):
    """Plan-A observation-noise augmentation for the teacher-forced branch.

    Simulates "the previous observation is the model's own, slightly wrong output" by perturbing
    the raw GT latent with a per-sample amplitude drawn from U[0, amplitude] (units: per-dim
    transport std -- the same relative scale ControlOperators use for their input noise), then
    applying the exact inference feedback policy (clip -> quantize -> decode) so the motion token
    and the latent token stay mutually consistent and on the codebook manifold.
    """
    coeff = amplitude * torch.rand((z_gt.size(0), 1), device=z_gt.device, dtype=z_gt.dtype)
    eps = torch.randn_like(z_gt)
    shift = (model.latent_scale * model.latent_transport_std).to(dtype=z_gt.dtype)[None, :] * coeff * eps
    motion, latent = model.feedback_state(z_gt + shift, requantize=requantize, latent_clip=latent_clip)
    return motion, latent, coeff


def unrolled_flow_loss(
    *,
    model,
    x_window: torch.Tensor,
    style_label: torch.Tensor | None,
    prev_latent_0: torch.Tensor,
    target_latents: torch.Tensor,
    recon_targets: torch.Tensor,
    steps: int,
    observation_latents: torch.Tensor | None = None,
    flow_weight: float = 1.0,
    recon_weight: float = 0.0,
    root_recon_weight: float = 0.0,
    recon_inference_path: bool = False,
    recon_anchor_weight: float = 0.0,
    style_hist_weight: float = 0.0,
    style_entropy_weight: float = 0.0,
    noise_std: float = 1.0,
    solver_steps: int = 8,
    detach_feedback: bool = True,
    requantize_feedback: bool = True,
    latent_clip_feedback: bool = True,
    style_consistency_weight: float = 0.0,
    style_classifier=None,
    motion_mean=None,
    motion_std=None,
    joint_indices=None,
    classifier_label_map=None,
    manual_style_probs=None,
    clip_histograms=None,
    clip_histograms_validated=False,
    prior_labels=None,
    return_per_sample=False,
    flow_low_t_probability=0.,
    flow_low_t_max=.2,
    feedback_solver_steps: int | None = None,
    feedback_probability: float = 1.0,
    reuse_style_table: bool = True,
    observation_noise_std: float = 0.0,
    history_latent_0: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Mean per-step objectives plus one style regularizer per window.

    Reconstruction uses continuous predictions and GT or explicitly synthesized motion291.
    Feedback uses the shared inference policy and is detached by default. VQ stays frozen.

    `feedback_solver_steps` overrides the Euler steps used to build the rolled-out feedback
    latent on intermediate steps (the flow-matching loss itself is a single FM evaluation per
    step and is unaffected). The default None keeps it equal to `solver_steps`. Auxiliary motion
    losses (`recon_weight`, `root_recon_weight`, `style_consistency_weight`) are applied to the
    same intermediate prediction that feeds the next step, so what is supervised is exactly what
    the training rollout consumes; the final step has no feedback and always uses `solver_steps`.

    `feedback_probability` implements the TF -> rollout curriculum: a per-window gate (drawn once
    per window) decides whether the rolled history comes from the model's own feedback (p) or from
    the GT rows (1-p). p=0 is pure multi-step teacher forcing (no Euler/decode at all), p=1 is the
    plain rollout objective. GT-history windows observe `x_window[:, k+1]` and
    `observation_latents[:, k]`; feedback windows observe the model's decoded/requantized state.
    `observation_latents` defaults to `target_latents` and should be the *source* (pre-decoupling)
    latents when style decoupling replaced the targets.

    `observation_noise_std` enables the Plan-A observation-noise augmentation on the
    teacher-forced branch: at every rolled step k >= 1 the GT observation (latent *and* motion)
    is replaced by clip->quantize->decode(z_gt + eps), with a per-sample amplitude drawn from
    U[0, observation_noise_std] in units of the per-dim transport std. This teaches recovery
    from the model's own feedback errors during pure TF training, without running Euler
    integration. 0 disables it and draws nothing (the legacy RNG stream is preserved); it is
    inert when the GT branch is unreachable (steps == 1, or feedback_probability == 1).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    feedback_probability = float(feedback_probability)
    if not 0.0 <= feedback_probability <= 1.0:
        raise ValueError(f"feedback_probability must be in [0,1], got {feedback_probability}")
    observation_noise_std = float(observation_noise_std)
    if not np.isfinite(observation_noise_std) or observation_noise_std < 0.0:
        raise ValueError(f"observation_noise_std must be finite and >= 0, got {observation_noise_std}")
    # steps == 1 has no rolled history, so the gate is inert there (validation paths pass the
    # current curriculum value); the CLI rejects configuring a ramp for unroll_steps == 1.
    if feedback_solver_steps is None:
        feedback_solver_steps = solver_steps
    elif int(feedback_solver_steps) < 1:
        raise ValueError(f"feedback_solver_steps must be >= 1, got {feedback_solver_steps}")
    if x_window.dim() != 3 or x_window.size(1) < steps + 1:
        raise ValueError(f"x_window must be [B, T, 375] with T >= steps+1, got {tuple(x_window.shape)}")
    if target_latents.size(1) != steps or recon_targets.size(1) != steps:
        raise ValueError(
            f"target tensors must provide {steps} steps, got {target_latents.size(1)} and {recon_targets.size(1)}"
        )
    totals: dict[str, torch.Tensor] = {}
    window_metrics = {'style_hist_loss', 'style_entropy', 'selector_entropy', 'conditioning_entropy', 'style_top1_prob',
                      'style_prior_kl', 'style_prior_target_entropy', 'style_code_coverage'}
    regularizer = None
    x_curr = x_window[:, 0]
    prev_latent = prev_latent_0
    history_latent = history_latent_0
    source_latents = target_latents if observation_latents is None else observation_latents
    if source_latents.shape != target_latents.shape:
        raise ValueError(
            f"observation_latents must match target_latents shape {tuple(target_latents.shape)}, "
            f"got {tuple(source_latents.shape)}")
    # The style labels are fixed for the window, so the all-styles selector table (0.3ms) is
    # computed once here instead of once per unroll step. Bit-identical: the selector forward
    # is deterministic with dropout_prob=0, and the eligibility test below mirrors the one in
    # `compute_loss` (a non-eligible window falls back to per-step computation there).
    shared_style_table = None
    if (reuse_style_table and steps > 1 and getattr(model, 'training', False)
            and getattr(model, 'label_selector_dedup', False)
            and getattr(model, 'use_style_code_selector', False)
            and getattr(model, 'conditioning_mode', '') == 'label'
            and style_label is not None and manual_style_probs is None and clip_histograms is None
            and getattr(model, 'style_prior_loss', 'dense') == 'dense'
            and float(getattr(getattr(model, 'style_selector', None), 'dropout_prob', 0.0)) == 0.0):
        flat_labels = style_label.reshape(-1).long()
        if flat_labels.numel() >= 2 * int(model.num_styles):
            shared_style_table = model.selector_output_all_styles()
    use_feedback_path = steps > 1 and feedback_probability > 0.0
    # One gate per window: a window's rolled history is either entirely the model's own feedback
    # or entirely the GT rows of x_window. p=1 (default) draws nothing, so the plain rollout
    # objective keeps its exact RNG stream.
    feedback_gate = None
    if steps > 1 and 0.0 < feedback_probability < 1.0:
        feedback_gate = torch.rand((x_window.size(0), 1), device=x_window.device) < feedback_probability
    # Diagnostic accumulation for the observation-noise augmentation (device-side, no host sync).
    noise_coeff_sum = None
    noise_step_count = 0
    for step in range(steps):
        # The target motion is the GT (or explicitly synthesized) motion; never decode(target_latent).
        x_next_target = torch.cat((x_window[:, step + 1, :84], recon_targets[:, step]), dim=-1)
        # The FM loss is one field evaluation and does not depend on the Euler integration;
        # only the returned z_hat/motion_hat feed the next step. Intermediate rollout steps use
        # the cheaper feedback integration; the last step has no feedback and keeps the full
        # solver (which is also the only step whose *rendered motion* is supervised when
        # `recon_inference_path` is on -- the visual-loss dose at one step per window, not
        # `steps` times, because a 4x dose of visual loss measurably destabilised fine-tuning).
        step_solver_steps = feedback_solver_steps if (
            step + 1 < steps and use_feedback_path) else solver_steps
        # Intermediate GT-history steps in a pure teacher-forced epoch feed the next GT row, so
        # they do not need the integrated prediction at all (no Euler, no decode).
        need_prediction = (step + 1 < steps and use_feedback_path) or style_consistency_weight > 0
        # Visual-loss scope: the inference-path recon only applies to the final step
        # (full solver, no feedback). Intermediate steps keep their feedback prediction free of
        # the auxiliary loss; the style-consistency CE (which needs gradients) applies to every
        # step that produces a prediction.
        step_recon_weight = recon_weight if (step + 1 == steps or not recon_inference_path) else 0.0
        step_root_weight = root_recon_weight if (step + 1 == steps or not recon_inference_path) else 0.0
        step_recon_inference = recon_inference_path if step + 1 == steps else False
        if step_recon_inference and step_recon_weight > 0.0:
            # The displayed motion comes from a full-solver prediction at inference.
            step_solver_steps = solver_steps
        out = model.compute_loss(
            x_curr=x_curr,
            x_next=x_next_target,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            clip_histograms=clip_histograms,
            clip_histograms_validated=clip_histograms_validated,
            prior_labels=prior_labels,
            return_per_sample=return_per_sample,
            style_table=shared_style_table,
            prev_latent=prev_latent,
            target_latent=target_latents[:, step],
            flow_weight=flow_weight,
            recon_weight=step_recon_weight,
            root_recon_weight=step_root_weight,
            recon_inference_path=step_recon_inference,
            recon_anchor_weight=recon_anchor_weight,
            style_hist_weight=style_hist_weight if step == 0 else 0.0,
            style_entropy_weight=style_entropy_weight if step == 0 else 0.0,
            noise_std=noise_std,
            solver_steps=step_solver_steps,
            return_prediction=style_consistency_weight > 0,
            return_rollout_latent=need_prediction,
            history_latent=history_latent if getattr(model, "history_conditioning", False) else None,
            # The rollout feedback consumes the prediction detached; skip its autograd graph.
            # Steps whose prediction feeds an auxiliary loss (last step / consistency) keep it.
            prediction_no_grad=bool(
                step + 1 < steps and use_feedback_path and detach_feedback
                and style_consistency_weight == 0 and step_recon_weight == 0 and step_root_weight == 0),
            compute_style_metrics=step == 0,
            flow_low_t_probability=flow_low_t_probability,
            flow_low_t_max=flow_low_t_max,
        )
        if style_consistency_weight > 0:
            if style_classifier is None:
                raise ValueError("style_consistency_weight requires a frozen classifier")
            out["style_consistency_loss"] = style_consistency_loss(
                pred_motion=out["pred_motion"], style_label=style_label,
                classifier=style_classifier, motion_mean=motion_mean, motion_std=motion_std,
                joint_indices=joint_indices, label_map=classifier_label_map)
            out["loss"] = out["loss"] + style_consistency_weight * out["style_consistency_loss"]
        if step == 0:
            regularizer = style_hist_weight * out["style_hist_loss"] + style_entropy_weight * out["style_entropy"]
        # The window has one style condition; count its regularizers once, independently of k.
        value = out["loss"] - (regularizer if step == 0 else 0)
        totals["loss"] = value if "loss" not in totals else totals["loss"] + value
        for key, value in out.items():
            if key == '_flow_per_sample':
                totals[key] = value if key not in totals else totals[key] + value
                continue
            if key == "loss" or key.startswith("pred_") or value.ndim != 0:
                continue
            if key in window_metrics and step > 0:
                continue
            totals[key] = value if key not in totals else totals[key] + value
        if step + 1 < steps:
            traj_next = x_window[:, step + 1, :84]
            previous_latent = prev_latent
            if not use_feedback_path:
                # Pure multi-step teacher forcing (curriculum p=0): the observation is the next
                # GT row. Without the noise augmentation this path never integrates Euler or
                # decodes; with it, the observation is re-rendered from a noised latent (one
                # clip->quantize->decode per applied step) so nothing downstream drifts off the
                # inference feedback manifold.
                if observation_noise_std > 0.0:
                    with torch.no_grad():
                        motion_obs, latent_obs, coeff = _noised_gt_observation(
                            model, source_latents[:, step], observation_noise_std,
                            requantize=requantize_feedback, latent_clip=latent_clip_feedback)
                    x_curr = torch.cat((x_window[:, step + 1, :84], motion_obs), dim=-1)
                    prev_latent = latent_obs
                    noise_coeff_sum = coeff.mean() if noise_coeff_sum is None else noise_coeff_sum + coeff.mean()
                    noise_step_count += 1
                else:
                    x_curr = x_window[:, step + 1]
                    prev_latent = source_latents[:, step]
                history_latent = previous_latent.detach() if detach_feedback else previous_latent
                continue
            z_pred = out["pred_latent"]
            if detach_feedback:
                z_pred = z_pred.detach()
            feedback_motion, feedback_latent = model.feedback_state(
                z_pred, requantize=requantize_feedback, latent_clip=latent_clip_feedback)
            if detach_feedback:
                feedback_motion = feedback_motion.detach()
            if feedback_gate is None:
                x_curr = torch.cat((traj_next, feedback_motion), dim=-1)
                prev_latent = feedback_latent
            else:
                gt_motion = x_window[:, step + 1, 84:]
                gt_latent = source_latents[:, step]
                if observation_noise_std > 0.0:
                    with torch.no_grad():
                        gt_motion, gt_latent, coeff = _noised_gt_observation(
                            model, source_latents[:, step], observation_noise_std,
                            requantize=requantize_feedback, latent_clip=latent_clip_feedback)
                    noise_coeff_sum = coeff.mean() if noise_coeff_sum is None else noise_coeff_sum + coeff.mean()
                    noise_step_count += 1
                x_curr = torch.cat(
                    (traj_next, torch.where(feedback_gate, feedback_motion, gt_motion)), dim=-1)
                prev_latent = torch.where(feedback_gate, feedback_latent, gt_latent)
            history_latent = previous_latent.detach() if detach_feedback else previous_latent
    averaged = {key: value if key in window_metrics else value / float(steps) for key, value in totals.items()}
    averaged["loss"] = averaged["loss"] + regularizer
    averaged["observation_noise_coeff_mean"] = (
        averaged["loss"].new_zeros(()) if noise_step_count == 0
        else noise_coeff_sum / float(noise_step_count))
    if feedback_gate is None:
        averaged["feedback_fraction"] = averaged["loss"].new_tensor(
            1.0 if use_feedback_path else 0.0)
    else:
        averaged["feedback_fraction"] = feedback_gate.float().mean()
    return averaged


def _style_prior_entropy_floor(style_prior_payload) -> float:
    """E[H(pi_bar)] over the selected slots: the CE floor of the dense prior term (nats)."""
    try:
        if not style_prior_payload:
            return float("nan")
        prior = np.asarray(style_prior_payload["prior"], dtype=np.float64)
        if prior.ndim != 3:
            return float("nan")
        slots = np.asarray(style_prior_payload.get("selected_slots", []), dtype=np.int64).reshape(-1)
        if slots.size == 0:
            slots = np.arange(prior.shape[1])
        clipped = np.clip(prior, 1e-12, None)
        entropy = -(clipped * np.log(clipped)).sum(axis=-1)  # (styles, slots)
        return float(entropy[:, slots].mean())
    except Exception:
        return float("nan")


def style_consistency_loss(
    *,
    pred_motion: torch.Tensor,
    style_label: torch.Tensor,
    classifier,
    motion_mean: torch.Tensor,
    motion_std: torch.Tensor,
    joint_indices,
    window_size: int = 1,
    label_map=None,
) -> torch.Tensor:
    """CE between a frozen style classifier applied to the *decoded* predicted motion and the style.

    `pred_motion` is the decoder output in normalized space ([B, 291]); it is denormalized, the
    15 ybot joints' local positions are selected (the classifier's input joint set), and the frozen
    classifier must assign the requested style. This optimizes the metric that is reported
    downstream (style accuracy) rather than the latent-space flow loss.
    """
    if window_size != 1:
        raise ValueError(f"style consistency currently supports a frame-level surrogate (window=1), got {window_size}")
    if motion_mean.numel() == 375 and motion_std.numel() == 375:
        motion_mean, motion_std = motion_mean[84:], motion_std[84:]
    if motion_mean.numel() != 291 or motion_std.numel() != 291:
        raise ValueError("style consistency expects motion291 normalization (full stats[84:])")
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    motion_raw = pred_motion * motion_std + motion_mean
    local_pos = motion_raw[..., 3 : 3 + 24 * 3].reshape(-1, 24, 3)
    ybot = local_pos[:, joint_indices, :].reshape(-1, 45)
    logits = classifier(ybot.unsqueeze(-1))[1]          # [B, num_styles], frame-level surrogate
    labels = style_label if label_map is None else label_map.to(style_label.device)[style_label]
    return F.cross_entropy(logits, labels)


def _weighted_loss_breakdown(metrics: dict[str, float], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Reported values, with the dense prior's constant floor shown separately."""
    total = float(metrics.get("loss", 0.0))
    terms = {
        "flow": (float(metrics.get("flow_loss", 0.0)), 1.0),
        "recon": (float(metrics.get("recon_loss", 0.0)), float(getattr(args, "recon_weight", 0.0))),
        "root_recon": (float(metrics.get("root_recon_loss", 0.0)), float(getattr(args, "root_recon_weight", 0.0))),
        "style_prior": (
            float(metrics.get("style_hist_loss", 0.0)),
            float(getattr(args, "style_hist_weight", 0.0)),
        ),
        "style_consistency": (
            float(metrics.get("style_consistency_loss", 0.0)),
            float(getattr(args, "style_consistency_weight", 0.0) or 0.0),
        ),
        "style_entropy": (
            float(metrics.get("style_entropy", 0.0)),
            float(getattr(args, "style_entropy_weight", 0.0)),
        ),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (value, weight) in terms.items():
        weighted = value * weight
        out[name] = {"raw": value, "weight": weight, "weighted": weighted, "share": (weighted / total) if total else 0.0}
    if getattr(args, "style_prior_loss", "dense") == "dense":
        floor = float(metrics.get("style_prior_target_entropy", 0.0))
        out["style_prior"]["constant_floor"] = floor * float(getattr(args, "style_hist_weight", 0.0))
        out["style_prior"]["optimizable_kl"] = float(metrics.get("style_prior_kl", 0.0)) * float(getattr(args, "style_hist_weight", 0.0))
    return out


def _metric_keys(prediction_mode: str) -> list[str]:
    base = [
        "loss",
        "flow_loss",
        "recon_loss",
        "recon_anchor_loss",
        "recon_quant_error",
        "style_hist_loss",
        "style_entropy",
        "selector_entropy",
        "conditioning_entropy",
        "observation_noise_coeff_mean",
        "recipe_fraction",
        "recipe_exemplars_mean",
        "recipe_shortfall_fraction",
        "recipe_bank_hit_fraction",
        "style_top1_prob",
        "style_consistency_loss",
        "root_recon_loss",
        "style_prior_kl",
        "style_prior_target_entropy",
        "style_code_coverage",
        "feedback_fraction",
        "flow_time_mean",
        "latent_target_norm",
        "velocity_pred_norm",
        "velocity_target_norm",
    ]
    if prediction_mode == "target":
        base.extend(["target_pred_norm", "target_error", "target_weight_mean"])
    return base


def _save_json(path: Path, payload: dict) -> None:
    serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in payload.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_latent_stats(
    stats_path: Path,
    *,
    latent_dim: int,
) -> dict[str, np.ndarray | str]:
    data = np.load(stats_path, allow_pickle=True)
    required = {"offset", "scale", "transport_std", "data_min", "data_max"}
    if not required.issubset(data.files):
        raise ValueError(
            f"Latent stats file {stats_path} must contain {sorted(required)} for CO-style flow matching"
        )
    offset = np.asarray(data["offset"], dtype=np.float32).reshape(-1)
    scale = np.asarray(data["scale"], dtype=np.float32).reshape(-1)
    transport_std = np.asarray(data["transport_std"], dtype=np.float32).reshape(-1)
    data_min = np.asarray(data["data_min"], dtype=np.float32).reshape(-1)
    data_max = np.asarray(data["data_max"], dtype=np.float32).reshape(-1)
    if (
        offset.shape != (latent_dim,)
        or scale.shape != (latent_dim,)
        or transport_std.shape != (latent_dim,)
        or data_min.shape != (latent_dim,)
        or data_max.shape != (latent_dim,)
    ):
        raise ValueError(f"Invalid latent stats shape in {stats_path}")
    return {
        "mode": "co",
        "offset": offset,
        "scale": scale,
        "transport_std": transport_std,
        "data_min": data_min,
        "data_max": data_max,
        "fingerprints": json.loads(str(data["fingerprints"].item())) if "fingerprints" in data else None,
        "split": str(data["split"].item()) if "split" in data else "legacy_all",
    }


def _save_latent_stats(stats_path: Path, stats: dict[str, np.ndarray | str]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        mode=np.asarray(str(stats["mode"])),
        offset=np.asarray(stats["offset"], dtype=np.float32),
        scale=np.asarray(stats["scale"], dtype=np.float32),
        transport_std=np.asarray(stats["transport_std"], dtype=np.float32),
        data_min=np.asarray(stats["data_min"], dtype=np.float32),
        data_max=np.asarray(stats["data_max"], dtype=np.float32),
        fingerprints=np.asarray(json.dumps(stats.get("fingerprints"))),
        split=np.asarray("train"),
    )


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


@torch.no_grad()
def _estimate_raw_latent_stats(
    *,
    db_path: Path,
    vqvae,
    device: torch.device,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    vq_target_cache: VQTargetCache | None,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    x_source = None
    if vq_target_cache is not None:
        total_frames = int(vq_target_cache.meta.num_frames)
    else:
        with (db_path / MEMMAP_META_FILE).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        total_frames = int(meta["num_frames"])
        x_source = np.memmap(db_path / MEMMAP_X_FILE, dtype=np.float32, mode="r", shape=(total_frames, 375))

    current_chunk_rows = int(chunk_rows)
    segments = json.loads((db_path / "segments.json").read_text())
    train_ranges = [(int(s["frame_start"]), int(s["num_frames"])) for s in segments if s.get("split") == "train"]
    if not train_ranges:
        raise ValueError("Latent statistics require explicit non-empty train segments")
    train_ids = np.concatenate([np.arange(start, start + size, dtype=np.int64) for start, size in train_ranges])
    while True:
        try:
            sum_latent = torch.zeros(vqvae.latent_dim, dtype=torch.float64, device=device)
            sumsq_latent = torch.zeros(vqvae.latent_dim, dtype=torch.float64, device=device)
            min_latent = torch.full((vqvae.latent_dim,), float("inf"), dtype=torch.float64, device=device)
            max_latent = torch.full((vqvae.latent_dim,), -float("inf"), dtype=torch.float64, device=device)
            if vq_target_cache is not None:
                for start in range(0, len(train_ids), current_chunk_rows):
                    frame_ids = train_ids[start:start + current_chunk_rows]
                    latents = vq_target_cache.lookup_latents(frame_ids, vqvae=vqvae, device=device).to(torch.float64)
                    sum_latent += latents.sum(dim=0)
                    sumsq_latent += latents.square().sum(dim=0)
                    min_latent = torch.minimum(min_latent, latents.amin(dim=0))
                    max_latent = torch.maximum(max_latent, latents.amax(dim=0))
            else:
                if x_source is None:
                    raise RuntimeError("x_source was not initialized for latent stats estimation")
                for start in range(0, len(train_ids), current_chunk_rows):
                    x_chunk = np.asarray(x_source[train_ids[start:start + current_chunk_rows]], dtype=np.float32)
                    x_chunk = (x_chunk - norm_mean[None]) / norm_std[None]
                    x_t = torch.from_numpy(x_chunk).to(device=device)
                    latents, _, _ = vqvae.encode(x_t)
                    latents = latents.to(torch.float64)
                    sum_latent += latents.sum(dim=0)
                    sumsq_latent += latents.square().sum(dim=0)
                    min_latent = torch.minimum(min_latent, latents.amin(dim=0))
                    max_latent = torch.maximum(max_latent, latents.amax(dim=0))
            count = float(len(train_ids))
            mean = sum_latent / count
            var = (sumsq_latent / count) - mean.square()
            std = torch.sqrt(var.clamp_min(1e-6))
            return (
                mean.cpu().numpy().astype(np.float32),
                std.cpu().numpy().astype(np.float32),
                min_latent.cpu().numpy().astype(np.float32),
                max_latent.cpu().numpy().astype(np.float32),
            )
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc) or current_chunk_rows <= 1:
                raise
            next_chunk_rows = max(current_chunk_rows // 2, 1)
            print(
                f"[Warn] CUDA OOM while estimating latent stats with chunk_rows={current_chunk_rows}; "
                f"retrying with {next_chunk_rows}"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_chunk_rows = next_chunk_rows


def _build_latent_stats_payload(
    *,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    raw_min: np.ndarray,
    raw_max: np.ndarray,
) -> dict[str, np.ndarray | str]:
    offset = np.asarray(raw_mean, dtype=np.float32).reshape(-1)
    raw_std = np.asarray(raw_std, dtype=np.float32).reshape(-1).clip(min=1e-6)
    raw_min = np.asarray(raw_min, dtype=np.float32).reshape(-1)
    raw_max = np.asarray(raw_max, dtype=np.float32).reshape(-1)
    global_scale = float(np.clip(raw_std.mean(), a_min=1e-6, a_max=None))
    scale = np.full_like(raw_std, global_scale, dtype=np.float32)
    transport_std = (raw_std / scale).astype(np.float32, copy=False)
    data_min = ((raw_min - offset) / scale).astype(np.float32, copy=False)
    data_max = ((raw_max - offset) / scale).astype(np.float32, copy=False)
    return {
        "mode": "co",
        "offset": offset,
        "scale": scale.astype(np.float32, copy=False),
        "transport_std": transport_std.astype(np.float32, copy=False),
        "data_min": data_min,
        "data_max": data_max,
    }


def _seed_loader(loader, *, seed, epoch):
    """Separate shuffle/worker streams from model RNG and from persistent-worker lifetime."""
    loader.generator = torch.Generator().manual_seed(int(seed) + 2 * int(epoch))
    if hasattr(loader.sampler, "generator"):
        loader.sampler.generator = torch.Generator().manual_seed(int(seed) + 2 * int(epoch) + 1)



def feedback_probability_for_epoch(epoch, *, start_epoch, ramp_epochs, probability=1.0):
    """One-based TF -> rollout curriculum ramp for the unroll feedback share.

    Epochs before `start_epoch` train with pure teacher-forced (GT) history (p=0); from
    `start_epoch` the share of windows whose rolled history comes from the model's own feedback
    climbs linearly to `probability` over `ramp_epochs` epochs (p = probability * k / ramp_epochs
    at the k-th ramp epoch), then stays there. `ramp_epochs=0` jumps straight to `probability` at
    `start_epoch`, which reproduces the plain (ungated) rollout objective.
    """
    probability = float(probability)
    start_epoch = int(start_epoch)
    ramp_epochs = int(ramp_epochs)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"feedback probability must be in [0,1], got {probability}")
    if start_epoch < 1 or ramp_epochs < 0:
        raise ValueError("feedback ramp needs start_epoch >= 1 and ramp_epochs >= 0")
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs == 0:
        return probability
    progress = min(1.0, (int(epoch) - start_epoch + 1) / float(ramp_epochs))
    return probability * progress


def controller_collate(samples):
    if isinstance(samples, dict):
        return samples
    # traj/motion are views of x in the dataset, but default_collate copies both again.
    return default_collate([{key: sample[key] for key in ('x', 'label', 'start', 'segment_id')
                             if key in sample} for sample in samples])


def configure_controller_loader(loader, cache=None, *, worker_batches=True):
    loader.collate_fn = controller_collate
    if worker_batches and isinstance(loader.dataset, Motion375Dataset):
        loader.dataset.enable_controller_batches(
            cache if isinstance(cache, VQTargetCache) else None,
            pin_batches=bool(getattr(loader, 'pin_memory', False)),
        )


def _run_epoch(
    *, model, loader, device, prediction_mode, recon_weight, recon_inference_path=False,
    recon_anchor_weight=0.0, style_hist_weight,
    style_entropy_weight, noise_std, solver_steps, optimizer,
    vq_target_cache, cache_vqvae, grad_clip, log_interval, epoch, total_epochs,
    split, unroll_steps=1,
    style_consistency_weight=0.0, style_classifier=None, motion_mean=None,
    motion_std=None, joint_indices=None, style_consistency_window=1,
    classifier_label_map=None, root_recon_weight=0.0, microbatch_size=0, max_batches=0,
    recipe_pool=None, recipe_probability=0.,
    train_feedback_context=False,
    flow_low_t_probability=0., flow_low_t_max=.2,
    feedback_solver_steps=None,
    feedback_probability=1.0,
    reuse_style_table=True,
    feedback_batch_schedule="window",
    observation_noise_std=0.0,
):
    is_train = optimizer is not None
    pure_recipe = getattr(model, 'conditioning_mode', 'label') in ('recipe', 'clip_recipe')
    learned_recipe = getattr(model, 'conditioning_mode', 'label') == 'clip_recipe'
    model.train(is_train)
    keys = _metric_keys(prediction_mode)
    # The TF -> rollout share is computed by the caller per epoch and passed through; validation
    # runs at steps==1 where the gate is inert regardless of this value.
    effective_feedback_probability = float(feedback_probability)
    if feedback_batch_schedule not in ("window", "batch"):
        raise ValueError("Unknown feedback batch schedule")
    # One 0-dim accumulator per metric lets a microbatch update every metric with a single
    # fused `_foreach_add_` instead of a stack (+mul/add) subgraph per step.
    accum = [torch.zeros((), device=device, dtype=torch.float32) for _ in keys]
    num_samples = 0
    group_sums = torch.zeros(4, device=device)
    # Recipe conditioning needs the source segment metadata (train-only exemplar isolation).
    dataset = getattr(loader, "dataset", None)
    while dataset is not None and not hasattr(dataset, "segments") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    with tqdm(total=min(len(loader), max_batches) if max_batches else len(loader),
              desc=f"{split} {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True) as progress:
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True).long()
            labels_cpu = batch['label'].long() if batch['label'].device.type == 'cpu' else None
            # Cache indices are CPU memmap rows: keep frame IDs on CPU throughout lookup.
            starts = batch["start"].long()
            if x.ndim != 3 or x.size(1) < 2:
                raise ValueError("Training expects [B,T,375], T>=2")
            steps = int(unroll_steps) if is_train else 1
            if x.size(1) < steps + 1:
                raise ValueError("seq_len must be >= unroll_steps+1")
            segments = batch.get("segment_id")
            history_before = None
            teacher_motion = batch.get('teacher_motion')
            teacher_observations = None
            if getattr(model, 'history_conditioning', False):
                if x.size(1) < steps + 2:
                    raise ValueError('History conditioning requires seq_len >= unroll_steps+2')
                if vq_target_cache is None:
                    all_latents = model.frame_latent(x.flatten(0, 1)).reshape(x.size(0), x.size(1), -1)
                elif 'vq_indices' in batch:
                    all_latents = cache_vqvae.quantizer.lookup(batch['vq_indices'].to(device, non_blocking=True)).reshape(x.size(0), x.size(1), -1)
                else:
                    all_ids = starts[:, None] + torch.arange(x.size(1), device=starts.device)
                    all_latents = vq_target_cache.lookup_latents(all_ids.flatten(), vqvae=cache_vqvae, device=device).reshape(x.size(0), x.size(1), -1)
                if steps == 1:
                    repeats = x.size(1) - 2
                    window = torch.stack((x[:, 1:-1], x[:, 2:]), 2).reshape(-1, 2, 375)
                    history_before = all_latents[:, :-2].reshape(-1, model.latent_dim)
                    latents = torch.stack((all_latents[:, 1:-1], all_latents[:, 2:]), 2).reshape(-1, 2, model.latent_dim)
                    starts = (starts[:, None] + 1 + torch.arange(repeats, device=starts.device)).reshape(-1)
                    labels = labels.repeat_interleave(repeats)
                    if labels_cpu is not None: labels_cpu = labels_cpu.repeat_interleave(repeats)
                    if segments is not None: segments = segments.repeat_interleave(repeats)
                    if teacher_motion is not None:teacher_observations=teacher_motion[:,1:-1].reshape(-1,1,291)
                else:
                    window = x[:, 1:steps+2]
                    latents = all_latents[:, 1:steps+2]
                    history_before = all_latents[:, 0]
                    starts = starts + 1
                    if teacher_motion is not None:teacher_observations=teacher_motion[:,1:steps+1]
            elif steps == 1:
                repeats = x.size(1) - 1
                window = x if repeats == 1 else torch.stack((x[:, :-1], x[:, 1:]), dim=2).reshape(-1, 2, 375)
                starts = (starts[:, None] + torch.arange(repeats, device=starts.device)).reshape(-1)
                labels = labels.repeat_interleave(repeats)
                if labels_cpu is not None:
                    labels_cpu = labels_cpu.repeat_interleave(repeats)
                if segments is not None:
                    segments = segments.repeat_interleave(repeats)
                if teacher_motion is not None:teacher_observations=teacher_motion[:,:-1].reshape(-1,1,291)
            else:
                window = x[:, :steps + 1]
                if teacher_motion is not None:teacher_observations=teacher_motion[:,:steps]
            ids = starts[:, None] + torch.arange(steps + 1, device=starts.device)
            if getattr(model, "history_conditioning", False):
                pass  # Latents and prefix history came from the contiguous source batch above.
            elif vq_target_cache is None:
                latents = model.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
            elif 'vq_indices' in batch:
                indices = batch['vq_indices']
                if steps == 1 and repeats > 1:
                    indices = torch.stack((indices[:, :-1], indices[:, 1:]), 2).reshape(-1, 2, indices.size(-1))
                else:
                    indices = indices[:, :steps + 1]
                latents = cache_vqvae.quantizer.lookup(indices.to(device, non_blocking=True)).reshape(window.size(0), steps + 1, -1)
            else:
                latents = vq_target_cache.lookup_latents(ids.flatten(), vqvae=cache_vqvae,
                    device=device).reshape(window.size(0), steps + 1, -1)
            targets = latents[:, 1:]
            recon = window[:, 1:, 84:]
            original_labels = labels.clone()
            count = window.size(0)
            if train_feedback_context and (is_train or getattr(model,'history_conditioning',False)):
                # Frozen-VQ reconstructed GT conditions approximate the rendered feedback
                # distribution; supervision remains GT throughout.
                with torch.no_grad():
                    observed_motion = (teacher_observations if teacher_observations is not None else
                        model.vqvae.decode(latents[:, :-1].flatten(0, 1)).reshape(latents.size(0), steps, 291))
                window = window.clone()
                window[:, :steps, 84:] = observed_motion
            manual_probs, recipe_fraction = None, 0.
            if pure_recipe or (is_train and recipe_probability > 0):
                if recipe_pool is None:
                    raise ValueError('Recipe conditioning requires an independent train-only exemplar pool')
                if dataset is None or segments is None:
                    raise ValueError('Recipe conditioning requires source segment metadata for exemplar isolation')
                source_ids = [dataset.segments[int(i)].get('source_entry_id', int(i)) for i in segments.cpu().tolist()]
                if pure_recipe:
                    exclusions = [{source_ids[i]} for i in range(count)]
                    manual_probs, recipe_fraction = recipe_pool.sample(labels, probability=1.,
                        per_sample_exclusions=exclusions, require_all=True,
                        **({'labels_cpu': labels_cpu} if isinstance(recipe_pool, TrainStyleRecipePool) else {}))
                else:
                    manual_probs, recipe_fraction = recipe_pool.sample(labels, probability=recipe_probability,
                        excluded_source_ids=set(source_ids))
            batch_feedback = effective_feedback_probability
            if is_train and feedback_batch_schedule == 'batch':
                batch_feedback = float(math.floor((batch_idx+1)*effective_feedback_probability+1e-9)
                                       > math.floor(batch_idx*effective_feedback_probability+1e-9))
            micro = min(microbatch_size or count, count)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            for begin in range(0, count, micro):
                end = min(begin + micro, count)
                with torch.set_grad_enabled(is_train):
                    out = unrolled_flow_loss(
                        model=model, x_window=window[begin:end], style_label=None if pure_recipe else labels[begin:end],
                        prev_latent_0=latents[begin:end, 0], target_latents=targets[begin:end],
                        history_latent_0=None if history_before is None else history_before[begin:end],
                        recon_targets=recon[begin:end], steps=steps,
                        # Source (pre-decoupling) GT latents drive the GT-history branch of the
                        # TF->rollout curriculum; with no decoupling they equal the targets.
                        observation_latents=latents[begin:end, 1: 1 + steps],
                        feedback_probability=batch_feedback if is_train else 1.0,
                        recon_weight=recon_weight, recon_inference_path=recon_inference_path,
                        recon_anchor_weight=recon_anchor_weight,
                        root_recon_weight=root_recon_weight, style_hist_weight=style_hist_weight,
                        style_entropy_weight=style_entropy_weight, noise_std=noise_std,
                        solver_steps=solver_steps, requantize_feedback=model.inference_requantize,
                        latent_clip_feedback=model.inference_latent_clip,
                        feedback_solver_steps=feedback_solver_steps if is_train else solver_steps,
                        reuse_style_table=reuse_style_table,
                        observation_noise_std=observation_noise_std if is_train else 0.0,
                        style_consistency_weight=style_consistency_weight if is_train else 0.,
                        style_classifier=style_classifier, motion_mean=motion_mean, motion_std=motion_std,
                        joint_indices=joint_indices, classifier_label_map=classifier_label_map,
                        manual_style_probs=None if manual_probs is None or learned_recipe else manual_probs[begin:end],
                        clip_histograms=manual_probs[begin:end] if learned_recipe else None,
                        clip_histograms_validated=learned_recipe,
                        prior_labels=labels[begin:end] if learned_recipe else None,
                        return_per_sample=True,
                        flow_low_t_probability=flow_low_t_probability if is_train else 0.,
                        flow_low_t_max=flow_low_t_max)
                    if is_train:
                        (out["loss"] * ((end - begin) / count)).backward()
                cross = labels[begin:end] != original_labels[begin:end]
                per_sample = out.pop('_flow_per_sample').detach()
                # Bucket every sample once (0 clean / 1 cross) and reduce with a single
                # weighted bincount.
                bucket = cross.long()
                sums2 = torch.bincount(bucket, weights=per_sample, minlength=2)
                counts2 = torch.bincount(bucket, minlength=2).to(per_sample.dtype)
                group_sums += torch.cat((sums2, counts2))
                out['recipe_fraction'] = out['loss'].new_tensor(recipe_fraction)
                stats = getattr(recipe_pool, 'last_sample_stats', {}) if manual_probs is not None else {}
                out['recipe_exemplars_mean'] = out['loss'].new_tensor(stats.get('exemplars_mean', 0.))
                out['recipe_shortfall_fraction'] = out['loss'].new_tensor(stats.get('shortfall_fraction', 0.))
                out['recipe_bank_hit_fraction'] = out['loss'].new_tensor(stats.get('bank_hit_fraction', 0.))
                # Single fused add for every accumulable metric (weight = microbatch rows,
                # matching the previous `values += stack * (end - begin)` scaling).
                torch._foreach_add_(accum, [out[key].detach() for key in keys],
                                    alpha=float(end - begin))
            if is_train:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    grad_clip if grad_clip > 0 else float("inf"), error_if_nonfinite=True,
                    foreach=True)
                optimizer.step()
            num_samples += count
            progress.update(1)
            if (batch_idx + 1) % max(log_interval, 1) == 0:
                progress.set_postfix(loss=f"{float(accum[0] / max(num_samples, 1)):.4f}", unroll=steps)
    metrics = {key: float(value) / max(num_samples, 1) for key, value in zip(keys, accum)}
    if num_samples == 0:
        raise ValueError(f'{split} has no batches; check batch size and split/window lengths')
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f'Non-finite {split} metrics: {metrics}')
    metrics["num_samples"] = float(num_samples)
    # Accumulated as [matched_sum, cross_sum, matched_count, cross_count].
    matched_sum, cross_sum, matched_n, cross_n = group_sums.cpu().tolist()
    metrics.update(matched_flow_loss=matched_sum / max(matched_n, 1), cross_flow_loss=cross_sum / max(cross_n, 1),
                   matched_samples=matched_n, cross_samples=cross_n)
    return metrics


def _save_checkpoint(
    *,
    path: Path,
    model: FlowMatchVQVAEController,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "args": vars(args),
            "implementation_version": IMPLEMENTATION_VERSION,
            "training_objective_version": getattr(args, 'training_objective_version', IMPLEMENTATION_VERSION),
            "rng_state": rng_state(),
            "selection": getattr(args, "best_selection", None),
        },
        temporary,
    )
    temporary.replace(path)


def _move_optimizer_state_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Minimal flow-matching VQ controller trainer")
    parser.add_argument("--db", type=Path, default=script_dir / "database_100sty_375_memmap_nomirror")
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror" / "best.pt",
    )
    parser.add_argument("--vq-target-cache", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--latent-stats", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=script_dir / "checkpoints" / "flowmatch_tf50_fp32_norecon")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument('--worker-batches', action=argparse.BooleanOptionalAction, default=True,
                        help='Assemble tensor batches and frozen VQ cache indices in DataLoader workers')
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument('--fp32-math-attention', action=argparse.BooleanOptionalAction, default=True,
                        help='Use math SDPA for the six-token FP32 training field; eval keeps the original path')
    parser.add_argument('--style-val-batch-size', type=int, default=64)
    parser.add_argument('--stop-after-epoch', type=int, default=0,
                        help='Stop at a saved epoch boundary; scheduler/total epoch budget remain unchanged')
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--allow-legacy-artifacts", action="store_true")
    parser.add_argument("--rollout-val-every", type=int, default=0)
    parser.add_argument('--tf-val-every', type=int, default=0, help='Fixed 240f style validation interval; historical flag name, see --style-val-mode')
    parser.add_argument('--style-val-mode', choices=['rollout','reconstruct'], default='rollout',
                        help='Motion/style validation and checkpoint selection; reconstruct is an explicit TF diagnostic')
    parser.add_argument('--tf-val-classifier', type=Path, default=script_dir / 'checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth')
    parser.add_argument('--tf-val-root-tolerance', type=float, default=1.15)
    parser.add_argument('--motion-quality-selection', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--quality-max-jitter', type=float, default=.5)
    parser.add_argument('--quality-max-local-fs', type=float, default=.8)
    parser.add_argument('--quality-max-world-fs', type=float, default=1.)
    parser.add_argument('--tf-val-selection', choices=['absolute', 'relative'], default='absolute')
    parser.add_argument('--tf-val-min-matched-accuracy', type=float, default=.85)
    parser.add_argument('--tf-val-max-matched-ade', type=float, default=5.)
    parser.add_argument('--tf-val-max-cross-ade', type=float, default=10.)
    parser.add_argument('--tf-val-reference', type=Path, default=None, help='Reuse the initial baseline metrics and adjacent TF case manifest across stages')
    parser.add_argument('--tf-val-clips', type=int, default=90)
    parser.add_argument('--tf-val-windows', type=int, default=3)
    parser.add_argument("--rollout-val-classifier", type=Path, default=script_dir / "checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth")

    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument('--lr-schedule-epochs', type=int, default=0,
                        help='Cosine horizon; 0 uses --epochs. Set 50 for a bounded short prefix of a 50-epoch schedule.')
    parser.add_argument('--train-feedback-context', action=argparse.BooleanOptionalAction, default=False,
                        help='Optional frozen-VQ reconstruction of GT previous motion during training; GT targets stay unchanged')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument('--fused-adamw', action=argparse.BooleanOptionalAction, default=True,
                        help='Fuse CUDA AdamW kernels in FP32; CPU uses ordinary AdamW')
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--matmul-precision", type=str, choices=["highest", "high", "medium"], default="highest")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-data-cache-gb", type=float, default=0., help="Immutable FP32 train windows/code indices; 0 keeps the CPU loader")
    parser.add_argument("--cache-teacher-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-into-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-same-label", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-styles", type=int, default=0)
    parser.add_argument("--use-style-code-selector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--style-code-prior", type=Path, default=None)
    parser.add_argument("--style-num-slots", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--style-selector-hidden-dim", type=int, default=128)
    parser.add_argument("--style-selector-topk", type=int, default=4)
    parser.add_argument("--style-selector-temperature", type=float, default=1.0)
    parser.add_argument('--style-proto-normalization', choices=['none', 'layernorm'], default='none')
    parser.add_argument('--style-selector-init', choices=['prior', 'random'], default='prior')
    parser.add_argument('--shared-style-table', action=argparse.BooleanOptionalAction, default=True,
        help=("Unroll training: compute the all-styles selector table once per window and reuse it "
              "for every rolled step (deterministic, bit-identical). Disable for legacy replay."))
    parser.add_argument('--label-selector-dedup', action=argparse.BooleanOptionalAction, default=True,
        help=("Label mode: run the learned selector once on the full style table per step and "
              "gather rows per sample, instead of materializing [B, M, K] twice. Mathematically "
              "equivalent (FP32 reduction order aside); disable for bitwise legacy replay."))
    parser.add_argument('--use-motion-token', action=argparse.BooleanOptionalAction, default=True,
        help=("Feed the 291-dim previous motion as its own condition token (default, 6 tokens). "
              "--no-use-motion-token drops that token (5 tokens): at inference the fed motion is "
              "decoder(z_prev), so the quantized-latent token already carries it. Init from a "
              "6-token checkpoint drops motion_proj and its position-embedding row."))
    parser.add_argument('--field-arch', choices=['transformer', 'skip_cat_mlp'], default='skip_cat_mlp',
        help=("Velocity-field backbone. 'skip_cat_mlp' (default for new runs) is the "
              "concatenative skip MLP (ControlOperators networks.SkipCatMLP) over [z_t, z_prev, "
              "motion?, traj, style, time]; no attention. 'transformer' keeps the original "
              "6-token self-attention field (legacy checkpoints resolve to it from their saved "
              "args). Cross-architecture --init-ckpt transfer is rejected."))
    parser.add_argument('--recipe-prob', type=float, default=0., help='Optional recipe-input augmentation; 0 preserves the paper training path')
    parser.add_argument('--conditioning-mode', choices=['label', 'recipe', 'clip_recipe'], default='label',
                        help='recipe uses pure clip-derived conditions in both train/validation; disables unused selector losses')
    parser.add_argument('--recipe-frames', type=int, default=240)
    parser.add_argument('--recipe-exemplars', type=int, default=5)
    parser.add_argument('--recipe-exemplar-choices', type=int, nargs='+', default=None,
                        help='Sample an exemplar count uniformly per recipe, e.g. 1 3 5; no silent count reduction in pure mode')
    parser.add_argument('--recipe-bank-size', type=int, default=0,
                        help='Deterministic recipe variants per style/exclusion/count; sparse in-memory cache, 0 samples fresh crops')
    parser.add_argument('--recipe-bank-max-entries', type=int, default=131072)
    parser.add_argument("--time-embed-dim", type=int, default=64)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-layers", type=int, default=2)
    parser.add_argument("--flow-heads", type=int, default=4)
    parser.add_argument("--flow-ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prediction-mode", type=str, choices=["velocity", "target"], default="target")
    parser.add_argument("--flow-loss-space", choices=["normalized", "transport"], default="normalized")
    parser.add_argument("--inference-requantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-latent-clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-time-eps", type=float, default=1e-3)
    parser.add_argument('--flow-low-t-probability', type=float, default=0.,
                        help='Mixture fraction sampled from Uniform[0,flow-low-t-max]; remaining times cover [0,1]')
    parser.add_argument('--flow-low-t-max', type=float, default=.2)

    parser.add_argument("--recon-weight", type=float, default=0.0,
                        help="Motion reconstruction loss; 0 skips Euler integration and decoding when no other motion loss or feedback is needed")
    parser.add_argument("--recon-inference-path", action=argparse.BooleanOptionalAction, default=False,
                        help=("With --recon-weight > 0, supervise decode(requantize(clip(z_hat))) -- exactly "
                              "what inference displays (straight-through gradient on the predicted latent) -- "
                              "instead of the continuous decode. Default off preserves the legacy continuous-decode loss."))
    parser.add_argument("--recon-anchor-weight", type=float, default=0.0,
                        help=("With --recon-inference-path, weight of the commitment anchor "
                              "MSE(z_clipped, sg(z_quantized)). Keeps the predicted latent near its code; "
                              "recommended ~0.1-0.25 to stop the straight-through drift seen without it."))
    parser.add_argument("--root-recon-weight", type=float, default=0.0)
    parser.add_argument(
        "--style-hist-weight",
        type=float,
        default=0.02,
        help="Weight of selector prior CE. Its target entropy floor is logged separately from KL.",
    )
    parser.add_argument(
        "--style-entropy-weight",
        type=float,
        default=0.003,
        help="Weight of the selector entropy H(pi); minimizing it sharpens selection.",
    )
    parser.add_argument(
        "--style-prior-loss",
        type=str,
        choices=["dense", "js", "sparse_eps", "legacy"],
        default="dense",
        help=(
            "How the offline style-code prior supervises the selector. 'dense' uses CE against "
            "softmax(logits/tau), with the same gradient as forward KL, while TopK constructs "
            "the style token. KL against a sparse distribution is infinite for a positive prior. "
            "'js' uses bounded sparse divergence; 'sparse_eps' and 'legacy' retain historical "
            "sparse cross-entropy variants. CE's target entropy is a constant, not a gradient diagnostic."
        ),
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=1,
        help=(
            "Autoregressive (scheduled-sampling) training depth. 1 = ordinary one-step training. "
            ">1 replays the evaluation rollout for that many steps and applies the flow/recon loss at "
            "every step, feeding back the model's own decoded motion and latent. Requires "
            "--seq-len >= unroll-steps + 1. This targets the exposure-bias gap behind the rollout-vs-AE "
            "style-accuracy difference (see GOAL_STYLE_ACCURACY.md)."
        ),
    )
    parser.add_argument(
        "--feedback-start-epoch", type=int, default=1,
        help=("First epoch whose unroll history uses the model's own feedback (one-based). Epochs "
              "before it train as pure multi-step teacher forcing on GT rows (no Euler/decode)."))
    parser.add_argument(
        "--feedback-ramp-epochs", type=int, default=0,
        help=("Linearly ramp the per-window feedback share from 0 to --feedback-probability over "
              "this many epochs starting at --feedback-start-epoch. 0 jumps straight to the "
              "target share. The window gate is fixed per window, so a window's rolled history is "
              "either all feedback or all GT."))
    parser.add_argument(
        "--feedback-probability", type=float, default=1.0,
        help=("Target per-window share of model-feedback history during unroll training; 0 keeps "
              "pure GT history. Only meaningful with --unroll-steps > 1."))
    parser.add_argument("--history-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feedback-batch-schedule", choices=("window", "batch"), default="window")
    parser.add_argument("--observation-noise-std", type=float, default=0.0,
        help=("Plan-A observation noise: on the teacher-forced branch, replace the rolled GT "
              "observation by clip->quantize->decode(z_gt + U[0,std]*transport_std*eps). "
              "Simulates the model's own noisy feedback during pure TF without Euler rollout. "
              "0 disables (legacy RNG stream); inert at unroll_steps==1 or feedback_probability==1."))
    parser.add_argument(
        "--style-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the style-consistency CE: the decoded predicted motion must be classified as "
            "the requested style by a frozen frame-level style classifier. 0 disables; 0.02-0.1 is the "
            "useful range."
        ),
    )
    parser.add_argument(
        "--style-classifier-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "fid_feature_extractor_camdm100_frame_nomirror" / "100style_position_classifier.pth",
        help="Frozen frame-level (window=1) style classifier used by --style-consistency-weight",
    )
    parser.add_argument("--noise-std", type=float, default=1.0,
                        help="Initial flow-matching noise scale (the z0 = start distribution)")
    parser.add_argument("--solver-steps", type=int, default=8)
    parser.add_argument("--feedback-solver-steps", type=int, default=0,
        help=("Euler steps used only for the rolled-out feedback latent during unroll training "
              "(cheaper than the supervised/auxiliary solver). 0 keeps it equal to --solver-steps. "
              "Only affects the intermediate steps whose prediction is fed back; the FM loss is a "
              "single field evaluation and is unaffected. Requires no recon/root/consistency aux loss."))
    parser.add_argument("--latent-stats-chunk-rows", type=int, default=65536)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch < 0:
        raise ValueError('stop-after-epoch must be nonnegative')
    if not np.isfinite(args.gpu_data_cache_gb) or args.gpu_data_cache_gb<0:
        raise ValueError("GPU data cache budget must be finite and nonnegative")
    if args.style_val_batch_size < 1:
        raise ValueError('style-val-batch-size must be positive')
    if not 0 <= args.flow_low_t_probability < 1 or not 0 < args.flow_low_t_max <= 1:
        raise ValueError('Invalid flow-time probabilities')
    args.db = args.db.expanduser().resolve()
    args.vqvae_ckpt = args.vqvae_ckpt.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.resume = None if args.resume is None else args.resume.expanduser().resolve()
    args.vq_target_cache = None if args.vq_target_cache is None else args.vq_target_cache.expanduser().resolve()
    args.norm_stats = None if args.norm_stats is None else args.norm_stats.expanduser().resolve()
    args.latent_stats = (args.out_dir / "latent_stats.npz") if args.latent_stats is None else args.latent_stats.expanduser().resolve()
    args.style_code_prior = None if args.style_code_prior is None else args.style_code_prior.expanduser().resolve()
    args.init_ckpt = None if args.init_ckpt is None else args.init_ckpt.expanduser().resolve()
    if args.resume and args.init_ckpt:
        raise ValueError("--resume and --init-ckpt are mutually exclusive")
    if args.norm_stats is None:
        args.norm_stats = args.vqvae_ckpt.parent / "norm_stats.npz"
    if not args.norm_stats.is_file():
        raise FileNotFoundError("VQ normalization is required; never estimate it from evaluation data")
    for value in (args.recon_weight, args.root_recon_weight, args.style_hist_weight,
                  args.style_entropy_weight, args.style_consistency_weight):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Loss weights must be finite and non-negative")
    if args.microbatch_size < 0:
        raise ValueError("Invalid microbatch size")
    args.implementation_version = IMPLEMENTATION_VERSION
    if args.conditioning_mode in ('recipe', 'clip_recipe'):
        if not args.use_style_code_selector:
            raise ValueError('Recipe mode requires --use-style-code-selector')
        if args.style_consistency_weight > 0 or args.rollout_val_every > 0:
            raise ValueError('Recipe mode uses direct FM supervision; classifier loss and label rollout validation must be disabled')
        args.recipe_prob = 1.
        if args.conditioning_mode == 'recipe':
            args.style_hist_weight = args.style_entropy_weight = 0.
    if args.recipe_bank_max_entries < 1:
        raise ValueError('Invalid recipe bank dimensions')
    if args.tf_val_every < 0 or args.tf_val_root_tolerance < 1 or min(args.tf_val_clips, args.tf_val_windows) < 1:
        raise ValueError('Invalid teacher-forced validation settings')
    if args.tf_val_every and (args.rollout_val_every or not args.tf_val_classifier.is_file()):
        raise ValueError('Style validation requires its 240-frame classifier and rollout validation disabled')
    if (not 0 <= args.tf_val_min_matched_accuracy <= 1 or not all(np.isfinite(x) and x > 0
            for x in (args.tf_val_max_matched_ade, args.tf_val_max_cross_ade))):
        raise ValueError('Invalid TF accuracy/ADE selection limits')
    if args.tf_val_selection == 'absolute' and args.tf_val_reference:
        raise ValueError('Absolute TF selection does not use a checkpoint reference')
    if not args.use_style_code_selector:
        args.recipe_prob = 0.
        args.style_selector_init = 'random'
        args.style_proto_normalization = 'layernorm'
    if not 0 <= args.recipe_prob <= 1 or args.recipe_frames < 1 or args.recipe_exemplars < 1 or args.recipe_bank_size < 0:
        raise ValueError('Invalid recipe probability or exemplar dimensions')
    if args.recipe_exemplar_choices is not None and (not args.recipe_exemplar_choices
            or min(args.recipe_exemplar_choices) < 1
            or len(set(args.recipe_exemplar_choices)) != len(args.recipe_exemplar_choices)):
        raise ValueError('Recipe exemplar choices must be distinct positive integers')
    # Donor/decoupling supervision was removed on 2026-10-12: transplanted cross-style targets
    # measurably degraded closed-loop motion quality. Every run now trains under the donor-free
    # objective; the single tag 11 keeps every donor-era checkpoint (versions <= 10) from
    # silently resuming into it. Tag 12 adds the Plan-A observation-noise augmentation; runs
    # with the noise disabled keep the v11 RNG stream and remain resumable across the change.
    args.training_objective_version = 11 + (1 if args.observation_noise_std > 0.0 else 0)
    if args.motion_quality_selection and (not args.tf_val_every or args.style_val_mode != "rollout"):
        raise ValueError("Quality selection requires rollout style validation")
    if not all(np.isfinite(v) and v>0 for v in (args.quality_max_jitter,args.quality_max_local_fs,args.quality_max_world_fs)):
        raise ValueError("Quality limits must be finite and positive")
    if args.history_conditioning and (args.field_arch != "transformer" or args.seq_len < args.unroll_steps+2):
        raise ValueError("History requires transformer and seq_len >= unroll_steps+2")
    if args.lr_schedule_epochs < 0 or (args.lr_schedule_epochs and args.lr_schedule_epochs < args.epochs):
        raise ValueError('Cosine horizon must be 0 or at least the requested training epochs')
    if not args.resume and ((args.out_dir / "config.json").exists() or any(args.out_dir.glob("*.pt"))):
        raise ValueError("Training output already contains a run; use --resume or a new output directory")

    require_unmirrored_db(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ checkpoint not found: {args.vqvae_ckpt}")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.solver_steps < 1:
        raise ValueError("--solver-steps must be >= 1")
    if args.feedback_solver_steps < 0:
        raise ValueError("--feedback-solver-steps must be >= 0 (0 = same as --solver-steps)")
    args.feedback_solver_steps = None if args.feedback_solver_steps == 0 else int(args.feedback_solver_steps)
    if args.style_num_slots < 1:
        raise ValueError("--style-num-slots must be >= 1")
    if args.style_selector_topk < 1:
        raise ValueError("--style-selector-topk must be >= 1")
    if args.style_selector_temperature <= 0.0:
        raise ValueError("--style-selector-temperature must be > 0")
    if not args.normalize:
        raise ValueError("Minimal flowmatch training expects normalized inputs.")
    if args.use_style_code_selector and args.style_code_prior is None:
        raise ValueError("--style-code-prior is required when --use-style-code-selector is enabled.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.unroll_steps < 1:
        raise ValueError(f"--unroll-steps must be >= 1, got {args.unroll_steps}")
    if args.unroll_steps > 1 and args.seq_len < args.unroll_steps + 1:
        raise ValueError(
            f"--unroll-steps {args.unroll_steps} needs --seq-len >= {args.unroll_steps + 1} "
            f"(the window must contain the trajectory of every rolled-out step); got seq_len={args.seq_len}"
        )
    # TF -> rollout curriculum validation. A ramp only makes sense with unroll > 1; the epoch
    # window must fit inside the training budget (ramp end may exceed epochs only when the run
    # stops early, which still converges smoothly because the ramp is evaluated per epoch).
    if args.feedback_start_epoch < 1:
        raise ValueError(f"--feedback-start-epoch must be >= 1, got {args.feedback_start_epoch}")
    if args.feedback_ramp_epochs < 0:
        raise ValueError(f"--feedback-ramp-epochs must be >= 0, got {args.feedback_ramp_epochs}")
    if not 0.0 <= args.feedback_probability <= 1.0:
        raise ValueError(f"--feedback-probability must be in [0,1], got {args.feedback_probability}")
    if not np.isfinite(args.observation_noise_std) or args.observation_noise_std < 0.0:
        raise ValueError(f"--observation-noise-std must be finite and >= 0, got {args.observation_noise_std}")
    if args.unroll_steps == 1:
        if (args.feedback_start_epoch != 1 or args.feedback_ramp_epochs != 0
                or args.feedback_probability != 1.0):
            raise ValueError(
                "--feedback-start-epoch/--feedback-ramp-epochs/--feedback-probability only apply "
                "to --unroll-steps > 1")
    else:
        # Last epoch that still ramps up is start + ramp - 1 (ramp_epochs=0 jumps at start).
        last_ramp_epoch = (args.feedback_start_epoch if args.feedback_ramp_epochs == 0
                           else args.feedback_start_epoch + args.feedback_ramp_epochs - 1)
        if last_ramp_epoch > args.epochs:
            print(f"[Warn] feedback curriculum: ramp completes at epoch {last_ramp_epoch}, past "
                  f"--epochs {args.epochs}; the run ends before the target share is reached")
    set_seed(args.seed)
    device = resolve_device(args.device)
    configure_torch_runtime(
        device=device,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        matmul_precision=args.matmul_precision,
    )

    vq_target_cache = None
    if args.vq_target_cache is not None:
        vq_target_cache = VQTargetCache(args.vq_target_cache)
        if args.norm_stats is None:
            args.norm_stats = vq_target_cache.meta.norm_stats
    fixed_norm_mean = None
    fixed_norm_std = None
    if args.norm_stats is not None:
        fixed_norm_mean, fixed_norm_std = load_norm_stats(args.norm_stats)

    bundle = build_train_val_test_dataloaders(
        db_path=args.db,
        seq_len=args.seq_len,
        stride=args.stride,
        enforce_same_label=args.enforce_same_label,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        load_into_memory=args.load_into_memory,
        normalize=args.normalize,
        norm_mean=fixed_norm_mean,
        norm_std=fixed_norm_std,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    norm_stats_path = save_norm_stats_if_present(
        out_dir=args.out_dir,
        norm_mean=bundle.norm_mean,
        norm_std=bundle.norm_std,
    )
    for loader in (bundle.train_loader, bundle.val_loader, bundle.test_loader):
        if loader is not None:
            configure_controller_loader(loader, vq_target_cache, worker_batches=args.worker_batches)
    if args.norm_stats is None:
        args.norm_stats = norm_stats_path

    style_names = getattr(bundle.train_dataset, "style_names", None)
    if args.num_styles > 0:
        num_styles = int(args.num_styles)
    elif style_names:
        num_styles = len(style_names)
    else:
        labels = getattr(bundle.train_dataset, "labels", None)
        num_styles = int(np.max(labels)) + 1

    print(f"[Info] device          : {device}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] out dir         : {args.out_dir}")
    print(f"[Info] vq ckpt         : {args.vqvae_ckpt}")
    print(f"[Info] prediction mode : {args.prediction_mode}")
    print(f"[Info] conditioning    : {args.conditioning_mode}")
    print("[Info] latent norm     : co")
    print(f"[Info] num styles      : {num_styles}")
    if args.norm_stats is not None:
        print(f"[Info] norm stats      : {args.norm_stats}")
    if args.vq_target_cache is not None:
        print(f"[Info] vq target cache : {args.vq_target_cache}")

    args.num_styles = num_styles
    args.fingerprints = artifact_fingerprints(args.db, args.vqvae_ckpt, args.norm_stats)
    args.training_artifacts = {
        "prior_sha256": file_sha256(args.style_code_prior) if args.style_code_prior else None,
        "rollout_classifier_sha256": file_sha256(args.rollout_val_classifier) if args.rollout_val_every > 0 else None,
        "style_classifier_sha256": file_sha256(args.style_classifier_ckpt) if args.style_consistency_weight > 0 else None,
    }
    if args.tf_val_every:
        args.training_artifacts['tf_classifier_sha256'] = file_sha256(args.tf_val_classifier)
        args.training_artifacts['tf_reference_sha256'] = file_sha256(args.tf_val_reference) if args.tf_val_reference else None
    vq_model = build_vqvae_from_checkpoint(args.vqvae_ckpt).to(device)
    if vq_target_cache is not None:
        vq_target_cache.validate(
            db=args.db,
            vqvae_ckpt=args.vqvae_ckpt,
            norm_stats=args.norm_stats,
            vqvae=vq_model,
            fingerprints=None if args.allow_legacy_artifacts else args.fingerprints,
        )

    style_classifier = None
    style_consistency_motion_mean = None
    style_consistency_motion_std = None
    style_consistency_joint_indices = None
    classifier_label_map = None
    if args.style_consistency_weight > 0.0:
        from Utils.fid_feature_extractor import (
            YBOT15_TO_SELECTED24_INDICES,
            build_fid_feature_extractor_from_checkpoint,
        )

        if not Path(args.style_classifier_ckpt).exists():
            raise FileNotFoundError(
                f"--style-classifier-ckpt not found: {args.style_classifier_ckpt} (required by "
                "--style-consistency-weight)"
            )
        style_classifier, style_clf_meta = build_fid_feature_extractor_from_checkpoint(
            Path(args.style_classifier_ckpt), map_location="cpu"
        )
        if int(style_clf_meta.get("window_size", 60)) != 1:
            raise ValueError(
                "style consistency needs a frame-level (window_size=1) classifier; "
                f"got window_size={style_clf_meta.get('window_size')} in {args.style_classifier_ckpt}. "
                "Train one with: train_fid_feature_extractor_camdm.py --window-size 1 --include-mirror "
                "--clip-offset 10"
            )
        from Utils.controller_validation import classifier_label_mapping
        classifier_label_map = classifier_label_mapping(style_names, style_clf_meta, device=device)
        style_classifier = style_classifier.to(device).eval()
        for parameter in style_classifier.parameters():
            parameter.requires_grad_(False)
        if fixed_norm_mean is None or fixed_norm_std is None:
            raise RuntimeError("--style-consistency-weight requires --norm-stats (or a saved norm_stats.npz)")
        style_consistency_motion_mean = torch.from_numpy(np.asarray(fixed_norm_mean[84:], dtype=np.float32)).to(device)
        style_consistency_motion_std = torch.from_numpy(np.asarray(fixed_norm_std[84:], dtype=np.float32)).to(device)
        style_consistency_joint_indices = torch.tensor(YBOT15_TO_SELECTED24_INDICES, dtype=torch.long)
        print(f"[Info] style consistency: {args.style_classifier_ckpt} (window=1, frozen)")

    style_prior_payload = None
    if args.use_style_code_selector:
        style_prior_payload = load_style_code_prior(args.style_code_prior)
        validate_fingerprints(style_prior_payload.get("fingerprints"), args.fingerprints,
                              required=not args.allow_legacy_artifacts)
        if style_prior_payload["split"] != "train" or style_prior_payload["style_names"] != style_names:
            raise ValueError("Style prior must match train split and ordered generator style names")
        if vq_target_cache is not None and not args.allow_legacy_artifacts:
            if style_prior_payload.get("cache_sha256") != vq_target_cache.meta.indices_sha256:
                raise ValueError("Style prior was counted from a different VQ target cache")
        style_prior = style_prior_payload["prior"]
        if style_prior.shape[0] != num_styles:
            raise ValueError(
                f"style code prior num_styles mismatch: prior={style_prior.shape[0]}, dataset={num_styles}"
            )
        selected_slots = [int(slot) for slot in np.asarray(style_prior_payload["selected_slots"], dtype=np.int64).tolist()]
        if args.style_num_slots != len(selected_slots):
            print(
                f"[Warn] style_num_slots={args.style_num_slots} overridden by prior selected_slots={len(selected_slots)}"
            )
        args.style_num_slots = len(selected_slots)
        args.style_slot_indices = selected_slots
        print(f"[Info] style prior     : {args.style_code_prior}")
        prior_vq = style_prior_payload.get("vqvae_ckpt")
        if prior_vq is not None and str(prior_vq) != str(args.vqvae_ckpt) and not style_prior_payload.get("fingerprints"):
            print(
                f"[Warn] style prior was estimated from a different VQ checkpoint: prior="
                f"{prior_vq} vs controller VQ={args.vqvae_ckpt}; the prior's codebook indices refer "
                f"to that earlier codebook"
            )
        prior_slots_meta = style_prior_payload.get("num_codebooks")
        if prior_slots_meta is not None and int(prior_slots_meta) != int(vq_model.num_latents):
            raise ValueError(
                f"style prior num_codebooks={int(prior_slots_meta)} does not match the VQ's "
                f"num_latents={int(vq_model.num_latents)}"
            )
        print(f"[Info] style prior loss: {args.style_prior_loss}")
        print(f"[Info] style slots     : {selected_slots}")
    else:
        args.style_slot_indices = []

    model = build_flowmatch_controller_from_config(
        vars(args),
        vqvae=vq_model,
        num_styles=num_styles,
    ).to(device)
    # Inference-time solver for this run: rollout validation and export paths read this
    # (motion_eval.load_motion_model sets the same attribute from checkpoint args), so a
    # training/validation/inference step-count mismatch cannot reappear silently.
    model.eval_solver_steps = int(args.solver_steps)
    model.eval_noise_std = float(args.noise_std)
    if style_prior_payload is not None:
        model.set_style_code_prior(style_prior_payload["prior"])
    if args.latent_stats.exists():
        latent_stats = _load_latent_stats(
            args.latent_stats,
            latent_dim=model.latent_dim,
        )
        validate_fingerprints(latent_stats.get("fingerprints"), args.fingerprints,
                              required=not args.allow_legacy_artifacts)
        if latent_stats.get("split") != "train" and not args.allow_legacy_artifacts:
            raise ValueError("Latent stats must be estimated from train only")
    else:
        raw_mean, raw_std, raw_min, raw_max = _estimate_raw_latent_stats(
            db_path=args.db,
            vqvae=vq_model,
            device=device,
            norm_mean=bundle.norm_mean,
            norm_std=bundle.norm_std,
            vq_target_cache=vq_target_cache,
            chunk_rows=args.latent_stats_chunk_rows,
        )
        latent_stats = _build_latent_stats_payload(
            raw_mean=raw_mean,
            raw_std=raw_std,
            raw_min=raw_min,
            raw_max=raw_max,
        )
        latent_stats["fingerprints"] = args.fingerprints
        _save_latent_stats(args.latent_stats, latent_stats)
    args.training_artifacts["latent_stats_sha256"] = file_sha256(args.latent_stats)
    args.training_artifacts["cache_sha256"] = vq_target_cache.meta.indices_sha256 if vq_target_cache else None
    model.set_latent_prior_stats(
        latent_stats["offset"],
        latent_stats["scale"],
        transport_std=latent_stats["transport_std"],
        data_min=latent_stats["data_min"],
        data_max=latent_stats["data_max"],
    )
    print(f"[Info] latent stats    : {args.latent_stats}")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in flow controller.")
    args.fused_adamw = bool(args.fused_adamw and device.type == 'cuda')
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay,
                      fused=args.fused_adamw, foreach=False if args.fused_adamw else None)
    print(f'[Info] fused AdamW     : {args.fused_adamw} (FP32)')
    scheduler = CosineAnnealingLR(optimizer, T_max=args.lr_schedule_epochs or args.epochs, eta_min=args.min_lr)
    start_epoch = 1
    best_metric = float("inf")
    resume_rng = None
    best_loss = float("inf")
    best_key = (1, float("inf"))
    args.best_selection = None
    if args.init_ckpt is not None:
        initial = load_checkpoint(args.init_ckpt, map_location="cpu")
        prior_args = initial.get("args", {})
        if prior_args.get("prediction_mode") != args.prediction_mode:
            raise ValueError("Initialization checkpoint prediction mode differs")
        if list(prior_args.get("style_slot_indices", [])) != list(args.style_slot_indices):
            raise ValueError("Initialization checkpoint style slot identities/order differ")
        bundled = {k[6:]: v for k, v in sanitize_state_dict_keys(initial["model"]).items() if k.startswith("vqvae.")}
        if state_fingerprint(bundled) != args.fingerprints["vq_state_sha256"]:
            raise ValueError("Initialization checkpoint has a different VQ manifold")
        initial_state = sanitize_state_dict_keys(initial['model'])
        init_field_arch = str(prior_args.get('field_arch', 'transformer'))
        if init_field_arch != args.field_arch:
            raise ValueError(
                f"Initialization checkpoint uses field_arch={init_field_arch!r}, this run uses "
                f"{args.field_arch!r}; cross-architecture --init-ckpt transfer is not supported")
        init_uses_motion = bool(prior_args.get('use_motion_token', True))
        if init_uses_motion != args.use_motion_token:
            initial_state = remap_token_state_dict(
                initial_state, drop_motion_token=init_uses_motion and not args.use_motion_token)
            dropped = 'field.motion_proj.* + one position-embedding row'
            print(f'[Info] init remap      : 6->5 tokens ({dropped} dropped)')
        missing, unexpected = model.load_state_dict(initial_state, strict=False)
        allowed = {k for k in model.state_dict() if k.startswith('histogram_selector.')} if prior_args.get('conditioning_mode') != 'clip_recipe' else set()
        if args.history_conditioning and not prior_args.get("history_conditioning", False):
            allowed.update(k for k in model.state_dict() if k.startswith("history_delta_proj."))
        if set(missing) - allowed or unexpected:
            raise ValueError(f'Incompatible initialization weights: missing={missing}, unexpected={unexpected}')
        # Data/normalization buffers belong to this run, not the initialization run.
        model.set_latent_prior_stats(latent_stats["offset"], latent_stats["scale"],
            transport_std=latent_stats["transport_std"], data_min=latent_stats["data_min"], data_max=latent_stats["data_max"])
        if style_prior_payload is not None:
            model.set_style_code_prior(style_prior_payload["prior"])
    args.selector_initialization = {'method': 'checkpoint' if args.init_ckpt else 'random'}
    if args.conditioning_mode == 'recipe':
        args.selector_initialization = {'method': 'unused_recipe_selector'}
    elif args.conditioning_mode == 'clip_recipe':
        args.selector_initialization = {'method': 'histogram_residual_checkpoint' if args.init_ckpt and prior_args.get('conditioning_mode') == 'clip_recipe' else 'histogram_zero_residual'}
    elif args.use_style_code_selector and args.style_selector_init == 'prior' and not (args.init_ckpt or args.resume):
        args.selector_initialization = model.style_selector.initialize_from_prior(model.style_prior_probs)
        print(f'[Info] selector init   : {args.selector_initialization}')
    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        if (ckpt.get("implementation_version") != IMPLEMENTATION_VERSION
                or ckpt.get("training_objective_version") != args.training_objective_version or "rng_state" not in ckpt):
            raise ValueError('Resume objective version differs; initialize compatible weights into a new run with --init-ckpt')
        prior_args = ckpt.get("args", {})
        args.selector_initialization = prior_args.get('selector_initialization', {'method': 'legacy_checkpoint'})
        validate_fingerprints(prior_args.get("fingerprints"), args.fingerprints)
        if prior_args.get("training_artifacts") != args.training_artifacts:
            raise ValueError("Resume prior, latent stats, cache or classifier content differs")
        bundled = {k[6:]: v for k, v in sanitize_state_dict_keys(ckpt['model']).items() if k.startswith('vqvae.')}
        if state_fingerprint(bundled) != args.fingerprints['vq_state_sha256']:
            raise ValueError('Resume checkpoint embedded VQ weights differ from the frozen VQ')
        resume_fields = ("prediction_mode", "flow_loss_space", "seq_len", "unroll_steps",
            "root_recon_weight", "recon_weight", "style_hist_weight", "style_entropy_weight",
            "style_consistency_weight", "inference_requantize", "inference_latent_clip", "solver_steps",
            "noise_std", "seed", "batch_size", "microbatch_size", "lr", "min_lr", "weight_decay",
            "epochs", "dropout", "matmul_precision", "tf32", "cudnn_benchmark", "grad_clip", "fp32_math_attention",
            "stride", "style_slot_indices", "style_prior_loss", "style_selector_topk",
            "style_selector_temperature", "target_time_eps", "rollout_val_every", "val_every",
            "max_train_batches", "max_val_batches", "enforce_same_label", "val_ratio", "test_ratio",
            'style_proto_normalization', 'style_selector_init', 'recipe_prob', 'recipe_frames', 'recipe_exemplars',
            'lr_schedule_epochs', 'train_feedback_context')
        resume_fields += ('conditioning_mode', 'recipe_exemplar_choices', 'recipe_bank_size')
        resume_fields += ('recipe_bank_max_entries',
                          'tf_val_every', 'tf_val_root_tolerance', 'tf_val_clips', 'tf_val_windows')
        resume_fields += ('tf_val_selection', 'tf_val_min_matched_accuracy', 'tf_val_max_matched_ade', 'tf_val_max_cross_ade')
        if args.tf_val_every:
            resume_fields += ('style_val_mode', 'style_val_batch_size')
        resume_fields += ('flow_low_t_probability', 'flow_low_t_max')
        resume_fields += ('use_motion_token', 'field_arch', 'feedback_solver_steps',
                          'feedback_start_epoch', 'feedback_ramp_epochs', 'feedback_probability',
                          'observation_noise_std',
                          'recon_inference_path', 'recon_anchor_weight', 'shared_style_table')
        legacy_defaults = {'style_proto_normalization': 'layernorm', 'style_selector_init': 'random',
                           'recipe_prob': 0., 'recipe_frames': 240, 'recipe_exemplars': 5, 'lr_schedule_epochs': 0,
                           'train_feedback_context': False,
                           'conditioning_mode': 'label', 'recipe_exemplar_choices': None,
                           'recipe_bank_size': 0, 'recipe_bank_max_entries': 131072, 'tf_val_every': 0,
                           'tf_val_root_tolerance': 1.15, 'tf_val_clips': 90, 'tf_val_windows': 3}
        legacy_defaults.update(tf_val_selection='absolute', tf_val_min_matched_accuracy=.85,
                               tf_val_max_matched_ade=5., tf_val_max_cross_ade=10.)
        legacy_defaults['style_val_mode'] = 'reconstruct'
        legacy_defaults.update(fp32_math_attention=False, style_val_batch_size=8)
        legacy_defaults['label_selector_dedup'] = True
        legacy_defaults.update(flow_low_t_probability=0., flow_low_t_max=.2)
        legacy_defaults['use_motion_token'] = True
        legacy_defaults['field_arch'] = 'transformer'
        legacy_defaults['feedback_solver_steps'] = None
        legacy_defaults['recon_inference_path'] = False
        legacy_defaults['recon_anchor_weight'] = 0.0
        legacy_defaults['shared_style_table'] = True
        legacy_defaults['observation_noise_std'] = 0.0
        resume_fields += ("history_conditioning", "feedback_batch_schedule", "motion_quality_selection",
                          "quality_max_jitter", "quality_max_local_fs", "quality_max_world_fs", "gpu_data_cache_gb", "cache_teacher_motion")
        legacy_defaults.update(history_conditioning=False, feedback_batch_schedule="window",
                               motion_quality_selection=False,quality_max_jitter=.5,quality_max_local_fs=.8,quality_max_world_fs=1.,gpu_data_cache_gb=0.,cache_teacher_motion=False)
        legacy_defaults.update(feedback_start_epoch=1, feedback_ramp_epochs=0, feedback_probability=1.0)
        for name in resume_fields:
            if prior_args.get(name, legacy_defaults.get(name)) != getattr(args, name):
                raise ValueError(f"Resume configuration mismatch: {name}")
        resume_rng = ckpt["rng_state"]
        args.best_selection = ckpt.get("selection")
        args.tf_validation_reference = prior_args.get('tf_validation_reference')
        best_key = tuple(args.best_selection["key"]) if args.best_selection else best_key
        best_loss = float(prior_args.get("best_loss", float("inf")))
        load_state_dict_with_allowed_missing(
            model,
            sanitize_state_dict_keys(ckpt["model"]),
            allowed_missing=FLOWMATCH_VQ_COMPAT_MISSING_KEYS,
        )
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
            for group in optimizer.param_groups:
                group['fused'] = args.fused_adamw
                if args.fused_adamw:
                    group['foreach'] = False
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        print(f"[Info] resume         : {args.resume} (epoch {start_epoch})")

    recipe_pool = None
    if args.recipe_prob > 0:
        recipe_pool = TrainStyleRecipePool(db=args.db, model=model, cache=vq_target_cache,
            mean=bundle.norm_mean, std=bundle.norm_std, frames=args.recipe_frames, exemplars=args.recipe_exemplars,
            exemplar_choices=args.recipe_exemplar_choices, bank_size=args.recipe_bank_size, seed=args.seed,
            bank_max_entries=args.recipe_bank_max_entries)
        print(f'[Info] train recipes   : p={args.recipe_prob}, choices={recipe_pool.exemplar_choices} clips × {args.recipe_frames} frames')

    args.resident_data = None
    if args.gpu_data_cache_gb and device.type == 'cuda' and vq_target_cache is not None:
        from Utils.clip_recipe_resident import ResidentTrainingData, ResidentWindowLoader
        resident = ResidentTrainingData(bundle.train_loader.dataset, vq_target_cache, bundle.norm_mean,
            bundle.norm_std, device=device, budget_gb=args.gpu_data_cache_gb,
            decoder=vq_model if args.cache_teacher_motion else None)
        args.resident_data = resident.info
        if resident.features is not None and bundle.train_loader.drop_last:
            bundle.train_loader = ResidentWindowLoader(bundle.train_loader.dataset,bundle.train_loader.sampler,
                resident,batch_size=args.batch_size)
        print(f'[GPU train cache] {resident.info}')
    if args.resume:
        previous=prior_args.get('resident_data') or {}
        current=args.resident_data or {}
        if previous.get('mode')!=current.get('mode') or previous.get('codec_chunk_rows')!=current.get('codec_chunk_rows'):
            raise ValueError('Resume requires the same resident-data/codec cache mode')
    tf_validator = None
    if args.tf_val_every:
        from Utils.controller_validation import TeacherForcedStyleValidation
        with isolated_rng(args.seed + 1002):
            tf_validator = TeacherForcedStyleValidation(model, db=args.db, mean=bundle.norm_mean,
                std=bundle.norm_std, classifier_path=args.tf_val_classifier, seed=args.seed,
                clips=args.tf_val_clips, windows=args.tf_val_windows, out_dir=args.out_dir,
                eval_mode=args.style_val_mode, batch_size=args.style_val_batch_size,quality_metrics=args.motion_quality_selection)
            if args.tf_val_selection == 'relative' and not getattr(args, 'tf_validation_reference', None):
                if args.tf_val_reference:
                    reference_cases = args.tf_val_reference.parent / ('rollout_style_validation_cases.json'
                        if args.style_val_mode == 'rollout' else 'tf_validation_cases.json')
                    if json.loads(reference_cases.read_text()) != tf_validator.manifest:
                        raise ValueError('TF reference must use the identical validation case/exemplar protocol')
                    args.tf_validation_reference = json.loads(args.tf_val_reference.read_text())
                else:
                    args.tf_validation_reference = tf_validator.evaluate(model)
                _save_json(args.out_dir / 'tf_validation_reference.json', args.tf_validation_reference)
    _save_json(args.out_dir / "config.json", vars(args))
    history_path = args.out_dir / "history.jsonl"

    if resume_rng is not None:
        restore_rng_state(resume_rng)
    if args.stop_after_epoch and start_epoch > args.stop_after_epoch:
        print(f'[Pilot stop] epoch {args.stop_after_epoch} already saved; use --stop-after-epoch 0 to continue')
        return
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.8f}")
        _seed_loader(bundle.train_loader, seed=args.seed, epoch=epoch)

        train_t0 = time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        epoch_feedback_probability = feedback_probability_for_epoch(
            epoch, start_epoch=args.feedback_start_epoch,
            ramp_epochs=args.feedback_ramp_epochs, probability=args.feedback_probability)
        train_metrics = _run_epoch(
            model=model,
            loader=bundle.train_loader,
            device=device,
            prediction_mode=model.prediction_mode,
            recon_weight=args.recon_weight,
            recon_inference_path=args.recon_inference_path,
            recon_anchor_weight=args.recon_anchor_weight,
            style_hist_weight=args.style_hist_weight,
            style_entropy_weight=args.style_entropy_weight,
            noise_std=args.noise_std,
            solver_steps=args.solver_steps,
            feedback_solver_steps=args.feedback_solver_steps,
            feedback_probability=epoch_feedback_probability,
            feedback_batch_schedule=args.feedback_batch_schedule,
            reuse_style_table=args.shared_style_table,
            optimizer=optimizer,
            vq_target_cache=vq_target_cache,
            cache_vqvae=vq_model,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            unroll_steps=args.unroll_steps,
            style_consistency_weight=args.style_consistency_weight,
            style_classifier=style_classifier,
            motion_mean=style_consistency_motion_mean,
            motion_std=style_consistency_motion_std,
            joint_indices=style_consistency_joint_indices,
            split="train", root_recon_weight=args.root_recon_weight,
            microbatch_size=args.microbatch_size,
            classifier_label_map=classifier_label_map, max_batches=args.max_train_batches,
            recipe_pool=recipe_pool, recipe_probability=args.recipe_prob,
            train_feedback_context=args.train_feedback_context,
            observation_noise_std=args.observation_noise_std,
            flow_low_t_probability=args.flow_low_t_probability,
            flow_low_t_max=args.flow_low_t_max,
        )
        train_sec = time.perf_counter() - train_t0

        val_metrics = None
        metric_for_best = None
        if bundle.val_loader is None:
            metric_for_best = train_metrics["loss"]
        elif epoch % args.val_every == 0:
            with isolated_rng(args.seed + 1001), torch.no_grad():
                _seed_loader(bundle.val_loader, seed=args.seed + 1001, epoch=0)
                val_metrics = _run_epoch(
                    model=model,
                    loader=bundle.val_loader,
                    device=device,
                    prediction_mode=model.prediction_mode,
                    recon_weight=args.recon_weight,
                    recon_inference_path=args.recon_inference_path,
                    recon_anchor_weight=args.recon_anchor_weight,
                    style_hist_weight=args.style_hist_weight,
                    style_entropy_weight=args.style_entropy_weight,
                    noise_std=args.noise_std,
                    solver_steps=args.solver_steps,
                    optimizer=None,
                    style_consistency_weight=0.0,
                    style_classifier=None,
                    vq_target_cache=vq_target_cache,
                    cache_vqvae=vq_model,
                    grad_clip=0.0,
                    log_interval=args.log_interval,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    split="val", root_recon_weight=args.root_recon_weight,
                    microbatch_size=args.microbatch_size, max_batches=args.max_val_batches,
                    train_feedback_context=args.train_feedback_context,
                    recipe_pool=recipe_pool, recipe_probability=args.recipe_prob,
                )
            metric_for_best = val_metrics["loss"]

        scheduler.step()
        epoch_sec = time.time() - epoch_t0

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "weighted": _weighted_loss_breakdown(train_metrics, args),
            "time_sec": epoch_sec,
            "train_time_sec": train_sec,
            "train_num_batches": len(bundle.train_loader),
            "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.,
            "feedback_curriculum": {
                "start_epoch": args.feedback_start_epoch,
                "ramp_epochs": args.feedback_ramp_epochs,
                "target_probability": args.feedback_probability,
                "probability": epoch_feedback_probability,
                "actual_fraction": train_metrics["feedback_fraction"],
            },
        }
        _append_jsonl(history_path, record)
        breakdown = _weighted_loss_breakdown(train_metrics, args)
        print(
            f"[Epoch {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"train_flow={train_metrics['flow_loss']:.6f} "
            f"train_recon={train_metrics['recon_loss']:.6f} "
            f"| w_flow={breakdown['flow']['weighted']:.4f} "
            f"w_recon={breakdown['recon']['weighted']:.4f} "
            f"w_prior={breakdown['style_prior']['weighted']:.4f} "
            f"prior_KL={train_metrics['style_prior_kl']:.4f} "
            f"prior_H={train_metrics['style_prior_target_entropy']:.4f} "
            f"w_ent={breakdown['style_entropy']['weighted']:.4f} "
            f"selector_H={train_metrics['selector_entropy']:.4f} "
            f"selector_top1={train_metrics['style_top1_prob']:.4f} "
            f"recipe_fraction={train_metrics['recipe_fraction']:.3f} "
            f"recipe_n={train_metrics['recipe_exemplars_mean']:.2f} "
            f"recipe_bank_hit={train_metrics['recipe_bank_hit_fraction']:.3f}"
            + (
                ""
                if val_metrics is None
                else f" | val_loss={val_metrics['loss']:.6f} val_flow={val_metrics['flow_loss']:.6f} val_recon={val_metrics['recon_loss']:.6f}"
            )
            + f" | time={epoch_sec:.1f}s"
        )
        # CE includes a constant target entropy; its proportion is not a gradient diagnostic.
        if style_prior_payload is not None:
            prior_raw = float(train_metrics.get("style_hist_loss", float("nan")))
            floor = _style_prior_entropy_floor(style_prior_payload)
            if np.isfinite(prior_raw) and np.isfinite(floor) and prior_raw > max(6.0, floor + 1.0):
                diagnostic = ('Clip support and bounded residuals can impose a nonzero attainable KL floor; '
                    'high CE/KL alone does not establish optimization failure.' if args.conditioning_mode == 'clip_recipe'
                    else 'Inspect selector probabilities and gradients; the CE entropy floor is constant.')
                print(
                    f"[Warn] epoch {epoch}: L_prior={prior_raw:.3f} is far above its floor "
                    f"{floor:.3f} (CE >= E[H(pi_bar)]). {diagnostic}"
                )
        if metric_for_best is not None and metric_for_best < best_loss:
            best_loss = metric_for_best
            args.best_loss = best_loss
            _save_checkpoint(path=args.out_dir / "best_loss.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, best_metric=best_loss, args=args)
        if tf_validator is not None and (epoch % args.tf_val_every == 0 or epoch == args.epochs
                or (args.stop_after_epoch and epoch == args.stop_after_epoch)):
            from Utils.controller_validation import teacher_forced_selection_key
            tf_t0 = time.perf_counter()
            with isolated_rng(args.seed + 1002):
                metrics = tf_validator.evaluate(model)
            metrics['time_sec'] = time.perf_counter() - tf_t0
            key = teacher_forced_selection_key(metrics,
                getattr(args, 'tf_validation_reference', None) if args.tf_val_selection == 'relative' else None,
                args.tf_val_root_tolerance, min_matched_accuracy=args.tf_val_min_matched_accuracy,
                max_matched_ade=args.tf_val_max_matched_ade, max_cross_ade=args.tf_val_max_cross_ade)
            if args.motion_quality_selection:
                from Utils.controller_validation import motion_quality_selection_key
                key = motion_quality_selection_key(metrics,min_matched=args.tf_val_min_matched_accuracy,
                    max_matched_ade=args.tf_val_max_matched_ade,max_cross_ade=args.tf_val_max_cross_ade,
                    max_jitter=args.quality_max_jitter,max_local_fs=args.quality_max_local_fs,max_world_fs=args.quality_max_world_fs)
            validation_name = 'Rollout' if args.style_val_mode == 'rollout' else 'TF'
            validation_log = 'rollout_style_val' if args.style_val_mode == 'rollout' else 'tf_style_val'
            print(f'[{validation_name} validation] matched={metrics["matched_accuracy"]:.4f} cross={metrics["cross_accuracy"]:.4f} '
                  f'ADE={metrics["matched_root_ade_cm"]:.3f}/{metrics["cross_root_ade_cm"]:.3f}cm '
                  f'constraints={key[0] == 0} time={metrics["time_sec"]:.1f}s')
            _append_jsonl(history_path, {'epoch': epoch, validation_log: metrics, 'selection_key': list(key)})
            if args.best_selection is None or key < best_key:
                best_key, best_metric = key, float(key[1])
                args.best_selection = {'key': list(key), 'epoch': epoch, 'metrics': metrics,
                    'protocol': 'rollout_240f' if args.style_val_mode == 'rollout' else 'teacher_forced_240f',
                    'policy': 'motion_quality' if args.motion_quality_selection else args.tf_val_selection,
                    'constraints_passed': key[0] == 0}
                _save_checkpoint(path=args.out_dir / 'best.pt', model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, best_metric=best_metric, args=args)
        elif args.rollout_val_every > 0 and (epoch % args.rollout_val_every == 0 or epoch == args.epochs):
            from Utils.controller_validation import evaluate_validation, selection_key
            with isolated_rng(args.seed + 1002):
                rollout = evaluate_validation(model, db=args.db, mean=bundle.norm_mean,
                    std=bundle.norm_std, classifier_path=args.rollout_val_classifier, seed=args.seed)
            key = selection_key(rollout)
            _append_jsonl(history_path, {"epoch": epoch, "rollout_val": rollout})
            if args.best_selection is None or key < best_key:
                best_key = key
                best_metric = key[1]
                args.best_selection = {"key": list(key), "epoch": epoch, "metrics": rollout}
                _save_checkpoint(path=args.out_dir / "best.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, best_metric=best_metric, args=args)
        elif args.rollout_val_every == 0 and tf_validator is None and metric_for_best is not None and metric_for_best < best_metric:
            best_metric = metric_for_best
            _save_checkpoint(path=args.out_dir / "best.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, best_metric=best_metric, args=args)
        _save_checkpoint(path=args.out_dir / "latest.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, best_metric=best_metric, args=args)
        if args.save_every > 0 and (epoch % args.save_every == 0):
            _save_checkpoint(
                path=args.out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
            )

        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(f'[Pilot stop] saved epoch {epoch}; total budget remains {args.epochs}')
            break

    # Test evaluation is a separate, sealed phase after A/B/C have all finished.
    print(f"\n[Done] best metric: {best_metric:.6f}")
    print(f"[Done] outputs    : {args.out_dir}")


if __name__ == "__main__":
    main()
