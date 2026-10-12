"""
Minimal training script for the flow-matching VQ controller.

Core pipeline:
  1) load normalized 375-dim frame windows from the memmap DB
  2) convert each window into adjacent frame pairs
  3) freeze the basic VQ-VAE and train a flow field in latent space
  4) optionally reuse precomputed VQ target cache for faster training
"""

from __future__ import annotations

import argparse
import json
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
from dataset import build_train_val_test_dataloaders
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
    validate_memmap_db_dir,
)

FLOWMATCH_VQ_COMPAT_MISSING_KEYS = (
    "style_slot_indices",
    "style_prior_probs",
    "style_prior_ready",
)
from Utils.style_code_prior import load_style_code_prior
from Utils.vq_target_cache import VQTargetCache
from Utils.reproducibility import (
    IMPLEMENTATION_VERSION, artifact_fingerprints, isolated_rng, rng_state,
    restore_rng_state, state_fingerprint, validate_fingerprints, file_sha256,
)


def matched_root_targets(*, x_window, labels, latents, suffixes, norm_mean, norm_std,
                         frame_latent_fn, prob, generator=None, return_donors=False):
    """Build one donor per window, preserving source root motion and relative joint velocity."""
    batch, frames, dims = x_window.shape
    if dims != 375 or len(suffixes) != batch:
        raise ValueError("matched_root requires motion375 windows and per-window suffixes")
    if frames < 2 or not 0 <= prob <= 1:
        raise ValueError("matched_root requires at least two frames and probability in [0,1]")
    mean = torch.as_tensor(norm_mean, device=x_window.device, dtype=x_window.dtype).reshape(375)
    std = torch.as_tensor(norm_std, device=x_window.device, dtype=x_window.dtype).reshape(375)
    raw = x_window * std + mean
    motions = raw[:, 1:, 84:]
    root = motions[..., :3]
    planar = root[..., [0, 2]]
    dv = (planar[:, None] - planar[None]).norm(dim=-1)
    limits = torch.maximum(planar.norm(dim=-1) * .2, planar.new_full((batch, frames - 1), .25))
    yaw_difference = (root[:, None, :, 1] - root[None, :, :, 1]).abs()
    suffix_lookup = {name: i for i, name in enumerate(dict.fromkeys(suffixes))}
    suffix_ids = torch.tensor([suffix_lookup[s] for s in suffixes], device=x_window.device)
    valid = ((dv <= limits[:, None]).all(-1) & (yaw_difference <= 30).all(-1)
             & (labels[:, None] != labels[None]) & (suffix_ids[:, None] == suffix_ids[None]))
    feet = motions[..., 3:75].reshape(batch, frames - 1, 24, 3)[..., [18, 19, 22, 23], :]
    foot_cost = ((feet[:, None] - feet[None]) / .2).square().mean(dim=(-1, -2, -3))
    cost = foot_cost + (dv / .25).square().mean(-1) + (yaw_difference / 30).square().mean(-1)
    donors = torch.arange(batch, device=x_window.device)
    requested = torch.rand(batch, device=x_window.device, generator=generator) < prob
    # Uniform over qualified styles, then nearest donor within that style. No per-sample
    # nonzero/item calls or CPU loop; unequal donor counts must not bias style selection.
    styles, local_labels = labels.unique(sorted=True, return_inverse=True)
    style_cost = cost.new_full((batch, styles.numel()), float('inf'))
    style_cost.scatter_reduce_(1, local_labels[None].expand(batch, -1),
                               cost.masked_fill(~valid, float('inf')), reduce='amin')
    available = torch.isfinite(style_cost)
    rank = (torch.rand(batch, device=x_window.device, generator=generator) * available.sum(1)).long()
    chosen = ((available.cumsum(1) - 1 == rank[:, None]) & available).long().argmax(1)
    selected_cost = cost.masked_fill(~(valid & (local_labels[None] == chosen[:, None])), float('inf'))
    donors = torch.where(requested & available.any(1), selected_cost.argmin(1), donors)
    changed = donors != torch.arange(batch, device=donors.device)
    target_motion = motions.clone()
    target_labels = labels.clone()
    target_latents = latents[:, 1:].clone()
    if changed.any():
        src = torch.nonzero(changed, as_tuple=False).flatten()
        donor = donors[src]
        transplanted = motions[donor].clone()
        delta = root[src] - root[donor]
        translation = torch.zeros_like(delta)
        translation[..., 0] = delta[..., 0]
        translation[..., 2] = delta[..., 2]
        omega = torch.zeros_like(delta)
        omega[..., 1] = torch.deg2rad(delta[..., 1])
        positions = transplanted[..., 3:75].reshape(-1, frames - 1, 24, 3)
        correction = translation[..., None, :] + torch.linalg.cross(
            omega[..., None, :].expand_as(positions), positions, dim=-1)
        velocities = transplanted[..., 219:291].reshape_as(positions)
        transplanted[..., 219:291] = (velocities + correction).flatten(-2)
        transplanted[..., :3] = root[src]
        target_motion[src] = transplanted
        target_labels[src] = labels[donor]
        synthetic = torch.cat((raw[src, 1:, :84], transplanted), -1)
        synthetic = (synthetic - mean) / std
        target_latents[src] = frame_latent_fn(synthetic.reshape(-1, 375)).reshape(len(src), frames - 1, -1)
    result = (target_labels, target_latents, (target_motion - mean[84:]) / std[84:], float(changed.float().mean()))
    return result + (donors,) if return_donors else result


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


def apply_style_decoupling_windows(
    *,
    style_label: torch.Tensor,
    target_latents: torch.Tensor,
    recon_targets: torch.Tensor,
    prob: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Decoupling for the unrolled window path.

    Same semantics as `apply_style_decoupling`, but the per-step targets are whole sequences:
    `target_latents` [B, S, L] and `recon_targets` [B, S, 291] are replaced by the donor sample's
    (same relative steps), and the requested style becomes the donor's style. Trajectories are not
    touched -- the source window keeps providing the future trajectory conditioning.
    """
    if prob <= 0.0:
        return style_label, target_latents, recon_targets, 0.0
    batch = int(style_label.size(0))
    mask = torch.rand(batch, device=style_label.device, generator=generator) < float(prob)
    if not bool(mask.any()):
        return style_label, target_latents, recon_targets, 0.0
    by_style: dict[int, list[int]] = {}
    for idx, label in enumerate(style_label.detach().cpu().tolist()):
        by_style.setdefault(int(label), []).append(idx)
    styles_present = sorted(by_style)
    if len(styles_present) < 2:
        return style_label, target_latents, recon_targets, 0.0

    # donors are read from the pre-swap tensors: a donor that was itself decoupled would
    # otherwise hand over a target belonging to yet another style
    target_latent_source = target_latents
    recon_target_source = recon_targets
    style_label = style_label.clone()
    target_latents = target_latents.clone()
    recon_targets = recon_targets.clone()
    applied = 0
    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        own = int(style_label[idx].item())
        candidates = [style for style in styles_present if style != own]
        if not candidates:
            continue
        donor_style = candidates[int(torch.randint(len(candidates), (1,), device="cpu", generator=generator).item())]
        donors = by_style[donor_style]
        donor = donors[int(torch.randint(len(donors), (1,), device="cpu", generator=generator).item())]
        target_latents[idx] = target_latent_source[donor]
        recon_targets[idx] = recon_target_source[donor]
        style_label[idx] = donor_style
        applied += 1
    return style_label, target_latents, recon_targets, applied / float(batch)


def unrolled_flow_loss(
    *,
    model,
    x_window: torch.Tensor,
    style_label: torch.Tensor | None,
    prev_latent_0: torch.Tensor,
    target_latents: torch.Tensor,
    recon_targets: torch.Tensor,
    steps: int,
    flow_weight: float = 1.0,
    recon_weight: float = 0.0,
    root_recon_weight: float = 0.0,
    style_hist_weight: float = 0.0,
    style_entropy_weight: float = 0.0,
    noise_std: float = 1.0,
    prev_latent_noise_std: float = 0.0,
    prev_motion_noise_std: float = 0.0,
    prev_latent_dropout: float = 0.0,
    prev_motion_dropout: float = 0.0,
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
    history_noise_mode="legacy",
    history_noise_probability=0.0,
    history_root_noise_std=0.05,
    history_mask=None,
    flow_low_t_probability=0.,
    flow_low_t_max=.2,
) -> dict[str, torch.Tensor]:
    """Mean per-step objectives plus one style regularizer per window.

    Reconstruction uses continuous predictions and GT or explicitly synthesized motion291.
    Feedback uses the shared inference policy and is detached by default. VQ stays frozen.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
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
    for step in range(steps):
        # The target motion is GT or a synthesized donor; never decode(target_latent).
        x_next_target = torch.cat((x_window[:, step + 1, :84], recon_targets[:, step]), dim=-1)
        out = model.compute_loss(
            x_curr=x_curr,
            x_next=x_next_target,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            clip_histograms=clip_histograms,
            clip_histograms_validated=clip_histograms_validated,
            prior_labels=prior_labels,
            return_per_sample=return_per_sample,
            prev_latent=prev_latent,
            target_latent=target_latents[:, step],
            flow_weight=flow_weight,
            recon_weight=recon_weight,
            root_recon_weight=root_recon_weight,
            style_hist_weight=style_hist_weight if step == 0 else 0.0,
            style_entropy_weight=style_entropy_weight if step == 0 else 0.0,
            noise_std=noise_std,
            prev_latent_noise_std=prev_latent_noise_std,
            prev_motion_noise_std=prev_motion_noise_std,
            prev_latent_dropout=prev_latent_dropout,
            prev_motion_dropout=prev_motion_dropout,
            solver_steps=solver_steps,
            return_prediction=step + 1 < steps or style_consistency_weight > 0,
            compute_style_metrics=step == 0,
            history_noise_mode=history_noise_mode,
            history_noise_probability=history_noise_probability,
            history_root_noise_std=history_root_noise_std,
            history_mask=history_mask,
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
            z_pred = out["pred_latent"]
            if detach_feedback:
                z_pred = z_pred.detach()
            feedback_motion, feedback_latent = model.feedback_state(
                z_pred, requantize=requantize_feedback, latent_clip=latent_clip_feedback)
            if detach_feedback:
                feedback_motion = feedback_motion.detach()
            traj_next = x_window[:, step + 1, :84]
            x_curr = torch.cat((traj_next, feedback_motion), dim=-1)
            prev_latent = feedback_latent
    averaged = {key: value if key in window_metrics else value / float(steps) for key, value in totals.items()}
    averaged["loss"] = averaged["loss"] + regularizer
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


def apply_style_decoupling(
    *,
    x_next: torch.Tensor,
    style_label: torch.Tensor,
    target_latent: torch.Tensor | None,
    prob: float,
    generator: torch.Generator | None = None,
    frame_latent_fn=None,
    retarget_latent: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """Content/style decoupling: request another style and take that style's target frame.

    With probability `prob`, sample i keeps its own content and trajectory conditioning but the
    *requested* style becomes the style of a donor sample j from a different style, and the
    training target becomes j's motion block + j's latent (the trajectory block of x_next stays
    sample i's, matching inference where the user supplies the trajectory and asks for a style).

    Without this, content and style are always consistent, so the content already implies the
    style and the generator has no incentive to follow the style token -- which is why cross-style
    transfer stays at ~10x chance (see RUN_STYLE_PRIOR_50EP.md section 9).

    Returns (x_next, style_label, target_latent, fraction_of_samples_decoupled).
    """
    if prob <= 0.0:
        return x_next, style_label, target_latent, 0.0
    batch = int(x_next.size(0))
    if batch < 2:
        return x_next, style_label, target_latent, 0.0
    mask = torch.rand(batch, device=x_next.device, generator=generator) < float(prob)
    if not bool(mask.any()):
        return x_next, style_label, target_latent, 0.0

    by_style: dict[int, list[int]] = {}
    for idx, label in enumerate(style_label.detach().cpu().tolist()):
        by_style.setdefault(int(label), []).append(idx)
    styles_present = sorted(by_style)
    if len(styles_present) < 2:
        return x_next, style_label, target_latent, 0.0

    # Donors are always read from the *original* batch: otherwise a donor that was itself
    # decoupled earlier would hand over a target belonging to yet another style.
    x_next_source = x_next
    target_latent_source = target_latent
    x_next = x_next.clone()
    style_label = style_label.clone()
    if target_latent is not None:
        target_latent = target_latent.clone()
    applied = 0
    retarget: list[tuple[int, int]] = []
    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        own = int(style_label[idx].item())
        candidates = [style for style in styles_present if style != own]
        if not candidates:
            continue
        donor_style = candidates[int(torch.randint(len(candidates), (1,), device="cpu", generator=generator).item())]
        donors = by_style[donor_style]
        donor = donors[int(torch.randint(len(donors), (1,), device="cpu", generator=generator).item())]
        # target motion block from the donor (original batch), trajectory block (0:84) stays
        # with the source sample -- matching inference, where the user supplies the trajectory
        x_next[idx, 84:] = x_next_source[donor, 84:]
        if target_latent is not None:
            target_latent[idx] = target_latent_source[donor]
        style_label[idx] = donor_style
        applied += 1
        retarget.append((idx, donor))

    if retarget_latent and target_latent is not None and frame_latent_fn is not None and retarget:
        # The donor's cached latent encodes the donor's *trajectory* as well, so training against it
        # teaches the generator to follow the style instead of the requested path (measured:
        # RootADE 1.35 -> 3.93 cm). Re-encode (this sample's trajectory, donor's motion) so the
        # target keeps the source trajectory -- exactly the cross-style inference situation.
        idxs = torch.tensor([pair[0] for pair in retarget], device=x_next.device, dtype=torch.long)
        donors = torch.tensor([pair[1] for pair in retarget], device=x_next.device, dtype=torch.long)
        synthetic = torch.cat((x_next_source[idxs, :84], x_next_source[donors, 84:]), dim=-1)
        target_latent[idxs] = frame_latent_fn(synthetic).to(dtype=target_latent.dtype)
    return x_next, style_label, target_latent, applied / float(batch)


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
        "style_hist_loss",
        "style_entropy",
        "selector_entropy",
        "conditioning_entropy",
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
        "decoupled_fraction",
        "history_noise_fraction",
        "history_mask_fraction",
        "history_style_fraction",
        "gt_target_fraction",
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


def history_noise_probability_for_epoch(epoch, *, mode, start_epoch, probability):
    """One-based stage boundary; resume uses the restored epoch, never a reset counter."""
    return probability if mode == "structured" and epoch >= start_epoch else 0.0


def controller_collate(samples):
    # traj/motion are views of x in the dataset, but default_collate copies both again.
    return default_collate([{key: sample[key] for key in ('x', 'label', 'start', 'segment_id')
                             if key in sample} for sample in samples])


def _run_epoch(
    *, model, loader, device, prediction_mode, recon_weight, style_hist_weight,
    style_entropy_weight, noise_std, prev_latent_noise_std, prev_motion_noise_std,
    prev_latent_dropout, prev_motion_dropout, solver_steps, optimizer,
    vq_target_cache, cache_vqvae, grad_clip, log_interval, epoch, total_epochs,
    split, decouple_prob=0.0, decouple_latent="matched_root", unroll_steps=1,
    style_consistency_weight=0.0, style_classifier=None, motion_mean=None,
    motion_std=None, joint_indices=None, style_consistency_window=1,
    classifier_label_map=None, root_recon_weight=0.0, microbatch_size=0,
    norm_mean=None, norm_std=None, max_batches=0,
    recipe_pool=None, recipe_probability=0.,
    train_feedback_context=False,
    history_noise_mode="legacy", history_noise_probability=0.0,
    history_noise_start_epoch=1, history_root_noise_std=0.05,
    donor_bank=None,
    history_style_probability=0., history_mask_probability=0., history_condition_warmup_epochs=5,
    flow_low_t_probability=0., flow_low_t_max=.2,
):
    is_train = optimizer is not None
    pure_recipe = getattr(model, 'conditioning_mode', 'label') in ('recipe', 'clip_recipe')
    learned_recipe = getattr(model, 'conditioning_mode', 'label') == 'clip_recipe'
    model.train(is_train)
    keys = _metric_keys(prediction_mode)
    effective_history_probability = history_noise_probability_for_epoch(
        epoch, mode=history_noise_mode, start_epoch=history_noise_start_epoch,
        probability=history_noise_probability) if is_train else 0.0
    sums = torch.zeros(len(keys), device=device, dtype=torch.float32)
    num_samples = 0
    group_sums = torch.zeros(4, device=device)
    donor_sums = {'donor_candidates_mean': 0., 'donor_phase_cost': 0.}
    donor_count = 0
    task_sums = torch.zeros(6, device=device)
    practical_history = is_train and (history_style_probability > 0 or history_mask_probability > 0)
    if practical_history and (unroll_steps != 1 or decouple_prob > 0 or train_feedback_context):
        raise ValueError('GT-target history augmentation requires single-step TF without target decoupling/feedback')
    ramp = min(1., epoch / max(history_condition_warmup_epochs, 1))
    if donor_bank is not None:
        # Fixed normalization belongs on device, not two synchronous H2D copies/batch.
        norm_mean = torch.as_tensor(norm_mean, device=device, dtype=torch.float32)
        norm_std = torch.as_tensor(norm_std, device=device, dtype=torch.float32)
    suffix_table = None
    dataset = getattr(loader, "dataset", None)
    while dataset is not None and not hasattr(dataset, "segments") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if dataset is not None and getattr(dataset, "segments", None):
        suffix_table = [str(s.get("suffix", "")) for s in dataset.segments]
    with tqdm(total=min(len(loader), max_batches) if max_batches else len(loader),
              desc=f"{split} {epoch}/{total_epochs}", unit="batch", dynamic_ncols=True) as progress:
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            x = batch["x"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True).long()
            # Cache indices are CPU memmap rows: keep frame IDs on CPU throughout lookup.
            starts = batch["start"].long()
            if x.ndim != 3 or x.size(1) < 2:
                raise ValueError("Training expects [B,T,375], T>=2")
            steps = int(unroll_steps) if is_train else 1
            if x.size(1) < steps + 1:
                raise ValueError("seq_len must be >= unroll_steps+1")
            segments = batch.get("segment_id")
            if steps == 1:
                repeats = x.size(1) - 1
                window = x if repeats == 1 else torch.stack((x[:, :-1], x[:, 1:]), dim=2).reshape(-1, 2, 375)
                starts = (starts[:, None] + torch.arange(repeats, device=starts.device)).reshape(-1)
                labels = labels.repeat_interleave(repeats)
                if segments is not None:
                    segments = segments.repeat_interleave(repeats)
            else:
                window = x[:, :steps + 1]
            ids = starts[:, None] + torch.arange(steps + 1, device=starts.device)
            if vq_target_cache is None:
                latents = model.frame_latent(window.reshape(-1, 375)).reshape(window.size(0), steps + 1, -1)
            else:
                latents = vq_target_cache.lookup_latents(ids.flatten(), vqvae=cache_vqvae,
                    device=device).reshape(window.size(0), steps + 1, -1)
            targets = latents[:, 1:]
            recon = window[:, 1:, 84:]
            fraction = 0.0
            donor_rows = list(range(window.size(0)))
            donor_entries = None
            original_labels = labels.clone()
            history_mixed = torch.zeros(window.size(0), device=device, dtype=torch.bool)
            history_mask = torch.zeros_like(history_mixed)
            if practical_history:
                if steps != 1 or window.size(1) != 2 or segments is None:
                    raise ValueError('History augmentation requires adjacent frames and segment metadata')
                gate = torch.rand(window.size(0), device=device)
                history_mask = gate < history_mask_probability * ramp
                requested_history = ((gate >= history_mask_probability * ramp)
                    & (gate < (history_mask_probability + history_style_probability) * ramp))
                if history_style_probability > 0:
                    if donor_bank is None:
                        raise ValueError('Style history augmentation requires a train-only donor bank')
                    current, prev, _, history_mixed, donor_entries = donor_bank.perturb_history(
                        current=window[:, 0], previous_latent=latents[:, 0], labels=labels,
                        segment_ids=segments, selected=requested_history, norm_mean=norm_mean,
                        norm_std=norm_std, frame_latent_fn=model.frame_latent)
                    window = window.clone(); window[:, 0] = current
                    latents = latents.clone(); latents[:, 0] = prev
                    # Missing alternate styles remain useful conditional-generation cases.
                    history_mask |= requested_history & ~history_mixed
            if is_train and decouple_prob > 0:
                if donor_bank is not None:
                    labels, targets, recon, changed, donor_entries, donor_stats = donor_bank.targets(
                        x_window=window, labels=labels, latents=latents, starts=starts,
                        segment_ids=segments, norm_mean=norm_mean, norm_std=norm_std,
                        frame_latent_fn=model.frame_latent, probability=decouple_prob)
                    changed_count = donor_stats.pop('_changed_count', None)
                    if changed_count is None:
                        changed_count = int(changed.sum())
                    fraction = changed_count / window.size(0)
                    donor_count += changed_count
                    for name, value in donor_stats.items():
                        donor_sums[name] += value * changed_count
                elif decouple_latent == "matched_root":
                    if suffix_table is None or segments is None or (segments < 0).any():
                        raise ValueError("matched_root requires segment_id and DB segment suffix metadata")
                    suffixes = [suffix_table[int(i)] for i in segments.cpu().tolist()]
                    matched = matched_root_targets(
                        x_window=window, labels=labels, latents=latents, suffixes=suffixes,
                        norm_mean=norm_mean, norm_std=norm_std, frame_latent_fn=model.frame_latent,
                        prob=decouple_prob, return_donors=pure_recipe)
                    labels, targets, recon, fraction = matched[:4]
                    if pure_recipe:
                        donor_rows = matched[4].cpu().tolist()
                else:
                    original = labels
                    labels, targets, recon, fraction = apply_style_decoupling_windows(
                        style_label=labels, target_latents=targets, recon_targets=recon, prob=decouple_prob)
                    if decouple_latent == "retarget":
                        mask = labels != original
                        synthetic = torch.cat((window[mask, 1:, :84], recon[mask]), dim=-1)
                        if mask.any():
                            targets = targets.clone()
                            targets[mask] = model.frame_latent(synthetic.reshape(-1, 375)).reshape(-1, steps, model.latent_dim)
            count = window.size(0)
            if is_train and train_feedback_context:
                # Frozen-VQ reconstructed GT conditions approximate the rendered feedback
                # distribution; supervision and donor construction remain GT/synthetic GT.
                with torch.no_grad():
                    observed_motion = model.vqvae.decode(latents[:, 0])
                window = window.clone()
                window[:, 0, 84:] = observed_motion
            manual_probs, recipe_fraction = None, 0.
            if pure_recipe or (is_train and recipe_probability > 0):
                if recipe_pool is None:
                    raise ValueError('Recipe conditioning requires an independent train-only exemplar pool')
                if dataset is None or segments is None:
                    raise ValueError('Recipe conditioning requires source segment metadata for exemplar isolation')
                source_ids = [dataset.segments[int(i)].get('source_entry_id', int(i)) for i in segments.cpu().tolist()]
                if pure_recipe:
                    exclusions = [{source_ids[i], donor_entries[i] if donor_entries is not None else source_ids[donor_rows[i]]} for i in range(count)]
                    manual_probs, recipe_fraction = recipe_pool.sample(labels, probability=1.,
                        per_sample_exclusions=exclusions, require_all=True)
                else:
                    manual_probs, recipe_fraction = recipe_pool.sample(labels, probability=recipe_probability,
                        excluded_source_ids=set(source_ids))
            micro = min(microbatch_size or count, count)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            values = torch.zeros(len(keys), device=device, dtype=torch.float32)
            for begin in range(0, count, micro):
                end = min(begin + micro, count)
                with torch.set_grad_enabled(is_train):
                    out = unrolled_flow_loss(
                        model=model, x_window=window[begin:end], style_label=None if pure_recipe else labels[begin:end],
                        prev_latent_0=latents[begin:end, 0], target_latents=targets[begin:end],
                        recon_targets=recon[begin:end], steps=steps, recon_weight=recon_weight,
                        root_recon_weight=root_recon_weight, style_hist_weight=style_hist_weight,
                        style_entropy_weight=style_entropy_weight, noise_std=noise_std,
                        prev_latent_noise_std=prev_latent_noise_std if is_train else 0.,
                        prev_motion_noise_std=prev_motion_noise_std if is_train else 0.,
                        prev_latent_dropout=prev_latent_dropout if is_train else 0.,
                        prev_motion_dropout=prev_motion_dropout if is_train else 0.,
                        solver_steps=solver_steps, requantize_feedback=model.inference_requantize,
                        latent_clip_feedback=model.inference_latent_clip,
                        style_consistency_weight=style_consistency_weight if is_train else 0.,
                        style_classifier=style_classifier, motion_mean=motion_mean, motion_std=motion_std,
                        joint_indices=joint_indices, classifier_label_map=classifier_label_map,
                        manual_style_probs=None if manual_probs is None or learned_recipe else manual_probs[begin:end],
                        clip_histograms=manual_probs[begin:end] if learned_recipe else None,
                        clip_histograms_validated=learned_recipe,
                        prior_labels=labels[begin:end] if learned_recipe else None,
                        return_per_sample=True,
                        history_noise_mode=history_noise_mode,
                        history_noise_probability=effective_history_probability,
                        history_root_noise_std=history_root_noise_std,
                        history_mask=history_mask[begin:end] if practical_history else None,
                        flow_low_t_probability=flow_low_t_probability if is_train else 0.,
                        flow_low_t_max=flow_low_t_max)
                    # Loss/targets are always GT; only observed history was corrupted.
                    if is_train:
                        (out["loss"] * ((end - begin) / count)).backward()
                cross = history_mixed[begin:end] if practical_history else labels[begin:end] != original_labels[begin:end]
                missing = history_mask[begin:end]
                clean = ~cross & ~missing
                per_sample = out.pop('_flow_per_sample').detach()
                group_sums += torch.stack((per_sample.masked_fill(~clean, 0).sum(), clean.sum(),
                                           per_sample.masked_fill(~cross, 0).sum(), cross.sum()))
                task_sums += torch.stack((per_sample.masked_fill(~clean, 0).sum(), clean.sum(),
                    per_sample.masked_fill(~history_mixed[begin:end], 0).sum(), history_mixed[begin:end].sum(),
                    per_sample.masked_fill(~missing, 0).sum(), missing.sum()))
                out["decoupled_fraction"] = out["loss"].new_tensor(fraction)
                out['history_style_fraction'] = history_mixed.float().mean()
                out['gt_target_fraction'] = out['loss'].new_tensor(1. - fraction)
                out['recipe_fraction'] = out['loss'].new_tensor(recipe_fraction)
                stats = getattr(recipe_pool, 'last_sample_stats', {}) if manual_probs is not None else {}
                out['recipe_exemplars_mean'] = out['loss'].new_tensor(stats.get('exemplars_mean', 0.))
                out['recipe_shortfall_fraction'] = out['loss'].new_tensor(stats.get('shortfall_fraction', 0.))
                out['recipe_bank_hit_fraction'] = out['loss'].new_tensor(stats.get('bank_hit_fraction', 0.))
                values += torch.stack([out[key].detach() for key in keys]) * (end - begin)
            if is_train:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    grad_clip if grad_clip > 0 else float("inf"), error_if_nonfinite=True)
                optimizer.step()
            sums += values
            num_samples += count
            progress.update(1)
            if (batch_idx + 1) % max(log_interval, 1) == 0:
                progress.set_postfix(loss=f"{float(values[0] / count):.4f}", unroll=steps)
    metrics = dict(zip(keys, (sums / max(num_samples, 1)).cpu().tolist()))
    if num_samples == 0:
        raise ValueError(f'{split} has no batches; check batch size and split/window lengths')
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f'Non-finite {split} metrics: {metrics}')
    metrics["num_samples"] = float(num_samples)
    matched_sum, matched_n, cross_sum, cross_n = group_sums.cpu().tolist()
    metrics.update(matched_flow_loss=matched_sum / max(matched_n, 1), cross_flow_loss=cross_sum / max(cross_n, 1),
                   matched_samples=matched_n, cross_samples=cross_n,
                   **{k: float(v / max(donor_count, 1)) for k, v in donor_sums.items()})
    tasks = task_sums.cpu().tolist()
    for i, name in enumerate(('clean_history', 'mixed_style_history', 'masked_history')):
        value, count = tasks[2*i:2*i+2]
        metrics[name + '_flow_loss'] = value / max(count, 1)
        metrics[name + '_samples'] = count
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
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Minimal flow-matching VQ controller trainer")
    parser.add_argument("--db", type=Path, default=script_dir / "database_100sty_375_memmap")
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=script_dir / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep" / "best.pt",
    )
    parser.add_argument("--vq-target-cache", type=Path, default=None)
    parser.add_argument("--norm-stats", type=Path, default=None)
    parser.add_argument("--latent-stats", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=script_dir / "checkpoints" / "flowmatch_tf50_fp32_norecon")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument("--microbatch-size", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument('--stop-after-epoch', type=int, default=0,
                        help='Stop at a saved epoch boundary; scheduler/total epoch budget remain unchanged')
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--allow-legacy-artifacts", action="store_true")
    parser.add_argument("--rollout-val-every", type=int, default=0)
    parser.add_argument('--tf-val-every', type=int, default=0, help='Fixed 240f style validation interval; historical flag name, see --style-val-mode')
    parser.add_argument('--style-val-mode', choices=['rollout','reconstruct'], default='rollout',
                        help='Motion/style validation and checkpoint selection; reconstruct is an explicit TF diagnostic')
    parser.add_argument('--tf-val-classifier', type=Path, default=script_dir / 'checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth')
    parser.add_argument('--tf-val-root-tolerance', type=float, default=1.15)
    parser.add_argument('--tf-val-selection', choices=['absolute', 'relative'], default='absolute')
    parser.add_argument('--tf-val-min-matched-accuracy', type=float, default=.85)
    parser.add_argument('--tf-val-max-matched-ade', type=float, default=5.)
    parser.add_argument('--tf-val-max-cross-ade', type=float, default=10.)
    parser.add_argument('--tf-val-reference', type=Path, default=None, help='Reuse the initial baseline metrics and adjacent TF case manifest across stages')
    parser.add_argument('--tf-val-clips', type=int, default=90)
    parser.add_argument('--tf-val-windows', type=int, default=3)
    parser.add_argument('--cross-donor-source', choices=['batch', 'bank'], default='batch')
    parser.add_argument('--donor-window', type=int, default=16)
    parser.add_argument('--donor-windows-per-clip', type=int, default=16)
    parser.add_argument("--rollout-val-classifier", type=Path, default=script_dir / "checkpoints/fid_feature_extractor_camdm100_mirror/100style_position_classifier.pth")

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
    parser.add_argument('--history-style-probability', type=float, default=0.,
                        help='Perturb previous body/latent with another train style; retain real GT endpoint and root/trajectory')
    parser.add_argument('--history-mask-probability', type=float, default=0.,
                        help='Jointly mask normalized previous body and latent; disjoint from style perturbations')
    parser.add_argument('--history-condition-warmup-epochs', type=int, default=5,
                        help='Linearly ramp style/mask probabilities; 0 enables full strength immediately')
    parser.add_argument('--flow-low-t-probability', type=float, default=0.,
                        help='Mixture fraction sampled from Uniform[0,flow-low-t-max]; remaining times cover [0,1]')
    parser.add_argument('--flow-low-t-max', type=float, default=.2)

    parser.add_argument("--recon-weight", type=float, default=0.0,
                        help="Motion reconstruction loss; 0 skips Euler integration and decoding when no other motion loss or feedback is needed")
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
        "--decouple-prob",
        type=float,
        default=0.0,
        help="Requested donor probability; actual qualified donor fraction is logged. 0 disables.",
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
        "--decouple-latent",
        type=str,
        choices=["donor", "retarget", "matched_root"],
        default="matched_root",
        help=("'matched_root' matches suffix/root, preserves source root and re-encodes corrected donor motion. "
              "'donor' and 'retarget' retain historical diagnostic behavior."),
    )
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
        default=script_dir / "checkpoints" / "fid_feature_extractor_camdm100_mirror_frame" / "100style_position_classifier.pth",
        help="Frozen frame-level (window=1) style classifier used by --style-consistency-weight",
    )
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--prev-latent-noise-std", type=float, default=0.3)
    parser.add_argument("--prev-motion-noise-std", type=float, default=0.3)
    parser.add_argument("--history-noise-mode", choices=["legacy", "structured"], default="legacy",
                        help="legacy keeps independent noise; structured shares sample gate/strength and preserves rotations")
    parser.add_argument("--history-noise-start-epoch", type=int, default=1,
                        help="First noisy epoch (one-based), used only in structured mode; 51 gives 50 clean epochs")
    parser.add_argument("--history-noise-probability", type=float, default=0.3,
                        help="Structured-mode per-sample probability, distinct from the noise amplitude")
    parser.add_argument("--history-root-noise-std", type=float, default=0.05,
                        help="Maximum normalized root noise amplitude in structured mode")
    parser.add_argument("--prev-latent-dropout", type=float, default=0.0)
    parser.add_argument("--prev-motion-dropout", type=float, default=0.0)
    parser.add_argument("--solver-steps", type=int, default=8)
    parser.add_argument("--latent-stats-chunk-rows", type=int, default=65536)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch < 0:
        raise ValueError('stop-after-epoch must be nonnegative')
    if (not 0 <= args.history_style_probability <= 1 or not 0 <= args.history_mask_probability <= 1
            or args.history_style_probability + args.history_mask_probability > 1
            or args.history_condition_warmup_epochs < 0
            or not 0 <= args.flow_low_t_probability < 1 or not 0 < args.flow_low_t_max <= 1):
        raise ValueError('Invalid practical-history/flow-time probabilities')
    if args.history_style_probability or args.history_mask_probability:
        if (args.seq_len != 2 or args.unroll_steps != 1 or args.decouple_prob > 0 or args.train_feedback_context
                or (args.history_style_probability > 0 and args.cross_donor_source != 'bank')):
            raise ValueError('Practical history requires seq_len=2/unroll=1, GT targets and a bank for style perturbations')
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
    if not 0 <= args.decouple_prob <= 1 or args.microbatch_size < 0:
        raise ValueError("Invalid decouple probability or microbatch size")
    noise_values = (args.prev_latent_noise_std, args.prev_motion_noise_std,
                    args.history_root_noise_std, args.history_noise_probability)
    if (not all(np.isfinite(v) and v >= 0 for v in noise_values)
            or args.history_noise_probability > 1 or args.history_noise_start_epoch < 1):
        raise ValueError("Invalid history noise parameters")
    args.implementation_version = IMPLEMENTATION_VERSION
    if args.conditioning_mode in ('recipe', 'clip_recipe'):
        if not args.use_style_code_selector:
            raise ValueError('Recipe mode requires --use-style-code-selector')
        if args.style_consistency_weight > 0 or args.rollout_val_every > 0:
            raise ValueError('Recipe mode uses direct FM supervision; classifier loss and label rollout validation must be disabled')
        if args.decouple_prob > 0 and args.decouple_latent != 'matched_root':
            raise ValueError('Recipe mode supports matched_root cross supervision')
        args.recipe_prob = 1.
        if args.conditioning_mode == 'recipe':
            args.style_hist_weight = args.style_entropy_weight = 0.
    if args.cross_donor_source == 'bank' and (args.unroll_steps != 1 or args.seq_len != 2
            or args.decouple_latent != 'matched_root' or args.conditioning_mode == 'label'):
        raise ValueError('Donor bank requires recipe/clip_recipe, matched_root and seq_len=2/unroll=1')
    if args.donor_window < 2 or args.donor_windows_per_clip < 1 or args.recipe_bank_max_entries < 1:
        raise ValueError('Invalid donor/recipe bank dimensions')
    if args.tf_val_every < 0 or args.tf_val_root_tolerance < 1 or min(args.tf_val_clips, args.tf_val_windows) < 1:
        raise ValueError('Invalid teacher-forced validation settings')
    if args.tf_val_every and (args.conditioning_mode == 'label' or args.rollout_val_every or not args.tf_val_classifier.is_file()):
        raise ValueError('TF style validation requires recipe mode, its classifier and rollout validation disabled')
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
    args.training_objective_version = (3 if args.use_style_code_selector and
        (args.recipe_prob > 0 or args.style_proto_normalization != 'layernorm' or args.style_selector_init != 'random')
        else IMPLEMENTATION_VERSION)
    if args.train_feedback_context:
        args.training_objective_version = 4
    if args.history_noise_mode == "structured":
        args.training_objective_version = 5
    if args.conditioning_mode == 'recipe':
        args.training_objective_version = 6
    if args.conditioning_mode == 'clip_recipe' or args.cross_donor_source == 'bank':
        args.training_objective_version = 7
    if args.history_style_probability or args.history_mask_probability or args.flow_low_t_probability:
        args.training_objective_version = 8
    if args.lr_schedule_epochs < 0 or (args.lr_schedule_epochs and args.lr_schedule_epochs < args.epochs):
        raise ValueError('Cosine horizon must be 0 or at least the requested training epochs')
    if not args.resume and ((args.out_dir / "config.json").exists() or any(args.out_dir.glob("*.pt"))):
        raise ValueError("Training output already contains a run; use --resume or a new output directory")

    validate_memmap_db_dir(args.db)
    if not args.vqvae_ckpt.exists():
        raise FileNotFoundError(f"VQ checkpoint not found: {args.vqvae_ckpt}")
    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")
    if args.solver_steps < 1:
        raise ValueError("--solver-steps must be >= 1")
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
            loader.collate_fn = controller_collate
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
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay, fused=args.fused_adamw)
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
        missing, unexpected = model.load_state_dict(initial_state, strict=False)
        allowed = {k for k in model.state_dict() if k.startswith('histogram_selector.')} if prior_args.get('conditioning_mode') != 'clip_recipe' else set()
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
        resume_fields = ("prediction_mode", "flow_loss_space", "seq_len", "unroll_steps", "decouple_latent",
            "decouple_prob", "root_recon_weight", "recon_weight", "style_hist_weight", "style_entropy_weight",
            "style_consistency_weight", "inference_requantize", "inference_latent_clip", "solver_steps",
            "noise_std", "prev_latent_noise_std", "prev_motion_noise_std", "prev_latent_dropout",
            "prev_motion_dropout", "seed", "batch_size", "microbatch_size", "lr", "min_lr", "weight_decay",
            "epochs", "dropout", "matmul_precision", "tf32", "cudnn_benchmark", "grad_clip",
            "stride", "style_slot_indices", "style_prior_loss", "style_selector_topk",
            "style_selector_temperature", "target_time_eps", "rollout_val_every", "val_every",
            "max_train_batches", "max_val_batches", "enforce_same_label", "val_ratio", "test_ratio",
            'style_proto_normalization', 'style_selector_init', 'recipe_prob', 'recipe_frames', 'recipe_exemplars',
            'lr_schedule_epochs', 'train_feedback_context', 'history_noise_mode',
            'history_noise_start_epoch', 'history_noise_probability', 'history_root_noise_std')
        resume_fields += ('conditioning_mode', 'recipe_exemplar_choices', 'recipe_bank_size')
        resume_fields += ('cross_donor_source', 'donor_window', 'donor_windows_per_clip', 'recipe_bank_max_entries',
                          'tf_val_every', 'tf_val_root_tolerance', 'tf_val_clips', 'tf_val_windows')
        resume_fields += ('tf_val_selection', 'tf_val_min_matched_accuracy', 'tf_val_max_matched_ade', 'tf_val_max_cross_ade')
        if args.tf_val_every:
            resume_fields += ('style_val_mode',)
        resume_fields += ('history_style_probability', 'history_mask_probability', 'history_condition_warmup_epochs',
                          'flow_low_t_probability', 'flow_low_t_max')
        legacy_defaults = {'style_proto_normalization': 'layernorm', 'style_selector_init': 'random',
                           'recipe_prob': 0., 'recipe_frames': 240, 'recipe_exemplars': 5, 'lr_schedule_epochs': 0,
                           'train_feedback_context': False, 'history_noise_mode': 'legacy',
                           'history_noise_start_epoch': 1, 'history_noise_probability': .3,
                           'history_root_noise_std': .05, 'conditioning_mode': 'label', 'recipe_exemplar_choices': None,
                           'recipe_bank_size': 0, 'cross_donor_source': 'batch', 'donor_window': 16,
                           'donor_windows_per_clip': 16, 'recipe_bank_max_entries': 131072, 'tf_val_every': 0,
                           'tf_val_root_tolerance': 1.15, 'tf_val_clips': 90, 'tf_val_windows': 3}
        legacy_defaults.update(tf_val_selection='absolute', tf_val_min_matched_accuracy=.85,
                               tf_val_max_matched_ade=5., tf_val_max_cross_ade=10.)
        legacy_defaults['style_val_mode'] = 'reconstruct'
        legacy_defaults.update(history_style_probability=0., history_mask_probability=0.,
                               history_condition_warmup_epochs=5, flow_low_t_probability=0., flow_low_t_max=.2)
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
        from Utils.style_code_prior import TrainStyleRecipePool
        recipe_pool = TrainStyleRecipePool(db=args.db, model=model, cache=vq_target_cache,
            mean=bundle.norm_mean, std=bundle.norm_std, frames=args.recipe_frames, exemplars=args.recipe_exemplars,
            exemplar_choices=args.recipe_exemplar_choices, bank_size=args.recipe_bank_size, seed=args.seed,
            bank_max_entries=args.recipe_bank_max_entries)
        print(f'[Info] train recipes   : p={args.recipe_prob}, choices={recipe_pool.exemplar_choices} clips × {args.recipe_frames} frames')

    donor_bank = None
    if args.cross_donor_source == 'bank':
        from Utils.style_donor_bank import StyleDonorBank
        donor_bank = StyleDonorBank(args.db, window=args.donor_window,
            windows_per_clip=args.donor_windows_per_clip, device=device)
        print(f'[Info] donor bank      : {len(donor_bank.starts)} train-only windows')
    tf_validator = None
    if args.tf_val_every:
        from Utils.controller_validation import TeacherForcedStyleValidation
        with isolated_rng(args.seed + 1002):
            tf_validator = TeacherForcedStyleValidation(model, db=args.db, mean=bundle.norm_mean,
                std=bundle.norm_std, classifier_path=args.tf_val_classifier, seed=args.seed,
                clips=args.tf_val_clips, windows=args.tf_val_windows, out_dir=args.out_dir,
                eval_mode=args.style_val_mode)
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
        train_metrics = _run_epoch(
            model=model,
            loader=bundle.train_loader,
            device=device,
            prediction_mode=model.prediction_mode,
            recon_weight=args.recon_weight,
            style_hist_weight=args.style_hist_weight,
            style_entropy_weight=args.style_entropy_weight,
            noise_std=args.noise_std,
            prev_latent_noise_std=args.prev_latent_noise_std,
            prev_motion_noise_std=args.prev_motion_noise_std,
            prev_latent_dropout=args.prev_latent_dropout,
            prev_motion_dropout=args.prev_motion_dropout,
            solver_steps=args.solver_steps,
            optimizer=optimizer,
            vq_target_cache=vq_target_cache,
            cache_vqvae=vq_model,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            total_epochs=args.epochs,
            decouple_prob=args.decouple_prob,
            decouple_latent=args.decouple_latent,
            unroll_steps=args.unroll_steps,
            style_consistency_weight=args.style_consistency_weight,
            style_classifier=style_classifier,
            motion_mean=style_consistency_motion_mean,
            motion_std=style_consistency_motion_std,
            joint_indices=style_consistency_joint_indices,
            split="train", root_recon_weight=args.root_recon_weight,
            microbatch_size=args.microbatch_size, norm_mean=bundle.norm_mean, norm_std=bundle.norm_std,
            classifier_label_map=classifier_label_map, max_batches=args.max_train_batches,
            recipe_pool=recipe_pool, recipe_probability=args.recipe_prob,
            train_feedback_context=args.train_feedback_context,
            history_noise_mode=args.history_noise_mode,
            history_noise_start_epoch=args.history_noise_start_epoch,
            history_noise_probability=args.history_noise_probability,
            history_root_noise_std=args.history_root_noise_std,
            donor_bank=donor_bank,
            history_style_probability=args.history_style_probability,
            history_mask_probability=args.history_mask_probability,
            history_condition_warmup_epochs=args.history_condition_warmup_epochs,
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
                    style_hist_weight=args.style_hist_weight,
                    style_entropy_weight=args.style_entropy_weight,
                    noise_std=args.noise_std,
                    prev_latent_noise_std=0.0,
                    prev_motion_noise_std=0.0,
                    prev_latent_dropout=0.0,
                    prev_motion_dropout=0.0,
                    solver_steps=args.solver_steps,
                    optimizer=None,
                    decouple_prob=0.0,
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
            "history_augmentation": {
                "mode": args.history_noise_mode,
                "start_epoch": args.history_noise_start_epoch,
                "probability": history_noise_probability_for_epoch(epoch,
                    mode=args.history_noise_mode, start_epoch=args.history_noise_start_epoch,
                    probability=args.history_noise_probability),
                "actual_fraction": train_metrics["history_noise_fraction"],
                "style_probability": args.history_style_probability,
                "mask_probability": args.history_mask_probability,
                "style_actual": train_metrics['history_style_fraction'],
                "mask_actual": train_metrics['history_mask_fraction'],
                "gt_target_fraction": train_metrics['gt_target_fraction'],
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
                    'policy': args.tf_val_selection,
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
