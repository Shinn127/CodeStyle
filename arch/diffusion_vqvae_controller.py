from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .basic_vqvae_375to291 import BasicVQVAE375to291
from .camdm_controller import get_named_beta_schedule
from .flowmatch_vqvae_controller import FlowMatchVQVAEController


class DiffusionVQVAEController(FlowMatchVQVAEController):
    def __init__(
        self,
        *,
        vqvae: BasicVQVAE375to291,
        num_styles: int,
        use_style_code_selector: bool = False,
        selected_style_slots: Sequence[int] | None = None,
        style_dim: int = 64,
        style_selector_hidden_dim: int = 128,
        style_selector_topk: int = 4,
        style_selector_temperature: float = 1.0,
        time_embed_dim: int = 64,
        flow_hidden_dim: int = 256,
        flow_layers: int = 2,
        flow_heads: int = 4,
        flow_ffn_dim: int = 1024,
        dropout: float = 0.1,
        target_time_eps: float = 1e-3,
        diffusion_steps: int = 16,
        beta_schedule: str = "cosine",
    ):
        super().__init__(
            vqvae=vqvae,
            num_styles=num_styles,
            use_style_code_selector=use_style_code_selector,
            selected_style_slots=selected_style_slots,
            style_dim=style_dim,
            style_selector_hidden_dim=style_selector_hidden_dim,
            style_selector_topk=style_selector_topk,
            style_selector_temperature=style_selector_temperature,
            time_embed_dim=time_embed_dim,
            flow_hidden_dim=flow_hidden_dim,
            flow_layers=flow_layers,
            flow_heads=flow_heads,
            flow_ffn_dim=flow_ffn_dim,
            dropout=dropout,
            prediction_mode="velocity",
            target_time_eps=target_time_eps,
        )
        if diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1, got {diffusion_steps}")
        self.diffusion_steps = int(diffusion_steps)
        self.beta_schedule = str(beta_schedule)
        self.prediction_target = "x0"

        betas_np = get_named_beta_schedule(self.beta_schedule, self.diffusion_steps)
        betas = torch.from_numpy(np.asarray(betas_np, dtype=np.float32))
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    def _extract_alpha_bar(self, timesteps: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        t = timesteps.reshape(-1).long()
        return self.alpha_bars.index_select(0, t).to(device=like.device, dtype=like.dtype).unsqueeze(-1)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_bar = self._extract_alpha_bar(timesteps, like=x_start)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1.0 - alpha_bar) * noise

    def _time_condition(self, timesteps: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        denom = max(self.diffusion_steps - 1, 1)
        return timesteps.to(device=like.device, dtype=like.dtype).reshape(-1, 1) / float(denom)

    def predict_x0(
        self,
        *,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        timesteps: torch.Tensor,
        e_c: torch.Tensor,
        e_traj: torch.Tensor,
        e_s: torch.Tensor,
    ) -> torch.Tensor:
        return self.field(
            z_t=z_t,
            z_prev=z_prev,
            t=self._time_condition(timesteps, like=z_t),
            content=e_c,
            traj=e_traj,
            style=e_s,
        )

    def _resolve_sampling_timesteps(self, num_steps: int) -> list[int]:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if num_steps > self.diffusion_steps:
            raise ValueError(
                f"num_steps must be <= trained diffusion_steps={self.diffusion_steps}, got {num_steps}"
            )
        timeline = np.linspace(self.diffusion_steps - 1, 0, num_steps, dtype=np.float64)
        timesteps = np.rint(timeline).astype(np.int64).tolist()
        for idx in range(1, len(timesteps)):
            timesteps[idx] = min(timesteps[idx], timesteps[idx - 1] - 1)
        timesteps[-1] = 0
        if any(step < 0 for step in timesteps):
            raise RuntimeError(f"Invalid diffusion sampling schedule resolved from num_steps={num_steps}: {timesteps}")
        return timesteps

    def compute_loss(
        self,
        *,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        style_label: torch.Tensor,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        prev_latent: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        diffusion_weight: float = 1.0,
        recon_weight: float = 0.0,
        style_hist_weight: float = 0.0,
        style_entropy_weight: float = 0.0,
        prev_latent_noise_std: float = 0.0,
        prev_motion_noise_std: float = 0.0,
        prev_latent_dropout: float = 0.0,
        prev_motion_dropout: float = 0.0,
        sample_steps: int = 16,
        sample_noise_std: float = 1.0,
        return_prediction: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        x_next = self._flatten_frames(x_next, self.input_dim)
        traj_next, motion_next = self.split_traj_motion(x_next)
        x_curr_for_conditions = x_curr
        if self.training and prev_motion_noise_std > 0.0:
            traj_curr, motion_curr = self.split_traj_motion(x_curr)
            sample_scale = torch.rand((motion_curr.size(0), 1), device=motion_curr.device, dtype=motion_curr.dtype)
            motion_curr = motion_curr + (
                float(prev_motion_noise_std) * sample_scale * torch.randn_like(motion_curr)
            )
            x_curr_for_conditions = torch.cat((traj_curr, motion_curr), dim=-1)
        if self.training and prev_motion_dropout > 0.0:
            traj_curr, motion_curr = self.split_traj_motion(x_curr_for_conditions)
            motion_curr = F.dropout(motion_curr, p=float(prev_motion_dropout), training=True)
            x_curr_for_conditions = torch.cat((traj_curr, motion_curr), dim=-1)

        e_c, e_traj, e_s, style_details = self.encode_conditions(
            x_curr=x_curr_for_conditions,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            return_style_details=True,
        )

        with torch.no_grad():
            z_prev_raw = self.frame_latent(x_curr) if prev_latent is None else prev_latent.float()
            z1_raw = self.frame_latent(x_next) if target_latent is None else target_latent.float()

        z_prev = self.normalize_latent(z_prev_raw)
        z1 = self.normalize_latent(z1_raw)
        if self.training and prev_latent_noise_std > 0.0:
            sample_scale = torch.rand((z_prev.size(0), 1), device=z_prev.device, dtype=z_prev.dtype)
            z_prev = z_prev + sample_scale * self.sample_transport_noise_like(z_prev, noise_std=prev_latent_noise_std)
        if self.training and prev_latent_dropout > 0.0:
            z_prev = F.dropout(z_prev, p=float(prev_latent_dropout), training=True)

        timesteps = torch.randint(0, self.diffusion_steps, (z1.size(0),), device=z1.device)
        noise = torch.randn_like(z1)
        z_t = self.q_sample(z1, timesteps, noise)
        x0_pred = self.predict_x0(
            z_t=z_t,
            z_prev=z_prev,
            timesteps=timesteps,
            e_c=e_c,
            e_traj=e_traj,
            e_s=e_s,
        )

        diffusion_loss = F.mse_loss(x0_pred, z1)
        style_hist_loss = diffusion_loss.new_zeros(())
        style_entropy_loss = diffusion_loss.new_zeros(())
        style_top1_prob = diffusion_loss.new_zeros(())
        if self.use_style_code_selector:
            selector_probs = style_details["style_probs"]
            if style_hist_weight > 0.0 and not bool(self.style_prior_ready.item()):
                raise RuntimeError("Style code prior is required when style_hist_weight > 0.")
            style_hist_loss, style_entropy_loss, style_top1_prob = self._compute_style_regularizers(
                style_label=style_label,
                style_probs=selector_probs,
            )

        recon_loss = z1.new_zeros(())
        z_hat = None
        motion_hat = None
        if recon_weight > 0.0 or return_prediction:
            z_hat = self.sample_latent(
                x_curr=x_curr,
                style_label=style_label,
                manual_style_probs=manual_style_probs,
                manual_style_strength=manual_style_strength,
                prev_latent=z_prev_raw,
                num_steps=sample_steps,
                noise_std=sample_noise_std,
            )
            motion_hat = self.vqvae.decode(z_hat, from_indices=False)
        if recon_weight > 0.0:
            recon_loss = F.l1_loss(motion_hat, motion_next)

        total = (
            float(diffusion_weight) * diffusion_loss
            + float(recon_weight) * recon_loss
            + float(style_hist_weight) * style_hist_loss
            + float(style_entropy_weight) * style_entropy_loss
        )
        result = {
            "loss": total,
            "diffusion_loss": diffusion_loss,
            "recon_loss": recon_loss,
            "style_hist_loss": style_hist_loss,
            "style_entropy": style_entropy_loss,
            "style_top1_prob": style_top1_prob,
            "latent_target_norm": z1.norm(dim=-1).mean(),
            "noisy_latent_norm": z_t.norm(dim=-1).mean(),
            "pred_x0_norm": x0_pred.norm(dim=-1).mean(),
        }
        if return_prediction:
            if z_hat is None or motion_hat is None:
                raise RuntimeError("Prediction tensors were not computed.")
            result.update(
                {
                    "pred_latent": z_hat,
                    "pred_motion": motion_hat,
                    "pred_frame": torch.cat((traj_next, motion_hat), dim=-1),
                }
            )
        return result

    @torch.no_grad()
    def sample_latent(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        prev_latent: torch.Tensor | None = None,
        num_steps: int = 16,
        noise_std: float = 1.0,
        zT: torch.Tensor | None = None,
        latent_clip: bool = True,
    ) -> torch.Tensor:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        e_c, e_traj, e_s = self.encode_conditions(
            x_curr=x_curr,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
        )
        if prev_latent is None:
            z_prev = self.normalize_latent(self.frame_latent(x_curr))
        else:
            prev_latent = prev_latent.to(device=x_curr.device, dtype=torch.float32)
            if prev_latent.shape != (x_curr.size(0), self.latent_dim):
                raise ValueError(
                    f"Expected prev_latent shape {(x_curr.size(0), self.latent_dim)}, got {tuple(prev_latent.shape)}"
                )
            z_prev = self.normalize_latent(prev_latent)

        if zT is None:
            z = float(noise_std) * torch.randn((x_curr.size(0), self.latent_dim), device=x_curr.device, dtype=z_prev.dtype)
        else:
            z = self.normalize_latent(zT.to(device=x_curr.device, dtype=torch.float32))

        timesteps = self._resolve_sampling_timesteps(num_steps)
        for idx, step_idx in enumerate(timesteps):
            t_tensor = torch.full((x_curr.size(0),), int(step_idx), device=x_curr.device, dtype=torch.long)
            x0_pred = self.predict_x0(
                z_t=z,
                z_prev=z_prev,
                timesteps=t_tensor,
                e_c=e_c,
                e_traj=e_traj,
                e_s=e_s,
            )
            alpha_t = self.alpha_bars[step_idx].to(device=z.device, dtype=z.dtype)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp_min(1e-8))
            eps_pred = (z - sqrt_alpha_t * x0_pred) / sqrt_one_minus_alpha_t

            next_step = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
            if next_step >= 0:
                alpha_next = self.alpha_bars[next_step].to(device=z.device, dtype=z.dtype)
            else:
                alpha_next = torch.ones((), device=z.device, dtype=z.dtype)
            z = torch.sqrt(alpha_next) * x0_pred + torch.sqrt((1.0 - alpha_next).clamp_min(1e-8)) * eps_pred

        z_hat = self.denormalize_latent(z)
        if latent_clip:
            z_hat = self.clip_latent_to_data_bounds(z_hat)
        return z_hat

    @torch.no_grad()
    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor | None,
        manual_style_probs: torch.Tensor | None = None,
        manual_style_strength: float = 1.0,
        prev_latent: torch.Tensor | None = None,
        num_steps: int = 16,
        noise_std: float = 1.0,
        zT: torch.Tensor | None = None,
        latent_clip: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_hat = self.sample_latent(
            x_curr=x_curr,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            prev_latent=prev_latent,
            num_steps=num_steps,
            noise_std=noise_std,
            zT=zT,
            latent_clip=latent_clip,
        )
        motion_hat = self.vqvae.decode(z_hat, from_indices=False)
        return motion_hat, z_hat


def build_diffusion_vqvae_controller(
    *,
    vqvae: BasicVQVAE375to291,
    num_styles: int,
    use_style_code_selector: bool = False,
    selected_style_slots: Sequence[int] | None = None,
    style_dim: int = 64,
    style_selector_hidden_dim: int = 128,
    style_selector_topk: int = 4,
    style_selector_temperature: float = 1.0,
    time_embed_dim: int = 64,
    flow_hidden_dim: int = 256,
    flow_layers: int = 2,
    flow_heads: int = 4,
    flow_ffn_dim: int = 1024,
    dropout: float = 0.1,
    target_time_eps: float = 1e-3,
    diffusion_steps: int = 16,
    beta_schedule: str = "cosine",
) -> DiffusionVQVAEController:
    return DiffusionVQVAEController(
        vqvae=vqvae,
        num_styles=num_styles,
        use_style_code_selector=use_style_code_selector,
        selected_style_slots=selected_style_slots,
        style_dim=style_dim,
        style_selector_hidden_dim=style_selector_hidden_dim,
        style_selector_topk=style_selector_topk,
        style_selector_temperature=style_selector_temperature,
        time_embed_dim=time_embed_dim,
        flow_hidden_dim=flow_hidden_dim,
        flow_layers=flow_layers,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        dropout=dropout,
        target_time_eps=target_time_eps,
        diffusion_steps=diffusion_steps,
        beta_schedule=beta_schedule,
    )


def build_diffusion_vqvae_controller_from_config(
    config: Mapping[str, object],
    *,
    vqvae: BasicVQVAE375to291,
    num_styles: int,
) -> DiffusionVQVAEController:
    raw_style_slots = config.get("style_slot_indices", None)
    selected_style_slots = None
    if raw_style_slots not in (None, (), []):
        selected_style_slots = tuple(int(slot) for slot in raw_style_slots)
    return build_diffusion_vqvae_controller(
        vqvae=vqvae,
        num_styles=num_styles,
        use_style_code_selector=bool(config.get("use_style_code_selector", False)),
        selected_style_slots=selected_style_slots,
        style_dim=int(config.get("style_dim", 64)),
        style_selector_hidden_dim=int(config.get("style_selector_hidden_dim", 128)),
        style_selector_topk=int(config.get("style_selector_topk", 4)),
        style_selector_temperature=float(config.get("style_selector_temperature", 1.0)),
        time_embed_dim=int(config.get("time_embed_dim", 64)),
        flow_hidden_dim=int(config.get("flow_hidden_dim", config.get("hidden_dim", 256))),
        flow_layers=int(config.get("flow_layers", config.get("num_layers", 2))),
        flow_heads=int(config.get("flow_heads", 4)),
        flow_ffn_dim=int(config.get("flow_ffn_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        target_time_eps=float(config.get("target_time_eps", 1e-3)),
        diffusion_steps=int(config.get("diffusion_steps", 16)),
        beta_schedule=str(config.get("beta_schedule", "cosine")),
    )


__all__ = [
    "DiffusionVQVAEController",
    "build_diffusion_vqvae_controller",
    "build_diffusion_vqvae_controller_from_config",
]
