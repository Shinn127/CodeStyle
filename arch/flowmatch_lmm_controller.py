from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flowmatch_vqvae_controller import (
    ConditionalVelocityField,
    FlowPredictionMode,
    StyleEncoder,
    resolve_flowmatch_prediction_mode,
)
from .lmm_375to291 import LMM375to291


class FlowMatchLMMController(nn.Module):
    def __init__(
        self,
        *,
        lmm: LMM375to291,
        num_styles: int,
        style_dim: int = 64,
        time_embed_dim: int = 64,
        flow_hidden_dim: int = 256,
        flow_layers: int = 2,
        flow_heads: int = 4,
        flow_ffn_dim: int = 1024,
        dropout: float = 0.1,
        prediction_mode: FlowPredictionMode = "velocity",
        target_time_eps: float = 1e-3,
    ):
        super().__init__()
        self.lmm = lmm
        self.lmm.eval()
        for parameter in self.lmm.parameters():
            parameter.requires_grad_(False)

        self.input_dim = int(lmm.input_dim)
        self.traj_dim = int(lmm.traj_dim)
        self.motion_dim = int(lmm.motion_dim)
        self.latent_dim = int(lmm.latent_dim)
        self.hidden_dim = int(flow_hidden_dim)
        self.prediction_mode = resolve_flowmatch_prediction_mode(prediction_mode)
        self.target_time_eps = float(target_time_eps)

        self.style_encoder = StyleEncoder(num_styles=num_styles, style_dim=style_dim)
        self.field = ConditionalVelocityField(
            latent_dim=self.latent_dim,
            motion_dim=self.motion_dim,
            traj_dim=self.traj_dim,
            style_dim=style_dim,
            hidden_dim=self.hidden_dim,
            time_embed_dim=time_embed_dim,
            num_heads=flow_heads,
            ff_hidden_dim=flow_ffn_dim,
            num_layers=flow_layers,
            dropout=dropout,
        )

        self.register_buffer("latent_offset", torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_scale", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_transport_std", torch.ones(self.latent_dim, dtype=torch.float32))
        self.register_buffer("latent_data_min", torch.full((self.latent_dim,), -float("inf"), dtype=torch.float32))
        self.register_buffer("latent_data_max", torch.full((self.latent_dim,), float("inf"), dtype=torch.float32))
        self.register_buffer("latent_stats_ready", torch.tensor(False, dtype=torch.bool))

    def train(self, mode: bool = True):
        super().train(mode)
        self.lmm.eval()
        return self

    def set_latent_prior_stats(
        self,
        offset: torch.Tensor,
        scale: torch.Tensor,
        *,
        transport_std: torch.Tensor | None = None,
        data_min: torch.Tensor | None = None,
        data_max: torch.Tensor | None = None,
    ) -> None:
        offset = torch.as_tensor(offset, dtype=torch.float32, device=self.latent_offset.device).reshape(-1)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.latent_scale.device).reshape(-1)
        if offset.shape != (self.latent_dim,) or scale.shape != (self.latent_dim,):
            raise ValueError("Invalid latent stats shape")
        if transport_std is None:
            transport_std_t = torch.ones(self.latent_dim, dtype=torch.float32, device=self.latent_transport_std.device)
        else:
            transport_std_t = torch.as_tensor(
                transport_std,
                dtype=torch.float32,
                device=self.latent_transport_std.device,
            ).reshape(-1)
        if transport_std_t.shape != (self.latent_dim,):
            raise ValueError("Invalid latent transport_std shape")
        if data_min is None:
            data_min_t = torch.full((self.latent_dim,), -float("inf"), dtype=torch.float32, device=self.latent_data_min.device)
        else:
            data_min_t = torch.as_tensor(data_min, dtype=torch.float32, device=self.latent_data_min.device).reshape(-1)
        if data_max is None:
            data_max_t = torch.full((self.latent_dim,), float("inf"), dtype=torch.float32, device=self.latent_data_max.device)
        else:
            data_max_t = torch.as_tensor(data_max, dtype=torch.float32, device=self.latent_data_max.device).reshape(-1)
        if data_min_t.shape != (self.latent_dim,) or data_max_t.shape != (self.latent_dim,):
            raise ValueError("Invalid latent min/max stats shape")
        self.latent_offset.copy_(offset)
        self.latent_scale.copy_(scale.clamp_min(1e-6))
        self.latent_transport_std.copy_(transport_std_t.clamp_min(1e-6))
        self.latent_data_min.copy_(data_min_t)
        self.latent_data_max.copy_(data_max_t)
        self.latent_stats_ready.fill_(True)

    def _require_latent_stats(self) -> None:
        if not bool(self.latent_stats_ready.item()):
            raise RuntimeError("Call set_latent_prior_stats(...) before training or sampling.")

    def normalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        offset = self.latent_offset.to(device=latents.device, dtype=latents.dtype)
        scale = self.latent_scale.to(device=latents.device, dtype=latents.dtype)
        return (latents - offset) / scale

    def denormalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        offset = self.latent_offset.to(device=latents.device, dtype=latents.dtype)
        scale = self.latent_scale.to(device=latents.device, dtype=latents.dtype)
        return offset + scale * latents

    def transport_normalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        transport_std = self.latent_transport_std.to(device=latents.device, dtype=latents.dtype)
        return latents / transport_std

    def transport_denormalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        self._require_latent_stats()
        transport_std = self.latent_transport_std.to(device=latents.device, dtype=latents.dtype)
        return transport_std * latents

    def sample_transport_noise_like(self, reference: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
        return float(noise_std) * self.transport_denormalize_latent(torch.randn_like(reference))

    def clip_latent_to_data_bounds(self, latents: torch.Tensor) -> torch.Tensor:
        latents_norm = self.normalize_latent(latents.to(dtype=torch.float32))
        data_min = self.latent_data_min.to(device=latents.device, dtype=latents_norm.dtype)
        data_max = self.latent_data_max.to(device=latents.device, dtype=latents_norm.dtype)
        latents_norm = latents_norm.clamp(min=data_min, max=data_max)
        return self.denormalize_latent(latents_norm).to(dtype=latents.dtype)

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    def _flatten_frames(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.reshape(-1, dim)
        raise ValueError(f"Expected rank 2 or 3 input, got shape={tuple(x.shape)}")

    def _expand_style(self, style_label: torch.Tensor, n: int) -> torch.Tensor:
        style_label = style_label.reshape(-1)
        if style_label.numel() == n:
            return style_label.long()
        if style_label.numel() == 1:
            return style_label.long().expand(n)
        if n % style_label.numel() == 0:
            repeat = n // style_label.numel()
            return style_label.long().repeat_interleave(repeat)
        raise ValueError(f"Cannot align {style_label.numel()} style labels with batch size {n}")

    def encode_conditions(
        self,
        *,
        x_curr: torch.Tensor,
        style_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        style_label = self._expand_style(style_label.to(device=x_curr.device), x_curr.size(0))
        traj_curr, motion_curr = self.split_traj_motion(x_curr)
        return motion_curr, traj_curr, self.style_encoder(style_label)

    @torch.no_grad()
    def frame_latent(self, x_frame: torch.Tensor) -> torch.Tensor:
        x_frame = self._flatten_frames(x_frame, self.input_dim)
        _, motion = self.split_traj_motion(x_frame)
        return self.lmm.encode(motion.float())

    def decode_motion(self, *, traj: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.lmm.decode(traj=traj, latent=latent)

    def _predict_velocity(
        self,
        *,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        t: torch.Tensor,
        e_c: torch.Tensor,
        e_traj: torch.Tensor,
        e_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field_out = self.field(
            z_t=z_t,
            z_prev=z_prev,
            t=t,
            content=e_c,
            traj=e_traj,
            style=e_s,
        )
        if self.prediction_mode == "velocity":
            velocity = self.transport_denormalize_latent(field_out)
            remaining = (1.0 - t).clamp_min(self.target_time_eps)
            target_pred = z_t + remaining * velocity
            return velocity, field_out, target_pred
        target_pred = self.transport_denormalize_latent(field_out)
        remaining = (1.0 - t).clamp_min(self.target_time_eps)
        velocity = (target_pred - z_t) / remaining
        return velocity, field_out, target_pred

    def _integrate(
        self,
        *,
        z_prev: torch.Tensor,
        e_c: torch.Tensor,
        e_traj: torch.Tensor,
        e_s: torch.Tensor,
        num_steps: int,
        noise_std: float,
        z0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        n = z_prev.size(0)
        z = self.sample_transport_noise_like(z_prev, noise_std=noise_std) if z0 is None else z0
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = z.new_full((n, 1), step / float(num_steps))
            velocity, _, _ = self._predict_velocity(
                z_t=z,
                z_prev=z_prev,
                t=t,
                e_c=e_c,
                e_traj=e_traj,
                e_s=e_s,
            )
            z = z + dt * velocity
        return z

    def compute_loss(
        self,
        *,
        x_curr: torch.Tensor,
        x_next: torch.Tensor,
        style_label: torch.Tensor,
        prev_latent: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        flow_weight: float = 1.0,
        recon_weight: float = 0.0,
        noise_std: float = 1.0,
        prev_latent_noise_std: float = 0.0,
        prev_motion_noise_std: float = 0.0,
        prev_latent_dropout: float = 0.0,
        prev_motion_dropout: float = 0.0,
        solver_steps: int = 8,
        return_prediction: bool = False,
    ) -> dict[str, torch.Tensor]:
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        x_next = self._flatten_frames(x_next, self.input_dim)
        traj_next, motion_next = self.split_traj_motion(x_next)
        x_curr_for_conditions = x_curr
        if self.training and prev_motion_noise_std > 0.0:
            traj_curr, motion_curr = self.split_traj_motion(x_curr)
            sample_scale = torch.rand((motion_curr.size(0), 1), device=motion_curr.device, dtype=motion_curr.dtype)
            motion_curr = motion_curr + float(prev_motion_noise_std) * sample_scale * torch.randn_like(motion_curr)
            x_curr_for_conditions = torch.cat((traj_curr, motion_curr), dim=-1)
        if self.training and prev_motion_dropout > 0.0:
            traj_curr, motion_curr = self.split_traj_motion(x_curr_for_conditions)
            motion_curr = F.dropout(motion_curr, p=float(prev_motion_dropout), training=True)
            x_curr_for_conditions = torch.cat((traj_curr, motion_curr), dim=-1)
        e_c, e_traj, e_s = self.encode_conditions(
            x_curr=x_curr_for_conditions,
            style_label=style_label,
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
        z0 = self.sample_transport_noise_like(z1, noise_std=noise_std)
        t = torch.rand((z1.size(0), 1), device=z1.device, dtype=z1.dtype)
        z_t = (1.0 - t) * z0 + t * z1
        velocity_target = z1 - z0
        velocity_pred, field_out, target_pred = self._predict_velocity(
            z_t=z_t,
            z_prev=z_prev,
            t=t,
            e_c=e_c,
            e_traj=e_traj,
            e_s=e_s,
        )

        if self.prediction_mode == "velocity":
            flow_loss = F.mse_loss(velocity_pred, velocity_target)
        else:
            flow_loss = F.mse_loss(field_out, self.transport_normalize_latent(z1))

        recon_loss = z1.new_zeros(())
        z_hat = None
        motion_hat = None
        if recon_weight > 0.0 or return_prediction:
            z_hat = self.denormalize_latent(
                self._integrate(
                    z_prev=z_prev,
                    e_c=e_c,
                    e_traj=e_traj,
                    e_s=e_s,
                    num_steps=solver_steps,
                    noise_std=noise_std,
                    z0=z0,
                )
            )
            motion_hat = self.decode_motion(traj=traj_next, latent=z_hat)
        if recon_weight > 0.0:
            recon_loss = F.l1_loss(motion_hat, motion_next)

        total = flow_weight * flow_loss + recon_weight * recon_loss
        result = {
            "loss": total,
            "flow_loss": flow_loss,
            "recon_loss": recon_loss,
            "latent_target_norm": z1.norm(dim=-1).mean(),
            "velocity_pred_norm": velocity_pred.norm(dim=-1).mean(),
            "velocity_target_norm": velocity_target.norm(dim=-1).mean(),
        }
        if self.prediction_mode == "target":
            result.update(
                {
                    "target_pred_norm": target_pred.norm(dim=-1).mean(),
                    "target_error": (target_pred - z1).norm(dim=-1).mean(),
                    "target_weight_mean": self.latent_transport_std.to(device=z1.device, dtype=z1.dtype).mean(),
                }
            )
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
        style_label: torch.Tensor,
        prev_latent: torch.Tensor | None = None,
        num_steps: int = 8,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> torch.Tensor:
        if solver != "euler":
            raise ValueError("Minimal flowmatch controller only keeps Euler integration.")
        x_curr = self._flatten_frames(x_curr, self.input_dim)
        e_c, e_traj, e_s = self.encode_conditions(
            x_curr=x_curr,
            style_label=style_label,
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
        z0_norm = None if z0 is None else self.normalize_latent(z0.float().to(device=x_curr.device))
        z_hat = self._integrate(
            z_prev=z_prev,
            e_c=e_c,
            e_traj=e_traj,
            e_s=e_s,
            num_steps=num_steps,
            noise_std=noise_std,
            z0=z0_norm,
        )
        return self.denormalize_latent(z_hat)

    @torch.no_grad()
    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        prev_latent: torch.Tensor | None = None,
        num_steps: int = 8,
        noise_std: float = 1.0,
        z0: torch.Tensor | None = None,
        solver: Literal["euler", "midpoint", "heun"] = "euler",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_hat = self.sample_latent(
            x_curr=x_curr,
            style_label=style_label,
            prev_latent=prev_latent,
            num_steps=num_steps,
            noise_std=noise_std,
            z0=z0,
            solver=solver,
        )
        motion_hat = self.decode_motion(traj=traj_next, latent=z_hat)
        return motion_hat, z_hat


def build_flowmatch_lmm_controller(
    *,
    lmm: LMM375to291,
    num_styles: int,
    style_dim: int = 64,
    time_embed_dim: int = 64,
    flow_hidden_dim: int = 256,
    flow_layers: int = 2,
    flow_heads: int = 4,
    flow_ffn_dim: int = 1024,
    dropout: float = 0.1,
    prediction_mode: FlowPredictionMode = "velocity",
    target_time_eps: float = 1e-3,
) -> FlowMatchLMMController:
    return FlowMatchLMMController(
        lmm=lmm,
        num_styles=num_styles,
        style_dim=style_dim,
        time_embed_dim=time_embed_dim,
        flow_hidden_dim=flow_hidden_dim,
        flow_layers=flow_layers,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        dropout=dropout,
        prediction_mode=prediction_mode,
        target_time_eps=target_time_eps,
    )


def build_flowmatch_lmm_controller_from_config(
    config: Mapping[str, object],
    *,
    lmm: LMM375to291,
    num_styles: int,
) -> FlowMatchLMMController:
    return build_flowmatch_lmm_controller(
        lmm=lmm,
        num_styles=num_styles,
        style_dim=int(config.get("style_dim", 64)),
        time_embed_dim=int(config.get("time_embed_dim", 64)),
        flow_hidden_dim=int(config.get("flow_hidden_dim", config.get("hidden_dim", 256))),
        flow_layers=int(config.get("flow_layers", config.get("num_layers", 2))),
        flow_heads=int(config.get("flow_heads", 4)),
        flow_ffn_dim=int(config.get("flow_ffn_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        prediction_mode=resolve_flowmatch_prediction_mode(str(config.get("prediction_mode", "velocity"))),
        target_time_eps=float(config.get("target_time_eps", 1e-3)),
    )


__all__ = [
    "FlowMatchLMMController",
    "build_flowmatch_lmm_controller",
    "build_flowmatch_lmm_controller_from_config",
]
