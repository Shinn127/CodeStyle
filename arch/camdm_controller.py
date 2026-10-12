from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn


def get_named_beta_schedule(
    schedule_name: str,
    num_diffusion_timesteps: int,
    *,
    scale_betas: float = 1.0,
) -> np.ndarray:
    if num_diffusion_timesteps < 1:
        raise ValueError(f"num_diffusion_timesteps must be >= 1, got {num_diffusion_timesteps}")

    if schedule_name == "linear":
        scale = scale_betas * 1000.0 / float(num_diffusion_timesteps)
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2,
        )
    raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar,
    *,
    max_beta: float = 0.999,
) -> np.ndarray:
    betas = []
    for step in range(num_diffusion_timesteps):
        t1 = step / num_diffusion_timesteps
        t2 = (step + 1) / num_diffusion_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing="ij")
        pe_2i = torch.sin(pos / (10000 ** (two_i / d_model)))
        pe_2i_1 = torch.cos(pos / (10000 ** (two_i / d_model)))
        pe = torch.stack((pe_2i, pe_2i_1), dim=-1).reshape(1, max_seq_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x rank 3 [B,T,C], got {tuple(x.shape)}")
        _, seq_len, d_model = x.shape
        return x * math.sqrt(float(d_model)) + self.pe[:, :seq_len, :]


class TimestepEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.long, device=self.pe.device).reshape(-1)
        if torch.any(t < 0) or torch.any(t >= self.pe.size(0)):
            raise ValueError(f"timestep out of range for max_len={self.pe.size(0)}")
        return self.pe.index_select(0, t)


class CAMDMController(nn.Module):
    def __init__(
        self,
        *,
        num_styles: int,
        traj_dim: int = 84,
        motion_dim: int = 291,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        activation: str = "gelu",
        diffusion_steps: int = 32,
        beta_schedule: str = "cosine",
        simple_var: bool = True,
    ) -> None:
        super().__init__()
        if num_styles < 1:
            raise ValueError(f"num_styles must be >= 1, got {num_styles}")
        if diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be >= 1, got {diffusion_steps}")

        self.num_styles = int(num_styles)
        self.traj_dim = int(traj_dim)
        self.motion_dim = int(motion_dim)
        self.input_dim = self.traj_dim + self.motion_dim
        self.latent_dim = int(latent_dim)
        self.ff_size = int(ff_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.activation = str(activation)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_schedule = str(beta_schedule)
        self.simple_var = bool(simple_var)

        self.embed_style = nn.Embedding(self.num_styles, self.latent_dim)
        self.pos_encoding = PositionalEncoding(d_model=self.latent_dim, max_seq_len=5)
        self.time_encoding = TimestepEncoding(d_model=self.latent_dim, max_len=self.diffusion_steps)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.traj_proj = nn.Linear(self.traj_dim, self.latent_dim)
        self.motion_curr_proj = nn.Linear(self.motion_dim, self.latent_dim)
        self.noisy_motion_proj = nn.Linear(self.motion_dim, self.latent_dim)
        self.out_proj = nn.Linear(self.latent_dim, self.motion_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        betas_np = get_named_beta_schedule(self.beta_schedule, self.diffusion_steps)
        betas = torch.from_numpy(betas_np.astype(np.float32))
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[0] = 1.0
        alpha_prev[1:] = alpha_bars[:-1]
        posterior_var = ((1.0 - alpha_prev) / (1.0 - alpha_bars)) * betas
        posterior_var[0] = betas[0]
        posterior_coef1 = torch.sqrt(alphas) * (1.0 - alpha_prev) / (1.0 - alpha_bars)
        posterior_coef2 = torch.sqrt(alpha_prev) * betas / (1.0 - alpha_bars)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)
        self.register_buffer("alpha_prev", alpha_prev, persistent=False)
        self.register_buffer("posterior_var", posterior_var, persistent=False)
        self.register_buffer("posterior_coef1", posterior_coef1, persistent=False)
        self.register_buffer("posterior_coef2", posterior_coef2, persistent=False)

    def _expand_style_label(self, style_label: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        style_label = torch.as_tensor(style_label, device=device).reshape(-1).long()
        if style_label.numel() == 1:
            style_label = style_label.expand(batch_size)
        elif style_label.numel() != batch_size:
            raise ValueError(f"style_label must have 1 or {batch_size} elements, got {style_label.numel()}")
        if torch.any(style_label < 0) or torch.any(style_label >= self.num_styles):
            raise ValueError(f"style_label out of range for num_styles={self.num_styles}")
        return style_label

    def _extract_alpha_bar(self, t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return self.alpha_bars.index_select(0, t).to(device=like.device, dtype=like.dtype).unsqueeze(-1)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_bar = self._extract_alpha_bar(timesteps, like=x_start)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1.0 - alpha_bar) * noise

    def forward(
        self,
        *,
        x_noisy: torch.Tensor,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if x_noisy.dim() != 2 or x_noisy.size(-1) != self.motion_dim:
            raise ValueError(f"x_noisy must be shape [B,{self.motion_dim}], got {tuple(x_noisy.shape)}")
        if x_curr.dim() != 2 or x_curr.size(-1) != self.input_dim:
            raise ValueError(f"x_curr must be shape [B,{self.input_dim}], got {tuple(x_curr.shape)}")
        if traj_next.dim() != 2 or traj_next.size(-1) != self.traj_dim:
            raise ValueError(f"traj_next must be shape [B,{self.traj_dim}], got {tuple(traj_next.shape)}")
        if x_noisy.size(0) != x_curr.size(0) or x_noisy.size(0) != traj_next.size(0):
            raise ValueError("Batch size mismatch in CAMDMController.forward")

        batch_size = x_noisy.size(0)
        motion_curr = x_curr[:, self.traj_dim :]
        style_ids = self._expand_style_label(style_label, batch_size, x_noisy.device)
        time_embed = self.time_mlp(self.time_encoding(timesteps.to(device=x_noisy.device)))

        tokens = torch.stack(
            (
                self.noisy_motion_proj(x_noisy),
                self.traj_proj(traj_next),
                self.motion_curr_proj(motion_curr),
                self.embed_style(style_ids),
                time_embed,
            ),
            dim=1,
        )
        tokens = self.pos_encoding(tokens)
        encoded = self.seq_encoder(tokens)[:, 0]
        return self.out_proj(encoded)

    def predict_x0(
        self,
        *,
        x_noisy: torch.Tensor,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self(
            x_noisy=x_noisy,
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            timesteps=timesteps,
        )

    def p_sample_step(
        self,
        *,
        x_t: torch.Tensor,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        if not (0 <= step_idx < self.diffusion_steps):
            raise ValueError(f"step_idx out of range: {step_idx}")

        batch_size = x_t.size(0)
        t = torch.full((batch_size,), step_idx, device=x_t.device, dtype=torch.long)
        x0_pred = self.predict_x0(
            x_noisy=x_t,
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            timesteps=t,
        )
        coef1 = self.posterior_coef1[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        coef2 = self.posterior_coef2[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        mean = coef1 * x_t + coef2 * x0_pred

        if step_idx == 0:
            return mean
        var = self.betas[step_idx] if self.simple_var else self.posterior_var[step_idx]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var.to(device=x_t.device, dtype=x_t.dtype)) * noise

    def sample_motion(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.diffusion_steps
        if int(num_steps) != self.diffusion_steps:
            raise ValueError(
                f"CAMDMController currently samples with the full trained diffusion_steps={self.diffusion_steps}, "
                f"got num_steps={num_steps}"
            )

        batch_size = x_curr.size(0)
        x_t = torch.randn((batch_size, self.motion_dim), device=x_curr.device, dtype=x_curr.dtype)
        for step_idx in range(self.diffusion_steps - 1, -1, -1):
            x_t = self.p_sample_step(
                x_t=x_t,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_label,
                step_idx=step_idx,
            )
        return x_t

    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        return self.sample_motion(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            num_steps=num_steps,
        )

    def predict_next_frame(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        motion_hat = self.predict_next_motion(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
            num_steps=num_steps,
        )
        return torch.cat((traj_next, motion_hat), dim=-1)


def build_camdm_controller_from_config(
    config: Mapping[str, object],
    *,
    num_styles: int | None = None,
) -> CAMDMController:
    resolved_num_styles = int(config.get("num_styles", 0)) if num_styles is None else int(num_styles)
    if resolved_num_styles < 1:
        raise ValueError("num_styles must be provided either in config or via the function argument")

    return CAMDMController(
        num_styles=resolved_num_styles,
        traj_dim=int(config.get("traj_dim", 84)),
        motion_dim=int(config.get("motion_dim", 291)),
        latent_dim=int(config.get("latent_dim", 256)),
        ff_size=int(config.get("ff_size", 1024)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.2)),
        activation=str(config.get("activation", "gelu")),
        diffusion_steps=int(config.get("diffusion_steps", config.get("denoising_steps", 32))),
        beta_schedule=str(config.get("beta_schedule", "cosine")),
        simple_var=bool(config.get("simple_var", True)),
    )


__all__ = [
    "CAMDMController",
    "PositionalEncoding",
    "TimestepEncoding",
    "betas_for_alpha_bar",
    "build_camdm_controller_from_config",
    "get_named_beta_schedule",
]
