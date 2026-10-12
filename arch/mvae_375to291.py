from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MVAEEncoder(nn.Module):
    """
    Conditional VAE encoder for motion transition.

    Inputs:
      - x_future: [B, F, input_dim] flattened target future frames.
      - condition: [B, C, input_dim] flattened autoregressive condition frames.
    Outputs:
      - z, mu, logvar: [B, latent_dim]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_condition_frames: int,
        num_future_predictions: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_condition_frames = int(num_condition_frames)
        self.num_future_predictions = int(num_future_predictions)

        future_dim = self.input_dim * self.num_future_predictions
        condition_dim = self.input_dim * self.num_condition_frames
        in_dim = future_dim + condition_dim

        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(future_dim + self.hidden_dim, self.hidden_dim)
        self.mu = nn.Linear(future_dim + self.hidden_dim, self.latent_dim)
        self.logvar = nn.Linear(future_dim + self.hidden_dim, self.latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x_future: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_future.dim() != 2:
            raise ValueError(f"x_future must be rank-2 [B, F*D], got {tuple(x_future.shape)}")
        if condition.dim() != 2:
            raise ValueError(f"condition must be rank-2 [B, C*D], got {tuple(condition.shape)}")
        if x_future.size(0) != condition.size(0):
            raise ValueError(
                f"x_future/condition batch mismatch: {x_future.size(0)} vs {condition.size(0)}"
            )

        h1 = F.elu(self.fc1(torch.cat((x_future, condition), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x_future, h1), dim=1)))
        s = torch.cat((x_future, h2), dim=1)
        mu = self.mu(s)
        logvar = self.logvar(s)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class MVAEMixedDecoder(nn.Module):
    """
    MoE decoder where z is injected at every layer (paper-aligned trick).

    Decoder conditions:
      - autoregressive history condition: C frames of full features.
      - deterministic future trajectory condition: F frames of trajectory features.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        traj_dim: int,
        motion_dim: int,
        latent_dim: int,
        hidden_dim: int,
        gate_hidden_dim: int,
        num_condition_frames: int,
        num_future_predictions: int,
        num_experts: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.traj_dim = int(traj_dim)
        self.motion_dim = int(motion_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.num_condition_frames = int(num_condition_frames)
        self.num_future_predictions = int(num_future_predictions)
        self.num_experts = int(num_experts)

        condition_dim = self.input_dim * self.num_condition_frames
        future_traj_dim = self.traj_dim * self.num_future_predictions
        self.cond_plus_traj_dim = condition_dim + future_traj_dim
        self.decoder_input_dim = self.latent_dim + self.cond_plus_traj_dim
        self.inter_dim = self.latent_dim + self.hidden_dim
        self.output_dim = self.motion_dim * self.num_future_predictions

        self.decoder_layers = [
            (
                nn.Parameter(
                    torch.empty(self.num_experts, self.decoder_input_dim, self.hidden_dim)
                ),
                nn.Parameter(torch.empty(self.num_experts, self.hidden_dim)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, self.inter_dim, self.hidden_dim)),
                nn.Parameter(torch.empty(self.num_experts, self.hidden_dim)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(self.num_experts, self.inter_dim, self.output_dim)),
                nn.Parameter(torch.empty(self.num_experts, self.output_dim)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter(f"w{index}", weight)
            self.register_parameter(f"b{index}", bias)

        self.gate = nn.Sequential(
            nn.Linear(self.decoder_input_dim, self.gate_hidden_dim),
            nn.ELU(),
            nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim),
            nn.ELU(),
            nn.Linear(self.gate_hidden_dim, self.num_experts),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        future_traj: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be rank-2 [B, latent_dim], got {tuple(z.shape)}")
        if condition.dim() != 2:
            raise ValueError(f"condition must be rank-2 [B, C*input_dim], got {tuple(condition.shape)}")
        if future_traj.dim() != 2:
            raise ValueError(
                f"future_traj must be rank-2 [B, F*traj_dim], got {tuple(future_traj.shape)}"
            )
        if not (z.size(0) == condition.size(0) == future_traj.size(0)):
            raise ValueError(
                "Batch mismatch among z/condition/future_traj: "
                f"{z.size(0)}, {condition.size(0)}, {future_traj.size(0)}"
            )

        cond_traj = torch.cat((condition, future_traj), dim=1)
        gate_in = torch.cat((z, cond_traj), dim=1)
        coefficients = F.softmax(self.gate(gate_in), dim=1)

        layer_out = cond_traj
        for weight, bias, activation in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)

            layer_input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, layer_input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class MVAE375to291(nn.Module):
    """
    Motion VAE adapted to this project's 375-d features:
      375 = traj(84) + motion(291)

    At each prediction step:
      condition: previous C full frames [traj + motion]
      future_traj: deterministic trajectory features for next F frames
      z ~ N(0, I)
      decoder(condition, future_traj, z) -> predicted future motion (291 * F)
    """

    def __init__(
        self,
        *,
        input_dim: int = 375,
        traj_dim: int = 84,
        motion_dim: int = 291,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        gate_hidden_dim: int = 64,
        num_condition_frames: int = 1,
        num_future_predictions: int = 1,
        num_experts: int = 6,
    ):
        super().__init__()

        if traj_dim + motion_dim != input_dim:
            raise ValueError(
                f"Expected traj_dim + motion_dim == input_dim, got "
                f"{traj_dim} + {motion_dim} != {input_dim}"
            )
        if num_condition_frames < 1:
            raise ValueError("num_condition_frames must be >= 1")
        if num_future_predictions < 1:
            raise ValueError("num_future_predictions must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if gate_hidden_dim < 1:
            raise ValueError("gate_hidden_dim must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")

        self.input_dim = int(input_dim)
        self.traj_dim = int(traj_dim)
        self.motion_dim = int(motion_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.num_condition_frames = int(num_condition_frames)
        self.num_future_predictions = int(num_future_predictions)
        self.num_experts = int(num_experts)

        self.encoder = MVAEEncoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_condition_frames=self.num_condition_frames,
            num_future_predictions=self.num_future_predictions,
        )
        self.decoder = MVAEMixedDecoder(
            input_dim=self.input_dim,
            traj_dim=self.traj_dim,
            motion_dim=self.motion_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            gate_hidden_dim=self.gate_hidden_dim,
            num_condition_frames=self.num_condition_frames,
            num_future_predictions=self.num_future_predictions,
            num_experts=self.num_experts,
        )

    @staticmethod
    def _to_btd(
        x: torch.Tensor,
        *,
        name: str,
        expected_last_dim: int,
    ) -> tuple[torch.Tensor, bool]:
        if x.size(-1) != expected_last_dim:
            raise ValueError(
                f"Expected {name} last dim == {expected_last_dim}, got {x.size(-1)}"
            )
        if x.dim() == 2:
            return x.unsqueeze(1), True
        if x.dim() == 3:
            return x, False
        raise ValueError(
            f"Expected {name} shape [B,{expected_last_dim}] or [B,T,{expected_last_dim}], got {tuple(x.shape)}"
        )

    @staticmethod
    def _restore_from_btd(x: torch.Tensor, squeezed_time: bool) -> torch.Tensor:
        return x[:, 0] if squeezed_time else x

    @staticmethod
    def _loss_by_type(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["l1", "mse", "smooth_l1"],
    ) -> torch.Tensor:
        if loss_type == "l1":
            return (pred - target).abs()
        if loss_type == "mse":
            return (pred - target).pow(2)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Mean KL over batch and latent dims.
        return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()

    def split_traj_motion(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected x last dim == {self.input_dim}, got {x.size(-1)}"
            )
        return x[..., : self.traj_dim], x[..., self.traj_dim :]

    def forward(
        self,
        *,
        x_future: torch.Tensor,
        future_traj: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_bfd, x_squeezed = self._to_btd(
            x_future,
            name="x_future",
            expected_last_dim=self.input_dim,
        )
        traj_bfd, traj_squeezed = self._to_btd(
            future_traj,
            name="future_traj",
            expected_last_dim=self.traj_dim,
        )
        cond_bcd, _ = self._to_btd(
            condition,
            name="condition",
            expected_last_dim=self.input_dim,
        )

        if x_squeezed != traj_squeezed:
            raise ValueError(
                "x_future and future_traj rank mismatch: "
                f"{tuple(x_future.shape)} vs {tuple(future_traj.shape)}"
            )

        if x_bfd.size(1) != self.num_future_predictions:
            raise ValueError(
                f"x_future time dim must be {self.num_future_predictions}, got {x_bfd.size(1)}"
            )
        if traj_bfd.size(1) != self.num_future_predictions:
            raise ValueError(
                f"future_traj time dim must be {self.num_future_predictions}, got {traj_bfd.size(1)}"
            )
        if cond_bcd.size(1) != self.num_condition_frames:
            raise ValueError(
                f"condition time dim must be {self.num_condition_frames}, got {cond_bcd.size(1)}"
            )

        if not (
            x_bfd.size(0) == traj_bfd.size(0) == cond_bcd.size(0)
        ):
            raise ValueError(
                "Batch mismatch among x_future/future_traj/condition: "
                f"{x_bfd.size(0)}, {traj_bfd.size(0)}, {cond_bcd.size(0)}"
            )

        x_flat = x_bfd.flatten(start_dim=1, end_dim=2)
        traj_flat = traj_bfd.flatten(start_dim=1, end_dim=2)
        cond_flat = cond_bcd.flatten(start_dim=1, end_dim=2)

        z, mu, logvar = self.encoder(x_flat, cond_flat)
        motion_hat_flat = self.decoder(z, cond_flat, traj_flat)
        motion_hat = motion_hat_flat.view(
            -1, self.num_future_predictions, self.motion_dim
        )
        motion_hat = self._restore_from_btd(motion_hat, x_squeezed)
        return motion_hat, mu, logvar

    @torch.no_grad()
    def encode(
        self,
        *,
        x_future: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_bfd, _ = self._to_btd(
            x_future,
            name="x_future",
            expected_last_dim=self.input_dim,
        )
        cond_bcd, _ = self._to_btd(
            condition,
            name="condition",
            expected_last_dim=self.input_dim,
        )
        if x_bfd.size(1) != self.num_future_predictions:
            raise ValueError(
                f"x_future time dim must be {self.num_future_predictions}, got {x_bfd.size(1)}"
            )
        if cond_bcd.size(1) != self.num_condition_frames:
            raise ValueError(
                f"condition time dim must be {self.num_condition_frames}, got {cond_bcd.size(1)}"
            )
        x_flat = x_bfd.flatten(start_dim=1, end_dim=2)
        cond_flat = cond_bcd.flatten(start_dim=1, end_dim=2)
        _, mu, logvar = self.encoder(x_flat, cond_flat)
        return mu, logvar

    @torch.no_grad()
    def sample(
        self,
        *,
        z: torch.Tensor,
        future_traj: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"Expected z shape [B,{self.latent_dim}], got {tuple(z.shape)}")
        if z.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected z last dim == {self.latent_dim}, got {z.size(-1)}"
            )

        traj_bfd, traj_squeezed = self._to_btd(
            future_traj,
            name="future_traj",
            expected_last_dim=self.traj_dim,
        )
        cond_bcd, _ = self._to_btd(
            condition,
            name="condition",
            expected_last_dim=self.input_dim,
        )

        if traj_bfd.size(1) != self.num_future_predictions:
            raise ValueError(
                f"future_traj time dim must be {self.num_future_predictions}, got {traj_bfd.size(1)}"
            )
        if cond_bcd.size(1) != self.num_condition_frames:
            raise ValueError(
                f"condition time dim must be {self.num_condition_frames}, got {cond_bcd.size(1)}"
            )
        if not (z.size(0) == traj_bfd.size(0) == cond_bcd.size(0)):
            raise ValueError(
                "Batch mismatch among z/future_traj/condition: "
                f"{z.size(0)}, {traj_bfd.size(0)}, {cond_bcd.size(0)}"
            )

        traj_flat = traj_bfd.flatten(start_dim=1, end_dim=2)
        cond_flat = cond_bcd.flatten(start_dim=1, end_dim=2)
        motion_hat_flat = self.decoder(z, cond_flat, traj_flat)
        motion_hat = motion_hat_flat.view(
            -1, self.num_future_predictions, self.motion_dim
        )
        return self._restore_from_btd(motion_hat, traj_squeezed)


__all__ = ["MVAE375to291", "MVAEEncoder", "MVAEMixedDecoder"]
