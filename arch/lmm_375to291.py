from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class Compressor(nn.Module):
    """
    LMM compressor network (same architecture as Motion-Matching-main/resources/train_decompressor.py):
      Linear(input, 512) -> ELU
      Linear(512, 512) -> ELU
      Linear(512, 512) -> ELU
      Linear(512, latent_dim)

    Accepts [B, D] or [B, T, D].
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)

        self.linear0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim == {self.input_dim}, got {x.size(-1)}"
            )
        if x.dim() == 2:
            h = F.elu(self.linear0(x))
            h = F.elu(self.linear1(h))
            h = F.elu(self.linear2(h))
            return self.linear3(h)
        if x.dim() == 3:
            b, t, _ = x.shape
            y = x.reshape(b * t, self.input_dim)
            y = F.elu(self.linear0(y))
            y = F.elu(self.linear1(y))
            y = F.elu(self.linear2(y))
            y = self.linear3(y)
            return y.reshape(b, t, self.latent_dim)
        raise ValueError(
            f"Expected shape [B,{self.input_dim}] or [B,T,{self.input_dim}], got {tuple(x.shape)}"
        )


class Decompressor(nn.Module):
    """
    LMM decompressor network (same architecture as Motion-Matching-main/resources/train_decompressor.py):
      Linear(input, 512) -> ReLU
      Linear(512, output)

    Accepts [B, D] or [B, T, D].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)

        self.linear0 = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim == {self.input_dim}, got {x.size(-1)}"
            )
        if x.dim() == 2:
            return self.linear1(F.relu(self.linear0(x)))
        if x.dim() == 3:
            b, t, _ = x.shape
            y = x.reshape(b * t, self.input_dim)
            y = self.linear1(F.relu(self.linear0(y)))
            return y.reshape(b, t, self.output_dim)
        raise ValueError(
            f"Expected shape [B,{self.input_dim}] or [B,T,{self.input_dim}], got {tuple(x.shape)}"
        )


class LMM375to291(nn.Module):
    """
    LMM autoencoder for the 375-dim feature split used in this project.

    Full frame feature layout:
      input(375) = traj(84) + motion(291)

    Model data flow:
      motion(291) -> Compressor -> latent(32)
      [traj(84), latent(32)] -> Decompressor -> motion(291)

    Here `traj(84)` is the real trajectory feature from dataset (not predicted by compressor).

    This supports both frame-wise input [B, D] and sequence input [B, T, D].

    Architecture alignment note:
      - latent_dim is fixed to 32
      - hidden_dim is fixed to 512
    to match `Motion-Matching-main/resources/train_decompressor.py`.
    """

    latent_dim_default = 32
    hidden_dim_default = 512
    root_dim = 3
    selected_joint_count = 24
    pos_dim = selected_joint_count * 3
    rot_dim = selected_joint_count * 3 * 2
    vel_dim = selected_joint_count * 3

    def __init__(
        self,
        motion_dim: int = 291,
        traj_dim: int = 84,
        latent_dim: int = latent_dim_default,
        hidden_dim: int = hidden_dim_default,
    ):
        super().__init__()

        if int(latent_dim) != self.latent_dim_default:
            raise ValueError(
                f"LMM375to291 latent_dim must be {self.latent_dim_default} to match "
                "Motion-Matching-main/resources/train_decompressor.py"
            )
        if int(hidden_dim) != self.hidden_dim_default:
            raise ValueError(
                f"LMM375to291 hidden_dim must be {self.hidden_dim_default} to match "
                "Motion-Matching-main/resources/train_decompressor.py"
            )

        self.motion_dim = int(motion_dim)
        self.traj_dim = int(traj_dim)
        self.latent_dim = int(latent_dim)
        self.input_dim = self.traj_dim + self.motion_dim
        self.bottleneck_dim = self.traj_dim + self.latent_dim
        self.hidden_dim = int(hidden_dim)

        expected_motion_dim = self.root_dim + self.pos_dim + self.rot_dim + self.vel_dim
        if self.motion_dim != expected_motion_dim:
            raise ValueError(
                f"LMM375to291 motion_dim must be {expected_motion_dim} for the current 100STYLE "
                f"feature layout, got {self.motion_dim}"
            )

        self.compressor = Compressor(
            input_dim=self.motion_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        self.decompressor = Decompressor(
            input_dim=self.bottleneck_dim,
            output_dim=self.motion_dim,
            hidden_dim=self.hidden_dim,
        )

    @staticmethod
    def _loss_by_type(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: Literal["l1", "mse", "smooth_l1"],
    ) -> torch.Tensor:
        if loss_type == "l1":
            return F.l1_loss(pred, target)
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        if loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _split_motion(
        self,
        motion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if motion.size(-1) != self.motion_dim:
            raise ValueError(
                f"Expected motion last dim == {self.motion_dim}, got {motion.size(-1)}"
            )

        offset = 0
        root = motion[..., offset : offset + self.root_dim]
        offset += self.root_dim

        pos = motion[..., offset : offset + self.pos_dim].reshape(
            *motion.shape[:-1], self.selected_joint_count, 3
        )
        offset += self.pos_dim

        rot = motion[..., offset : offset + self.rot_dim].reshape(
            *motion.shape[:-1], self.selected_joint_count, 3, 2
        )
        offset += self.rot_dim

        vel = motion[..., offset : offset + self.vel_dim].reshape(
            *motion.shape[:-1], self.selected_joint_count, 3
        )
        offset += self.vel_dim

        if offset != self.motion_dim:
            raise RuntimeError(f"Motion parsing consumed {offset} dims, expected {self.motion_dim}")

        return {
            "root_vel": torch.stack((root[..., 0], root[..., 2]), dim=-1),
            "root_ang": root[..., 1:2],
            "pos": pos,
            "rot": rot,
            "vel": vel,
        }

    def _denormalize_motion(
        self,
        motion: torch.Tensor,
        motion_mean: torch.Tensor | None,
        motion_std: torch.Tensor | None,
    ) -> torch.Tensor:
        if (motion_mean is None) != (motion_std is None):
            raise ValueError("motion_mean and motion_std must both be provided or both omitted")
        if motion_mean is None:
            return motion
        if motion_mean.shape != (self.motion_dim,):
            raise ValueError(f"motion_mean must have shape ({self.motion_dim},), got {tuple(motion_mean.shape)}")
        if motion_std.shape != (self.motion_dim,):
            raise ValueError(f"motion_std must have shape ({self.motion_dim},), got {tuple(motion_std.shape)}")
        return motion * motion_std + motion_mean

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.size(-1) != self.motion_dim:
            raise ValueError(
                f"Expected motion last dim == {self.motion_dim}, got {motion.size(-1)}"
            )
        return self.compressor(motion)

    def decode(
        self,
        traj: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if traj.dim() != latent.dim():
            raise ValueError(
                f"traj/latent rank mismatch: {traj.dim()} vs {latent.dim()}"
            )
        if traj.size(-1) != self.traj_dim:
            raise ValueError(
                f"Expected traj last dim == {self.traj_dim}, got {traj.size(-1)}"
            )
        if latent.size(-1) != self.latent_dim:
            raise ValueError(
                f"Expected latent last dim == {self.latent_dim}, got {latent.size(-1)}"
            )
        if traj.shape[:-1] != latent.shape[:-1]:
            raise ValueError(
                f"traj/latent leading shape mismatch: {traj.shape[:-1]} vs {latent.shape[:-1]}"
            )
        bottleneck = torch.cat([traj, latent], dim=-1)
        return self.decompressor(bottleneck)

    def forward(
        self,
        motion: torch.Tensor,
        traj: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(motion)
        motion_hat = self.decode(traj=traj, latent=latent)
        if return_details:
            return motion_hat, latent
        return motion_hat

    def compute_loss(
        self,
        motion: torch.Tensor,
        traj: torch.Tensor,
        recon_type: Literal["l1", "mse", "smooth_l1"] = "mse",
        recon_weight: float = 1.0,
        pos_weight: float = 15.0,
        rot_weight: float = 5.0,
        vel_weight: float = 2.0,
        root_vel_weight: float = 2.0,
        root_ang_weight: float = 2.0,
        pos_delta_weight: float = 2.0,
        rot_delta_weight: float = 0.75,
        latent_l1_weight: float = 0.1,
        latent_l2_weight: float = 0.1,
        latent_vel_weight: float = 0.01,
        dt: float = 1.0 / 60.0,
        motion_mean: torch.Tensor | None = None,
        motion_std: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        motion_hat, latent = self.forward(motion=motion, traj=traj, return_details=True)

        motion_ref = self._denormalize_motion(motion, motion_mean=motion_mean, motion_std=motion_std)
        motion_hat_ref = self._denormalize_motion(
            motion_hat,
            motion_mean=motion_mean,
            motion_std=motion_std,
        )

        motion_terms = self._split_motion(motion_ref)
        motion_hat_terms = self._split_motion(motion_hat_ref)

        pos_loss = pos_weight * self._loss_by_type(
            motion_hat_terms["pos"],
            motion_terms["pos"],
            recon_type,
        )
        rot_loss = rot_weight * self._loss_by_type(
            motion_hat_terms["rot"],
            motion_terms["rot"],
            recon_type,
        )
        vel_loss = vel_weight * self._loss_by_type(
            motion_hat_terms["vel"],
            motion_terms["vel"],
            recon_type,
        )
        root_vel_loss = root_vel_weight * self._loss_by_type(
            motion_hat_terms["root_vel"],
            motion_terms["root_vel"],
            recon_type,
        )
        root_ang_loss = root_ang_weight * self._loss_by_type(
            motion_hat_terms["root_ang"],
            motion_terms["root_ang"],
            recon_type,
        )

        pos_delta_loss = motion_hat.new_zeros(())
        rot_delta_loss = motion_hat.new_zeros(())
        if motion_hat.dim() == 3 and motion_hat.size(1) >= 2:
            pos_delta_hat = (motion_hat_terms["pos"][:, 1:] - motion_hat_terms["pos"][:, :-1]) / dt
            pos_delta_ref = (motion_terms["pos"][:, 1:] - motion_terms["pos"][:, :-1]) / dt
            pos_delta_loss = pos_delta_weight * self._loss_by_type(pos_delta_hat, pos_delta_ref, recon_type)

            rot_delta_hat = (motion_hat_terms["rot"][:, 1:] - motion_hat_terms["rot"][:, :-1]) / dt
            rot_delta_ref = (motion_terms["rot"][:, 1:] - motion_terms["rot"][:, :-1]) / dt
            rot_delta_loss = rot_delta_weight * self._loss_by_type(rot_delta_hat, rot_delta_ref, recon_type)

        recon_loss = (
            pos_loss
            + rot_loss
            + vel_loss
            + root_vel_loss
            + root_ang_loss
            + pos_delta_loss
            + rot_delta_loss
        )

        latent_l1_loss = torch.mean(torch.abs(latent))
        latent_l2_loss = torch.mean(torch.square(latent))

        latent_vel_loss = latent.new_zeros(())
        if latent_vel_weight > 0.0:
            if latent.dim() == 3 and latent.size(1) >= 2:
                latent_vel = (latent[:, 1:] - latent[:, :-1]) / dt
                latent_vel_loss = torch.mean(torch.abs(latent_vel))

        total_loss = (
            recon_weight * recon_loss
            + latent_l1_weight * latent_l1_loss
            + latent_l2_weight * latent_l2_loss
            + latent_vel_weight * latent_vel_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "pos_loss": pos_loss,
            "rot_loss": rot_loss,
            "vel_loss": vel_loss,
            "root_vel_loss": root_vel_loss,
            "root_ang_loss": root_ang_loss,
            "pos_delta_loss": pos_delta_loss,
            "rot_delta_loss": rot_delta_loss,
            "latent_l1_loss": latent_l1_loss,
            "latent_l2_loss": latent_l2_loss,
            "latent_vel_loss": latent_vel_loss,
            "motion_hat": motion_hat,
            "latent": latent,
        }


LMMCompressorDecompressorAE = LMM375to291


__all__ = [
    "Compressor",
    "Decompressor",
    "LMM375to291",
    "LMMCompressorDecompressorAE",
]
