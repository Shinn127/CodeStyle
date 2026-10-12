from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_mann_gating_indices(
    *,
    traj_dim: int,
    motion_dim: int,
    num_styles: int,
    include_style_dims: bool = True,
    gated_joint_ids: Sequence[int] = (11, 15, 19, 23),
) -> list[int]:
    if traj_dim != 84:
        raise ValueError(f"default_mann_gating_indices currently expects traj_dim=84, got {traj_dim}")
    if motion_dim != 291:
        raise ValueError(f"default_mann_gating_indices currently expects motion_dim=291, got {motion_dim}")
    if num_styles < 1:
        raise ValueError(f"num_styles must be >= 1, got {num_styles}")

    local_pos_start = traj_dim + 3
    local_vel_start = traj_dim + 3 + (24 * 3) + (24 * 3) + (24 * 3)
    indices: list[int] = []
    for joint_id in gated_joint_ids:
        if not (0 <= int(joint_id) < 24):
            raise ValueError(f"Invalid gated joint id {joint_id}; expected in [0, 23]")
        joint_offset = int(joint_id) * 3
        indices.extend(range(local_pos_start + joint_offset, local_pos_start + joint_offset + 3))
    for joint_id in gated_joint_ids:
        joint_offset = int(joint_id) * 3
        indices.extend(range(local_vel_start + joint_offset, local_vel_start + joint_offset + 3))

    if include_style_dims:
        style_start = traj_dim + motion_dim
        indices.extend(range(style_start, style_start + num_styles))
    return indices


class ExpertLinear(nn.Module):
    def __init__(self, experts: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if experts < 1:
            raise ValueError(f"experts must be >= 1, got {experts}")
        if input_dim < 1 or output_dim < 1:
            raise ValueError(f"Invalid ExpertLinear shape: input={input_dim}, output={output_dim}")

        self.experts = int(experts)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.empty(self.experts, self.input_dim, self.output_dim))
        self.bias = nn.Parameter(torch.zeros(self.experts, self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_idx in range(self.experts):
            nn.init.xavier_uniform_(self.weight[expert_idx])
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"ExpertLinear expects x rank 2 [B,I], got {tuple(x.shape)}")
        if weights.dim() != 2:
            raise ValueError(f"ExpertLinear expects weights rank 2 [B,E], got {tuple(weights.shape)}")
        if x.size(0) != weights.size(0):
            raise ValueError("Batch size mismatch between x and weights")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected x feature dim {self.input_dim}, got {x.size(1)}")
        if weights.size(1) != self.experts:
            raise ValueError(f"Expected weights expert dim {self.experts}, got {weights.size(1)}")

        expert_outputs = torch.einsum("bi,eio->beo", x, self.weight)
        expert_outputs = expert_outputs + self.bias.unsqueeze(0)
        return torch.einsum("be,beo->bo", weights, expert_outputs)


class MANNController(nn.Module):
    def __init__(
        self,
        *,
        num_styles: int,
        traj_dim: int = 84,
        motion_dim: int = 291,
        main_hidden_dim: int = 512,
        gating_hidden_dim: int = 32,
        num_experts: int = 8,
        dropout: float = 0.2,
        gating_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_styles < 1:
            raise ValueError(f"num_styles must be >= 1, got {num_styles}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.num_styles = int(num_styles)
        self.traj_dim = int(traj_dim)
        self.motion_dim = int(motion_dim)
        self.input_dim = self.traj_dim + self.motion_dim + self.num_styles
        self.main_hidden_dim = int(main_hidden_dim)
        self.gating_hidden_dim = int(gating_hidden_dim)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)

        if gating_indices is None:
            gating_indices = default_mann_gating_indices(
                traj_dim=self.traj_dim,
                motion_dim=self.motion_dim,
                num_styles=self.num_styles,
            )
        gating_indices_np = np.asarray(gating_indices, dtype=np.int64).reshape(-1)
        if gating_indices_np.size == 0:
            raise ValueError("gating_indices must not be empty")
        if gating_indices_np.min() < 0 or gating_indices_np.max() >= self.input_dim:
            raise ValueError(
                f"gating_indices out of range for input_dim={self.input_dim}: "
                f"min={int(gating_indices_np.min())}, max={int(gating_indices_np.max())}"
            )
        self.register_buffer("gating_indices", torch.from_numpy(gating_indices_np), persistent=True)

        gating_input_dim = int(gating_indices_np.size)
        self.gating_fc1 = nn.Linear(gating_input_dim, self.gating_hidden_dim)
        self.gating_fc2 = nn.Linear(self.gating_hidden_dim, self.gating_hidden_dim)
        self.gating_fc3 = nn.Linear(self.gating_hidden_dim, self.num_experts)

        self.main_fc1 = ExpertLinear(self.num_experts, self.input_dim, self.main_hidden_dim)
        self.main_fc2 = ExpertLinear(self.num_experts, self.main_hidden_dim, self.main_hidden_dim)
        self.main_fc3 = ExpertLinear(self.num_experts, self.main_hidden_dim, self.motion_dim)

    def _flatten_input(self, controller_input: torch.Tensor) -> torch.Tensor:
        if controller_input.dim() == 2:
            return controller_input
        if controller_input.dim() == 3:
            return controller_input.reshape(-1, controller_input.size(-1))
        raise ValueError(f"Expected rank 2 or 3 controller_input, got {tuple(controller_input.shape)}")

    def _style_one_hot(self, style_label: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        style_label = torch.as_tensor(style_label, device=device).reshape(-1).long()
        if style_label.numel() == 1:
            style_label = style_label.expand(batch_size)
        elif style_label.numel() != batch_size:
            raise ValueError(
                f"style_label must have 1 or {batch_size} elements, got {style_label.numel()}"
            )
        if torch.any(style_label < 0) or torch.any(style_label >= self.num_styles):
            raise ValueError(f"style_label out of range for num_styles={self.num_styles}")
        return F.one_hot(style_label, num_classes=self.num_styles).to(dtype=dtype)

    def build_controller_input(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        if x_curr.dim() != 2 or x_curr.size(1) != (self.traj_dim + self.motion_dim):
            raise ValueError(
                f"x_curr must be shape [B,{self.traj_dim + self.motion_dim}], got {tuple(x_curr.shape)}"
            )
        if traj_next.dim() != 2 or traj_next.size(1) != self.traj_dim:
            raise ValueError(f"traj_next must be shape [B,{self.traj_dim}], got {tuple(traj_next.shape)}")
        if x_curr.size(0) != traj_next.size(0):
            raise ValueError("Batch size mismatch between x_curr and traj_next")

        motion_curr = x_curr[:, self.traj_dim :]
        style_one_hot = self._style_one_hot(
            style_label,
            batch_size=x_curr.size(0),
            device=x_curr.device,
            dtype=x_curr.dtype,
        )
        return torch.cat((traj_next, motion_curr, style_one_hot), dim=-1)

    def forward(self, controller_input: torch.Tensor) -> torch.Tensor:
        controller_input = self._flatten_input(controller_input)
        if controller_input.size(-1) != self.input_dim:
            raise ValueError(f"Expected controller_input dim {self.input_dim}, got {controller_input.size(-1)}")

        gating_input = torch.index_select(controller_input, dim=-1, index=self.gating_indices)
        gating_input = F.dropout(gating_input, p=self.dropout, training=self.training)
        gating_hidden = F.elu(self.gating_fc1(gating_input))
        gating_hidden = F.dropout(gating_hidden, p=self.dropout, training=self.training)
        gating_hidden = F.elu(self.gating_fc2(gating_hidden))
        gating_hidden = F.dropout(gating_hidden, p=self.dropout, training=self.training)
        mixture_weights = F.softmax(self.gating_fc3(gating_hidden), dim=-1)

        hidden = F.dropout(controller_input, p=self.dropout, training=self.training)
        hidden = F.elu(self.main_fc1(hidden, mixture_weights))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.elu(self.main_fc2(hidden, mixture_weights))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.main_fc3(hidden, mixture_weights)

    def predict_next_motion(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        controller_input = self.build_controller_input(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
        )
        return self(controller_input)

    def predict_next_frame(
        self,
        *,
        x_curr: torch.Tensor,
        traj_next: torch.Tensor,
        style_label: torch.Tensor,
    ) -> torch.Tensor:
        motion_hat = self.predict_next_motion(
            x_curr=x_curr,
            traj_next=traj_next,
            style_label=style_label,
        )
        return torch.cat((traj_next, motion_hat), dim=-1)


def build_mann_controller_from_config(
    config: Mapping[str, object],
    *,
    num_styles: int | None = None,
) -> MANNController:
    resolved_num_styles = int(config.get("num_styles", 0)) if num_styles is None else int(num_styles)
    if resolved_num_styles < 1:
        raise ValueError("num_styles must be provided either in config or via the function argument")

    return MANNController(
        num_styles=resolved_num_styles,
        traj_dim=int(config.get("traj_dim", 84)),
        motion_dim=int(config.get("motion_dim", 291)),
        main_hidden_dim=int(config.get("main_hidden_dim", config.get("main_hidden", 512))),
        gating_hidden_dim=int(config.get("gating_hidden_dim", config.get("gating_hidden", 32))),
        num_experts=int(config.get("num_experts", config.get("gating_output", 8))),
        dropout=float(config.get("dropout", 0.2)),
        gating_indices=config.get("gating_indices", None),
    )


__all__ = [
    "ExpertLinear",
    "MANNController",
    "build_mann_controller_from_config",
    "default_mann_gating_indices",
]
