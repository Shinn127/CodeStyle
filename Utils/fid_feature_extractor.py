from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_WINDOW_SIZE = 60
DEFAULT_FEATURE_DIM = 24 * 3
MOTION_ROOT_DIM = 3
MOTION_POS_DIM = 24 * 3

JOINT_SET_SELECTED24 = "selected24"
JOINT_SET_YBOT15 = "ybot15"
JOINT_SETS = (JOINT_SET_SELECTED24, JOINT_SET_YBOT15)

# CAMDM (https://github.com/AIGAnimation/CAMDM) trains its FID classifier on the 15 joints of
# the `ybot_skeleton` in exactly this order (Evaluation/feature_extractor/process_train_data.py)
# -> in_feat = 15 * 3 = 45.
YBOT15_JOINT_NAMES = [
    "Hips",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
]


def _ybot15_to_selected24_indices() -> list[int]:
    """Positions of CAMDM's 15 ybot joints inside this project's 24-joint block.

    Both lists are joint *names* on the same 100STYLE skeleton, so the mapping is exact
    (verified at import time); the classifier sees the same joints CAMDM feeds it.
    """
    from Utils.motion375 import SELECTED_24_NAMES

    missing = [name for name in YBOT15_JOINT_NAMES if name not in SELECTED_24_NAMES]
    if missing:
        raise RuntimeError(f"ybot joints missing from SELECTED_24_NAMES: {missing}")
    return [SELECTED_24_NAMES.index(name) for name in YBOT15_JOINT_NAMES]


YBOT15_TO_SELECTED24_INDICES = _ybot15_to_selected24_indices()


def joint_count_for(joint_set: str) -> int:
    if joint_set == JOINT_SET_SELECTED24:
        return 24
    if joint_set == JOINT_SET_YBOT15:
        return len(YBOT15_JOINT_NAMES)
    raise ValueError(f"Unknown joint_set={joint_set!r}; expected one of {JOINT_SETS}")


def infer_joint_set_from_in_feat(in_feat: int) -> str:
    """Recover which joint set a classifier checkpoint was trained on from its input width."""
    in_feat = int(in_feat)
    if in_feat == 24 * 3:
        return JOINT_SET_SELECTED24
    if in_feat == len(YBOT15_JOINT_NAMES) * 3:
        return JOINT_SET_YBOT15
    raise ValueError(
        f"Cannot infer the joint set for in_feat={in_feat}; "
        f"supported: {24 * 3} (selected24) and {len(YBOT15_JOINT_NAMES) * 3} (ybot15)"
    )


class MotionFeatureClassifier(nn.Module):
    """
    Convolutional classifier whose penultimate activations are used as FID features.

    This mirrors the structure of CAMDM-main/Evaluation/feature_extractor/train_classifier.py,
    but is packaged so we can retrain it on the local 100STYLE database and reuse
    the same checkpoint in evaluation.
    """

    def __init__(
        self,
        in_feat: int,
        class_num: int,
        window_size: int = DEFAULT_WINDOW_SIZE,
        dropout: float = 0.3,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.in_feat = int(in_feat)
        self.class_num = int(class_num)
        self.window_size = int(window_size)
        if not 0.0 <= float(dropout) < 1.0 or not 0.0 <= float(head_dropout) < 1.0:
            raise ValueError("dropout and head_dropout must be in [0, 1)")
        # Recorded as attributes so trainers can write the rates into the checkpoint provenance.
        self.dropout_prob = float(dropout)
        self.head_dropout_prob = float(head_dropout)

        self.conv1 = nn.Conv1d(in_channels=self.in_feat, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.residual1x1_1 = nn.Conv1d(in_channels=self.in_feat, out_channels=64, kernel_size=1)
        self.residual1x1_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.dropout = nn.Dropout(self.dropout_prob)
        # Extra regularisation between the 1024-d hidden and the 512-d FID feature.
        self.head_dropout = nn.Dropout(self.head_dropout_prob)
        self.fc1 = nn.Linear(in_features=128 * self.window_size, out_features=1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(in_features=512, out_features=self.class_num)

    def get_feats(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.in_feat:
            raise ValueError(f"Expected feature dim {self.in_feat}, got {x.size(1)}")
        if x.size(2) != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {x.size(2)}")

        residual = self.residual1x1_1(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = out + residual

        residual = self.residual1x1_2(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = out + residual

        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        out = self.head_dropout(out)
        out = F.leaky_relu(self.bn4(self.fc2(out)))
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.get_feats(x)
        logits = self.fc3(feats)
        return feats, logits


class MotionReconFeatureExtractor(nn.Module):
    """Feature extractor trained by *reconstruction*, in CAMDM's architectural family.

    Same conv trunk and same 512-dim feature width as `MotionFeatureClassifier`, but instead of a
    100-way style head the 512-dim code feeds a decoder that reconstructs the input window. The
    FID feature is that 512-dim code (the bottleneck), which is what makes this the
    "motion reconstruction objective" variant of the extractor:

        x [B, 45, 60] -> trunk (conv 64/128, residuals) -> FC 1024 -> FC 512  (feature)
                                                            -> FC 1024 -> FC 128*60 -> deconv -> x_hat

    `get_feats` returns the 512-dim code so the evaluation path can treat it exactly like the
    classifier variant; `forward` returns `(feats, reconstruction)`.
    """

    def __init__(
        self,
        in_feat: int,
        window_size: int = DEFAULT_WINDOW_SIZE,
        hidden_dim: int = 1024,
        feat_dim: int = 512,
        dropout: float = 0.3,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_feat = int(in_feat)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.feat_dim = int(feat_dim)
        if not 0.0 <= float(dropout) < 1.0 or not 0.0 <= float(head_dropout) < 1.0:
            raise ValueError("dropout and head_dropout must be in [0, 1)")
        self.dropout_prob = float(dropout)
        self.head_dropout_prob = float(head_dropout)

        self.conv1 = nn.Conv1d(in_channels=self.in_feat, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.residual1x1_1 = nn.Conv1d(in_channels=self.in_feat, out_channels=64, kernel_size=1)
        self.residual1x1_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.head_dropout = nn.Dropout(self.head_dropout_prob)

        self.fc1 = nn.Linear(in_features=128 * self.window_size, out_features=self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=self.feat_dim)
        self.bn4 = nn.BatchNorm1d(self.feat_dim)

        self.fc3 = nn.Linear(in_features=self.feat_dim, out_features=self.hidden_dim)
        self.bn5 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(in_features=self.hidden_dim, out_features=128 * self.window_size)
        self.bn6 = nn.BatchNorm1d(128 * self.window_size)
        self.deconv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv1d(in_channels=64, out_channels=self.in_feat, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.in_feat:
            raise ValueError(f"Expected feature dim {self.in_feat}, got {x.size(1)}")
        if x.size(2) != self.window_size:
            raise ValueError(f"Expected window size {self.window_size}, got {x.size(2)}")

        residual = self.residual1x1_1(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = out + residual
        residual = self.residual1x1_2(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = out + residual

        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        out = self.head_dropout(out)
        return F.leaky_relu(self.bn4(self.fc2(out)))

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn5(self.fc3(feats)))
        out = F.leaky_relu(self.bn6(self.fc4(out)))
        out = out.reshape(out.size(0), 128, self.window_size)
        out = F.leaky_relu(self.deconv1(out))
        out = F.leaky_relu(self.deconv2(out))
        return self.deconv3(out)

    def get_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encode(x)
        return feats, self.decode(feats)


def build_fid_feature_extractor_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    joint_set: str | None = None,
) -> tuple[nn.Module, dict]:
    """Load either FID extractor flavour, dispatching on the checkpoint's `objective` field.

    `objective="classification"` (or absent, for the CAMDM/legacy checkpoints) ->
    `MotionFeatureClassifier`; `objective="reconstruction"` -> `MotionReconFeatureExtractor`.
    Both expose `get_feats(x) -> [B, feat_dim]`, so the evaluation path is unchanged.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    objective = str(ckpt.get("objective", "classification")).strip().lower()
    if objective in {"classification", "style_classification"}:
        return build_motion_feature_classifier_from_checkpoint(
            ckpt_path, map_location=map_location, joint_set=joint_set
        )
    if objective != "reconstruction":
        raise ValueError(f"Unknown FID extractor objective={objective!r} in {ckpt_path}")

    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing state_dict: {ckpt_path}")
    in_feat = int(ckpt.get("in_feat", DEFAULT_FEATURE_DIM))
    window_size = int(ckpt.get("window_size", DEFAULT_WINDOW_SIZE))
    if joint_set is not None:
        if joint_set not in JOINT_SETS:
            raise ValueError(f"joint_set must be one of {JOINT_SETS}, got {joint_set!r}")
        if joint_count_for(joint_set) * 3 != in_feat:
            raise ValueError(
                f"joint_set={joint_set!r} implies in_feat={joint_count_for(joint_set) * 3}, "
                f"but the checkpoint uses in_feat={in_feat}"
            )
    model = MotionReconFeatureExtractor(
        in_feat=in_feat,
        window_size=window_size,
        hidden_dim=int(ckpt.get("hidden_dim", 1024)),
        feat_dim=int(ckpt.get("feat_dim", 512)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    ckpt["joint_set"] = str(joint_set) if joint_set is not None else infer_joint_set_from_in_feat(in_feat)
    ckpt["class_names"] = list(ckpt.get("class_names", []))
    ckpt["class_num"] = int(ckpt.get("class_num", 0))
    return model, ckpt


def build_motion_feature_classifier_from_checkpoint(
    ckpt_path: Path,
    *,
    map_location: str = "cpu",
    joint_set: str | None = None,
) -> tuple[MotionFeatureClassifier, dict]:
    """Load a classifier checkpoint and report which joint set its input expects.

    `joint_set=None` infers it from `in_feat` (72 -> selected24, 45 -> ybot15); a value in the
    checkpoint's `joint_set` field, when present, wins and is cross-checked against the width.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing state_dict: {ckpt_path}")
    in_feat = int(ckpt.get("in_feat", DEFAULT_FEATURE_DIM))
    class_names = list(ckpt.get("class_names", []))
    window_size = int(ckpt.get("window_size", DEFAULT_WINDOW_SIZE))
    class_num = int(ckpt.get("class_num", len(class_names) if class_names else 1))
    recorded = ckpt.get("joint_set")
    inferred = infer_joint_set_from_in_feat(in_feat)
    if joint_set is not None:
        if joint_set not in JOINT_SETS:
            raise ValueError(f"joint_set must be one of {JOINT_SETS}, got {joint_set!r}")
        if joint_count_for(joint_set) * 3 != in_feat:
            raise ValueError(
                f"joint_set={joint_set!r} implies in_feat={joint_count_for(joint_set) * 3}, "
                f"but the checkpoint uses in_feat={in_feat}"
            )
    elif recorded is not None:
        if joint_count_for(str(recorded)) * 3 != in_feat:
            raise ValueError(
                f"checkpoint records joint_set={recorded!r}, which needs in_feat="
                f"{joint_count_for(str(recorded)) * 3}, but it declares in_feat={in_feat}"
            )
        joint_set = str(recorded)
    else:
        joint_set = inferred
        if str(joint_set) != str(inferred):
            raise RuntimeError(f"inconsistent joint set inference for {ckpt_path}")
    model = MotionFeatureClassifier(
        in_feat=in_feat,
        class_num=class_num,
        window_size=window_size,
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    # normalise the metadata the evaluation path reads, for both extractor flavours
    ckpt["window_size"] = window_size
    ckpt["joint_set"] = str(joint_set)
    ckpt["class_names"] = class_names
    ckpt["class_num"] = class_num
    return model, ckpt


def extract_local_position_features(
    x_375: np.ndarray | torch.Tensor,
    *,
    selected_joint_count: int = 24,
    joint_set: str = JOINT_SET_SELECTED24,
) -> np.ndarray:
    x = x_375.detach().cpu().numpy() if isinstance(x_375, torch.Tensor) else np.asarray(x_375)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[-1] != 375:
        raise ValueError(f"Expected [B,T,375] or [T,375] input, got {x.shape}")
    motion = x[..., 84:]
    expected_motion_dim = MOTION_ROOT_DIM + 3 * selected_joint_count + 3 * selected_joint_count * 2 + 3 * selected_joint_count
    if motion.shape[-1] < expected_motion_dim:
        raise ValueError(f"Expected motion dim at least {expected_motion_dim}, got {motion.shape[-1]}")
    local_pos = motion[..., MOTION_ROOT_DIM : MOTION_ROOT_DIM + selected_joint_count * 3]
    if local_pos.shape[-1] != selected_joint_count * 3:
        raise ValueError(
            f"Expected {selected_joint_count * 3} local-position dims, got {local_pos.shape[-1]}"
        )
    if joint_set == JOINT_SET_SELECTED24:
        return local_pos
    if joint_set == JOINT_SET_YBOT15:
        picked = local_pos.reshape(*local_pos.shape[:-1], selected_joint_count, 3)
        picked = picked[..., YBOT15_TO_SELECTED24_INDICES, :]
        return picked.reshape(*local_pos.shape[:-1], len(YBOT15_JOINT_NAMES) * 3)
    raise ValueError(f"Unknown joint_set={joint_set!r}; expected one of {JOINT_SETS}")


def motion375_clip_to_classifier_input(
    x_375: np.ndarray | torch.Tensor,
    *,
    selected_joint_count: int = 24,
    joint_set: str = JOINT_SET_SELECTED24,
) -> torch.Tensor:
    local_pos = extract_local_position_features(
        x_375,
        selected_joint_count=selected_joint_count,
        joint_set=joint_set,
    )
    n_joints = joint_count_for(joint_set)
    local_pos = local_pos.reshape(local_pos.shape[0], local_pos.shape[1], n_joints * 3)
    return torch.from_numpy(local_pos).permute(0, 2, 1).contiguous().float()


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_WINDOW_SIZE",
    "JOINT_SET_SELECTED24",
    "JOINT_SET_YBOT15",
    "JOINT_SETS",
    "MotionFeatureClassifier",
    "MotionReconFeatureExtractor",
    "YBOT15_JOINT_NAMES",
    "YBOT15_TO_SELECTED24_INDICES",
    "build_fid_feature_extractor_from_checkpoint",
    "build_motion_feature_classifier_from_checkpoint",
    "extract_local_position_features",
    "infer_joint_set_from_in_feat",
    "joint_count_for",
    "motion375_clip_to_classifier_input",
]
