from __future__ import annotations

import numpy as np

from Utils import quat


FEATURE_DIM = 375
TRAJ_DIM = 84
MOTION_DIM = FEATURE_DIM - TRAJ_DIM
TRAJECTORY_SAMPLE_STRIDE = 10
SELECTED_24_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]
SELECTED_24_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Neck1",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]


def _as_feature_sequence(features_375: np.ndarray) -> np.ndarray:
    x = np.asarray(features_375, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T, {FEATURE_DIM}) or ({FEATURE_DIM},), got {x.shape}")
    return x


def decode_selected_state(features_375: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _as_feature_sequence(features_375)
    state = x[:, TRAJ_DIM:]
    if state.shape[1] != MOTION_DIM:
        raise ValueError(f"Expected motion slice dim {MOTION_DIM}, got {state.shape[1]}")

    n_joints = len(SELECTED_24_NAMES)
    root_motion = state[:, :3]

    pos_start = 3
    pos_end = pos_start + n_joints * 3
    rot_x_end = pos_end + n_joints * 3
    rot_y_end = rot_x_end + n_joints * 3
    vel_end = rot_y_end + n_joints * 3

    local_pos = state[:, pos_start:pos_end].reshape(-1, n_joints, 3)
    rot_x = state[:, pos_end:rot_x_end].reshape(-1, n_joints, 3)
    rot_y = state[:, rot_x_end:rot_y_end].reshape(-1, n_joints, 3)
    local_vel = state[:, rot_y_end:vel_end].reshape(-1, n_joints, 3)

    local_rot = quat.from_xform_xy(np.stack([rot_x, rot_y], axis=-1))
    local_rot = quat.normalize(local_rot)
    return root_motion, local_pos, local_rot, local_vel


def decode_trajectory_context(
    features_375: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _as_feature_sequence(features_375)
    num_samples = TRAJ_DIM // 7
    if num_samples * 7 != TRAJ_DIM:
        raise ValueError(f"Expected trajectory dim {TRAJ_DIM} to be divisible into 7 chunks, got {TRAJ_DIM}")

    traj = x[:, :TRAJ_DIM]
    px = traj[:, 0:num_samples]
    pz = traj[:, num_samples : 2 * num_samples]
    dx = traj[:, 2 * num_samples : 3 * num_samples]
    dz = traj[:, 3 * num_samples : 4 * num_samples]
    vx = traj[:, 4 * num_samples : 5 * num_samples]
    vz = traj[:, 5 * num_samples : 6 * num_samples]
    speed = traj[:, 6 * num_samples : 7 * num_samples]

    positions = np.zeros((x.shape[0], num_samples, 3), dtype=np.float32)
    directions = np.zeros_like(positions)
    velocities = np.zeros_like(positions)
    positions[..., 0] = px
    positions[..., 2] = pz
    directions[..., 0] = dx
    directions[..., 2] = dz
    velocities[..., 0] = vx
    velocities[..., 2] = vz
    directions = directions / np.clip(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-8, None)
    return positions, directions.astype(np.float32), velocities, speed.astype(np.float32)


def apply_coordinate_convert_to_positions(
    positions: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    positions_out = np.asarray(positions, dtype=np.float32)
    if not enabled:
        return positions_out
    positions_out = positions_out.copy()
    positions_out[..., 0] *= -1.0
    return positions_out


def _integrate_root_motion(
    root_motion: np.ndarray,
    *,
    dt: float,
    root_wy_in_degrees: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    root_motion = np.asarray(root_motion, dtype=np.float32)
    total_frames = int(root_motion.shape[0])
    root_pos = np.zeros((total_frames, 3), dtype=np.float32)
    root_rot = np.zeros((total_frames, 4), dtype=np.float32)
    root_rot[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    if total_frames <= 1:
        return root_pos, root_rot

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Vectorized recurrence. Every step rotates about the same world axis, so step k uses the
    # closed-form rotation by the cumulative yaw of the first k steps (yaw itself is applied on
    # the same frame k+1 as in the loop: frame j's rotation carries yaw[0..j-2]... see below).
    # Conventions kept identical to the per-frame loop:
    #   frame j rotation = rotation by sum(yaw[0:j-1])   (loop applies yaw[j-1] when building j)
    #   step k->k+1 world displacement = R(cumulative yaw of frames <= k) @ [dx_k, 0, dz_k]
    # Verified against the loop on real 240-frame sequences; float32 rounding differences only.
    count = total_frames - 1
    # float64 accumulation: a 240-step fp32 cumsum drifts ~2.6e-5 m against the loop, which is
    # still tiny but 10x above this pipeline's float32 noise floor; float64 keeps the vectorized
    # form within ~1e-7 of the per-frame loop (measured), for negligible cost at these sizes.
    deltas = (root_motion[:count] * float(dt)).astype(np.float64)
    yaw = np.radians(deltas[:, 1]) if root_wy_in_degrees else deltas[:, 1]
    cs = np.cumsum(yaw)                                        # cs[k] = yaw of frames 0..k
    angles_for_steps = np.concatenate((np.zeros(1, dtype=np.float64), cs[:-1]))
    cos_yaw = np.cos(angles_for_steps)
    sin_yaw = np.sin(angles_for_steps)
    dx, dz = deltas[:, 0], deltas[:, 2]
    root_pos[1:, 0] = np.cumsum(dx * cos_yaw + dz * sin_yaw).astype(np.float32)
    root_pos[1:, 2] = np.cumsum(-dx * sin_yaw + dz * cos_yaw).astype(np.float32)
    root_rot[1:] = quat.normalize(
        quat.from_angle_axis(cs.astype(np.float32), np.broadcast_to(up, (count, 3))))

    return root_pos, root_rot


def reconstruct_root_track_from_375(
    features_375: np.ndarray,
    *,
    dt: float = 1.0 / 60.0,
    root_wy_in_degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    root_motion, _, _, _ = decode_selected_state(features_375)
    return _integrate_root_motion(root_motion, dt=dt, root_wy_in_degrees=root_wy_in_degrees)


def reconstruct_selected_globals_from_375(
    features_375: np.ndarray,
    *,
    dt: float = 1.0 / 60.0,
    root_wy_in_degrees: bool = True,
    use_global: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    root_motion, local_pos, local_rot, _ = decode_selected_state(features_375)
    if not use_global:
        return local_pos.astype(np.float32), local_rot.astype(np.float32)

    root_pos, root_rot = _integrate_root_motion(
        root_motion,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
    )
    global_pos = quat.mul_vec(root_rot[:, None, :], local_pos) + root_pos[:, None, :]
    global_rot = quat.mul(root_rot[:, None, :], local_rot)
    return np.asarray(global_pos, dtype=np.float32), quat.normalize(np.asarray(global_rot, dtype=np.float32))


__all__ = [
    "FEATURE_DIM",
    "MOTION_DIM",
    "SELECTED_24_INDICES",
    "SELECTED_24_NAMES",
    "TRAJECTORY_SAMPLE_STRIDE",
    "TRAJ_DIM",
    "apply_coordinate_convert_to_positions",
    "decode_selected_state",
    "decode_trajectory_context",
    "reconstruct_root_track_from_375",
    "reconstruct_selected_globals_from_375",
]
