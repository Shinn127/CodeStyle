from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.motion375 import (  # noqa: E402
    apply_coordinate_convert_to_positions,
    decode_selected_state as _decode_selected_state_full,
    reconstruct_selected_globals_from_375,
)


def _prepare_motion(animations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize floor height and center x/z by root per frame."""
    motion = np.asarray(animations, dtype=np.float32).copy()
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(f"Expected animations shape (T, J, 3), got {motion.shape}")

    root_traj = motion[:, 0, [0, 2]].copy()
    floor = float(motion[..., 1].min())
    motion[..., 1] -= floor
    motion[..., 0] -= motion[:, 0:1, 0]
    motion[..., 2] -= motion[:, 0:1, 2]
    return motion, root_traj


def _limits_from_motion(motion: np.ndarray, pad: float = 0.15) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x = motion[..., 0]
    y = motion[..., 2]
    z = motion[..., 1]

    radius = max(float(np.abs(x).max()), float(np.abs(y).max()), 1.0)
    radius *= 1.0 + pad
    z_max = max(float(z.max()) * (1.0 + pad), 1.0)
    return (-radius, radius), (-radius, radius), (0.0, z_max)


def _style_axes(ax, xlim: tuple[float, float], ylim: tuple[float, float], zlim: tuple[float, float]) -> None:
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.view_init(elev=20, azim=45)
    ax.grid(False)
    ax.set_axis_off()

    try:
        ax.set_box_aspect((1.0, 1.0, 0.85))
    except Exception:
        pass


def _add_ground_plane(ax, xlim: tuple[float, float], ylim: tuple[float, float], z: float = 0.0) -> None:
    verts = [[
        [xlim[0], ylim[0], z],
        [xlim[0], ylim[1], z],
        [xlim[1], ylim[1], z],
        [xlim[1], ylim[0], z],
    ]]
    plane = Poly3DCollection(verts, alpha=0.25, facecolor=(0.75, 0.78, 0.84), edgecolor="none")
    ax.add_collection3d(plane)


def _bone_indices(parents: np.ndarray) -> list[int]:
    return [joint for joint, parent in enumerate(parents) if parent != -1]


def _bone_colors(n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def animation_plot(
    animations: np.ndarray,
    parents: list[int] | np.ndarray,
    interval: int,
    save_path: str | Path | None = None,
    title: str | None = None,
    trail_len: int = 90,
):
    parents = np.asarray(parents, dtype=np.int32)
    motion, root_traj = _prepare_motion(animations)
    if motion.shape[1] != len(parents):
        raise ValueError(f"parents length {len(parents)} does not match joints {motion.shape[1]}")

    xlim, ylim, zlim = _limits_from_motion(motion)
    fig = plt.figure(figsize=(7, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, xlim, ylim, zlim)
    _add_ground_plane(ax, xlim, ylim, z=0.0)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents)
    colors = _bone_colors(len(bones))
    lines = []
    for index, joint in enumerate(bones):
        parent = int(parents[joint])
        line_width = 2.8 if parent == 0 else 2.0
        (line,) = ax.plot([], [], [], color=colors[index], lw=line_width, solid_capstyle="round")
        lines.append((joint, line))

    (traj_line,) = ax.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)

    def init():
        for _, line in lines:
            line.set_data_3d([], [], [])
        traj_line.set_data_3d([], [], [])
        return [line for _, line in lines] + [traj_line]

    def animate(frame_idx: int):
        for joint, line in lines:
            parent = int(parents[joint])
            line.set_data_3d(
                [motion[frame_idx, joint, 0], motion[frame_idx, parent, 0]],
                [motion[frame_idx, joint, 2], motion[frame_idx, parent, 2]],
                [motion[frame_idx, joint, 1], motion[frame_idx, parent, 1]],
            )

        if frame_idx > 1:
            start = max(0, frame_idx - trail_len)
            trail = root_traj[start : frame_idx + 1] - root_traj[frame_idx : frame_idx + 1]
            traj_line.set_data_3d(trail[:, 0], trail[:, 1], np.zeros(trail.shape[0], dtype=np.float32))
        else:
            traj_line.set_data_3d([], [], [])

        return [line for _, line in lines] + [traj_line]

    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=motion.shape[0],
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1000.0 / max(interval, 1))))
        ani.save(str(save_path), writer="pillow", fps=fps)
        plt.close(fig)
    else:
        plt.show()

    return ani


def animation_plot_frame(
    animations: np.ndarray,
    parents: list[int] | np.ndarray,
    save_path: str | Path | None = None,
    frame_idx: int = 0,
    title: str | None = None,
) -> None:
    parents = np.asarray(parents, dtype=np.int32)
    motion, root_traj = _prepare_motion(animations)
    if motion.shape[1] != len(parents):
        raise ValueError(f"parents length {len(parents)} does not match joints {motion.shape[1]}")

    frame_idx = int(np.clip(frame_idx, 0, motion.shape[0] - 1))
    xlim, ylim, zlim = _limits_from_motion(motion)

    fig = plt.figure(figsize=(7, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, xlim, ylim, zlim)
    _add_ground_plane(ax, xlim, ylim, z=0.0)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents)
    colors = _bone_colors(len(bones))
    for index, joint in enumerate(bones):
        parent = int(parents[joint])
        line_width = 3.0 if parent == 0 else 2.2
        ax.plot(
            [motion[frame_idx, joint, 0], motion[frame_idx, parent, 0]],
            [motion[frame_idx, joint, 2], motion[frame_idx, parent, 2]],
            [motion[frame_idx, joint, 1], motion[frame_idx, parent, 1]],
            color=colors[index],
            lw=line_width,
            solid_capstyle="round",
        )

    if frame_idx > 1:
        trail = root_traj[: frame_idx + 1] - root_traj[frame_idx : frame_idx + 1]
        ax.plot(trail[:, 0], trail[:, 1], np.zeros(trail.shape[0], dtype=np.float32), color="#2E6FBB", lw=1.5)

    ax.scatter(
        motion[frame_idx, :, 0],
        motion[frame_idx, :, 2],
        motion[frame_idx, :, 1],
        s=9,
        color="#1f1f1f",
        alpha=0.85,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()


def _decode_375_state(features_375: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root_vel, local_pos, _, _ = _decode_selected_state_full(features_375)
    return root_vel, local_pos


def _reconstruct_motion_from_375(
    features_375: np.ndarray,
    *,
    dt: float,
    root_wy_in_degrees: bool,
    use_global: bool,
) -> np.ndarray:
    if use_global:
        global_pos, _ = reconstruct_selected_globals_from_375(
            features_375,
            dt=dt,
            root_wy_in_degrees=root_wy_in_degrees,
            use_global=True,
        )
        return global_pos

    _, local_pos = _decode_375_state(features_375)
    return local_pos


def animation_plot_from_375(
    features_375: np.ndarray,
    parents: list[int] | np.ndarray,
    interval: int = 33,
    save_path: str | Path | None = None,
    title: str | None = None,
    dt: float = 1.0 / 60.0,
    root_wy_in_degrees: bool = True,
    use_global: bool = True,
    coordinate_convert: bool = False,
    frame_idx: int = 0,
    single_frame: bool = False,
    trail_len: int = 90,
):
    parents = np.asarray(parents, dtype=np.int32)
    _, local_pos = _decode_375_state(features_375)
    if local_pos.shape[1] != len(parents):
        raise ValueError(
            f"parents length {len(parents)} does not match joints decoded from 375 ({local_pos.shape[1]})."
        )

    motion = _reconstruct_motion_from_375(
        features_375,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )
    motion = apply_coordinate_convert_to_positions(
        motion,
        enabled=bool(coordinate_convert),
    )

    if single_frame:
        return animation_plot_frame(
            motion,
            parents,
            save_path=save_path,
            frame_idx=frame_idx,
            title=title,
        )

    return animation_plot(
        motion,
        parents,
        interval=interval,
        save_path=save_path,
        title=title,
        trail_len=trail_len,
    )


def visualize_side_by_side_from_375(
    ref_features_375: np.ndarray,
    pred_features_375: np.ndarray,
    parents: list[int] | np.ndarray,
    *,
    fps: int,
    dt: float,
    use_global: bool,
    root_wy_in_degrees: bool = True,
    coordinate_convert: bool = False,
    trail_len: int = 120,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    parents_np = np.asarray(parents, dtype=np.int32)

    ref_motion = _reconstruct_motion_from_375(
        ref_features_375,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )
    pred_motion = _reconstruct_motion_from_375(
        pred_features_375,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
    )

    ref_motion = apply_coordinate_convert_to_positions(ref_motion, enabled=bool(coordinate_convert))
    pred_motion = apply_coordinate_convert_to_positions(pred_motion, enabled=bool(coordinate_convert))

    if ref_motion.shape[1] != len(parents_np):
        raise ValueError(f"Reference joints {ref_motion.shape[1]} mismatch parents length {len(parents_np)}")
    if pred_motion.shape[1] != len(parents_np):
        raise ValueError(f"Prediction joints {pred_motion.shape[1]} mismatch parents length {len(parents_np)}")

    t_total = min(ref_motion.shape[0], pred_motion.shape[0])
    if t_total <= 1:
        raise ValueError(f"Need at least 2 frames for animation, got {t_total}")
    ref_motion = ref_motion[:t_total]
    pred_motion = pred_motion[:t_total]

    ref_motion, ref_root_traj = _prepare_motion(ref_motion)
    pred_motion, pred_root_traj = _prepare_motion(pred_motion)

    ref_xlim, ref_ylim, ref_zlim = _limits_from_motion(ref_motion)
    pred_xlim, pred_ylim, pred_zlim = _limits_from_motion(pred_motion)
    xlim = (min(ref_xlim[0], pred_xlim[0]), max(ref_xlim[1], pred_xlim[1]))
    ylim = (min(ref_ylim[0], pred_ylim[0]), max(ref_ylim[1], pred_ylim[1]))
    zlim = (min(ref_zlim[0], pred_zlim[0]), max(ref_zlim[1], pred_zlim[1]))

    fig = plt.figure(figsize=(13, 6.5), dpi=120)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    _style_axes(ax_left, xlim, ylim, zlim)
    _style_axes(ax_right, xlim, ylim, zlim)
    _add_ground_plane(ax_left, xlim, ylim)
    _add_ground_plane(ax_right, xlim, ylim)
    ax_left.set_title("Reference", fontsize=11, pad=6)
    ax_right.set_title("Prediction", fontsize=11, pad=6)
    if title is not None:
        fig.suptitle(title, fontsize=12)

    bones = _bone_indices(parents_np)
    colors = _bone_colors(len(bones))
    left_lines = []
    right_lines = []
    for index, joint in enumerate(bones):
        parent = int(parents_np[joint])
        line_width = 2.8 if parent == 0 else 2.0
        (line_left,) = ax_left.plot([], [], [], color=colors[index], lw=line_width, solid_capstyle="round")
        (line_right,) = ax_right.plot([], [], [], color=colors[index], lw=line_width, solid_capstyle="round")
        left_lines.append((joint, line_left))
        right_lines.append((joint, line_right))

    (left_traj_line,) = ax_left.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)
    (right_traj_line,) = ax_right.plot([], [], [], color="#2E6FBB", lw=1.6, alpha=0.9)
    artists = [line for _, line in left_lines] + [line for _, line in right_lines] + [left_traj_line, right_traj_line]

    def init():
        for line in artists:
            line.set_data_3d([], [], [])
        return artists

    def animate(frame_idx: int):
        for joint, line in left_lines:
            parent = int(parents_np[joint])
            line.set_data_3d(
                [ref_motion[frame_idx, joint, 0], ref_motion[frame_idx, parent, 0]],
                [ref_motion[frame_idx, joint, 2], ref_motion[frame_idx, parent, 2]],
                [ref_motion[frame_idx, joint, 1], ref_motion[frame_idx, parent, 1]],
            )
        for joint, line in right_lines:
            parent = int(parents_np[joint])
            line.set_data_3d(
                [pred_motion[frame_idx, joint, 0], pred_motion[frame_idx, parent, 0]],
                [pred_motion[frame_idx, joint, 2], pred_motion[frame_idx, parent, 2]],
                [pred_motion[frame_idx, joint, 1], pred_motion[frame_idx, parent, 1]],
            )

        if frame_idx > 1:
            start = max(0, frame_idx - trail_len)
            left_trail = ref_root_traj[start : frame_idx + 1] - ref_root_traj[frame_idx : frame_idx + 1]
            right_trail = pred_root_traj[start : frame_idx + 1] - pred_root_traj[frame_idx : frame_idx + 1]
            left_traj_line.set_data_3d(
                left_trail[:, 0],
                left_trail[:, 1],
                np.zeros(left_trail.shape[0], dtype=np.float32),
            )
            right_traj_line.set_data_3d(
                right_trail[:, 0],
                right_trail[:, 1],
                np.zeros(right_trail.shape[0], dtype=np.float32),
            )
        else:
            left_traj_line.set_data_3d([], [], [])
            right_traj_line.set_data_3d([], [], [])
        return artists

    interval = int(round(1000.0 / fps))
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=t_total,
        interval=interval,
        blit=False,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer="pillow", fps=fps)
    if show:
        plt.show()
    else:
        plt.close(fig)
    del ani


__all__ = [
    "animation_plot",
    "animation_plot_frame",
    "animation_plot_from_375",
    "visualize_side_by_side_from_375",
]
