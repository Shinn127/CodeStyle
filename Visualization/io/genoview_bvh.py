from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent
REPO_ROOT = NEWCODE_DIR.parent
GENOVIEW_RESOURCES_DIR = REPO_ROOT / "GenoViewPython-main" / "resources"

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils import bvh, quat  # noqa: E402
from Utils.motion375 import SELECTED_24_NAMES, reconstruct_selected_globals_from_375  # noqa: E402


MEMMAP_META_FILE = "meta.json"
MEMMAP_X_FILE = "X.f32"
MEMMAP_SEGMENTS_FILE = "segments.json"
FEATURE_DIM = 375


def _load_sci_rotation():
    try:
        from scipy.spatial.transform import Rotation as sci_rotation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to export Geno-compatible BVH rotations. "
            "Install scipy in the active Python environment and retry."
        ) from exc
    return sci_rotation


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_meta(db_path: Path) -> dict[str, Any]:
    meta_path = db_path / MEMMAP_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Memmap meta not found: {meta_path}")
    meta = _load_json(meta_path)
    if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError(f"Expected feature_dim={FEATURE_DIM}, got {meta.get('feature_dim')} in {meta_path}")
    return meta


def _load_segments(db_path: Path) -> list[dict[str, Any]]:
    segments_path = db_path / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        return []
    payload = _load_json(segments_path)
    if not isinstance(payload, list):
        raise ValueError(f"{segments_path} must contain a JSON list")
    return [dict(item) for item in payload]


def _resolve_frame_range(
    *,
    args: argparse.Namespace,
    total_frames: int,
    segments: list[dict[str, Any]],
) -> tuple[int, int, dict[str, Any] | None]:
    if args.frame_start is not None:
        start = int(args.frame_start)
        if start < 0 or start >= total_frames:
            raise ValueError(f"--frame-start must be within [0, {total_frames - 1}], got {start}")
        stop = total_frames if args.num_frames <= 0 else min(start + int(args.num_frames), total_frames)
        if stop <= start:
            raise ValueError(f"Resolved empty frame range [{start}, {stop})")
        return start, stop, None

    if not segments:
        raise ValueError("segments.json is required when --frame-start is not provided")

    segment_index = int(args.segment_index)
    if segment_index < 0 or segment_index >= len(segments):
        raise ValueError(f"--segment-index must be within [0, {len(segments) - 1}], got {segment_index}")

    segment = segments[segment_index]
    start = int(segment["frame_start"]) + int(args.segment_offset)
    stop_limit = int(segment["frame_start"]) + int(segment["num_frames"])
    if start < int(segment["frame_start"]) or start >= stop_limit:
        raise ValueError(
            f"--segment-offset resolves outside segment bounds: start={start}, "
            f"segment=[{segment['frame_start']}, {stop_limit})"
        )

    stop = stop_limit if args.num_frames <= 0 else min(start + int(args.num_frames), stop_limit)
    if stop <= start:
        raise ValueError(f"Resolved empty frame range [{start}, {stop}) inside segment {segment_index}")
    return start, stop, segment


def _load_motion375_clip(db_path: Path, *, start: int, stop: int, total_frames: int) -> np.ndarray:
    x_path = db_path / MEMMAP_X_FILE
    if not x_path.exists():
        raise FileNotFoundError(f"Memmap feature file not found: {x_path}")
    x_mm = np.memmap(x_path, dtype=np.float32, mode="r", shape=(total_frames, FEATURE_DIM))
    clip = np.asarray(x_mm[start:stop], dtype=np.float32)
    if clip.ndim != 2 or clip.shape[1] != FEATURE_DIM or clip.shape[0] == 0:
        raise ValueError(f"Unexpected clip shape {clip.shape}, expected (T, {FEATURE_DIM})")
    return clip


def _load_template(template_bvh: Path) -> dict[str, Any]:
    if not template_bvh.exists():
        raise FileNotFoundError(f"Template BVH not found: {template_bvh}")
    template = bvh.load(str(template_bvh))
    if len(template["names"]) != 75:
        raise ValueError(f"Expected Geno template with 75 joints, got {len(template['names'])} from {template_bvh}")
    return template


def _template_local_transforms(template: dict[str, Any], frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    frame_count = int(template["positions"].shape[0])
    if frame_index < 0 or frame_index >= frame_count:
        raise ValueError(f"--template-frame must be within [0, {frame_count - 1}], got {frame_index}")

    local_pos_m = 0.01 * np.asarray(template["positions"][frame_index], dtype=np.float32)
    local_rot_q = quat.from_euler(
        np.radians(np.asarray(template["rotations"][frame_index], dtype=np.float32)),
        order=template["order"],
    )
    return local_pos_m, quat.normalize(local_rot_q)


def _mirror_local_animation(
    local_rot: np.ndarray,
    local_pos: np.ndarray,
    *,
    names: list[str],
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    joints_mirror = np.array(
        [
            names.index("Left" + name[5:])
            if name.startswith("Right")
            else names.index("Right" + name[4:])
            if name.startswith("Left")
            else names.index(name)
            for name in names
        ],
        dtype=np.int32,
    )
    mirror_pos = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    mirror_rot = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], dtype=np.float32)
    global_rot, global_pos = quat.fk(local_rot, local_pos, parents)
    global_pos = mirror_pos * global_pos[:, joints_mirror]
    global_rot = quat.from_xform(mirror_rot * quat.to_xform(global_rot[:, joints_mirror]))
    mirrored_rot, mirrored_pos = quat.ik(global_rot, global_pos, parents)
    return quat.normalize(quat.unroll(mirrored_rot)), np.asarray(mirrored_pos, dtype=np.float32)


def _prepare_template_locals(
    template: dict[str, Any],
    *,
    frame_index: int,
    mirror: bool,
) -> tuple[np.ndarray, np.ndarray]:
    local_pos, local_rot = _template_local_transforms(template, frame_index)
    if not mirror:
        return local_pos, local_rot
    mirrored_rot, mirrored_pos = _mirror_local_animation(
        local_rot[None, ...],
        local_pos[None, ...],
        names=list(template["names"]),
        parents=np.asarray(template["parents"], dtype=np.int32),
    )
    return np.asarray(mirrored_pos[0], dtype=np.float32), np.asarray(mirrored_rot[0], dtype=np.float32)


def _convert_to_geno_basis(
    positions: np.ndarray,
    rotations: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return positions, rotations

    positions_out = np.asarray(positions, dtype=np.float32).copy()
    positions_out[..., 0] *= -1
    xform = quat.to_xform(np.asarray(rotations, dtype=np.float32))
    xform[..., 0, 1] *= -1
    xform[..., 0, 2] *= -1
    xform[..., 1, 0] *= -1
    xform[..., 2, 0] *= -1
    rotations_out = quat.normalize(quat.from_xform(xform))
    return positions_out, rotations_out


def _expand_selected_to_full(
    selected_global_pos: np.ndarray,
    selected_global_rot: np.ndarray,
    *,
    full_names: list[str],
    full_parents: np.ndarray,
    template_local_pos: np.ndarray,
    template_local_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    name_to_full = {name: idx for idx, name in enumerate(full_names)}
    missing = [name for name in SELECTED_24_NAMES if name not in name_to_full]
    if missing:
        raise ValueError(f"Selected 24-joint mapping missing bones in template skeleton: {missing}")

    selected_indices = [name_to_full[name] for name in SELECTED_24_NAMES]
    t_total = int(selected_global_pos.shape[0])
    full_joint_count = len(full_names)

    full_pos = np.zeros((t_total, full_joint_count, 3), dtype=np.float32)
    full_rot = np.zeros((t_total, full_joint_count, 4), dtype=np.float32)
    full_rot[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    selected_mask = np.zeros((full_joint_count,), dtype=bool)
    selected_mask[selected_indices] = True
    for local_idx, full_idx in enumerate(selected_indices):
        full_pos[:, full_idx] = selected_global_pos[:, local_idx]
        full_rot[:, full_idx] = selected_global_rot[:, local_idx]

    for full_idx in range(full_joint_count):
        if selected_mask[full_idx]:
            continue
        parent_idx = int(full_parents[full_idx])
        if parent_idx < 0:
            full_pos[:, full_idx] = template_local_pos[full_idx]
            full_rot[:, full_idx] = template_local_rot[full_idx]
            continue

        parent_rot = full_rot[:, parent_idx : parent_idx + 1]
        local_rot = template_local_rot[None, full_idx : full_idx + 1]
        local_pos = template_local_pos[None, full_idx : full_idx + 1]
        full_rot[:, full_idx : full_idx + 1] = quat.mul(parent_rot, local_rot)
        full_pos[:, full_idx : full_idx + 1] = quat.mul_vec(parent_rot, local_pos) + full_pos[:, parent_idx : parent_idx + 1]

    return full_pos, quat.normalize(full_rot)


def _quat_to_euler_degrees_xyz(rotations_wxyz: np.ndarray, *, order: str) -> np.ndarray:
    sci_rotation = _load_sci_rotation()
    rotations_wxyz = quat.normalize(np.asarray(rotations_wxyz, dtype=np.float32))
    rotations_xyzw = np.concatenate([rotations_wxyz[..., 1:], rotations_wxyz[..., :1]], axis=-1)
    euler_order = sci_rotation.from_quat(rotations_xyzw.reshape(-1, 4)).as_euler(order.upper(), degrees=False)
    euler_order = np.unwrap(euler_order.reshape(rotations_wxyz.shape[:-1] + (3,)), axis=0)

    # NewCode/Utils/bvh.py::save expects rotations stored in xyz slots, then
    # picks axes according to `order` while writing channels. Our quaternions are
    # converted in the BVH channel order, so remap them back into xyz slots here.
    euler_xyz = np.zeros_like(euler_order, dtype=np.float32)
    axis_to_slot = {"x": 0, "y": 1, "z": 2}
    for src_idx, axis_name in enumerate(order):
        euler_xyz[..., axis_to_slot[axis_name]] = euler_order[..., src_idx]
    return np.degrees(euler_xyz).astype(np.float32)


def resolve_default_output_bvh_path(
    *,
    segment: dict[str, Any] | None,
    start: int,
    stop: int,
) -> Path:
    out_dir = NEWCODE_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    if segment is None:
        return out_dir / f"motion375_frames_{start}_{stop}.bvh"
    style_name = str(segment.get("style_name", "segment"))
    suffix = str(segment.get("suffix", "clip"))
    mirror_tag = "_mirror" if bool(segment.get("mirror", False)) else ""
    return out_dir / f"{style_name}_{suffix}_{start}_{stop}{mirror_tag}.bvh"


def export_motion375_segment_to_bvh(
    *,
    db_path: Path,
    template_bvh: Path,
    template_frame: int,
    segment_index: int,
    segment_offset: int,
    frame_start: int | None,
    num_frames: int,
    fps: int,
    root_wy_in_degrees: bool,
    mirror_template: bool,
    out_bvh: Path | None,
) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    db_path = db_path.expanduser().resolve()
    template_bvh = template_bvh.expanduser().resolve()
    meta = _load_meta(db_path)
    segments = _load_segments(db_path)
    total_frames = int(meta["num_frames"])
    args = argparse.Namespace(
        segment_index=segment_index,
        segment_offset=segment_offset,
        frame_start=frame_start,
        num_frames=num_frames,
    )
    start, stop, segment = _resolve_frame_range(args=args, total_frames=total_frames, segments=segments)
    clip = _load_motion375_clip(db_path, start=start, stop=stop, total_frames=total_frames)
    resolved_out_bvh = (
        out_bvh.expanduser().resolve()
        if out_bvh is not None
        else resolve_default_output_bvh_path(segment=segment, start=start, stop=stop)
    )
    resolved_out_bvh.parent.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / float(fps)
    selected_pos, selected_rot = reconstruct_selected_globals_from_375(
        clip,
        dt=dt,
        root_wy_in_degrees=bool(root_wy_in_degrees),
        use_global=True,
    )
    selected_pos, selected_rot = _convert_to_geno_basis(
        selected_pos,
        selected_rot,
        enabled=bool(meta.get("coordinate_convert", False)),
    )

    template = _load_template(template_bvh)
    full_names = list(template["names"])
    full_parents = np.asarray(template["parents"], dtype=np.int32)
    apply_mirror = bool(mirror_template or (segment is not None and bool(segment.get("mirror", False))))
    template_local_pos, template_local_rot = _prepare_template_locals(
        template,
        frame_index=template_frame,
        mirror=apply_mirror,
    )
    full_pos, full_rot = _expand_selected_to_full(
        selected_pos,
        selected_rot,
        full_names=full_names,
        full_parents=full_parents,
        template_local_pos=template_local_pos,
        template_local_rot=template_local_rot,
    )

    local_rot, local_pos = quat.ik(full_rot, full_pos, full_parents)
    local_rot = quat.normalize(quat.unroll(local_rot))
    local_rot_deg = _quat_to_euler_degrees_xyz(local_rot, order=template["order"])
    local_pos_cm = 100.0 * np.asarray(local_pos, dtype=np.float32)

    animation = {
        "rotations": local_rot_deg.astype(np.float32),
        "positions": local_pos_cm.astype(np.float32),
        "offsets": np.asarray(template["offsets"], dtype=np.float32),
        "parents": full_parents,
        "names": full_names,
        "order": str(template["order"]),
    }
    bvh.save(str(resolved_out_bvh), animation, frametime=dt, save_positions=True)

    print(f"[Info] DB                 : {db_path}")
    print(f"[Info] Template BVH       : {template_bvh}")
    print(f"[Info] Frames exported    : {clip.shape[0]} [{start}, {stop})")
    print(f"[Info] FPS                : {fps}")
    print(f"[Info] Root wy degrees    : {bool(root_wy_in_degrees)}")
    print(f"[Info] Undo coord convert : {bool(meta.get('coordinate_convert', False))}")
    print(f"[Info] Mirror template    : {apply_mirror}")
    if segment is not None:
        print(
            "[Info] Segment meta       : "
            f"style={segment.get('style_name')} suffix={segment.get('suffix')} mirror={segment.get('mirror')}"
        )
    print(f"[Done] BVH saved          : {resolved_out_bvh}")
    return resolved_out_bvh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a motion375 memmap segment as a GenoView-compatible BVH")
    parser.add_argument("--db", type=Path, default=NEWCODE_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--out-bvh", type=Path, default=None, help="Default: NewCode/exports/<derived_name>.bvh")
    parser.add_argument("--template-bvh", type=Path, default=GENOVIEW_RESOURCES_DIR / "Geno_stance.bvh")
    parser.add_argument("--template-frame", type=int, default=0)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--segment-offset", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means full segment/range")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--root-wy-rad", action="store_true", help="Interpret root yaw velocity as radians/sec")
    parser.add_argument("--mirror-template", action="store_true", help="Mirror the full Geno template before filling missing joints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_motion375_segment_to_bvh(
        db_path=args.db,
        template_bvh=args.template_bvh,
        template_frame=int(args.template_frame),
        segment_index=int(args.segment_index),
        segment_offset=int(args.segment_offset),
        frame_start=args.frame_start,
        num_frames=int(args.num_frames),
        fps=int(args.fps),
        root_wy_in_degrees=not args.root_wy_rad,
        mirror_template=bool(args.mirror_template),
        out_bvh=args.out_bvh,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GENOVIEW_RESOURCES_DIR",
    "export_motion375_segment_to_bvh",
    "resolve_default_output_bvh_path",
]
