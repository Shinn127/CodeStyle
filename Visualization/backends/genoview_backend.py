from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent
REPO_ROOT = NEWCODE_DIR.parent

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

# The full render chain and the GenoView module/resources live in genoview_pipeline; the
# names below stay importable from this module for existing callers.
from Visualization.backends.genoview_pipeline import (  # noqa: E402
    GENOVIEW_DIR,
    GENOVIEW_RESOURCES_DIR,
    GenoViewPipeline,
    _as_bytes,
    _load_genoview_module,
)
from Utils import quat  # noqa: E402
from Utils.motion375 import (  # noqa: E402
    TRAJECTORY_SAMPLE_STRIDE,
    decode_trajectory_context,
    reconstruct_root_track_from_375,
    reconstruct_selected_globals_from_375,
)
from Visualization.io.genoview_bvh import (  # noqa: E402
    _convert_to_geno_basis,
    _expand_selected_to_full,
    _load_meta,
    _load_motion375_clip,
    _load_segments,
    _load_template,
    _resolve_frame_range,
    _template_local_transforms,
)


def _trajectory_sample_offsets(window: int, sample_count: int) -> np.ndarray:
    offsets = np.arange(-int(window), int(window), TRAJECTORY_SAMPLE_STRIDE, dtype=np.int32)
    if len(offsets) != sample_count:
        center = sample_count // 2
        offsets = (np.arange(sample_count, dtype=np.int32) - center) * TRAJECTORY_SAMPLE_STRIDE
    return offsets


def _prepare_trajectory_debug(
    clip: np.ndarray,
    *,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj_pos_local, traj_dir_local, _, _ = decode_trajectory_context(clip)
    traj_pos_world = quat.mul_vec(root_rot[:, None, :], traj_pos_local) + root_pos[:, None, :]
    traj_dir_world = quat.mul_vec(root_rot[:, None, :], traj_dir_local)
    offsets = _trajectory_sample_offsets(window, traj_pos_local.shape[1])
    return np.asarray(traj_pos_world, dtype=np.float32), np.asarray(traj_dir_world, dtype=np.float32), offsets


def _draw_trajectory_overlay(
    gv: Any,
    *,
    positions: np.ndarray,
    directions: np.ndarray,
    offsets: np.ndarray,
    frame_index: int,
    future_only: bool,
) -> None:
    pts = positions[frame_index]
    dirs = directions[frame_index]
    visible = np.where(offsets >= 0)[0] if future_only else np.arange(len(offsets))
    if visible.size <= 1:
        return

    current_color = gv.Color(255, 196, 61, 255)
    future_color = gv.Color(67, 181, 129, 255)
    past_color = gv.Color(116, 133, 151, 220)

    prev = pts[visible[0]]
    for idx in visible[1:]:
        pos = pts[idx]
        direction = dirs[idx]
        color = future_color if offsets[idx] > 0 else past_color if offsets[idx] < 0 else current_color
        gv.DrawSphereWires(gv.Vector3(*pos), 0.045, 4, 10, color)
        gv.DrawLine3D(gv.Vector3(*prev), gv.Vector3(*pos), color)
        gv.DrawLine3D(gv.Vector3(*pos), gv.Vector3(*(pos + 0.45 * direction)), color)
        prev = pos


def _prepare_full_animation(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    db_path = args.db.expanduser().resolve()
    template_bvh = args.template_bvh.expanduser().resolve()
    meta = _load_meta(db_path)
    segments = _load_segments(db_path)
    total_frames = int(meta["num_frames"])
    start, stop, segment = _resolve_frame_range(args=args, total_frames=total_frames, segments=segments)
    clip = _load_motion375_clip(db_path, start=start, stop=stop, total_frames=total_frames)

    fps = int(args.fps)
    if fps <= 0:
        raise ValueError(f"--fps must be > 0, got {fps}")

    full_pos, full_rot, full_parents, info = _prepare_full_animation_from_features(
        clip,
        fps=fps,
        dt=1.0 / float(fps),
        root_wy_in_degrees=not args.root_wy_rad,
        use_global=True,
        coordinate_convert=bool(meta.get("coordinate_convert", False)),
        template_bvh=template_bvh,
        template_frame=args.template_frame,
        trajectory_window=int(meta.get("window", 60)),
    )
    info = {
        **info,
        "db_path": db_path,
        "segment": segment,
        "meta": meta,
        "frame_range": (start, stop),
    }
    return full_pos, full_rot, full_parents, info


def _prepare_full_animation_from_features(
    features_375: np.ndarray,
    *,
    fps: int,
    dt: float,
    root_wy_in_degrees: bool,
    use_global: bool,
    coordinate_convert: bool,
    template_bvh: Path,
    template_frame: int,
    trajectory_window: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clip = np.asarray(features_375, dtype=np.float32)
    if clip.ndim != 2 or clip.shape[1] != 375 or clip.shape[0] == 0:
        raise ValueError(f"Expected clip shape (T, 375), got {clip.shape}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    template_bvh = template_bvh.expanduser().resolve()
    selected_pos, selected_rot = reconstruct_selected_globals_from_375(
        clip,
        dt=float(dt),
        root_wy_in_degrees=bool(root_wy_in_degrees),
        use_global=bool(use_global),
    )
    selected_pos, selected_rot = _convert_to_geno_basis(
        selected_pos,
        selected_rot,
        enabled=bool(coordinate_convert),
    )

    template = _load_template(template_bvh)
    full_names = list(template["names"])
    full_parents = np.asarray(template["parents"], dtype=np.int32)
    template_local_pos, template_local_rot = _template_local_transforms(template, int(template_frame))
    full_pos, full_rot = _expand_selected_to_full(
        selected_pos,
        selected_rot,
        full_names=full_names,
        full_parents=full_parents,
        template_local_pos=template_local_pos,
        template_local_rot=template_local_rot,
    )

    if use_global:
        root_pos_seq, root_rot_seq = reconstruct_root_track_from_375(
            clip,
            dt=float(dt),
            root_wy_in_degrees=bool(root_wy_in_degrees),
        )
        if coordinate_convert:
            root_pos_seq, root_rot_seq = _convert_to_geno_basis(
                root_pos_seq[:, None, :],
                root_rot_seq[:, None, :],
                enabled=True,
            )
            root_pos_seq = root_pos_seq[:, 0]
            root_rot_seq = root_rot_seq[:, 0]
        traj_pos_world, traj_dir_world, traj_offsets = _prepare_trajectory_debug(
            clip,
            root_pos=root_pos_seq,
            root_rot=root_rot_seq,
            window=int(trajectory_window),
        )
    else:
        traj_pos_world = np.zeros((clip.shape[0], 0, 3), dtype=np.float32)
        traj_dir_world = np.zeros((clip.shape[0], 0, 3), dtype=np.float32)
        traj_offsets = np.zeros((0,), dtype=np.int32)

    info = {
        "template_bvh": template_bvh,
        "num_frames": int(full_pos.shape[0]),
        "full_names": full_names,
        "trajectory_positions": np.asarray(traj_pos_world, dtype=np.float32),
        "trajectory_directions": np.asarray(traj_dir_world, dtype=np.float32),
        "trajectory_offsets": np.asarray(traj_offsets, dtype=np.int32),
        "coordinate_convert": bool(coordinate_convert),
        "use_global": bool(use_global),
    }
    return full_pos, full_rot, full_parents, info


def _resolve_coordinate_convert_flag(
    *,
    coordinate_convert: bool | None,
    db_path: Path | None,
) -> bool:
    if coordinate_convert is not None:
        return bool(coordinate_convert)
    if db_path is None:
        return False

    db_path = Path(db_path).expanduser().resolve()
    if not db_path.is_dir() or not (db_path / "meta.json").exists():
        return False
    try:
        return bool(_load_meta(db_path).get("coordinate_convert", False))
    except Exception:
        return False


def visualize_dual_motion375_genoview_from_375(
    *,
    ref_features_375: np.ndarray,
    pred_features_375: np.ndarray,
    fps: int,
    dt: float,
    use_global: bool,
    root_wy_in_degrees: bool = True,
    coordinate_convert: bool | None = None,
    db_path: Path | None = None,
    template_bvh: Path = GENOVIEW_RESOURCES_DIR / "Geno_stance.bvh",
    template_frame: int = 0,
    title: str | None = None,
    screen_width: int = 1600,
    screen_height: int = 900,
    model_gap: float = 2.4,
    draw_skeleton: bool = False,
    draw_trajectory: bool = False,
    trajectory_future_only: bool = False,
    start_paused: bool = False,
    screenshots: dict[int, Path] | None = None,
) -> None:
    """`screenshots` maps an animation frame index to a PNG path written with raylib's
    TakeScreenshot during that frame (for headless/archival captures of a live run)."""
    coordinate_convert_enabled = _resolve_coordinate_convert_flag(
        coordinate_convert=coordinate_convert,
        db_path=db_path,
    )

    ref_full_pos, ref_full_rot, ref_full_parents, ref_info = _prepare_full_animation_from_features(
        ref_features_375,
        fps=fps,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
        coordinate_convert=coordinate_convert_enabled,
        template_bvh=template_bvh,
        template_frame=template_frame,
    )
    pred_full_pos, pred_full_rot, pred_full_parents, pred_info = _prepare_full_animation_from_features(
        pred_features_375,
        fps=fps,
        dt=dt,
        root_wy_in_degrees=root_wy_in_degrees,
        use_global=use_global,
        coordinate_convert=coordinate_convert_enabled,
        template_bvh=template_bvh,
        template_frame=template_frame,
    )

    if ref_full_parents.shape != pred_full_parents.shape or not np.array_equal(ref_full_parents, pred_full_parents):
        raise ValueError("Reference/prediction full skeleton parents do not match")

    t_total = min(int(ref_full_pos.shape[0]), int(pred_full_pos.shape[0]))
    if t_total <= 1:
        raise ValueError(f"Need at least 2 frames for GenoView animation, got {t_total}")
    ref_full_pos = np.asarray(ref_full_pos[:t_total], dtype=np.float32)
    ref_full_rot = np.asarray(ref_full_rot[:t_total], dtype=np.float32)
    pred_full_pos = np.asarray(pred_full_pos[:t_total], dtype=np.float32)
    pred_full_rot = np.asarray(pred_full_rot[:t_total], dtype=np.float32)
    ref_traj_pos = np.asarray(ref_info["trajectory_positions"][:t_total], dtype=np.float32)
    ref_traj_dir = np.asarray(ref_info["trajectory_directions"][:t_total], dtype=np.float32)
    pred_traj_pos = np.asarray(pred_info["trajectory_positions"][:t_total], dtype=np.float32)
    pred_traj_dir = np.asarray(pred_info["trajectory_directions"][:t_total], dtype=np.float32)
    traj_offsets = np.asarray(ref_info["trajectory_offsets"], dtype=np.int32)

    ref_offset = np.array([-0.5 * float(model_gap), 0.0, 0.0], dtype=np.float32)
    pred_offset = np.array([+0.5 * float(model_gap), 0.0, 0.0], dtype=np.float32)
    ref_full_pos = ref_full_pos + ref_offset[None, None, :]
    pred_full_pos = pred_full_pos + pred_offset[None, None, :]
    if ref_traj_pos.size:
        ref_traj_pos = ref_traj_pos + ref_offset[None, None, :]
    if pred_traj_pos.size:
        pred_traj_pos = pred_traj_pos + pred_offset[None, None, :]

    gv = _load_genoview_module()
    old_cwd = Path.cwd()
    os.chdir(GENOVIEW_DIR)

    pipeline = None
    ref_model = None
    pred_model = None
    try:
        gv.SetConfigFlags(gv.FLAG_VSYNC_HINT)
        gv.InitWindow(int(screen_width), int(screen_height), (title or "motion375 GenoView compare").encode("utf-8"))
        gv.SetTargetFPS(int(fps))

        ref_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        pred_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        bind_pos, bind_rot = gv.GetModelBindPoseAsNumpyArrays(ref_model)

        camera = gv.Camera()
        camera.distance = 7.5
        camera.altitude = 0.35

        pipeline = GenoViewPipeline(
            gv,
            width=int(screen_width),
            height=int(screen_height),
            shadow_size=1024,
            shadow_area=(10.0, 8.0),
            shadow_extent=6.0,
            ground_size=20.0,
        )

        ref_color = gv.Color(110, 159, 255, 255)
        pred_color = gv.Color(255, 171, 82, 255)
        ref_skeleton_color = gv.Color(70, 105, 180, 255)
        pred_skeleton_color = gv.Color(191, 108, 10, 255)

        def overlay(gv_mod: Any, cam: Any) -> None:
            if draw_bone_transforms_ptr[0]:
                gv_mod.DrawSkeleton(ref_full_pos[animation_frame], ref_full_rot[animation_frame], ref_full_parents, ref_skeleton_color)
                gv_mod.DrawSkeleton(pred_full_pos[animation_frame], pred_full_rot[animation_frame], ref_full_parents, pred_skeleton_color)
            if draw_trajectory_ptr[0]:
                if ref_traj_pos.size:
                    _draw_trajectory_overlay(
                        gv_mod,
                        positions=ref_traj_pos,
                        directions=ref_traj_dir,
                        offsets=traj_offsets,
                        frame_index=animation_frame,
                        future_only=bool(trajectory_future_only),
                    )
                if pred_traj_pos.size:
                    _draw_trajectory_overlay(
                        gv_mod,
                        positions=pred_traj_pos,
                        directions=pred_traj_dir,
                        offsets=traj_offsets,
                        frame_index=animation_frame,
                        future_only=bool(trajectory_future_only),
                    )

        animation_frame = 0
        paused = bool(start_paused)
        draw_bone_transforms_ptr = gv.ffi.new("bool*")
        draw_bone_transforms_ptr[0] = bool(draw_skeleton)
        draw_trajectory_ptr = gv.ffi.new("bool*")
        draw_trajectory_ptr[0] = bool(draw_trajectory)

        while not gv.WindowShouldClose():
            if gv.IsKeyPressed(gv.KEY_SPACE):
                paused = not paused
            if paused and gv.IsKeyPressed(gv.KEY_RIGHT):
                animation_frame = (animation_frame + 1) % t_total
            elif paused and gv.IsKeyPressed(gv.KEY_LEFT):
                animation_frame = (animation_frame - 1) % t_total
            elif not paused:
                animation_frame = (animation_frame + 1) % t_total

            gv.UpdateModelPoseFromNumpyArrays(
                ref_model,
                bind_pos,
                bind_rot,
                ref_full_pos[animation_frame],
                ref_full_rot[animation_frame],
            )
            gv.UpdateModelPoseFromNumpyArrays(
                pred_model,
                bind_pos,
                bind_rot,
                pred_full_pos[animation_frame],
                pred_full_rot[animation_frame],
            )

            hip_ref = gv.Vector3(*ref_full_pos[animation_frame, 0])
            hip_pred = gv.Vector3(*pred_full_pos[animation_frame, 0])
            focus_target = gv.Vector3(
                0.5 * (hip_ref.x + hip_pred.x),
                0.75 + 0.5 * (hip_ref.y + hip_pred.y),
                0.5 * (hip_ref.z + hip_pred.z),
            )
            camera.update(
                focus_target,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseWheelMove(),
                gv.GetFrameTime(),
            )

            pipeline.render(
                camera,
                models=[(ref_model, ref_color), (pred_model, pred_color)],
                shadow_focus=(focus_target.x, focus_target.z),
                overlay=overlay,
            )
            gv.GuiGroupBox(gv.Rectangle(20, 10, 190, 180), b"Camera")
            gv.GuiLabel(gv.Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
            gv.GuiLabel(gv.Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
            gv.GuiLabel(gv.Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
            gv.GuiLabel(
                gv.Rectangle(30, 80, 150, 20),
                (b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z)),
            )
            gv.GuiLabel(
                gv.Rectangle(30, 100, 150, 20),
                (b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z)),
            )
            gv.GuiLabel(gv.Rectangle(30, 120, 150, 20), (b"Azimuth: %5.3f" % camera.azimuth))
            gv.GuiLabel(gv.Rectangle(30, 140, 150, 20), (b"Altitude: %5.3f" % camera.altitude))
            gv.GuiLabel(gv.Rectangle(30, 160, 150, 20), (b"Distance: %5.3f" % camera.distance))
            gv.GuiGroupBox(gv.Rectangle(int(screen_width) - 300, 10, 280, 95), b"Compare")
            gv.GuiLabel(gv.Rectangle(int(screen_width) - 290, 20, 240, 20), b"Left: Reference")
            gv.GuiLabel(gv.Rectangle(int(screen_width) - 290, 40, 240, 20), b"Right: Prediction")
            gv.GuiCheckBox(gv.Rectangle(int(screen_width) - 290, 65, 20, 20), b"Draw Transforms", draw_bone_transforms_ptr)
            gv.GuiCheckBox(gv.Rectangle(int(screen_width) - 290, 85, 20, 20), b"Draw Trajectory", draw_trajectory_ptr)
            if screenshots and animation_frame in screenshots:
                # raylib's TakeScreenshot writes relative to the process CWD (chdir'd to
                # GENOVIEW_DIR, so absolute paths collapse to their basename); grab the frame
                # and export it explicitly so the caller's path is honored.
                shot = screenshots[animation_frame]
                shot.parent.mkdir(parents=True, exist_ok=True)
                gv.rlDrawRenderBatchActive()
                image = gv.LoadImageFromScreen()
                try:
                    if not gv.ExportImage(image, str(shot).encode()):
                        raise RuntimeError(f"Screenshot export failed: {shot}")
                finally:
                    gv.UnloadImage(image)
                print(f"[Screenshot] frame {animation_frame} -> {shot}", flush=True)
            gv.EndDrawing()
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        for model in (pred_model, ref_model):
            if model is not None:
                try:
                    gv.UnloadModel(model)
                except Exception:
                    pass
        try:
            gv.CloseWindow()
        except Exception:
            pass
        os.chdir(old_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly visualize motion375 features in GenoView")
    parser.add_argument("--db", type=Path, default=NEWCODE_DIR / "database_100sty_375_memmap_nomirror")
    parser.add_argument("--template-bvh", type=Path, default=GENOVIEW_RESOURCES_DIR / "Geno_stance.bvh")
    parser.add_argument("--template-frame", type=int, default=0)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--segment-offset", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means full segment/range")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--root-wy-rad", action="store_true", help="Interpret root yaw velocity as radians/sec")
    parser.add_argument("--screen-width", type=int, default=1440)
    parser.add_argument("--screen-height", type=int, default=900)
    parser.add_argument("--title", type=str, default="motion375 GenoView")
    parser.add_argument("--draw-skeleton", action="store_true")
    parser.add_argument("--draw-trajectory", action="store_true")
    parser.add_argument("--trajectory-future-only", action="store_true")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare motion and print metadata without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_pos, full_rot, full_parents, info = _prepare_full_animation(args)

    print(f"[Info] DB                 : {info['db_path']}")
    print(f"[Info] Template BVH       : {info['template_bvh']}")
    print(f"[Info] Frames prepared    : {info['num_frames']} [{info['frame_range'][0]}, {info['frame_range'][1]})")
    print(f"[Info] FPS                : {args.fps}")
    print(f"[Info] Root wy degrees    : {not args.root_wy_rad}")
    print(f"[Info] Undo coord convert : {bool(info['meta'].get('coordinate_convert', False))}")
    if info["segment"] is not None:
        segment = info["segment"]
        print(
            "[Info] Segment meta       : "
            f"style={segment.get('style_name')} suffix={segment.get('suffix')} mirror={segment.get('mirror')}"
        )

    if args.dry_run:
        print(f"[Info] Full pose shapes   : pos={tuple(full_pos.shape)} rot={tuple(full_rot.shape)}")
        print(
            f"[Info] Trajectory shapes  : pos={tuple(info['trajectory_positions'].shape)} "
            f"dir={tuple(info['trajectory_directions'].shape)}"
        )
        return

    gv = _load_genoview_module()
    old_cwd = Path.cwd()
    os.chdir(GENOVIEW_DIR)

    pipeline = None
    geno_model = None
    try:
        gv.SetConfigFlags(gv.FLAG_VSYNC_HINT)
        gv.InitWindow(int(args.screen_width), int(args.screen_height), args.title.encode("utf-8"))
        gv.SetTargetFPS(int(args.fps))

        geno_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        bind_pos, bind_rot = gv.GetModelBindPoseAsNumpyArrays(geno_model)

        camera = gv.Camera()

        pipeline = GenoViewPipeline(
            gv,
            width=int(args.screen_width),
            height=int(args.screen_height),
            shadow_size=1024,
            shadow_area=(5.0, 5.0),
            shadow_extent=5.0,
            ground_size=20.0,
            clip_far=50.0,
        )

        animation_frame = 0
        paused = bool(args.start_paused)
        draw_bone_transforms_ptr = gv.ffi.new("bool*")
        draw_bone_transforms_ptr[0] = bool(args.draw_skeleton)
        draw_trajectory_ptr = gv.ffi.new("bool*")
        draw_trajectory_ptr[0] = bool(args.draw_trajectory)

        def overlay(gv_mod: Any, cam: Any) -> None:
            if draw_bone_transforms_ptr[0]:
                gv_mod.DrawSkeleton(full_pos[animation_frame], full_rot[animation_frame], full_parents, gv.GRAY)
            if draw_trajectory_ptr[0]:
                _draw_trajectory_overlay(
                    gv_mod,
                    positions=info["trajectory_positions"],
                    directions=info["trajectory_directions"],
                    offsets=info["trajectory_offsets"],
                    frame_index=animation_frame,
                    future_only=bool(args.trajectory_future_only),
                )

        while not gv.WindowShouldClose():
            if gv.IsKeyPressed(gv.KEY_SPACE):
                paused = not paused
            if paused and gv.IsKeyPressed(gv.KEY_RIGHT):
                animation_frame = (animation_frame + 1) % info["num_frames"]
            elif paused and gv.IsKeyPressed(gv.KEY_LEFT):
                animation_frame = (animation_frame - 1) % info["num_frames"]
            elif not paused:
                animation_frame = (animation_frame + 1) % info["num_frames"]

            gv.UpdateModelPoseFromNumpyArrays(
                geno_model,
                bind_pos,
                bind_rot,
                full_pos[animation_frame],
                full_rot[animation_frame],
            )

            hip = gv.Vector3(*full_pos[animation_frame, 0])
            camera.update(
                gv.Vector3(hip.x, 0.75, hip.z),
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseWheelMove(),
                gv.GetFrameTime(),
            )

            pipeline.render(
                camera,
                models=[(geno_model, gv.ORANGE)],
                shadow_focus=(hip.x, hip.z),
                overlay=overlay,
            )
            gv.GuiGroupBox(gv.Rectangle(20, 10, 190, 180), b"Camera")
            gv.GuiLabel(gv.Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
            gv.GuiLabel(gv.Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
            gv.GuiLabel(gv.Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
            gv.GuiLabel(
                gv.Rectangle(30, 80, 150, 20),
                (b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z)),
            )
            gv.GuiLabel(
                gv.Rectangle(30, 100, 150, 20),
                (b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z)),
            )
            gv.GuiLabel(gv.Rectangle(30, 120, 150, 20), (b"Azimuth: %5.3f" % camera.azimuth))
            gv.GuiLabel(gv.Rectangle(30, 140, 150, 20), (b"Altitude: %5.3f" % camera.altitude))
            gv.GuiLabel(gv.Rectangle(30, 160, 150, 20), (b"Distance: %5.3f" % camera.distance))
            gv.GuiGroupBox(gv.Rectangle(int(args.screen_width) - 260, 10, 240, 70), b"Rendering")
            gv.GuiCheckBox(gv.Rectangle(int(args.screen_width) - 250, 20, 20, 20), b"Draw Transforms", draw_bone_transforms_ptr)
            gv.GuiCheckBox(gv.Rectangle(int(args.screen_width) - 250, 45, 20, 20), b"Draw Trajectory", draw_trajectory_ptr)
            gv.EndDrawing()
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        if geno_model is not None:
            try:
                gv.UnloadModel(geno_model)
            except Exception:
                pass
        try:
            gv.CloseWindow()
        except Exception:
            pass
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()


__all__ = [
    "GENOVIEW_DIR",
    "GENOVIEW_RESOURCES_DIR",
    "visualize_dual_motion375_genoview_from_375",
]
