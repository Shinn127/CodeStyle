from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent
REPO_ROOT = NEWCODE_DIR.parent
GENOVIEW_DIR = REPO_ROOT / "GenoViewPython-main"
GENOVIEW_RESOURCES_DIR = GENOVIEW_DIR / "resources"

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

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


def _as_bytes(path: Path) -> bytes:
    return str(path.expanduser().resolve()).encode("utf-8")


def _load_genoview_module() -> Any:
    if not GENOVIEW_DIR.exists():
        raise FileNotFoundError(f"GenoView project directory not found: {GENOVIEW_DIR}")
    if str(GENOVIEW_DIR) not in sys.path:
        sys.path.insert(0, str(GENOVIEW_DIR))
    try:
        return importlib.import_module("genoview")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import GenoView runtime. Install the Python raylib bindings in the selected environment."
        ) from exc


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

    shadow_shader = None
    skinned_shadow_shader = None
    skinned_basic_shader = None
    basic_shader = None
    lighting_shader = None
    ssao_shader = None
    blur_shader = None
    fxaa_shader = None
    ground_model = None
    shadow_map = None
    gbuffer = None
    lighted = None
    ssao_front = None
    ssao_back = None
    ref_model = None
    pred_model = None
    try:
        gv.SetConfigFlags(gv.FLAG_VSYNC_HINT)
        gv.InitWindow(int(screen_width), int(screen_height), (title or "motion375 GenoView compare").encode("utf-8"))
        gv.SetTargetFPS(int(fps))
        gv.rlSetClipPlanes(0.01, 50.0)

        shadow_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"))
        shadow_shader_light_clip_near = gv.GetShaderLocation(shadow_shader, b"lightClipNear")
        shadow_shader_light_clip_far = gv.GetShaderLocation(shadow_shader, b"lightClipFar")

        skinned_shadow_shader = gv.LoadShader(
            _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedShadow.vs"),
            _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"),
        )
        skinned_shadow_shader_light_clip_near = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipNear")
        skinned_shadow_shader_light_clip_far = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipFar")

        skinned_basic_shader = gv.LoadShader(
            _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedBasic.vs"),
            _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"),
        )
        shader_specularity = gv.GetShaderLocation(skinned_basic_shader, b"specularity")
        shader_glossiness = gv.GetShaderLocation(skinned_basic_shader, b"glossiness")
        shader_cam_clip_near = gv.GetShaderLocation(skinned_basic_shader, b"camClipNear")
        shader_cam_clip_far = gv.GetShaderLocation(skinned_basic_shader, b"camClipFar")

        basic_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "basic.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"))
        basic_shader_specularity = gv.GetShaderLocation(basic_shader, b"specularity")
        basic_shader_glossiness = gv.GetShaderLocation(basic_shader, b"glossiness")
        basic_shader_cam_clip_near = gv.GetShaderLocation(basic_shader, b"camClipNear")
        basic_shader_cam_clip_far = gv.GetShaderLocation(basic_shader, b"camClipFar")

        lighting_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "lighting.fs"))
        lighting_shader_gbuffer_color = gv.GetShaderLocation(lighting_shader, b"gbufferColor")
        lighting_shader_gbuffer_normal = gv.GetShaderLocation(lighting_shader, b"gbufferNormal")
        lighting_shader_gbuffer_depth = gv.GetShaderLocation(lighting_shader, b"gbufferDepth")
        lighting_shader_ssao = gv.GetShaderLocation(lighting_shader, b"ssao")
        lighting_shader_cam_pos = gv.GetShaderLocation(lighting_shader, b"camPos")
        lighting_shader_cam_inv_view_proj = gv.GetShaderLocation(lighting_shader, b"camInvViewProj")
        lighting_shader_light_dir = gv.GetShaderLocation(lighting_shader, b"lightDir")
        lighting_shader_sun_color = gv.GetShaderLocation(lighting_shader, b"sunColor")
        lighting_shader_sun_strength = gv.GetShaderLocation(lighting_shader, b"sunStrength")
        lighting_shader_sky_color = gv.GetShaderLocation(lighting_shader, b"skyColor")
        lighting_shader_sky_strength = gv.GetShaderLocation(lighting_shader, b"skyStrength")
        lighting_shader_ground_strength = gv.GetShaderLocation(lighting_shader, b"groundStrength")
        lighting_shader_ambient_strength = gv.GetShaderLocation(lighting_shader, b"ambientStrength")
        lighting_shader_exposure = gv.GetShaderLocation(lighting_shader, b"exposure")
        lighting_shader_cam_clip_near = gv.GetShaderLocation(lighting_shader, b"camClipNear")
        lighting_shader_cam_clip_far = gv.GetShaderLocation(lighting_shader, b"camClipFar")

        ssao_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "ssao.fs"))
        ssao_shader_gbuffer_normal = gv.GetShaderLocation(ssao_shader, b"gbufferNormal")
        ssao_shader_gbuffer_depth = gv.GetShaderLocation(ssao_shader, b"gbufferDepth")
        ssao_shader_cam_view = gv.GetShaderLocation(ssao_shader, b"camView")
        ssao_shader_cam_proj = gv.GetShaderLocation(ssao_shader, b"camProj")
        ssao_shader_cam_inv_proj = gv.GetShaderLocation(ssao_shader, b"camInvProj")
        ssao_shader_cam_inv_view_proj = gv.GetShaderLocation(ssao_shader, b"camInvViewProj")
        ssao_shader_light_view_proj = gv.GetShaderLocation(ssao_shader, b"lightViewProj")
        ssao_shader_shadow_map = gv.GetShaderLocation(ssao_shader, b"shadowMap")
        ssao_shader_shadow_inv_resolution = gv.GetShaderLocation(ssao_shader, b"shadowInvResolution")
        ssao_shader_cam_clip_near = gv.GetShaderLocation(ssao_shader, b"camClipNear")
        ssao_shader_cam_clip_far = gv.GetShaderLocation(ssao_shader, b"camClipFar")
        ssao_shader_light_clip_near = gv.GetShaderLocation(ssao_shader, b"lightClipNear")
        ssao_shader_light_clip_far = gv.GetShaderLocation(ssao_shader, b"lightClipFar")
        ssao_shader_light_dir = gv.GetShaderLocation(ssao_shader, b"lightDir")

        blur_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "blur.fs"))
        blur_shader_gbuffer_normal = gv.GetShaderLocation(blur_shader, b"gbufferNormal")
        blur_shader_gbuffer_depth = gv.GetShaderLocation(blur_shader, b"gbufferDepth")
        blur_shader_input_texture = gv.GetShaderLocation(blur_shader, b"inputTexture")
        blur_shader_cam_inv_proj = gv.GetShaderLocation(blur_shader, b"camInvProj")
        blur_shader_cam_clip_near = gv.GetShaderLocation(blur_shader, b"camClipNear")
        blur_shader_cam_clip_far = gv.GetShaderLocation(blur_shader, b"camClipFar")
        blur_shader_inv_texture_resolution = gv.GetShaderLocation(blur_shader, b"invTextureResolution")
        blur_shader_blur_direction = gv.GetShaderLocation(blur_shader, b"blurDirection")

        fxaa_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "fxaa.fs"))
        fxaa_shader_input_texture = gv.GetShaderLocation(fxaa_shader, b"inputTexture")
        fxaa_shader_inv_texture_resolution = gv.GetShaderLocation(fxaa_shader, b"invTextureResolution")

        ground_mesh = gv.GenMeshPlane(20.0, 20.0, 10, 10)
        ground_model = gv.LoadModelFromMesh(ground_mesh)
        ground_position = gv.Vector3(0.0, -0.01, 0.0)
        zero_position = gv.Vector3(0.0, 0.0, 0.0)

        ref_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        pred_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        bind_pos, bind_rot = gv.GetModelBindPoseAsNumpyArrays(ref_model)

        camera = gv.Camera()
        camera.distance = 7.5
        camera.altitude = 0.35

        specularity_ptr = gv.ffi.new("float*")
        glossiness_ptr = gv.ffi.new("float*")
        cam_clip_near_ptr = gv.ffi.new("float*")
        cam_clip_far_ptr = gv.ffi.new("float*")
        specularity_ptr[0] = 0.5
        glossiness_ptr[0] = 10.0
        cam_clip_near_ptr[0] = 0.01
        cam_clip_far_ptr[0] = 50.0

        light_dir = gv.Vector3Normalize(gv.Vector3(0.35, -1.0, -0.35))
        shadow_light = gv.ShadowLight()
        shadow_light.target = gv.Vector3Zero()
        shadow_light.position = gv.Vector3Scale(light_dir, -6.0)
        shadow_light.up = gv.Vector3(0.0, 1.0, 0.0)
        shadow_light.width = 10.0
        shadow_light.height = 8.0
        shadow_light.near = 0.01
        shadow_light.far = 16.0

        shadow_width = 1024
        shadow_height = 1024
        shadow_inv_resolution = gv.Vector2(1.0 / shadow_width, 1.0 / shadow_height)
        shadow_map = gv.LoadShadowMap(shadow_width, shadow_height)

        gbuffer = gv.LoadGBuffer(int(screen_width), int(screen_height))
        lighted = gv.LoadRenderTexture(int(screen_width), int(screen_height))
        ssao_front = gv.LoadRenderTexture(int(screen_width), int(screen_height))
        ssao_back = gv.LoadRenderTexture(int(screen_width), int(screen_height))

        ref_color = gv.Color(110, 159, 255, 255)
        pred_color = gv.Color(255, 171, 82, 255)
        ref_skeleton_color = gv.Color(70, 105, 180, 255)
        pred_skeleton_color = gv.Color(191, 108, 10, 255)

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
            if screenshots and animation_frame in screenshots:
                shot = screenshots[animation_frame]
                shot.parent.mkdir(parents=True, exist_ok=True)
                gv.TakeScreenshot(str(shot).encode())
                print(f"[Screenshot] frame {animation_frame} -> {shot}", flush=True)
            focus_target = gv.Vector3(
                0.5 * (hip_ref.x + hip_pred.x),
                0.75 + 0.5 * (hip_ref.y + hip_pred.y),
                0.5 * (hip_ref.z + hip_pred.z),
            )
            shadow_light.target = gv.Vector3(focus_target.x, 0.0, focus_target.z)
            shadow_light.position = gv.Vector3Add(shadow_light.target, gv.Vector3Scale(light_dir, -6.0))
            camera.update(
                focus_target,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseWheelMove(),
                gv.GetFrameTime(),
            )

            gv.rlDisableColorBlend()
            gv.SetShaderValue(skinned_basic_shader, shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.BeginDrawing()

            gv.BeginShadowMap(shadow_map, shadow_light)
            light_view_proj = gv.MatrixMultiply(gv.rlGetMatrixModelview(), gv.rlGetMatrixProjection())
            light_clip_near = gv.rlGetCullDistanceNear()
            light_clip_far = gv.rlGetCullDistanceFar()
            light_clip_near_ptr = gv.ffi.new("float*")
            light_clip_far_ptr = gv.ffi.new("float*")
            light_clip_near_ptr[0] = light_clip_near
            light_clip_far_ptr[0] = light_clip_far
            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_shadow_shader, skinned_shadow_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_shadow_shader, skinned_shadow_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            ground_model.materials[0].shader = shadow_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.WHITE)
            ref_model.materials[0].shader = skinned_shadow_shader
            pred_model.materials[0].shader = skinned_shadow_shader
            gv.DrawModel(ref_model, zero_position, 1.0, gv.WHITE)
            gv.DrawModel(pred_model, zero_position, 1.0, gv.WHITE)
            gv.EndShadowMap()

            gv.BeginGBuffer(gbuffer, camera.cam3d)
            cam_view = gv.rlGetMatrixModelview()
            cam_proj = gv.rlGetMatrixProjection()
            cam_inv_proj = gv.MatrixInvert(cam_proj)
            cam_inv_view_proj = gv.MatrixInvert(gv.MatrixMultiply(cam_view, cam_proj))
            cam_clip_near = gv.rlGetCullDistanceNear()
            cam_clip_far = gv.rlGetCullDistanceFar()
            cam_clip_near_ptr[0] = cam_clip_near
            cam_clip_far_ptr[0] = cam_clip_far
            gv.SetShaderValue(basic_shader, basic_shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            ground_model.materials[0].shader = basic_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.Color(190, 190, 190, 255))
            ref_model.materials[0].shader = skinned_basic_shader
            pred_model.materials[0].shader = skinned_basic_shader
            gv.DrawModel(ref_model, zero_position, 1.0, ref_color)
            gv.DrawModel(pred_model, zero_position, 1.0, pred_color)
            gv.EndGBuffer(int(screen_width), int(screen_height))

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(ssao_shader)
            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_view, cam_view)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_proj, cam_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_light_view_proj, light_view_proj)
            gv.SetShaderValueShadowMap(ssao_shader, ssao_shader_shadow_map, shadow_map)
            gv.SetShaderValue(ssao_shader, ssao_shader_shadow_inv_resolution, gv.ffi.addressof(shadow_inv_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)
            gv.ClearBackground(gv.WHITE)
            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0.0, 0.0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_back)
            gv.BeginShaderMode(blur_shader)
            blur_direction = gv.Vector2(1.0, 0.0)
            blur_inv_texture_resolution = gv.Vector2(1.0 / ssao_front.texture.width, 1.0 / ssao_front.texture.height)
            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_front.texture)
            gv.SetShaderValueMatrix(blur_shader, blur_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(blur_shader, blur_shader_inv_texture_resolution, gv.ffi.addressof(blur_inv_texture_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                ssao_back.texture,
                gv.Rectangle(0, 0, ssao_back.texture.width, -ssao_back.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(blur_shader)
            blur_direction = gv.Vector2(0.0, 1.0)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_back.texture)
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(lighted)
            gv.BeginShaderMode(lighting_shader)
            sun_color = gv.Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
            sun_strength_ptr = gv.ffi.new("float*")
            sky_color = gv.Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
            sky_strength_ptr = gv.ffi.new("float*")
            ground_strength_ptr = gv.ffi.new("float*")
            ambient_strength_ptr = gv.ffi.new("float*")
            exposure_ptr = gv.ffi.new("float*")
            sun_strength_ptr[0] = 0.25
            sky_strength_ptr[0] = 0.15
            ground_strength_ptr[0] = 0.1
            ambient_strength_ptr[0] = 1.0
            exposure_ptr[0] = 0.9
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_color, gbuffer.color)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_ssao, ssao_front.texture)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_pos, gv.ffi.addressof(camera.cam3d.position), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValueMatrix(lighting_shader, lighting_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValue(lighting_shader, lighting_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_color, gv.ffi.addressof(sun_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_strength, sun_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_color, gv.ffi.addressof(sky_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_strength, sky_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ground_strength, ground_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ambient_strength, ambient_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_exposure, exposure_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.ClearBackground(gv.RAYWHITE)
            gv.DrawTextureRec(
                gbuffer.color,
                gv.Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            gv.BeginMode3D(camera.cam3d)
            if draw_bone_transforms_ptr[0]:
                gv.DrawSkeleton(ref_full_pos[animation_frame], ref_full_rot[animation_frame], ref_full_parents, ref_skeleton_color)
                gv.DrawSkeleton(pred_full_pos[animation_frame], pred_full_rot[animation_frame], ref_full_parents, pred_skeleton_color)
            if draw_trajectory_ptr[0]:
                if ref_traj_pos.size:
                    _draw_trajectory_overlay(
                        gv,
                        positions=ref_traj_pos,
                        directions=ref_traj_dir,
                        offsets=traj_offsets,
                        frame_index=animation_frame,
                        future_only=bool(trajectory_future_only),
                    )
                if pred_traj_pos.size:
                    _draw_trajectory_overlay(
                        gv,
                        positions=pred_traj_pos,
                        directions=pred_traj_dir,
                        offsets=traj_offsets,
                        frame_index=animation_frame,
                        future_only=bool(trajectory_future_only),
                    )
            gv.EndMode3D()
            gv.EndTextureMode()

            gv.BeginShaderMode(fxaa_shader)
            fxaa_inv_texture_resolution = gv.Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
            gv.SetShaderValueTexture(fxaa_shader, fxaa_shader_input_texture, lighted.texture)
            gv.SetShaderValue(fxaa_shader, fxaa_shader_inv_texture_resolution, gv.ffi.addressof(fxaa_inv_texture_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                lighted.texture,
                gv.Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            gv.rlEnableColorBlend()
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
            gv.EndDrawing()
    finally:
        for target in (lighted, ssao_back, ssao_front):
            if target is not None:
                try:
                    gv.UnloadRenderTexture(target)
                except Exception:
                    pass
        if gbuffer is not None:
            try:
                gv.UnloadGBuffer(gbuffer)
            except Exception:
                pass
        if shadow_map is not None:
            try:
                gv.UnloadShadowMap(shadow_map)
            except Exception:
                pass
        for model in (pred_model, ref_model):
            if model is not None:
                try:
                    gv.UnloadModel(model)
                except Exception:
                    pass
        if ground_model is not None:
            try:
                gv.UnloadModel(ground_model)
            except Exception:
                pass
        for shader in (
            fxaa_shader,
            blur_shader,
            ssao_shader,
            lighting_shader,
            basic_shader,
            skinned_basic_shader,
            skinned_shadow_shader,
            shadow_shader,
        ):
            if shader is not None:
                try:
                    gv.UnloadShader(shader)
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

    shadow_shader = None
    skinned_shadow_shader = None
    skinned_basic_shader = None
    basic_shader = None
    lighting_shader = None
    ssao_shader = None
    blur_shader = None
    fxaa_shader = None
    ground_model = None
    shadow_map = None
    gbuffer = None
    lighted = None
    ssao_front = None
    ssao_back = None
    geno_model = None
    try:
        gv.SetConfigFlags(gv.FLAG_VSYNC_HINT)
        gv.InitWindow(int(args.screen_width), int(args.screen_height), args.title.encode("utf-8"))
        gv.SetTargetFPS(int(args.fps))
        gv.rlSetClipPlanes(0.01, 50.0)

        shadow_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"))
        shadow_shader_light_clip_near = gv.GetShaderLocation(shadow_shader, b"lightClipNear")
        shadow_shader_light_clip_far = gv.GetShaderLocation(shadow_shader, b"lightClipFar")

        skinned_shadow_shader = gv.LoadShader(
            _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedShadow.vs"),
            _as_bytes(GENOVIEW_RESOURCES_DIR / "shadow.fs"),
        )
        skinned_shadow_shader_light_clip_near = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipNear")
        skinned_shadow_shader_light_clip_far = gv.GetShaderLocation(skinned_shadow_shader, b"lightClipFar")

        skinned_basic_shader = gv.LoadShader(
            _as_bytes(GENOVIEW_RESOURCES_DIR / "skinnedBasic.vs"),
            _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"),
        )
        shader_specularity = gv.GetShaderLocation(skinned_basic_shader, b"specularity")
        shader_glossiness = gv.GetShaderLocation(skinned_basic_shader, b"glossiness")
        shader_cam_clip_near = gv.GetShaderLocation(skinned_basic_shader, b"camClipNear")
        shader_cam_clip_far = gv.GetShaderLocation(skinned_basic_shader, b"camClipFar")

        basic_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "basic.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "basic.fs"))
        basic_shader_specularity = gv.GetShaderLocation(basic_shader, b"specularity")
        basic_shader_glossiness = gv.GetShaderLocation(basic_shader, b"glossiness")
        basic_shader_cam_clip_near = gv.GetShaderLocation(basic_shader, b"camClipNear")
        basic_shader_cam_clip_far = gv.GetShaderLocation(basic_shader, b"camClipFar")

        lighting_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "lighting.fs"))
        lighting_shader_gbuffer_color = gv.GetShaderLocation(lighting_shader, b"gbufferColor")
        lighting_shader_gbuffer_normal = gv.GetShaderLocation(lighting_shader, b"gbufferNormal")
        lighting_shader_gbuffer_depth = gv.GetShaderLocation(lighting_shader, b"gbufferDepth")
        lighting_shader_ssao = gv.GetShaderLocation(lighting_shader, b"ssao")
        lighting_shader_cam_pos = gv.GetShaderLocation(lighting_shader, b"camPos")
        lighting_shader_cam_inv_view_proj = gv.GetShaderLocation(lighting_shader, b"camInvViewProj")
        lighting_shader_light_dir = gv.GetShaderLocation(lighting_shader, b"lightDir")
        lighting_shader_sun_color = gv.GetShaderLocation(lighting_shader, b"sunColor")
        lighting_shader_sun_strength = gv.GetShaderLocation(lighting_shader, b"sunStrength")
        lighting_shader_sky_color = gv.GetShaderLocation(lighting_shader, b"skyColor")
        lighting_shader_sky_strength = gv.GetShaderLocation(lighting_shader, b"skyStrength")
        lighting_shader_ground_strength = gv.GetShaderLocation(lighting_shader, b"groundStrength")
        lighting_shader_ambient_strength = gv.GetShaderLocation(lighting_shader, b"ambientStrength")
        lighting_shader_exposure = gv.GetShaderLocation(lighting_shader, b"exposure")
        lighting_shader_cam_clip_near = gv.GetShaderLocation(lighting_shader, b"camClipNear")
        lighting_shader_cam_clip_far = gv.GetShaderLocation(lighting_shader, b"camClipFar")

        ssao_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "ssao.fs"))
        ssao_shader_gbuffer_normal = gv.GetShaderLocation(ssao_shader, b"gbufferNormal")
        ssao_shader_gbuffer_depth = gv.GetShaderLocation(ssao_shader, b"gbufferDepth")
        ssao_shader_cam_view = gv.GetShaderLocation(ssao_shader, b"camView")
        ssao_shader_cam_proj = gv.GetShaderLocation(ssao_shader, b"camProj")
        ssao_shader_cam_inv_proj = gv.GetShaderLocation(ssao_shader, b"camInvProj")
        ssao_shader_cam_inv_view_proj = gv.GetShaderLocation(ssao_shader, b"camInvViewProj")
        ssao_shader_light_view_proj = gv.GetShaderLocation(ssao_shader, b"lightViewProj")
        ssao_shader_shadow_map = gv.GetShaderLocation(ssao_shader, b"shadowMap")
        ssao_shader_shadow_inv_resolution = gv.GetShaderLocation(ssao_shader, b"shadowInvResolution")
        ssao_shader_cam_clip_near = gv.GetShaderLocation(ssao_shader, b"camClipNear")
        ssao_shader_cam_clip_far = gv.GetShaderLocation(ssao_shader, b"camClipFar")
        ssao_shader_light_clip_near = gv.GetShaderLocation(ssao_shader, b"lightClipNear")
        ssao_shader_light_clip_far = gv.GetShaderLocation(ssao_shader, b"lightClipFar")
        ssao_shader_light_dir = gv.GetShaderLocation(ssao_shader, b"lightDir")

        blur_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "blur.fs"))
        blur_shader_gbuffer_normal = gv.GetShaderLocation(blur_shader, b"gbufferNormal")
        blur_shader_gbuffer_depth = gv.GetShaderLocation(blur_shader, b"gbufferDepth")
        blur_shader_input_texture = gv.GetShaderLocation(blur_shader, b"inputTexture")
        blur_shader_cam_inv_proj = gv.GetShaderLocation(blur_shader, b"camInvProj")
        blur_shader_cam_clip_near = gv.GetShaderLocation(blur_shader, b"camClipNear")
        blur_shader_cam_clip_far = gv.GetShaderLocation(blur_shader, b"camClipFar")
        blur_shader_inv_texture_resolution = gv.GetShaderLocation(blur_shader, b"invTextureResolution")
        blur_shader_blur_direction = gv.GetShaderLocation(blur_shader, b"blurDirection")

        fxaa_shader = gv.LoadShader(_as_bytes(GENOVIEW_RESOURCES_DIR / "post.vs"), _as_bytes(GENOVIEW_RESOURCES_DIR / "fxaa.fs"))
        fxaa_shader_input_texture = gv.GetShaderLocation(fxaa_shader, b"inputTexture")
        fxaa_shader_inv_texture_resolution = gv.GetShaderLocation(fxaa_shader, b"invTextureResolution")

        ground_mesh = gv.GenMeshPlane(20.0, 20.0, 10, 10)
        ground_model = gv.LoadModelFromMesh(ground_mesh)
        ground_position = gv.Vector3(0.0, -0.01, 0.0)

        geno_model = gv.LoadGenoModel(_as_bytes(GENOVIEW_RESOURCES_DIR / "Geno.bin"))
        bind_pos, bind_rot = gv.GetModelBindPoseAsNumpyArrays(geno_model)

        geno_position = gv.Vector3(0.0, 0.0, 0.0)
        camera = gv.Camera()

        specularity_ptr = gv.ffi.new("float*")
        glossiness_ptr = gv.ffi.new("float*")
        cam_clip_near_ptr = gv.ffi.new("float*")
        cam_clip_far_ptr = gv.ffi.new("float*")
        specularity_ptr[0] = 0.5
        glossiness_ptr[0] = 10.0
        cam_clip_near_ptr[0] = 0.01
        cam_clip_far_ptr[0] = 50.0

        light_dir = gv.Vector3Normalize(gv.Vector3(0.35, -1.0, -0.35))
        shadow_light = gv.ShadowLight()
        shadow_light.target = gv.Vector3Zero()
        shadow_light.position = gv.Vector3Scale(light_dir, -5.0)
        shadow_light.up = gv.Vector3(0.0, 1.0, 0.0)
        shadow_light.width = 5.0
        shadow_light.height = 5.0
        shadow_light.near = 0.01
        shadow_light.far = 10.0

        shadow_width = 1024
        shadow_height = 1024
        shadow_inv_resolution = gv.Vector2(1.0 / shadow_width, 1.0 / shadow_height)
        shadow_map = gv.LoadShadowMap(shadow_width, shadow_height)

        gbuffer = gv.LoadGBuffer(int(args.screen_width), int(args.screen_height))
        lighted = gv.LoadRenderTexture(int(args.screen_width), int(args.screen_height))
        ssao_front = gv.LoadRenderTexture(int(args.screen_width), int(args.screen_height))
        ssao_back = gv.LoadRenderTexture(int(args.screen_width), int(args.screen_height))

        animation_frame = 0
        paused = bool(args.start_paused)
        draw_bone_transforms_ptr = gv.ffi.new("bool*")
        draw_bone_transforms_ptr[0] = bool(args.draw_skeleton)
        draw_trajectory_ptr = gv.ffi.new("bool*")
        draw_trajectory_ptr[0] = bool(args.draw_trajectory)

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
            shadow_light.target = gv.Vector3(hip.x, 0.0, hip.z)
            shadow_light.position = gv.Vector3Add(shadow_light.target, gv.Vector3Scale(light_dir, -5.0))
            camera.update(
                gv.Vector3(hip.x, 0.75, hip.z),
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(0) else 0.0,
                gv.GetMouseDelta().x if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseDelta().y if gv.IsKeyDown(gv.KEY_LEFT_CONTROL) and gv.IsMouseButtonDown(1) else 0.0,
                gv.GetMouseWheelMove(),
                gv.GetFrameTime(),
            )

            gv.rlDisableColorBlend()
            gv.SetShaderValue(skinned_basic_shader, shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.BeginDrawing()

            gv.BeginShadowMap(shadow_map, shadow_light)
            light_view_proj = gv.MatrixMultiply(gv.rlGetMatrixModelview(), gv.rlGetMatrixProjection())
            light_clip_near = gv.rlGetCullDistanceNear()
            light_clip_far = gv.rlGetCullDistanceFar()
            light_clip_near_ptr = gv.ffi.new("float*")
            light_clip_far_ptr = gv.ffi.new("float*")
            light_clip_near_ptr[0] = light_clip_near
            light_clip_far_ptr[0] = light_clip_far
            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(shadow_shader, shadow_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_shadow_shader, skinned_shadow_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_shadow_shader, skinned_shadow_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            ground_model.materials[0].shader = shadow_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.WHITE)
            geno_model.materials[0].shader = skinned_shadow_shader
            gv.DrawModel(geno_model, geno_position, 1.0, gv.WHITE)
            gv.EndShadowMap()

            gv.BeginGBuffer(gbuffer, camera.cam3d)
            cam_view = gv.rlGetMatrixModelview()
            cam_proj = gv.rlGetMatrixProjection()
            cam_inv_proj = gv.MatrixInvert(cam_proj)
            cam_inv_view_proj = gv.MatrixInvert(gv.MatrixMultiply(cam_view, cam_proj))
            cam_clip_near = gv.rlGetCullDistanceNear()
            cam_clip_far = gv.rlGetCullDistanceFar()
            cam_clip_near_ptr[0] = cam_clip_near
            cam_clip_far_ptr[0] = cam_clip_far
            gv.SetShaderValue(basic_shader, basic_shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(basic_shader, basic_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_specularity, specularity_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_glossiness, glossiness_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(skinned_basic_shader, shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            ground_model.materials[0].shader = basic_shader
            gv.DrawModel(ground_model, ground_position, 1.0, gv.Color(190, 190, 190, 255))
            geno_model.materials[0].shader = skinned_basic_shader
            gv.DrawModel(geno_model, geno_position, 1.0, gv.ORANGE)
            gv.EndGBuffer(int(args.screen_width), int(args.screen_height))

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(ssao_shader)
            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(ssao_shader, ssao_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_view, cam_view)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_proj, cam_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValueMatrix(ssao_shader, ssao_shader_light_view_proj, light_view_proj)
            gv.SetShaderValueShadowMap(ssao_shader, ssao_shader_shadow_map, shadow_map)
            gv.SetShaderValue(ssao_shader, ssao_shader_shadow_inv_resolution, gv.ffi.addressof(shadow_inv_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_near, light_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_clip_far, light_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(ssao_shader, ssao_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)
            gv.ClearBackground(gv.WHITE)
            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0.0, 0.0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_back)
            gv.BeginShaderMode(blur_shader)
            blur_direction = gv.Vector2(1.0, 0.0)
            blur_inv_texture_resolution = gv.Vector2(1.0 / ssao_front.texture.width, 1.0 / ssao_front.texture.height)
            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(blur_shader, blur_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_front.texture)
            gv.SetShaderValueMatrix(blur_shader, blur_shader_cam_inv_proj, cam_inv_proj)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(blur_shader, blur_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(blur_shader, blur_shader_inv_texture_resolution, gv.ffi.addressof(blur_inv_texture_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                ssao_back.texture,
                gv.Rectangle(0, 0, ssao_back.texture.width, -ssao_back.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(ssao_front)
            gv.BeginShaderMode(blur_shader)
            blur_direction = gv.Vector2(0.0, 1.0)
            gv.SetShaderValueTexture(blur_shader, blur_shader_input_texture, ssao_back.texture)
            gv.SetShaderValue(blur_shader, blur_shader_blur_direction, gv.ffi.addressof(blur_direction), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                ssao_front.texture,
                gv.Rectangle(0, 0, ssao_front.texture.width, -ssao_front.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()
            gv.EndTextureMode()

            gv.BeginTextureMode(lighted)
            gv.BeginShaderMode(lighting_shader)
            sun_color = gv.Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
            sun_strength_ptr = gv.ffi.new("float*")
            sky_color = gv.Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
            sky_strength_ptr = gv.ffi.new("float*")
            ground_strength_ptr = gv.ffi.new("float*")
            ambient_strength_ptr = gv.ffi.new("float*")
            exposure_ptr = gv.ffi.new("float*")
            sun_strength_ptr[0] = 0.25
            sky_strength_ptr[0] = 0.15
            ground_strength_ptr[0] = 0.1
            ambient_strength_ptr[0] = 1.0
            exposure_ptr[0] = 0.9
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_color, gbuffer.color)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_normal, gbuffer.normal)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_gbuffer_depth, gbuffer.depth)
            gv.SetShaderValueTexture(lighting_shader, lighting_shader_ssao, ssao_front.texture)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_pos, gv.ffi.addressof(camera.cam3d.position), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValueMatrix(lighting_shader, lighting_shader_cam_inv_view_proj, cam_inv_view_proj)
            gv.SetShaderValue(lighting_shader, lighting_shader_light_dir, gv.ffi.addressof(light_dir), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_color, gv.ffi.addressof(sun_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sun_strength, sun_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_color, gv.ffi.addressof(sky_color), gv.SHADER_UNIFORM_VEC3)
            gv.SetShaderValue(lighting_shader, lighting_shader_sky_strength, sky_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ground_strength, ground_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_ambient_strength, ambient_strength_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_exposure, exposure_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_near, cam_clip_near_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.SetShaderValue(lighting_shader, lighting_shader_cam_clip_far, cam_clip_far_ptr, gv.SHADER_UNIFORM_FLOAT)
            gv.ClearBackground(gv.RAYWHITE)
            gv.DrawTextureRec(
                gbuffer.color,
                gv.Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            gv.BeginMode3D(camera.cam3d)
            if draw_bone_transforms_ptr[0]:
                gv.DrawSkeleton(full_pos[animation_frame], full_rot[animation_frame], full_parents, gv.GRAY)
            if draw_trajectory_ptr[0]:
                _draw_trajectory_overlay(
                    gv,
                    positions=info["trajectory_positions"],
                    directions=info["trajectory_directions"],
                    offsets=info["trajectory_offsets"],
                    frame_index=animation_frame,
                    future_only=bool(args.trajectory_future_only),
                )
            gv.EndMode3D()
            gv.EndTextureMode()

            gv.BeginShaderMode(fxaa_shader)
            fxaa_inv_texture_resolution = gv.Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
            gv.SetShaderValueTexture(fxaa_shader, fxaa_shader_input_texture, lighted.texture)
            gv.SetShaderValue(fxaa_shader, fxaa_shader_inv_texture_resolution, gv.ffi.addressof(fxaa_inv_texture_resolution), gv.SHADER_UNIFORM_VEC2)
            gv.DrawTextureRec(
                lighted.texture,
                gv.Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
                gv.Vector2(0, 0),
                gv.WHITE,
            )
            gv.EndShaderMode()

            gv.rlEnableColorBlend()
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
        for target in (lighted, ssao_back, ssao_front):
            if target is not None:
                try:
                    gv.UnloadRenderTexture(target)
                except Exception:
                    pass
        if gbuffer is not None:
            try:
                gv.UnloadGBuffer(gbuffer)
            except Exception:
                pass
        if shadow_map is not None:
            try:
                gv.UnloadShadowMap(shadow_map)
            except Exception:
                pass
        if geno_model is not None:
            try:
                gv.UnloadModel(geno_model)
            except Exception:
                pass
        if ground_model is not None:
            try:
                gv.UnloadModel(ground_model)
            except Exception:
                pass
        for shader in (
            fxaa_shader,
            blur_shader,
            ssao_shader,
            lighting_shader,
            basic_shader,
            skinned_basic_shader,
            skinned_shadow_shader,
            shadow_shader,
        ):
            if shader is not None:
                try:
                    gv.UnloadShader(shader)
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
