"""Shared full GenoView render chain: shadow -> G-Buffer -> SSAO -> blur x2 -> lighting -> FXAA.

Extracted 2026-10-14 so the interactive label viewer, the single-clip preview and the dual
compare view share one implementation instead of three copies. The pipeline owns every
resource of the chain (shaders, ground mesh, shadow map, G-buffer, ping-pong targets),
caches the ffi pointers / Vector2 / Rectangle objects the chain needs per frame, and
exposes :meth:`render` which draws one frame up to (and including) the FXAA pass, leaving
the default framebuffer active so callers can draw their own GUI before ``EndDrawing``.

The caller must create the window first (``InitWindow``); the pipeline sets the clip planes
it encodes into the ssao/blur/lighting uniforms. Overlay geometry (skeleton, trajectory,
grid, ...) is passed as a callback that runs inside ``BeginMode3D`` on the lit texture, so
it is depth-tested against the shaded scene and picked up by FXAA, exactly like the
original ``genoview_backend`` loops.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
VISUALIZATION_DIR = SCRIPT_DIR.parent
NEWCODE_DIR = VISUALIZATION_DIR.parent
REPO_ROOT = NEWCODE_DIR.parent
GENOVIEW_DIR = REPO_ROOT / "GenoViewPython-main"
GENOVIEW_RESOURCES_DIR = GENOVIEW_DIR / "resources"

if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))


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


def _as_bytes(path: Path) -> bytes:
    return str(path.expanduser().resolve()).encode("utf-8")


class GenoViewPipeline:
    """Full deferred GenoView renderer for one window.

    Per-frame :meth:`render` sequence (matching ``genoview_backend``):
    shadow map -> G-buffer -> SSAO -> blur H -> blur V -> lighting -> 3D overlay hook ->
    FXAA to the window. Constant uniforms (specularity/glossiness, sun/sky/ground/ambient
    strengths, exposure, colors, inverse shadow resolution) are uploaded once at
    construction; per-frame allocations are avoided entirely.
    """

    def __init__(
        self,
        gv: Any,
        *,
        width: int,
        height: int,
        shadow_size: int = 1024,
        shadow_area: tuple[float, float] = (5.0, 5.0),
        shadow_extent: float = 5.0,
        light_dir: tuple[float, float, float] = (0.35, -1.0, -0.35),
        clip_near: float = 0.01,
        clip_far: float = 50.0,
        ground_size: float = 20.0,
        ground_height: float = -0.01,
        ground_color: tuple[int, int, int, int] = (190, 190, 190, 255),
        specularity: float = 0.5,
        glossiness: float = 10.0,
        sun_strength: float = 0.25,
        sky_strength: float = 0.15,
        ground_strength: float = 0.1,
        ambient_strength: float = 1.0,
        exposure: float = 0.9,
    ) -> None:
        self.gv = gv
        self.width = int(width)
        self.height = int(height)
        self.clip_near = float(clip_near)
        self.clip_far = float(clip_far)
        self.ground_height = float(ground_height)
        self.ground_color = gv.Color(*ground_color)
        self.shadow_extent = float(shadow_extent)
        gv.rlSetClipPlanes(self.clip_near, self.clip_far)

        res = GENOVIEW_RESOURCES_DIR
        self.shadow_shader = gv.LoadShader(_as_bytes(res / "shadow.vs"), _as_bytes(res / "shadow.fs"))
        self.skinned_shadow_shader = gv.LoadShader(_as_bytes(res / "skinnedShadow.vs"), _as_bytes(res / "shadow.fs"))
        self.skinned_basic_shader = gv.LoadShader(_as_bytes(res / "skinnedBasic.vs"), _as_bytes(res / "basic.fs"))
        self.basic_shader = gv.LoadShader(_as_bytes(res / "basic.vs"), _as_bytes(res / "basic.fs"))
        self.lighting_shader = gv.LoadShader(_as_bytes(res / "post.vs"), _as_bytes(res / "lighting.fs"))
        self.ssao_shader = gv.LoadShader(_as_bytes(res / "post.vs"), _as_bytes(res / "ssao.fs"))
        self.blur_shader = gv.LoadShader(_as_bytes(res / "post.vs"), _as_bytes(res / "blur.fs"))
        self.fxaa_shader = gv.LoadShader(_as_bytes(res / "post.vs"), _as_bytes(res / "fxaa.fs"))
        default_id = gv.rlGetShaderIdDefault()
        for name in ("shadow_shader", "skinned_shadow_shader", "skinned_basic_shader", "basic_shader",
                     "lighting_shader", "ssao_shader", "blur_shader", "fxaa_shader"):
            if getattr(self, name).id in (0, default_id):
                raise RuntimeError(f"Could not compile the {name} (check the GenoView resources)")

        loc = gv.GetShaderLocation
        # shadow shaders
        self.shadow_light_clip_near = loc(self.shadow_shader, b"lightClipNear")
        self.shadow_light_clip_far = loc(self.shadow_shader, b"lightClipFar")
        self.skinned_shadow_light_clip_near = loc(self.skinned_shadow_shader, b"lightClipNear")
        self.skinned_shadow_light_clip_far = loc(self.skinned_shadow_shader, b"lightClipFar")
        # basic / skinned basic
        self.skinned_specularity = loc(self.skinned_basic_shader, b"specularity")
        self.skinned_glossiness = loc(self.skinned_basic_shader, b"glossiness")
        self.skinned_cam_clip_near = loc(self.skinned_basic_shader, b"camClipNear")
        self.skinned_cam_clip_far = loc(self.skinned_basic_shader, b"camClipFar")
        self.basic_specularity = loc(self.basic_shader, b"specularity")
        self.basic_glossiness = loc(self.basic_shader, b"glossiness")
        self.basic_cam_clip_near = loc(self.basic_shader, b"camClipNear")
        self.basic_cam_clip_far = loc(self.basic_shader, b"camClipFar")
        # lighting
        self.lighting_gbuffer_color = loc(self.lighting_shader, b"gbufferColor")
        self.lighting_gbuffer_normal = loc(self.lighting_shader, b"gbufferNormal")
        self.lighting_gbuffer_depth = loc(self.lighting_shader, b"gbufferDepth")
        self.lighting_ssao = loc(self.lighting_shader, b"ssao")
        self.lighting_cam_pos = loc(self.lighting_shader, b"camPos")
        self.lighting_cam_inv_view_proj = loc(self.lighting_shader, b"camInvViewProj")
        self.lighting_light_dir = loc(self.lighting_shader, b"lightDir")
        self.lighting_sun_color = loc(self.lighting_shader, b"sunColor")
        self.lighting_sun_strength = loc(self.lighting_shader, b"sunStrength")
        self.lighting_sky_color = loc(self.lighting_shader, b"skyColor")
        self.lighting_sky_strength = loc(self.lighting_shader, b"skyStrength")
        self.lighting_ground_strength = loc(self.lighting_shader, b"groundStrength")
        self.lighting_ambient_strength = loc(self.lighting_shader, b"ambientStrength")
        self.lighting_exposure = loc(self.lighting_shader, b"exposure")
        self.lighting_cam_clip_near = loc(self.lighting_shader, b"camClipNear")
        self.lighting_cam_clip_far = loc(self.lighting_shader, b"camClipFar")
        # ssao
        self.ssao_gbuffer_normal = loc(self.ssao_shader, b"gbufferNormal")
        self.ssao_gbuffer_depth = loc(self.ssao_shader, b"gbufferDepth")
        self.ssao_cam_view = loc(self.ssao_shader, b"camView")
        self.ssao_cam_proj = loc(self.ssao_shader, b"camProj")
        self.ssao_cam_inv_proj = loc(self.ssao_shader, b"camInvProj")
        self.ssao_cam_inv_view_proj = loc(self.ssao_shader, b"camInvViewProj")
        self.ssao_light_view_proj = loc(self.ssao_shader, b"lightViewProj")
        self.ssao_shadow_map = loc(self.ssao_shader, b"shadowMap")
        self.ssao_shadow_inv_resolution = loc(self.ssao_shader, b"shadowInvResolution")
        self.ssao_cam_clip_near = loc(self.ssao_shader, b"camClipNear")
        self.ssao_cam_clip_far = loc(self.ssao_shader, b"camClipFar")
        self.ssao_light_clip_near = loc(self.ssao_shader, b"lightClipNear")
        self.ssao_light_clip_far = loc(self.ssao_shader, b"lightClipFar")
        self.ssao_light_dir = loc(self.ssao_shader, b"lightDir")
        # blur
        self.blur_gbuffer_normal = loc(self.blur_shader, b"gbufferNormal")
        self.blur_gbuffer_depth = loc(self.blur_shader, b"gbufferDepth")
        self.blur_input_texture = loc(self.blur_shader, b"inputTexture")
        self.blur_cam_inv_proj = loc(self.blur_shader, b"camInvProj")
        self.blur_cam_clip_near = loc(self.blur_shader, b"camClipNear")
        self.blur_cam_clip_far = loc(self.blur_shader, b"camClipFar")
        self.blur_inv_texture_resolution = loc(self.blur_shader, b"invTextureResolution")
        self.blur_blur_direction = loc(self.blur_shader, b"blurDirection")
        # fxaa
        self.fxaa_input_texture = loc(self.fxaa_shader, b"inputTexture")
        self.fxaa_inv_texture_resolution = loc(self.fxaa_shader, b"invTextureResolution")

        ground_mesh = gv.GenMeshPlane(float(ground_size), float(ground_size), 10, 10)
        self.ground_model = gv.LoadModelFromMesh(ground_mesh)
        self.shadow_map = gv.LoadShadowMap(int(shadow_size), int(shadow_size))
        self.gbuffer = gv.LoadGBuffer(self.width, self.height)
        self.lighted = gv.LoadRenderTexture(self.width, self.height)
        self.ssao_front = gv.LoadRenderTexture(self.width, self.height)
        self.ssao_back = gv.LoadRenderTexture(self.width, self.height)

        self.light_dir = gv.Vector3Normalize(gv.Vector3(*light_dir))
        self.shadow_light = gv.ShadowLight()
        self.shadow_light.target = gv.Vector3Zero()
        self.shadow_light.position = gv.Vector3Scale(self.light_dir, -self.shadow_extent)
        self.shadow_light.up = gv.Vector3(0.0, 1.0, 0.0)
        self.shadow_light.width = float(shadow_area[0])
        self.shadow_light.height = float(shadow_area[1])
        self.shadow_light.near = self.clip_near
        self.shadow_light.far = self.clip_far

        # Cached cdata: allocated once instead of per frame (the original loops allocated
        # 6+ ffi pointers and several Vector2/Rectangle objects every frame).
        ffi = gv.ffi
        self._specularity = ffi.new("float*"); self._specularity[0] = float(specularity)
        self._glossiness = ffi.new("float*"); self._glossiness[0] = float(glossiness)
        self._cam_clip_near = ffi.new("float*"); self._cam_clip_near[0] = self.clip_near
        self._cam_clip_far = ffi.new("float*"); self._cam_clip_far[0] = self.clip_far
        self._light_clip_near = ffi.new("float*"); self._light_clip_near[0] = self.clip_near
        self._light_clip_far = ffi.new("float*"); self._light_clip_far[0] = self.clip_far
        self._sun_strength = ffi.new("float*"); self._sun_strength[0] = float(sun_strength)
        self._sky_strength = ffi.new("float*"); self._sky_strength[0] = float(sky_strength)
        self._ground_strength = ffi.new("float*"); self._ground_strength[0] = float(ground_strength)
        self._ambient_strength = ffi.new("float*"); self._ambient_strength[0] = float(ambient_strength)
        self._exposure = ffi.new("float*"); self._exposure[0] = float(exposure)

        self._sun_color = gv.Vector3(253.0 / 255.0, 1.0, 232.0 / 255.0)
        self._sky_color = gv.Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        self._blur_dir_x = gv.Vector2(1.0, 0.0)
        self._blur_dir_y = gv.Vector2(0.0, 1.0)
        self._origin2 = gv.Vector2(0.0, 0.0)
        self._shadow_inv_resolution = gv.Vector2(1.0 / float(shadow_size), 1.0 / float(shadow_size))
        self._inv_resolution = gv.Vector2(1.0 / self.width, 1.0 / self.height)
        self._rect_ssao_front = gv.Rectangle(0, 0, self.ssao_front.texture.width, -self.ssao_front.texture.height)
        self._rect_ssao_back = gv.Rectangle(0, 0, self.ssao_back.texture.width, -self.ssao_back.texture.height)
        self._rect_gbuffer_color = gv.Rectangle(0, 0, self.gbuffer.color.width, -self.gbuffer.color.height)
        self._rect_lighted = gv.Rectangle(0, 0, self.lighted.texture.width, -self.lighted.texture.height)

        # Upload the frame-invariant uniforms once; the values live in the shader programs.
        u = gv.SHADER_UNIFORM_FLOAT
        v3 = gv.SHADER_UNIFORM_VEC3
        v2 = gv.SHADER_UNIFORM_VEC2
        for shader, spec, gloss, near, far in (
            (self.skinned_basic_shader, self.skinned_specularity, self.skinned_glossiness,
             self.skinned_cam_clip_near, self.skinned_cam_clip_far),
            (self.basic_shader, self.basic_specularity, self.basic_glossiness,
             self.basic_cam_clip_near, self.basic_cam_clip_far),
        ):
            gv.SetShaderValue(shader, spec, self._specularity, u)
            gv.SetShaderValue(shader, gloss, self._glossiness, u)
            gv.SetShaderValue(shader, near, self._cam_clip_near, u)
            gv.SetShaderValue(shader, far, self._cam_clip_far, u)
        for shader, near, far in (
            (self.shadow_shader, self.shadow_light_clip_near, self.shadow_light_clip_far),
            (self.skinned_shadow_shader, self.skinned_shadow_light_clip_near, self.skinned_shadow_light_clip_far),
        ):
            gv.SetShaderValue(shader, near, self._light_clip_near, u)
            gv.SetShaderValue(shader, far, self._light_clip_far, u)
        gv.SetShaderValue(self.ssao_shader, self.ssao_shadow_inv_resolution, ffi.addressof(self._shadow_inv_resolution), v2)
        gv.SetShaderValue(self.ssao_shader, self.ssao_light_dir, ffi.addressof(self.light_dir), v3)
        gv.SetShaderValue(self.blur_shader, self.blur_inv_texture_resolution, ffi.addressof(self._inv_resolution), v2)
        gv.SetShaderValue(self.fxaa_shader, self.fxaa_inv_texture_resolution, ffi.addressof(self._inv_resolution), v2)
        # Clip planes are set once (__init__`s rlSetClipPlanes) and never change, so every
        # clip-distance uniform is frame-invariant too.
        for shader, near_loc, far_loc, near_ptr, far_ptr in (
            (self.ssao_shader, self.ssao_cam_clip_near, self.ssao_cam_clip_far,
             self._cam_clip_near, self._cam_clip_far),
            (self.ssao_shader, self.ssao_light_clip_near, self.ssao_light_clip_far,
             self._light_clip_near, self._light_clip_far),
            (self.blur_shader, self.blur_cam_clip_near, self.blur_cam_clip_far,
             self._cam_clip_near, self._cam_clip_far),
            (self.lighting_shader, self.lighting_cam_clip_near, self.lighting_cam_clip_far,
             self._cam_clip_near, self._cam_clip_far),
        ):
            gv.SetShaderValue(shader, near_loc, near_ptr, u)
            gv.SetShaderValue(shader, far_loc, far_ptr, u)
        gv.SetShaderValue(self.lighting_shader, self.lighting_sun_color, ffi.addressof(self._sun_color), v3)
        gv.SetShaderValue(self.lighting_shader, self.lighting_sky_color, ffi.addressof(self._sky_color), v3)
        gv.SetShaderValue(self.lighting_shader, self.lighting_light_dir, ffi.addressof(self.light_dir), v3)
        gv.SetShaderValue(self.lighting_shader, self.lighting_sun_strength, self._sun_strength, u)
        gv.SetShaderValue(self.lighting_shader, self.lighting_sky_strength, self._sky_strength, u)
        gv.SetShaderValue(self.lighting_shader, self.lighting_ground_strength, self._ground_strength, u)
        gv.SetShaderValue(self.lighting_shader, self.lighting_ambient_strength, self._ambient_strength, u)
        gv.SetShaderValue(self.lighting_shader, self.lighting_exposure, self._exposure, u)

    # ------------------------------------------------------------------ rendering
    def render(
        self,
        camera: Any,
        *,
        models: Sequence[tuple[Any, Any]],
        ground_center: tuple[float, float] = (0.0, 0.0),
        shadow_focus: tuple[float, float] | None = None,
        overlay: Callable[[Any, Any], None] | None = None,
    ) -> None:
        """Draw one full frame; leaves the default framebuffer active (caller draws GUI then
        ``EndDrawing``). ``models`` is ``[(model, color), ...]`` with poses already updated."""
        gv = self.gv
        ffi = gv.ffi
        ground_position = gv.Vector3(float(ground_center[0]), self.ground_height, float(ground_center[1]))
        focus = ground_center if shadow_focus is None else shadow_focus
        self.shadow_light.target = gv.Vector3(float(focus[0]), 0.0, float(focus[1]))
        self.shadow_light.position = gv.Vector3Add(
            self.shadow_light.target, gv.Vector3Scale(self.light_dir, -self.shadow_extent))

        gv.rlDisableColorBlend()
        gv.BeginDrawing()

        # ---- shadow map ----
        gv.BeginShadowMap(self.shadow_map, self.shadow_light)
        light_view_proj = gv.MatrixMultiply(gv.rlGetMatrixModelview(), gv.rlGetMatrixProjection())
        self.ground_model.materials[0].shader = self.shadow_shader
        gv.DrawModel(self.ground_model, ground_position, 1.0, gv.WHITE)
        for model, _ in models:
            model.materials[0].shader = self.skinned_shadow_shader
            gv.DrawModel(model, gv.Vector3(0.0, 0.0, 0.0), 1.0, gv.WHITE)
        gv.EndShadowMap()

        # ---- G-buffer ----
        gv.BeginGBuffer(self.gbuffer, camera.cam3d)
        cam_view = gv.rlGetMatrixModelview()
        cam_proj = gv.rlGetMatrixProjection()
        cam_inv_proj = gv.MatrixInvert(cam_proj)
        cam_inv_view_proj = gv.MatrixInvert(gv.MatrixMultiply(cam_view, cam_proj))
        self.ground_model.materials[0].shader = self.basic_shader
        gv.DrawModel(self.ground_model, ground_position, 1.0, self.ground_color)
        for model, color in models:
            model.materials[0].shader = self.skinned_basic_shader
            gv.DrawModel(model, gv.Vector3(0.0, 0.0, 0.0), 1.0, color)
        gv.EndGBuffer(self.width, self.height)

        # ---- SSAO ----
        gv.BeginTextureMode(self.ssao_front)
        gv.BeginShaderMode(self.ssao_shader)
        gv.SetShaderValueTexture(self.ssao_shader, self.ssao_gbuffer_normal, self.gbuffer.normal)
        gv.SetShaderValueTexture(self.ssao_shader, self.ssao_gbuffer_depth, self.gbuffer.depth)
        gv.SetShaderValueMatrix(self.ssao_shader, self.ssao_cam_view, cam_view)
        gv.SetShaderValueMatrix(self.ssao_shader, self.ssao_cam_proj, cam_proj)
        gv.SetShaderValueMatrix(self.ssao_shader, self.ssao_cam_inv_proj, cam_inv_proj)
        gv.SetShaderValueMatrix(self.ssao_shader, self.ssao_cam_inv_view_proj, cam_inv_view_proj)
        gv.SetShaderValueMatrix(self.ssao_shader, self.ssao_light_view_proj, light_view_proj)
        gv.SetShaderValueShadowMap(self.ssao_shader, self.ssao_shadow_map, self.shadow_map)
        gv.ClearBackground(gv.WHITE)
        gv.DrawTextureRec(self.ssao_front.texture, self._rect_ssao_front, self._origin2, gv.WHITE)
        gv.EndShaderMode()
        gv.EndTextureMode()

        # ---- separable blur: H into ssao_back, V back into ssao_front ----
        gv.BeginTextureMode(self.ssao_back)
        gv.BeginShaderMode(self.blur_shader)
        gv.SetShaderValueTexture(self.blur_shader, self.blur_gbuffer_normal, self.gbuffer.normal)
        gv.SetShaderValueTexture(self.blur_shader, self.blur_gbuffer_depth, self.gbuffer.depth)
        gv.SetShaderValueTexture(self.blur_shader, self.blur_input_texture, self.ssao_front.texture)
        gv.SetShaderValueMatrix(self.blur_shader, self.blur_cam_inv_proj, cam_inv_proj)
        gv.SetShaderValue(self.blur_shader, self.blur_blur_direction, ffi.addressof(self._blur_dir_x), gv.SHADER_UNIFORM_VEC2)
        gv.DrawTextureRec(self.ssao_back.texture, self._rect_ssao_back, self._origin2, gv.WHITE)
        gv.EndShaderMode()
        gv.EndTextureMode()

        gv.BeginTextureMode(self.ssao_front)
        gv.BeginShaderMode(self.blur_shader)
        gv.SetShaderValueTexture(self.blur_shader, self.blur_input_texture, self.ssao_back.texture)
        gv.SetShaderValue(self.blur_shader, self.blur_blur_direction, ffi.addressof(self._blur_dir_y), gv.SHADER_UNIFORM_VEC2)
        gv.DrawTextureRec(self.ssao_front.texture, self._rect_ssao_front, self._origin2, gv.WHITE)
        gv.EndShaderMode()
        gv.EndTextureMode()

        # ---- lighting (plus caller overlays, depth-tested, before FXAA) ----
        gv.BeginTextureMode(self.lighted)
        gv.BeginShaderMode(self.lighting_shader)
        gv.SetShaderValueTexture(self.lighting_shader, self.lighting_gbuffer_color, self.gbuffer.color)
        gv.SetShaderValueTexture(self.lighting_shader, self.lighting_gbuffer_normal, self.gbuffer.normal)
        gv.SetShaderValueTexture(self.lighting_shader, self.lighting_gbuffer_depth, self.gbuffer.depth)
        gv.SetShaderValueTexture(self.lighting_shader, self.lighting_ssao, self.ssao_front.texture)
        gv.SetShaderValue(self.lighting_shader, self.lighting_cam_pos, ffi.addressof(camera.cam3d.position), gv.SHADER_UNIFORM_VEC3)
        gv.SetShaderValueMatrix(self.lighting_shader, self.lighting_cam_inv_view_proj, cam_inv_view_proj)
        gv.ClearBackground(gv.RAYWHITE)
        gv.DrawTextureRec(self.gbuffer.color, self._rect_gbuffer_color, self._origin2, gv.WHITE)
        gv.EndShaderMode()
        if overlay is not None:
            gv.BeginMode3D(camera.cam3d)
            overlay(gv, camera)
            gv.EndMode3D()
        gv.EndTextureMode()

        # ---- FXAA to the window ----
        gv.BeginShaderMode(self.fxaa_shader)
        gv.SetShaderValueTexture(self.fxaa_shader, self.fxaa_input_texture, self.lighted.texture)
        gv.DrawTextureRec(self.lighted.texture, self._rect_lighted, self._origin2, gv.WHITE)
        gv.EndShaderMode()

        gv.rlEnableColorBlend()

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        gv = self.gv
        for target in (self.lighted, self.ssao_back, self.ssao_front):
            try:
                gv.UnloadRenderTexture(target)
            except Exception:
                pass
        try:
            gv.UnloadGBuffer(self.gbuffer)
        except Exception:
            pass
        try:
            gv.UnloadShadowMap(self.shadow_map)
        except Exception:
            pass
        try:
            gv.UnloadModel(self.ground_model)
        except Exception:
            pass
        for shader in (self.fxaa_shader, self.blur_shader, self.ssao_shader, self.lighting_shader,
                       self.basic_shader, self.skinned_basic_shader, self.skinned_shadow_shader,
                       self.shadow_shader):
            try:
                gv.UnloadShader(shader)
            except Exception:
                pass


__all__ = ["GENOVIEW_DIR", "GENOVIEW_RESOURCES_DIR", "GenoViewPipeline", "_as_bytes", "_load_genoview_module"]
