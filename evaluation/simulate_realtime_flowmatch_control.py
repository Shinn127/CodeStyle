"""
Simulate real-time character control with flow-matching controllers built on the basic VQ model or LMM.

Workflow:
  1) Load a source trajectory segment (84-d traj per frame) from DB.
  2) Use the trained flow-matching controller in reconstruct or rollout mode.
  3) Export generated/reference 375-d sequences into NPZ files.
  4) Show side-by-side real-time visualization, defaulting to GenoView.

Example:
  python NewCode/evaluation/simulate_realtime_flowmatch_control.py \
    --db NewCode/database_100sty_375_memmap_nomirror \
    --flow-ckpt NewCode/checkpoints/<your_run>/best.pt \
    --vqvae-ckpt NewCode/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt \
    --mode rollout \
    --num-frames 300 \
    --target-style Zombie \
    --vis-backend quick \
    --out-dir NewCode/rollout_vis
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import secrets
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from Utils.model_loading import build_lmm_from_checkpoint, build_vqvae_from_checkpoint
from Utils.reproducibility import state_fingerprint
from dataset import norm_stats_fingerprint
from Utils.realtime import (
    choose_segment,
    load_db_arrays,
    parse_style_selector,
    prepare_output_dir,
    resolve_db_coordinate_convert,
    write_generated_reference_npz,
    write_meta_json,
)
from Utils.runtime import (
    load_checkpoint,
    load_norm_stats,
    load_state_dict_with_allowed_missing,
    resolve_device,
    sanitize_state_dict_keys,
    set_seed,
)
from Visualization.backends.dispatch_backend import canonicalize_vis_backend, visualize_pair_from_375
from arch import (
    FlowMatchLMMController,
    FlowMatchVQVAEController,
    build_flowmatch_controller_from_config,
    build_flowmatch_lmm_controller_from_config,
    resolve_flowmatch_prediction_mode,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
FLOWMATCH_VQ_COMPAT_MISSING_KEYS = (
    "style_slot_indices",
    "style_prior_probs",
    "style_prior_ready",
)


def _sync_device_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_rollout_benchmark(
    *,
    frame_latencies_sec: np.ndarray,
    warmup_frames: int,
    target_fps: float,
) -> dict[str, float | int]:
    if frame_latencies_sec.ndim != 1:
        raise ValueError(f"Expected frame_latencies_sec rank 1, got {frame_latencies_sec.shape}")
    if frame_latencies_sec.size == 0:
        raise ValueError("No frame latencies collected for benchmark")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    warmup = min(int(warmup_frames), int(frame_latencies_sec.size))
    measured = frame_latencies_sec[warmup:]
    if measured.size == 0:
        raise ValueError(
            f"warmup_frames={warmup_frames} leaves no measured frames; collected={frame_latencies_sec.size}"
        )

    lat_ms = measured * 1000.0
    fps_inst = 1.0 / np.maximum(measured, 1e-12)
    total_sec = float(measured.sum())
    target_frame_ms = 1000.0 / float(target_fps)
    over_budget = lat_ms > target_frame_ms

    return {
        "warmup_frames": int(warmup),
        "measured_frames": int(measured.size),
        "target_fps": float(target_fps),
        "target_frame_ms": float(target_frame_ms),
        "latency_ms_mean": float(lat_ms.mean()),
        "latency_ms_std": float(lat_ms.std()),
        "latency_ms_p50": float(np.percentile(lat_ms, 50)),
        "latency_ms_p95": float(np.percentile(lat_ms, 95)),
        "latency_ms_p99": float(np.percentile(lat_ms, 99)),
        "latency_ms_min": float(lat_ms.min()),
        "latency_ms_max": float(lat_ms.max()),
        "fps_mean_effective": float(measured.size / max(total_sec, 1e-12)),
        "fps_mean_instant": float(fps_inst.mean()),
        "fps_std_instant": float(fps_inst.std()),
        "fps_p05": float(np.percentile(fps_inst, 5)),
        "fps_p50": float(np.percentile(fps_inst, 50)),
        "fps_p95": float(np.percentile(fps_inst, 95)),
        "fps_cv_instant": float(fps_inst.std() / max(fps_inst.mean(), 1e-12)),
        "over_budget_ratio": float(over_budget.mean()),
    }


def _print_benchmark_summary(summary: dict[str, float | int]) -> None:
    print(
        "[Benchmark] "
        f"frames={summary['measured_frames']} warmup={summary['warmup_frames']} "
        f"lat_mean={summary['latency_ms_mean']:.3f}ms "
        f"lat_p95={summary['latency_ms_p95']:.3f}ms "
        f"lat_p99={summary['latency_ms_p99']:.3f}ms "
        f"fps_eff={summary['fps_mean_effective']:.2f} "
        f"fps_p05={summary['fps_p05']:.2f} "
        f"fps_p95={summary['fps_p95']:.2f} "
        f"fps_cv={summary['fps_cv_instant']:.4f} "
        f"over_budget={100.0 * float(summary['over_budget_ratio']):.2f}%"
    )


def _resolve_path_candidate(path_like: str | Path) -> Path:
    return Path(path_like).expanduser()


def _resolve_existing_path(
    path_like: str | Path,
    *,
    label: str,
    search_roots: tuple[Path, ...] = (),
) -> Path:
    raw = _resolve_path_candidate(path_like)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw))
        for root in search_roots:
            candidates.append(root / raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"{label} not found: {path_like} (searched: {', '.join(str(c.resolve()) for c in candidates)})"
    )


def _resolve_base_ckpt_path(
    explicit_vq_path: Path | None,
    explicit_lmm_path: Path | None,
    *,
    flow_args: dict,
    flow_ckpt_path: Path,
) -> tuple[str, Path]:
    search_roots = (flow_ckpt_path.parent, SCRIPT_DIR, REPO_ROOT)
    base_model_type = str(flow_args.get("base_model_type", "vqvae")).strip().lower()
    if base_model_type in {"vq", "basic_vqvae"}:
        base_model_type = "vqvae"
    if base_model_type not in {"vqvae", "lmm"}:
        raise ValueError(f"Unsupported flow base_model_type: {base_model_type!r}")
    if base_model_type == "vqvae":
        if explicit_vq_path is not None:
            return "vqvae", _resolve_existing_path(explicit_vq_path, label="--vqvae-ckpt", search_roots=search_roots)
        vqvae_ckpt_raw = flow_args.get("vqvae_ckpt", None)
        if vqvae_ckpt_raw is None:
            raise ValueError("--vqvae-ckpt is required because flow checkpoint args has no vqvae_ckpt")
        return "vqvae", _resolve_existing_path(vqvae_ckpt_raw, label="vqvae_ckpt from flow checkpoint args", search_roots=search_roots)

    if explicit_lmm_path is not None:
        return "lmm", _resolve_existing_path(explicit_lmm_path, label="--lmm-ckpt", search_roots=search_roots)
    lmm_ckpt_raw = flow_args.get("lmm_ckpt", None)
    if lmm_ckpt_raw is None:
        raise ValueError("--lmm-ckpt is required because flow checkpoint args has no lmm_ckpt")
    return "lmm", _resolve_existing_path(lmm_ckpt_raw, label="lmm_ckpt from flow checkpoint args", search_roots=search_roots)


def _resolve_norm_stats_path(
    explicit_path: Path | None,
    *,
    flow_args: dict,
    flow_ckpt_path: Path,
    base_ckpt_path: Path,
) -> Path:
    search_roots = (flow_ckpt_path.parent, base_ckpt_path.parent, SCRIPT_DIR, REPO_ROOT)
    if explicit_path is not None:
        return _resolve_existing_path(explicit_path, label="--norm-stats", search_roots=search_roots)

    flow_arg_norm = flow_args.get("norm_stats", None)
    if flow_arg_norm is not None:
        try:
            return _resolve_existing_path(
                flow_arg_norm,
                label="norm_stats from flow checkpoint args",
                search_roots=search_roots,
            )
        except FileNotFoundError:
            pass

    flow_sibling = flow_ckpt_path.parent / "norm_stats.npz"
    if flow_sibling.exists():
        return flow_sibling.resolve()

    base_sibling = base_ckpt_path.parent / "norm_stats.npz"
    if base_sibling.exists():
        return base_sibling.resolve()

    raise FileNotFoundError(
        "Could not resolve norm stats. Tried flow checkpoint args, "
        f"{flow_sibling}, and {base_sibling}."
    )


def _resolve_solver(explicit_solver: Optional[str], *, flow_args: dict) -> tuple[str, str]:
    del flow_args
    if explicit_solver is not None:
        return explicit_solver, "cli"
    return "euler", "minimal-default"


def _resolve_ode_steps(explicit_steps: Optional[int], *, flow_args: dict) -> tuple[int, str]:
    if explicit_steps is not None:
        return int(explicit_steps), "cli"
    ckpt_solver_steps = flow_args.get("solver_steps", flow_args.get("recon_solver_steps", None))
    if ckpt_solver_steps is not None:
        return int(ckpt_solver_steps), "checkpoint"
    return 16, "fallback"


def _resolve_seed(explicit_seed: Optional[int]) -> tuple[int, str]:
    if explicit_seed is not None:
        return int(explicit_seed), "cli"
    return int(secrets.randbelow(2**31)), "random"


def _build_flow_controller(
    *,
    flow_args: dict,
    flow_ckpt: dict,
    base_model,
    base_model_type: str,
    num_styles: int,
    device: torch.device,
) -> FlowMatchVQVAEController | FlowMatchLMMController:
    if base_model_type == "vqvae":
        model = build_flowmatch_controller_from_config(
            flow_args,
            vqvae=base_model,
            num_styles=num_styles,
        ).to(device)
    elif base_model_type == "lmm":
        model = build_flowmatch_lmm_controller_from_config(
            flow_args,
            lmm=base_model,
            num_styles=num_styles,
        ).to(device)
    else:
        raise ValueError(f"Unsupported base_model_type: {base_model_type!r}")

    state_dict = sanitize_state_dict_keys(flow_ckpt["model"])
    if base_model_type == "vqvae":
        bundled = {k[6:]: v for k, v in state_dict.items() if k.startswith("vqvae.")}
        if bundled and state_fingerprint(base_model.state_dict()) != state_fingerprint(bundled):
            raise ValueError("VQ override differs from the controller latent manifold")
        load_state_dict_with_allowed_missing(
            model,
            state_dict,
            allowed_missing=FLOWMATCH_VQ_COMPAT_MISSING_KEYS,
        )
    else:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_style_recipe(
    recipe_path: Path,
    *,
    style_names: list[str],
    selected_slots: tuple[int, ...],
    num_embeddings: int,
    fallback_style_label: int | None,
    vq_fingerprint=None,
    allow_legacy=False,
    norm_fingerprint=None,
) -> tuple[np.ndarray, float, int | None, dict]:
    with recipe_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"style recipe must be a JSON object, got {type(payload)}")
    if vq_fingerprint is not None:
        from Utils.style_code_prior import validate_recipe_metadata
        validate_recipe_metadata(payload.get("metadata"), vq_fingerprint=vq_fingerprint,
            selected_slots=selected_slots, allow_legacy=allow_legacy, norm_fingerprint=norm_fingerprint)
    slot_entries = payload.get("slots", None)
    if not isinstance(slot_entries, list) or not slot_entries:
        raise ValueError("style recipe must contain a non-empty 'slots' list")

    slot_to_local = {int(slot): idx for idx, slot in enumerate(selected_slots)}
    manual_probs = np.zeros((len(selected_slots), num_embeddings), dtype=np.float32)
    has_base_style = "base_style" in payload and str(payload["base_style"]).strip() != ""
    base_style_label = (
        parse_style_selector(str(payload["base_style"]), style_names) if has_base_style else fallback_style_label
    )
    strength = float(payload.get("strength", 1.0))
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Recipe strength must be finite and in [0,1]')
    overrides_summary: list[dict[str, object]] = []
    seen_slots = set()

    for entry_idx, entry in enumerate(slot_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"style recipe slot entry {entry_idx} must be an object, got {type(entry)}")
        if "slot" not in entry or "codes" not in entry:
            raise KeyError(f"style recipe slot entry {entry_idx} must contain 'slot' and 'codes'")
        slot = int(entry["slot"])
        if slot in seen_slots:
            raise ValueError(f'Recipe repeats slot {slot}')
        seen_slots.add(slot)
        if slot not in slot_to_local:
            raise ValueError(
                f"style recipe slot {slot} is not controllable by this checkpoint. "
                f"Selected slots: {list(selected_slots)}"
            )
        codes = [int(code) for code in entry["codes"]]
        if len(set(codes)) != len(codes):
            raise ValueError(f'Recipe slot {slot} repeats code indices')
        if not codes:
            raise ValueError(f"style recipe slot entry {entry_idx} must contain at least one code")
        if min(codes) < 0 or max(codes) >= num_embeddings:
            raise ValueError(
                f"style recipe codes for slot {slot} must lie within [0, {num_embeddings - 1}], got {codes}"
            )
        weights_raw = entry.get("weights", None)
        if weights_raw is None:
            weights = np.full((len(codes),), 1.0 / float(len(codes)), dtype=np.float32)
        else:
            weights = np.asarray(weights_raw, dtype=np.float32).reshape(-1)
            if weights.size != len(codes):
                raise ValueError(
                    f"style recipe slot {slot} weights length mismatch: expected {len(codes)}, got {weights.size}"
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0:
                raise ValueError(f"style recipe slot {slot} weights must be finite, non-negative and non-empty")
            weights = weights / np.clip(weights.sum(), 1e-8, None)
        local_slot = slot_to_local[slot]
        manual_probs[local_slot] = 0.0
        manual_probs[local_slot, np.asarray(codes, dtype=np.int64)] = weights
        overrides_summary.append(
            {
                "slot": slot,
                "codes": codes,
                "weights": [float(weight) for weight in weights.tolist()],
            }
        )

    if base_style_label is None and (strength != 1 or np.any(manual_probs.sum(-1) <= 0)):
        raise ValueError("Pure recipes must specify every selected slot and strength=1")
    return manual_probs, strength, base_style_label, {"path": str(recipe_path), "overrides": overrides_summary}


def _predict_next_frame(
    *,
    model: FlowMatchVQVAEController | FlowMatchLMMController,
    base_model_type: str,
    x_curr: torch.Tensor,
    traj_next: torch.Tensor,
    style_label: torch.Tensor | None,
    manual_style_probs: torch.Tensor | None,
    manual_style_strength: float,
    prev_latent: torch.Tensor | None,
    ode_steps: int,
    noise_std: float,
    solver: str,
    requantize_z: bool,
    latent_clip: bool,
    history_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base_model_type == "vqvae":
        z_hat = model.sample_latent(
            x_curr=x_curr,
            style_label=style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            prev_latent=prev_latent,
            **({"history_latent":history_latent} if getattr(model,"history_conditioning",False) else {}),
            num_steps=ode_steps,
            noise_std=noise_std,
            solver=solver,
        )
        motion_hat, z_hat = model.feedback_state(z_hat, requantize=requantize_z, latent_clip=latent_clip)
        return torch.cat((traj_next, motion_hat), dim=-1), z_hat

    if requantize_z:
        raise ValueError("--requantize-z is only supported for VQ-based flow controllers")
    if manual_style_probs is not None:
        raise ValueError("--style-recipe is only supported for VQ-based flow controllers")
    motion_hat, z_hat = model.predict_next_motion(
        x_curr=x_curr,
        traj_next=traj_next,
        style_label=style_label,
        prev_latent=prev_latent,
        num_steps=ode_steps,
        noise_std=noise_std,
        solver=solver,
    )
    if latent_clip:
        z_hat = model.clip_latent_to_data_bounds(z_hat)
        motion_hat = model.decode_motion(traj=traj_next, latent=z_hat)
    return torch.cat((traj_next, motion_hat), dim=-1), z_hat


def reconstruct_sequence(
    *,
    model: FlowMatchVQVAEController | FlowMatchLMMController,
    base_model_type: str,
    x_ref_norm: np.ndarray,
    style_label: int | None,
    manual_style_probs: torch.Tensor | None,
    manual_style_strength: float,
    ode_steps: int,
    noise_std: float,
    solver: str,
    device: torch.device,
    requantize_z: bool = False,
    latent_clip: bool = True,
    benchmark: bool = False,
    warmup_frames: int = 0,
    target_fps: float = 60.0,
) -> tuple[np.ndarray, dict[str, float | int] | None]:
    if x_ref_norm.ndim != 2 or x_ref_norm.shape[1] != 375:
        raise ValueError(f"x_ref_norm must be shape (T,375), got {x_ref_norm.shape}")
    if x_ref_norm.shape[0] < 2:
        raise ValueError(f"x_ref_norm must contain at least 2 frames, got {x_ref_norm.shape[0]}")
    if ode_steps < 1:
        raise ValueError(f"ode_steps must be >= 1, got {ode_steps}")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be > 0, got {noise_std}")
    if solver != "euler":
        raise ValueError(f"Unsupported solver: {solver}")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    t_total = x_ref_norm.shape[0]
    seq = torch.empty((t_total, 375), device=device, dtype=torch.float32)
    x_ref_t = torch.from_numpy(x_ref_norm.astype(np.float32, copy=False)).to(device=device)
    seq[0] = x_ref_t[0]
    style_t = None if style_label is None else torch.tensor([style_label], dtype=torch.long, device=device)
    frame_latencies_sec: list[float] = []
    model.eval()
    with torch.no_grad():
        history_codes = model.frame_latent(x_ref_t) if getattr(model,"history_conditioning",False) else None
        for i in range(1, t_total):
            x_curr = x_ref_t[i - 1 : i]
            if history_codes is not None:
                x_curr = torch.cat((x_curr[:,:84],model.vqvae.decode(history_codes[i-1:i])),dim=-1)
            traj_next = x_ref_t[i : i + 1, :84]
            if benchmark:
                _sync_device_if_needed(device)
                t0 = time.perf_counter()

            x_next, _ = _predict_next_frame(
                model=model,
                base_model_type=base_model_type,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_t,
                manual_style_probs=manual_style_probs,
                manual_style_strength=manual_style_strength,
                prev_latent=None if history_codes is None else history_codes[i-1:i],
                history_latent=None if history_codes is None else history_codes[max(0,i-2):max(0,i-2)+1],
                ode_steps=ode_steps,
                noise_std=noise_std,
                solver=solver,
                requantize_z=requantize_z,
                latent_clip=latent_clip,
            )
            if benchmark:
                _sync_device_if_needed(device)
                frame_latencies_sec.append(time.perf_counter() - t0)
            seq[i] = x_next[0]

    benchmark_summary = None
    if benchmark:
        benchmark_summary = _summarize_rollout_benchmark(
            frame_latencies_sec=np.asarray(frame_latencies_sec, dtype=np.float64),
            warmup_frames=warmup_frames,
            target_fps=target_fps,
        )
    return seq.detach().cpu().numpy().astype(np.float32, copy=False), benchmark_summary


def rollout_sequence(
    *,
    model: FlowMatchVQVAEController | FlowMatchLMMController,
    base_model_type: str,
    x_seed_norm: np.ndarray,
    traj_plan_norm: np.ndarray,
    style_label: int | None,
    manual_style_probs: torch.Tensor | None,
    manual_style_strength: float,
    ode_steps: int,
    noise_std: float,
    solver: str,
    device: torch.device,
    requantize_z: bool = False,
    latent_clip: bool = True,
    benchmark: bool = False,
    warmup_frames: int = 0,
    target_fps: float = 60.0,
) -> tuple[np.ndarray, dict[str, float | int] | None]:
    if x_seed_norm.shape != (375,):
        raise ValueError(f"x_seed_norm must be shape (375,), got {x_seed_norm.shape}")
    if traj_plan_norm.ndim != 2 or traj_plan_norm.shape[1] != 84:
        raise ValueError(f"traj_plan_norm must be shape (T,84), got {traj_plan_norm.shape}")
    if ode_steps < 1:
        raise ValueError(f"ode_steps must be >= 1, got {ode_steps}")
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be > 0, got {noise_std}")
    if solver != "euler":
        raise ValueError(f"Unsupported solver: {solver}")
    if warmup_frames < 0:
        raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    t_total = traj_plan_norm.shape[0]
    seq = torch.empty((t_total, 375), device=device, dtype=torch.float32)
    x_curr = torch.from_numpy(x_seed_norm.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)
    seq[0] = x_curr[0]
    traj_plan_t = torch.from_numpy(traj_plan_norm.astype(np.float32, copy=False)).to(device=device)
    style_t = None if style_label is None else torch.tensor([style_label], dtype=torch.long, device=device)
    prev_latent = model.frame_latent(x_curr)
    history_latent = prev_latent
    if getattr(model,"history_conditioning",False):
        x_curr = torch.cat((x_curr[:,:84],model.vqvae.decode(prev_latent)),dim=-1)
        seq[0] = x_curr[0]
    frame_latencies_sec: list[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(1, t_total):
            traj_next = traj_plan_t[i : i + 1]
            if benchmark:
                _sync_device_if_needed(device)
                t0 = time.perf_counter()

            x_next, z_next = _predict_next_frame(
                model=model,
                base_model_type=base_model_type,
                x_curr=x_curr,
                traj_next=traj_next,
                style_label=style_t,
                manual_style_probs=manual_style_probs,
                manual_style_strength=manual_style_strength,
                prev_latent=prev_latent,
                history_latent=history_latent,
                ode_steps=ode_steps,
                noise_std=noise_std,
                solver=solver,
                requantize_z=requantize_z,
                latent_clip=latent_clip,
            )
            if benchmark:
                _sync_device_if_needed(device)
                frame_latencies_sec.append(time.perf_counter() - t0)
            seq[i] = x_next[0]
            x_curr = x_next
            history_latent = prev_latent
            prev_latent = z_next

    benchmark_summary = None
    if benchmark:
        benchmark_summary = _summarize_rollout_benchmark(
            frame_latencies_sec=np.asarray(frame_latencies_sec, dtype=np.float64),
            warmup_frames=warmup_frames,
            target_fps=target_fps,
        )
    return seq.detach().cpu().numpy().astype(np.float32, copy=False), benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate real-time control with flow-matching controllers over the project base autoencoders"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=SCRIPT_DIR / "database_100sty_375_memmap_nomirror",
        help="Input DB path (memmap directory)",
    )
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "flowmatch_label_nomirror_150ep" / "best.pt",
        help="Trained flow matching checkpoint",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=None,
        help="Basic VQ checkpoint (optional; default tries to read from flow checkpoint args)",
    )
    parser.add_argument(
        "--lmm-ckpt",
        type=Path,
        default=None,
        help="LMM checkpoint (optional; default tries to read from flow checkpoint args)",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Normalization stats npz with mean/std. Default: sibling of flow ckpt (norm_stats.npz)",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Generated rollout length")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for segment sampling and latent noise. Default: generate a fresh random seed each run.",
    )
    parser.add_argument(
        "--source-style",
        type=str,
        default="",
        help="Style selector for source trajectory segment (name or label id)",
    )
    parser.add_argument(
        "--target-style",
        type=str,
        default="",
        help="Target style for generation (name or label id). Empty means same as source style.",
    )
    parser.add_argument(
        "--style-recipe",
        type=Path,
        default=None,
        help="Optional JSON file that manually overrides selected style slots with code mixtures.",
    )
    parser.add_argument(
        "--style-strength",
        type=float,
        default=None,
        help="Optional override for recipe strength in [0,1]. If omitted, uses the recipe's 'strength'.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional fixed start index in DB (must keep same label for num_frames)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reconstruct", "rollout"],
        default="rollout",
        help="Generation mode. 'reconstruct' uses teacher-forced next-frame prediction; 'rollout' is autoregressive.",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=None,
        help="ODE integration steps for latent generation. Default: inherit solver_steps from flow checkpoint args.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["euler"],
        default=None,
        help="ODE solver used for generation. Minimal pipeline keeps Euler only.",
    )
    parser.add_argument("--noise-std", type=float, default=1.0, help="Initial latent noise std")
    parser.add_argument(
        "--latent-clip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clip normalized latent to dataset min/max after each predicted step, matching ControlOperators.",
    )
    parser.add_argument(
        "--requantize-z",
        action=argparse.BooleanOptionalAction, default=None,
        help="Snap each predicted latent to the nearest VQ codebook entry before decoding.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--allow-legacy-recipe", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure per-frame generation latency and report tail latency / FPS stability",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of initial predicted frames excluded from latency summary when --benchmark is enabled",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "rollout_vis",
        help="Directory for generated npz, metadata, and visualization outputs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Default: flowmatch_<mode>_<num_frames>",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Show side-by-side reference/prediction visualization after generation (default true)",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip quick visualization call",
    )
    parser.add_argument("--fps", type=int, default=60, help="Visualization fps")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Visualization dt")
    parser.add_argument("--local-only", action="store_true", help="Visualize local positions only")
    parser.add_argument(
        "--vis-backend",
        type=str,
        choices=["genoview", "quick", "matplotlib"],
        default="genoview",
        help="Visualization backend. 'quick' uses the Matplotlib quick-visualize path; 'matplotlib' is kept as an alias of 'quick'.",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="Optional GIF output path for quick visualization backend.",
    )
    parser.add_argument(
        "--geno-gap",
        type=float,
        default=2.4,
        help="Horizontal spacing between the two GenoView characters.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "100sty",
        help="100STYLE BVH root used for side-by-side visualization",
    )
    parser.add_argument(
        "--bvh-ref",
        type=Path,
        default=None,
        help="Optional BVH ref used for side-by-side visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.db = args.db.expanduser().resolve()
    args.flow_ckpt = _resolve_existing_path(args.flow_ckpt, label="--flow-ckpt", search_roots=(SCRIPT_DIR, REPO_ROOT))
    args.vis_backend = canonicalize_vis_backend(args.vis_backend)

    flow_ckpt = load_checkpoint(args.flow_ckpt, map_location="cpu")
    flow_args = flow_ckpt.get("args", {}) or {}

    base_model_type, base_ckpt = _resolve_base_ckpt_path(
        args.vqvae_ckpt,
        args.lmm_ckpt,
        flow_args=flow_args,
        flow_ckpt_path=args.flow_ckpt,
    )
    args.vqvae_ckpt = base_ckpt if base_model_type == "vqvae" else None
    args.lmm_ckpt = base_ckpt if base_model_type == "lmm" else None
    args.norm_stats = _resolve_norm_stats_path(
        args.norm_stats,
        flow_args=flow_args,
        flow_ckpt_path=args.flow_ckpt,
        base_ckpt_path=base_ckpt,
    )

    args.seed, seed_source = _resolve_seed(args.seed)
    args.solver, solver_source = _resolve_solver(args.solver, flow_args=flow_args)
    args.ode_steps, ode_source = _resolve_ode_steps(args.ode_steps, flow_args=flow_args)
    prediction_mode = resolve_flowmatch_prediction_mode(str(flow_args.get("prediction_mode", "velocity")))
    if args.gif_path is not None:
        args.gif_path = args.gif_path.expanduser().resolve()
    if args.prefix is None:
        args.prefix = f"flowmatch_{args.mode}_{args.num_frames}"

    if args.num_frames < 2:
        raise ValueError("--num-frames must be >= 2")
    if args.solver != "euler":
        raise ValueError(f"--solver must be 'euler', got {args.solver!r}")
    if args.ode_steps < 1:
        raise ValueError("--ode-steps must be >= 1")
    if args.noise_std <= 0.0:
        raise ValueError("--noise-std must be > 0")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")

    set_seed(args.seed)

    device = resolve_device(args.device)
    coordinate_convert = resolve_db_coordinate_convert(args.db)
    if args.requantize_z is None:
        args.requantize_z = bool(flow_args.get("inference_requantize", False))
    if args.latent_clip is None:
        args.latent_clip = bool(flow_args.get("inference_latent_clip", True))
    print(f"[Info] mode            : {args.mode}")
    print(f"[Info] device          : {device}")
    print(f"[Info] flow ckpt       : {args.flow_ckpt}")
    print(f"[Info] base model      : {base_model_type}")
    print(f"[Info] base ckpt       : {base_ckpt}")
    print(f"[Info] norm stats      : {args.norm_stats}")
    print(f"[Info] seed            : {args.seed} ({seed_source})")
    print(f"[Info] solver          : {args.solver} ({solver_source})")
    print(f"[Info] ode steps       : {args.ode_steps} ({ode_source})")
    print(f"[Info] prediction mode : {prediction_mode} (checkpoint)")
    print(f"[Info] requantize z    : {args.requantize_z}")
    print(f"[Info] latent clip     : {args.latent_clip}")
    print(f"[Info] benchmark       : {args.benchmark}")
    if args.benchmark:
        print(f"[Info] warmup frames   : {args.warmup_frames}")
    print(f"[Info] db              : {args.db}")
    print(f"[Info] coord convert   : {coordinate_convert}")

    x_source, labels, style_names = load_db_arrays(args.db)
    if labels.size == 0:
        raise RuntimeError("DB is empty")
    print("[Info] db format       : memmap")
    print(f"[Info] db frames       : {labels.shape[0]}")
    print(f"[Info] num styles      : {len(style_names)}")

    src_style_label = parse_style_selector(args.source_style, style_names)
    start, end, src_label, src_run = choose_segment(
        labels=labels,
        num_frames=args.num_frames,
        seed=args.seed,
        target_label=src_style_label,
        start_index=args.start_index,
    )
    target_style_label = parse_style_selector(args.target_style, style_names)
    if target_style_label is not None and (target_style_label < 0 or target_style_label >= len(style_names)):
        raise ValueError(
            f"target style label out of range: {target_style_label} vs [0, {len(style_names)-1}]"
        )

    x_ref_raw = np.asarray(x_source[start:end], dtype=np.float32)
    if x_ref_raw.shape != (args.num_frames, 375):
        raise RuntimeError(
            f"Unexpected sampled segment shape {x_ref_raw.shape}, expected {(args.num_frames, 375)}"
        )

    norm_mean, norm_std = load_norm_stats(args.norm_stats)
    recorded_norm = flow_args.get('fingerprints', {}).get('norm_fingerprint')
    if recorded_norm:
        if recorded_norm != norm_stats_fingerprint(norm_mean, norm_std):
            raise ValueError('Simulation normalization differs from the controller/VQ manifold')
    x_ref_norm = (x_ref_raw - norm_mean[None, :]) / norm_std[None, :]

    if base_model_type == "vqvae":
        base_model = build_vqvae_from_checkpoint(base_ckpt).to(device)
    else:
        base_model = build_lmm_from_checkpoint(base_ckpt).to(device)

    num_styles = int(flow_args.get("num_styles", 0))
    if num_styles <= 0:
        num_styles = len(style_names)
    if target_style_label is not None and target_style_label >= num_styles:
        raise ValueError(
            f"target style {target_style_label} exceeds model num_styles={num_styles}. "
            "Use matching style set or retrain with larger num_styles."
        )

    model = _build_flow_controller(
        flow_args=flow_args,
        flow_ckpt=flow_ckpt,
        base_model=base_model,
        base_model_type=base_model_type,
        num_styles=num_styles,
        device=device,
    )
    print(f"[Info] controller      : {type(model).__name__}({model.prediction_mode})")

    manual_style_probs = None
    manual_style_strength = 1.0
    conditioning_style_label = target_style_label
    pure_recipe_model = getattr(model, 'conditioning_mode', 'label') in ('recipe', 'clip_recipe')
    if pure_recipe_model and args.style_recipe is None:
        raise ValueError('This controller requires --style-recipe; export a clip recipe with its --flow-ckpt first')
    recipe_meta = None
    if args.style_recipe is not None:
        args.style_recipe = _resolve_existing_path(
            args.style_recipe,
            label="--style-recipe",
            search_roots=(SCRIPT_DIR, REPO_ROOT),
        )
        if base_model_type != "vqvae" or not isinstance(model, FlowMatchVQVAEController):
            raise ValueError("--style-recipe is only supported for VQ-based flow controllers")
        if not model.use_style_code_selector:
            raise ValueError(
                "--style-recipe requires a checkpoint trained with --use-style-code-selector. "
                "This checkpoint still uses the embedding-based style encoder."
            )
        recipe_probs_np, recipe_strength, recipe_base_style_label, recipe_meta = _load_style_recipe(
            args.style_recipe,
            style_names=style_names,
            selected_slots=model.selected_style_slots,
            num_embeddings=int(model.vqvae.quantizer.num_embeddings),
            fallback_style_label=None if pure_recipe_model else target_style_label,
            vq_fingerprint=state_fingerprint(model.vqvae.state_dict()),
            allow_legacy=args.allow_legacy_recipe,
            norm_fingerprint=norm_stats_fingerprint(norm_mean, norm_std),
        )
        manual_style_strength = float(recipe_strength if args.style_strength is None else args.style_strength)
        if not (0.0 <= manual_style_strength <= 1.0):
            raise ValueError(f"--style-strength must lie in [0,1], got {manual_style_strength}")
        conditioning_style_label = recipe_base_style_label
        manual_style_probs = torch.from_numpy(recipe_probs_np).to(device=device, dtype=torch.float32)
        if pure_recipe_model:
            if manual_style_strength != 1 or (manual_style_probs.sum(-1) <= 0).any():
                raise ValueError('Recipe-only controllers require strength=1 and all selected slots')
            conditioning_style_label = None
        print(f"[Info] style recipe    : {args.style_recipe}")
        print(f"[Info] style strength  : {manual_style_strength:.3f}")
        print(f"[Info] style slots     : {list(model.selected_style_slots)}")

    if target_style_label is None:
        target_style_label = conditioning_style_label if conditioning_style_label is not None else src_label
    if conditioning_style_label is None and manual_style_probs is None:
        conditioning_style_label = target_style_label
    if target_style_label < 0 or target_style_label >= len(style_names):
        raise ValueError(
            f"target style label out of range: {target_style_label} vs [0, {len(style_names)-1}]"
        )
    if conditioning_style_label is not None and (
        conditioning_style_label < 0 or conditioning_style_label >= len(style_names)
    ):
        raise ValueError(
            f"conditioning style label out of range: {conditioning_style_label} vs [0, {len(style_names)-1}]"
        )

    if args.mode == "reconstruct":
        x_gen_norm, benchmark_summary = reconstruct_sequence(
            model=model,
            base_model_type=base_model_type,
            x_ref_norm=x_ref_norm,
            style_label=conditioning_style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            ode_steps=args.ode_steps,
            noise_std=args.noise_std,
            solver=args.solver,
            device=device,
            requantize_z=bool(args.requantize_z),
            latent_clip=bool(args.latent_clip),
            benchmark=bool(args.benchmark),
            warmup_frames=args.warmup_frames,
            target_fps=float(args.fps),
        )
    else:
        x_gen_norm, benchmark_summary = rollout_sequence(
            model=model,
            base_model_type=base_model_type,
            x_seed_norm=x_ref_norm[0],
            traj_plan_norm=x_ref_norm[:, :84],
            style_label=conditioning_style_label,
            manual_style_probs=manual_style_probs,
            manual_style_strength=manual_style_strength,
            ode_steps=args.ode_steps,
            noise_std=args.noise_std,
            solver=args.solver,
            device=device,
            requantize_z=bool(args.requantize_z),
            latent_clip=bool(args.latent_clip),
            benchmark=bool(args.benchmark),
            warmup_frames=args.warmup_frames,
            target_fps=float(args.fps),
        )
    x_gen_raw = x_gen_norm * norm_std[None, :] + norm_mean[None, :]
    x_gen_raw = np.asarray(x_gen_raw, dtype=np.float32)

    metric_start_frame = 1
    motion_mse_norm: float | None = None
    motion_mae_norm: float | None = None
    if target_style_label == src_label and manual_style_probs is None:
        motion_diff = x_gen_norm[metric_start_frame:, 84:] - x_ref_norm[metric_start_frame:, 84:]
        motion_mse_norm = float(np.mean(motion_diff**2))
        motion_mae_norm = float(np.mean(np.abs(motion_diff)))

    args.out_dir = prepare_output_dir(args.out_dir)
    gen_npz_path, ref_npz_path = write_generated_reference_npz(
        out_dir=args.out_dir,
        prefix=args.prefix,
        generated_x=x_gen_raw,
        reference_x=x_ref_raw,
        generated_label=target_style_label,
        reference_label=src_label,
        style_names=style_names,
        coordinate_convert=coordinate_convert,
    )

    meta = {
        "mode": args.mode,
        "flow_ckpt": str(args.flow_ckpt),
        "base_model_type": base_model_type,
        "vqvae_ckpt": str(args.vqvae_ckpt) if args.vqvae_ckpt is not None else None,
        "lmm_ckpt": str(args.lmm_ckpt) if args.lmm_ckpt is not None else None,
        "norm_stats": str(args.norm_stats),
        "db": str(args.db),
        "db_format": "memmap",
        "segment_start": int(start),
        "segment_end": int(end),
        "segment_source_label": int(src_label),
        "segment_source_style": style_names[src_label] if 0 <= src_label < len(style_names) else f"label_{src_label}",
        "source_run": {"start": int(src_run[0]), "end": int(src_run[1]), "label": int(src_run[2])},
        "target_style_label": int(target_style_label),
        "target_style_name": style_names[target_style_label],
        "conditioning_style_label": None if conditioning_style_label is None else int(conditioning_style_label),
        "conditioning_style_name": None
        if conditioning_style_label is None
        else style_names[conditioning_style_label],
        "num_frames": int(args.num_frames),
        "seed": int(args.seed),
        "seed_source": seed_source,
        "prediction_mode": prediction_mode,
        "controller_class": type(model).__name__,
        "solver": str(args.solver),
        "ode_steps": int(args.ode_steps),
        "noise_std": float(args.noise_std),
        "latent_clip": bool(args.latent_clip),
        "requantize_z": bool(args.requantize_z),
        "vis_backend": str(args.vis_backend),
        "coordinate_convert": bool(coordinate_convert),
        "reference_npz": str(ref_npz_path),
        "generated_npz": str(gen_npz_path),
        "metric_start_frame": metric_start_frame,
        "motion_mse_norm": motion_mse_norm,
        "motion_mae_norm": motion_mae_norm,
        "benchmark": benchmark_summary,
    }
    if recipe_meta is not None:
        meta["style_recipe"] = {
            "path": recipe_meta["path"],
            "strength": float(manual_style_strength),
            "selected_slots": list(model.selected_style_slots) if isinstance(model, FlowMatchVQVAEController) else [],
            "overrides": recipe_meta["overrides"],
        }
    if args.gif_path is not None:
        meta["visualization_gif"] = str(args.gif_path)
    meta_path = write_meta_json(
        out_dir=args.out_dir,
        prefix=args.prefix,
        meta=meta,
    )

    print(f"[Info] source segment  : [{start}, {end}) label={src_label} ({meta['segment_source_style']})")
    print(f"[Info] target style    : {target_style_label} ({style_names[target_style_label]})")
    if motion_mse_norm is not None and motion_mae_norm is not None:
        print(f"[Info] motion mse(norm): {motion_mse_norm:.6f} (frames {metric_start_frame}:{args.num_frames})")
        print(f"[Info] motion mae(norm): {motion_mae_norm:.6f} (frames {metric_start_frame}:{args.num_frames})")
    else:
        print("[Info] motion metrics  : skipped because target style differs from source style")
    print(f"[Info] generated npz   : {gen_npz_path}")
    print(f"[Info] reference npz   : {ref_npz_path}")
    print(f"[Info] meta json       : {meta_path}")
    if args.gif_path is not None:
        print(f"[Info] gif path        : {args.gif_path}")
    if benchmark_summary is not None:
        _print_benchmark_summary(benchmark_summary)

    if args.visualize or args.gif_path is not None:
        vis_title = (
            f"FlowMatch compare | mode={args.mode} | "
            f"src={meta['segment_source_style']} -> tgt={style_names[target_style_label]} | "
            f"seg=[{start},{end})"
        )
        print(f"[Info] visualize backend: {args.vis_backend}")
        print("[Info] visualize mode   : left=reference | right=prediction")
        vis_info = visualize_pair_from_375(
            ref_features_375=x_ref_raw,
            pred_features_375=x_gen_raw,
            vis_backend=args.vis_backend,
            db_path=args.db,
            data_root=args.data_root,
            bvh_ref=args.bvh_ref,
            fps=args.fps,
            dt=args.dt,
            use_global=not args.local_only,
            root_wy_in_degrees=True,
            coordinate_convert=coordinate_convert,
            trail_len=min(120, args.num_frames),
            title=vis_title,
            gif_path=args.gif_path,
            show=args.visualize,
            geno_gap=args.geno_gap,
        )
        if vis_info["reference_bvh"] is not None:
            print(f"[Info] visualize bvh  : {vis_info['reference_bvh']}")
        if args.gif_path is not None:
            print(f"[Done] GIF saved      : {args.gif_path}")
        if args.visualize:
            print("[Done] Visualization complete.")
        else:
            print("[Done] GIF export complete.")
    else:
        print("[Done] Generation complete (visualization skipped).")


if __name__ == "__main__":
    main()
