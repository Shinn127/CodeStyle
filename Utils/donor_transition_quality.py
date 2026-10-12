"""Quality calibration and acceptance checks for the donor-transition pilot.

Contains:
  - continuity / contact / activity / fidelity metric computation on the frozen
    375-dim feature representation (raw path and frozen-VQ codec path);
  - reference calibration that freezes `quality_thresholds.json`;
  - raw/codec acceptance gates used by the builder;
  - an independent reader that validates manifest/sample artifacts without
    touching the training dataset or the VQ target cache.

Units follow the guide: positions cm, linear velocity m/s, acceleration m/s^2,
jerk m/s^3, angles deg, angular velocity deg/s, durations in frames and seconds.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from Utils import quat, motion375
from Utils.donor_transition import (
    DT,
    FPS,
    MARKERS,
    SELECTED_JOINTS,
    file_sha256,
    mask_intervals,
    shortest_arc_deg,
)
from tools.generate_database_100sty import extract_features_375  # noqa: F401  (API re-export)

THRESHOLDS_SCHEMA = "donor_transition_thresholds_v1"
MANIFEST_SCHEMA = "donor_transition_pilot_v1"

NUMERICAL_FLOORS = {
    "pos_step_cm": 1e-3,
    "lin_vel_mps": 1e-4,
    "lin_acc_mps2": 1e-2,
    "lin_jerk_mps3": 1.0,
    "rot_step_deg": 1e-3,
    "ang_vel_dps": 1e-2,
    "ang_acc_dps2": 1.0,
    "ang_jerk_dps3": 60.0,
}

METRIC_FAMILIES = (
    ("local_pos_step_cm", "pos_step_cm"),
    ("local_lin_vel_mps", "lin_vel_mps"),
    ("local_lin_acc_mps2", "lin_acc_mps2"),
    ("local_lin_jerk_mps3", "lin_jerk_mps3"),
    ("world_pos_step_cm", "pos_step_cm"),
    ("world_lin_vel_mps", "lin_vel_mps"),
    ("world_lin_acc_mps2", "lin_acc_mps2"),
    ("world_lin_jerk_mps3", "lin_jerk_mps3"),
    ("local_rot_step_deg", "rot_step_deg"),
    ("local_ang_vel_dps", "ang_vel_dps"),
    ("local_ang_acc_dps2", "ang_acc_dps2"),
    ("local_ang_jerk_dps3", "ang_jerk_dps3"),
    ("world_rot_step_deg", "rot_step_deg"),
    ("world_ang_vel_dps", "ang_vel_dps"),
    ("world_ang_acc_dps2", "ang_acc_dps2"),
    ("world_ang_jerk_dps3", "ang_jerk_dps3"),
)
FLOOR_OF = dict(METRIC_FAMILIES)

ABORT_REFERENCE_FAILURE_FRACTION = 0.20

_GLOBAL_JOINT_INDEX = {
    "RightFoot": 63, "RightToeBase": 64, "LeftFoot": 70, "LeftToeBase": 71,
}


# --------------------------------------------------------------------------
# pose extraction from the frozen feature layout
# --------------------------------------------------------------------------

@dataclass
class PoseSeries:
    local_pos: np.ndarray          # [T,24,3] root-relative selected positions (m)
    local_rot: np.ndarray          # [T,24,4] selected local rotations
    local_vel: np.ndarray          # [T,24,3] root-frame selected velocities (m/s)
    root_pos: np.ndarray           # [T,3]
    root_rot: np.ndarray           # [T,4]
    world_pos: np.ndarray          # [T,24,3]
    world_rot: np.ndarray          # [T,24,4]


def pose_from_features(
    features: np.ndarray,
    *,
    root_pos: np.ndarray | None = None,
    root_rot: np.ndarray | None = None,
) -> PoseSeries:
    """Decode selected-24 pose from features.

    Without an explicit root track the root channels inside the features are
    integrated (this is the pipeline convention; for raw features this equals the
    prescribed source root path).  Passing an explicit root track implements the
    codec 'prescribed' root口径 (decoded pose on the source root path).
    """
    _root_motion, local_pos, local_rot, local_vel = motion375.decode_selected_state(features)
    if root_pos is None or root_rot is None:
        root_pos, root_rot = motion375.reconstruct_root_track_from_375(features, dt=DT)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot = np.asarray(root_rot, dtype=np.float64)
    world_pos = quat.mul_vec(root_rot[:, None, :], local_pos) + root_pos[:, None, :]
    world_rot = quat.normalize(quat.mul(root_rot[:, None, :], local_rot))
    return PoseSeries(
        local_pos=np.asarray(local_pos, dtype=np.float64),
        local_rot=np.asarray(local_rot, dtype=np.float64),
        local_vel=np.asarray(local_vel, dtype=np.float64),
        root_pos=root_pos,
        root_rot=root_rot,
        world_pos=np.asarray(world_pos, dtype=np.float64),
        world_rot=np.asarray(world_rot, dtype=np.float64),
    )


def attach_exact_world(
    pose: PoseSeries,
    rotations: np.ndarray,
    positions: np.ndarray,
    parents: np.ndarray,
) -> PoseSeries:
    """Replace world/root fields with the exact FK of the final skeleton.

    The feature layout reconstructs the root by integrating velocity channels (the
    pipeline's inference convention); for raw acceptance metrics the exact FK of the
    final work clip is the ground truth, while local/velocity fields still come from
    the feature representation.
    """
    g_rot, g_pos = quat.fk(np.asarray(rotations, dtype=np.float32), np.asarray(positions, dtype=np.float32), parents)
    sel = SELECTED_JOINTS
    return PoseSeries(
        local_pos=pose.local_pos,
        local_rot=pose.local_rot,
        local_vel=pose.local_vel,
        root_pos=np.asarray(np.asarray(positions)[:, 0], dtype=np.float64),
        root_rot=np.asarray(g_rot[:, 0], dtype=np.float64),
        world_pos=np.asarray(g_pos[:, sel], dtype=np.float64),
        world_rot=np.asarray(quat.normalize(g_rot[:, sel]), dtype=np.float64),
    )


def _diff_series(x: np.ndarray, order: int) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    for _ in range(order):
        d = np.empty_like(out)
        if len(out) > 2:
            d[1:-1] = (out[2:] - out[:-2]) * (0.5 * FPS)
            d[0] = d[1]
            d[-1] = d[-2]
        else:
            d[:] = 0.0
        out = d
    return out


def _vec_norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=-1))


def continuity_series(pose: PoseSeries) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tag, pos in (("local", pose.local_pos), ("world", pose.world_pos)):
        out[f"{tag}_pos_step_cm"] = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * 100.0
        out[f"{tag}_lin_vel_mps"] = _vec_norm(_diff_series(pos, 1))
        out[f"{tag}_lin_acc_mps2"] = _vec_norm(_diff_series(pos, 2))
        out[f"{tag}_lin_jerk_mps3"] = _vec_norm(_diff_series(pos, 3))
    for tag, rot in (("local", pose.local_rot), ("world", pose.world_rot)):
        out[f"{tag}_rot_step_deg"] = shortest_arc_deg(rot[:-1], rot[1:]).astype(np.float64)
        out[f"{tag}_ang_vel_dps"] = _vec_norm(quat.to_scaled_angle_axis(_diff_series(rot, 1)))
        out[f"{tag}_ang_acc_dps2"] = _vec_norm(quat.to_scaled_angle_axis(_diff_series(rot, 2)))
        out[f"{tag}_ang_jerk_dps3"] = _vec_norm(quat.to_scaled_angle_axis(_diff_series(rot, 3)))
    return out


def window_peaks(series: np.ndarray) -> tuple[float, float]:
    joint_max = np.max(series, axis=-1)
    joint_mean = np.mean(series, axis=-1)
    return float(np.max(joint_max)), float(np.max(joint_mean))


def velocity_consistency(pose: PoseSeries) -> float:
    """Max |implied world velocity - finite-difference world velocity| over joints/frames."""
    implied = quat.mul_vec(pose.root_rot[:, None, :], pose.local_vel)
    diff = np.empty_like(pose.world_pos)
    if len(pose.world_pos) > 2:
        diff[1:-1] = (pose.world_pos[2:] - pose.world_pos[:-2]) * (0.5 * FPS)
        diff[0] = diff[1]
        diff[-1] = diff[-2]
    else:
        diff[:] = 0.0
    return float(np.max(np.linalg.norm(implied - diff, axis=-1)))


def marker_indices() -> dict[str, int]:
    return {name: SELECTED_JOINTS.index(idx) for name, idx in _GLOBAL_JOINT_INDEX.items()}


def marker_world_positions(pose: PoseSeries) -> dict[str, np.ndarray]:
    idx = marker_indices()
    return {name: pose.world_pos[:, i] for name, i in idx.items()}


def marker_local_positions(local_pos: np.ndarray) -> dict[str, np.ndarray]:
    idx = marker_indices()
    return {name: local_pos[:, i] for name, i in idx.items()}


def marker_speed_series(points: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(points)
    if len(points) > 2:
        vel[1:-1] = (points[2:] - points[:-2]) * (0.5 * FPS)
        vel[0] = vel[1]
        vel[-1] = vel[-2]
    return np.linalg.norm(vel[:, [0, 2]], axis=-1)


def contact_from_features(
    features: np.ndarray,
    *,
    pose: PoseSeries | None = None,
    proxies: dict | None = None,
) -> dict:
    """Proxies (frozen when given) + stable masks computed from a feature sequence."""
    from Utils.donor_transition import contact_mask

    pose = pose if pose is not None else pose_from_features(features)
    world = marker_world_positions(pose)
    heights = {name: world[name][:, 1] for name in MARKERS}
    speeds = {name: marker_speed_series(world[name]) for name in MARKERS}
    if proxies is None:
        proxies = {}
        for name in MARKERS:
            slow = speeds[name] <= 0.10
            values = heights[name][slow] if slow.sum() >= 5 else heights[name]
            proxies[name] = float(np.percentile(values, 1))
    masks = {name: contact_mask(heights[name], speeds[name], proxies[name]) for name in MARKERS}
    return {"proxies": dict(proxies), "heights": heights, "speeds": speeds, "world": world, "masks": masks, "pose": pose}


def stance_slip_stats(markers: dict, masks: dict) -> dict:
    """Slip metrics on each marker's own stable mask: P95/max speed and anchor drift."""
    per_marker = {}
    worst = {"p95": 0.0, "max": 0.0, "drift": 0.0}
    frames = 0
    for name in MARKERS:
        mask = masks[name]
        if not mask.any():
            per_marker[name] = None
            continue
        speed = markers["speeds"][name]
        pts = markers[name]
        drift_max = 0.0
        for a, b in mask_intervals(mask, 1):
            anchor = pts[a:b].mean(axis=0)
            d = np.linalg.norm((pts[a:b] - anchor)[:, [0, 2]], axis=-1)
            drift_max = max(drift_max, float(np.max(d)))
        entry = {
            "p95": float(np.percentile(speed[mask], 95)),
            "max": float(np.max(speed[mask])),
            "drift": float(drift_max),
            "frames": int(mask.sum()),
        }
        per_marker[name] = entry
        frames += entry["frames"]
        worst["p95"] = max(worst["p95"], entry["p95"])
        worst["max"] = max(worst["max"], entry["max"])
        worst["drift"] = max(worst["drift"], entry["drift"])
    return {"per_marker": per_marker, "worst": worst, "contact_frames": int(frames)}


def ground_proxy_depth(heights: dict, masks: dict, proxies: dict) -> float:
    depth = -1e9
    for name in MARKERS:
        mask = masks[name]
        if not mask.any():
            continue
        depth = max(depth, float(proxies[name] - np.min(heights[name][mask])))
    return float(depth) if depth > -1e8 else 0.0


def feet_activity(local_pos: np.ndarray) -> dict:
    """Root-removed foot activity: first-difference RMS and 60-frame separation amplitude."""
    markers = marker_local_positions(local_pos)
    rms = []
    for name in MARKERS:
        d = np.linalg.norm(np.diff(markers[name], axis=0), axis=-1)
        rms.append(float(np.sqrt(np.mean(d**2)) if len(d) else 0.0))
    feet_rms = float(np.mean(rms))
    left_mid = 0.5 * (markers["LeftFoot"] + markers["LeftToeBase"])
    right_mid = 0.5 * (markers["RightFoot"] + markers["RightToeBase"])
    rel = left_mid - right_mid
    amplitude = 0.0
    if len(rel) >= 2:
        for start in range(0, max(len(rel) - 60 + 1, 1)):
            seg = rel[start:start + 60]
            if len(seg) < 2:
                continue
            amplitude = max(amplitude, float(np.max(seg[:, 0]) - np.min(seg[:, 0])))
            amplitude = max(amplitude, float(np.max(seg[:, 2]) - np.min(seg[:, 2])))
    return {"feet_rms": feet_rms, "separation_amplitude": amplitude}


def root_planar_speed(root_pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_diff_series(root_pos, 1)[:, [0, 2]], axis=-1)


# --------------------------------------------------------------------------
# VQ codec path
# --------------------------------------------------------------------------

def codec_encode_decode(
    model,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device,
    chunk_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Frozen VQ encode->quantize->decode over rows; returns (codec_features, indices, gpu_seconds)."""
    import time

    import torch

    x = np.asarray(features, dtype=np.float32)
    n = x.shape[0]
    out = np.empty_like(x)
    indices = np.empty((n, int(model.num_latents)), dtype=np.uint16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, n, chunk_rows):
            end = min(begin + chunk_rows, n)
            norm = (x[begin:end] - mean) / std
            batch = torch.from_numpy(np.ascontiguousarray(norm)).to(device)
            z_q, idx, _ = model.encode(batch)
            motion_hat = model.decode(z_q)
            out[begin:end, :84] = x[begin:end, :84]
            out[begin:end, 84:] = motion_hat.float().cpu().numpy() * std[84:] + mean[84:]
            indices[begin:end] = idx.cpu().numpy().astype(np.uint16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, indices, float(time.perf_counter() - start)


def codec_variants(raw_features: np.ndarray, codec: np.ndarray, raw_pose: PoseSeries) -> dict[str, PoseSeries]:
    """The two frozen codec root口径."""
    return {
        "prescribed": pose_from_features(codec, root_pos=raw_pose.root_pos, root_rot=raw_pose.root_rot),
        "integrated": pose_from_features(codec),
    }


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

@dataclass
class ReferenceWindow:
    entry_id: int
    style: str
    suffix: str
    row_start: int
    label: int
    speed_bucket: int = -1


def sample_reference_windows(index: dict, plan: Sequence[dict], *, seed: int = 42) -> list[ReferenceWindow]:
    """Deterministic per-clip window sampling; `plan` lists dicts with entry_id/windows."""
    rng = np.random.default_rng(seed)
    windows: list[ReferenceWindow] = []
    for item in plan:
        entry_id = int(item["entry_id"])
        seg = next((s for s in index["segments"] if int(s["source_entry_id"]) == entry_id), None)
        if seg is None:
            continue
        n_rows = int(seg["num_frames"])
        count = min(int(item["windows"]), max(n_rows - 240, 0) // 240)
        if count <= 0:
            continue
        max_start = n_rows - 240
        starts = np.linspace(0, max_start, num=count, dtype=int)
        starts = np.unique(np.clip(starts + int(rng.integers(0, 60)), 0, max_start))
        for start in starts.tolist():
            windows.append(
                ReferenceWindow(
                    entry_id=entry_id,
                    style=str(seg["style_name"]),
                    suffix=str(seg["suffix"]),
                    row_start=int(seg["frame_start"]) + int(start),
                    label=int(seg["label"]),
                )
            )
    return windows


def read_db_rows(index: dict, row_start: int, count: int) -> np.ndarray:
    meta = index["meta"]
    shape = (int(meta["num_frames"]), 375)
    x = np.memmap(index["paths"]["x"], dtype=np.float32, mode="r", shape=shape)
    return np.array(x[row_start:row_start + count], dtype=np.float32, copy=True)


def calibrate_thresholds(
    index: dict,
    windows: list[ReferenceWindow],
    *,
    proxies: dict | None = None,
    model=None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    device=None,
    chunk_rows: int = 2048,
    want_codec: bool = True,
) -> tuple[dict, dict]:
    """Compute reference distributions and freeze raw/codec limits.

    `proxies` are the frozen marker floor proxies from all train references; when
    omitted they are estimated from these windows alone (callers should pass the
    frozen set so contact evidence and samples share one floor definition).
    """
    n_windows = len(windows)
    all_feats = (
        np.stack([read_db_rows(index, w.row_start, 240) for w in windows])
        if windows
        else np.empty((0, 240, 375), np.float32)
    )
    raw_peaks: dict[str, list[tuple[float, float]]] = {name: [] for name, _ in METRIC_FAMILIES}
    raw_activity, raw_slip, raw_depth, raw_velc = [], [], [], []
    speed_buckets: list[int] = []
    for i, window in enumerate(windows):
        feats = all_feats[i]
        pose = pose_from_features(feats)
        series = continuity_series(pose)
        for name, _floor in METRIC_FAMILIES:
            raw_peaks[name].append(window_peaks(series[name]))
        contact = contact_from_features(feats, pose=pose, proxies=proxies)
        raw_slip.append(stance_slip_stats({**contact["world"], "speeds": contact["speeds"]}, contact["masks"]))
        raw_depth.append(ground_proxy_depth(contact["heights"], contact["masks"], contact["proxies"]))
        raw_activity.append(feet_activity(pose.local_pos))
        raw_velc.append(velocity_consistency(pose))
        speed = float(np.mean(root_planar_speed(pose.root_pos)))
        window.speed_bucket = int(speed // 0.25)
        speed_buckets.append(window.speed_bucket)

    codec_seconds = 0.0
    codec_peaks: dict[str, list[tuple[float, float]]] = {name: [] for name, _ in METRIC_FAMILIES}
    codec_activity, codec_slip, codec_depth, codec_velc = [], [], [], []
    if model is not None and want_codec and n_windows:
        flat = all_feats.reshape(-1, 375)
        codec_flat, _indices, codec_seconds = codec_encode_decode(
            model, flat, mean, std, device=device, chunk_rows=chunk_rows
        )
        codec_all = codec_flat.reshape(n_windows, 240, 375)
        for i, window in enumerate(windows):
            feats = all_feats[i]
            codec = codec_all[i]
            raw_pose = pose_from_features(feats)
            contact = contact_from_features(feats, pose=raw_pose, proxies=proxies)
            for _tag, pose in codec_variants(feats, codec, raw_pose).items():
                series_c = continuity_series(pose)
                for name, _floor in METRIC_FAMILIES:
                    codec_peaks[name].append(window_peaks(series_c[name]))
                codec_activity.append(feet_activity(pose.local_pos))
                codec_velc.append(velocity_consistency(pose))
            codec_pose = pose_from_features(codec, root_pos=raw_pose.root_pos, root_rot=raw_pose.root_rot)
            codec_contact = contact_from_features(codec, pose=codec_pose, proxies=proxies)
            codec_slip.append(
                stance_slip_stats({**codec_contact["world"], "speeds": codec_contact["speeds"]}, contact["masks"])
            )
            codec_depth.append(ground_proxy_depth(codec_contact["heights"], contact["masks"], contact["proxies"]))

    thresholds = {
        "schema": THRESHOLDS_SCHEMA,
        "dt": DT,
        "fps": FPS,
        "seed": 42,
        "numerical_floors": NUMERICAL_FLOORS,
        "raw": {"continuity": {}, "contact": {}, "activity": {}, "velocity_consistency": {}, "ground_proxy": {}},
        "codec": {"continuity": {}, "contact": {}, "activity": {}, "velocity_consistency": {}, "ground_proxy": {}},
        "reference": {
            "window_count": n_windows,
            "speed_buckets": speed_buckets,
            "suffix_counts": {},
        },
        "notes": {
            "limit_rule": "limit = max(P99(reference window peaks), numerical floor)",
            "contact_mask_rule": (
                "stable runs: height<=proxy+2cm and speed<=0.10 m/s for >=3 frames, bridged over "
                "<=2 failing frames below height<=proxy+4cm and speed<=0.20 m/s"
            ),
            "raw_contact_caps": "raw slip P95<=0.10 m/s, max<=0.20 m/s, anchor drift<=2cm; references recorded",
        },
    }
    for window in windows:
        key = window.suffix
        thresholds["reference"]["suffix_counts"][key] = thresholds["reference"]["suffix_counts"].get(key, 0) + 1
    for target, peaks in (("raw", raw_peaks), ("codec", codec_peaks)):
        for name, floor_key in METRIC_FAMILIES:
            floor = NUMERICAL_FLOORS[floor_key]
            arr = np.asarray(peaks[name], dtype=np.float64)
            if arr.size == 0:
                continue
            p99_max = float(np.percentile(arr[:, 0], 99))
            p99_mean = float(np.percentile(arr[:, 1], 99))
            thresholds[target]["continuity"][name] = {
                "floor": floor,
                "ref_p99_peak_max": p99_max,
                "ref_p99_peak_mean": p99_mean,
                "limit_max": max(p99_max, floor),
                "limit_mean": max(p99_mean, floor),
            }

    def _limit(records: list[dict], key: str, floor: float) -> tuple[float, float]:
        vals = np.asarray([r["worst"][key] for r in records], dtype=np.float64)
        if vals.size == 0:
            return floor, float("nan")
        p99 = float(np.percentile(vals, 99))
        return max(p99, floor), p99

    for target, records, depth_records, vel_records in (
        ("raw", raw_slip, raw_depth, raw_velc),
        ("codec", codec_slip, codec_depth, codec_velc),
    ):
        p95_limit, p95_ref = _limit(records, "p95", 1e-4)
        max_limit, max_ref = _limit(records, "max", 1e-4)
        drift_limit, drift_ref = _limit(records, "drift", 1e-4)
        if target == "raw":
            p95_limit = min(0.10, p95_limit)
            max_limit = min(0.20, max_limit)
            drift_limit = min(0.02, drift_limit)
        thresholds[target]["contact"] = {
            "slip_p95_limit_mps": p95_limit,
            "slip_max_limit_mps": max_limit,
            "anchor_drift_limit_m": drift_limit,
            "ref_p99_window_p95": p95_ref,
            "ref_p99_window_max": max_ref,
            "ref_p99_window_drift": drift_ref,
        }
        depth_arr = np.asarray(depth_records, dtype=np.float64)
        depth_p99 = float(np.percentile(depth_arr, 99)) if depth_arr.size else float("nan")
        limits_depth = 0.01 if target == "raw" else max(depth_p99, 0.01)
        thresholds[target]["ground_proxy"] = {
            "max_penetration_limit_m": float(limits_depth),
            "ref_p99_penetration_m": depth_p99,
        }
        vel_arr = np.asarray(vel_records, dtype=np.float64)
        vel_p99 = float(np.percentile(vel_arr, 99)) if vel_arr.size else 0.0
        thresholds[target]["velocity_consistency"] = {
            "limit_mps": float(max(vel_p99, 1e-4)),
            "ref_p99_mps": vel_p99,
        }

    def _activity_bounds(records: list[dict], buckets_in: list[int], suffixes_in: list[str]) -> dict:
        """P01 lower bounds by suffix+speed bucket, with documented suffix/global fallbacks."""
        out = {"bucket_scope": 0.25, "min_windows_in_bucket": 8, "bounds": {}}
        for metric in ("feet_rms", "separation_amplitude"):
            all_vals = np.asarray([r[metric] for r in records], dtype=np.float64)
            global_p01 = float(np.percentile(all_vals, 1)) if all_vals.size else 0.0
            by_suffix: dict[str, dict] = {}
            for suffix in sorted({str(s) for s in suffixes_in}):
                vals = np.asarray(
                    [r[metric] for r, s in zip(records, suffixes_in) if str(s) == suffix], dtype=np.float64
                )
                entry = {"suffix_p01": float(np.percentile(vals, 1)) if vals.size else 0.0,
                         "suffix_windows": int(vals.size), "buckets": {}}
                for bucket in sorted({int(b) for b, s in zip(buckets_in, suffixes_in) if str(s) == suffix}):
                    vals_b = np.asarray(
                        [r[metric] for r, s, b in zip(records, suffixes_in, buckets_in)
                         if str(s) == suffix and int(b) == bucket],
                        dtype=np.float64,
                    )
                    if vals_b.size >= 8:
                        entry["buckets"][str(bucket)] = {
                            "bound_p01": float(np.percentile(vals_b, 1)),
                            "windows": int(vals_b.size),
                        }
                by_suffix[suffix] = entry
            out["bounds"][metric] = {"global_p01": global_p01, "suffixes": by_suffix}
        return out

    raw_suffixes = [w.suffix for w in windows]
    thresholds["raw"]["activity"] = _activity_bounds(raw_activity, speed_buckets, raw_suffixes)
    codec_suffixes = [w.suffix for w in windows for _ in range(2)] if codec_activity else []
    codec_buckets = [b for b in speed_buckets for _ in range(2)] if codec_activity else []
    thresholds["codec"]["activity"] = _activity_bounds(codec_activity, codec_buckets, codec_suffixes)
    details = {
        "gpu_seconds": codec_seconds,
        "raw_activity_windows": len(raw_activity),
        "codec_activity_windows": len(codec_activity),
        "reference_windows": [
            {"entry_id": w.entry_id, "style": w.style, "suffix": w.suffix, "row_start": w.row_start,
             "speed_bucket": w.speed_bucket}
            for w in windows
        ],
        "raw_slip_reference": {
            "worst_p95_values": [r["worst"]["p95"] for r in raw_slip],
            "worst_max_values": [r["worst"]["max"] for r in raw_slip],
            "worst_drift_values": [r["worst"]["drift"] for r in raw_slip],
        },
        "raw_depth_values": raw_depth,
    }
    return thresholds, details


def activity_bound(thresholds: dict, target: str, metric: str, suffix: str, speed: float) -> tuple[float, str]:
    """Resolve the activity lower bound: suffix+bucket, then suffix, then global (recorded)."""
    info = thresholds[target]["activity"]["bounds"][metric]
    scope = float(thresholds[target]["activity"]["bucket_scope"])
    entry = info.get("suffixes", {}).get(str(suffix))
    if entry is not None:
        bucket = str(int(speed // scope))
        if bucket in entry["buckets"]:
            return float(entry["buckets"][bucket]["bound_p01"]), f"suffix_bucket:{suffix}:{bucket}"
        if entry.get("suffix_windows", 0) >= 1:
            return float(entry["suffix_p01"]), f"suffix_fallback:{suffix}"
    return float(info["global_p01"]), "global_fallback"


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def evaluate_continuity(thresholds: dict, target: str, series: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    failures = []
    metrics = {}
    for name, values in series.items():
        entry = thresholds[target]["continuity"].get(name)
        if entry is None:
            continue
        mx, mn = window_peaks(values)
        metrics[name] = {"peak_max": mx, "peak_mean": mn,
                         "limit_max": entry["limit_max"], "limit_mean": entry["limit_mean"]}
        if mx > entry["limit_max"]:
            failures.append({"check": f"{target}_continuity_{name}_max", "value": mx, "limit": entry["limit_max"]})
        if mn > entry["limit_mean"]:
            failures.append({"check": f"{target}_continuity_{name}_mean", "value": mn, "limit": entry["limit_mean"]})
    return failures, metrics


def evaluate_velocity_consistency(thresholds: dict, target: str, value: float) -> tuple[list[dict], float]:
    limit = thresholds[target]["velocity_consistency"]["limit_mps"]
    if value > limit:
        return [{"check": f"{target}_velocity_consistency", "value": value, "limit": limit}], limit
    return [], limit


def evaluate_contact(thresholds: dict, target: str, stats: dict, depth: float) -> tuple[list[dict], dict]:
    failures = []
    if stats["contact_frames"] == 0:
        failures.append({"check": f"{target}_zero_contact_markers", "value": 0, "limit": ">0"})
        return failures, {"frames": 0}
    worst = stats["worst"]
    limits = thresholds[target]["contact"]
    if worst["p95"] > limits["slip_p95_limit_mps"]:
        failures.append({"check": f"{target}_contact_slip_p95", "value": worst["p95"], "limit": limits["slip_p95_limit_mps"]})
    if worst["max"] > limits["slip_max_limit_mps"]:
        failures.append({"check": f"{target}_contact_slip_max", "value": worst["max"], "limit": limits["slip_max_limit_mps"]})
    if worst["drift"] > limits["anchor_drift_limit_m"]:
        failures.append({"check": f"{target}_contact_anchor_drift", "value": worst["drift"], "limit": limits["anchor_drift_limit_m"]})
    depth_limit = thresholds[target]["ground_proxy"]["max_penetration_limit_m"]
    if depth > depth_limit:
        failures.append({"check": f"{target}_ground_penetration", "value": depth, "limit": depth_limit})
    return failures, {"worst": worst, "penetration_m": depth, "frames": stats["contact_frames"]}


def evaluate_activity(thresholds: dict, target: str, activity: dict, suffix: str, speed: float) -> tuple[list[dict], dict]:
    failures = []
    resolved = {}
    for metric, value in activity.items():
        bound, scope = activity_bound(thresholds, target, metric, suffix, speed)
        resolved[metric] = {"value": value, "bound_p01": bound, "scope": scope}
        if value < bound:
            failures.append({"check": f"{target}_activity_{metric}", "value": value, "limit": bound})
    return failures, resolved


def bone_length_check(rotations: np.ndarray, positions: np.ndarray, parents: np.ndarray, offsets: np.ndarray) -> float:
    """Max |FK bone length - header rest offset length| over the FULL skeleton (m)."""
    _g_rot, g_pos = quat.fk(rotations, positions, parents)
    worst = 0.0
    for i in range(1, len(parents)):
        rest = float(np.linalg.norm(offsets[i]))
        if rest < 1e-6:
            continue
        actual = float(np.max(np.linalg.norm(g_pos[:, i] - g_pos[:, int(parents[i])], axis=-1)))
        worst = max(worst, abs(actual - rest))
    return worst


def bone_length_reference_stats(
    deviations: Sequence[float],
    *,
    limit: float = 1e-4,
) -> dict:
    """Reference-side summary for the literal raw bone-length gate (guide section 7).

    Returns the per-window failure fraction and distribution so a run can decide
    whether the fixed 1e-4 m gate is compatible with the source data at all
    (guide section 5.1: >20% real-reference failure stops generation).
    """
    values = np.asarray(list(deviations), dtype=np.float64)
    if values.size == 0:
        return {"windows": 0, "failure_fraction": float("nan"), "limit_m": limit}
    failing = values > limit
    return {
        "windows": int(values.size),
        "limit_m": float(limit),
        "failure_fraction": float(np.mean(failing)),
        "failing_windows": int(np.sum(failing)),
        "p50_m": float(np.percentile(values, 50)),
        "p90_m": float(np.percentile(values, 90)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(values.max()),
        "abort_threshold_fraction": ABORT_REFERENCE_FAILURE_FRACTION,
        "abort_required": bool(float(np.mean(failing)) > ABORT_REFERENCE_FAILURE_FRACTION),
    }


def quaternion_norm_error(*arrays: np.ndarray) -> float:
    worst = 0.0
    for arr in arrays:
        worst = max(worst, float(np.max(np.abs(np.linalg.norm(arr, axis=-1) - 1.0))))
    return worst


def rigid_align_xz_yaw(src_xz: np.ndarray, dst_xz: np.ndarray) -> tuple[float, np.ndarray]:
    """Single yaw+translation aligning src onto dst (no scale).

    Returns (theta, shift) with the convention of `apply_align_xz_yaw`; theta is the
    rotation that turns dst into src (its inverse aligns src onto dst).
    """
    a0 = src_xz - src_xz.mean(axis=0)
    b0 = dst_xz - dst_xz.mean(axis=0)
    num = float(np.sum(a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0]))
    den = float(np.sum(a0[:, 0] * b0[:, 0] + a0[:, 1] * b0[:, 1]))
    theta = float(np.arctan2(num, den))
    # rotate src by the inverse yaw so that it lines up with dst, then match the means
    c, s = np.cos(theta), np.sin(theta)
    rotated = np.stack([c * src_xz[:, 0] - s * src_xz[:, 1], s * src_xz[:, 0] + c * src_xz[:, 1]], axis=1)
    shift = dst_xz.mean(axis=0) - rotated.mean(axis=0)
    return theta, shift


def apply_align_xz_yaw(points: np.ndarray, theta: float, shift: np.ndarray) -> np.ndarray:
    """Apply the inverse of the yaw rotation measured by `rigid_align_xz_yaw`."""
    out = np.array(points, dtype=np.float64, copy=True)
    c, s = np.cos(theta), np.sin(theta)
    x, z = out[..., 0].copy(), out[..., 2].copy()
    out[..., 0] = c * x - s * z + shift[0]
    out[..., 2] = s * x + c * z + shift[1]
    return out


def root_up_tilt_deg(root_rot: np.ndarray) -> np.ndarray:
    up = quat.mul_vec(np.asarray(root_rot, dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    return np.degrees(np.arccos(np.clip(up[:, 1], -1.0, 1.0)))


def suffix_fidelity(
    constructed_global_pos: np.ndarray,
    constructed_root_rot: np.ndarray,
    constructed_root_pos: np.ndarray,
    constructed_proxy: float,
    donor_global_pos: np.ndarray,
    donor_root_rot: np.ndarray,
    donor_root_pos: np.ndarray,
    donor_proxy: float,
) -> dict:
    """B-suffix fidelity vs the original donor after one rigid XZ/yaw alignment (guide 7)."""
    sel = SELECTED_JOINTS
    src_hip = np.asarray(constructed_global_pos)[:, sel[0]][:, [0, 2]]
    dst_hip = np.asarray(donor_global_pos)[:, sel[0]][:, [0, 2]]
    theta, shift = rigid_align_xz_yaw(src_hip, dst_hip)
    aligned = apply_align_xz_yaw(np.asarray(constructed_global_pos), theta, shift)
    diff = aligned[:, sel] - np.asarray(donor_global_pos)[:, sel]
    per_joint = np.linalg.norm(diff, axis=-1)
    tilt_c = root_up_tilt_deg(constructed_root_rot)
    tilt_d = root_up_tilt_deg(donor_root_rot)
    height_c = np.asarray(constructed_root_pos)[:, 1] - constructed_proxy
    height_d = np.asarray(donor_root_pos)[:, 1] - donor_proxy
    return {
        "mean_pose_dev_m": float(per_joint.mean()),
        "max_pose_dev_m": float(per_joint.max()),
        "yaw_offset_deg": float(np.degrees(theta)),
        "root_tilt_diff_deg": float(np.max(np.abs(tilt_c - tilt_d))),
        "root_height_diff_m": float(np.max(np.abs(height_c - height_d))),
    }


# --------------------------------------------------------------------------
# independent reader
# --------------------------------------------------------------------------

class ArtifactError(RuntimeError):
    pass


def read_manifest(run_dir: Path, *, require_complete: bool = True) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(f"manifest.json missing in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ArtifactError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    if require_complete and not manifest.get("complete", False):
        raise ArtifactError("manifest.complete is false; the run did not finish writing")
    for record in manifest.get("artifacts", []):
        path = run_dir / record["path"]
        if not path.exists():
            raise ArtifactError(f"artifact missing: {record['path']}")
        if file_sha256(path) != record["sha256"]:
            raise ArtifactError(f"artifact fingerprint mismatch (replaced or corrupted): {record['path']}")
    return manifest


def load_sample(run_dir: Path, sample_id: str, *, allow_pending_review: bool = False) -> dict:
    """Independent reader for one machine-pass sample (never train-eligible by default)."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    record = next((item for item in manifest.get("samples", []) if item["sample_id"] == sample_id), None)
    if record is None:
        raise ArtifactError(f"sample not registered in manifest: {sample_id}")
    if record.get("quality_status") != "machine_pass":
        raise ArtifactError(f"sample {sample_id} did not pass machine quality checks")
    if record.get("review_status") != "approved" and not allow_pending_review:
        raise ArtifactError(
            f"sample {sample_id} is review_status={record.get('review_status')!r}; pass "
            "allow_pending_review=True for audit access (not train-eligible)"
        )
    npz_path = run_dir / record["file"]
    if not npz_path.exists():
        raise ArtifactError(f"sample file missing: {record['file']}")
    if file_sha256(npz_path) != record["sha256"]:
        raise ArtifactError(f"sample fingerprint mismatch: {record['file']}")
    with np.load(npz_path) as data:
        x = data["x"]
        labels = data["command_labels"]
        raw_ids = data["raw_frame_ids"]
        codes = data["code_indices"]
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 375:
            raise ArtifactError(f"bad x shape/dtype: {x.shape} {x.dtype}")
        if len(labels) != len(x) or len(raw_ids) != len(x):
            raise ArtifactError("row-aligned arrays disagree")
        if int(raw_ids[0]) != int(record["raw_frame_ids"][0]) or int(raw_ids[-1]) != int(record["raw_frame_ids"][1]):
            raise ArtifactError("raw frame ids disagree with the manifest")
        lo, hi = record["allowed_raw_range"]
        if int(raw_ids[0]) < lo or int(raw_ids[-1]) > hi:
            raise ArtifactError("sample rows cross the recorded segment boundary")
        if codes.ndim != 2 or codes.shape[0] != len(x):
            raise ArtifactError("code indices shape mismatch")
        return {
            "x": x.astype(np.float32),
            "command_labels": labels.astype(np.int8),
            "raw_frame_ids": raw_ids.astype(np.int64),
            "code_indices": codes.astype(np.uint16),
            "transition_rows": data["transition_rows"].astype(np.int64),
            "valid_window_starts": data["valid_window_starts"].astype(np.int64),
            "local_rotations": data["local_rotations"].astype(np.float32),
            "positions": data["positions"].astype(np.float32),
            "meta": record,
            "train_eligible": bool(record.get("train_eligible", False)),
        }


def sha256_json_dump(payload: object, path: Path) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)
    return file_sha256(path)
