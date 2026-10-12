"""Offline donor-transition pilot builder (CLI, scheduling, artifacts).

Usage (bounded validation first, then the full pilot in a fresh run dir):

  python tools/build_donor_transition_dataset.py \
    --db database_100sty_375_memmap_nomirror \
    --vq-ckpt checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt \
    --norm-stats checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz \
    --out analysis/donor_transition_pilot_v1/<run_id> \
    --device cuda --seed 42 --max-pairs 4 --max-accepted 4 --max-videos 2

All motion arrays and VQ computation stay FP32 / matmul precision 'highest' /
TF32 disabled.  The tool never writes outside its run directory and never
modifies the database, caches or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version as _dist_version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Utils.bvh as bvh  # noqa: E402
from Utils import quat  # noqa: E402
from Utils.donor_transition import (  # noqa: E402
    DT,
    FPS,
    IK_MAX_CORRECTION,
    IK_MAX_JOINT_DELTA_DEG,
    LEAD_CONTEXT,
    MARKERS,
    PILOT_STYLES,
    PILOT_SUFFIXES,
    PREFIX_ROWS,
    SEED,
    SELECTED_JOINTS,
    SUFFIX_ROWS,
    TRANSITION_FRAMES,
    anchor_candidates,
    blend_start_local,
    build_transition,
    command_schedule,
    file_sha256,
    find_segment,
    json_sha256,
    load_clip,
    load_db_index,
    lock_feet_ik,
    mask_intervals,
    plan_contact_masks,
    raw_frame_ids_for_rows,
    scored_rows,
    select_pairs,
    set_proxies,
    skeleton_compatible,
    suffix_rows_start_local,
    valid_window_starts,
    work_length,
)
from Utils.donor_transition_quality import (  # noqa: E402
    ABORT_REFERENCE_FAILURE_FRACTION,
    MANIFEST_SCHEMA,
    attach_exact_world,
    bone_length_check,
    bone_length_reference_stats,
    calibrate_thresholds,
    codec_encode_decode,
    codec_variants,
    contact_from_features,
    continuity_series,
    evaluate_activity,
    evaluate_contact,
    evaluate_continuity,
    evaluate_velocity_consistency,
    feet_activity,
    ground_proxy_depth,
    pose_from_features,
    quaternion_norm_error,
    read_db_rows,
    root_planar_speed,
    sample_reference_windows,
    sha256_json_dump,
    stance_slip_stats,
    suffix_fidelity,
    velocity_consistency,
)
from tools.generate_database_100sty import extract_features_375  # noqa: E402

VERSION = "donor_transition_pilot_v1.0"

CALIBRATION_PLAN = {
    "targets": [
        {"style": "Neutral", "suffixes": ["BW", "SW"], "windows": 6},
        {"style": "ArmsAboveHead", "suffixes": ["BW", "SW"], "windows": 6},
        {"style": "ArmsFolded", "suffixes": ["FW", "BW"], "windows": 6},
        {"style": "ArmsBehindBack", "suffixes": ["FW", "BW", "SW"], "windows": 6},
    ],
    "others": [
        {"style": "Angry", "suffix": "BW", "windows": 6},
        {"style": "BouncyLeft", "suffix": "FW", "windows": 6},
        {"style": "DuckFoot", "suffix": "BW", "windows": 6},
        {"style": "HighKnees", "suffix": "FW", "windows": 6},
        {"style": "March", "suffix": "SW", "windows": 6},
        {"style": "Old", "suffix": "BW", "windows": 6},
        {"style": "Proud", "suffix": "SW", "windows": 6},
        {"style": "Rushed", "suffix": "FW", "windows": 6},
    ],
}
CALIBRATION_PLAN["total_windows"] = sum(
    item["windows"] * len(item.get("suffixes", [item.get("suffix")]))
    for item in CALIBRATION_PLAN["targets"] + CALIBRATION_PLAN["others"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative offline donor-transition pilot builder")
    parser.add_argument("--db", type=Path, default=ROOT / "database_100sty_375_memmap_nomirror")
    parser.add_argument(
        "--vq-ckpt",
        type=Path,
        default=ROOT / "checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt",
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=ROOT / "checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Run directory; default analysis/donor_transition_pilot_v1/<run_id>")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--max-accepted", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--max-gpu-seconds", type=float, default=120.0)
    parser.add_argument("--max-total-seconds", type=float, default=30 * 60.0)
    return parser.parse_args()


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_runtime(device) -> None:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False


class RunState:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.started = time.time()
        self.gpu_seconds = 0.0
        self.attempts: list[dict] = []
        self.artifacts: list[dict] = []
        self.thresholds: dict = {}

    def elapsed(self) -> float:
        return time.time() - self.started

    def register_artifact(self, path: Path) -> dict:
        record = {"path": str(Path(path).relative_to(self.run_dir)), "sha256": file_sha256(path)}
        self.artifacts.append(record)
        return record


def load_target_clips(index: dict, proxies: dict) -> dict:
    clips = {}
    for style in PILOT_STYLES:
        for suffix in PILOT_SUFFIXES:
            seg = find_segment(index, style, suffix)
            if seg is None or seg["split"] != "train":
                continue
            clip = load_clip(
                Path(seg["source_path"]), int(seg["source_start"]), int(seg["source_stop"]),
                style=style, suffix=suffix, split=seg["split"],
                entry_id=int(seg["source_entry_id"]), label=int(seg["label"]),
            )
            set_proxies(clip, proxies)
            clips[clip.entry_id] = clip
    return clips


# --------------------------------------------------------------------------
# phase 0: identity
# --------------------------------------------------------------------------

_norm_stats_cache: dict[str, np.ndarray] = {}


def identity_check(index: dict, identity_ids: list[int], progress=print) -> dict:
    from tools.generate_database_100sty import prepare_motion

    results, ok = [], True
    mean, std = _norm_stats_cache["mean"], _norm_stats_cache["std"]
    for entry_id in identity_ids:
        seg = next(s for s in index["segments"] if int(s["source_entry_id"]) == int(entry_id))
        raw = bvh.load_zeroeggs(str(seg["source_path"]))
        frametime = float(raw.get("frametime") or 0.0)
        parents, rotations, positions = prepare_motion(
            raw, start=int(seg["source_start"]), stop=int(seg["source_stop"]), coordinate_convert=False
        )
        feats = extract_features_375(rotations, positions, parents, SELECTED_JOINTS, 60, 10)
        ref = read_db_rows(index, int(seg["frame_start"]), int(seg["num_frames"]))
        n = min(len(feats), len(ref))
        raw_diff = np.abs(feats[:n] - ref[:n])
        norm_diff = raw_diff / std
        entry = {
            "entry_id": int(entry_id),
            "style": seg["style_name"], "suffix": seg["suffix"], "split": seg["split"],
            "rows": int(n),
            "raw_max_abs_diff": float(raw_diff.max()),
            "raw_mean_abs_diff": float(raw_diff.mean()),
            "normalized_max_abs_diff": float(norm_diff.max()),
            "frametime": frametime,
            "frametime_ok": bool(abs(frametime - 1.0 / 60.0) <= 1e-6),
            "source_start": int(seg["source_start"]), "source_stop": int(seg["source_stop"]),
            "db_frame_start": int(seg["frame_start"]),
            "row_mapping": "row r -> trajectory source_start+60+r, motion source_start+59+r",
        }
        entry["passed"] = entry["normalized_max_abs_diff"] <= 2e-4 and entry["frametime_ok"]
        ok = ok and entry["passed"]
        results.append(entry)
        progress(
            f"[Identity] {seg['style_name']}:{seg['suffix']} rows={n} "
            f"normalized_max={entry['normalized_max_abs_diff']:.3e} passed={entry['passed']}"
        )
    return {"passed": bool(ok), "atol": 2e-4, "rtol": 1e-5, "windows": results}


# --------------------------------------------------------------------------
# construction attempt
# --------------------------------------------------------------------------

def assemble_attempt(clip_a, clip_b, pair: dict, t_frames: int) -> dict:
    a, d = int(pair["a"]), int(pair["d"])
    built = build_transition(clip_a, clip_b, a, d, t_frames)
    rot, pos = built["rotations"], built["positions"]
    nw, g0, d0 = built["nw"], built["g0"], built["d0"]
    planned = plan_contact_masks(clip_a, clip_b, a, d, t_frames)
    ik = lock_feet_ik(rot, pos, clip_a.parents, clip_a.names, planned,
                      offsets=clip_a.offsets, blend_start=blend_start_local())
    rot_final = ik["rotations"]
    g_rot, g_pos = quat.fk(rot_final, pos, clip_a.parents)
    features_all = extract_features_375(rot_final, pos, clip_a.parents, SELECTED_JOINTS, 60, 10)
    n_rows = built["n_rows"]
    features = features_all[:n_rows]
    labels = command_schedule(n_rows)
    starts = valid_window_starts(labels)
    raw_ids = raw_frame_ids_for_rows(a, n_rows)
    a_ref_all = extract_features_375(
        clip_a.rotations[g0:g0 + nw], clip_a.positions[g0:g0 + nw],
        clip_a.parents, SELECTED_JOINTS, 60, 10,
    )
    return {
        "pair": pair, "t_frames": int(t_frames), "built": built, "planned": planned, "ik": ik,
        "rot_final": rot_final, "positions": pos, "global_pos": g_pos, "global_rot": g_rot,
        "features": features, "a_ref_features": a_ref_all[:n_rows], "labels": labels,
        "valid_window_starts": starts, "raw_frame_ids": raw_ids,
        "unreachable_frames": int(ik["unreachable_frames"]), "n_rows": n_rows,
    }


def _suffix_window(t_frames: int, n_rows: int) -> tuple[int, int]:
    """Row range of the pure-D B suffix inside the scored region."""
    del n_rows
    lo = suffix_rows_start_local(t_frames)
    return lo, lo + SUFFIX_ROWS


def evaluate_attempt(
    attempt: dict,
    clip_a,
    clip_b,
    thresholds: dict,
    proxies: dict,
    *,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    device,
    chunk_rows: int,
) -> dict:
    """Raw + codec acceptance for one (pair, duration) attempt."""
    t_frames = attempt["t_frames"]
    nw = attempt["built"]["nw"]
    n_rows = attempt["n_rows"]
    features = attempt["features"]
    # gates cover the scored rows (what enters the dataset); the lead/trail protection
    # frames stay in the work clip but are only used for extraction boundaries
    planned = {name: mask[:n_rows] for name, mask in attempt["planned"].items()}
    ik = attempt["ik"]
    failures: list[dict] = []

    # ---- IK caps (per frame, weighted by the contact ramp) ----
    req_max = 0.0
    for side in ("Right", "Left"):
        for item in ik["intervals"][side]:
            req_max = max(req_max, item["required_correction_max"])
    if attempt["unreachable_frames"] > 0:
        failures.append({"check": "ik_unreachable", "value": attempt["unreachable_frames"], "limit": 0})
    if req_max > IK_MAX_CORRECTION:
        failures.append({"check": "ik_correction", "value": req_max, "limit": IK_MAX_CORRECTION})
    if ik["joint_delta_max_deg"] > IK_MAX_JOINT_DELTA_DEG:
        failures.append({"check": "ik_joint_delta", "value": ik["joint_delta_max_deg"], "limit": IK_MAX_JOINT_DELTA_DEG})

    # ---- basic ----
    if not np.isfinite(features).all():
        failures.append({"check": "nonfinite_features", "value": 1, "limit": 0})
    qerr = quaternion_norm_error(attempt["rot_final"], attempt["global_rot"])
    if qerr > 1e-5:
        failures.append({"check": "quaternion_norm", "value": qerr, "limit": 1e-5})

    # ---- raw bone lengths over the full skeleton ----
    bone_err = bone_length_check(attempt["rot_final"], attempt["positions"], clip_a.parents, clip_a.offsets)
    if bone_err > 1e-4:
        failures.append({"check": "bone_length", "value": bone_err, "limit": 1e-4})

    # ---- root/trajectory identity vs the same A extraction ----
    norm_err = np.abs((features - attempt["a_ref_features"]) / std)
    traj_max = float(norm_err[:, :84].max())
    root_max = float(norm_err[:, 84:87].max())
    if max(traj_max, root_max) > 2e-4:
        failures.append({"check": "root_trajectory_identity", "value": max(traj_max, root_max), "limit": 2e-4})

    # ---- raw path ----
    pose_raw = pose_from_features(features)          # pipeline convention (integrated root)
    pose_world = attach_exact_world(
        pose_raw, attempt["rot_final"][:n_rows], attempt["positions"][:n_rows], clip_a.parents
    )
    series = continuity_series(pose_world)
    raw_fail, raw_metrics = evaluate_continuity(thresholds, "raw", series)
    failures.extend(raw_fail)
    contact_raw = contact_from_features(features, pose=pose_world, proxies=proxies)
    stats_raw = stance_slip_stats(
        {**contact_raw["world"], "speeds": contact_raw["speeds"]}, planned
    )
    depth_raw = ground_proxy_depth(contact_raw["heights"], planned, proxies)
    raw_cfail, raw_cmetrics = evaluate_contact(thresholds, "raw", stats_raw, depth_raw)
    failures.extend(raw_cfail)
    vel_err = velocity_consistency(pose_raw)
    vfail, vlimit = evaluate_velocity_consistency(thresholds, "raw", vel_err)
    failures.extend(vfail)

    sfx_lo, sfx_hi = _suffix_window(t_frames, n_rows)
    b_speed = float(np.mean(root_planar_speed(pose_world.root_pos[sfx_lo:sfx_hi])))
    act_raw = feet_activity(pose_raw.local_pos[sfx_lo:sfx_hi])
    act_fail, act_metrics = evaluate_activity(thresholds, "raw", act_raw, attempt["pair"]["suffix"], b_speed)
    failures.extend(act_fail)

    # ---- B suffix fidelity vs the original donor ----
    d0 = attempt["built"]["d0"]
    donor_lo = d0 + sfx_lo
    constructed_pos = attempt["global_pos"][sfx_lo:sfx_hi]
    donor_pos = clip_b.global_pos[donor_lo:donor_lo + SUFFIX_ROWS]
    constructed_root_rot = attempt["rot_final"][sfx_lo:sfx_hi, 0]
    donor_root_rot = clip_b.rotations[donor_lo:donor_lo + SUFFIX_ROWS, 0]
    constructed_root_pos = attempt["positions"][sfx_lo:sfx_hi, 0]
    donor_root_pos = clip_b.positions[donor_lo:donor_lo + SUFFIX_ROWS, 0]
    proxy_min = float(min(proxies[name] for name in MARKERS))
    del proxy_min  # contact proxies are frozen globally; root-height base uses each clip's own floor
    fidelity = suffix_fidelity(
        constructed_pos, constructed_root_rot, constructed_root_pos, clip_a.floor_ref,
        donor_pos, donor_root_rot, donor_root_pos, clip_b.floor_ref,
    )
    if fidelity["mean_pose_dev_m"] > 0.03:
        failures.append({"check": "b_suffix_mean_pose", "value": fidelity["mean_pose_dev_m"], "limit": 0.03})
    if fidelity["max_pose_dev_m"] > 0.08:
        failures.append({"check": "b_suffix_max_pose", "value": fidelity["max_pose_dev_m"], "limit": 0.08})
    if fidelity["root_tilt_diff_deg"] > 5.0:
        failures.append({"check": "b_suffix_root_tilt", "value": fidelity["root_tilt_diff_deg"], "limit": 5.0})
    if fidelity["root_height_diff_m"] > 0.02:
        failures.append({"check": "b_suffix_root_height", "value": fidelity["root_height_diff_m"], "limit": 0.02})

    # ---- codec path ----
    codec, indices, gpu_seconds = codec_encode_decode(
        model, features, mean, std, device=device, chunk_rows=chunk_rows
    )
    codec_results = {}
    for tag, pose_c in codec_variants(features, codec, pose_raw).items():
        c_series = continuity_series(pose_c)
        c_fail, c_metrics = evaluate_continuity(thresholds, "codec", c_series)
        contact_c = contact_from_features(codec, pose=pose_c, proxies=proxies)
        c_stats = stance_slip_stats({**contact_c["world"], "speeds": contact_c["speeds"]}, planned)
        c_depth = ground_proxy_depth(contact_c["heights"], planned, proxies)
        cc_fail, cc_metrics = evaluate_contact(thresholds, "codec", c_stats, c_depth)
        c_act = feet_activity(pose_c.local_pos[sfx_lo:sfx_hi])
        ca_fail, ca_metrics = evaluate_activity(thresholds, "codec", c_act, attempt["pair"]["suffix"], b_speed)
        c_vel = velocity_consistency(pose_c)
        cv_fail, cv_limit = evaluate_velocity_consistency(thresholds, "codec", c_vel)
        for item in c_fail + cc_fail + ca_fail + cv_fail:
            tagged = dict(item)
            tagged["check"] = f"{tag}_{item['check']}"
            failures.append(tagged)
        codec_results[tag] = {
            "continuity": c_metrics,
            "contact": {"stats": c_stats, "penetration_m": c_depth},
            "activity": ca_metrics,
            "velocity_consistency": {"value": c_vel, "limit": cv_limit},
        }
    raw_results = {
        "continuity": raw_metrics,
        "contact": {"stats": stats_raw, "penetration_m": depth_raw},
        "activity": act_metrics,
        "b_suffix_speed_mps": b_speed,
        "velocity_consistency": {"value": vel_err, "limit": vlimit},
        "bone_length_error_m": bone_err,
        "quaternion_norm_error": qerr,
        "root_trajectory_identity": {"traj_max_norm_err": traj_max, "root_max_norm_err": root_max},
        "ik": {
            "required_correction_max_m": req_max,
            "post_ik_max_drift_m": ik["post_ik_max_drift_m"],
            "joint_delta_max_deg": ik["joint_delta_max_deg"],
            "unreachable_frames": attempt["unreachable_frames"],
        },
    }
    return {
        "failures": failures, "raw": raw_results, "codec": codec_results, "fidelity": fidelity,
        "gpu_seconds": gpu_seconds, "codec_features": codec, "code_indices": indices,
    }


def worst_margin(evaluation: dict, thresholds: dict) -> dict:
    """Worst value/limit ratio across machine gates (higher = closer to failing)."""
    worst = {"ratio": 0.0, "check": None}

    def consider(check: str, value, limit) -> None:
        nonlocal worst
        if value is None or limit in (None, 0):
            return
        ratio = abs(float(value)) / abs(float(limit))
        if ratio > worst["ratio"]:
            worst = {"ratio": float(ratio), "check": check}

    for target in ("raw", "codec"):
        block = evaluation[target]
        for name, entry in block.get("continuity", {}).items():
            for key in ("max", "mean"):
                consider(f"{target}_{name}_{key}", entry.get(f"peak_{key}"), entry.get(f"limit_{key}"))
        stats = (block.get("contact") or {}).get("stats") or {}
        limits = thresholds[target]["contact"]
        consider(f"{target}_contact_slip_p95", stats.get("worst", {}).get("p95"), limits["slip_p95_limit_mps"])
        consider(f"{target}_contact_slip_max", stats.get("worst", {}).get("max"), limits["slip_max_limit_mps"])
        consider(f"{target}_contact_anchor_drift", stats.get("worst", {}).get("drift"), limits["anchor_drift_limit_m"])
        consider(
            f"{target}_ground_penetration",
            (block.get("contact") or {}).get("penetration_m"),
            thresholds[target]["ground_proxy"]["max_penetration_limit_m"],
        )
        vel = block.get("velocity_consistency") or {}
        consider(f"{target}_velocity_consistency", vel.get("value"), vel.get("limit"))
    raw = evaluation["raw"]
    consider("ik_correction", raw["ik"]["required_correction_max_m"], IK_MAX_CORRECTION)
    consider("ik_joint_delta", raw["ik"]["joint_delta_max_deg"], IK_MAX_JOINT_DELTA_DEG)
    consider("bone_length", raw["bone_length_error_m"], 1e-4)
    consider("root_trajectory_identity", max(
        raw["root_trajectory_identity"]["traj_max_norm_err"],
        raw["root_trajectory_identity"]["root_max_norm_err"]), 2e-4)
    fidelity = evaluation["fidelity"]
    consider("b_suffix_mean_pose", fidelity["mean_pose_dev_m"], 0.03)
    consider("b_suffix_max_pose", fidelity["max_pose_dev_m"], 0.08)
    consider("b_suffix_root_tilt", fidelity["root_tilt_diff_deg"], 5.0)
    consider("b_suffix_root_height", fidelity["root_height_diff_m"], 0.02)
    return worst


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

def _npz_bytes(x: np.ndarray) -> int:
    return int(x.nbytes)


def save_sample(state: RunState, attempt: dict, evaluation: dict, *, sample_index: int) -> dict:
    pair = attempt["pair"]
    t_frames = attempt["t_frames"]
    sample_id = f"s{sample_index:03d}_{pair['style_a']}_{pair['style_b']}_{pair['suffix']}_T{t_frames}"
    samples_dir = state.run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    npz_path = samples_dir / f"{sample_id}.npz"
    np.savez_compressed(
        npz_path,
        x=attempt["features"].astype(np.float32),
        command_labels=attempt["labels"].astype(np.int8),
        raw_frame_ids=attempt["raw_frame_ids"],
        code_indices=evaluation["code_indices"],
        transition_rows=np.array([PREFIX_ROWS, PREFIX_ROWS + t_frames], dtype=np.int64),
        valid_window_starts=attempt["valid_window_starts"],
        local_rotations=attempt["rot_final"].astype(np.float32),
        positions=attempt["positions"].astype(np.float32),
        alpha=attempt["built"]["alpha"].astype(np.float32),
    )
    record = state.register_artifact(npz_path)
    nw = attempt["built"]["nw"]
    record.update({
        "sample_id": sample_id,
        "quality_status": "machine_pass",
        "review_status": "pending",
        "train_eligible": False,
        "style_source": pair["style_a"], "style_donor": pair["style_b"], "suffix": pair["suffix"],
        "entry_source": pair["entry_a"], "entry_donor": pair["entry_b"],
        "label_source": pair["label_a"], "label_donor": pair["label_b"],
        "source_bvh": pair["source_path_a"], "donor_bvh": pair["source_path_b"],
        "source_cut": [pair["source_start_a"], pair["source_stop_a"]],
        "donor_cut": [pair["source_start_b"], pair["source_stop_b"]],
        "anchor_source": pair["a"], "anchor_donor": pair["d"],
        "pair_rank": pair.get("pair_rank"),
        "pair_stats": {k: v for k, v in pair.items() if k != "key"},
        "transition_frames": t_frames,
        "transition_duration_s": t_frames / FPS,
        "root_policy": "copy_source_full_root",
        "time_rate": 1,
        "rows": int(attempt["n_rows"]),
        "raw_frame_ids": [int(attempt["raw_frame_ids"][0]), int(attempt["raw_frame_ids"][-1])],
        "allowed_raw_range": [int(attempt["built"]["g0"]), int(attempt["built"]["g0"] + nw - 1)],
        "planned_contact_intervals": {
            name: [[int(a), int(b)] for a, b in mask_intervals(attempt["planned"][name], 1)]
            for name in MARKERS
        },
        "ik_intervals": attempt["ik"]["intervals"],
        "command_schedule": {
            "k_switch_motion_raw": int(attempt["raw_frame_ids"][PREFIX_ROWS] + 1),
            "first_B_row": PREFIX_ROWS,
            "note": "row r predicts motion frame k=raw_frame_ids[r]+1; command A if k<k_switch else B",
        },
        "raw_metrics": evaluation["raw"],
        "codec_metrics": evaluation["codec"],
        "fidelity": evaluation["fidelity"],
        "margin": worst_margin(evaluation, state.thresholds),
        "file": str(npz_path.relative_to(state.run_dir)),
    })
    return record


# --------------------------------------------------------------------------
# videos
# --------------------------------------------------------------------------

REDUCED_NAMES = (
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
)


def _reduced_sel_and_bones(names, parents):
    sel = [names.index(n) for n in REDUCED_NAMES]
    selected_set = set(sel)
    bones = []
    for node in sel:
        walker = int(parents[node])
        while walker != -1 and walker not in selected_set:
            walker = int(parents[walker])
        if walker != -1:
            bones.append((sel.index(walker), sel.index(node)))
    return sel, bones


def render_comparison_video(
    path: Path,
    *,
    title: str,
    panels: list[dict],
    subtitles: list[str],
    switch_row: int | None,
    transition_rows: tuple[int, int] | None,
    fps: int = 30,
    frame_step: int = 2,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    bones = panels[0]["bones"]
    n_frames = min(len(p["pos"]) for p in panels)
    # single common display offset taken from the first panel's frame-0 hip, so the
    # relative root motion between panels is preserved (never per-panel re-centering)
    base = np.asarray(panels[0]["pos"], dtype=np.float64)[0, 0]
    for panel in panels:
        pos = np.asarray(panel["pos"], dtype=np.float64).copy()
        pos[:, :, 0] -= base[0]
        pos[:, :, 2] -= base[2]
        panel["pos"] = pos
    # ONE common camera for every panel (guide section 10): centered on the mean hip
    # position of all panels at the command switch row, fixed for the whole clip. The
    # relative root/path differences between panels stay visible because the display
    # offset is shared; panels may leave this zoomed frame on long walks.
    focus = int(switch_row) if switch_row is not None else 0
    focus = min(max(focus, 0), n_frames - 1)
    centers = np.stack([p["pos"][focus, 0] for p in panels])
    center = centers.mean(axis=0)
    span = 1.15  # meters half-width of the shared view
    fig = plt.figure(figsize=(12, 7.2), dpi=115)
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    fig.suptitle(title, fontsize=9)
    collections = []
    for ax, panel, subtitle in zip(axes, panels, subtitles):
        segs = [[panel["pos"][0, j], panel["pos"][0, k]] for j, k in bones]
        col = Line3DCollection(segs, colors=panel["color"], linewidths=2.4)
        ax.add_collection3d(col)
        collections.append((col, panel))
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(0.0, 1.9)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect((1, 0.85, 1))
        ax.view_init(elev=12, azim=-72)
        ax.set_axis_off()
        ax.set_title(subtitle, fontsize=8)
    # timeline strip: 0..1 maps to rows of the scored region (uniform timeline for all panels)
    bar = fig.add_axes([0.08, 0.075, 0.84, 0.022])
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    bar.set_yticks([])
    bar.set_xticks([])
    for spine in bar.spines.values():
        spine.set_visible(False)
    if transition_rows is not None:
        t0 = transition_rows[0] / n_frames
        t1 = transition_rows[1] / n_frames
        bar.axvspan(0, t0, color="#4c72b0", alpha=0.55)
        bar.axvspan(t0, min(t1, 1.0), color="#dd8452", alpha=0.75)
        bar.axvspan(min(t1, 1.0), 1, color="#55a868", alpha=0.55)
    if switch_row is not None:
        bar.axvline(switch_row / n_frames, color="k", lw=1.4)
    status = fig.text(0.5, 0.02, "", ha="center", fontsize=8)
    writer = FFMpegWriter(fps=fps, bitrate=2600)
    with writer.saving(fig, str(path), dpi=115):
        for frame in range(0, n_frames, frame_step):
            for col, panel in collections:
                segs = [[panel["pos"][frame, j], panel["pos"][frame, k]] for j, k in bones]
                col.set_segments(segs)
            status.set_text(
                f"row {frame}/{n_frames}  t={(frame) / FPS:.2f}s(frames/60)  "
                f"blue=A command, orange=transition, green=B command, black line=command switch row {switch_row}"
            )
            writer.grab_frame()
    plt.close(fig)


def build_video_from_attempt(
    state: RunState, attempt: dict, evaluation: dict, clip_a, clip_b, *, name: str, kind: str
) -> Path:
    from Utils.donor_transition_quality import pose_from_features as _pose

    n_rows = attempt["n_rows"]
    t_frames = attempt["t_frames"]
    g0, d0 = attempt["built"]["g0"], attempt["built"]["d0"]
    sel, bones = _reduced_sel_and_bones(clip_a.names, clip_a.parents)
    # every panel on the scored-row timeline (row r = motion state of raw frame a-2+r)
    panels = [
        {"pos": clip_a.global_pos[g0 + 59:g0 + 59 + n_rows][:, sel], "color": "#7f7f7f", "bones": bones},
        {"pos": attempt["global_pos"][59:59 + n_rows][:, sel], "color": "#d62728", "bones": bones},
        {"pos": clip_b.global_pos[d0 + 59:d0 + 59 + n_rows][:, sel], "color": "#1f77b4", "bones": bones},
        {
            # the codec pose comes from the features and already has exactly the
            # reduced 24 joints in the same order (motion375.SELECTED_24_NAMES)
            "pos": _pose(evaluation["codec_features"]).world_pos,
            "color": "#ff7f0e",
            "bones": bones,
        },
    ]
    videos_dir = state.run_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    path = videos_dir / f"{name}.mp4"
    pair = attempt["pair"]
    reasons = "; ".join(f"{f['check']}={f['value']:.4g}" for f in evaluation["failures"][:3]) or "machine_pass"
    render_comparison_video(
        path,
        title=(
            f"[{kind}] A={pair['style_a']}({pair['suffix']}) -> B={pair['style_b']} T={t_frames}f "
            f"({t_frames / FPS:.2f}s) | panels: A orig | transition (offline IK applied) | B orig | VQ recon "
            f"| one common display offset; no extra smoothing "
            f"| {reasons}"
        ),
        panels=panels,
        subtitles=[
            f"A original ({pair['style_a']} {pair['suffix']})",
            f"constructed transition ({pair['style_a']}->{pair['style_b']}, offline IK applied)",
            f"original B ({pair['style_b']} {pair['suffix']})",
            "VQ codec reconstruction (integrated root)",
        ],
        switch_row=PREFIX_ROWS,
        transition_rows=(PREFIX_ROWS, PREFIX_ROWS + t_frames),
    )
    return path


# --------------------------------------------------------------------------
# report / blocked handling
# --------------------------------------------------------------------------

def blocked_report(state: RunState, reason: str, details: dict) -> None:
    lines = [
        "# Donor transition pilot - blocked before generation",
        "",
        f"- reason: {reason}",
        f"- details: {json.dumps(details, ensure_ascii=False, indent=2, default=_json_default)[:20000]}",
        "",
        "Per docs/DONOR_AGENT_EXECUTION_GUIDE.md section 5.1, more than 20% of real reference",
        "windows failing an applicable hard gate stops generation; this report carries the",
        "metric/reference distribution problem instead of pretending the threshold is reasonable.",
    ]
    (state.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_report(
    state: RunState, manifest: dict, thresholds: dict, cal_details: dict, rejected_attempts: list,
    *, final_status: str, complete: bool,
) -> None:
    run_dir = state.run_dir
    accepted = manifest["samples"]
    lines = [
        "# Donor transition pilot (v1) report",
        "",
        f"- run_id: {manifest['run_id']}  seed={manifest['seed']}",
        f"- status: {final_status}  complete={complete}",
        "- Every sample is review_status=pending / train_eligible=false: machine-passed only,",
        "  human naturalness review is still required before any training use.",
        "",
        "## Funnel",
        "",
        f"- reference windows (calibration): {manifest['calibration']['windows'] if manifest.get('calibration') else 0}",
        f"- geometric pairs selected: {manifest['candidates']['selected'] if manifest.get('candidates') else 0} "
        f"(cap {manifest['budgets']['max_pairs']})",
        f"- duration attempts: {len(state.attempts)} (cap 3 x max_pairs)",
        f"- accepted samples: {len(accepted)} (cap {manifest['budgets']['max_accepted']})",
        f"- videos: {len(manifest['videos'])} (cap {manifest['budgets']['max_videos']})",
        "",
        "### Rejection reasons (attempt level)",
        "",
    ]
    reason_counts: dict[str, int] = {}
    for item in rejected_attempts:
        for failure in item.get("failures", []):
            reason_counts[failure["check"]] = reason_counts.get(failure["check"], 0) + 1
    if reason_counts:
        for key, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none (every attempted pair/duration was accepted)")
    lines += ["", "### Near-miss attempts (fewest failing gates)", ""]
    ranked = sorted(
        [item for item in rejected_attempts if item.get("failures")],
        key=lambda item: (len(item["failures"]), item["pair_id"], item["t_frames"]),
    )[:5]
    if ranked:
        for item in ranked:
            lines.append(
                f"- {item['pair_id']} T={item['t_frames']}: {len(item['failures'])} failing gates "
                f"({', '.join(f['check'] for f in item['failures'][:5])})"
            )
        lines.append(
            "- With zero accepted samples these are the closest constructed candidates; "
            "they are still rejected and are not fabricated into accepted data."
        )
    else:
        lines.append("- none")
    lines += ["", "### Accepted coverage (style pair / suffix)", ""]
    matrix: dict[str, int] = {}
    for record in accepted:
        key = f"{record['style_source']}->{record['style_donor']}:{record['suffix']}"
        matrix[key] = matrix.get(key, 0) + 1
    if matrix:
        for key, count in sorted(matrix.items()):
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines += ["", "### Unavailable directed groups (train side missing or skeleton mismatch)", ""]
    unavailable = manifest.get("unavailable_groups", [])
    if unavailable:
        for group in unavailable:
            lines.append(
                f"- {group['style_a']}->{group['style_b']}:{group['suffix']} "
                f"segments={group.get('segments')} reject={group.get('reject', 'split_or_missing')}"
            )
    else:
        lines.append("- none")
    lines += ["", "### Reference hard-gate findings (guide section 5.1)", ""]
    findings = manifest.get("reference_hard_gate_findings", [])
    if findings:
        for finding in findings:
            lines.append(
                f"- {finding['gate']}: limit={finding['limit_m']} m, reference failure fraction="
                f"{finding['reference_failure_fraction']:.4f} -> {finding['decision']}"
            )
    else:
        lines.append("- none")
    bone = (cal_details.get("bone_length_reference") or {})
    if bone:
        lines.append(
            f"- bone-length reference distribution: windows={bone.get('windows')} "
            f"p50={bone.get('p50_m'):.2e} p90={bone.get('p90_m'):.2e} p99={bone.get('p99_m'):.2e} "
            f"max={bone.get('max_m'):.2e} m"
        )
        offenders = bone.get("per_joint_worst_m") or {}
        lines.append(
            "- worst joints (reference max deviation): "
            + ", ".join(f"{k}={v:.2e}" for k, v in list(offenders.items())[:5])
        )
        lines.append(f"- {bone.get('note', '')}")
    lines += ["", "### Calibration notes", ""]
    lines.append(
        f"- raw contact hard-gate failure fraction on references: "
        f"{cal_details.get('raw_contact_hard_gate_failure_fraction', float('nan')):.4f}"
    )
    lines.append(f"- counts: {cal_details.get('raw_contact_gate_counts')}")
    lines.append(f"- contact mask rule: {thresholds['notes']['contact_mask_rule']}")
    lines.append(f"- thresholds frozen at quality_thresholds.json sha256={manifest['threshold_file_sha256']}")
    lines.append(
        "- caveat: continuity/codec reference windows are 240 rows while accepted samples span "
        "301+T rows; the sample gates use P99 window maxima from the shorter references."
    )
    lines += ["", "## Accepted samples (machine-pass; human review pending)", ""]
    for record in sorted(accepted, key=lambda r: -r["margin"]["ratio"]):
        lines.append(
            f"- {record['sample_id']}: margin={record['margin']['ratio']:.2f} ({record['margin']['check']}) "
            f"rows={record['rows']} T={record['transition_frames']} "
            f"file={record['file']} review={record['review_status']} train_eligible={record['train_eligible']}"
        )
    if not accepted:
        lines.append(
            "- none: every constructed attempt failed at least one machine gate; no sample was fabricated "
            "and no threshold was relaxed."
        )
    lines += ["", "## Videos", ""]
    for item in manifest["videos"]:
        lines.append(f"- {item['path']} ({item['kind']})")
    if not manifest["videos"]:
        lines.append("- none")
    lines += [
        "",
        "## Budget",
        "",
        f"- GPU seconds (frozen VQ encode/quantize/decode): {state.gpu_seconds:.1f}",
        f"- wall seconds: {state.elapsed():.0f}",
        "",
        "## Unresolved / honest caveats",
        "",
    ]
    for note in manifest.get("unresolved", []):
        lines.append(f"- {note}")
    if not manifest.get("unresolved"):
        lines.append("- none recorded")
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from Utils.runtime import load_norm_stats

    return load_norm_stats(path)


def _global_proxies(index: dict, windows: list) -> dict:
    """Marker floor proxies (P01 of slow-frame world heights) over the train reference windows."""
    heights: dict[str, list[np.ndarray]] = {name: [] for name in MARKERS}
    for window in windows:
        feats = read_db_rows(index, window.row_start, 240)
        contact = contact_from_features(feats)
        for name in MARKERS:
            slow = contact["speeds"][name] <= 0.10
            values = contact["heights"][name][slow]
            if values.size:
                heights[name].append(values)
    proxies = {}
    for name in MARKERS:
        values = np.concatenate(heights[name]) if heights[name] else np.array([0.01])
        proxies[name] = float(np.percentile(values, 1))
    return proxies


def _bone_length_reference(index: dict, windows: list, progress=print) -> dict:
    """Literal full-skeleton bone-length gate measured on the real reference windows.

    The gate (|FK bone length - header rest offset| <= 1e-4 m, guide section 7) is a
    fixed raw hard gate.  Guide section 5.1 requires that a >20% real-reference failure
    be delivered as a reference/threshold problem rather than silently relaxed; this
    function measures that distribution and the caller records the finding.
    """
    from tools.generate_database_100sty import prepare_motion

    caches: dict[int, tuple] = {}
    deviations: list[float] = []
    per_joint_worst: dict[str, float] = {}
    for window in windows:
        if window.entry_id not in caches:
            seg = next(s for s in index["segments"] if int(s["source_entry_id"]) == int(window.entry_id))
            raw = bvh.load_zeroeggs(str(seg["source_path"]))
            parents, rotations, positions = prepare_motion(
                raw, start=int(seg["source_start"]), stop=int(seg["source_stop"]), coordinate_convert=False
            )
            caches[window.entry_id] = (
                seg, raw["names"], parents, rotations, positions,
                np.asarray(raw["offsets"], dtype=np.float32) * 0.01,
            )
        seg, names, parents, rotations, positions, offsets = caches[window.entry_id]
        lo = int(window.row_start) - int(seg["frame_start"]) + 59  # motion local frame of the first row
        dev = bone_length_reference_deviation(
            rotations[lo:lo + 240], positions[lo:lo + 240], parents, offsets, names
        )
        deviations.append(dev["max_m"])
        for name, value in dev["per_joint"].items():
            per_joint_worst[name] = max(per_joint_worst.get(name, 0.0), value)
    stats = bone_length_reference_stats(deviations, limit=1e-4)
    stats["per_joint_worst_m"] = dict(sorted(per_joint_worst.items(), key=lambda kv: -kv[1])[:10])
    stats["gate"] = "full-skeleton |FK bone length - header rest offset| <= 1e-4 m (guide section 7)"
    stats["note"] = (
        "measured on the same real train reference windows as the other calibration; "
        "construction copies every per-frame joint translation verbatim, so this deviation "
        "is a source-data property and the construction adds exactly zero to it"
    )
    progress(
        f"[Calibration] bone-length reference: failure_fraction={stats['failure_fraction']:.4f} "
        f"(p50={stats['p50_m']:.2e} p90={stats['p90_m']:.2e} max={stats['max_m']:.2e} m)"
    )
    return stats


def bone_length_reference_deviation(
    rotations: np.ndarray, positions: np.ndarray, parents: np.ndarray, offsets: np.ndarray, names: list,
) -> dict:
    _g_rot, g_pos = quat.fk(rotations, positions, parents)
    per_joint = {}
    worst = 0.0
    for i in range(1, len(parents)):
        rest = float(np.linalg.norm(offsets[i]))
        if rest < 1e-6:
            continue
        actual = float(np.max(np.linalg.norm(g_pos[:, i] - g_pos[:, int(parents[i])], axis=-1)))
        dev = abs(actual - rest)
        per_joint[names[i]] = dev
        worst = max(worst, dev)
    return {"max_m": worst, "per_joint": per_joint}


def _safe_dist_version(name: str) -> str | None:
    try:
        return _dist_version(name)
    except Exception:
        return None


def _which(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None


def _peak_vram_mb(device) -> float | None:
    import torch

    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def default_run_id(seed: int) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"_seed{seed}"


def _dry_run(args: argparse.Namespace, index: dict) -> int:
    print("[DryRun] Donor transition pilot plan")
    print(f"[DryRun] db            : {args.db} exists={args.db.exists()}")
    print(f"[DryRun] vq checkpoint : {args.vq_ckpt} exists={args.vq_ckpt.exists()}")
    print(f"[DryRun] norm stats    : {args.norm_stats} exists={args.norm_stats.exists()}")
    out = args.out or (ROOT / "analysis/donor_transition_pilot_v1" / default_run_id(args.seed))
    print(f"[DryRun] out           : {out} (non-empty -> refuse; nothing is created now)")
    print(f"[DryRun] styles        : {list(PILOT_STYLES)} suffixes={list(PILOT_SUFFIXES)}")
    print(f"[DryRun] layout        : 60 A rows + (T+1) blend rows + 240 B rows; T={list(TRANSITION_FRAMES)}")
    print(f"[DryRun] root policy   : copy_source_full_root; time_rate=1; lead={LEAD_CONTEXT} trail>=62")
    print(f"[DryRun] budgets       : max_pairs={args.max_pairs} max_accepted={args.max_accepted} "
          f"max_videos={args.max_videos} gpu<= {args.max_gpu_seconds}s wall<= {args.max_total_seconds:.0f}s")
    print(f"[DryRun] calibration   : {CALIBRATION_PLAN['total_windows']} train windows (<=128), "
          "incl. non-pilot styles for activity/limits")
    ids = []
    for style, suffix in (("Neutral", "BW"), ("ArmsFolded", "FW"), ("ArmsBehindBack", "SW")):
        seg = find_segment(index, style, suffix)
        ids.append((style, suffix, None if seg is None else (seg["source_entry_id"], seg["split"])))
    print(f"[DryRun] identity clips: {ids}")
    for style in PILOT_STYLES:
        row = []
        for suffix in PILOT_SUFFIXES:
            seg = find_segment(index, style, suffix)
            row.append(f"{suffix}:{(seg or {}).get('split', 'missing')}")
        print(f"[DryRun]   {style:>14s}  " + "  ".join(row))
    print("[DryRun] nothing was written; no GPU was touched")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    index = load_db_index(args.db)
    mean, std = _load_norm_stats(args.norm_stats)
    _norm_stats_cache["mean"], _norm_stats_cache["std"] = mean, std

    if args.dry_run:
        return _dry_run(args, index)

    run_dir = Path(args.out) if args.out else (ROOT / "analysis/donor_transition_pilot_v1" / default_run_id(args.seed))
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"[Error] output directory is not empty; refusing to overwrite: {run_dir}")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir)

    device = resolve_device(args.device)
    configure_runtime(device)
    import torch

    print(f"[Info] run_dir={run_dir} device={device} torch={torch.__version__}")

    manifest: dict = {
        "schema": MANIFEST_SCHEMA,
        "version": VERSION,
        "complete": False,
        "quality_status": "running",
        "review_status": "pending",
        "train_eligible": False,
        "run_id": run_dir.name,
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root_policy": "copy_source_full_root",
        "time_rate": 1,
        "transition_frames_attempted": list(TRANSITION_FRAMES),
        "config": {
            "styles": list(PILOT_STYLES),
            "suffixes": list(PILOT_SUFFIXES),
            "prefix_rows": PREFIX_ROWS,
            "suffix_rows": SUFFIX_ROWS,
            "lead_context": LEAD_CONTEXT,
            "blend": "slerp alpha(u)=6u^5-15u^4+10u^3 at work-local frames 119..119+T",
            "ik": "two-bone lower-limb, root/pelvis fixed, correction<=3cm, joint delta<=10deg",
            "contact_rule": (
                "stable runs height<=proxy+2cm & speed<=0.10m/s >=3 frames, "
                "bridged <=2 failing frames under exit caps h<=proxy+4cm, v<=0.20m/s"
            ),
            "command_schedule": (
                "row r predicts motion frame raw_frame_ids[r]+1 (=a-1+r); A while k<k_switch; "
                "k_switch = a+59 = first blend sample (alpha=0); first B row = row 60"
            ),
            "row_mapping": "row r -> motion state a-2+r, trajectory context centered at a-1+r",
            "calibration_plan": CALIBRATION_PLAN,
        },
        "budgets": {
            "max_pairs": int(args.max_pairs), "max_accepted": int(args.max_accepted),
            "max_videos": int(args.max_videos), "max_gpu_seconds": float(args.max_gpu_seconds),
            "max_total_seconds": float(args.max_total_seconds),
        },
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "matmul_precision": "highest", "tf32": False,
            "dtype": "float32", "chunk_rows": int(args.chunk_rows),
            "vector_quantize_pytorch": _safe_dist_version("vector-quantize-pytorch"),
            "matplotlib": _safe_dist_version("matplotlib"),
            "ffmpeg": "system" if _which("ffmpeg") else "missing",
        },
        "inputs": {
            "db": str(args.db),
            "db_meta_sha256": file_sha256(index["paths"]["meta"]),
            "db_segments_sha256": file_sha256(index["paths"]["segments"]),
            "db_styles_sha256": file_sha256(index["paths"]["styles"]),
            "db_labels_sha256": file_sha256(index["paths"]["labels"]),
            "vq_ckpt": str(args.vq_ckpt), "vq_ckpt_sha256": file_sha256(args.vq_ckpt),
            "norm_stats": str(args.norm_stats), "norm_stats_sha256": file_sha256(args.norm_stats),
        },
        "generator_code": {
            "tools/build_donor_transition_dataset.py": file_sha256(Path(__file__)),
            "Utils/donor_transition.py": file_sha256(ROOT / "Utils/donor_transition.py"),
            "Utils/donor_transition_quality.py": file_sha256(ROOT / "Utils/donor_transition_quality.py"),
            "tests/test_donor_transition.py": file_sha256(ROOT / "tests/test_donor_transition.py")
            if (ROOT / "tests/test_donor_transition.py").exists() else None,
        },
        "identity": None, "calibration": None, "threshold_file": None, "threshold_file_sha256": None,
        "proxies": None, "candidates": None, "unavailable_groups": [], "pairs_materialized": [],
        "samples": [], "rejections": [], "videos": [], "artifacts": [], "budget_used": {}, "unresolved": [],
    }
    sha256_json_dump(manifest, run_dir / "manifest.json")

    try:
        # ---------------- phase 0: identity ----------------
        identity_ids = []
        for style, suffix in (("Neutral", "BW"), ("ArmsFolded", "FW"), ("ArmsBehindBack", "SW")):
            seg = find_segment(index, style, suffix)
            if seg is not None:
                identity_ids.append(int(seg["source_entry_id"]))
        identity = identity_check(index, identity_ids)
        sha256_json_dump(identity, run_dir / "identity.json")
        state.register_artifact(run_dir / "identity.json")
        manifest["identity"] = {"passed": identity["passed"], "windows": len(identity["windows"])}
        if not identity["passed"]:
            blocked_report(state, "identity check failed (guide section 4)", identity)
            manifest["quality_status"] = "identity_failed"
            manifest["complete"] = True
            manifest["budget_used"] = {"gpu_seconds_vq": state.gpu_seconds, "wall_seconds": state.elapsed()}
            sha256_json_dump(manifest, run_dir / "manifest.json")
            print("[Error] identity check failed; stopping before generation")
            return 3

        # ---------------- phase 1: calibration ----------------
        plan = []
        for item in CALIBRATION_PLAN["targets"]:
            for suffix in item["suffixes"]:
                seg = find_segment(index, item["style"], suffix)
                if seg is not None and seg["split"] == "train":
                    plan.append({"entry_id": int(seg["source_entry_id"]), "windows": item["windows"]})
        for item in CALIBRATION_PLAN["others"]:
            seg = find_segment(index, item["style"], item["suffix"])
            if seg is not None and seg["split"] == "train":
                plan.append({"entry_id": int(seg["source_entry_id"]), "windows": item["windows"]})
        windows = sample_reference_windows(index, plan, seed=args.seed)
        print(f"[Calibration] reference windows: {len(windows)}")
        proxies = _global_proxies(index, windows)
        print(f"[Calibration] floor proxies (m): { {k: round(v, 4) for k, v in proxies.items()} }")
        from Utils.model_loading import build_vqvae_from_checkpoint

        model = build_vqvae_from_checkpoint(args.vq_ckpt, freeze=True).to(device)
        model.eval()
        thresholds, cal_details = calibrate_thresholds(
            index, windows, proxies=proxies, model=model, mean=mean, std=std,
            device=device, chunk_rows=args.chunk_rows,
        )
        state.gpu_seconds += float(cal_details.get("gpu_seconds", 0.0))

        slip = np.asarray(cal_details["raw_slip_reference"]["worst_p95_values"], dtype=float)
        slip_max = np.asarray(cal_details["raw_slip_reference"]["worst_max_values"], dtype=float)
        drift = np.asarray(cal_details["raw_slip_reference"]["worst_drift_values"], dtype=float)
        fails = (slip > 0.10) | (slip_max > 0.20) | (drift > 0.02)
        frac = float(np.mean(fails)) if fails.size else 0.0
        cal_details["raw_contact_hard_gate_failure_fraction"] = frac
        cal_details["raw_contact_gate_counts"] = {
            "windows": int(fails.size),
            "slip_p95_fail": int(np.sum(slip > 0.10)),
            "slip_max_fail": int(np.sum(slip_max > 0.20)),
            "drift_fail": int(np.sum(drift > 0.02)),
        }
        threshold_hash = sha256_json_dump(thresholds, run_dir / "quality_thresholds.json")
        state.thresholds = thresholds
        state.register_artifact(run_dir / "quality_thresholds.json")
        sha256_json_dump(cal_details, run_dir / "calibration.json")
        state.register_artifact(run_dir / "calibration.json")
        manifest["threshold_file"] = "quality_thresholds.json"
        manifest["threshold_file_sha256"] = threshold_hash
        manifest["proxies"] = proxies
        manifest["calibration"] = {
            "windows": len(windows), "gpu_seconds": cal_details.get("gpu_seconds", 0.0),
            "raw_contact_hard_gate_failure_fraction": frac,
        }
        print(
            f"[Calibration] raw contact hard-gate failure fraction: {frac:.4f} "
            f"(slip_p95>{0.10}: {cal_details['raw_contact_gate_counts']['slip_p95_fail']}, "
            f"slip_max>{0.20}: {cal_details['raw_contact_gate_counts']['slip_max_fail']}, "
            f"drift>{0.02}: {cal_details['raw_contact_gate_counts']['drift_fail']})"
        )
        if frac > ABORT_REFERENCE_FAILURE_FRACTION:
            blocked_report(state, "raw contact hard gates reject too many real references", cal_details)
            manifest["quality_status"] = "calibration_blocked"
            manifest["complete"] = True
            manifest["budget_used"] = {"gpu_seconds_vq": state.gpu_seconds, "wall_seconds": state.elapsed()}
            sha256_json_dump(manifest, run_dir / "manifest.json")
            print("[Error] calibration guard tripped; stopping before generation (guide section 5.1)")
            return 4

        # guide section 5.1: literal full-skeleton bone-length hard gate vs real references
        bone_stats = _bone_length_reference(index, windows)
        cal_details["bone_length_reference"] = bone_stats
        manifest["reference_hard_gate_findings"] = []
        if bone_stats["abort_required"]:
            finding = {
                "gate": "raw_bone_length_full_skeleton",
                "limit_m": bone_stats["limit_m"],
                "reference_failure_fraction": bone_stats["failure_fraction"],
                "guide_rule": (
                    "section 5.1: >20% real-reference failure means the rule/applicability is a "
                    "problem to deliver, not to relax"
                ),
                "decision": (
                    "gate kept unchanged and enforced on every sample; the finding is delivered "
                    "with the reference distribution, p50/p90/p99/max and per-joint offenders "
                    "instead of lowering or skipping the gate"
                ),
            }
            manifest["reference_hard_gate_findings"].append(finding)
            cal_details["bone_length_finding"] = finding
            print(
                "[Warn] guide 5.1 finding: literal bone-length gate fails "
                f"{bone_stats['failure_fraction']:.1%} of real references; gate kept unchanged"
            )
        # rewrite calibration.json with the bone stats included
        sha256_json_dump(cal_details, run_dir / "calibration.json")
        state.artifacts = [rec for rec in state.artifacts if rec["path"] != "calibration.json"]
        state.register_artifact(run_dir / "calibration.json")

        # ---------------- phase 1b: candidates ----------------
        clips = load_target_clips(index, proxies)
        print(f"[Candidates] target clips loaded: {len(clips)}")
        span = PREFIX_ROWS + SUFFIX_ROWS + TRANSITION_FRAMES[-1]
        pairs, pair_stats = select_pairs(
            index, clips, span=span, t_max=TRANSITION_FRAMES[-1], max_pairs=args.max_pairs,
        )
        unavailable = pair_stats.pop("unavailable_groups", [])
        candidates_payload = {
            "seed": args.seed, "span_frames": span,
            "pair_stats": pair_stats, "unavailable_groups": unavailable,
            "pairs": [{k: v for k, v in item.items() if k != "key"} for item in pairs],
        }
        candidates_hash = sha256_json_dump(candidates_payload, run_dir / "candidates.json")
        state.register_artifact(run_dir / "candidates.json")
        manifest["candidates"] = {"file": "candidates.json", "sha256": candidates_hash, "selected": len(pairs)}
        manifest["unavailable_groups"] = unavailable
        manifest["pairs_materialized"] = [item["pair_id"] for item in pairs]
        print(f"[Candidates] selected pairs: {len(pairs)}")

        # ---------------- phase 2: construction attempts ----------------
        accepted_attempts, rejected_attempts = [], []
        for pair in pairs:
            if len(accepted_attempts) >= args.max_accepted:
                rejected_attempts.append({"pair_id": pair["pair_id"], "t_frames": None,
                                          "failures": [{"check": "skipped_quota_reached", "value": 0, "limit": 0}]})
                continue
            if state.elapsed() > args.max_total_seconds or state.gpu_seconds > args.max_gpu_seconds:
                manifest["unresolved"].append(
                    f"budget exceeded after {len(state.attempts)} attempts; remaining pairs not attempted"
                )
                break
            clip_a, clip_b = clips[pair["entry_a"]], clips[pair["entry_b"]]
            accepted_here = False
            for t_frames in TRANSITION_FRAMES:
                if state.elapsed() > args.max_total_seconds or state.gpu_seconds > args.max_gpu_seconds:
                    manifest["unresolved"].append(f"budget exceeded mid-pair {pair['pair_id']}")
                    break
                attempt = assemble_attempt(clip_a, clip_b, pair, t_frames)
                evaluation = evaluate_attempt(
                    attempt, clip_a, clip_b, thresholds, proxies,
                    model=model, mean=mean, std=std, device=device, chunk_rows=args.chunk_rows,
                )
                state.gpu_seconds += evaluation["gpu_seconds"]
                record = {
                    "pair_id": pair["pair_id"], "t_frames": int(t_frames),
                    "ok": not evaluation["failures"], "failures": evaluation["failures"][:10],
                }
                state.attempts.append(record)
                if not evaluation["failures"]:
                    sample = save_sample(state, attempt, evaluation, sample_index=len(accepted_attempts) + 1)
                    accepted_attempts.append((attempt, evaluation, sample))
                    print(f"[Accept] {sample['sample_id']} margin={sample['margin']['ratio']:.2f} "
                          f"({sample['margin']['check']})")
                    accepted_here = True
                    break
                rejected_attempts.append({**record, "attempt": attempt, "evaluation": evaluation})
                print(f"[Reject] {pair['pair_id']} T={t_frames}: "
                      + ", ".join(f"{f['check']}" for f in evaluation["failures"][:4]))
            if not accepted_here:
                pass
        manifest["samples"] = [rec for _, _, rec in accepted_attempts]
        manifest["rejections"] = [
            {k: v for k, v in item.items() if k not in ("attempt", "evaluation")} for item in rejected_attempts
        ]
        rejections_payload = {
            "run_id": manifest["run_id"],
            "attempts": [
                {k: v for k, v in item.items() if k not in ("attempt", "evaluation")} for item in rejected_attempts
            ],
            "reason_counts": {
                check: sum(1 for item in rejected_attempts for f in item.get("failures", []) if f["check"] == check)
                for check in sorted({f["check"] for item in rejected_attempts for f in item.get("failures", [])})
            },
        }
        sha256_json_dump(rejections_payload, run_dir / "rejections.json")
        state.register_artifact(run_dir / "rejections.json")

        # ---------------- videos ----------------
        video_records = []
        if args.max_videos > 0:
            n_pass = min(6, args.max_videos, len(accepted_attempts))
            accepted_sorted = sorted(accepted_attempts, key=lambda item: item[2]["margin"]["ratio"])
            for attempt, evaluation, sample in accepted_sorted[:n_pass]:
                path = build_video_from_attempt(
                    state, attempt, evaluation,
                    clips[attempt["pair"]["entry_a"]], clips[attempt["pair"]["entry_b"]],
                    name=f"pass_{sample['sample_id']}", kind="machine-pass",
                )
                record = state.register_artifact(path)
                record.update({"kind": "machine_pass", "sample_id": sample["sample_id"],
                               "file": record["path"]})
                video_records.append(record)
            remaining = max(0, args.max_videos - len(video_records))
            seen_reasons = set()
            for item in rejected_attempts:
                if remaining <= 0 or "attempt" not in item:
                    continue
                reasons = tuple(f["check"] for f in item.get("failures", [])[:2])
                if reasons in seen_reasons:
                    continue
                seen_reasons.add(reasons)
                path = build_video_from_attempt(
                    state, item["attempt"], item["evaluation"],
                    clips[item["attempt"]["pair"]["entry_a"]], clips[item["attempt"]["pair"]["entry_b"]],
                    name=f"reject_{item['pair_id']}_T{item['t_frames']}", kind="rejected",
                )
                record = state.register_artifact(path)
                record.update({"kind": "rejected", "pair_id": item["pair_id"],
                               "t_frames": item["t_frames"], "file": record["path"]})
                video_records.append(record)
                remaining -= 1
        manifest["videos"] = video_records

        # ---------------- report + completion ----------------
        manifest["budget_used"] = {
            "gpu_seconds_vq": state.gpu_seconds, "wall_seconds": state.elapsed(),
            "peak_vram_mb": _peak_vram_mb(device),
        }
        if not accepted_attempts:
            manifest["unresolved"].append(
                "zero accepted samples: every constructed (pair, duration) attempt failed at least "
                "one frozen machine gate; no threshold was relaxed and no sample was fabricated"
            )
            reason_counts = {}
            for item in rejected_attempts:
                for f in item.get("failures", []):
                    reason_counts[f["check"]] = reason_counts.get(f["check"], 0) + 1
            manifest["unresolved"].append(
                "dominant rejection drivers: "
                + ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:6])
            )
        for finding in manifest.get("reference_hard_gate_findings", []):
            manifest["unresolved"].append(
                f"reference hard-gate finding recorded: {finding['gate']} "
                f"failure_fraction={finding['reference_failure_fraction']:.3f} (gate unchanged)"
            )
        final_status = "complete" if accepted_attempts else "complete_zero_accepted"
        write_report(
            state, manifest, thresholds, cal_details, rejected_attempts,
            final_status=final_status, complete=True,
        )
        report_path = run_dir / "report.md"
        manifest["artifacts"] = state.artifacts + [{"path": "report.md", "sha256": file_sha256(report_path)}]
        sha256_json_dump(manifest, run_dir / "manifest.json")
        for record in manifest["artifacts"]:
            path = run_dir / record["path"]
            if not path.exists() or file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"artifact verification failed: {record['path']}")
        manifest["complete"] = True
        manifest["quality_status"] = final_status
        sha256_json_dump(manifest, run_dir / "manifest.json")
        print(f"[Done] samples={len(accepted_attempts)} rejects={len(rejected_attempts)} "
              f"videos={len(video_records)}")
        print(f"[Done] gpu_seconds={state.gpu_seconds:.1f} wall_seconds={state.elapsed():.0f}")
        return 0
    except Exception as exc:
        manifest["quality_status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["budget_used"] = {"gpu_seconds_vq": state.gpu_seconds, "wall_seconds": state.elapsed()}
        sha256_json_dump(manifest, run_dir / "manifest.json")
        raise


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
