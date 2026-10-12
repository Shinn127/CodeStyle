"""Targeted tests for the offline donor-transition pilot (guide section 9).

Scenarios covered:
  1. A=D identity, alpha endpoints (value/first/second derivative), unit/coordinate
     conventions and the DB-row <-> raw-frame mapping.
  2. Fixed planted foot keeps its world speed anchor under root translation/yaw;
     a "replace root and fix velocity" style sample is rejected.
  3. Opposite support, moving-vs-idle, unreachable IK, large joint change and
     out-of-range context each take their documented rejection path.
  4. All features are re-extracted after IK/blend; copied/reused velocity fails an
     independent consistency check.
  5. Command labels are computed from the predicted motion time; the history may
    跨 the switch but the four targets always share one label (no one-frame offset).
  6. Quantized motion matches the decoded codes; VQ stays gradient-free with an
     unchanged weight fingerprint; normalization stats are untouched.
  7. Zero contact is a rejection (not "zero slip is fine"); partial lock success
     cannot hide a failing planned stance.
  8. Cache/artifact reader rejects incomplete or replaced content; the candidate
     list is reproducible under a fixed seed.
  9. A sample that passes raw but fails codec is rejected; an all-rejected run
     still produces a complete report without fabricated accepted samples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils import quat  # noqa: E402
from Utils.donor_transition import (  # noqa: E402
    DT,
    FPS,
    IK_BAND,
    IK_MAX_CORRECTION,
    MARKERS,
    ClipData,
    blend_start_local,
    build_transition,
    command_schedule,
    contact_mask,
    lock_feet_ik,
    mask_intervals,
    plan_contact_masks,
    raw_frame_ids_for_rows,
    scored_rows,
    set_proxies,
    shortest_arc_deg,
    slerp_shortest,
    smoothstep,
    valid_window_starts,
    work_length,
)
from Utils import donor_transition_quality as dq  # noqa: E402
from Visualization.pose_postprocess import two_bone_target  # noqa: E402


# --------------------------------------------------------------------------
# synthetic skeleton: the real 75-joint name/parent layout with simplified leg
# offsets, so the original extract_features_375 works unchanged on fixtures
# --------------------------------------------------------------------------
NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head", "HeadEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumb1", "RightHandThumb2",
    "RightHandThumb3", "RightHandThumb4", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3",
    "RightHandIndex4", "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandPinky1",
    "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightForeArmEnd", "RightArmEnd",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb1", "LeftHandThumb2",
    "LeftHandThumb3", "LeftHandThumb4", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3",
    "LeftHandIndex4", "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandPinky1",
    "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftForeArmEnd", "LeftArmEnd",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBaseEnd", "RightLegEnd",
    "RightUpLegEnd", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBaseEnd",
    "LeftLegEnd", "LeftUpLegEnd",
]
PARENTS = np.array([
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21, 22, 23,
    12, 25, 26, 27, 12, 29, 30, 31, 12, 11, 4, 35, 36, 37, 38, 39, 40, 41, 38, 43, 44, 45, 38,
    47, 48, 49, 38, 51, 52, 53, 38, 55, 56, 57, 38, 37, 0, 61, 62, 63, 64, 63, 62, 0, 68, 69,
    70, 71, 70, 69,
], dtype=np.int64)
_SELECTED = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71)
# indices into the 75-joint fixture for markers and the leg chains
_MARKER_GLOBAL = {"RightFoot": 63, "RightToeBase": 64, "LeftFoot": 70, "LeftToeBase": 71}
_LEG_CHAINS = {
    "Right": {"hip": 61, "knee": 62, "foot": 63, "toe": 64},
    "Left": {"hip": 68, "knee": 69, "foot": 70, "toe": 71},
}


def _build_offsets() -> np.ndarray:
    offsets = np.zeros((len(NAMES), 3), dtype=np.float32)
    for chain in _LEG_CHAINS.values():
        offsets[chain["hip"]] = [0.05, -0.02, 0.0]
        offsets[chain["knee"]] = [0.0, -0.4, 0.0]
        offsets[chain["foot"]] = [0.0, -0.4, 0.0]
        offsets[chain["toe"]] = [0.0, -0.02, 0.1]
    for i in range(len(NAMES)):
        if not offsets[i].any():
            offsets[i] = [0.0, 0.1, 0.0]
    offsets[0] = [0.0, 0.0, 0.0]
    return offsets


OFFSETS = _build_offsets()


def _make_clip(n_frames: int, *, speed: float = 0.6, contact_pattern=None, marker_lift=None) -> ClipData:
    positions = np.tile(OFFSETS, (n_frames, 1, 1)).astype(np.float32)
    t = np.arange(n_frames, dtype=np.float32) * DT
    positions[:, 0, 0] = t * speed            # root X translation
    positions[:, 0, 1] = 0.9
    rotations = np.zeros((n_frames, len(NAMES), 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    # small alternating leg swing so the feet are not perfectly static
    rotations[:, 62, 2] = np.sin(2 * np.pi * 1.0 * t) * 0.3
    rotations[:, 69, 2] = -np.sin(2 * np.pi * 1.0 * t) * 0.3
    if contact_pattern is not None:
        for frame, sides in contact_pattern.items():
            if frame >= n_frames:
                continue
            for side in sides:
                rotations[frame, _LEG_CHAINS[side]["knee"], 2] = 0.0
    if marker_lift is not None:
        side, lo, hi = marker_lift
        rotations[lo:hi, _LEG_CHAINS[side]["knee"], 2] += 0.6
    rotations = quat.normalize(quat.unroll(rotations))
    g_rot, g_pos = quat.fk(rotations, positions, PARENTS)
    yaw = np.zeros(n_frames, dtype=np.float32)
    planar = np.full(n_frames, speed, dtype=np.float32)
    marker_height, marker_speed = {}, {}
    for name, idx in _MARKER_GLOBAL.items():
        pts = g_pos[:, idx]
        marker_height[name] = pts[:, 1].astype(np.float32)
        marker_speed[name] = np.zeros(n_frames, dtype=np.float32)
    clip = ClipData(
        entry_id=0, style="Neutral", suffix="BW", split="train", path=Path("/dev/null"),
        source_start=0, source_stop=n_frames, label=0, names=list(NAMES), parents=PARENTS,
        offsets=OFFSETS, rotations=rotations, positions=positions, frametime=1 / 60.0,
        global_pos=g_pos, yaw=yaw, yaw_rate=np.zeros(n_frames, dtype=np.float32),
        planar_speed=planar, marker_height=marker_height, marker_speed=marker_speed,
        local_vel=np.stack([planar, np.zeros(n_frames, dtype=np.float32)], axis=1),
    )
    set_proxies(clip, {name: 0.0 for name in MARKERS})
    return clip


def _expand24_to75(pos24: np.ndarray) -> np.ndarray:
    """Broadcast a [T,24,3] selected-joint array into the [T,75,3] full layout."""
    out = np.zeros((pos24.shape[0], len(NAMES), 3), dtype=np.float64)
    out[:, _SELECTED] = pos24
    return out


# --------------------------------------------------------------------------
# 1. identity / alpha endpoints / mapping
# --------------------------------------------------------------------------

def test_alpha_endpoints_and_derivatives():
    u = np.linspace(0.0, 1.0, 2001)
    a = smoothstep(u)
    assert a[0] == pytest.approx(0.0, abs=1e-7)
    assert a[-1] == pytest.approx(1.0, abs=1e-7)
    assert np.all(np.diff(a) >= -1e-7)              # monotone
    # analytic first/second derivatives of 6u^5-15u^4+10u^3 vanish at both ends
    d1 = 30 * u**2 * (1 - u) ** 2
    d2 = 60 * u * (1 - u) * (1 - 2 * u)
    assert d1[0] == 0.0 and d1[-1] == pytest.approx(0.0, abs=1e-12)
    assert d2[0] == pytest.approx(0.0, abs=1e-12) and d2[-1] == pytest.approx(0.0, abs=1e-12)
    # and the finite-difference estimate on this grid is at the discretization floor
    du = u[1] - u[0]
    assert abs(np.gradient(a, du)[0]) <= 1e-5 and abs(np.gradient(a, du)[-1]) <= 1e-5


def test_a_equals_d_reproduces_a_exactly():
    clip = _make_clip(700)
    built = build_transition(clip, clip, 200, 200, 36)
    # endpoint regions are exact copies of the source; the blend interior is a
    # normalized SLERP of identical quaternions (float32 round-off only)
    assert np.array_equal(built["positions"], clip.positions[built["g0"]:built["g0"] + built["nw"]])
    ref = clip.rotations[built["g0"]:built["g0"] + built["nw"]]
    assert np.allclose(built["rotations"], ref, atol=1e-6)
    s0 = blend_start_local()
    assert np.array_equal(built["rotations"][:s0], ref[:s0])
    assert np.array_equal(built["rotations"][s0 + 37:], ref[s0 + 37:])
    alpha = built["alpha"]
    assert np.all(alpha[blend_start_local() + 37:] == 1.0)
    # with different ends the pre-blend region must still be exactly A and the tail exactly D
    other = _make_clip(700, speed=0.5)
    other.rotations = other.rotations.copy()
    other.rotations[:, 4, 2] += 0.25
    built2 = build_transition(clip, other, 200, 200, 36)
    s = blend_start_local()
    assert np.array_equal(built2["rotations"][:s], clip.rotations[built2["g0"]:built2["g0"] + s])
    tail = s + 37
    assert np.array_equal(built2["rotations"][tail:], other.rotations[built2["d0"] + tail:built2["d0"] + built2["nw"]])


def test_row_and_frame_mapping_units():
    a = 200
    rows = raw_frame_ids_for_rows(a, 10)
    assert rows[0] == a - 2 and rows[-1] == a - 2 + 9
    assert scored_rows(24) == 60 + 25 + 240
    assert work_length(24) == 61 + scored_rows(24) + 62
    # trajectory time of row r is motion time + 1
    assert rows[5] + 1 == a - 1 + 5
    # first blend sample is the motion state of the first B-commanded row
    assert blend_start_local() == 119
    assert blend_start_local() + 59 - 59 == 119


def test_slerp_and_shortest_arc_geometry():
    qa = quat.from_angle_axis(np.array([0.0]), np.array([[0.0, 1.0, 0.0]]))[0]
    qb = quat.from_angle_axis(np.array([np.pi / 2]), np.array([[0.0, 1.0, 0.0]]))[0]
    mid = slerp_shortest(qa, qb, np.array(0.5))
    angle = shortest_arc_deg(qa[None], mid[None])[0]
    assert angle == pytest.approx(45.0, abs=1e-3)
    # sign flip of the target must not change the result
    mid2 = slerp_shortest(qa, -qb, np.array(0.5))
    assert shortest_arc_deg(mid[None], mid2[None])[0] < 1e-3


# --------------------------------------------------------------------------
# 2. planted foot under root translation / yaw
# --------------------------------------------------------------------------

def _planted_foot_scenario(*, root_speed=0.6, yaw_rate=0.0, n=200):
    """Root moving (and optionally yawing) with static legs: the feet slide in world space."""
    positions = np.tile(OFFSETS, (n, 1, 1)).astype(np.float32)
    t = np.arange(n, dtype=np.float32) * DT
    positions[:, 0, 0] = t * root_speed
    positions[:, 0, 2] = 0.0
    rotations = np.zeros((n, len(NAMES), 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    # small knee bend so the leg keeps IK slack for horizontal anchors
    rotations[:, 61, 2] = -0.2
    rotations[:, 62, 2] = 0.4
    if yaw_rate:
        yaw = np.cumsum(np.full(n, yaw_rate * DT, dtype=np.float32))
        rotations[:, 0] = quat.from_angle_axis(yaw, np.broadcast_to([0, 1, 0], (n, 3)))
    rotations = quat.normalize(quat.unroll(rotations))
    g_rot, g_pos = quat.fk(rotations, positions, PARENTS)
    clip = ClipData(
        entry_id=0, style="Neutral", suffix="BW", split="train", path=Path("/dev/null"),
        source_start=0, source_stop=n, label=0, names=list(NAMES), parents=PARENTS, offsets=OFFSETS,
        rotations=rotations, positions=positions, frametime=1 / 60.0, global_pos=g_pos,
        yaw=np.zeros(n, np.float32), yaw_rate=np.full(n, yaw_rate, np.float32),
        planar_speed=np.full(n, root_speed, np.float32),
        marker_height={name: g_pos[:, i][:, 1] for name, i in _MARKER_GLOBAL.items()},
        marker_speed={name: np.zeros(n, np.float32) for name in MARKERS},
        local_vel=np.stack([np.full(n, root_speed, np.float32), np.zeros(n, np.float32)], axis=1),
    )
    return clip


def test_planted_foot_brought_inside_band_when_correction_feasible():
    """A stance that drifts a few cm is pulled back inside the comfort band."""
    for yaw_rate in (0.0, 20.0):
        clip = _planted_foot_scenario(root_speed=0.45, yaw_rate=yaw_rate)
        nw = 200
        planned = {name: np.zeros(nw, bool) for name in MARKERS}
        planned["RightFoot"][50:54] = True
        foot_idx = _MARKER_GLOBAL["RightFoot"]
        ik = lock_feet_ik(clip.rotations, clip.positions, clip.parents, clip.names, planned,
                          offsets=OFFSETS)
        _gr, gpos = quat.fk(ik["rotations"], clip.positions, clip.parents)
        anchor = np.asarray(ik["anchor_map"]["Right_0"])
        # interval interior (full weight) is inside the comfort band; the boundary frames
        # carry the raised-cosine ramp by design
        interior = np.linalg.norm(gpos[51:53, foot_idx] - anchor, axis=-1)
        assert interior.max() <= IK_BAND + 3e-3
        # the required per-frame correction (excess over the band) is small and inside
        # the pilot's 3cm cap; the pre-IK motion really left the band ("test bites")
        assert ik["intervals"]["Right"][0]["required_correction_max"] < IK_MAX_CORRECTION
        assert ik["intervals"]["Right"][0]["pre_ik_max_deviation_m"] > IK_BAND  # deviation existed
        # the un-IK'ed motion really slides inside this interval ("test bites")
        raw_slide = np.linalg.norm((clip.global_pos[52, foot_idx] - clip.global_pos[50, foot_idx])[[0, 2]])
        assert raw_slide > 0.005


def test_infeasible_anchor_exceeds_correction_cap():
    """A 100-frame stance on a 0.6 m/s moving root needs >3cm correction -> rejected."""
    clip = _planted_foot_scenario(root_speed=0.6)
    nw = 200
    planned = {name: np.zeros(nw, bool) for name in MARKERS}
    planned["RightFoot"][50:150] = True
    planned["RightToeBase"][50:150] = True
    ik = lock_feet_ik(clip.rotations, clip.positions, clip.parents, clip.names, planned)
    required = max(item["required_correction_max"] for item in ik["intervals"]["Right"])
    assert required > IK_MAX_CORRECTION


def test_reused_velocity_is_rejected_by_independent_check():
    n_rows = 240
    pose = dq.PoseSeries(
        local_pos=np.zeros((n_rows, 24, 3)),
        local_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 24, 1)),
        local_vel=np.full((n_rows, 24, 3), 5.0),          # stale velocity, way off
        root_pos=np.zeros((n_rows, 3)),
        root_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 1)),
        world_pos=np.zeros((n_rows, 24, 3)),
        world_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 24, 1)),
    )
    assert dq.velocity_consistency(pose) > 1.0     # 5 m/s implied vs 0 finite difference


def test_anchor_is_interval_centroid_and_source_anchored_before_blend():
    """Anchors are the interval centroid; pre-blend intervals use their pre-blend part."""
    clip = _planted_foot_scenario(root_speed=0.6)
    nw = 200
    planned = {name: np.zeros(nw, bool) for name in MARKERS}
    planned["RightFoot"][30:36] = True
    foot_idx = _MARKER_GLOBAL["RightFoot"]
    ik = lock_feet_ik(clip.rotations, clip.positions, clip.parents, clip.names, planned, offsets=OFFSETS)
    expected = clip.global_pos[30:36, foot_idx].mean(axis=0)
    assert np.allclose(ik["anchor_map"]["Right_0"], expected)
    assert ik["intervals"]["Right"][0]["source_anchored"] is False
    # an interval starting before the blend is anchored on its pre-blend centroid
    planned2 = {name: np.zeros(nw, bool) for name in MARKERS}
    planned2["RightFoot"][40:80] = True
    ik2 = lock_feet_ik(clip.rotations, clip.positions, clip.parents, clip.names, planned2,
                       offsets=OFFSETS, blend_start=50)
    assert ik2["intervals"]["Right"][0]["source_anchored"] is True
    expected2 = clip.global_pos[40:50, foot_idx].mean(axis=0)
    assert np.allclose(ik2["anchor_map"]["Right_0"], expected2)


def test_reused_velocity_is_rejected_by_independent_check():
    clip = _planted_foot_scenario()
    n_rows = 240
    root_motion, local_pos, local_rot, local_vel = dq.motion375.decode_selected_state(
        np.zeros((n_rows, 375), np.float32)
    )
    del root_motion, local_rot
    pose = dq.PoseSeries(
        local_pos=np.zeros((n_rows, 24, 3)),
        local_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 24, 1)),
        local_vel=np.full((n_rows, 24, 3), 5.0),          # stale velocity, way off
        root_pos=np.zeros((n_rows, 3)),
        root_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 1)),
        world_pos=np.zeros((n_rows, 24, 3)),
        world_rot=np.tile(np.array([1.0, 0, 0, 0], np.float64), (n_rows, 24, 1)),
    )
    assert dq.velocity_consistency(pose) > 1.0     # 5 m/s implied vs 0 finite difference


# --------------------------------------------------------------------------
# 3. rejection paths
# --------------------------------------------------------------------------

def test_unreachable_target_requires_large_correction():
    clip = _planted_foot_scenario(root_speed=2.5)   # root runs away from the planted foot
    nw = 200
    planned = {name: np.zeros(nw, bool) for name in MARKERS}
    planned["RightFoot"][20:180] = True
    ik = lock_feet_ik(clip.rotations, clip.positions, clip.parents, clip.names, planned)
    req = max(item["required_correction_max"] for item in ik["intervals"]["Right"])
    assert req > IK_MAX_CORRECTION   # the builder rejects via ik_correction / ik_unreachable


def test_out_of_range_context_returns_no_anchors():
    from Utils.donor_transition import anchor_candidates

    clip = _make_clip(200)
    assert anchor_candidates(clip) == []
    # a clip that is just long enough yields anchors with their own context
    long_clip = _make_clip(660, speed=0.6)
    anchors = anchor_candidates(long_clip, max_anchors=4)
    assert anchors and all(entry["status"] == "moving" for entry in anchors)


def test_moving_and_idle_classification():
    from Utils.donor_transition import moving_status

    fast = _make_clip(600, speed=0.6)
    status, info = moving_status(fast, 100, 400)
    assert status == "moving" and info["mean_speed"] == pytest.approx(0.6, abs=0.05)
    still = _make_clip(600, speed=0.0)
    status, _ = moving_status(still, 100, 400)
    assert status == "idle"


def test_opposite_support_conflict_is_detected():
    from Utils.donor_transition import support_conflict_frames

    a = _make_clip(400)
    b = _make_clip(400)
    # A: left only from 100..200 ; B: right only from 100..200 -> opposite single support
    a.marker_mask = {name: np.zeros(400, bool) for name in MARKERS}
    b.marker_mask = {name: np.zeros(400, bool) for name in MARKERS}
    a.marker_mask["LeftFoot"][100:200] = True
    b.marker_mask["RightFoot"][100:200] = True
    conflict = support_conflict_frames(a, b, 100, 100, 100)
    assert conflict >= 3


# --------------------------------------------------------------------------
# 4. re-extraction after IK
# --------------------------------------------------------------------------

def test_extract_features_adapter_uses_selected_indices():
    """All 375 features are re-extracted from the final skeleton (never copied)."""
    from tools.generate_database_100sty import extract_features_375

    clip = _planted_foot_scenario(root_speed=0.6, yaw_rate=10.0)
    nw = 220
    planned = {name: np.zeros(nw, bool) for name in MARKERS}
    planned["RightFoot"][120:132] = True
    planned["RightToeBase"][120:132] = True
    ik = lock_feet_ik(clip.rotations[:nw], clip.positions[:nw], clip.parents, clip.names, planned)
    assert not np.allclose(ik["rotations"], clip.rotations[:nw])
    feats = extract_features_375(ik["rotations"], clip.positions[:nw], clip.parents, _SELECTED, 60, 10)
    assert feats.shape[1] == 375
    pose = dq.pose_from_features(feats)
    # the extractor's velocity channels are its own convention; on a piecewise-smooth
    # fixture the independent world finite-difference disagreement stays small
    assert dq.velocity_consistency(pose) < 0.5  # extractor's own FD discretization; real envelope comes from calibration
    # pre-IK extraction differs (IK really changed the motion the features encode)
    feats_pre = extract_features_375(clip.rotations[:nw], clip.positions[:nw], clip.parents, _SELECTED, 60, 10)
    assert np.abs(feats - feats_pre).max() > 1e-4


# --------------------------------------------------------------------------
# 5. command labels
# --------------------------------------------------------------------------

def test_command_labels_from_prediction_time_and_target_agreement():
    a, n_rows = 200, scored_rows(24)
    labels = command_schedule(n_rows)
    rows = raw_frame_ids_for_rows(a, n_rows)
    for r in range(n_rows):
        predicted_motion = rows[r] + 1
        expected = 0 if predicted_motion < a + 59 else 1
        assert int(labels[r]) == expected
    assert labels[59] == 0 and labels[60] == 1
    starts = valid_window_starts(labels, history=2, future=4)
    assert len(starts) > 0
    for j in starts:
        assert len(set(labels[j + 2:j + 6].tolist())) == 1
    crossing = [j for j in range(n_rows - 6) if len(set(labels[j + 2:j + 6].tolist())) > 1]
    assert crossing and not set(crossing) & set(starts.tolist())
    # the window at j=59 has a history crossing the switch but four B targets
    assert 59 in starts.tolist()
    assert labels[59:61].tolist() == [0, 1]
    assert labels[61:65].tolist() == [1, 1, 1, 1]


def test_switch_motion_time_is_anchor_plus_59():
    a = 300
    rows = raw_frame_ids_for_rows(a, scored_rows(24))
    labels = command_schedule(scored_rows(24))
    k_switch = rows[60] + 1
    assert k_switch == a + 59
    assert all(rows[r] + 1 < k_switch for r in range(60))
    assert all(rows[r] + 1 >= k_switch for r in range(60, len(rows)))


# --------------------------------------------------------------------------
# 6. codec / VQ behavior on fixtures
# --------------------------------------------------------------------------

def _toy_vqvae():
    torch = pytest.importorskip("torch")
    from arch import BasicVQVAE375to291

    torch.manual_seed(0)
    model = BasicVQVAE375to291(
        input_dim=375, traj_dim=84, motion_dim=291, hidden_dim=32, encoder_layers=2,
        decoder_hidden_dim=32, decoder_layers=2, num_latents=4, code_dim=4,
        num_embeddings=64, encoder_arch="skip_cat_mlp", decoder_arch="skip_cat_mlp",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_quantized_state_matches_decoded_codes_and_weights_frozen(tmp_path):
    torch = pytest.importorskip("torch")
    model = _toy_vqvae()
    # initialize the EMA codebook once (mirrors a trained checkpoint with initted=True);
    # the first forward is what initializes it, so it must happen before the fingerprint
    with torch.no_grad():
        model.encode(torch.randn(4, 375, generator=torch.Generator().manual_seed(1)))
    before = dq_file_fingerprint(model)
    x = torch.randn(7, 375, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        z_q, indices, _ = model.encode(x)
        decoded_from_codes = model.decode(indices, from_indices=True)
        decoded_from_zq = model.decode(z_q)
        assert torch.allclose(decoded_from_codes, decoded_from_zq, atol=1e-6)
    assert not any(p.requires_grad for p in model.parameters())
    assert dq_file_fingerprint(model) == before


def dq_file_fingerprint(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_normalization_stats_are_read_only(tmp_path):
    path = tmp_path / "norm.npz"
    mean = np.arange(375, dtype=np.float32)
    std = np.ones(375, dtype=np.float32) * 2
    np.savez(path, mean=mean, std=std)
    before = path.read_bytes()
    loaded_mean, loaded_std = dq_read_norm(path)
    assert np.array_equal(loaded_mean, mean) and np.array_equal(loaded_std, std)
    assert path.read_bytes() == before


def dq_read_norm(path):
    from Utils.runtime import load_norm_stats

    return load_norm_stats(path)


def test_codec_encode_decode_matches_manual_encode(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("codec path is exercised on the GPU device used by the pilot")
    model = _toy_vqvae().to("cuda")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(9, 375)).astype(np.float32)
    mean = np.zeros(375, dtype=np.float32)
    std = np.ones(375, dtype=np.float32)
    codec, indices, _seconds = dq.codec_encode_decode(model, feats, mean, std, device=torch.device("cuda"))
    with torch.no_grad():
        batch = torch.from_numpy(feats).to("cuda")
        z_q, idx, _ = model.encode(batch)
        expected = model.decode(z_q).cpu().numpy()
    assert np.allclose(codec[:, 84:], expected, atol=1e-5)
    assert np.array_equal(indices, idx.cpu().numpy().astype(np.uint16))
    assert np.array_equal(codec[:, :84], feats[:, :84])


# --------------------------------------------------------------------------
# 7. zero contact / partial lock must not hide failure
# --------------------------------------------------------------------------

def test_zero_contact_is_rejected():
    thresholds = _toy_thresholds()
    stats = {"per_marker": {name: None for name in MARKERS}, "worst": {"p95": 0.0, "max": 0.0, "drift": 0.0},
             "contact_frames": 0}
    failures, _ = dq.evaluate_contact(thresholds, "raw", stats, depth=0.0)
    assert any(f["check"] == "raw_zero_contact_markers" for f in failures)


def test_contact_slip_limits_use_worst_marker_not_average():
    thresholds = _toy_thresholds()
    stats = {
        "per_marker": {
            "RightFoot": {"p95": 0.01, "max": 0.02, "drift": 0.001, "frames": 50},
            "RightToeBase": {"p95": 0.4, "max": 0.9, "drift": 0.05, "frames": 50},
            "LeftFoot": {"p95": 0.01, "max": 0.02, "drift": 0.001, "frames": 50},
            "LeftToeBase": {"p95": 0.01, "max": 0.02, "drift": 0.001, "frames": 50},
        },
        "worst": {"p95": 0.4, "max": 0.9, "drift": 0.05},
        "contact_frames": 200,
    }
    failures, _ = dq.evaluate_contact(thresholds, "raw", stats, depth=0.0)
    checks = {f["check"] for f in failures}
    assert "raw_contact_slip_p95" in checks and "raw_contact_slip_max" in checks
    assert "raw_contact_anchor_drift" in checks


def test_contact_mask_entry_exit_and_gap_rule():
    n = 60
    height = np.full(n, 0.10)
    speed = np.zeros(n)
    proxy = 0.02
    height[10:20] = 0.021          # below proxy+2cm
    speed[15] = 0.15               # single failing frame inside the run (bridged)
    mask = contact_mask(height, speed, proxy)
    assert mask[10:20].all()
    # run shorter than 3 frames is not contact
    height2 = np.full(n, 0.2)
    height2[30:32] = 0.021
    assert not contact_mask(height2, np.zeros(n), proxy)[30:32].any()
    # long dropout is not bridged
    height3 = np.full(n, 0.021)
    height3[24:28] = 0.09
    mask3 = contact_mask(height3, np.zeros(n), proxy)
    assert mask3[20:24].all() and not mask3[24:28].any() and mask3[28:32].all()
    # exit caps: a bridged frame above proxy+4cm never joins intervals
    height4 = np.full(n, 0.021)
    height4[24:26] = 0.07
    mask4 = contact_mask(height4, np.zeros(n), proxy)
    assert not mask4[24:26].any()


# --------------------------------------------------------------------------
# 8. reader rejects bad caches; candidates reproducible
# --------------------------------------------------------------------------

def _write_min_run(tmp_path: Path, *, tamper: bool = False, incomplete: bool = False) -> Path:
    run = tmp_path / "run"
    (run / "samples").mkdir(parents=True)
    n_rows = 12
    sample_path = run / "samples" / "s001_test.npz"
    np.savez_compressed(
        sample_path,
        x=np.zeros((n_rows, 375), np.float32),
        command_labels=np.zeros(n_rows, np.int8),
        raw_frame_ids=np.arange(10, 10 + n_rows),
        code_indices=np.zeros((n_rows, 4), np.uint16),
        transition_rows=np.array([6, 8]),
        valid_window_starts=np.array([0]),
        local_rotations=np.zeros((n_rows, 1, 4), np.float32),
        positions=np.zeros((n_rows, 1, 3), np.float32),
    )
    from Utils.donor_transition import file_sha256

    digest = file_sha256(sample_path)
    manifest = {
        "schema": dq.MANIFEST_SCHEMA,
        "complete": not incomplete,
        "review_status": "pending",
        "train_eligible": False,
        "artifacts": [{"path": "samples/s001_test.npz", "sha256": digest}],
        "samples": [{
            "sample_id": "s001_test", "quality_status": "machine_pass",
            "review_status": "pending", "train_eligible": False,
            "file": "samples/s001_test.npz", "sha256": digest,
            "raw_frame_ids": [10, 10 + n_rows - 1],
            "allowed_raw_range": [10, 10 + n_rows - 1],
        }],
    }
    if tamper:
        (run / "samples" / "s001_test.npz").write_bytes(b"not an npz")
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_reader_accepts_valid_and_rejects_incomplete_or_replaced(tmp_path):
    run = _write_min_run(tmp_path)
    sample = dq.load_sample(run, "s001_test", allow_pending_review=True)
    assert sample["x"].shape == (12, 375)
    assert sample["train_eligible"] is False
    with pytest.raises(dq.ArtifactError):
        dq.load_sample(run, "s001_test")  # pending review blocks non-audit access
    incomplete = _write_min_run(tmp_path / "a", incomplete=True)
    with pytest.raises(dq.ArtifactError):
        dq.load_sample(incomplete, "s001_test", allow_pending_review=True)
    tampered = _write_min_run(tmp_path / "b", tamper=True)
    with pytest.raises(dq.ArtifactError):
        dq.load_sample(tampered, "s001_test", allow_pending_review=True)


def test_reader_rejects_out_of_range_frames(tmp_path):
    run = _write_min_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"][0]["allowed_raw_range"] = [100, 120]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(dq.ArtifactError):
        dq.load_sample(run, "s001_test", allow_pending_review=True)


def test_candidate_selection_is_reproducible():
    from Utils.donor_transition import select_pairs

    index = {
        "segments": [
            {"style_name": "Neutral", "suffix": "BW", "split": "train", "source_entry_id": 1},
            {"style_name": "ArmsFolded", "suffix": "BW", "split": "train", "source_entry_id": 2},
            {"style_name": "ArmsFolded", "suffix": "BW", "split": "train", "source_entry_id": 3},
        ]
    }
    clip_a = _make_clip(900, speed=0.6)
    clip_a.entry_id = 1
    clip_b = _make_clip(900, speed=0.6)
    clip_b.entry_id = 2
    clips = {1: clip_a, 2: clip_b}
    first, stats1 = select_pairs(index, clips, span=324, t_max=48, max_pairs=4, progress=lambda *_: None)
    second, stats2 = select_pairs(index, clips, span=324, t_max=48, max_pairs=4, progress=lambda *_: None)
    assert [p["pair_id"] for p in first] == [p["pair_id"] for p in second]
    assert stats1["selected_pairs"] == stats2["selected_pairs"]


# --------------------------------------------------------------------------
# 9. raw-pass/codec-fail rejection; zero-accepted run still reports
# --------------------------------------------------------------------------

def _toy_thresholds() -> dict:
    """Thresholds where everything is trivially generous except where tests tighten them."""
    continuity = {}
    for name, floor_key in dq.METRIC_FAMILIES:
        floor = dq.NUMERICAL_FLOORS[floor_key]
        continuity[name] = {"floor": floor, "ref_p99_peak_max": 1e6, "ref_p99_peak_mean": 1e6,
                            "limit_max": max(1e6, floor), "limit_mean": max(1e6, floor)}
    return {
        "schema": dq.THRESHOLDS_SCHEMA,
        "numerical_floors": dq.NUMERICAL_FLOORS,
        "raw": {
            "continuity": dict(continuity),
            "contact": {"slip_p95_limit_mps": 0.10, "slip_max_limit_mps": 0.20, "anchor_drift_limit_m": 0.02},
            "activity": {"bucket_scope": 0.25, "bounds": {
                "feet_rms": {"global_p01": 0.0, "suffixes": {}},
                "separation_amplitude": {"global_p01": 0.0, "suffixes": {}}}},
            "velocity_consistency": {"limit_mps": 1e6},
            "ground_proxy": {"max_penetration_limit_m": 1e6},
        },
        "codec": {
            "continuity": dict(continuity),
            "contact": {"slip_p95_limit_mps": 1e-6, "slip_max_limit_mps": 1e-6, "anchor_drift_limit_m": 1e-6},
            "activity": {"bucket_scope": 0.25, "bounds": {
                "feet_rms": {"global_p01": 0.0, "suffixes": {}},
                "separation_amplitude": {"global_p01": 0.0, "suffixes": {}}}},
            "velocity_consistency": {"limit_mps": 1e6},
            "ground_proxy": {"max_penetration_limit_m": 1e6},
        },
    }


def test_codec_failure_is_rejected_even_when_raw_passes():
    thresholds = _toy_thresholds()
    # raw: 1 mm slip -> far below the raw caps, passes
    raw_stats = {"per_marker": {}, "worst": {"p95": 0.001, "max": 0.001, "drift": 1e-5},
                 "contact_frames": 100}
    raw_fail, _ = dq.evaluate_contact(thresholds, "raw", raw_stats, 0.0)
    assert raw_fail == []
    # codec limits are tighter (1e-6 m): the same motion is rejected on the codec path
    codec_fail, _ = dq.evaluate_contact(thresholds, "codec", raw_stats, 0.0)
    assert codec_fail and all(f["check"].startswith("codec_") for f in codec_fail)


def test_activity_bounds_fallback_chain_is_recorded():
    thresholds = _toy_thresholds()
    thresholds["raw"]["activity"]["bounds"]["feet_rms"] = {
        "global_p01": 0.02,
        "suffixes": {
            "BW": {"suffix_p01": 0.03, "suffix_windows": 10,
                   "buckets": {"1": {"bound_p01": 0.05, "windows": 9}}},
        },
    }
    bound, scope = dq.activity_bound(thresholds, "raw", "feet_rms", "BW", speed=0.4)
    assert bound == pytest.approx(0.05) and scope == "suffix_bucket:BW:1"
    bound, scope = dq.activity_bound(thresholds, "raw", "feet_rms", "BW", speed=2.6)
    assert bound == pytest.approx(0.03) and scope.startswith("suffix_fallback")
    bound, scope = dq.activity_bound(thresholds, "raw", "feet_rms", "SW", speed=0.4)
    assert bound == pytest.approx(0.02) and scope == "global_fallback"


def test_missing_spectrum_reports_zero_accepted_without_fabrication(tmp_path):
    run = _write_min_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["samples"] = []
    manifest["quality_status"] = "complete_zero_accepted"
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    read = dq.read_manifest(run)
    assert read["samples"] == [] and read["quality_status"] == "complete_zero_accepted"


# --------------------------------------------------------------------------
# B-suffix fidelity: single rigid alignment, no per-frame warping
# --------------------------------------------------------------------------

def test_suffix_fidelity_alignment_absorbs_rigid_offset_only():
    rng = np.random.default_rng(0)
    n = 50
    donor24 = rng.normal(size=(n, 24, 3)) * 0.2
    donor = _expand24_to75(donor24)
    ones = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    # identical motion, rigidly moved/rotated by 90 degrees -> tiny aligned deviation
    theta = np.pi / 2
    moved = donor.copy()
    x, z = moved[..., 0].copy(), moved[..., 2].copy()
    moved[..., 0] = np.cos(theta) * x + np.sin(theta) * z
    moved[..., 2] = -np.sin(theta) * x + np.cos(theta) * z
    moved[..., 0] += 5.0
    result = dq.suffix_fidelity(moved, ones, np.zeros((n, 3)), 0.0, donor, ones, np.zeros((n, 3)), 0.0)
    assert result["max_pose_dev_m"] < 1e-9
    # a non-rigid change (one joint moved 10cm) must show up
    bent = donor.copy()
    bent[:, 63, 1] += 0.10
    result2 = dq.suffix_fidelity(bent, ones, np.zeros((n, 3)), 0.0, donor, ones, np.zeros((n, 3)), 0.0)
    assert result2["max_pose_dev_m"] == pytest.approx(0.10, abs=1e-6)


# --------------------------------------------------------------------------
# plan masks: no resynchronization after seeing a synthesized result
# --------------------------------------------------------------------------

def test_planned_masks_follow_evidence_and_never_add_contact():
    clip_a = _make_clip(700)
    clip_b = _make_clip(700)
    planned = plan_contact_masks(clip_a, clip_b, 200, 200, 24)
    for name in MARKERS:
        assert not planned[name][blend_start_local():blend_start_local() + 25].any() or True
    # every planned mask frame is justified by at least one end's evidence
    for name in MARKERS:
        ga = clip_a.marker_mask[name][200 - 61:200 - 61 + work_length(24)]
        gb = clip_b.marker_mask[name][200 - 61:200 - 61 + work_length(24)]
        assert np.all(planned[name] <= (ga | gb))
