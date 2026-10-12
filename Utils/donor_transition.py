"""Conservative offline donor-transition construction (pilot v1).

Scope (see docs/DONOR_AGENT_EXECUTION_GUIDE.md):
  - 4 upper-body pose styles, same FW/BW/SW suffix on both ends, train split only.
  - Full source-A root translation/rotation for the whole work clip.
  - Pure geometric SLERP of non-root local joint rotations with
    alpha(u) = 6u^5 - 15u^4 + 10u^3, plus a limited two-bone lower-limb IK that
    keeps root/pelvis translation fixed.
  - Time rate is fixed at 1; no time warping anywhere.

Row/timeline layout (frozen):
  - Work clip local frame i corresponds to raw frame `g0 + i` with
    `g0 = anchor - LEAD_CONTEXT` (61), so the first scored row's motion state is
    raw frame `anchor - 2`.
  - Scored rows: 60 command-A rows + (T+1) blend sample rows + 240 B-suffix rows.
    Row r holds the motion state of raw frame `anchor - 2 + r` and the
    trajectory context centered at `anchor - 1 + r` (verified bit-exactly
    against the existing database in the identity phase).
  - Blend sample points sit at work-local frames `s .. s+T` (s = 120), i.e. raw
    motion frames `anchor + 59 .. anchor + 59 + T`, with
    `alpha(k/T) = smoothstep(k/T)`; frames after the blend are pure D.
  - Command label: row r is A while the *target* motion frame `anchor - 1 + r`
    is below `k_switch`; the first B-commanded row is row 60 and its target is
    the first blend sample (alpha = 0), so the response starts exactly at the
    command and builds up over the T-frame blend.

Contact semantics (frozen): a marker is in stable contact on maximal runs of
frames satisfying `height <= proxy + 2cm` and `horizontal speed <= 0.10 m/s`
for at least 3 consecutive frames; runs are bridged across at most 2 failing
frames as long as every bridged frame stays below the exit caps
`height <= proxy + 4cm` and `speed <= 0.20 m/s`.

This module is read-only with respect to the existing pipeline: it imports the
original `prepare_motion` / `extract_features_375` conventions and never writes
into the database, caches or checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

import Utils.bvh as bvh
from Utils import quat
from tools.generate_database_100sty import (  # noqa: F401  (original conventions)
    SELECTED_JOINTS,
    extract_features_375,
    prepare_motion,
)

FPS = 60.0
DT = 1.0 / FPS

PILOT_STYLES = ("Neutral", "ArmsAboveHead", "ArmsFolded", "ArmsBehindBack")
PILOT_SUFFIXES = ("FW", "BW", "SW")
SEED = 42

# Work-clip layout, in raw frames / rows (guide section 6.1).
LEAD_CONTEXT = 61          # raw frames before the first scored row's motion frame
PREFIX_ROWS = 60           # command-A rows (visible A prefix)
SUFFIX_ROWS = 240          # B suffix rows
TRAIL_MIN = 62             # protection frames after the last scored trajectory frame
TRANSITION_FRAMES = (24, 36, 48)

# Moving/idle classification (guide section 5.2).
MOVE_MEAN_SPEED = 0.20     # m/s over the 120-frame context
MOVE_SUBWINDOW_PATH = 0.15  # m over every complete 60-frame subwindow
IDLE_MEAN_SPEED = 0.05     # m/s or below is idle

# Pair pre-screen caps (guide section 5.2).
PAIR_SPEED_RMS = 0.10      # m/s
PAIR_YAW_RMS = 10.0        # deg/s
PAIR_EVENT_TOL = 2         # frames, hard cap on matched contact-event error (handover window)
EVENT_MARGIN = 24          # frames of boundary margin around the switch/transition window
SUPPORT_CONFLICT_RUN = 3   # frames of clear opposite single-support -> reject

# Contact rules (guide section 6.3).
CONTACT_ENTRY_HEIGHT = 0.02
CONTACT_ENTRY_SPEED = 0.10
CONTACT_EXIT_HEIGHT = 0.04
CONTACT_EXIT_SPEED = 0.20
CONTACT_MIN_RUN = 3
CONTACT_MAX_GAP = 2
CONTACT_FLOOR_SLOW_SPEED = 0.10

# IK caps (guide section 6.3).
IK_MAX_CORRECTION = 0.03   # m, per-frame required correction cap (excess over IK_BAND)
IK_MAX_JOINT_DELTA_DEG = 10.0
IK_RAMP_FRAMES = 3         # raised-cosine weight ramp at interval boundaries
IK_BAND = 0.01             # m, comfort band: deviation inside it is never corrected

MARKERS = ("RightFoot", "RightToeBase", "LeftFoot", "LeftToeBase")
SIDE_OF_MARKER = {
    "RightFoot": "Right", "RightToeBase": "Right",
    "LeftFoot": "Left", "LeftToeBase": "Left",
}

_TAIL_REQUIRED = LEAD_CONTEXT + PREFIX_ROWS + 1 + SUFFIX_ROWS + TRAIL_MIN - 1  # 363


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def smoothstep(u: np.ndarray) -> np.ndarray:
    """alpha(u) = 6u^5 - 15u^4 + 10u^3 with zero first/second derivatives at both ends."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return (6 * u**5 - 15 * u**4 + 10 * u**3).astype(np.float32)


def shortest_arc_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Angle in degrees of the shortest-arc relative rotation qa -> qb."""
    rel = quat.mul_inv(quat.normalize(qb), quat.normalize(qa))
    rel = quat.abs(rel)
    return np.degrees(2.0 * np.linalg.norm(quat.log(rel), axis=-1))


def slerp_shortest(qa: np.ndarray, qb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Stable shortest-arc SLERP; `weight` broadcasts to the quaternion leading axes."""
    qa = quat.normalize(np.asarray(qa, dtype=np.float32))
    qb = quat.normalize(np.asarray(qb, dtype=np.float32))
    dot = np.sum(qa * qb, axis=-1, keepdims=True)
    qb = np.where(dot < 0.0, -qb, qb)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    w = np.asarray(weight, dtype=np.float32)
    while w.ndim < qa.ndim:
        w = w[..., None]
    small = theta < 1e-4
    sin_theta = np.sin(theta)
    safe = np.where(sin_theta < 1e-8, 1.0, sin_theta)
    slerped = np.sin((1.0 - w) * theta) / safe * qa + np.sin(w * theta) / safe * qb
    linear = qa + (qb - qa) * w
    return quat.normalize(np.where(small, linear, slerped).astype(np.float32))


def raised_cosine_ramp(length: int) -> np.ndarray:
    """Raised-cosine 0-><1 ramp of `length` values (1 excluded; 1 is the interval interior)."""
    length = int(max(length, 0))
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = np.arange(1, length + 1, dtype=np.float64) / (length + 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _run_intervals(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i, n = 0, int(len(mask))
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def mask_intervals(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    return _run_intervals(mask, min_len)


def _side_mask(marker_mask_map: dict, side: str) -> np.ndarray:
    return marker_mask_map[side + "Foot"] | marker_mask_map[side + "ToeBase"]


# --------------------------------------------------------------------------
# clip loading
# --------------------------------------------------------------------------

@dataclass
class ClipData:
    entry_id: int
    style: str
    suffix: str
    split: str
    path: Path
    source_start: int
    source_stop: int
    label: int
    names: list
    parents: np.ndarray
    offsets: np.ndarray
    rotations: np.ndarray          # local rotations (quaternions), full clip
    positions: np.ndarray          # local positions (rest offsets + root translation), meters
    frametime: float
    global_pos: np.ndarray         # FK global positions, meters
    yaw: np.ndarray                # savgol-smoothed facing yaw following the extractor convention
    yaw_rate: np.ndarray           # deg/s
    planar_speed: np.ndarray       # local planar root speed, m/s
    local_vel: np.ndarray          # [T,2] root local planar velocity (vx, vz) in the heading frame
    marker_height: dict            # marker -> height series (m)
    marker_speed: dict             # marker -> horizontal world speed (m/s)
    marker_mask: dict = field(default_factory=dict)
    marker_intervals: dict = field(default_factory=dict)
    proxies: dict = field(default_factory=dict)
    floor_ref: float = 0.0        # own ground reference: P01 slow-frame height of the toe markers

    @property
    def length(self) -> int:
        return int(self.positions.shape[0])

    def joint_index(self, name: str) -> int:
        return self.names.index(name)


def load_db_index(db_dir: Path) -> dict:
    db_dir = Path(db_dir)
    meta = json.loads((db_dir / "meta.json").read_text(encoding="utf-8"))
    segments = json.loads((db_dir / "segments.json").read_text(encoding="utf-8"))
    styles = json.loads((db_dir / "StyleNames.json").read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "segments": segments,
        "style_names": styles,
        "paths": {
            "meta": db_dir / "meta.json",
            "segments": db_dir / "segments.json",
            "styles": db_dir / "StyleNames.json",
            "x": db_dir / "X.f32",
            "labels": db_dir / "Labels.i32",
        },
    }


def find_segment(index: dict, style: str, suffix: str) -> dict | None:
    for seg in index["segments"]:
        if seg["style_name"] == style and seg["suffix"] == suffix:
            return seg
    return None


def _heading_from_rotations(rotations: np.ndarray) -> np.ndarray:
    """Facing yaw following the extractor convention (savgol-smoothed root +Z direction)."""
    import scipy.signal as signal

    direction = quat.mul_vec(rotations[:, 0:1], np.array([0.0, 0.0, 1.0], dtype=np.float32))[:, 0]
    direction = direction / np.clip(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8, None)
    savgol_window = min(61, len(direction) if len(direction) % 2 == 1 else len(direction) - 1)
    if savgol_window > 3:
        direction = signal.savgol_filter(direction, savgol_window, 3, axis=0, mode="interp")
    direction = direction / np.clip(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8, None)
    return np.arctan2(direction[:, 0], direction[:, 2]).astype(np.float32)


def marker_speed_series(points: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(points)
    if len(points) > 2:
        vel[1:-1] = (points[2:] - points[:-2]) * (0.5 * FPS)
        vel[0] = vel[1]
        vel[-1] = vel[-2]
    return np.linalg.norm(vel[:, [0, 2]], axis=-1).astype(np.float32)


def _root_local_planar_velocity(yaw: np.ndarray, global_root: np.ndarray) -> np.ndarray:
    """Root local planar velocity (vx, vz) expressed in the smoothed heading frame.

    Same convention as the extractor's root velocity channels: central difference of the
    root world position rotated by the smoothed facing (the extractor divides by dt; the
    planar components are what the pair pre-screen compares).
    """
    vel = np.zeros((len(global_root), 3), dtype=np.float64)
    if len(global_root) > 2:
        vel[1:-1] = (global_root[2:] - global_root[:-2]) * (0.5 * FPS)
        vel[0] = vel[1]
        vel[-1] = vel[-2]
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx = c * vel[:, 0] - s * vel[:, 2]
    lz = s * vel[:, 0] + c * vel[:, 2]
    return np.stack([lx, lz], axis=1).astype(np.float32)


def _root_local_planar_speed(yaw: np.ndarray, global_root: np.ndarray) -> np.ndarray:
    vel = _root_local_planar_velocity(yaw, global_root)
    return np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2).astype(np.float32)


def load_clip(path: Path, start: int, stop: int, *, style: str, suffix: str, split: str,
              entry_id: int, label: int) -> ClipData:
    raw = bvh.load_zeroeggs(str(path))
    frametime = float(raw.get("frametime") or 0.0)
    if abs(frametime - 1.0 / 60.0) > 1e-6:
        raise ValueError(f"{path}: Frame Time {frametime!r} is not ~1/60")
    parents, rotations, positions = prepare_motion(
        raw, start=int(start), stop=int(stop), coordinate_convert=False
    )
    global_rot, global_pos = quat.fk(rotations, positions, parents)
    names = list(raw["names"])
    yaw = _heading_from_rotations(rotations)
    yaw_unwrapped = np.unwrap(yaw.astype(np.float64))
    yaw_rate = np.zeros_like(yaw)
    if len(yaw) > 2:
        yaw_rate[1:-1] = (yaw_unwrapped[2:] - yaw_unwrapped[:-2]) * (0.5 * FPS)
        yaw_rate[0] = yaw_rate[1]
        yaw_rate[-1] = yaw_rate[-2]
    planar = _root_local_planar_speed(yaw, global_pos[:, 0])
    local_vel = _root_local_planar_velocity(yaw, global_pos[:, 0])
    marker_height, marker_speed = {}, {}
    for side in ("Right", "Left"):
        for part in ("Foot", "ToeBase"):
            name = side + part
            pts = global_pos[:, names.index(name)]
            marker_height[name] = pts[:, 1].astype(np.float32)
            marker_speed[name] = marker_speed_series(pts)
    floor_candidates = []
    for name in ("RightToeBase", "LeftToeBase"):
        slow = marker_speed[name] <= CONTACT_FLOOR_SLOW_SPEED
        values = marker_height[name][slow] if slow.sum() >= 5 else marker_height[name]
        floor_candidates.append(float(np.percentile(values, 1)))
    return ClipData(
        entry_id=int(entry_id), style=style, suffix=suffix, split=split, path=Path(path),
        source_start=int(start), source_stop=int(stop), label=int(label), names=names,
        # offsets converted to meters exactly like prepare_motion does for positions;
        # the root OFFSET (initial root position) is kept for reference but unused
        parents=parents, offsets=np.asarray(raw["offsets"], dtype=np.float32) * 0.01,
        rotations=rotations, positions=positions, frametime=frametime, global_pos=global_pos,
        yaw=yaw, yaw_rate=yaw_rate.astype(np.float32), planar_speed=planar,
        local_vel=local_vel,
        marker_height=marker_height, marker_speed=marker_speed,
        floor_ref=float(min(floor_candidates)),
    )


SKELETON_OFFSET_ATOL = 1e-3  # meters; 100STYLE files carry ~6e-4 m (0.06 cm) noise on non-root offsets


def skeleton_offsets_diff(a: ClipData, b: ClipData) -> float:
    """Max non-root rest-offset difference in meters (ClipData offsets are meters).

    The root OFFSET line is each file's initial root position (superseded by the
    per-frame root translation), so it is excluded from the compatibility check.
    """
    diff = np.abs(np.asarray(a.offsets, dtype=np.float64)[1:] - np.asarray(b.offsets, dtype=np.float64)[1:])
    return float(np.max(diff))


def skeleton_compatible(a: ClipData, b: ClipData) -> bool:
    if list(a.names) != list(b.names):
        return False
    if not np.array_equal(a.parents, b.parents):
        return False
    if a.offsets.shape != b.offsets.shape:
        return False
    return skeleton_offsets_diff(a, b) <= SKELETON_OFFSET_ATOL


# --------------------------------------------------------------------------
# contact detection
# --------------------------------------------------------------------------

def set_proxies(clip: ClipData, proxies: dict) -> None:
    """Freeze the marker floor proxies on a clip and derive its stable-contact masks."""
    clip.proxies = dict(proxies)
    clip.marker_mask = {}
    clip.marker_intervals = {}
    for name in MARKERS:
        mask = contact_mask(clip.marker_height[name], clip.marker_speed[name], clip.proxies[name])
        clip.marker_mask[name] = mask
        clip.marker_intervals[name] = mask_intervals(mask)


def contact_mask(height: np.ndarray, speed: np.ndarray, proxy: float) -> np.ndarray:
    """Stable-contact mask: entry runs bridged over <=CONTACT_MAX_GAP frames below exit caps."""
    entry = (height <= proxy + CONTACT_ENTRY_HEIGHT) & (speed <= CONTACT_ENTRY_SPEED)
    n = int(len(height))
    mask = np.zeros(n, dtype=bool)
    for a, b in _run_intervals(entry, CONTACT_MIN_RUN):
        mask[a:b] = True
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        k = j
        while k < n and not mask[k]:
            k += 1
        if k < n and (k - j) <= CONTACT_MAX_GAP:
            gap_ok = np.all(height[j:k] <= proxy + CONTACT_EXIT_HEIGHT) and np.all(
                speed[j:k] <= CONTACT_EXIT_SPEED
            )
            if gap_ok:
                mask[j:k] = True
        i = max(k, j)
    return mask


def contact_events(clip: ClipData, lo: int, hi: int) -> list[tuple[int, str, str]]:
    """Contact events (frame, marker, 'in'/'out') with frames inside [lo, hi)."""
    events: list[tuple[int, str, str]] = []
    for name in MARKERS:
        for a, b in clip.marker_intervals[name]:
            if lo <= a < hi:
                events.append((int(a), name, "in"))
            if lo <= b < hi:
                events.append((int(b), name, "out"))
    events.sort()
    return events


# --------------------------------------------------------------------------
# moving/idle classification and anchors
# --------------------------------------------------------------------------

def moving_status(clip: ClipData, lo: int, hi: int) -> tuple[str, dict]:
    speed = clip.planar_speed[lo:hi]
    mean_speed = float(np.mean(speed)) if len(speed) else 0.0
    paths = []
    for start in range(0, max(len(speed) - 61 + 1, 0), 60):
        paths.append(path_len(clip, lo + start, lo + start + 61))
    min_path = float(min(paths)) if paths else 0.0
    info = {"mean_speed": mean_speed, "min_subwindow_path": min_path}
    if mean_speed <= IDLE_MEAN_SPEED:
        return "idle", info
    if mean_speed >= MOVE_MEAN_SPEED and (min_path >= MOVE_SUBWINDOW_PATH or not paths):
        return "moving", info
    return "ambiguous", info


def path_len(clip: ClipData, lo: int, hi: int) -> float:
    pts = clip.global_pos[lo:hi, 0][:, [0, 2]]
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=-1)))


def max_anchor(clip_length: int, t_frames: int) -> int:
    """Largest anchor whose work clip (length 424+T) stays inside the clip."""
    return int(clip_length) - _TAIL_REQUIRED - int(t_frames)


def anchor_candidates(clip: ClipData, *, max_anchors: int = 16, t_max: int | None = None) -> list[dict]:
    t_max = TRANSITION_FRAMES[-1] if t_max is None else int(t_max)
    lo = 120
    hi = max_anchor(clip.length, t_max)
    if hi < lo:
        return []
    values = np.unique(np.linspace(lo, hi, num=min(max_anchors, hi - lo + 1)).astype(int))
    out = []
    for a in values.tolist():
        status, stats = moving_status(clip, a - 120, a)
        out.append({"anchor": int(a), "status": status, **stats})
    return out


# --------------------------------------------------------------------------
# pair alignment and scoring
# --------------------------------------------------------------------------

def _events_by_key(events: list[tuple[int, str, str]]) -> dict[tuple[str, str], np.ndarray]:
    by_key: dict[tuple[str, str], list[int]] = {}
    for t, marker, kind in events:
        by_key.setdefault((marker, kind), []).append(int(t))
    return {key: np.asarray(sorted(values), dtype=np.int64) for key, values in by_key.items()}


def _match_events(
    by_key: dict[tuple[str, str], np.ndarray], ev_a: list, delta: int, *, window: tuple[int, int]
) -> tuple[float, int]:
    """Nearest same-(marker,type) event matching after shifting B by delta frames.

    Events are matched inside the handover `window` (motion-frame coordinates of A);
    donor events outside it are irrelevant and are not counted. With delta = u - t
    (B events shifted onto A), an A event at t corresponds to a B event at t + delta.
    Returns (max |dt| over matched events, unmatched events on either side in the window).
    """
    best_err = 0.0
    matched = 0
    used: set[tuple[str, str, int]] = set()
    for t, marker, kind in ev_a:
        times = by_key.get((marker, kind))
        if times is None or len(times) == 0:
            continue
        pos = int(np.searchsorted(times, t + delta))
        candidates = []
        for idx in (pos - 1, pos, pos + 1):
            if 0 <= idx < len(times) and (marker, kind, idx) not in used:
                candidates.append((abs(int(times[idx]) - (t + delta)), idx))
        if not candidates:
            continue
        err, idx = min(candidates)
        used.add((marker, kind, idx))
        matched += 1
        best_err = max(best_err, float(err))
    b_lo, b_hi = int(window[0]) + delta, int(window[1]) + delta
    unmatched_b = 0
    for key, times in by_key.items():
        bounds = np.searchsorted(times, [b_lo, b_hi])
        for idx in range(int(bounds[0]), int(bounds[1])):
            if (key[0], key[1], idx) not in used:
                unmatched_b += 1
    unmatched = (len(ev_a) - matched) + unmatched_b
    return best_err, unmatched


def _longest_run(flags: np.ndarray) -> int:
    if flags.size == 0 or not flags.any():
        return 0
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2]))


def foot_separation_series(clip: ClipData) -> np.ndarray:
    """Precomputed (left_foot - right_foot) world XZ separation, shape [T,2]."""
    li = clip.joint_index("LeftFoot")
    ri = clip.joint_index("RightFoot")
    diff = clip.global_pos[:, li] - clip.global_pos[:, ri]
    return np.stack([diff[:, 0], diff[:, 2]], axis=1).astype(np.float64)


def phase_distance(sep_a: np.ndarray, sep_b: np.ndarray, a: int, d: int, span: int) -> tuple[float, float]:
    """Mean angle between the left-right foot separation direction of the two ends.

    Frames where either side has no reliable separation (<5cm) are skipped; when
    fewer than 30% of frames are reliable the pair receives the worst score
    instead of a fake perfect match on a zero vector.
    """
    n = min(span, len(sep_a) - a, len(sep_b) - d)
    if n <= 0:
        return 90.0, 0.0
    fa = sep_a[a:a + n]
    fb = sep_b[d:d + n]
    na = np.linalg.norm(fa, axis=1)
    nb = np.linalg.norm(fb, axis=1)
    valid = (na >= 0.05) & (nb >= 0.05)
    frac = float(valid.mean())
    if frac < 0.3:
        return 90.0, frac
    cosang = np.clip(np.sum(fa[valid] * fb[valid], axis=1) / (na[valid] * nb[valid]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)).mean()), frac


def _series_rms(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def support_conflict_frames(
    clip_a: ClipData, clip_b: ClipData, a: int, d: int, t_frames: int, margin: int = EVENT_MARGIN
) -> int:
    """Longest run of frames with opposite single support inside the handover window."""
    lo, hi = handover_window(a, t_frames, margin)
    n = hi - lo
    if n <= 0:
        return 0
    al = _side_mask(clip_a.marker_mask, "Left")[lo:hi]
    ar = _side_mask(clip_a.marker_mask, "Right")[lo:hi]
    bl = _side_mask(clip_b.marker_mask, "Left")[lo + (d - a):hi + (d - a)]
    br = _side_mask(clip_b.marker_mask, "Right")[lo + (d - a):hi + (d - a)]
    conflict = ((al & ~ar) & (br & ~bl)) | ((ar & ~al) & (bl & ~br))
    return _longest_run(conflict)


def boundary_rotation_rms(clip_a: ClipData, clip_b: ClipData, a: int, d: int, window: int = 3) -> float:
    qa = clip_a.rotations[max(a - window, 0):a]
    qb = clip_b.rotations[max(d - window, 0):d]
    n = min(len(qa), len(qb))
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean(shortest_arc_deg(qa[-n:], qb[-n:]) ** 2)))


def handover_window(a: int, t_frames: int, margin: int = EVENT_MARGIN) -> tuple[int, int]:
    """Motion-frame window where A/B contact events must align (guide: boundary+transition).

    The suffix is pure-D local motion on A's root, so donor events there are D's own
    and are validated by the contact/B-fidelity gates instead of being matched to A;
    only the command boundary and the blend need the two ends' evidence to agree.
    Switch motion frame = a + 58; window = [switch - margin, switch + T + margin].
    """
    sw = a + 58
    return sw - int(margin), sw + int(t_frames) + int(margin)


def search_donor_offset(
    clip_a: ClipData,
    clip_b: ClipData,
    a: int,
    *,
    span: int,
    t_max: int,
) -> dict | None:
    """Find the integer donor offset d that best aligns B onto A at anchor a.

    Hard filters, in order:
      1. contact events inside the handover window must match within PAIR_EVENT_TOL
         frames (nearest same-(marker,type) event after shifting B by d);
      2. root planar speed RMS over the whole candidate span <= 0.10 m/s;
      3. yaw-rate RMS over the whole candidate span <= 10 deg/s;
      4. no >=3-frame run of clear opposite single support inside the handover window;
      5. both anchors classified as moving (120-frame context).
    Ranking tuple (deterministic): (max event error, phase distance, speed RMS,
    yaw RMS, boundary rotation RMS, donor entry id, donor offset).
    """
    lo = 120
    hi = max_anchor(clip_b.length, t_max)
    w_lo, w_hi = handover_window(a, t_max)
    ev_a = contact_events(clip_a, w_lo, w_hi)
    if not ev_a:
        return None
    by_key = _events_by_key(contact_events(clip_b, 0, clip_b.length))
    candidate_deltas: set[int] = set()
    for t, marker, kind in ev_a:
        times = by_key.get((marker, kind))
        if times is None:
            continue
        for u in times.tolist():
            delta = u - t
            if lo <= a + delta <= hi:
                candidate_deltas.add(int(delta))
    if not candidate_deltas:
        return None
    sep_a = foot_separation_series(clip_a)
    sep_b = foot_separation_series(clip_b)
    survivors = []
    for delta in sorted(candidate_deltas):
        d = a + delta
        max_err, unmatched = _match_events(by_key, ev_a, delta, window=(w_lo, w_hi))
        if max_err > PAIR_EVENT_TOL or unmatched > max(2, len(ev_a) // 10):
            continue
        speed_rms = _series_rms(clip_a.planar_speed[a:a + span], clip_b.planar_speed[d:d + span])
        if speed_rms > PAIR_SPEED_RMS:
            continue
        yaw_rms = _series_rms(clip_a.yaw_rate[a:a + span], clip_b.yaw_rate[d:d + span])
        if yaw_rms > PAIR_YAW_RMS:
            continue
        pdist, valid_frac = phase_distance(sep_a, sep_b, a, d, span)
        survivors.append(
            {"delta": int(delta), "d": int(d), "max_event_err": float(max_err),
             "unmatched_events": int(unmatched), "speed_rms": float(speed_rms),
             "yaw_rms": float(yaw_rms), "phase_distance": float(pdist),
             "phase_valid_fraction": float(valid_frac)}
        )
    if not survivors:
        return None
    survivors.sort(key=lambda s: (s["max_event_err"], s["phase_distance"], s["speed_rms"],
                                  s["yaw_rms"], s["d"]))
    status_a = moving_status(clip_a, a - 120, a)[0]
    if status_a != "moving":
        return None
    best = None
    for cand in survivors[:12]:
        d = cand["d"]
        status_b = moving_status(clip_b, d, d + 120)[0]
        if status_b != "moving":
            continue
        conflict = support_conflict_frames(clip_a, clip_b, a, d, t_max)
        if conflict >= SUPPORT_CONFLICT_RUN:
            continue
        boundary_rms = boundary_rotation_rms(clip_a, clip_b, a, d)
        entry = {
            **cand,
            "total_events_in_window": int(len(ev_a)),
            "handover_window": [int(w_lo), int(w_hi)],
            "boundary_rot_rms_deg": float(boundary_rms),
            "support_conflict_frames": int(conflict),
        }
        key = (entry["max_event_err"], entry["phase_distance"], entry["speed_rms"],
               entry["yaw_rms"], entry["boundary_rot_rms_deg"], clip_b.entry_id, d)
        entry["key"] = key
        if best is None or key < best["key"]:
            best = entry
    if best is not None:
        # recorded diagnostics for the alternative (stricter) reading of the guide's
        # "root local planar velocity difference": vector RMS instead of magnitude RMS
        best["local_vel_vec_rms_diag"] = _local_vec_rms(clip_a, clip_b, a, best["d"], span)
        best["world_vel_vec_rms_diag"] = _world_vec_rms(clip_a, clip_b, a, best["d"], span)
    return best


def _local_vec_rms(clip_a: ClipData, clip_b: ClipData, a: int, d: int, span: int) -> float:
    va = clip_a.local_vel[a:a + span]
    vb = clip_b.local_vel[d:d + span]
    n = min(len(va), len(vb))
    if n == 0:
        return float("nan")
    diff = va[:n] - vb[:n]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _world_vec_rms(clip_a: ClipData, clip_b: ClipData, a: int, d: int, span: int) -> float:
    """World planar velocity vector RMS including the heading difference at the anchor."""
    la = clip_a.local_vel[a:a + span]
    lb = clip_b.local_vel[d:d + span]
    ya = clip_a.yaw[a:a + span]
    yb = clip_b.yaw[d:d + span]
    n = min(len(la), len(lb))
    if n == 0:
        return float("nan")
    ya, yb, la, lb = ya[:n], yb[:n], la[:n], lb[:n]
    ca_, sa_ = np.cos(ya), np.sin(ya)
    cb_, sb_ = np.cos(yb), np.sin(yb)
    wax = ca_ * la[:, 0] + sa_ * la[:, 1]
    waz = -sa_ * la[:, 0] + ca_ * la[:, 1]
    wbx = cb_ * lb[:, 0] + sb_ * lb[:, 1]
    wbz = -sb_ * lb[:, 0] + cb_ * lb[:, 1]
    diff = np.stack([wax - wbx, waz - wbz], axis=1)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# --------------------------------------------------------------------------
# group selection (round-robin over directed (styleA, styleB, suffix) groups)
# --------------------------------------------------------------------------

def directed_groups(index: dict) -> tuple[list[dict], list[dict]]:
    available, unavailable = [], []
    for suffix in PILOT_SUFFIXES:
        for style_a in PILOT_STYLES:
            for style_b in PILOT_STYLES:
                if style_a == style_b:
                    continue
                seg_a = find_segment(index, style_a, suffix)
                seg_b = find_segment(index, style_b, suffix)
                group = {"style_a": style_a, "style_b": style_b, "suffix": suffix}
                if seg_a is None or seg_b is None or seg_a["split"] != "train" or seg_b["split"] != "train":
                    group["segments"] = [
                        None if seg is None else {"entry_id": seg["source_entry_id"], "split": seg["split"]}
                        for seg in (seg_a, seg_b)
                    ]
                    unavailable.append(group)
                else:
                    group["ids"] = (int(seg_a["source_entry_id"]), int(seg_b["source_entry_id"]))
                    available.append(group)
    return available, unavailable


def select_pairs(
    index: dict,
    clips: dict,
    *,
    span: int,
    t_max: int,
    max_pairs: int,
    progress=print,
) -> tuple[list[dict], dict]:
    available, unavailable = directed_groups(index)
    per_group: dict[tuple[str, str, str], list[dict]] = {}
    stats = {"anchor_candidates": 0, "groups_available": 0, "groups_unavailable": len(unavailable),
             "group_scan": {}, "unavailable_groups": unavailable}
    for group in available:
        id_a, id_b = group["ids"]
        clip_a, clip_b = clips[id_a], clips[id_b]
        if not skeleton_compatible(clip_a, clip_b):
            group["reject"] = "skeleton_mismatch"
            unavailable.append(group)
            continue
        candidates = []
        for anchor in anchor_candidates(clip_a, t_max=t_max):
            stats["anchor_candidates"] += 1
            if anchor["status"] != "moving":
                continue
            best = search_donor_offset(clip_a, clip_b, anchor["anchor"], span=span, t_max=t_max)
            if best is None:
                continue
            candidates.append(
                {
                    "style_a": group["style_a"],
                    "style_b": group["style_b"],
                    "suffix": group["suffix"],
                    "entry_a": id_a,
                    "entry_b": id_b,
                    "label_a": clip_a.label,
                    "label_b": clip_b.label,
                    "source_start_a": clip_a.source_start,
                    "source_stop_a": clip_a.source_stop,
                    "source_start_b": clip_b.source_start,
                    "source_stop_b": clip_b.source_stop,
                    "source_path_a": str(clip_a.path),
                    "source_path_b": str(clip_b.path),
                    "a": int(anchor["anchor"]),
                    "moving_status": anchor["status"],
                    **best,
                }
            )
        candidates.sort(key=lambda c: c["key"])
        per_group[(group["style_a"], group["style_b"], group["suffix"])] = candidates
        stats["groups_available"] += 1
        stats["group_scan"][f"{group['style_a']}->{group['style_b']}:{group['suffix']}"] = len(candidates)
    order = sorted(per_group.keys())
    selected: list[dict] = []
    cursor = {key: 0 for key in order}
    while len(selected) < max_pairs:
        progressed = False
        for key in order:
            if len(selected) >= max_pairs:
                break
            idx = cursor[key]
            if idx < len(per_group[key]):
                selected.append(per_group[key][idx])
                cursor[key] += 1
                progressed = True
        if not progressed:
            break
    for rank, item in enumerate(selected):
        item["pair_rank"] = rank
        item["pair_id"] = f"{item['style_a']}__{item['style_b']}__{item['suffix']}__a{item['a']}_d{item['d']}"
    stats["selected_pairs"] = len(selected)
    stats["per_group_selected"] = {f"{k[0]}->{k[1]}:{k[2]}": cursor[k] for k in order}
    return selected, stats


# --------------------------------------------------------------------------
# construction: blend + planned contacts + IK
# --------------------------------------------------------------------------

def scored_rows(t_frames: int) -> int:
    """Scored rows: 60 command-A rows + (T+1) blend sample rows + 240 B-suffix rows."""
    return PREFIX_ROWS + (int(t_frames) + 1) + SUFFIX_ROWS


def blend_start_local() -> int:
    """Work-local frame of the first blend sample.

    Chosen so the motion state of the first B-commanded row (row 60) is exactly
    this sample with alpha = 0: row r holds local frame 59+r, therefore the first
    B row's motion is `LEAD_CONTEXT + PREFIX_ROWS - 2` local = anchor + 58 raw,
    and the last A-commanded target (row 59 -> local 119 = anchor + 58 raw) is
    still pure A.
    """
    return LEAD_CONTEXT + PREFIX_ROWS - 2  # 119


def suffix_rows_start_local(t_frames: int) -> int:
    """Work-local frame of the first pure-D B-suffix *row*'s motion state (alpha = 1)."""
    return blend_start_local() + int(t_frames) + 1  # 120 + T


def blend_end_local(t_frames: int) -> int:
    """Work-local frame of the last blend sample (alpha = 1); pure D from here on."""
    return blend_start_local() + int(t_frames)


def work_length(t_frames: int) -> int:
    return LEAD_CONTEXT + scored_rows(t_frames) + TRAIL_MIN


def build_transition(clip_a: ClipData, clip_b: ClipData, a: int, d: int, t_frames: int) -> dict:
    """Assemble the raw work clip: A root + alpha-blended local rotations."""
    t_frames = int(t_frames)
    n_rows = scored_rows(t_frames)
    nw = work_length(t_frames)
    g0, d0 = a - LEAD_CONTEXT, d - LEAD_CONTEXT
    rot_a = clip_a.rotations[g0:g0 + nw]
    s = blend_start_local()
    local_t = np.arange(nw) - s
    alpha = np.zeros(nw, dtype=np.float32)
    in_transition = (local_t >= 0) & (local_t <= t_frames)
    alpha[in_transition] = smoothstep(local_t[in_transition] / float(t_frames))
    alpha[local_t > t_frames] = 1.0
    rot_d = clip_b.rotations[d0:d0 + nw]
    rot = slerp_shortest(rot_a, rot_d, alpha)
    # exact endpoint assignment: alpha == 0 regions reproduce A bit-exactly and the
    # tail reproduces D bit-exactly (no normalization drift at the endpoints)
    rot[local_t < 0] = rot_a[local_t < 0]
    rot[local_t > t_frames] = rot_d[local_t > t_frames]
    rot[:, 0] = rot_a[:, 0]  # full source-A root transform
    positions = clip_a.positions[g0:g0 + nw].copy()  # rest offsets + A root translation
    return {
        "rotations": rot, "positions": positions, "alpha": alpha,
        "t_start_local": s, "nw": nw, "n_rows": n_rows, "g0": g0, "d0": d0,
    }


def plan_contact_masks(clip_a: ClipData, clip_b: ClipData, a: int, d: int, t_frames: int) -> dict:
    """Frozen planned stable-contact masks per marker over the work timeline (guide 6.3).

    - frames before the blend: A's own evidence;
    - blend samples: A's evidence AND the donor's evidence (conservative handover);
    - pure-D frames: the donor's own evidence.
    """
    nw = work_length(int(t_frames))
    g0, d0 = a - LEAD_CONTEXT, d - LEAD_CONTEXT
    s = blend_start_local()
    e = blend_end_local(t_frames)
    timeline = np.arange(nw)
    planned = {}
    for name in MARKERS:
        ma = clip_a.marker_mask[name][g0:g0 + nw]
        mb = clip_b.marker_mask[name][d0:d0 + nw]
        mask = np.zeros(nw, dtype=bool)
        before = timeline < s
        blend = (timeline >= s) & (timeline <= e)
        after = timeline > e
        mask[before] = ma[before]
        mask[blend] = ma[blend] & mb[blend]
        mask[after] = mb[after]
        planned[name] = mask
    return planned


PLANNED_SIDE_INTERVALS_DEPRECATED = True  # only per-marker masks drive the IK


def _contact_weights(mask: np.ndarray) -> np.ndarray:
    """Contact weights with continuous raised-cosine ramps at interval boundaries."""
    weight = mask.astype(np.float32).copy()
    for a, b in mask_intervals(mask, 1):
        length = b - a
        ramp = min(IK_RAMP_FRAMES, max((length - 1) // 2, 0))
        if ramp <= 0:
            continue
        values = raised_cosine_ramp(ramp)
        weight[a:a + ramp] = values
        weight[b - ramp:b] = values[::-1]
    return weight


def lock_feet_ik(
    rotations: np.ndarray,
    positions: np.ndarray,
    parents: np.ndarray,
    names: Sequence[str],
    planned: dict,
    *,
    offsets: np.ndarray | None = None,
    band: float = IK_BAND,
    blend_start: int | None = None,
) -> dict:
    """Limited two-bone ankle IK bringing planned stances inside the contact band.

    Frozen semantics (documented in the manifest/report):

    - IK intervals are the per-side FOOT (ankle) stable-contact intervals from the
      planned mask; the ToeBase mask stays planned scoring evidence. The ankle is
      targeted exactly like the offline render precedent (ToeBase follows the foot
      rotation, it is not a second IK chain).
    - Anchor per interval: for an interval that already started before the blend
      (`start < blend_start`) the anchor is the centroid of the interval's pre-blend
      portion -- i.e. the source motion's own contact position -- so feet in contact
      before the command keep their source world anchor. Otherwise the anchor is the
      interval centroid of the pre-IK constructed motion (the donor stance's landing
      area on A's root path). The anchor minimizes the maximum distance of the fixed
      anchor (`每个stance最大锚点漂移` is measured against exactly this anchor).
    - The IK only corrects the part of the deviation that exceeds the comfort band
      (`band`, 1cm): the per-frame requested target is
      `anchor + clip(pre_ik_foot - anchor, band)` so that a stance already inside the
      band is never moved (the source's natural foot roll is preserved) and the post-IK
      anchoring drift is <= band (comfortably inside the 2cm acceptance cap).
    - Required correction = the per-frame excess `max(0, |foot - anchor| - band)`;
      its maximum over the whole clip must be <= IK_MAX_CORRECTION (3cm), the joint
      change versus the un-IK'ed result must be <= IK_MAX_JOINT_DELTA_DEG (10deg),
      and unreachable targets (beyond hip+knee reach) are counted; any violation
      rejects the attempt instead of relaxing the cap.
    """
    from Visualization.pose_postprocess import rotation_between, two_bone_target

    rot = np.array(rotations, dtype=np.float32, copy=True)
    pos = np.array(positions, dtype=np.float32, copy=True)
    nw = rot.shape[0]
    idx = {name: i for i, name in enumerate(names)}
    leg_ids = {side: (idx[side + "UpLeg"], idx[side + "Leg"], idx[side + "Foot"]) for side in ("Right", "Left")}
    weights = {side: _contact_weights(planned[side + "Foot"]) for side in ("Right", "Left")}
    _g_rot_pre, g_pos_pre = quat.fk(rot, pos, parents)

    intervals: dict[str, list[dict]] = {"Right": [], "Left": []}
    anchors: dict[tuple[str, int], np.ndarray] = {}
    for side in ("Right", "Left"):
        foot_local = idx[side + "Foot"]
        for n, (i0, i1) in enumerate(mask_intervals(planned[side + "Foot"], 1)):
            if blend_start is not None and i0 < blend_start:
                hi = min(i1, blend_start)
                anchor = g_pos_pre[i0:hi, foot_local].mean(axis=0).astype(np.float32)
                source_anchored = True
            else:
                anchor = g_pos_pre[i0:i1, foot_local].mean(axis=0).astype(np.float32)
                source_anchored = False
            anchors[(side, n)] = anchor
            deviation = np.linalg.norm(g_pos_pre[i0:i1, foot_local] - anchor, axis=-1)
            required = np.maximum(0.0, deviation - band)
            intervals[side].append({
                "start": int(i0), "end": int(i1), "anchor": anchor.tolist(),
                "source_anchored": bool(source_anchored),
                "pre_ik_max_deviation_m": float(deviation.max()),
                "required_correction_max": float(required.max()),
                "required_correction_mean": float(required.mean()),
            })

    joint_delta = 0.0
    unreachable_frames = 0
    for frame in range(nw):
        active = [side for side in ("Right", "Left") if weights[side][frame] > 0.0]
        if not active:
            continue
        gr, gp = quat.fk(rot[frame][None], pos[frame][None], parents)
        gr, gp = gr[0], gp[0]
        for side in active:
            w = float(weights[side][frame])
            hip, knee, foot = leg_ids[side]
            interval = None
            for item in intervals[side]:
                if item["start"] <= frame < item["end"]:
                    interval = item
                    break
            if interval is None:
                continue
            anchor = np.asarray(interval["anchor"], dtype=np.float32)
            vec = gp[foot] - anchor
            dist = float(np.linalg.norm(vec))
            if dist <= band:
                continue
            target = anchor + vec * (band / dist) if w >= 1.0 else gp[foot] + w * (anchor + vec * (band / dist) - gp[foot])
            if offsets is not None:
                upper = float(np.linalg.norm(offsets[knee]))
                lower = float(np.linalg.norm(offsets[foot]))
                if float(np.linalg.norm(target - gp[hip])) > upper + lower - 1e-4:
                    unreachable_frames += 1
            new_knee, new_foot = two_bone_target(gp[hip], gp[knee], gp[foot], target)
            hip_rot = quat.normalize(quat.mul(rotation_between(gp[knee] - gp[hip], new_knee - gp[hip]), gr[hip]))
            knee_rot = quat.normalize(quat.mul(rotation_between(gp[foot] - gp[knee], new_foot - new_knee), gr[knee]))
            rot[frame, hip] = quat.inv_mul(gr[int(parents[hip])], hip_rot)
            rot[frame, knee] = quat.inv_mul(hip_rot, knee_rot)
            d_hip = float(shortest_arc_deg(gr[hip][None], hip_rot[None])[0])
            d_knee = float(shortest_arc_deg(gr[knee][None], knee_rot[None])[0])
            joint_delta = max(joint_delta, d_hip, d_knee)

    # post-IK anchoring drift per interval (diagnostics; the acceptance gate re-measures
    # the final motion independently)
    _g_rot_post, g_pos_post = quat.fk(rot, pos, parents)
    post_max = 0.0
    for side in ("Right", "Left"):
        foot_local = idx[side + "Foot"]
        for item in intervals[side]:
            anchor = np.asarray(item["anchor"], dtype=np.float32)
            drift = np.linalg.norm(g_pos_post[item["start"]:item["end"], foot_local] - anchor, axis=-1)
            item["post_ik_max_drift_m"] = float(drift.max())
            post_max = max(post_max, item["post_ik_max_drift_m"])

    return {
        "rotations": rot,
        "weights": weights,
        "intervals": intervals,
        "joint_delta_max_deg": float(joint_delta),
        "unreachable_frames": int(unreachable_frames),
        "post_ik_max_drift_m": float(post_max),
        "anchor_map": {f"{side}_{n}": v.tolist() for (side, n), v in anchors.items()},
    }


# --------------------------------------------------------------------------
# labels and row mapping
# --------------------------------------------------------------------------

def command_schedule(n_rows: int, *, prefix_rows: int = PREFIX_ROWS) -> np.ndarray:
    """Command label per scored row: A for row index < prefix_rows else B.

    Row r's motion state is raw frame `anchor - 2 + r` and its target is
    `anchor - 1 + r`; the first B-commanded row (r = prefix_rows) targets the
    first blend sample (alpha = 0), so the switch motion time is
    k_switch = anchor + prefix_rows - 1.  The prefix is never relabelled.
    """
    labels = np.empty(int(n_rows), dtype=np.int8)
    labels[:prefix_rows] = 0  # A
    labels[prefix_rows:] = 1  # B
    return labels


def valid_window_starts(labels: np.ndarray, *, history: int = 2, future: int = 4) -> np.ndarray:
    """Starts of 6-row TF windows whose four target rows share one command label."""
    n = int(len(labels))
    starts = [
        j for j in range(0, n - (history + future) + 1)
        if np.all(labels[j + history:j + history + future] == labels[j + history])
    ]
    return np.asarray(starts, dtype=np.int64)


def raw_frame_ids_for_rows(a: int, n_rows: int) -> np.ndarray:
    """Motion-state raw frame id of each scored row (row r -> raw frame a - 2 + r)."""
    return np.arange(a - 2, a - 2 + int(n_rows), dtype=np.int64)
