"""Shared protocol code for the VQ temporal-continuity source experiment (S0/S1/S2/S4).

All definitions here mirror analysis/vq_temporal_source/COMMON.md, which was frozen
before the B0/B1 pilot started. Read that file first; do not change a metric identity
here without either withdrawing the corresponding report claim or re-registering.

Everything is read-only against the database and checkpoints: the probe never writes
a checkpoint and never leaves a model in training mode.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.motion_eval import (  # noqa: E402
    MotionEvalConfig,
    TOE_JOINT_INDICES,
    _toe_contact_mask_from_reference,
    compute_sequence_metric_sums,
)
from Utils.motion375 import (  # noqa: E402
    SELECTED_24_NAMES,
    reconstruct_root_track_from_375,
    reconstruct_selected_globals_from_375,
)
from Utils.model_loading import build_vqvae_from_checkpoint  # noqa: E402
from Utils.reproducibility import file_sha256, state_fingerprint  # noqa: E402
from Utils.runtime import load_norm_stats  # noqa: E402

DB = ROOT / "database_100sty_375_memmap_nomirror"
SRC_RUN = ROOT / "checkpoints" / "vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror"
SRC_CKPT = SRC_RUN / "best.pt"
SRC_NORM = SRC_RUN / "norm_stats.npz"
ARCHIVED_VAL12 = ROOT / "analysis" / "vq_code_stability_2026-10-14" / "audit" / "baseline_val12.json"
ARCHIVED_RESIDUAL = ROOT / "analysis" / "vq_code_stability_2026-10-14" / "residual_baseline.json"
OUT_ROOT = ROOT / "analysis" / "vq_temporal_source"
CLASSIFIER_CKPT = (
    ROOT / "checkpoints" / "fid_feature_extractor_camdm100_60fps_240f_nomirror"
    / "100style_position_classifier.pth"
)

N_JOINTS = len(SELECTED_24_NAMES)  # 24
# Foot/toe joints used by the activity proxies (RightFoot, RightToeBase, LeftFoot, LeftToeBase).
FOOT_JOINT_INDICES = (18, 19, 22, 23)
TOE_INDICES = tuple(TOE_JOINT_INDICES)
MOVING_FRAME_THRESHOLD_CM = 0.5
STRONG_MOVING_FRAME_THRESHOLD_CM = 1.0
FOOT_LIFT_HEIGHT_CM = 5.0

SUFFIX_GROUPS = {
    "ID": "idle",
    "FW": "walk", "BW": "walk", "SW": "walk",
    "FR": "run", "BR": "run", "SR": "run",
    "TR1": "turn", "TR2": "turn", "TR3": "turn",
}


def group_of(suffix: str) -> str:
    if suffix not in SUFFIX_GROUPS:
        raise ValueError(f"unknown 100STYLE suffix: {suffix!r}")
    return SUFFIX_GROUPS[suffix]


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))


def read_db():
    meta = json.loads((DB / "meta.json").read_text())
    segments = json.loads((DB / "segments.json").read_text())
    x = np.memmap(DB / "X.f32", mode="r", dtype=np.float32, shape=(int(meta["num_frames"]), 375))
    return meta, segments, x


def fixed_val12(*, clips: int = 12, frames: int = 120, seed: int = 42) -> list[dict]:
    """The archived audit cases: same filter, same RNG call, same frame count.

    Asserts against analysis/vq_code_stability_2026-10-14/audit/baseline_val12.json so a
    silent protocol change cannot masquerade as "the same fixed validation set".
    """
    _, segments, _ = read_db()
    pool = [
        s for s in segments
        if s.get("split") == "val" and not s.get("mirror", False) and s["num_frames"] >= frames + 2
    ]
    rng = np.random.RandomState(seed)
    picks = [pool[i] for i in rng.choice(len(pool), min(clips, len(pool)), replace=False)]
    archived = json.loads(ARCHIVED_VAL12.read_text())
    got = [int(s["frame_start"]) for s in picks]
    if got != list(archived["starts"]):
        raise RuntimeError(
            "fixed_val12 selection drifted from the archived audit starts; "
            f"got {got[:4]}... expected {list(archived['starts'])[:4]}..."
        )
    return picks


def load_probe_model(ckpt: Path, device: torch.device):
    """Frozen eval-mode VQ + its own normalization (each run wrote/explicit-passed its norm)."""
    ckpt = Path(ckpt).resolve()
    model = build_vqvae_from_checkpoint(ckpt, freeze=True).to(device).eval()
    norm_path = ckpt.parent / "norm_stats.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"norm_stats.npz next to checkpoint is required: {norm_path}")
    mean, std = load_norm_stats(norm_path)
    return model, mean, std


def positions_from_motion_norm(motion_norm: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor):
    """normalized motion291 [T,291] -> (local_cm, world_cm) for the 24 selected joints.

    The world reconstruction fills the 84-dim trajectory slot with zeros: it reads only the
    motion block's own root-motion channels, and mixing the normalized trajectory block with
    raw motion units was withdrawn from the archived reports as misleading.
    """
    motion_raw = motion_norm * std_t[84:] + mean_t[84:]
    local = (motion_raw[:, 3:75].reshape(-1, N_JOINTS, 3) * 100.0).detach().cpu().numpy()
    layout = torch.cat((torch.zeros_like(motion_raw[:, :84]), motion_raw), -1).detach().cpu().numpy()
    world, _ = reconstruct_selected_globals_from_375(layout)
    return local, np.asarray(world, dtype=np.float32) * 100.0


def d2(values: np.ndarray) -> np.ndarray:
    """Second difference along the frame axis: (T, J, 3) -> (T-2, J, 3)."""
    return values[2:] - 2.0 * values[1:-1] + values[:-2]


def joint_norm(values: np.ndarray) -> np.ndarray:
    """Per joint-frame L2 over the 3 coordinates, flattened (archived probe's aggregation:
    (T, J, 3) -> (T*J,)). Positions are already cm; do NOT rescale again."""
    return np.linalg.norm(values.astype(np.float64), axis=-1).reshape(-1)


def jitter_values_cm(local_cm: np.ndarray) -> np.ndarray:
    """Per joint-frame ||second difference|| in cm; mean equals the official Jitter sum/count."""
    return joint_norm(d2(local_cm))


def local_mpjpe_sum_count(local_cm: np.ndarray, reference_cm: np.ndarray) -> tuple[float, int]:
    """Root-relative joint error, same formula as Utils.vqvae_eval's local_mpjpe (sum, count)."""
    per_joint_frame = np.linalg.norm(local_cm.astype(np.float64) - reference_cm.astype(np.float64), axis=-1)
    return float(per_joint_frame.sum()), int(per_joint_frame.size)


def _stat_block(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return dict(mean=float("nan"), p50=float("nan"), p95=float("nan"), count=0)
    return dict(
        mean=float(values.mean()),
        p50=float(np.quantile(values, 0.5)),
        p95=float(np.quantile(values, 0.95)),
        count=int(values.size),
    )


def _pairwise(values: np.ndarray) -> np.ndarray:
    """Per joint-frame displacement, cm (positions are already cm)."""
    return np.linalg.norm(values[1:] - values[:-1], axis=-1)


def activity_proxies(local_cm: np.ndarray, cfg: MotionEvalConfig) -> dict:
    """Registered activity/anti-degeneration proxies (COMMON.md section 3), all cm-based."""
    steps = _pairwise(local_cm.astype(np.float64))  # [T-1, 24]
    pose_step = steps.mean(-1)
    foot = local_cm[:, FOOT_JOINT_INDICES, :].astype(np.float64)
    toe = local_cm[:, TOE_INDICES, :].astype(np.float64)
    toe_height = toe[..., 1]
    toe_speed_step = np.linalg.norm(toe[1:] - toe[:-1], axis=-1)  # [T-1, 2] cm/frame
    return {
        "moving_frame_fraction": float((pose_step > MOVING_FRAME_THRESHOLD_CM).mean()),
        "strong_moving_frame_fraction": float((pose_step > STRONG_MOVING_FRAME_THRESHOLD_CM).mean()),
        "pose_step_mean_cm": float(pose_step.mean()),
        "leg_travel_cm": float(np.linalg.norm(foot[1:] - foot[:-1], axis=-1).sum(axis=0).mean()),
        "toe_height_range_cm": float((toe_height.max(0) - toe_height.min(0)).mean()),
        "foot_lift_fraction": float((toe_height > FOOT_LIFT_HEIGHT_CM).mean()),
        "toe_speed_mean_cm_per_frame": float(toe_speed_step.mean()),
    }


@torch.no_grad()
def probe_source(model, mean, std, rows: np.ndarray, device: torch.device, cfg: MotionEvalConfig):
    """One fixed val12 source: residual decomposition + official metrics + activity + codes.

    Returns (json_entry, raw_curves). `raw_curves` holds the per-frame scalar curves so the
    caller can build both per-source and pooled statistics without re-running the model.
    """
    mean_t = torch.as_tensor(mean, device=device)
    std_t = torch.as_tensor(std, device=device)
    x = torch.as_tensor((rows - mean) / std, device=device)

    z_e = model.encode_pre_vq(x)
    z_q, idx, _ = model.encode(x)
    idx = idx.reshape(len(x), model.num_latents)
    motion_q = model.decode(z_q)
    motion_c = model.decode(z_e)

    # codec roundtrip on the decoded motion (trajectory block kept from the input)
    motion_q_raw = motion_q * std_t[84:] + mean_t[84:]
    redo_like = torch.cat((x[:, :84], motion_q), -1)
    _, idx_rt, _ = model.encode(redo_like)
    idx_rt = idx_rt.reshape(len(x), model.num_latents)
    flip = (idx_rt != idx).float()
    # Count of slots that change between adjacent frames (mean over transitions) and the same
    # quantity divided by the slot count = fraction of slot-transitions. Both go to JSON under
    # the archived probe's names; do not divide twice.
    gt_switch_count = (idx[1:] != idx[:-1]).float().sum(-1).mean().item() if len(idx) > 1 else 0.0
    gt_switch_fraction = (idx[1:] != idx[:-1]).float().mean().item() if len(idx) > 1 else 0.0

    lq, wq = positions_from_motion_norm(motion_q, mean_t, std_t)
    lc, wc = positions_from_motion_norm(motion_c, mean_t, std_t)
    lg, wg = positions_from_motion_norm(x[:, 84:], mean_t, std_t)

    r_local = lq - lc
    e_local = lq - lg
    r_world = wq - wc
    e_world = wq - wg
    # Aggregation = per joint-frame L2, the archived residual probe's convention; the mean then
    # equals the official Jitter sum/count for the jitter_* entries (asserted in run_probe).
    curves = {
        "r_local": joint_norm(r_local),
        "dr_local": joint_norm(r_local[1:] - r_local[:-1]),
        "d2r_local": joint_norm(d2(r_local)),
        "r_world": joint_norm(r_world),
        "d2r_world": joint_norm(d2(r_world)),
        "d2e_local": joint_norm(d2(e_local)),
        "d2e_world": joint_norm(d2(e_world)),
        "jitter_quant_local": jitter_values_cm(lq),
        "jitter_cont_local": jitter_values_cm(lc),
        "jitter_gt_local": jitter_values_cm(lg),
        "jitter_quant_world": jitter_values_cm(wq),
        "jitter_cont_world": jitter_values_cm(wc),
        "jitter_gt_world": jitter_values_cm(wg),
    }
    out = {
        "frames": int(len(rows)),
        **{key: _stat_block(values) for key, values in curves.items()},
        "codec_flip_rate": float(flip.mean()),
        "codec_flip_per_head": flip.mean(0).cpu().tolist(),
        "gt_switch_slots_per_frame": float(gt_switch_count),
        "gt_switch_fraction_per_slot": float(gt_switch_fraction),
        "codes_used": int(torch.unique(idx.reshape(-1)).numel()),
        "codes_total": int(model.quantizer.num_embeddings),
    }
    for name, gt, pred in (("activity_gt", lg, lg), ("activity_q", lq, lq), ("activity_c", lc, lc)):
        out[name] = activity_proxies(pred, cfg)

    # Official metrics on the raw 375 layout (reference = GT, prediction = GT with decoded motion).
    def raw_with(motion_norm: torch.Tensor) -> np.ndarray:
        raw = np.array(rows, copy=True)
        raw[:, 84:] = (motion_norm * std_t[84:] + mean_t[84:]).detach().cpu().numpy()
        return raw

    for tag, motion_norm, local_pred in (("q", motion_q, lq), ("c", motion_c, lc)):
        sums = compute_sequence_metric_sums(
            reference_375=np.array(rows, copy=True), prediction_375=raw_with(motion_norm), cfg=cfg,
        )
        # `local_mpjpe` is not part of SequenceMetricSums; use vqvae_eval's own formula on the
        # 24-joint root-frame positions so the number is the same quantity the trainer gates on.
        lsum, lcount = local_mpjpe_sum_count(local_pred, lg)
        out[f"official_{tag}"] = dict(
            jitter_sum_cm=float(sums.jitter_sum_cm), jitter_count=int(sums.jitter_count),
            mpjpe_sum_cm=float(sums.mpjpe_sum_cm), mpjpe_count=int(sums.mpjpe_count),
            local_mpjpe_sum_cm=lsum, local_mpjpe_count=lcount,
            root_ade_sum_cm=float(sums.root_ade_sum_cm), root_ade_count=int(sums.root_ade_count),
            fs_local_sum_cm=float(sums.foot_sliding_sum_cm), fs_local_count=int(sums.foot_sliding_count),
            fs_world_sum_cm=float(sums.foot_sliding_world_sum_cm),
            fs_world_count=int(sums.foot_sliding_world_count),
            contact_possible=int(sums.foot_contact_possible_count),
        )
    gt_sums = compute_sequence_metric_sums(
        reference_375=np.array(rows, copy=True), prediction_375=np.array(rows, copy=True), cfg=cfg,
    )
    gsum, gcount = local_mpjpe_sum_count(lg, lg)
    out["official_gt"] = dict(
        jitter_sum_cm=float(gt_sums.jitter_sum_cm), jitter_count=int(gt_sums.jitter_count),
        mpjpe_sum_cm=0.0, mpjpe_count=int(gt_sums.mpjpe_count),
        local_mpjpe_sum_cm=gsum, local_mpjpe_count=gcount,
        root_ade_sum_cm=0.0, root_ade_count=int(gt_sums.root_ade_count),
        fs_local_sum_cm=float(gt_sums.foot_sliding_sum_cm), fs_local_count=int(gt_sums.foot_sliding_count),
        fs_world_sum_cm=float(gt_sums.foot_sliding_world_sum_cm),
        fs_world_count=int(gt_sums.foot_sliding_world_count),
        contact_possible=int(gt_sums.foot_contact_possible_count),
    )

    # Foot-contact diagnostics: GT-masked (official) and predicted-masked (new, COMMON.md §2/§3).
    # `_toe_contact_mask_from_reference` expects positions in METERS (it rescales to cm inside);
    # local_cm is already cm, so convert back. Passing cm would multiply heights by 100.
    def _mask(positions_cm: np.ndarray) -> np.ndarray:
        return _toe_contact_mask_from_reference(
            (positions_cm / 100.0).astype(np.float32), dt=cfg.dt,
            contact_height_cm=cfg.foot_contact_height_cm,
            contact_speed_cm_s=cfg.foot_contact_speed_cm_s,
        )

    gt_local_mask = _mask(lg)
    pred_local_mask = _mask(lq)
    gt_gl_mask = _mask(wg)
    pred_gl_mask = _mask(wq)
    out["contact"] = dict(
        gt_local_true=int(gt_local_mask.sum()), gt_local_possible=int(gt_local_mask.size),
        pred_local_true=int(pred_local_mask.sum()), pred_local_possible=int(pred_local_mask.size),
        gt_world_true=int(gt_gl_mask.sum()), gt_world_possible=int(gt_gl_mask.size),
        pred_world_true=int(pred_gl_mask.sum()), pred_world_possible=int(pred_gl_mask.size),
        pred_contact_coverage=float(pred_local_mask.mean()) if pred_local_mask.size else float("nan"),
        gt_contact_coverage=float(gt_local_mask.mean()) if gt_local_mask.size else float("nan"),
    )
    return out, curves


def mean_of(per_source: list[dict], getter) -> dict:
    values = np.asarray([getter(c) for c in per_source], dtype=np.float64)
    return _stat_block(values)


def aggregate_probe(per_source: list[dict], curves: list[dict], cfg: MotionEvalConfig) -> dict:
    """Pooled + cross-source aggregates over the fixed val12 set.

    `pooled_*` concatenates every frame of every source (the archived audit's reading);
    `per_source_*` treats each source as one sample (what the paired bootstrap pairs on).
    """
    agg: dict = {}
    for tag in ("q", "c", "gt"):
        rows = [c[f"official_{tag}"] for c in per_source]
        agg[f"official_{tag}"] = {
            key: float(sum(r[key + "_sum_cm"] for r in rows) / max(sum(r[key + "_count"] for r in rows), 1))
            for key in ("jitter", "mpjpe", "local_mpjpe", "root_ade", "fs_local", "fs_world")
        }
        agg[f"official_{tag}"]["contact_possible"] = int(sum(r["contact_possible"] for r in rows))
    for key in curves[0]:
        pooled = np.concatenate([c[key] for c in curves])
        per_source_means = np.asarray([c[key].mean() for c in curves])
        per_source_p95 = np.asarray([np.quantile(c[key], 0.95) if c[key].size else np.nan for c in curves])
        agg[key] = dict(
            pooled_mean=float(pooled.mean()),
            pooled_p50=float(np.quantile(pooled, 0.5)),
            pooled_p95=float(np.quantile(pooled, 0.95)),
            frames=int(pooled.size),
            per_source_mean=float(per_source_means.mean()),
            per_source_mean_std=float(per_source_means.std(ddof=0)),
            per_source_p95=float(np.nanmean(per_source_p95)),
        )
    agg["codec_flip_rate"] = mean_of(per_source, lambda c: c["codec_flip_rate"])
    agg["gt_switch_slots_per_frame"] = mean_of(per_source, lambda c: c["gt_switch_slots_per_frame"])
    agg["gt_switch_fraction_per_slot"] = mean_of(per_source, lambda c: c["gt_switch_fraction_per_slot"])
    agg["codes_used_fraction"] = float(
        np.mean([c["codes_used"] for c in per_source]) / per_source[0]["codes_total"]
    )
    for key in ("activity_gt", "activity_q", "activity_c"):
        for sub in per_source[0][key]:
            agg[f"{key}.{sub}"] = mean_of(per_source, lambda c, k=key, s=sub: c[k][s])
    for sub in per_source[0]["contact"]:
        agg[f"contact.{sub}"] = float(np.mean([c["contact"][sub] for c in per_source]))
    return agg


@torch.no_grad()
def dead_code_report(model, device) -> dict:
    codebook = model.quantizer.vq._codebook
    cluster = codebook.cluster_size.detach().reshape(-1).float()
    threshold = float(model.quantizer.dead_code_threshold)
    return dict(
        codes_total=int(cluster.numel()),
        codes_never_assigned=int((cluster == 0).sum().item()),
        codes_below_threshold=int((cluster < threshold).sum().item()),
        threshold_units="EMA assignments per batch (library threshold_ema_dead_code)",
        cluster_size_min=float(cluster.min()), cluster_size_max=float(cluster.max()),
        cluster_size_median=float(cluster.median()),
        initted=bool(getattr(codebook, "initted", torch.tensor(False)).reshape(-1)[0].item()),
    )


@torch.no_grad()
def style_proxy(model, mean, std, device, *, frames: int = 240) -> dict:
    """Frozen 100-class classifier applied to VQ-reconstructed clips (IND style proxy).

    Covers ALL non-mirror val sources with >= `frames` frames (the full 90-clip val split),
    so one flipped source is 1.1pp rather than 8.3pp and the registered <=2pp gate has
    meaningful granularity. Representation-level proxy: the report classifier is frozen
    (never optimized here) and only val sources are read. Not the controller-level metric.
    """
    from Utils.controller_validation import classifier_label_mapping
    from Utils.fid_feature_extractor import (
        build_fid_feature_extractor_from_checkpoint,
        motion375_clip_to_classifier_input,
    )

    _, segments, x_source = read_db()
    picks = sorted(
        (s for s in segments if s.get("split") == "val" and not s.get("mirror", False)
         and int(s["num_frames"]) >= frames),
        key=lambda s: (int(s["source_entry_id"]), int(s["frame_start"])),
    )
    names = json.loads((DB / "StyleNames.json").read_text())
    classifier, meta = build_fid_feature_extractor_from_checkpoint(CLASSIFIER_CKPT, map_location="cpu")
    if int(meta.get("window_size", 0)) != frames:
        raise ValueError(f"style proxy needs the {frames}-frame classifier, got window_size={meta.get('window_size')}")
    classifier = classifier.to(device).eval().requires_grad_(False)
    label_map = classifier_label_mapping(names, meta, device=device)

    per_source = []
    for s in picks:
        raw = np.array(x_source[s["frame_start"]:s["frame_start"] + frames], dtype=np.float32, copy=True)
        x = torch.as_tensor((raw - mean) / std, device=device).unsqueeze(0)
        z_q, _, _ = model.encode(x)
        motion = model.decode(z_q)
        pred = raw.copy()
        pred[:, 84:] = motion[0].cpu().numpy() * std[84:] + mean[84:]
        _, logits = classifier(motion375_clip_to_classifier_input(pred, joint_set=meta["joint_set"]).to(device))
        target = int(label_map[int(s["label"])].item())
        per_source.append(dict(
            source_entry_id=int(s["source_entry_id"]), style=s["style_name"], suffix=s["suffix"],
            correct=int(logits.argmax(1).item() == target),
        ))
    acc = float(np.mean([p["correct"] for p in per_source]))
    return dict(
        protocol=f"frozen_100class_classifier_on_{frames}f_vq_reconstruction_of_all_{len(per_source)}_val_sources",
        classifier_sha256=file_sha256(CLASSIFIER_CKPT), accuracy=acc,
        sources=int(len(per_source)), per_source=per_source,
        limitation="representation-level proxy on the full val split; not controller rollout accuracy",
    )


def full_probe(ckpt: Path, *, device: str = "cuda", style: bool = True, tag: str | None = None) -> dict:
    device = torch.device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, mean, std = load_probe_model(Path(ckpt), device)
    before = state_fingerprint(model.state_dict())
    picks = fixed_val12()
    _, _, x_source = read_db()
    cfg = MotionEvalConfig()
    per_source, curves = [], []
    for s in picks:
        rows = np.array(x_source[s["frame_start"]:s["frame_start"] + 121], dtype=np.float32, copy=True)
        entry, raw = probe_source(model, mean, std, rows, device, cfg)
        entry.update(source_entry_id=int(s["source_entry_id"]), style=s["style_name"], suffix=s["suffix"],
                     group=group_of(s["suffix"]), frame_start=int(s["frame_start"]))
        per_source.append(entry)
        curves.append(raw)
    after = state_fingerprint(model.state_dict())
    if before != after:
        raise RuntimeError("probe mutated the model state; the frozen-model contract is broken")
    agg = aggregate_probe(per_source, curves, cfg)
    # Identity check (S1 DoD): the probe's own jitter curves must reproduce Utils.motion_eval's
    # Jitter sum/count exactly; a mismatch means one of the two paths is measuring something else.
    for tag, key in (("q", "jitter_quant_local"), ("c", "jitter_cont_local"), ("gt", "jitter_gt_local")):
        reference = agg[f"official_{tag}"]["jitter"]
        measured = agg[key]["pooled_mean"]
        if abs(reference - measured) > 1e-6:
            raise RuntimeError(f"jitter identity failed for {tag}: official={reference} probe={measured}")
    payload = dict(
        tag=tag or Path(ckpt).parent.name,
        checkpoint=str(Path(ckpt).resolve()),
        checkpoint_sha256=file_sha256(Path(ckpt).resolve()),
        checkpoint_state_sha256=before,
        source_commit=git_commit(),
        split="val", clips=len(picks), frames=120, seed=42,
        starts=[int(s["frame_start"]) for s in picks],
        per_source=per_source,
        # Raw per-frame curves are intentionally NOT serialized: every registered statistic
        # (per-source mean/p50/p95, pooled same) is a function of them and is already stored;
        # keeping ~4000 floats x 12 sources x 13 curves would add ~11MB per probe of redundancy.
        aggregate=agg,
        dead_codes=dead_code_report(model, device),
        style_proxy=style_proxy(model, mean, std, device) if style else None,
        protocol_note=(
            "r = P(D(q)) - P(D(z_e)); d2* are per-frame L2 norms of second differences of the 24-joint "
            "positions in cm; official_* are Utils.motion_eval sums/counts; local uses root-frame "
            "positions, world integrates root motion; traj slot is a zero block in world reconstruction"
        ),
    )
    return payload


def git_commit() -> dict:
    import subprocess

    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return dict(
        commit=run("git", "rev-parse", "HEAD"),
        branch=run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        dirty=bool(run("git", "status", "--porcelain")),
    )


def environment_report() -> dict:
    import importlib.metadata as md

    try:
        vq_version = md.version("vector-quantize-pytorch")
    except md.PackageNotFoundError:
        vq_version = None
    return dict(
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        vector_quantize_pytorch=vq_version,
        fp32=True, tf32=False, matmul_precision="highest",
    )


def key_metric(payload: dict, key: str) -> float:
    """Canonical scalar values used by S4 gates (single source of truth for names)."""
    agg = payload["aggregate"]
    table = {
        "d2r_local_mean": agg["d2r_local"]["per_source_mean"],
        "d2r_local_p95": agg["d2r_local"]["per_source_p95"],
        "d2r_world_mean": agg["d2r_world"]["per_source_mean"],
        "d2e_local_mean": agg["d2e_local"]["per_source_mean"],
        "d2e_world_mean": agg["d2e_world"]["per_source_mean"],
        "jitter_quant_local": agg["official_q"]["jitter"],
        "jitter_cont_local": agg["official_c"]["jitter"],
        "jitter_gt_local": agg["official_gt"]["jitter"],
        "mpjpe_q": agg["official_q"]["mpjpe"],
        "local_mpjpe_q": agg["official_q"]["local_mpjpe"],
        "root_ade_q": agg["official_q"]["root_ade"],
        "fs_local_q": agg["official_q"]["fs_local"],
        "fs_world_q": agg["official_q"]["fs_world"],
    }
    return float(table[key])


def per_source_metric(payload: dict, name: str) -> np.ndarray:
    """One scalar per source for paired bootstrap (means/sums, never pooled)."""
    per_source = payload["per_source"]
    if name == "d2r_local_mean":
        return np.asarray([c["d2r_local"]["mean"] for c in per_source])
    if name == "d2r_local_p95":
        return np.asarray([c["d2r_local"]["p95"] for c in per_source])
    if name == "d2r_world_mean":
        return np.asarray([c["d2r_world"]["mean"] for c in per_source])
    if name == "d2e_local_mean":
        return np.asarray([c["d2e_local"]["mean"] for c in per_source])
    if name == "d2e_world_mean":
        return np.asarray([c["d2e_world"]["mean"] for c in per_source])
    if name == "jitter_quant_local":
        return np.asarray([c["official_q"]["jitter_sum_cm"] / c["official_q"]["jitter_count"] for c in per_source])
    if name == "jitter_cont_local":
        return np.asarray([c["official_c"]["jitter_sum_cm"] / c["official_c"]["jitter_count"] for c in per_source])
    if name == "mpjpe_q":
        return np.asarray([c["official_q"]["mpjpe_sum_cm"] / c["official_q"]["mpjpe_count"] for c in per_source])
    if name == "local_mpjpe_q":
        return np.asarray([c["official_q"]["local_mpjpe_sum_cm"] / c["official_q"]["local_mpjpe_count"] for c in per_source])
    if name == "root_ade_q":
        return np.asarray([c["official_q"]["root_ade_sum_cm"] / c["official_q"]["root_ade_count"] for c in per_source])
    if name == "fs_local_q":
        counts = np.asarray([c["official_q"]["fs_local_count"] for c in per_source], dtype=np.float64)
        sums = np.asarray([c["official_q"]["fs_local_sum_cm"] for c in per_source])
        return np.divide(sums, np.maximum(counts, 1.0))
    if name == "fs_world_q":
        counts = np.asarray([c["official_q"]["fs_world_count"] for c in per_source], dtype=np.float64)
        sums = np.asarray([c["official_q"]["fs_world_sum_cm"] for c in per_source])
        return np.divide(sums, np.maximum(counts, 1.0))
    if name.startswith("activity_q."):
        sub = name.split(".", 1)[1]
        return np.asarray([c["activity_q"][sub] for c in per_source])
    if name.startswith("activity_gt."):
        sub = name.split(".", 1)[1]
        return np.asarray([c["activity_gt"][sub] for c in per_source])
    if name == "pred_contact_coverage":
        return np.asarray([c["contact"]["pred_contact_coverage"] for c in per_source])
    if name == "style_correct":
        return np.asarray([c["correct"] for c in payload["style_proxy"]["per_source"]], dtype=np.float64)
    raise KeyError(f"unknown per-source metric: {name}")
