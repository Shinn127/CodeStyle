"""
Representation-layer evaluation for MotionVQ (the VQ-VAE) during training.

Reuses `Utils.motion_eval.compute_sequence_metric_sums`, which is the same code path the
offline evaluation uses, so the numbers reported here are directly comparable with the
paper's MPJPE / Jitter / F.S. tables. No metric is reimplemented here.

What is measured: teacher-forced reconstruction of whole clips from the validation split.
Every frame is encoded and decoded once (the trajectory block is kept from the input,
because MotionVQ is trained to reconstruct the 291-dim pose block only), the normalized
sequence is handed to the shared metric code, and the metrics are aggregated over clips.

The evaluation set is chosen deterministically from `seed`, so the same clips are scored
at every checkpoint and "best" comparisons across epochs are meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from Utils.motion_eval import MotionEvalConfig, compute_sequence_metric_sums, denormalize_motion_batch
from Utils.motion375 import reconstruct_selected_globals_from_375
from Utils.runtime import MEMMAP_X_FILE
from dataset import MEMMAP_SEGMENTS_FILE


@dataclass(frozen=True)
class VQVAEEvalConfig:
    clips: int = 90
    window_frames: int = 60
    windows_per_clip: int = 1
    min_frames_per_clip: int = 60
    split: str = "val"
    seed: int = 42
    batch_windows: int = 512
    motion_cfg: MotionEvalConfig = MotionEvalConfig()


def select_eval_windows(
    db_path: str | Path,
    *,
    split: str,
    clips: int,
    window_frames: int,
    windows_per_clip: int,
    min_frames: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pick deterministic `window_frames`-long windows from one split.

    Whole source clips are the unit of selection (mirror variants collapse onto their
    source clip so one motion is never counted twice), and each selected clip contributes
    `windows_per_clip` windows drawn from random offsets inside the clip. Sampling inside
    the clip rather than always taking the prefix avoids measuring only the idle opening
    frames of every motion.

    Keeping the window at ~60 frames matters for comparability: the shared metric
    integrates the reconstructed root velocity, so a longer window turns MPJPE into a
    measure of accumulated root drift instead of reconstruction quality.
    """
    db_path = Path(db_path).expanduser().resolve()
    segments_path = db_path / MEMMAP_SEGMENTS_FILE
    if not segments_path.exists():
        raise FileNotFoundError(f"{segments_path} not found; metric evaluation needs offline splits")
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    grouped: dict[int, list[dict[str, object]]] = {}
    for segment in segments:
        if str(segment.get("split", "")) != split:
            continue
        key = int(segment.get("source_entry_id", len(grouped)))
        grouped.setdefault(key, []).append(segment)
    if not grouped:
        raise RuntimeError(f"No segments with split={split!r} in {segments_path}")

    required = max(min_frames, window_frames)
    sources: list[tuple[int, int]] = []
    for group in grouped.values():
        unmirrored = [s for s in group if not s.get("mirror")] or group
        for segment in unmirrored:
            frames = int(segment["num_frames"])
            if frames >= required:
                sources.append((int(segment["frame_start"]), frames))
    sources.sort()
    if not sources:
        raise RuntimeError(
            f"No {split} clip is long enough for a {window_frames}-frame window "
            f"(need >= {required} frames)"
        )

    rng = np.random.default_rng(seed)
    if len(sources) > clips:
        picked = np.sort(rng.choice(len(sources), size=clips, replace=False))
        sources = [sources[int(idx)] for idx in picked]
    if windows_per_clip < 1:
        raise ValueError(f"windows_per_clip must be >= 1, got {windows_per_clip}")

    windows: list[tuple[int, int]] = []
    for frame_start, frames in sources:
        slack = frames - window_frames
        if slack == 0:
            offsets = [0] * windows_per_clip
        else:
            offsets = [int(offset) for offset in rng.integers(0, slack + 1, size=windows_per_clip)]
        for offset in offsets:
            windows.append((frame_start + offset, window_frames))
    return windows


@torch.no_grad()
def reconstruct_clips(
    model,
    db_path: str | Path,
    spans: list[tuple[int, int]],
    *,
    num_frames_total: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    input_dim: int = 375,
    traj_dim: int = 84,
    batch_windows: int = 512,
) -> list[dict[str, object]]:
    """Encode/decode the requested clips and return normalized reference/prediction pairs.

    Frame t is reconstructed as the first frame of the window that starts at t; the final
    frame of each clip comes from the second slot of the last window, so every frame is
    scored exactly once.
    """
    x_source = np.memmap(
        Path(db_path).expanduser().resolve() / MEMMAP_X_FILE,
        dtype=np.float32,
        mode="r",
        shape=(num_frames_total, input_dim),
    )
    mean = np.asarray(norm_mean, dtype=np.float32)
    std = np.asarray(norm_std, dtype=np.float32)
    motion_dim = input_dim - traj_dim
    results: list[dict[str, object]] = []

    def decode_windows(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode+decode a [B,2,375] window batch; return motion [B,2,291] and codes [B,2,C]."""
        block = torch.from_numpy(windows).to(device)
        latents, indices, _ = model.encode(block)
        motion_hat = model.decode(latents)
        return (
            motion_hat.float().cpu().numpy(),
            indices.cpu().numpy() if indices is not None else np.empty((0,), dtype=np.int64),
        )

    for frame_start, num_frames in spans:
        raw = np.array(x_source[frame_start : frame_start + num_frames], dtype=np.float32, copy=True)
        if raw.shape[0] < 2:
            continue
        norm = (raw - mean) / std

        windows = np.stack([norm[:-1], norm[1:]], axis=1)  # [N-1, 2, 375]
        recon_motion = np.empty((norm.shape[0], motion_dim), dtype=np.float32)
        codes: Optional[np.ndarray] = None
        for begin in range(0, windows.shape[0], batch_windows):
            chunk = windows[begin : begin + batch_windows]
            motion_hat, indices = decode_windows(chunk)
            recon_motion[begin : begin + chunk.shape[0]] = motion_hat[:, 0]
            if indices.ndim == 3:
                if codes is None:
                    # indices is [B, T, num_codebooks]; one code row per frame.
                    codes = np.empty((norm.shape[0], indices.shape[2]), dtype=np.int64)
                codes[begin : begin + chunk.shape[0]] = indices[:, 0]
        # Last frame comes from the second slot of the final window.
        motion_hat, indices = decode_windows(windows[-1:])
        recon_motion[-1] = motion_hat[0, 1]
        if codes is not None and indices.ndim == 3:
            codes[-1] = indices[0, 1]

        prediction = np.array(norm, copy=True)
        prediction[:, traj_dim:] = recon_motion
        results.append(
            {
                "reference": norm,
                "prediction": prediction,
                "codes": codes if codes is not None else np.empty((0,), dtype=np.int64),
                "frame_start": frame_start,
                "num_frames": int(norm.shape[0]),
            }
        )
    return results


def evaluate_vqvae(
    model,
    *,
    db_path: str | Path,
    num_frames_total: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    config: VQVAEEvalConfig,
    num_embeddings: int,
    traj_dim: int = 84,
) -> dict[str, float]:
    """Run the representation-layer evaluation and return the metric dict.

    Metric names: `local_mpjpe_cm` is root-relative joint error and sits on the same scale as
    the paper's MotionVQ MPJPE (3.653 cm for $32\\times4$, K=2048), because it does not
    accumulate root drift across the clip. `mpjpe_cm` does integrate the reconstructed root
    velocity, so on long clips it is dominated by drift and is an order of magnitude larger;
    it is reported for completeness and for parity with rollout-style evaluation.
    `jitter_cm` and `foot_sliding_cm` come straight from the shared metric code.
    `codebook_usage` reports the fraction of codebook entries used.

    FID is deliberately not computed here: the paper's FID runs frames through a trained
    feature extractor, so reporting a Fréchet distance on raw pose frames would be a
    different quantity under the same name. Use `evaluate_motion_375to291.py`, which wires
    the extractor and the bootstrap CI, when FID is needed.
    """
    spans = select_eval_windows(
        db_path,
        split=config.split,
        clips=config.clips,
        window_frames=config.window_frames,
        windows_per_clip=config.windows_per_clip,
        min_frames=config.min_frames_per_clip,
        seed=config.seed,
    )
    if not spans:
        raise RuntimeError("Metric evaluation selected zero windows")

    results = reconstruct_clips(
        model,
        db_path,
        spans,
        num_frames_total=num_frames_total,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        traj_dim=traj_dim,
        batch_windows=config.batch_windows,
    )
    if not results:
        raise RuntimeError("Metric evaluation produced zero reconstructable clips")

    sums = {
        "mpjpe": 0.0,
        "local_mpjpe": 0.0,
        "root_ade": 0.0,
        "heading": 0.0,
        "velocity_dir": 0.0,
        "foot": 0.0,
        "jitter": 0.0,
    }
    counts = {key: 0 for key in sums}
    used_codes: set[int] = set()
    total_code_slots = 0

    for item in results:
        metric = compute_sequence_metric_sums(
            reference_375=item["reference"],
            prediction_375=item["prediction"],
            cfg=config.motion_cfg,
            motion_mean=norm_mean,
            motion_std=norm_std,
        )
        sums["mpjpe"] += metric.mpjpe_sum_cm
        counts["mpjpe"] += metric.mpjpe_count

        # `mpjpe` integrates the reconstructed root velocity over the whole clip, so on long
        # clips it mostly measures root drift. This diagnostic skips the integration and
        # reports root-relative joint error, which isolates pose reconstruction quality.
        ref_local, _ = reconstruct_selected_globals_from_375(
            denormalize_motion_batch(item["reference"], mean=norm_mean, std=norm_std),
            dt=config.motion_cfg.dt,
            root_wy_in_degrees=True,
            use_global=False,
        )
        pred_local, _ = reconstruct_selected_globals_from_375(
            denormalize_motion_batch(item["prediction"], mean=norm_mean, std=norm_std),
            dt=config.motion_cfg.dt,
            root_wy_in_degrees=True,
            use_global=False,
        )
        sums["local_mpjpe"] += float(np.linalg.norm(pred_local - ref_local, axis=-1).sum() * 100.0)
        counts["local_mpjpe"] += int(pred_local.shape[0] * pred_local.shape[1])

        sums["root_ade"] += metric.root_ade_sum_cm
        counts["root_ade"] += metric.root_ade_count
        sums["heading"] += metric.heading_error_sum_deg
        counts["heading"] += metric.heading_error_count
        sums["velocity_dir"] += metric.velocity_dir_error_sum_deg
        counts["velocity_dir"] += metric.velocity_dir_error_count
        sums["foot"] += metric.foot_sliding_sum_cm
        counts["foot"] += metric.foot_sliding_count
        sums["jitter"] += metric.jitter_sum_cm
        counts["jitter"] += metric.jitter_count

        codes = np.asarray(item["codes"])
        if codes.size:
            used_codes.update(np.unique(codes).tolist())
            total_code_slots += int(codes.shape[0] * codes.shape[1])

    def mean_of(key: str) -> float:
        return float(sums[key] / counts[key]) if counts[key] else float("nan")

    metrics = {
        "mpjpe_cm": mean_of("mpjpe"),
        "local_mpjpe_cm": mean_of("local_mpjpe"),
        "jitter_cm": mean_of("jitter"),
        "foot_sliding_cm": mean_of("foot"),
        "root_ade_cm": mean_of("root_ade"),
        "heading_error_deg": mean_of("heading"),
        "velocity_dir_error_deg": mean_of("velocity_dir"),
        "codebook_usage": float(len(used_codes) / max(num_embeddings, 1)),
        "eval_clips": float(len(results)),
        "eval_frames": float(sum(int(item["num_frames"]) for item in results)),
    }
    return metrics


def eval_score(metrics: dict[str, float], metric_name: str) -> Optional[float]:
    """Map an --eval-metric choice onto a scalar; lower is better."""
    if metric_name in {"none", "off", ""}:
        return None
    if metric_name == "mpjpe":
        return float(metrics["mpjpe_cm"])
    if metric_name == "local_mpjpe":
        return float(metrics["local_mpjpe_cm"])
    if metric_name == "jitter":
        return float(metrics["jitter_cm"])
    if metric_name == "mpjpe_jitter":
        return float(metrics["mpjpe_cm"] + metrics["jitter_cm"])
    raise ValueError(f"Unsupported eval metric: {metric_name!r}")


# --- best-checkpoint gate ---------------------------------------------------------------
# Canonical gate metric name -> key returned by `evaluate_vqvae`. The four representation
# metrics the paper reports are all in cm, so one absolute `min_delta` is meaningful across
# them.
BEST_METRIC_KEYS: dict[str, str] = {
    "mpjpe": "mpjpe_cm",
    "local_mpjpe": "local_mpjpe_cm",
    "jitter": "jitter_cm",
    "foot_sliding": "foot_sliding_cm",
    "root_ade": "root_ade_cm",
    "heading_error": "heading_error_deg",
    "velocity_dir_error": "velocity_dir_error_deg",
    "codebook_usage": "codebook_usage",
}
_BEST_METRIC_ALIASES: dict[str, str] = {
    "pose_mpjpe": "local_mpjpe",
    "fs": "foot_sliding",
    "foot_slide": "foot_sliding",
    "rootade": "root_ade",
    "root_ade_error": "root_ade",
    "usage": "codebook_usage",
}
# Everything else is lower-is-better.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"codebook_usage"})
DEFAULT_BEST_METRICS: tuple[str, ...] = ("mpjpe", "jitter", "foot_sliding", "root_ade")
_DISABLED_SPEC_TOKENS = frozenset({"", "none", "off", "no", "disable", "disabled"})


def normalize_best_metric(name: str) -> str:
    """Canonicalise one --best-metrics entry onto a key of `BEST_METRIC_KEYS`."""
    key = str(name).strip().lower().replace("-", "_")
    key = _BEST_METRIC_ALIASES.get(key, key)
    if key not in BEST_METRIC_KEYS:
        raise ValueError(
            f"Unsupported best-checkpoint metric: {name!r}; known metrics: {sorted(BEST_METRIC_KEYS)}"
        )
    return key


def resolve_best_metrics(
    spec: str | Sequence[str],
    *,
    primary: Optional[str] = None,
) -> tuple[str, ...]:
    """Parse a --best-metrics spec into canonical names, deduplicated and order-preserving.

    `primary` (the --eval-metric choice) is appended when it names a real metric, so the
    scalar reported as `best_metric` always moves together with the gate. Non-metric
    choices (`none`, `loss`) are ignored, and an empty/disabled spec falls back to
    `DEFAULT_BEST_METRICS` -- except that a usable `primary` is then the whole gate, which
    reproduces the plain single-metric rule.
    """
    raw = spec.split(",") if isinstance(spec, str) else [str(part) for part in spec]
    tokens = [token.strip() for token in raw]
    names: list[str] = []
    for token in tokens:
        if token.lower() in _DISABLED_SPEC_TOKENS:
            continue
        name = normalize_best_metric(token)
        if name not in names:
            names.append(name)
    if not names:
        if primary is not None and str(primary).strip().lower() in BEST_METRIC_KEYS:
            names.append(normalize_best_metric(primary))
        else:
            names.extend(DEFAULT_BEST_METRICS)
    if primary is not None and str(primary).strip().lower() in BEST_METRIC_KEYS:
        primary_name = normalize_best_metric(primary)
        if primary_name not in names:
            names.append(primary_name)
    return tuple(names)


@dataclass(frozen=True)
class BestGateResult:
    """Outcome of the "every tracked metric improved" check."""

    accept: bool
    values: dict[str, float]
    reference: dict[str, float]
    improvements: dict[str, tuple[float, float]]
    blockers: tuple[str, ...]
    reason: str

    def describe(self) -> str:
        if self.reason == "first_eval":
            return "first evaluation (no reference checkpoint yet)"
        if self.accept:
            return ", ".join(
                f"{name} {ref:.4f}->{value:.4f}"
                for name, (ref, value) in self.improvements.items()
            )
        return ", ".join(
            f"{name} {self.reference.get(name, float('nan')):.4f}->{self.values.get(name, float('nan')):.4f}"
            for name in self.blockers
        )


def check_best_gate(
    metrics: dict[str, float],
    names: Sequence[str],
    *,
    reference: Optional[dict[str, float]] = None,
    min_delta: float = 0.0,
) -> BestGateResult:
    """Decide whether `metrics` may replace the current best checkpoint.

    Every metric in `names` must beat its value in `reference` -- not just the primary one --
    so `best.pt` only ever moves to a checkpoint that is better across the board. With
    `reference=None` (first evaluation) the checkpoint is always accepted. A metric that is
    missing from `reference` cannot block (e.g. after the tracked set changed), and a
    non-finite current value always blocks: a broken metric must not be saved as best.
    """
    if not names:
        raise ValueError("check_best_gate needs at least one metric name")
    values: dict[str, float] = {}
    for name in names:
        key = BEST_METRIC_KEYS[normalize_best_metric(name)]
        if key not in metrics:
            raise KeyError(f"Metric {name!r} is not part of the evaluation: available keys {sorted(metrics)}")
        values[normalize_best_metric(name)] = float(metrics[key])

    if reference is None:
        return BestGateResult(
            accept=True,
            values=values,
            reference={},
            improvements={name: (float("nan"), value) for name, value in values.items()},
            blockers=(),
            reason="first_eval",
        )

    ref_clean = {str(key): float(val) for key, val in reference.items()}
    improvements: dict[str, tuple[float, float]] = {}
    blockers: list[str] = []
    for name, value in values.items():
        if name not in ref_clean or not np.isfinite(ref_clean[name]):
            improvements[name] = (float("nan"), value)
            continue
        ref = ref_clean[name]
        if not np.isfinite(value):
            blockers.append(name)
            continue
        better = (
            value > ref + min_delta if name in HIGHER_IS_BETTER else value < ref - min_delta
        )
        if better:
            improvements[name] = (ref, value)
        else:
            blockers.append(name)
    return BestGateResult(
        accept=not blockers,
        values=values,
        reference=ref_clean,
        improvements=improvements,
        blockers=tuple(blockers),
        reason="all_improved" if not blockers else "blocked",
    )


__all__ = [
    "BEST_METRIC_KEYS",
    "BestGateResult",
    "DEFAULT_BEST_METRICS",
    "VQVAEEvalConfig",
    "check_best_gate",
    "eval_score",
    "evaluate_vqvae",
    "normalize_best_metric",
    "reconstruct_clips",
    "resolve_best_metrics",
    "select_eval_windows",
]
