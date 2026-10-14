"""C1 runner: inference-only bypass sweep (read-only, weights frozen, default OFF).

Seven configurations share the C0-C protocol (GT trajectory, official feedback policy);
each changes exactly one inference knob. Per-frame latency is measured for reference.
Writes C1/{per_source.json,manifest.json} (REPORT.md is written separately after review).

  python analysis/vq_controller_recovery/C1/c1_run.py [--sources 12] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
C0 = HERE.parent / "C0"
sys.path.insert(0, str(C0))
import loop_lib as lib  # noqa: E402

from Utils.motion_eval import MotionEvalConfig  # noqa: E402
from Utils.reproducibility import state_fingerprint  # noqa: E402

# The registered sweep. `mode` follows C0-C semantics; other keys are the single variable.
CONFIGS = [
    dict(id="REF", note="C0-C baseline: requantize on, Euler 4, noise 1.0, history on"),
    dict(id="R0", requantize=False, note="requantize off (diagnostic only)"),
    dict(id="S2", steps=2, note="Euler 2 steps"),
    dict(id="S8", steps=8, note="Euler 8 steps"),
    dict(id="N0", noise_scale=0.0, note="inference noise std=0 (z0=latent_offset)"),
    dict(id="N05", noise_scale=0.5, note="inference noise std=0.5"),
    dict(id="H0", zero_history=True, note="history ablation (zero history latent; OOD diagnostic)"),
]


def run_one_config(model, config, *, x_norm, style_feature, z0_plan, device):
    """One 240-frame sequential rollout under a single-variable override."""
    requantize = config.get("requantize", True)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t0 = time.perf_counter()
    raw_t, latents, diag = lib.run_sequence(
        model, x_norm=x_norm, label=0, style_feature=style_feature, z0_plan=z0_plan,
        mode="official" if requantize else "continuous",
        steps=int(config.get("steps", 4)),
        noise_scale=float(config.get("noise_scale", 1.0)),
        zero_history=bool(config.get("zero_history", False)),
    )
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    frames = raw_t.shape[0] - 1
    return raw_t.numpy(), elapsed_ms / max(frames, 1), diag


def metrics_for(raw_np, rows, mean, std, cfg):
    """C1 metric block; identical identities to C0 (joint-frame L2 cm + official sums)."""
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32)
    motion_norm = torch.as_tensor((raw_np[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
    local, _ = lib.vt.positions_from_motion_norm(motion_norm, mean_t, std_t)
    gt_norm = torch.as_tensor((rows[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
    gt_local, _ = lib.vt.positions_from_motion_norm(gt_norm, mean_t, std_t)
    sum_err, count_err = lib.vt.local_mpjpe_sum_count(local, gt_local)
    # per-frame Jitter aligned to frame index (first two frames have no second difference)
    values = lib.vt.jitter_values_cm(local).reshape(-1, local.shape[1]).mean(-1)
    jitter = np.concatenate(([np.nan, np.nan], values))
    gen_vel = lib.root_velocity_series(raw_np)[1]
    gt_vel = lib.root_velocity_series(rows)[1]
    vel_err = lib.velocity_dir_error_deg(gen_vel, gt_vel)
    return dict(
        local_mpjpe=float(sum_err / count_err),
        jitter=float(np.nanmean(jitter)),
        official=lib.official_metrics(rows, raw_np, cfg),
        vel_dir_err_deg=float(np.nanmean(vel_err)),
        activity=lib.vt.activity_proxies(local, cfg),
        contact_coverage_pred=_contact_coverage(local),
    )


def _contact_coverage(local_cm: np.ndarray) -> float:
    from Utils.motion_eval import _toe_contact_mask_from_reference
    mask = _toe_contact_mask_from_reference(
        (local_cm / 100.0).astype(np.float32), dt=1 / 60.0,
        contact_height_cm=20.0, contact_speed_cm_s=20.0)
    return float(mask.mean()) if mask.size else float("nan")


def summarise(per_source):
    def agg(getter):
        values = [v for v in (getter(p) for p in per_source) if v is not None and np.isfinite(v)]
        return dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values)) if values else None

    return dict(
        local_mpjpe=agg(lambda p: p["local_mpjpe"]), jitter=agg(lambda p: p["jitter"]),
        world_mpjpe=agg(lambda p: p["official"]["mpjpe"]), root_ade=agg(lambda p: p["official"]["root_ade"]),
        fs_local=agg(lambda p: p["official"]["fs_local"]), fs_world=agg(lambda p: p["official"]["fs_world"]),
        vel_dir_err_deg=agg(lambda p: p["vel_dir_err_deg"]),
        moving=agg(lambda p: p["activity"]["moving_frame_fraction"]),
        strong=agg(lambda p: p["activity"]["strong_moving_frame_fraction"]),
        pose_step=agg(lambda p: p["activity"]["pose_step_mean_cm"]),
        contact_coverage=agg(lambda p: p["contact_coverage_pred"]),
        style_accuracy=float(np.mean([p["style_pred"]["correct"] for p in per_source])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--configs", type=str, default="",
                    help="comma-separated subset of ids (default: all)")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    started = time.time()

    model, mean, std = lib.load_controller(device)
    lib.set_norm(mean, std)
    before = state_fingerprint(model.state_dict())
    vq_before = state_fingerprint(model.vqvae.state_dict())
    cfg = MotionEvalConfig(metric_start_frame=1)
    style_clf = lib.StyleClassifier(device)
    picks = lib.vt.fixed_val12()[:max(1, args.sources)]
    _, _, x_source = lib.vt.read_db()

    selected = [c for c in CONFIGS if not args.configs or c["id"] in args.configs.split(",")]
    results = {"configs": []}
    for config in selected:
        per_source = []
        for pick in picks:
            start = int(pick["frame_start"])
            label = int(pick["label"])
            rows = np.array(x_source[start:start + lib.T_FRAMES], dtype=np.float32, copy=True)
            x_norm = torch.as_tensor((rows - mean) / std, device=device)
            style_feature = lib.style_feature_for(model, x_norm[0:1], label, device)
            z0_plan = lib.z0_plan_for(model, start, label, device)
            raw_np, latency_ms, diag = run_one_config(
                model, config, x_norm=x_norm, style_feature=style_feature,
                z0_plan=z0_plan, device=device)
            entry = dict(source_entry_id=int(pick["source_entry_id"]), style=pick["style_name"],
                         suffix=pick["suffix"], group=lib.vt.group_of(pick["suffix"]),
                         latency_ms_per_frame=latency_ms,
                         invariant_violations=int(diag["invariant_violations"]))
            entry.update(metrics_for(raw_np, rows, mean, std, cfg))
            entry["style_pred"] = style_clf.classify(raw_np, label)
            per_source.append(entry)
            print(f"[C1:{config['id']}] {pick['style_name']:<16s} "
                  f"lmpjpe={entry['local_mpjpe']:.2f} jitter={entry['jitter']:.3f} "
                  f"({time.time()-started:.0f}s)")
        results["configs"].append(dict(config=config, per_source=per_source, summary=summarise(per_source)))

    after = state_fingerprint(model.state_dict())
    vq_after = state_fingerprint(model.vqvae.state_dict())
    if before != after or vq_before != vq_after:
        raise RuntimeError("C1 mutated model or VQ state; aborting (INVALID_EXPERIMENT)")

    manifest = dict(
        stage="C1", protocol=str(HERE / "PROTOCOL.md"),
        configs=selected, sources=[int(p["source_entry_id"]) for p in picks], n_sources=len(picks),
        environment=lib.vt.environment_report(), git=lib.vt.git_commit(),
        controller_state_sha256=before, vq_state_sha256=vq_before, state_unchanged=True,
        invariant_violations_total=int(sum(
            e["invariant_violations"] for c in results["configs"] for e in c["per_source"])),
        wall_sec=round(time.time() - started, 1),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (args.out_dir / "per_source.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("[Done]", args.out_dir)


if __name__ == "__main__":
    main()
