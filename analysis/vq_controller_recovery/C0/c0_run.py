"""C0 runner: closed-loop causal decomposition + single-perturbation recovery (read-only).

Produces analysis/vq_controller_recovery/C0/{manifest.json,per_source.json,REPORT.md}.
No training, no parameter updates, VQ/Controller fully frozen (state fingerprints verified).

  python analysis/vq_controller_recovery/C0/c0_run.py [--sources N] [--skip-perturb] [--device cuda]
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
sys.path.insert(0, str(HERE))
import loop_lib as lib  # noqa: E402

from Utils.motion_eval import MotionEvalConfig  # noqa: E402
from Utils.reproducibility import state_fingerprint  # noqa: E402

GROUPS = ("A", "B", "C", "D", "E")


def aligned_jitter(local_cm: np.ndarray) -> np.ndarray:
    """Per-frame mean joint Jitter (cm) aligned to frames 1..240 (first two frames NaN)."""
    values = lib.vt.jitter_values_cm(local_cm)          # (T-2)*24
    per_frame = values.reshape(-1, local_cm.shape[1]).mean(-1)   # T-2 frame values
    out = np.full(local_cm.shape[0], np.nan)
    out[2:] = per_frame
    return out


def contact_coverage(local_cm: np.ndarray) -> float:
    from Utils.motion_eval import _toe_contact_mask_from_reference
    mask = _toe_contact_mask_from_reference(
        (local_cm / 100.0).astype(np.float32), dt=1 / 60.0,
        contact_height_cm=20.0, contact_speed_cm_s=20.0)
    return float(mask.mean()) if mask.size else float("nan")


def bucket_means(values: np.ndarray) -> dict:
    out = {}
    for lo, hi in lib.BUCKETS:
        window = values[lo - 1:hi]
        out[f"{lo}-{hi}"] = float(np.nanmean(window)) if np.isfinite(window).any() else None
    return out


def group_metrics(*, model, gen_raw, gt_raw, mean, std, cfg, latents_gen, latents_gt,
                  style: dict | None, gt_referenced: bool) -> dict:
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32)

    def local_world(raw: np.ndarray):
        motion_norm = torch.as_tensor((raw[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
        return lib.vt.positions_from_motion_norm(motion_norm, mean_t, std_t)

    lp, wp = local_world(gen_raw)
    entry: dict = dict(frames=int(gen_raw.shape[0]))
    entry["jitter"] = float(np.nanmean(aligned_jitter(lp)))
    entry["jitter_frame"] = [None if not np.isfinite(v) else round(float(v), 4) for v in aligned_jitter(lp)]
    entry["jitter_buckets"] = bucket_means(aligned_jitter(lp))
    entry["activity"] = lib.vt.activity_proxies(lp, cfg)
    entry["contact_coverage_pred"] = contact_coverage(lp)
    if style is not None:
        entry["style"] = style
    if not gt_referenced:
        entry["gt_referenced"] = None
        return entry

    lg, wg = local_world(gt_raw)
    sum_err, count_err = lib.vt.local_mpjpe_sum_count(lp, lg)
    per_frame_err = np.linalg.norm(lp - lg, axis=-1).mean(-1)
    entry.update(
        local_mpjpe=float(sum_err / count_err),
        local_mpjpe_frame=[round(float(v), 4) for v in per_frame_err],
        local_mpjpe_buckets=bucket_means(per_frame_err),
        world_local_mpjpe=None,   # filled from official metrics below (world integrates root)
        official=lib.official_metrics(gt_raw, gen_raw, cfg),
    )
    gen_pos, gen_vel = lib.root_velocity_series(gen_raw)
    gt_pos, gt_vel = lib.root_velocity_series(gt_raw)
    vel_err = lib.velocity_dir_error_deg(gen_vel, gt_vel)
    entry["vel_dir_err_deg"] = float(np.nanmean(vel_err))
    entry["vel_dir_err_deg_frame"] = [None if not np.isfinite(v) else round(float(v), 3) for v in vel_err]
    entry["vel_dir_err_deg_buckets"] = bucket_means(vel_err)
    entry["root_drift_cm"] = float(np.linalg.norm(gen_pos[:, [0, 2]] - gt_pos[:, [0, 2]], axis=-1)[-1] * 100)
    entry["activity_gt"] = lib.vt.activity_proxies(lg, cfg)
    idx_gen = lib.code_indices(model, latents_gen[1:])
    idx_gt = lib.code_indices(model, latents_gt[1:])
    match = (idx_gen == idx_gt).float()
    entry["code_match"] = float(match.mean())
    entry["code_match_buckets"] = bucket_means(match.mean(-1).cpu().numpy())
    entry["code_switch_gen_per_slot"] = float((idx_gen[1:] != idx_gen[:-1]).float().mean())
    entry["code_switch_gt_per_slot"] = float((idx_gt[1:] != idx_gt[:-1]).float().mean())
    if style is not None:
        entry["style"] = style
    return entry


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.cuda() if torch.cuda.is_available() else t


def perturbation_metrics(*, perturbed_raw, clean_raw, gt_raw, t0: int, cfg) -> dict:
    mean_t = torch.as_tensor(lib.MEAN, dtype=torch.float32)
    std_t = torch.as_tensor(lib.STD, dtype=torch.float32)

    def local(raw):
        motion_norm = torch.as_tensor((raw[1:, 84:] - lib.MEAN[84:]) / lib.STD[84:], dtype=torch.float32)
        return lib.vt.positions_from_motion_norm(motion_norm, mean_t, std_t)[0]

    lp_p, lp_c, lp_g = local(perturbed_raw), local(clean_raw), local(gt_raw)
    pos_p, vel_p = lib.root_velocity_series(perturbed_raw)
    pos_c, vel_c = lib.root_velocity_series(clean_raw)
    _, vel_g = lib.root_velocity_series(gt_raw)
    drift = np.linalg.norm(pos_p[:, [0, 2]] - pos_c[:, [0, 2]], axis=-1) * 100.0     # [240]
    dir_pc = lib.velocity_dir_error_deg(vel_p, vel_c, min_speed_cm_s=0.0)
    dir_pg = lib.velocity_dir_error_deg(vel_p, vel_g)

    out = dict(t0=int(t0))
    for horizon in (8, 16, 32):
        end = min(t0 + horizon, len(drift))
        window = slice(t0, end)          # transitions t0 .. t0+horizon-1
        pose_step = np.linalg.norm(lp_p[1:] - lp_p[:-1], axis=-1).mean(-1)
        pose_step_clean = np.linalg.norm(lp_c[1:] - lp_c[:-1], axis=-1).mean(-1)
        out[f"+{horizon}"] = dict(
            root_deviation_cm=float(drift[end - 1]),
            root_deviation_window_mean_cm=float(np.nanmean(drift[window])),
            vel_dir_err_vs_clean_deg=float(np.nanmean(dir_pc[window])),
            vel_dir_err_vs_gt_deg=float(np.nanmean(dir_pg[window])),
            jitter=float(np.nanmean(aligned_jitter(lp_p)[t0:end])),
            jitter_clean=float(np.nanmean(aligned_jitter(lp_c)[t0:end])),
            pose_step_cm=float(pose_step[t0:end].mean()),
            pose_step_clean_cm=float(pose_step_clean[t0:end].mean()),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=12)
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=HERE)
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
    direction = lib.perturb_direction(model, device)

    picks = lib.vt.fixed_val12()[:max(1, args.sources)]
    _, _, x_source = lib.vt.read_db()
    per_source = []
    for pick in picks:
        start = int(pick["frame_start"])
        label = int(pick["label"])
        rows = np.array(x_source[start:start + lib.T_FRAMES], dtype=np.float32, copy=True)
        x_norm = torch.as_tensor((rows - mean) / std, device=device)
        style_feature = lib.style_feature_for(model, x_norm[0:1], label, device)
        latents_gt = lib.gt_latents(model, x_norm)
        z0_plan = lib.z0_plan_for(model, start, label, device)
        entry = dict(source_entry_id=int(pick["source_entry_id"]), style=pick["style_name"],
                     suffix=pick["suffix"], group=lib.vt.group_of(pick["suffix"]),
                     frame_start=start, label=label, groups={})

        gen_raw = {}
        gen_lat = {}
        for name, runner in (
            ("A", lambda: lib.run_teacher_anchored(
                model, x_norm=x_norm, latents_gt=latents_gt, style_feature=style_feature,
                z0_plan=z0_plan, mode="raw")),
            ("B", lambda: lib.run_teacher_anchored(
                model, x_norm=x_norm, latents_gt=latents_gt, style_feature=style_feature,
                z0_plan=z0_plan, mode="decoded")),
            ("C", lambda: lib.run_sequence(
                model, x_norm=x_norm, label=label, style_feature=style_feature,
                z0_plan=z0_plan, mode="official")),
            ("D", lambda: lib.run_sequence(
                model, x_norm=x_norm, label=label, style_feature=style_feature,
                z0_plan=z0_plan, mode="planner", planner=lib.InteractiveTrajectory(window=60, stride=10))),
            ("E", lambda: lib.run_sequence(
                model, x_norm=x_norm, label=label, style_feature=style_feature,
                z0_plan=z0_plan, mode="continuous")),
        ):
            if name in ("A", "B"):
                out_norm, lat_pred, diag = runner()
                raw = out_norm.detach().cpu().numpy() * std[None, :] + mean[None, :]
                lat = lat_pred.detach().cpu()
            else:
                raw_t, lat_t, diag = runner()
                raw = raw_t.numpy()
                lat = lat_t
            gen_raw[name] = raw
            gen_lat[name] = lat
            style = style_clf.classify(raw, label)
            info = group_metrics(
                model=model, gen_raw=raw, gt_raw=rows, mean=mean, std=std, cfg=cfg,
                latents_gen=lat,
                latents_gt=latents_gt.detach().cpu(),
                style=style, gt_referenced=name in ("A", "B", "C", "E"))
            info["n_transitions"] = int(raw.shape[0] - 1)
            info["invariant_violations"] = int(diag["invariant_violations"])
            if name == "D":
                speed = [v for v in diag["planner_speed"]]
                derr = [v for v in diag["planner_dir_err_deg"] if v is not None]
                info["planner"] = dict(speed_mean=float(np.mean(speed)), dir_err_deg_mean=float(np.mean(derr)),
                                       dir_err_deg_p95=float(np.percentile(derr, 95)))
            entry["groups"][name] = info
        entry["groups"]["D"]["note"] = "planner-conditioned; GT-referenced metrics are NOT measured by design"

        if not args.skip_perturb:
            perturbed_raw, _, diag = lib.run_sequence(
                model, x_norm=x_norm, label=label, style_feature=style_feature,
                z0_plan=z0_plan, mode="official",
                perturb=(direction, lib.PERTURB_AMPLITUDE, lib.T0_PERTURB))
            entry["perturbation"] = perturbation_metrics(
                perturbed_raw=perturbed_raw.numpy(), clean_raw=gen_raw["C"], gt_raw=rows,
                t0=lib.T0_PERTURB, cfg=cfg)
            entry["perturbation"]["invariant_violations"] = int(diag["invariant_violations"])
            for key in ("perturbation_l2_raw", "perturbation_l2_after_clip",
                        "perturbation_slots_changed", "perturbation_slots_total"):
                entry["perturbation"][key] = diag.get(key)
        per_source.append(entry)
        print(f"[C0] {pick['style_name']:<16s} done ({time.time()-started:.0f}s)")

    after = state_fingerprint(model.state_dict())
    vq_after = state_fingerprint(model.vqvae.state_dict())
    if before != after or vq_before != vq_after:
        raise RuntimeError("C0 mutated model or VQ state; aborting (INVALID_EXPERIMENT)")

    def mean_scalar(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else None

    # Aggregates over sources
    def agg(getter):
        values = [getter(p) for p in per_source]
        values = [v for v in values if v is not None and np.isfinite(v)]
        return dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values)) if values else None

    summary = {}
    for name in GROUPS:
        if name == "D":
            summary[name] = dict(
                jitter=agg(lambda p, n=name: p["groups"][n]["jitter"]),
                activity_moving=agg(lambda p, n=name: p["groups"][n]["activity"]["moving_frame_fraction"]),
                activity_strong=agg(lambda p, n=name: p["groups"][n]["activity"]["strong_moving_frame_fraction"]),
                contact_coverage=agg(lambda p, n=name: p["groups"][n]["contact_coverage_pred"]),
                style_accuracy=float(np.mean([p["groups"][name]["style"]["correct"] for p in per_source])),
                planner_dir_err_deg=agg(lambda p, n=name: p["groups"][n]["planner"]["dir_err_deg_mean"]),
                planner_speed=agg(lambda p, n=name: p["groups"][n]["planner"]["speed_mean"]),
            )
            continue
        summary[name] = dict(
            local_mpjpe=agg(lambda p, n=name: p["groups"][n]["local_mpjpe"]),
            jitter=agg(lambda p, n=name: p["groups"][n]["jitter"]),
            world_mpjpe=agg(lambda p, n=name: p["groups"][n]["official"]["mpjpe"]),
            root_ade=agg(lambda p, n=name: p["groups"][n]["official"]["root_ade"]),
            fs_local=agg(lambda p, n=name: p["groups"][n]["official"]["fs_local"]),
            fs_world=agg(lambda p, n=name: p["groups"][n]["official"]["fs_world"]),
            vel_dir_err_deg=agg(lambda p, n=name: p["groups"][n]["vel_dir_err_deg"]),
            code_match=agg(lambda p, n=name: p["groups"][n]["code_match"]),
            activity_moving=agg(lambda p, n=name: p["groups"][n]["activity"]["moving_frame_fraction"]),
            activity_strong=agg(lambda p, n=name: p["groups"][n]["activity"]["strong_moving_frame_fraction"]),
            contact_coverage=agg(lambda p, n=name: p["groups"][n]["contact_coverage_pred"]),
            style_accuracy=float(np.mean([p["groups"][name]["style"]["correct"] for p in per_source])),
            gt_style_ref=agg(lambda p, n=name: p["groups"][n]["activity_gt"]["moving_frame_fraction"]),
        )
    perturb_summary = None
    if not args.skip_perturb:
        perturb_summary = dict(absorption=dict(
            l2_raw=mean_scalar([p["perturbation"]["perturbation_l2_raw"] for p in per_source]),
            l2_after_clip=mean_scalar([p["perturbation"]["perturbation_l2_after_clip"] for p in per_source]),
            slots_changed=mean_scalar([p["perturbation"]["perturbation_slots_changed"] for p in per_source]),
            slots_total=int(per_source[0]["perturbation"]["perturbation_slots_total"]),
        ))
        for horizon in (8, 16, 32):
            key = f"+{horizon}"
            perturb_summary[key] = dict(
                root_deviation_cm=agg(lambda p, k=key: p["perturbation"][k]["root_deviation_cm"]),
                vel_dir_err_vs_clean_deg=agg(lambda p, k=key: p["perturbation"][k]["vel_dir_err_vs_clean_deg"]),
                vel_dir_err_vs_gt_deg=agg(lambda p, k=key: p["perturbation"][k]["vel_dir_err_vs_gt_deg"]),
                jitter=agg(lambda p, k=key: p["perturbation"][k]["jitter"]),
                jitter_clean=agg(lambda p, k=key: p["perturbation"][k]["jitter_clean"]),
                pose_step_cm=agg(lambda p, k=key: p["perturbation"][k]["pose_step_cm"]),
                pose_step_clean_cm=agg(lambda p, k=key: p["perturbation"][k]["pose_step_clean_cm"]),
            )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        stage="C0",
        protocol=str(HERE / "PROTOCOL.md"),
        config=lib.c0_config(),
        sources=[p["source_entry_id"] for p in per_source],
        n_sources=len(per_source),
        environment=lib.vt.environment_report(),
        git=lib.vt.git_commit(),
        controller_state_sha256=before, vq_state_sha256=vq_before, state_unchanged=True,
        vq_checkpoint=str(lib.VQ / "best.pt"), controller_checkpoint=str(lib.CONTROLLER / "best.pt"),
        wall_sec=round(time.time() - started, 1),
        invariant_violations_total=int(sum(
            p["groups"][g].get("invariant_violations", 0) for p in per_source for g in GROUPS)),
        commands=[f"python {Path(__file__).relative_to(lib.ROOT)} --sources {args.sources}"
                  + (" --skip-perturb" if args.skip_perturb else "")],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out_dir / "per_source.json").write_text(json.dumps(
        dict(summary=summary, perturbation_summary=perturb_summary, per_source=per_source),
        indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1)[:3000])
    print("[Done]", out_dir)


if __name__ == "__main__":
    main()
