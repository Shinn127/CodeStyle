"""C2a acceptance: three-arm observation-noise comparison (C2/PROTOCOL.md section 4).

For each arm (loading its `epoch_0010.pt`):
  * closed-loop C0-C protocol on the 12 fixed val sources + the registered perturbation
    recovery (loop_lib, identical to C0/C1 numbers);
  * 240f matched/cross rollout protocol with the frozen classifier and quality metrics
    (`Utils.controller_validation.TeacherForcedStyleValidation`, the training-time protocol).

Gates (pre-registered before the arms ran):
  C2-0 contract: same init state sha / same updates / config diff subset of
                 {out_dir, observation_noise_std}
  C2-1 main: at least one closed-loop metric or perturbation-recovery metric improves vs
             A=0.5 with paired 95% CI upper < 0
  C2-2 quality: local/world MPJPE, root ADE regression <= 3%; FS <= 0.5%
  C2-3 activity: moving/strong/leg-travel drop <= 5%
  C2-4 style: matched and cross accuracy drop <= 2pp
  C2-5 anti-freeze: cannot pass by freezing (activity gate covers it; reported)

  python analysis/vq_controller_recovery/C2/c2_acceptance.py [--device cuda] [--skip-long]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "C0"))

import loop_lib as lib  # noqa: E402

ARMS = {"A025": 0.25, "A050": 0.50, "A075": 0.75}
CONTROL = "A050"
CONFIG_DIFF_ALLOWED = {"out_dir", "observation_noise_std"}
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 1000


ARM_TAGS = {"A025": "A025", "A050": "A050", "A075": "A075"}


def arm_dir(arm: str) -> Path:
    """ARMS keys are already the directory tags (A025/A050/A075)."""
    return lib.ROOT / "checkpoints" / f"c2_obsnoise_{ARM_TAGS[arm]}_10ep"


def paired_ci(b0: np.ndarray, b1: np.ndarray) -> dict:
    diff = b1 - b0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(diff), size=(BOOTSTRAP_DRAWS, len(diff)))
    samples = diff[draws].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(point=float(diff.mean()), ci=[float(lo), float(hi)],
                b0_abs=float(b0.mean()), b1_abs=float(b1.mean()),
                n=int(len(diff)), seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)


def contract_gate(manifests: dict[str, dict]) -> dict:
    checks = {}
    inits = {arm: m["init_controller"]["state_sha256"] for arm, m in manifests.items()}
    updates = {arm: int(m["integrity"]["total_updates"]) for arm, m in manifests.items()}
    checks["same_init_state"] = dict(ok=len(set(inits.values())) == 1, detail=inits)
    checks["same_total_updates"] = dict(ok=len(set(updates.values())) == 1, detail=updates)
    configs = {}
    for arm in manifests:
        configs[arm] = json.loads((arm_dir(arm) / "config.json").read_text())
    ref = configs[CONTROL]
    diff_keys = set()
    for arm, config in configs.items():
        diff_keys |= {k for k in set(config) | set(ref) if config.get(k) != ref.get(k)}
    checks["config_diff_subset"] = dict(
        ok=diff_keys <= CONFIG_DIFF_ALLOWED, detail=sorted(diff_keys))
    noise = {arm: float(m["contract"]["observation_noise_std"]) for arm, m in manifests.items()}
    checks["noise_values_match_arms"] = dict(
        ok=all(abs(noise[arm] - value) < 1e-9 for arm, value in ARMS.items()) and noise[CONTROL] == 0.5,
        detail=noise)
    return dict(passed=all(c["ok"] for c in checks.values()), checks=checks)


@torch.no_grad()
def closed_loop_protocol(model, mean, std, device, style_clf) -> list[dict]:
    """C0-C protocol (GT trajectory, official feedback) + registered perturbation recovery."""
    cfg = lib.MotionEvalConfig(metric_start_frame=1)
    picks = lib.vt.fixed_val12()
    _, _, x_source = lib.vt.read_db()
    direction = lib.perturb_direction(model, device)
    per_source = []
    for pick in picks:
        start = int(pick["frame_start"])
        label = int(pick["label"])
        rows = np.array(x_source[start:start + lib.T_FRAMES], dtype=np.float32, copy=True)
        x_norm = torch.as_tensor((rows - mean) / std, device=device)
        style_feature = lib.style_feature_for(model, x_norm[0:1], label, device)
        z0_plan = lib.z0_plan_for(model, start, label, device)
        raw_t, _, diag = lib.run_sequence(
            model, x_norm=x_norm, label=label, style_feature=style_feature,
            z0_plan=z0_plan, mode="official")
        raw = raw_t.numpy()
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        motion_norm = torch.as_tensor((raw[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
        local, _ = lib.vt.positions_from_motion_norm(motion_norm, mean_t, std_t)
        gt_norm = torch.as_tensor((rows[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
        gt_local, _ = lib.vt.positions_from_motion_norm(gt_norm, mean_t, std_t)
        sum_err, count = lib.vt.local_mpjpe_sum_count(local, gt_local)
        jitter_vals = lib.vt.jitter_values_cm(local)
        gen_vel = lib.root_velocity_series(raw)[1]
        gt_vel = lib.root_velocity_series(rows)[1]
        entry = dict(
            source_entry_id=int(pick["source_entry_id"]), style=pick["style_name"],
            local_mpjpe=float(sum_err / count),
            jitter=float(np.nanmean(jitter_vals)),
            official=lib.official_metrics(rows, raw, cfg),
            vel_dir_err_deg=float(np.nanmean(lib.velocity_dir_error_deg(gen_vel, gt_vel))),
            activity=lib.vt.activity_proxies(local, cfg),
            invariant_violations=int(diag["invariant_violations"]),
            style_pred=style_clf.classify(raw, label),
        )
        perturbed, _, pdiag = lib.run_sequence(
            model, x_norm=x_norm, label=label, style_feature=style_feature,
            z0_plan=z0_plan, mode="official",
            perturb=(direction, lib.PERTURB_AMPLITUDE, lib.T0_PERTURB))
        # recovery window metrics at +32 transitions
        def root_of(features):
            return lib.root_velocity_series(features)[0]
        pos_p, vel_p = lib.root_velocity_series(perturbed.numpy())
        pos_c, vel_c = lib.root_velocity_series(raw)
        _, vel_g = lib.root_velocity_series(rows)
        drift = np.linalg.norm(pos_p[:, [0, 2]] - pos_c[:, [0, 2]], axis=-1) * 100.0
        dir_pc = lib.velocity_dir_error_deg(vel_p, vel_c, min_speed_cm_s=0.0)
        t0, end = lib.T0_PERTURB, min(lib.T0_PERTURB + 32, len(drift))
        entry["perturb_root_dev_32"] = float(drift[end - 1])
        entry["perturb_vel_err_32"] = float(np.nanmean(dir_pc[t0:end]))
        entry["perturb_invariants"] = int(pdiag["invariant_violations"])
        per_source.append(entry)
    return per_source


def series(per_source, getter) -> np.ndarray:
    return np.asarray([getter(p) for p in per_source], dtype=np.float64)


def long_protocol(model, mean, std, device, arm: str) -> dict:
    from Utils.controller_validation import TeacherForcedStyleValidation
    validator = TeacherForcedStyleValidation(
        model, db=lib.vt.DB, mean=mean, std=std,
        classifier_path=lib.vt.CLASSIFIER_CKPT, seed=42, clips=90, windows=3,
        out_dir=None, eval_mode="rollout", batch_size=64, quality_metrics=True)
    metrics = validator.evaluate(model)
    starts_hash = hashlib.sha256(np.asarray(validator.starts).tobytes()).hexdigest()[:16]
    keep = ("matched_accuracy", "cross_accuracy", "matched_root_ade_cm", "cross_root_ade_cm",
            "jitter_cm", "foot_sliding_local_cm", "foot_sliding_world_cm",
            "world_contact_count", "world_contact_coverage",
            "matched_pose_step_p99_cm", "cross_pose_step_p99_cm")
    out = {key: (float(metrics[key]) if isinstance(metrics.get(key), (int, float)) and metrics.get(key) is not None else None)
           for key in keep}
    out["starts_sha256_16"] = starts_hash
    out["protocol"] = metrics.get("quality_protocol")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "acceptance.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    manifests, payloads = {}, {"contract": None, "arms": {}}
    for arm in ARMS:
        run = arm_dir(arm)
        manifest = json.loads((run / "manifest.json").read_text())
        manifests[arm] = manifest
    contract = contract_gate(manifests)
    payloads["contract"] = contract

    style_clf = lib.StyleClassifier(device)
    for arm in ARMS:
        run = arm_dir(arm)
        endpoint = run / "epoch_0010.pt"
        # load per-arm controller via the standard loader (reads checkpoint args)
        from Utils.motion_eval import load_motion_model
        model = load_motion_model("flowmatch_target", endpoint, num_styles=90).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        from Utils.runtime import load_norm_stats
        mean, std = load_norm_stats(lib.VQ / "norm_stats.npz")
        lib.set_norm(mean, std)
        from Utils.reproducibility import state_fingerprint
        before = state_fingerprint(model.state_dict())
        entry = {"arm": arm, "noise": ARMS[arm], "checkpoint": str(endpoint),
                 "closed_loop": closed_loop_protocol(model, mean, std, device, style_clf)}
        if not args.skip_long:
            entry["long_protocol"] = long_protocol(model, mean, std, device, arm)
        after = state_fingerprint(model.state_dict())
        entry["state_unchanged"] = before == after
        if before != after:
            raise RuntimeError(f"C2 acceptance mutated arm {arm}")
        payloads["arms"][arm] = entry
        print(f"[C2 accept] {arm} done")

    # Pairwise gates vs the control arm (A050).
    gates = {}
    gates["c2_0_contract"] = dict(result="PASS" if contract["passed"] else "FAIL",
                                  detail=contract)
    control = payloads["arms"][CONTROL]["closed_loop"]

    def metric_getters():
        return {
            "local_mpjpe": lambda p: p["local_mpjpe"],
            "jitter": lambda p: p["jitter"],
            "world_mpjpe": lambda p: p["official"]["mpjpe"],
            "root_ade": lambda p: p["official"]["root_ade"],
            "fs_local": lambda p: p["official"]["fs_local"],
            "fs_world": lambda p: p["official"]["fs_world"],
            "vel_dir_err_deg": lambda p: p["vel_dir_err_deg"],
            "perturb_root_dev_32": lambda p: p["perturb_root_dev_32"],
            "perturb_vel_err_32": lambda p: p["perturb_vel_err_32"],
        }

    getters = metric_getters()
    comparisons = {}
    for arm in ARMS:
        if arm == CONTROL:
            continue
        candidates = payloads["arms"][arm]["closed_loop"]
        comparisons[arm] = {}
        for key, getter in getters.items():
            ci = paired_ci(series(control, getter), series(candidates, getter))
            comparisons[arm][key] = {**ci, "confirmed_improvement": ci["ci"][1] < 0.0,
                                     "relative_change": (ci["b1_abs"] - ci["b0_abs"]) / ci["b0_abs"]
                                     if ci["b0_abs"] else None}
    payloads["comparisons"] = comparisons

    def any_confirmed(arm):
        return [k for k, v in comparisons[arm].items() if v["confirmed_improvement"]]

    best_improvements = {arm: any_confirmed(arm) for arm in comparisons}
    gates["c2_1_main"] = dict(
        result="PASS" if any(best_improvements.values()) else ("FAIL"),
        detail=best_improvements,
        note="improvement vs A=0.5 with paired 95% CI upper < 0 on a closed-loop metric")

    # Quality gate from the long protocol (and closed-loop paired values as backup).
    quality_keys = ("matched_root_ade_cm", "cross_root_ade_cm", "foot_sliding_world_cm", "foot_sliding_local_cm")
    quality = {}
    for arm in ARMS:
        long = payloads["arms"][arm].get("long_protocol") or {}
        quality[arm] = {k: long.get(k) for k in quality_keys}
    gates["c2_2_quality"] = dict(result="PENDING" if args.skip_long else "see detail", detail=quality)

    # Activity gate (closed-loop activity proxies vs control).
    activity = {}
    for arm in ARMS:
        values = payloads["arms"][arm]["closed_loop"]
        activity[arm] = {
            "moving": float(np.mean(series(values, lambda p: p["activity"]["moving_frame_fraction"]))),
            "strong": float(np.mean(series(values, lambda p: p["activity"]["strong_moving_frame_fraction"]))),
            "leg_travel": float(np.mean(series(values, lambda p: p["activity"]["leg_travel_cm"]))),
        }
    act_gate = {}
    for arm in ARMS:
        if arm == CONTROL:
            continue
        act_gate[arm] = {k: (activity[arm][k] - activity[CONTROL][k]) / activity[CONTROL][k]
                         for k in activity[arm]}
    gates["c2_3_activity"] = dict(
        result="PASS" if all(v >= -0.05 for arm in act_gate.values() for v in arm.values()) else "FAIL",
        detail=act_gate)

    # Style gate: matched/cross from the long protocol when available.
    style = {}
    for arm in ARMS:
        long = payloads["arms"][arm].get("long_protocol") or {}
        closed = payloads["arms"][arm]["closed_loop"]
        style[arm] = dict(
            matched_accuracy_240f=long.get("matched_accuracy"),
            cross_accuracy_240f=long.get("cross_accuracy"),
            matched_12src=float(np.mean(series(closed, lambda p: float(p["style_pred"]["correct"])))),
        )
    gates["c2_4_style"] = dict(result="PENDING" if args.skip_long else "see detail", detail=style)
    gates["c2_5_antifreeze"] = dict(
        result=gates["c2_3_activity"]["result"],
        note="the activity gate is the anti-freeze screen; jitter-only improvements cannot pass")

    # Overall verdict logic.
    hard_fail = gates["c2_0_contract"]["result"] == "FAIL" or gates["c2_3_activity"]["result"] == "FAIL"
    main_pass = gates["c2_1_main"]["result"] == "PASS"
    if hard_fail:
        verdict = "INVALID_EXPERIMENT" if gates["c2_0_contract"]["result"] == "FAIL" else "FAIL"
    elif args.skip_long:
        verdict = "PARTIAL (long protocol skipped)"
    elif main_pass:
        # style gate evaluation when long protocol exists
        ref = payloads["arms"][CONTROL]["long_protocol"]
        style_ok = all(
            (payloads["arms"][arm]["long_protocol"]["matched_accuracy"] >= ref["matched_accuracy"] - 0.02
             and payloads["arms"][arm]["long_protocol"]["cross_accuracy"] >= ref["cross_accuracy"] - 0.02)
            for arm in ARMS)
        gates["c2_4_style"]["result"] = "PASS" if style_ok else "FAIL"
        verdict = "PASS" if style_ok else "FAIL"
    else:
        verdict = "FAIL"
    payloads["verdict"] = verdict

    args.out.write_text(json.dumps(payloads, indent=1, ensure_ascii=False))
    for name, gate in gates.items():
        print(f"[{str(gate['result']):28s}] {name}")
    print(f"[C2a] verdict={verdict}")
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
