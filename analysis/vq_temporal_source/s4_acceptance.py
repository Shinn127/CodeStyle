"""S4: pre-registered joint acceptance for the B0 vs B1 pilot (COMMON.md section 7).

Paired comparison of the two probe JSONs: per-source values are paired by source_entry_id
and resampled together (bootstrap seed 42, 1000 draws). All thresholds were registered
before the runs started; this script only evaluates them.

  python analysis/vq_temporal_source/s4_acceptance.py \
      --b0 <B0 probe json> --b1 <B1 probe json> \
      [--b0-manifest ... --b1-manifest ...] [--out S4/acceptance_ep10.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vq_temporal_common import OUT_ROOT, json_dump, per_source_metric  # noqa: E402

BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 1000
JITTER_DROP_REQUIRED = 0.15          # gate 3: official quantized jitter -15%
QUALITY_REGRESSION_LIMIT = 0.03      # gate 5: mpjpe/root_ade <= +3%
FS_REGRESSION_LIMIT = 0.005          # gate 4: world/local FS <= +0.5%
D2E_LIMIT = 0.005                    # gate 2: ||d2 e_q|| <= +0.5%
ACTIVITY_DROP_LIMIT = 0.05           # gate 6: proxies >= -5%
STYLE_DROP_LIMIT = 0.02              # gate 7: style proxy accuracy -2pp


def paired_bootstrap(b0: np.ndarray, b1: np.ndarray, *, statistic: str) -> dict:
    """CI for B1-B0 (mean or p95-of-per-source) resampling sources with replacement."""
    diff = b1 - b0
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diff)
    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    if statistic == "mean":
        samples = diff[draws].mean(axis=1)
        point = float(diff.mean())
        absolute = np.mean
    elif statistic == "median":
        samples = np.median(diff[draws], axis=1)
        point = float(np.median(diff))
        absolute = np.median
    else:
        raise ValueError(statistic)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(statistic=statistic, point=point, ci=[float(lo), float(hi)],
                b0_absolute=float(absolute(b0)), b1_absolute=float(absolute(b1)),
                n_sources=int(n), seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS)


def ratio_change(b0_value: float, b1_value: float) -> float:
    return b1_value / b0_value - 1.0 if b0_value else float("inf")


def gate(name: str, passed: bool, detail: dict, *, note: str = "") -> dict:
    return dict(name=name, result="PASS" if passed else "FAIL", detail=detail, note=note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--b1", type=Path, required=True)
    ap.add_argument("--b0-manifest", type=Path, default=None)
    ap.add_argument("--b1-manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="ep10")
    args = ap.parse_args()

    b0 = json.loads(args.b0.read_text())
    b1 = json.loads(args.b1.read_text())
    ids0 = [c["source_entry_id"] for c in b0["per_source"]]
    ids1 = [c["source_entry_id"] for c in b1["per_source"]]
    if ids0 != ids1:
        raise RuntimeError(f"probe sources are not paired: {ids0} vs {ids1}")

    def series(payload, key):
        return per_source_metric(payload, key).astype(np.float64)

    report: dict = dict(
        label=args.label, b0=str(args.b0.resolve()), b1=str(args.b1.resolve()),
        protocol="COMMON.md section 7 (pre-registered before B0/B1 started)",
        thresholds=dict(jitter_drop_required=JITTER_DROP_REQUIRED,
                        quality_regression_limit=QUALITY_REGRESSION_LIMIT,
                        fs_regression_limit=FS_REGRESSION_LIMIT, d2e_limit=D2E_LIMIT,
                        activity_drop_limit=ACTIVITY_DROP_LIMIT, style_drop_limit=STYLE_DROP_LIMIT),
        gates={}, pairs={},
    )

    # Gate 1: ||d2r|| mean and p95 must both drop. Both keys are per-source arrays
    # (mean of frame-level values / p95 of frame-level values); the paired bootstrap resamples
    # the difference of their means, matching the aggregate the report table shows
    # (`aggregate.d2r_local.per_source_mean` / `.per_source_p95`). CI upper < 0 = confirmed drop.
    g1 = {}
    ok1 = True
    for key in ("d2r_local_mean", "d2r_local_p95"):
        pair = paired_bootstrap(series(b0, key), series(b1, key), statistic="mean")
        confirmed = pair["ci"][1] < 0.0
        g1[key] = dict(**pair, confirmed_drop=confirmed,
                       relative_change=ratio_change(pair["b0_absolute"], pair["b1_absolute"]))
        ok1 &= confirmed
    report["pairs"].update(g1)
    report["gates"]["g1_d2r_down"] = gate(
        "d2r mean & p95 both drop (95% CI upper < 0)", ok1, g1,
        note="CI including 0 counts as not confirmed for the final verdict",
    )

    # Gate 2: ||d2 e_q|| must not grow.
    pair = paired_bootstrap(series(b0, "d2e_local_mean"), series(b1, "d2e_local_mean"), statistic="mean")
    rel = ratio_change(pair["b0_absolute"], pair["b1_absolute"])
    report["pairs"]["d2e_local_mean"] = pair
    report["gates"]["g2_d2e_not_up"] = gate(
        f"||d2 e_q|| mean change <= +{D2E_LIMIT:.1%}", rel <= D2E_LIMIT, dict(**pair, relative_change=rel),
    )

    # Gate 3: official quantized jitter relative drop >= 15%.
    j0 = float(b0["aggregate"]["official_q"]["jitter"])
    j1 = float(b1["aggregate"]["official_q"]["jitter"])
    jrel = ratio_change(j0, j1)
    pair = paired_bootstrap(series(b0, "jitter_quant_local"), series(b1, "jitter_quant_local"), statistic="mean")
    report["pairs"]["jitter_quant_local"] = pair
    report["gates"]["g3_jitter_drop"] = gate(
        f"official quantized jitter drop >= {JITTER_DROP_REQUIRED:.0%}",
        jrel <= -JITTER_DROP_REQUIRED,
        dict(b0=j0, b1=j1, relative_change=jrel, paired=pair),
    )

    # Gate 4: world FS not worse; local FS reported as well.
    g4 = {}
    ok4 = True
    for key, agg_key in (("fs_world_q", "fs_world"), ("fs_local_q", "fs_local")):
        v0 = float(b0["aggregate"]["official_q"][agg_key])
        v1 = float(b1["aggregate"]["official_q"][agg_key])
        pair = paired_bootstrap(series(b0, key), series(b1, key), statistic="mean")
        rel = ratio_change(v0, v1)
        g4[key] = dict(b0=v0, b1=v1, relative_change=rel, paired=pair,
                       within_limit=rel <= FS_REGRESSION_LIMIT)
        ok4 &= rel <= FS_REGRESSION_LIMIT
        report["pairs"][key] = pair
    report["gates"]["g4_foot_sliding"] = gate(
        f"world & local FS change <= +{FS_REGRESSION_LIMIT:.1%}", ok4, g4,
    )

    # Gate 5: local/world MPJPE and root ADE regression <= 3% each.
    g5 = {}
    ok5 = True
    for key, agg_key in (("mpjpe_q", "mpjpe"), ("local_mpjpe_q", "local_mpjpe"), ("root_ade_q", "root_ade")):
        v0 = float(b0["aggregate"]["official_q"][agg_key])
        v1 = float(b1["aggregate"]["official_q"][agg_key])
        rel = ratio_change(v0, v1)
        g5[key] = dict(b0=v0, b1=v1, relative_change=rel, within_limit=rel <= QUALITY_REGRESSION_LIMIT)
        ok5 &= rel <= QUALITY_REGRESSION_LIMIT
    report["gates"]["g5_quality"] = gate(
        f"MPJPE/local/rootADE regression <= {QUALITY_REGRESSION_LIMIT:.0%}", ok5, g5,
    )

    # Gate 6: activity proxies may not drop more than 5% (higher-is-more-active direction).
    g6 = {}
    ok6 = True
    activity_keys = (
        "activity_q.moving_frame_fraction", "activity_q.strong_moving_frame_fraction",
        "activity_q.pose_step_mean_cm", "activity_q.leg_travel_cm",
        "activity_q.toe_height_range_cm", "activity_q.foot_lift_fraction",
        "activity_q.toe_speed_mean_cm_per_frame", "contact.pred_contact_coverage",
    )
    for key in activity_keys:
        # activity_* entries are mean_of() stat blocks ({"mean": ...}); contact.* entries are
        # plain floats in aggregate_probe -- read whichever shape the probe produced.
        def value_of(payload):
            raw = payload["aggregate"][key]
            return float(raw["mean"]) if isinstance(raw, dict) else float(raw)

        v0, v1 = value_of(b0), value_of(b1)
        rel = ratio_change(v0, v1) if v0 else float("inf")
        within = rel >= -ACTIVITY_DROP_LIMIT
        g6[key] = dict(b0=v0, b1=v1, relative_change=rel, within_limit=within)
        ok6 &= within
    report["gates"]["g6_activity"] = gate(
        f"activity proxies drop <= {ACTIVITY_DROP_LIMIT:.0%}", ok6, g6,
    )

    # Gate 7: style proxy accuracy drop <= 2pp.
    s0 = float(b0["style_proxy"]["accuracy"]) if b0.get("style_proxy") else float("nan")
    s1 = float(b1["style_proxy"]["accuracy"]) if b1.get("style_proxy") else float("nan")
    pair = paired_bootstrap(series(b0, "style_correct"), series(b1, "style_correct"), statistic="mean")
    report["pairs"]["style_correct"] = pair
    ok7 = (s1 - s0) >= -STYLE_DROP_LIMIT
    report["gates"]["g7_style"] = gate(
        f"style proxy accuracy drop <= {STYLE_DROP_LIMIT:.0%}",
        ok7, dict(b0=s0, b1=s1, change=s1 - s0, paired=pair),
    )

    # Gate 8 (context, not pass/fail): continuous AE contrast.
    report["continuous_contrast"] = {
        key: dict(b0=float(b0["aggregate"]["official_c"][key]), b1=float(b1["aggregate"]["official_c"][key]),
                  relative_change=ratio_change(float(b0["aggregate"]["official_c"][key]),
                                               float(b1["aggregate"]["official_c"][key])))
        for key in ("jitter", "fs_world", "fs_local", "mpjpe", "local_mpjpe", "root_ade")
    }
    report["gt_reference"] = {
        "jitter": float(b0["aggregate"]["official_gt"]["jitter"]),
        "fs_world": float(b0["aggregate"]["official_gt"]["fs_world"]),
    }

    # Group breakdown (diagnostic, small n per group; not part of the gates).
    groups = {}
    for payload, tag in ((b0, "b0"), (b1, "b1")):
        for c in payload["per_source"]:
            g = groups.setdefault(c["group"], {"b0": [], "b1": []})
            g[tag].append(c)
    report["group_breakdown"] = {
        g: dict(
            n=len(v["b0"]),
            d2r_local=dict(
                b0=float(np.mean([c["d2r_local"]["mean"] for c in v["b0"]])),
                b1=float(np.mean([c["d2r_local"]["mean"] for c in v["b1"]])),
            ),
            jitter_q=dict(
                b0=float(np.mean([c["official_q"]["jitter_sum_cm"] / c["official_q"]["jitter_count"] for c in v["b0"]])),
                b1=float(np.mean([c["official_q"]["jitter_sum_cm"] / c["official_q"]["jitter_count"] for c in v["b1"]])),
            ),
        )
        for g, v in groups.items()
    }

    # Manifest cross-checks: identical updates and matched configs except acc.
    if args.b0_manifest and args.b1_manifest:
        m0 = json.loads(args.b0_manifest.read_text())
        m1 = json.loads(args.b1_manifest.read_text())
        checks = dict(
            same_total_updates=m0["integrity"]["total_updates"] == m1["integrity"]["total_updates"],
            total_updates=m0["integrity"]["total_updates"],
            acc_weight=(m0["contract"]["acc_weight_actual"], m1["contract"]["acc_weight_actual"]),
            same_dataset_sha=m0["dataset"]["db_sha256"] == m1["dataset"]["db_sha256"],
            same_init_state=m0["init_checkpoint"]["state_sha256"] == m1["init_checkpoint"]["state_sha256"],
        )
        report["manifest_checks"] = checks
        if not checks["same_total_updates"]:
            report["gates"]["g0_updates"] = gate("B0/B1 actual updates identical", False, checks)
        else:
            report["gates"]["g0_updates"] = gate("B0/B1 actual updates identical", True, checks)
    else:
        report["manifest_checks"] = None

    # Verdict: all registered gates PASS -> PASS; a confirmed reversal on gate 1, a >3% quality
    # regression, a >5% activity drop or a style collapse -> FAIL; otherwise INCONCLUSIVE.
    g1_reversed = any(v["ci"][0] > 0 for v in g1.values())
    hard_fail = (
        g1_reversed
        or any(not v["within_limit"] and v["relative_change"] > QUALITY_REGRESSION_LIMIT for v in g5.values())
        or any(not v["within_limit"] and v["relative_change"] < -ACTIVITY_DROP_LIMIT for v in g6.values())
        or not ok7
    )
    all_pass = all(g["result"] == "PASS" for g in report["gates"].values()) and not any(
        v["confirmed_drop"] is False for v in g1.values()
    )
    if hard_fail:
        verdict = "FAIL"
    elif all_pass:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    report["verdict"] = verdict
    report["verdict_note"] = (
        "PASS = every registered gate met and d2r CIs exclude 0 below; "
        "FAIL = confirmed reversal or registered veto; INCONCLUSIVE = budget-limited/no separation"
    )

    out = args.out or (OUT_ROOT / "S4" / f"acceptance_{args.label}.json")
    json_dump(out, report)
    for name, g in report["gates"].items():
        print(f"[{g['result']:4s}] {name}")
    print(f"[S4] verdict={verdict}")
    print("[Done]", out)


if __name__ == "__main__":
    main()
