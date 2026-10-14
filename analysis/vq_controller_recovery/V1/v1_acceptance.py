"""V1 acceptance: pre-registered gates over the two probe JSONs (V1/PROTOCOL.md section 4).

Reuses the exact S4 gate/boot-strap machinery (analysis/vq_temporal_source/s4_acceptance.py)
so the V1 numbers share the verified aggregation and the mandatory contract gate. The V1
manifest contract additionally requires E/Q/EMA to be bitwise unchanged, which is checked
from each arm's manifest.json.

  python analysis/vq_controller_recovery/V1/v1_acceptance.py \
      --b0 <B0 probe.json> --b1 <B1 probe.json> \
      --b0-manifest ... --b1-manifest ... --out V1/acceptance.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "vq_temporal_source"))
sys.path.insert(0, str(ROOT / "analysis" / "vq_controller_recovery" / "C0"))

import s4_acceptance as s4  # noqa: E402  (verified gate + bootstrap implementations)
import loop_lib as lib  # noqa: E402


def v1_contract(m0: dict, m1: dict) -> dict:
    """Fail-closed V1 contract gate (V1/PROTOCOL.md V1-0)."""
    checks: dict[str, dict] = {}

    def add(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}

    def short(v):
        return f"{str(v)[:16]}…"

    add("same_init_state_sha",
        m0["init_checkpoint"]["state_sha256"] == m1["init_checkpoint"]["state_sha256"],
        f"{short(m0['init_checkpoint']['state_sha256'])} vs {short(m1['init_checkpoint']['state_sha256'])}")
    add("same_init_file_sha",
        m0["init_checkpoint"]["sha256"] == m1["init_checkpoint"]["sha256"],
        f"{short(m0['init_checkpoint']['sha256'])}")
    add("same_total_updates",
        int(m0["integrity"]["total_updates"]) == int(m1["integrity"]["total_updates"]),
        f"{m0['integrity']['total_updates']} vs {m1['integrity']['total_updates']}")
    for key, name in (("db_sha256", "same_dataset_sha"), ("split_sha256", "same_split_sha"),
                      ("vq_state_sha256", "same_vq_state_sha")):
        add(name, m0["dataset"].get(key) == m1["dataset"].get(key),
            f"{short(m0['dataset'].get(key))} vs {short(m1['dataset'].get(key))}")
    add("same_norm_sha", m0["norm_stats"]["sha256"] == m1["norm_stats"]["sha256"],
        short(m0["norm_stats"]["sha256"]))

    pinned_diff = {k: (m0["contract"]["pinned"].get(k), m1["contract"]["pinned"].get(k))
                   for k in set(m0["contract"]["pinned"]) | set(m1["contract"]["pinned"])
                   if m0["contract"]["pinned"].get(k) != m1["contract"]["pinned"].get(k)}
    # posacc_weight is intentionally NOT in the pinned dict (it is the variable).
    add("same_pinned_protocol", not pinned_diff, pinned_diff or "pinned dicts equal")

    w0 = float(m0["contract"]["posacc_weight"])
    w1 = float(m1["contract"]["posacc_weight"])
    add("posacc_is_single_variable", w0 == 0.0 and w1 > 0.0,
        f"B0={w0} B1={w1} (control must be 0, candidate > 0)")

    frozen_detail = {arm: {k: v["unchanged"] for k, v in m["frozen_groups"].items()}
                     for arm, m in (("B0", m0), ("B1", m1))}
    add("encoder_quantizer_bitwise_unchanged",
        all(flag for arm in frozen_detail.values() for flag in arm.values()),
        frozen_detail)
    add("decoder_actually_updated",
        int(m0["decoder_tensors_changed"]) > 0 and int(m1["decoder_tensors_changed"]) > 0,
        {"B0": m0["decoder_tensors_changed"], "B1": m1["decoder_tensors_changed"]})
    return dict(passed=all(item["ok"] for item in checks.values()), checks=checks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=Path, required=True)
    ap.add_argument("--b1", type=Path, required=True)
    ap.add_argument("--b0-manifest", type=Path, required=True)
    ap.add_argument("--b1-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "acceptance.json")
    args = ap.parse_args()

    b0 = json.loads(args.b0.read_text())
    b1 = json.loads(args.b1.read_text())
    ids0 = [c["source_entry_id"] for c in b0["per_source"]]
    ids1 = [c["source_entry_id"] for c in b1["per_source"]]
    if ids0 != ids1:
        raise RuntimeError(f"probes are not paired: {ids0} vs {ids1}")
    m0 = s4.load_manifest(args.b0_manifest)
    m1 = s4.load_manifest(args.b1_manifest)

    # V1-0: contract, adapted from S4's fail-closed gate to the V1 manifest schema:
    #   same init state/file sha; same updates; identical pinned protocol (excluding the
    #   single variable); encoder+quantizer bitwise unchanged per arm; only posacc differs.
    contract_v1 = v1_contract(m0, m1)

    def series(payload, key):
        return s4.per_source_metric(payload, key).astype(np.float64)

    report = dict(label="V1", b0=str(args.b0.resolve()), b1=str(args.b1.resolve()),
                  b0_manifest=str(args.b0_manifest.resolve()), b1_manifest=str(args.b1_manifest.resolve()),
                  protocol="V1/PROTOCOL.md section 4 (frozen before training)",
                  gates={}, pairs={})

    report["gates"]["v1_0_contract"] = s4.gate(
        "V1-0 contract: single variable, frozen E/Q/EMA, identical updates", contract_v1["passed"],
        contract_v1, note="mandatory; failure -> INVALID_EXPERIMENT")

    # V1-1: ||d2r|| mean & p95 drop with paired CI upper < 0.
    g1, ok1 = {}, True
    for key in ("d2r_local_mean", "d2r_local_p95"):
        pair = s4.paired_bootstrap(series(b0, key), series(b1, key), statistic="mean")
        confirmed = pair["ci"][1] < 0.0
        g1[key] = dict(**pair, confirmed_drop=confirmed,
                       relative_change=s4.ratio_change(pair["b0_absolute"], pair["b1_absolute"]))
        ok1 &= confirmed
    report["pairs"].update(g1)
    report["gates"]["v1_1_d2r"] = s4.gate("V1-1 ||d2r|| mean & p95 drop (CI upper < 0)", ok1, g1)

    # V1-2: official quantized jitter >= 15% drop.
    j0 = float(b0["aggregate"]["official_q"]["jitter"])
    j1 = float(b1["aggregate"]["official_q"]["jitter"])
    jrel = s4.ratio_change(j0, j1)
    report["gates"]["v1_2_jitter"] = s4.gate(
        "V1-2 quantized jitter drop >= 15%", jrel <= -0.15,
        dict(b0=j0, b1=j1, relative_change=jrel))

    # V1-3: local/world MPJPE + root ADE regression <= 3%.
    g3, ok3 = {}, True
    for key, agg_key in (("mpjpe_q", "mpjpe"), ("local_mpjpe_q", "local_mpjpe"), ("root_ade_q", "root_ade")):
        v0, v1 = float(b0["aggregate"]["official_q"][agg_key]), float(b1["aggregate"]["official_q"][agg_key])
        rel = s4.ratio_change(v0, v1)
        g3[key] = dict(b0=v0, b1=v1, relative_change=rel, within_limit=rel <= 0.03)
        ok3 &= rel <= 0.03
    report["gates"]["v1_3_quality"] = s4.gate("V1-3 pose/root regression <= 3%", ok3, g3)

    # V1-4: foot sliding regression <= 0.5%, both masks, with contact counts.
    g4, ok4 = {}, True
    for key, agg_key, count_key in (("fs_world_q", "fs_world", "fs_world_count"),
                                    ("fs_local_q", "fs_local", "fs_local_count")):
        v0, v1 = float(b0["aggregate"]["official_q"][agg_key]), float(b1["aggregate"]["official_q"][agg_key])
        rel = s4.ratio_change(v0, v1)
        g4[key] = dict(b0=v0, b1=v1, relative_change=rel, within_limit=rel <= 0.005,
                       contact_count_b0=float(b0["aggregate"]["official_q"].get(count_key, float("nan"))),
                       contact_count_b1=float(b1["aggregate"]["official_q"].get(count_key, float("nan"))))
        ok4 &= rel <= 0.005
    report["gates"]["v1_4_foot_sliding"] = s4.gate("V1-4 FS regression <= 0.5%", ok4, g4)

    # V1-5: activity/anti-freeze proxies drop <= 5%.
    g5, ok5 = {}, True
    for key in ("activity_q.moving_frame_fraction", "activity_q.strong_moving_frame_fraction",
                "activity_q.pose_step_mean_cm", "activity_q.leg_travel_cm",
                "activity_q.toe_height_range_cm", "activity_q.foot_lift_fraction",
                "activity_q.toe_speed_mean_cm_per_frame", "contact.pred_contact_coverage"):
        raw0, raw1 = b0["aggregate"][key], b1["aggregate"][key]
        # activity_* entries are stat blocks; contact.* entries are plain floats.
        v0 = float(raw0["mean"]) if isinstance(raw0, dict) else float(raw0)
        v1 = float(raw1["mean"]) if isinstance(raw1, dict) else float(raw1)
        rel = s4.ratio_change(v0, v1)
        g5[key] = dict(b0=v0, b1=v1, relative_change=rel, within_limit=rel >= -0.05)
        ok5 &= rel >= -0.05
    report["gates"]["v1_5_activity"] = s4.gate("V1-5 activity drop <= 5%", ok5, g5)

    # V1-6: style proxy drop <= 2pp.
    s0 = float(b0["style_proxy"]["accuracy"])
    s1 = float(b1["style_proxy"]["accuracy"])
    report["gates"]["v1_6_style"] = s4.gate("V1-6 style proxy drop <= 2pp", (s1 - s0) >= -0.02,
                                            dict(b0=s0, b1=s1, change=s1 - s0))

    # V1-7: context only.
    report["continuous_contrast"] = {
        key: dict(b0=float(b0["aggregate"]["official_c"][key]), b1=float(b1["aggregate"]["official_c"][key]),
                  relative_change=s4.ratio_change(float(b0["aggregate"]["official_c"][key]),
                                                  float(b1["aggregate"]["official_c"][key])))
        for key in ("jitter", "fs_world", "fs_local", "mpjpe", "local_mpjpe", "root_ade")}
    report["codec_contrast"] = dict(
        b0=float(b0["aggregate"]["codec_flip_rate"]["mean"]),
        b1=float(b1["aggregate"]["codec_flip_rate"]["mean"]))
    report["gt_reference"] = dict(jitter=float(b0["aggregate"]["official_gt"]["jitter"]),
                                  fs_world=float(b0["aggregate"]["official_gt"]["fs_world"]))

    hard_fail = (not ok1) or any(not v["within_limit"] and v["relative_change"] > 0.03 for v in g3.values()) \
        or any(not v["within_limit"] for v in g5.values())
    all_pass = all(g["result"] == "PASS" for g in report["gates"].values())
    if not contract_v1["passed"]:
        verdict = "INVALID_EXPERIMENT"
    elif hard_fail:
        verdict = "FAIL"
    elif all_pass:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    report["verdict"] = verdict
    report["verdict_note"] = ("PASS = all registered V1 gates met; V1 PASS is only a "
                              "representation-val candidate and must NOT be injected into the "
                              "live controller. FAIL = veto rule. INCONCLUSIVE = no separation.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for name, g in report["gates"].items():
        print(f"[{g['result']:4s}] {name}")
    print(f"[V1] verdict={verdict}")
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
