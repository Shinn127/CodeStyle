"""C2 follow-up diagnostic: closed-loop evaluation on an EXPANDED val source set.

DECISION.md left A075's quality question open because the registered closed-loop protocol
uses only 12 val sources (wide CI). This script re-runs the same C0-C closed-loop protocol
on a larger, deterministic subset of val sources (default 30) for the three C2 arms and
reports paired per-source diffs vs the A050 control with a source-level bootstrap CI.

Evaluation-only: no training, weights frozen, state fingerprint verified. The registered
12-source numbers stay untouched; this is an explicitly-labelled extension diagnostic.

  python analysis/vq_controller_recovery/C2/c2_expand_eval.py [--sources 30] [--device cuda]
"""
from __future__ import annotations

import argparse
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
SEED = 42
DRAWS = 1000


def expanded_sources(n: int) -> list[dict]:
    """Deterministic val sources beyond the 12 registered ones.

    The first 12 are exactly the registered fixed_val12 picks (same RNG call) so the
    extension is a strict superset; the rest are drawn from the same filtered val pool with
    a separate fixed seed and appended in sorted order.
    """
    base = lib.vt.fixed_val12()
    if n <= len(base):
        return base[:n]
    _, segments, _ = lib.vt.read_db()
    pool = [s for s in segments if s.get("split") == "val" and not s.get("mirror", False)
            and s["num_frames"] >= lib.T_FRAMES + 2]
    used = {int(s["source_entry_id"]) for s in base}
    rest = sorted((s for s in pool if int(s["source_entry_id"]) not in used),
                  key=lambda s: (int(s["source_entry_id"]), int(s["frame_start"])))
    return base + rest[: n - len(base)]


@torch.no_grad()
def closed_loop(model, mean, std, device, picks) -> list[dict]:
    cfg = lib.MotionEvalConfig(metric_start_frame=1)
    _, _, x_source = lib.vt.read_db()
    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    std_t = torch.as_tensor(std, dtype=torch.float32)
    rows_out = []
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
        motion_norm = torch.as_tensor((raw[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
        local, _ = lib.vt.positions_from_motion_norm(motion_norm, mean_t, std_t)
        gt_norm = torch.as_tensor((rows[1:, 84:] - mean[84:]) / std[84:], dtype=torch.float32)
        gt_local, _ = lib.vt.positions_from_motion_norm(gt_norm, mean_t, std_t)
        sum_err, count = lib.vt.local_mpjpe_sum_count(local, gt_local)
        official = lib.official_metrics(rows, raw, cfg)
        activity = lib.vt.activity_proxies(local, cfg)
        rows_out.append(dict(
            source_entry_id=int(pick["source_entry_id"]), style=pick["style_name"],
            suffix=pick.get("suffix"), group=lib.vt.group_of(pick["suffix"]),
            local_mpjpe=float(sum_err / count),
            jitter=float(np.nanmean(lib.vt.jitter_values_cm(local))),
            world_mpjpe=official["mpjpe"], root_ade=official["root_ade"],
            fs_world=official["fs_world"], fs_local=official["fs_local"],
            moving=activity["moving_frame_fraction"], strong=activity["strong_moving_frame_fraction"],
            invariant_violations=int(diag["invariant_violations"]),
        ))
    return rows_out


def paired_ci(control_values: np.ndarray, candidate_values: np.ndarray) -> dict:
    diff = candidate_values - control_values
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(diff), size=(DRAWS, len(diff)))
    samples = diff[draws].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(point=float(diff.mean()), ci=[float(lo), float(hi)],
                control_abs=float(control_values.mean()), candidate_abs=float(candidate_values.mean()),
                relative_change=float(diff.mean() / control_values.mean()),
                n=int(len(diff)), seed=SEED, draws=DRAWS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=HERE / "expand_eval.json")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    picks = expanded_sources(int(args.sources))
    extension = [int(p["source_entry_id"]) for p in picks[12:]]

    from Utils.motion_eval import load_motion_model
    from Utils.reproducibility import state_fingerprint
    from Utils.runtime import load_norm_stats

    mean, std = load_norm_stats(lib.VQ / "norm_stats.npz")
    lib.set_norm(mean, std)

    results = {}
    for arm in ARMS:
        run = lib.ROOT / "checkpoints" / f"c2_obsnoise_{arm}_10ep"
        endpoint = run / "epoch_0010.pt"
        model = load_motion_model("flowmatch_target", endpoint, num_styles=90).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = state_fingerprint(model.state_dict())
        per_source = closed_loop(model, mean, std, device, picks)
        after = state_fingerprint(model.state_dict())
        if before != after:
            raise RuntimeError(f"expansion diagnostic mutated arm {arm}")
        results[arm] = per_source
        agg = {k: float(np.mean([r[k] for r in per_source]))
               for k in ("local_mpjpe", "jitter", "world_mpjpe", "root_ade", "fs_world", "moving", "strong")}
        print(f"[expand {arm}] n={len(per_source)} " +
              " ".join(f"{k}={v:.4f}" for k, v in agg.items()))

    control = results[CONTROL]
    def col(per_source, key):
        return np.asarray([r[key] for r in per_source], dtype=np.float64)

    comparisons = {}
    for arm in ARMS:
        if arm == CONTROL:
            continue
        comparisons[arm] = {
            key: paired_ci(col(control, key), col(results[arm], key))
            for key in ("local_mpjpe", "jitter", "world_mpjpe", "root_ade", "fs_world",
                        "moving", "strong")
        }

    payload = dict(
        stage="C2-followup", kind="expanded closed-loop evaluation (evaluation-only)",
        note=("registered protocol uses fixed val12; this extension is diagnostic and does "
              "not replace the registered numbers"),
        sources=int(len(picks)), extension_source_ids=extension,
        protocol="C0-C (GT traj, official clip->Q->D feedback), identical noise plan and seed",
        git=lib.vt.git_commit(), environment=lib.vt.environment_report(),
        per_arm_aggregate={arm: {k: float(np.mean([r[k] for r in rows]))
                                 for k in ("local_mpjpe", "jitter", "world_mpjpe", "root_ade",
                                           "fs_world", "fs_local", "moving", "strong")}
                           for arm, rows in results.items()},
        comparisons_vs_A050=comparisons,
        per_source=results,
    )
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"[expand] n={len(picks)} sources ({len(extension)} beyond the registered 12)")
    print("[Done]", args.out)


if __name__ == "__main__":
    main()
