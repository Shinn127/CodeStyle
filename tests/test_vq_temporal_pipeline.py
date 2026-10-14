"""Regression checks for the vq_temporal_source experiment pipeline (S0/S3/S4).

Covers the failure mode that actually happened: the entry wrapper expanded a command that
the trainer's argparse rejected (--no-decoder-only) because the DRY_RUN text check only
looked for flag presence. The new checks parse the expanded argv with the trainer's own
parser, pin the A/B wrapper contract, validate the S4 gate logic on synthetic probes, and
guard the val12 selection against the archived audit (skipped without the local database).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "vq_temporal_source"))

import training.train_basic_vqvae_375to291 as trainer  # noqa: E402
import vq_temporal_common as common  # noqa: E402


def _parse(command: str):
    import contextlib
    import io
    import shlex

    # Strip the "[Dry run]" prefix first (it is two shell tokens), then drop <python> and the
    # train-script path so only the flags remain.
    body = command[len("[Dry run]"):] if command.startswith("[Dry run]") else command
    argv = shlex.split(body)[2:]
    old = sys.argv
    sys.argv = ["train_basic_vqvae_375to291.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.parse_args()
        return True, None
    except SystemExit as exc:
        return False, str(exc.code)
    finally:
        sys.argv = old


@pytest.mark.parametrize(
    "env_extra,expect_flag,reject_flag",
    [({}, None, "--no-decoder-only"), ({"DECODER_ONLY": "1"}, "--decoder-only", "--no-decoder-only")],
)
def test_entry_wrapper_dry_run_expands_trainer_parseable_argv(tmp_path, env_extra, expect_flag, reject_flag):
    env = dict(os.environ, DRY_RUN="1", PYTHON_BIN=sys.executable, OUT=str(tmp_path / "out"), **env_extra)
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_train_basic_vqvae_375to291.sh")],
        env=env, cwd=ROOT, capture_output=True, text=True, check=True,
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("[Dry run]"))
    assert reject_flag not in line
    if expect_flag is not None:
        assert expect_flag in line
    ok, error = _parse(line)
    assert ok, f"expanded command fails the trainer parser: {error}"


@pytest.mark.parametrize("arm,acc_weight", [("B0", "0"), ("B1", "4.9842")])
def test_ab_wrapper_dry_run_pins_protocol(tmp_path, arm, acc_weight):
    init = tmp_path / "init.pt"
    norm = tmp_path / "norm_stats.npz"
    init.write_bytes(b"placeholder")
    norm.write_bytes(b"placeholder")
    env = dict(
        os.environ, DRY_RUN="1", PYTHON_BIN=sys.executable, ARM=arm,
        INIT_CKPT=str(init), NORM_STATS=str(norm),
        OUT=str(tmp_path / "out"), LOG=str(tmp_path / "run.log"),
    )
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_train_vq_temporal_ab.sh")],
        env=env, cwd=ROOT, capture_output=True, text=True, check=True,
    )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("[Dry run]"))
    assert f"--acc-weight {acc_weight}" in line
    for pinned in ("--seq-len 8", "--epochs 10", "--loader-generator-seed 4242",
                   "--latent-margin-weight 0.0", "--roundtrip-weight 0.0",
                   f"--init-ckpt {init}", f"--norm-stats {norm}"):
        assert pinned in line.replace("\\", "")
    ok, error = _parse(line)
    assert ok, f"expanded command fails the trainer parser: {error}"


def test_ab_wrapper_rejects_conflicting_acc_weight(tmp_path):
    env = dict(
        os.environ, DRY_RUN="1", PYTHON_BIN=sys.executable, ARM="B0",
        ACC_WEIGHT="3.0", INIT_CKPT="/tmp/x", NORM_STATS="/tmp/y",
        OUT=str(tmp_path / "out"), LOG=str(tmp_path / "run.log"),
    )
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_train_vq_temporal_ab.sh")],
        env=env, cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "conflicts with ARM" in proc.stderr


def test_jitter_curves_match_official_formula_on_synthetic_positions():
    rng = np.random.default_rng(0)
    local_cm = rng.normal(size=(40, 24, 3)) * 5.0
    values = common.jitter_values_cm(local_cm)
    manual = np.linalg.norm(np.diff(local_cm, n=2, axis=0), axis=-1).reshape(-1)
    np.testing.assert_allclose(values, manual, rtol=1e-12, atol=1e-12)
    # `_jitteriness_cm` in Utils.motion_eval rescales by 100 assuming meters; positions here
    # are already cm, so the probe's values must NOT carry another factor of 100.
    assert values.mean() < 362 * 100


def test_activity_proxies_track_known_motion():
    frames = 60
    zero = np.zeros((frames, 24, 3))
    cfg = common.MotionEvalConfig()
    still = common.activity_proxies(zero, cfg)
    assert still["moving_frame_fraction"] == 0.0
    assert still["pose_step_mean_cm"] == 0.0

    moving = zero.copy()
    moving[:, :, 0] = np.arange(frames)[:, None] * 1.5  # 1.5 cm/frame every joint
    moved = common.activity_proxies(moving, cfg)
    assert moved["moving_frame_fraction"] == 1.0
    assert moved["strong_moving_frame_fraction"] == 1.0
    assert moved["pose_step_mean_cm"] == pytest.approx(1.5)
    lift = zero.copy()
    lift[:, common.TOE_INDICES, 1] = 10.0
    lifted = common.activity_proxies(lift, cfg)
    assert lifted["foot_lift_fraction"] == 1.0
    assert lifted["toe_height_range_cm"] == 0.0


def _load_s4_module():
    spec = importlib.util.spec_from_file_location(
        "s4_acceptance", ROOT / "analysis" / "vq_temporal_source" / "s4_acceptance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ACTIVITY_DEFAULTS = (
    ("moving_frame_fraction", 0.9), ("strong_moving_frame_fraction", 0.5),
    ("pose_step_mean_cm", 0.8), ("leg_travel_cm", 120.0),
    ("toe_height_range_cm", 6.0), ("foot_lift_fraction", 0.2),
    ("toe_speed_mean_cm_per_frame", 1.2),
)


def _synthetic_probe(
    *, ids, d2r, jitter_q, jitter_c, jitter_gt, fs_world, fs_local_q, mpjpe, local_mpjpe,
    root_ade, d2e=None, activity_scale=1.0, style_correct=None, checkpoint=None,
):
    """Minimal payload honouring every key s4_acceptance reads (schema mirrors the probe)."""
    ids = list(ids)
    n = len(ids)
    d2e = d2r * 0.9 if d2e is None else np.asarray(d2e, dtype=np.float64)
    style_correct = np.ones(n) if style_correct is None else np.asarray(style_correct, dtype=np.float64)
    per_source = []
    for i, sid in enumerate(ids):
        per_source.append(dict(
            source_entry_id=int(sid), style=f"s{sid}", suffix="FW", group="walk",
            d2r_local=dict(mean=float(d2r[i]), p50=float(d2r[i]), p95=float(d2r[i] * 1.5), count=100),
            d2e_local=dict(mean=float(d2e[i]), p50=float(d2e[i]), p95=float(d2e[i]), count=100),
            official_q=dict(
                jitter_sum_cm=float(jitter_q[i]) * 100, jitter_count=100,
                mpjpe_sum_cm=float(mpjpe[i]) * 100, mpjpe_count=100,
                local_mpjpe_sum_cm=float(local_mpjpe[i]) * 100, local_mpjpe_count=100,
                root_ade_sum_cm=float(root_ade[i]) * 100, root_ade_count=100,
                fs_local_sum_cm=float(fs_local_q[i]) * 100, fs_local_count=100,
                fs_world_sum_cm=float(fs_world[i]) * 100, fs_world_count=100,
                contact_possible=200,
            ),
            official_c=dict(
                jitter_sum_cm=float(jitter_c[i]) * 100, jitter_count=100,
                mpjpe_sum_cm=float(mpjpe[i]) * 90, mpjpe_count=100,
                local_mpjpe_sum_cm=float(local_mpjpe[i]) * 90, local_mpjpe_count=100,
                root_ade_sum_cm=float(root_ade[i]) * 90, root_ade_count=100,
                fs_local_sum_cm=float(fs_local_q[i]) * 100, fs_local_count=100,
                fs_world_sum_cm=float(fs_world[i]) * 90, fs_world_count=100,
                contact_possible=200,
            ),
            official_gt=dict(
                jitter_sum_cm=float(jitter_gt[i]) * 100, jitter_count=100,
                mpjpe_sum_cm=0.0, mpjpe_count=100, local_mpjpe_sum_cm=0.0, local_mpjpe_count=100,
                root_ade_sum_cm=0.0, root_ade_count=100,
                fs_local_sum_cm=float(fs_local_q[i]) * 100, fs_local_count=100,
                fs_world_sum_cm=float(fs_world[i]) * 100, fs_world_count=100,
                contact_possible=200,
            ),
            activity_q={key: float(value) * activity_scale for key, value in ACTIVITY_DEFAULTS},
            contact=dict(pred_contact_coverage=0.12 * activity_scale),
        ))

    aggregate = {
        "official_q": {
            "jitter": float(np.mean(jitter_q)), "mpjpe": float(np.mean(mpjpe)),
            "local_mpjpe": float(np.mean(local_mpjpe)), "root_ade": float(np.mean(root_ade)),
            "fs_local": float(np.mean(fs_local_q)), "fs_world": float(np.mean(fs_world)),
        },
        "official_c": {
            "jitter": float(np.mean(jitter_c)), "mpjpe": float(np.mean(mpjpe)) * 0.9,
            "local_mpjpe": float(np.mean(local_mpjpe)) * 0.9,
            "root_ade": float(np.mean(root_ade)) * 0.9,
            "fs_local": float(np.mean(fs_local_q)) * 0.9,
            "fs_world": float(np.mean(fs_world)) * 0.9,
        },
        "official_gt": {"jitter": float(np.mean(jitter_gt)), "fs_world": float(np.mean(fs_world)) * 0.2},
        "contact.pred_contact_coverage": float(0.12 * activity_scale),
        **{f"activity_q.{key}": {"mean": float(value) * activity_scale} for key, value in ACTIVITY_DEFAULTS},
    }
    return dict(
        checkpoint=str(checkpoint) if checkpoint else None,
        per_source=per_source,
        aggregate=aggregate,
        style_proxy=dict(accuracy=float(style_correct.mean()), per_source=[
            dict(correct=int(style_correct[i])) for i in range(n)]),
    )


def _synthetic_run(tmp_path, arm, *, acc_weight, mutate=None):
    """Write a minimal run dir (config.json) + manifest.json + probe payload scaffold.

    `mutate` can override manifest fields to exercise the INVALID_EXPERIMENT contracts.
    """
    run_dir = tmp_path / f"run_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "epoch_0010.pt"
    ckpt.write_bytes(b"stub")
    config = dict(
        seq_len=8, stride=1, batch_size=512, epochs=10, lr=1e-4, min_lr=1e-5,
        seed=42, loader_generator_seed=4242, acc_weight=acc_weight,
        out_dir=str(run_dir),
    )
    (run_dir / "config.json").write_text(json.dumps(config))
    manifest = dict(
        arm=arm, run_dir=str(run_dir),
        contract=dict(pinned=dict(seq_len=8, epochs=10), acc_weight_expected=acc_weight,
                      acc_weight_actual=acc_weight),
        init_checkpoint=dict(path="/x/init.pt", sha256="init-file", state_sha256="init-state"),
        final_checkpoint=dict(path=str(ckpt), sha256="final", state_sha256=f"final-{arm}"),
        dataset=dict(db_sha256="db", split_sha256="split", vq_state_sha256="vq"),
        norm_stats=dict(path="/x/norm.npz", sha256="norm"),
        epochs=[], integrity=dict(total_updates=51650.0),
    )
    if mutate:
        mutate(manifest)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return dict(run_dir=run_dir, ckpt=ckpt, manifest=manifest, manifest_path=manifest_path)


def _run_s4(tmp_path, b0, b1, *, manifest_b0=None, manifest_b1=None):
    module = _load_s4_module()
    b0_path, b1_path = tmp_path / "b0.json", tmp_path / "b1.json"
    out = tmp_path / "acceptance.json"
    b0_path.write_text(json.dumps(b0))
    b1_path.write_text(json.dumps(b1))
    if manifest_b0 is None or manifest_b1 is None:
        raise ValueError("contract gate is mandatory: pass both manifests")
    old = sys.argv
    sys.argv = ["s4", "--b0", str(b0_path), "--b1", str(b1_path),
                "--b0-manifest", str(manifest_b0), "--b1-manifest", str(manifest_b1),
                "--out", str(out)]
    try:
        module.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


def _default_runs(tmp_path, *, b1_mutate=None, b0_mutate=None):
    r0 = _synthetic_run(tmp_path, "B0", acc_weight=0.0, mutate=b0_mutate)
    r1 = _synthetic_run(tmp_path, "B1", acc_weight=4.9842, mutate=b1_mutate)
    return r0, r1


IDS = list(range(1, 13))


def test_s4_passes_when_every_gate_is_met(tmp_path):
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.01, size=12)
    r0, r1 = _default_runs(tmp_path)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, d2e=0.35 + noise, jitter_q=0.40 + noise,
        jitter_c=0.12 + noise, jitter_gt=0.09 + noise, fs_world=0.49 + noise,
        fs_local_q=0.25 + noise, mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=1.0, checkpoint=r0["ckpt"],
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.28 + noise, d2e=0.35 + noise, jitter_q=0.30 + noise,
        jitter_c=0.11 + noise, jitter_gt=0.09 + noise, fs_world=0.47 + noise,
        fs_local_q=0.24 + noise, mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=1.0, checkpoint=r1["ckpt"],
    )
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "PASS", json.dumps(report["gates"], indent=1)
    assert report["gates"]["g0_contract"]["result"] == "PASS"
    assert report["gates"]["g1_d2r_down"]["result"] == "PASS"
    assert report["pairs"]["d2r_local_mean"]["point"] == pytest.approx(-0.12, abs=0.02)


def test_s4_invalid_experiment_when_contract_breaks(tmp_path):
    """Different initialization/dataset must short-circuit to INVALID_EXPERIMENT, never PASS."""
    rng = np.random.default_rng(3)
    noise = rng.normal(scale=0.01, size=12)

    def break_init(manifest):
        manifest["init_checkpoint"]["state_sha256"] = "different-init"

    def break_db(manifest):
        manifest["dataset"]["db_sha256"] = "different-db"

    r0, r1 = _default_runs(tmp_path, b1_mutate=break_init)
    b0 = _synthetic_probe(ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r0["ckpt"])
    b1 = _synthetic_probe(ids=IDS, d2r=0.28 + noise, jitter_q=0.30 + noise, jitter_c=0.11 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.47 + noise, fs_local_q=0.24 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r1["ckpt"])
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "INVALID_EXPERIMENT"
    assert report["gates"]["g0_contract"]["result"] == "FAIL"
    assert report["manifest_checks"]["checks"]["same_init_state_sha"]["ok"] is False

    # Same-updates-but-different-DB must also be caught (the exact hole the audit found).
    r0, r1 = _default_runs(tmp_path / "second", b1_mutate=break_db)
    assert r0["manifest"]["integrity"]["total_updates"] == r1["manifest"]["integrity"]["total_updates"]
    b0["checkpoint"] = str(r0["ckpt"])
    b1["checkpoint"] = str(r1["ckpt"])
    report = _run_s4(tmp_path / "second", b0, b1,
                     manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "INVALID_EXPERIMENT"
    assert report["manifest_checks"]["checks"]["same_dataset_sha"]["ok"] is False


def test_s4_invalid_when_config_diff_exceeds_acc_weight(tmp_path):
    rng = np.random.default_rng(4)
    noise = rng.normal(scale=0.01, size=12)
    r0, r1 = _default_runs(tmp_path)
    config = json.loads((r1["run_dir"] / "config.json").read_text())
    config["seq_len"] = 4  # not an allowed difference
    (r1["run_dir"] / "config.json").write_text(json.dumps(config))
    b0 = _synthetic_probe(ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r0["ckpt"])
    b1 = _synthetic_probe(ids=IDS, d2r=0.28 + noise, jitter_q=0.30 + noise, jitter_c=0.11 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.47 + noise, fs_local_q=0.24 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r1["ckpt"])
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "INVALID_EXPERIMENT"
    assert "seq_len" in report["manifest_checks"]["config_diff_keys"]


def test_s4_probe_checkpoint_must_belong_to_its_manifest_run(tmp_path):
    rng = np.random.default_rng(5)
    noise = rng.normal(scale=0.01, size=12)
    r0, r1 = _default_runs(tmp_path)
    b0 = _synthetic_probe(ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r1["ckpt"])  # B0 probe points at B1's run
    b1 = _synthetic_probe(ids=IDS, d2r=0.28 + noise, jitter_q=0.30 + noise, jitter_c=0.11 + noise,
                          jitter_gt=0.09 + noise, fs_world=0.47 + noise, fs_local_q=0.24 + noise,
                          mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
                          checkpoint=r1["ckpt"])
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "INVALID_EXPERIMENT"
    assert report["manifest_checks"]["checks"]["probe_b0_is_from_manifest_run"]["ok"] is False


def test_s4_fails_on_activity_collapse(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=0.01, size=12)
    r0, r1 = _default_runs(tmp_path)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        checkpoint=r0["ckpt"],
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.28 + noise, jitter_q=0.30 + noise, jitter_c=0.11 + noise,
        jitter_gt=0.09 + noise, fs_world=0.47 + noise, fs_local_q=0.24 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=0.85, checkpoint=r1["ckpt"],
    )
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "FAIL"
    assert report["gates"]["g6_activity"]["result"] == "FAIL"


def test_s4_inconclusive_when_no_separation(tmp_path):
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.01, size=12)
    r0, r1 = _default_runs(tmp_path)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        checkpoint=r0["ckpt"],
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.395 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        checkpoint=r1["ckpt"],
    )
    report = _run_s4(tmp_path, b0, b1, manifest_b0=r0["manifest_path"], manifest_b1=r1["manifest_path"])
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["gates"]["g3_jitter_drop"]["result"] == "FAIL"


def test_probe_val12_selection_matches_archived_audit_when_db_present():
    if not (common.DB / "segments.json").exists():
        pytest.skip("local database not available in this environment")
    picks = common.fixed_val12()
    assert [p["frame_start"] for p in picks][:3] == [1577711, 863033, 2213202]


def test_probe_common_rejects_unknown_suffix():
    with pytest.raises(ValueError):
        common.group_of("XX")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="probe requires CUDA")
def test_common_per_source_metric_schema_is_stable():
    """The probe -> S4 contract: every metric name s4 reads must resolve on a real probe payload."""
    if not (common.SRC_CKPT).exists():
        pytest.skip("source checkpoint not available")
    if not common.CLASSIFIER_CKPT.exists():
        pytest.skip("style classifier checkpoint not available")
    payload = common.full_probe(common.SRC_CKPT, device="cuda", style=True, tag="schema_check")
    for name in (
        "d2r_local_mean", "d2r_local_p95", "d2e_local_mean", "jitter_quant_local",
        "jitter_cont_local", "mpjpe_q", "local_mpjpe_q", "root_ade_q", "fs_local_q",
        "fs_world_q", "activity_q.moving_frame_fraction",
        "activity_q.strong_moving_frame_fraction", "activity_q.pose_step_mean_cm",
        "activity_q.leg_travel_cm", "activity_q.toe_height_range_cm",
        "activity_q.foot_lift_fraction", "activity_q.toe_speed_mean_cm_per_frame",
        "pred_contact_coverage", "style_correct",
    ):
        values = common.per_source_metric(payload, name)
        expected = 90 if name == "style_correct" else 12  # style proxy spans the full val split
        assert values.shape == (expected,), f"{name} has shape {values.shape}"
        assert np.isfinite(values).all(), f"{name} has non-finite entries"
    # The registered baseline numbers must be reproducible from the probe itself.
    assert payload["aggregate"]["official_q"]["jitter"] == pytest.approx(0.4116350422386362, rel=1e-6)
    assert payload["aggregate"]["codec_flip_rate"]["mean"] == pytest.approx(0.6259685, rel=1e-4)
