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
    root_ade, d2e=None, activity_scale=1.0, style_correct=None,
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
        per_source=per_source,
        aggregate=aggregate,
        style_proxy=dict(accuracy=float(style_correct.mean()), per_source=[
            dict(correct=int(style_correct[i])) for i in range(n)]),
    )


def _run_s4(tmp_path, b0, b1):
    module = _load_s4_module()
    b0_path, b1_path = tmp_path / "b0.json", tmp_path / "b1.json"
    out = tmp_path / "acceptance.json"
    b0_path.write_text(json.dumps(b0))
    b1_path.write_text(json.dumps(b1))
    old = sys.argv
    sys.argv = ["s4", "--b0", str(b0_path), "--b1", str(b1_path), "--out", str(out)]
    try:
        module.main()
    finally:
        sys.argv = old
    return json.loads(out.read_text())


IDS = list(range(1, 13))


def test_s4_passes_when_every_gate_is_met(tmp_path):
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.01, size=12)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, d2e=0.35 + noise, jitter_q=0.40 + noise,
        jitter_c=0.12 + noise, jitter_gt=0.09 + noise, fs_world=0.49 + noise,
        fs_local_q=0.25 + noise, mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=1.0,
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.28 + noise, d2e=0.35 + noise, jitter_q=0.30 + noise,
        jitter_c=0.11 + noise, jitter_gt=0.09 + noise, fs_world=0.47 + noise,
        fs_local_q=0.24 + noise, mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=1.0,
    )
    report = _run_s4(tmp_path, b0, b1)
    assert report["verdict"] == "PASS", json.dumps(report["gates"], indent=1)
    assert report["gates"]["g1_d2r_down"]["result"] == "PASS"
    assert report["pairs"]["d2r_local_mean"]["point"] == pytest.approx(-0.12, abs=0.02)


def test_s4_fails_on_activity_collapse(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.normal(scale=0.01, size=12)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.28 + noise, jitter_q=0.30 + noise, jitter_c=0.11 + noise,
        jitter_gt=0.09 + noise, fs_world=0.47 + noise, fs_local_q=0.24 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
        activity_scale=0.85,
    )
    report = _run_s4(tmp_path, b0, b1)
    assert report["verdict"] == "FAIL"
    assert report["gates"]["g6_activity"]["result"] == "FAIL"


def test_s4_inconclusive_when_no_separation(tmp_path):
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.01, size=12)
    b0 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.40 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
    )
    b1 = _synthetic_probe(
        ids=IDS, d2r=0.40 + noise, jitter_q=0.395 + noise, jitter_c=0.12 + noise,
        jitter_gt=0.09 + noise, fs_world=0.49 + noise, fs_local_q=0.25 + noise,
        mpjpe=1.58 + noise, local_mpjpe=1.07 + noise, root_ade=0.85 + noise,
    )
    report = _run_s4(tmp_path, b0, b1)
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
        assert values.shape == (12,), f"{name} has shape {values.shape}"
        assert np.isfinite(values).all(), f"{name} has non-finite entries"
    # The registered baseline numbers must be reproducible from the probe itself.
    assert payload["aggregate"]["official_q"]["jitter"] == pytest.approx(0.4116350422386362, rel=1e-6)
    assert payload["aggregate"]["codec_flip_rate"]["mean"] == pytest.approx(0.6259685, rel=1e-4)
