"""C0 closed-loop diagnostic library: invariants and helpers (no real weights required).

Covers the pieces that can be wrong silently in the causal decomposition:
  * planner command schedule boundaries (240 frames, four segments)
  * deterministic per-(source, step) noise plan and perturbation direction
  * bucket indexing for the five registered time windows
  * velocity direction error math and NaN policy on low speed
  * Jitter identity with the official formula on synthetic positions
The full 5-group run is an experiment, not a unit test; it is exercised by c0_run.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
C0 = ROOT / "analysis" / "vq_controller_recovery" / "C0"
sys.path.insert(0, str(C0))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lib():
    return _load("c0_loop_lib", C0 / "loop_lib.py")


def test_planner_schedule_covers_240_frames_with_registered_segments(lib):
    assert lib.planner_command(1) == (1.0, 0.0, 0.0, 1.2)
    assert lib.planner_command(60) == (1.0, 0.0, 0.0, 1.2)
    assert lib.planner_command(61) == (1.0, 0.0, 0.3, 1.2)
    assert lib.planner_command(121) == (0.8, 0.2, 0.0, 1.0)
    assert lib.planner_command(181) == (1.0, 0.0, -0.2, 1.2)
    assert lib.planner_command(240) == (1.0, 0.0, -0.2, 1.2)
    with pytest.raises(ValueError):
        lib.planner_command(241)


def test_buckets_cover_1_to_240_exactly(lib):
    covered = []
    for lo, hi in lib.BUCKETS:
        covered.extend(range(lo, hi + 1))
    assert covered == list(range(1, 241))


def test_bucket_slice_indexes_transition_arrays(lib):
    # transition t is stored at index t-1; bucket [1,16] must read indices 0..15
    assert lib.bucket_slice((1, 16)) == slice(0, 16)
    assert lib.bucket_slice((121, 240)) == slice(120, 240)


def test_perturbation_direction_is_deterministic_and_unit_norm(lib):
    class FakeModel:
        latent_dim = 128

    first = lib.perturb_direction(FakeModel(), torch.device("cpu"))
    second = lib.perturb_direction(FakeModel(), torch.device("cpu"))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert float(first.norm()) == pytest.approx(1.0, abs=1e-6)


def test_velocity_dir_error_math_and_low_speed_nan_policy(lib):
    gen = np.array([[1.0, 0.0], [0.0, 1.0], [1e-6, 0.0], [1.0, 1.0]])
    ref = np.array([[1.0, 0.0], [1.0, 0.0], [1e-6, 0.0], [1.0, 0.0]])
    err = lib.velocity_dir_error_deg(gen * 100.0, ref * 100.0, min_speed_cm_s=1.0)
    assert err[0] == pytest.approx(0.0, abs=1e-6)
    assert err[1] == pytest.approx(90.0, abs=1e-4)
    assert np.isnan(err[2])          # reference below the speed floor -> NaN, not fabricated
    assert err[3] == pytest.approx(45.0, abs=1e-4)


def test_local_metrics_helpers_match_official_jitter_identity(lib):
    rng = np.random.default_rng(0)
    local_cm = rng.normal(size=(30, 24, 3)) * 5.0
    values = lib.vt.jitter_values_cm(local_cm)
    manual = np.linalg.norm(np.diff(local_cm, n=2, axis=0), axis=-1).reshape(-1)
    np.testing.assert_allclose(values, manual, rtol=1e-12, atol=1e-12)


def test_c0_config_records_registered_constants(lib):
    config = lib.c0_config()
    assert config["seed"] == 42 and config["frames"] == 241
    assert config["perturb"] == {"t0": 120, "amplitude": 0.5, "seed": 4242}
    assert [tuple(b) for b in config["buckets"]] == list(lib.BUCKETS)
    assert config["vq_state_sha256"] == "918b5e07046266f7b38d15f3ba9b35cf33987cb4843f3d6f00a5ef6ce90e7f5c"
