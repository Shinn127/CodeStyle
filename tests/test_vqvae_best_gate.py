from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


NEWCODE_DIR = Path(__file__).resolve().parents[1]
if str(NEWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(NEWCODE_DIR))

from Utils.vqvae_eval import (
    DEFAULT_BEST_METRICS,
    check_best_gate,
    resolve_best_metrics,
)

FOUR = ("mpjpe", "jitter", "foot_sliding", "root_ade")


def metrics(mpjpe=10.0, jitter=1.0, foot_sliding=0.5, root_ade=3.0, usage=0.9):
    return {
        "mpjpe_cm": mpjpe,
        "local_mpjpe_cm": mpjpe - 1.0,
        "jitter_cm": jitter,
        "foot_sliding_cm": foot_sliding,
        "root_ade_cm": root_ade,
        "heading_error_deg": 1.0,
        "velocity_dir_error_deg": 2.0,
        "codebook_usage": usage,
    }


class ResolveBestMetricsTests(unittest.TestCase):
    def test_default_spec_keeps_order(self) -> None:
        self.assertEqual(resolve_best_metrics("mpjpe,jitter,foot_sliding,root_ade"), FOUR)

    def test_whitespace_and_duplicates(self) -> None:
        self.assertEqual(
            resolve_best_metrics(" mpjpe , jitter ,mpjpe, foot_sliding"),
            ("mpjpe", "jitter", "foot_sliding"),
        )

    def test_disabled_spec_falls_back_to_primary(self) -> None:
        self.assertEqual(resolve_best_metrics("none", primary="jitter"), ("jitter",))
        self.assertEqual(resolve_best_metrics("", primary="mpjpe"), ("mpjpe",))

    def test_disabled_spec_without_usable_primary_uses_defaults(self) -> None:
        self.assertEqual(resolve_best_metrics("none", primary="loss"), DEFAULT_BEST_METRICS)
        self.assertEqual(resolve_best_metrics("none"), DEFAULT_BEST_METRICS)

    def test_primary_is_appended_when_missing(self) -> None:
        self.assertEqual(
            resolve_best_metrics("jitter,root_ade", primary="mpjpe"),
            ("jitter", "root_ade", "mpjpe"),
        )

    def test_non_metric_primary_is_ignored(self) -> None:
        self.assertEqual(resolve_best_metrics("jitter", primary="loss"), ("jitter",))

    def test_aliases_and_unknown_names(self) -> None:
        self.assertEqual(resolve_best_metrics("fs,pose_mpjpe"), ("foot_sliding", "local_mpjpe"))
        with self.assertRaises(ValueError):
            resolve_best_metrics("mpjpe,fid")


class BestGateTests(unittest.TestCase):
    def test_first_evaluation_always_accepts(self) -> None:
        result = check_best_gate(metrics(), FOUR)
        self.assertTrue(result.accept)
        self.assertEqual(result.reason, "first_eval")
        self.assertEqual(result.blockers, ())

    def test_accepts_only_when_every_metric_improves(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        improved = check_best_gate(
            metrics(mpjpe=9.0, jitter=0.9, foot_sliding=0.4, root_ade=2.5),
            FOUR,
            reference=reference,
        )
        self.assertTrue(improved.accept)
        self.assertEqual(improved.blockers, ())
        self.assertEqual(improved.reason, "all_improved")

    def test_single_regression_blocks_the_save(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        result = check_best_gate(
            metrics(mpjpe=9.0, jitter=1.1, foot_sliding=0.4, root_ade=2.5),
            FOUR,
            reference=reference,
        )
        self.assertFalse(result.accept)
        self.assertEqual(result.blockers, ("jitter",))
        self.assertIn("jitter", result.describe())

    def test_unchanged_metric_blocks_a_strict_gate(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        result = check_best_gate(
            metrics(mpjpe=9.0, jitter=1.0, foot_sliding=0.4, root_ade=2.5),
            FOUR,
            reference=reference,
        )
        self.assertFalse(result.accept)
        self.assertEqual(result.blockers, ("jitter",))

    def test_min_delta_requires_a_headline_improvement(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        tiny = check_best_gate(
            metrics(mpjpe=9.999, jitter=0.9995, foot_sliding=0.4995, root_ade=2.999),
            FOUR,
            reference=reference,
            min_delta=0.01,
        )
        self.assertFalse(tiny.accept)
        self.assertEqual(set(tiny.blockers), set(FOUR))
        clear = check_best_gate(
            metrics(mpjpe=9.0, jitter=0.9, foot_sliding=0.4, root_ade=2.5),
            FOUR,
            reference=reference,
            min_delta=0.01,
        )
        self.assertTrue(clear.accept)

    def test_negative_min_delta_tolerates_noise(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        result = check_best_gate(
            metrics(mpjpe=9.9, jitter=1.005, foot_sliding=0.495, root_ade=2.99),
            FOUR,
            reference=reference,
            min_delta=-0.01,
        )
        self.assertTrue(result.accept)

    def test_higher_is_better_metric(self) -> None:
        reference = check_best_gate(metrics(), ("codebook_usage",)).values
        self.assertTrue(
            check_best_gate(metrics(usage=0.95), ("codebook_usage",), reference=reference).accept
        )
        self.assertFalse(
            check_best_gate(metrics(usage=0.85), ("codebook_usage",), reference=reference).accept
        )

    def test_metric_missing_from_reference_cannot_block(self) -> None:
        partial = {"mpjpe_cm": 12.0}
        result = check_best_gate(metrics(), FOUR, reference=partial)
        self.assertTrue(result.accept)
        self.assertEqual(result.blockers, ())

    def test_non_finite_value_blocks(self) -> None:
        reference = check_best_gate(metrics(), FOUR).values
        result = check_best_gate(
            metrics(mpjpe=float("nan"), jitter=0.9, foot_sliding=0.4, root_ade=2.5),
            FOUR,
            reference=reference,
        )
        self.assertFalse(result.accept)
        self.assertEqual(result.blockers, ("mpjpe",))
        self.assertTrue(math.isnan(result.values["mpjpe"]))

    def test_unknown_metric_and_empty_gate(self) -> None:
        with self.assertRaises(ValueError):
            check_best_gate(metrics(), ())
        with self.assertRaises(KeyError):
            check_best_gate(
                {"mpjpe_cm": 1.0}, ("root_ade",)
            )


if __name__ == "__main__":
    unittest.main()
