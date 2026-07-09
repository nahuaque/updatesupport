from __future__ import annotations

import importlib.util
import json
import unittest

import updatesupport as us


def _single_fiber_history(
    shares: tuple[tuple[str, float], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for period, x_share in shares:
        rows.extend(
            [
                {
                    "period": period,
                    "segment": "all",
                    "driver": "x",
                    "target": 0.0,
                    "weight": 100.0 * x_share,
                },
                {
                    "period": period,
                    "segment": "all",
                    "driver": "y",
                    "target": 1.0,
                    "weight": 100.0 * (1.0 - x_share),
                },
            ]
        )
    return rows


def _claim() -> us.ClaimSpec:
    return us.claim(
        "historical target rate remains positive",
        public=["segment"],
        hidden=["segment", "driver"],
        target="target",
        weight="weight",
        ambiguity_limit=0.5,
        decision=us.threshold_decision(">=", 0.25),
    )


@unittest.skipUnless(importlib.util.find_spec("cvxpy"), "cvxpy not installed")
class HistoricalTVCalibrationTests(unittest.TestCase):
    def test_calibration_uses_prior_transitions_for_rolling_backtests(self):
        rows = _single_fiber_history(
            (
                ("P1", 0.5),
                ("P2", 0.6),
                ("P3", 0.4),
                ("P4", 0.5),
                ("P5", 0.8),
            )
        )

        report = us.calibrate_tv_radius(
            rows,
            _claim(),
            period="period",
            coverage=0.75,
            min_train_transitions=2,
        )

        self.assertIsInstance(report, us.HistoricalTVCalibrationReport)
        self.assertEqual(report.eligible_transition_count, 4)
        self.assertEqual(report.backtest_count, 2)
        self.assertAlmostEqual(report.calibrated_radius, 0.2)
        self.assertAlmostEqual(report.q.radius, 0.2)

        first, second = report.backtests
        self.assertEqual(first.training_transition_count, 2)
        self.assertAlmostEqual(first.calibrated_radius, 0.2)
        self.assertAlmostEqual(first.actual_tv_radius, 0.1)
        self.assertTrue(first.shift_covered)
        self.assertTrue(first.target_covered)

        self.assertEqual(second.training_transition_count, 3)
        self.assertAlmostEqual(second.calibrated_radius, 0.2)
        self.assertAlmostEqual(second.actual_tv_radius, 0.3)
        self.assertFalse(second.shift_covered)
        self.assertFalse(second.target_covered)
        self.assertAlmostEqual(report.rolling_shift_coverage, 0.5)
        self.assertAlmostEqual(report.rolling_target_coverage, 0.5)
        self.assertAlmostEqual(report.rolling_decision_preservation, 0.5)

        current_audit = report.audit([row for row in rows if row["period"] == "P5"])
        self.assertEqual(current_audit.primary.grouped.q_name, "tv_budget(radius=0.2)")

        tables = report.to_tables()
        payload = json.loads(report.to_json())
        markdown = report.to_markdown()
        self.assertIn("transitions", tables)
        self.assertIn("backtests", tables)
        self.assertEqual(payload["calibrated_radius"], report.calibrated_radius)
        self.assertIn("Rolling Backtests", markdown)
        self.assertIn("restandardizing", markdown)

    def test_claim_method_routes_to_historical_calibration(self):
        rows = _single_fiber_history((("P1", 0.5), ("P2", 0.6), ("P3", 0.5)))

        report = _claim().calibrate_tv(
            rows,
            period="period",
            coverage=1.0,
            min_train_transitions=1,
        )

        self.assertIsInstance(report, us.HistoricalTVCalibrationReport)
        self.assertEqual(report.backtest_count, 1)

    def test_public_law_changes_are_removed_before_tv_is_measured(self):
        rows = [
            {
                "period": "P1",
                "segment": "A",
                "driver": "x",
                "target": 0.0,
                "weight": 40,
            },
            {
                "period": "P1",
                "segment": "A",
                "driver": "y",
                "target": 1.0,
                "weight": 40,
            },
            {"period": "P1", "segment": "B", "driver": "x", "target": 0.0, "weight": 4},
            {
                "period": "P1",
                "segment": "B",
                "driver": "y",
                "target": 1.0,
                "weight": 16,
            },
            {
                "period": "P2",
                "segment": "A",
                "driver": "x",
                "target": 0.0,
                "weight": 10,
            },
            {
                "period": "P2",
                "segment": "A",
                "driver": "y",
                "target": 1.0,
                "weight": 10,
            },
            {
                "period": "P2",
                "segment": "B",
                "driver": "x",
                "target": 0.0,
                "weight": 16,
            },
            {
                "period": "P2",
                "segment": "B",
                "driver": "y",
                "target": 1.0,
                "weight": 64,
            },
        ]

        report = us.calibrate_tv_radius(
            rows,
            _claim(),
            period="period",
            coverage=0.9,
            min_train_transitions=1,
        )

        self.assertAlmostEqual(report.transitions[0].tv_radius, 0.0)
        self.assertAlmostEqual(report.calibrated_radius, 0.0)
        self.assertEqual(report.backtest_count, 0)

    def test_new_hidden_cells_are_reported_as_support_drift(self):
        rows = _single_fiber_history((("P1", 0.5), ("P2", 0.6)))
        rows.extend(
            [
                {
                    "period": "P3",
                    "segment": "all",
                    "driver": "x",
                    "target": 0.0,
                    "weight": 50.0,
                },
                {
                    "period": "P3",
                    "segment": "all",
                    "driver": "y",
                    "target": 1.0,
                    "weight": 30.0,
                },
                {
                    "period": "P3",
                    "segment": "all",
                    "driver": "z",
                    "target": 0.5,
                    "weight": 20.0,
                },
            ]
        )

        report = us.calibrate_tv_radius(
            rows,
            _claim(),
            period="period",
            coverage=1.0,
            min_train_transitions=1,
        )

        transition = report.transitions[1]
        self.assertFalse(transition.calibration_eligible)
        self.assertFalse(transition.support_compatible)
        self.assertEqual(transition.new_hidden_cells, (("all", "z"),))
        self.assertEqual(report.unsupported_transition_count, 1)
        self.assertEqual(report.backtests[0].status, "unsupported_support")
        self.assertIsNone(report.backtests[0].shift_covered)
        self.assertIsNone(report.backtests[0].target_covered)

    def test_missing_reference_public_fiber_is_reported_as_support_drift(self):
        rows = [
            {
                "period": "P1",
                "segment": "A",
                "driver": "x",
                "target": 0.0,
                "weight": 30,
            },
            {
                "period": "P1",
                "segment": "A",
                "driver": "y",
                "target": 1.0,
                "weight": 30,
            },
            {
                "period": "P1",
                "segment": "B",
                "driver": "x",
                "target": 0.0,
                "weight": 20,
            },
            {
                "period": "P1",
                "segment": "B",
                "driver": "y",
                "target": 1.0,
                "weight": 20,
            },
            {
                "period": "P2",
                "segment": "A",
                "driver": "x",
                "target": 0.0,
                "weight": 36,
            },
            {
                "period": "P2",
                "segment": "A",
                "driver": "y",
                "target": 1.0,
                "weight": 24,
            },
            {
                "period": "P2",
                "segment": "B",
                "driver": "x",
                "target": 0.0,
                "weight": 16,
            },
            {
                "period": "P2",
                "segment": "B",
                "driver": "y",
                "target": 1.0,
                "weight": 24,
            },
            {
                "period": "P3",
                "segment": "A",
                "driver": "x",
                "target": 0.0,
                "weight": 50,
            },
            {
                "period": "P3",
                "segment": "A",
                "driver": "y",
                "target": 1.0,
                "weight": 50,
            },
        ]

        report = us.calibrate_tv_radius(
            rows,
            _claim(),
            period="period",
            coverage=1.0,
            min_train_transitions=1,
        )

        transition = report.transitions[1]
        self.assertIsNone(transition.tv_radius)
        self.assertFalse(transition.calibration_eligible)
        self.assertEqual(transition.missing_reference_public_cells, (("B",),))
        self.assertAlmostEqual(transition.missing_reference_public_mass, 0.4)
        self.assertEqual(report.backtests[0].status, "unsupported_support")

    def test_period_order_must_cover_observed_periods(self):
        rows = _single_fiber_history((("P1", 0.5), ("P2", 0.6)))

        with self.assertRaisesRegex(ValueError, "every observed period"):
            us.calibrate_tv_radius(
                rows,
                _claim(),
                period="period",
                period_order=["P1"],
            )


if __name__ == "__main__":
    unittest.main()
