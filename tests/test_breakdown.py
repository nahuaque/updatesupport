from __future__ import annotations

import unittest

import updatesupport as us


def _rows():
    return [
        {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
        {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
        {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
    ]


class BreakdownPointTests(unittest.TestCase):
    def test_bounded_shift_breakdown_radius_matches_threshold_crossing(self):
        report = us.breakdown_point(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            decision=us.threshold_decision(">=", 0.45),
            q_family="bounded_shift",
            radius_max=0.5,
            tolerance=1e-4,
            grid_size=6,
        )

        self.assertEqual(report.status, "found")
        self.assertAlmostEqual(report.breakdown_radius, 1.0 / 6.0, delta=1e-3)
        self.assertEqual(report.observed_decision, "pass")
        self.assertTrue(report.curve[0].decision_stable)
        self.assertFalse(report.curve[-1].decision_stable)
        self.assertIn("Breakdown radius", report.to_markdown())

        tables = report.to_tables()
        self.assertIn("summary", tables)
        self.assertIn("curve", tables)
        self.assertEqual(tables["summary"][0]["status"], "found")
        self.assertIn("curve", us.report_tables(report))

    def test_breakdown_not_found_when_decision_survives_search_range(self):
        report = us.breakdown_point(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            decision={"operator": ">=", "threshold": 0.3},
            q_family="bounded_shift",
            radius_max=0.5,
            tolerance=1e-4,
            grid_size=4,
        )

        self.assertEqual(report.status, "not_found")
        self.assertIsNone(report.breakdown_radius)
        self.assertAlmostEqual(report.stable_radius, 0.5)
        self.assertTrue(all(row.decision_stable for row in report.curve))

    def test_breakdown_can_be_already_broken_at_minimum_radius(self):
        report = us.breakdown_point(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            decision=us.threshold_decision(">=", 0.49),
            q_family="bounded_shift",
            radius_min=0.2,
            radius_max=0.5,
            tolerance=1e-4,
            grid_size=4,
        )

        self.assertEqual(report.status, "already_broken")
        self.assertAlmostEqual(report.breakdown_radius, 0.2)
        self.assertIsNone(report.stable_radius)
        self.assertAlmostEqual(report.broken_radius, 0.2)

    def test_breakdown_rejects_invalid_search_range(self):
        with self.assertRaisesRegex(ValueError, "radius_max"):
            us.breakdown_point(
                _rows(),
                public=["public"],
                hidden=["public", "hidden"],
                target="target",
                decision=us.threshold_decision(">=", 0.5),
                radius_min=1.0,
                radius_max=0.0,
            )


if __name__ == "__main__":
    unittest.main()
