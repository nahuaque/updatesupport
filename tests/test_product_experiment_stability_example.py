from __future__ import annotations

import unittest

import updatesupport as us
from examples import product_experiment_stability as example


class ProductExperimentStabilityExampleTests(unittest.TestCase):
    def test_synthetic_experiment_report_crosses_launch_decision_threshold(self):
        rows = example.synthetic_experiment_rows()
        report = example.build_public_report(rows)
        verdict = example.build_launch_claim(rows)

        self.assertEqual(len(rows), 13)
        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertGreater(report.observed_value, 0.0)
        self.assertLess(report.interval.lower, 0.0)
        self.assertGreater(report.interval.upper, 0.0)
        self.assertFalse(report.public_adequate)
        self.assertEqual(report.refinements[0].column, "acquisition_channel")

        self.assertEqual(verdict.status, "fail")
        self.assertIsNotNone(verdict.decision)
        self.assertFalse(verdict.decision.invariant)
        self.assertIsNotNone(verdict.decision_repair_candidate)
        self.assertEqual(
            verdict.decision_repair_candidate.added_columns,
            ("geo_market",),
        )

    def test_synthetic_experiment_report_renders_markdown(self):
        markdown = example.render_report(include_frontier=False)

        self.assertIn(
            "# Product Experimentation / A/B Test Stability Example",
            markdown,
        )
        self.assertIn("Observed treatment-minus-control lift", markdown)
        self.assertIn("Launch Decision Audit", markdown)
        self.assertIn("Decision invariant: no", markdown)
        self.assertIn("base + geo_market", markdown)


if __name__ == "__main__":
    unittest.main()
