from __future__ import annotations

import unittest

import updatesupport as us
from examples import conformal_reporting_stability as example


class ConformalReportingStabilityExampleTests(unittest.TestCase):
    def test_conformal_result_contains_split_conformal_targets(self):
        result = example.build_conformal_result()

        self.assertIsInstance(result, us.ConformalAdapterResult)
        self.assertEqual(result.source, "split_conformal_regression")
        self.assertEqual(result.source_rows, 16)
        self.assertEqual(result.interval_width_column, "interval_width")
        self.assertEqual(result.crosses_threshold_column, "crosses_threshold")
        self.assertTrue(any(row["crosses_threshold"] for row in result.rows))
        self.assertTrue(any(row["miscovered"] for row in result.rows))

    def test_conformal_report_audits_multiple_uncertainty_targets(self):
        report = example.build_stability_report()

        self.assertIsInstance(report, us.ConformalReportingStabilityReport)
        self.assertEqual(report.target_count, 7)
        self.assertEqual(report.status, "needs_refinement")

        tables = report.to_tables()
        target_rows = {row["target"]: row for row in tables["targets"]}
        self.assertGreater(target_rows["interval_width"]["ambiguity"], 0.03)
        self.assertGreater(target_rows["crosses_threshold"]["ambiguity"], 0.25)
        self.assertIn("refinement_recommendations", tables)

    def test_conformal_example_renders_markdown(self):
        markdown = example.render_report()

        self.assertIn("# Conformal Prediction Reporting Stability Example", markdown)
        self.assertIn("What The Conformal Layer Supplies", markdown)
        self.assertIn("Hidden-Composition Readout", markdown)
        self.assertIn("threshold-crossing burden", markdown.lower())
        self.assertIn("Conformal prediction quantifies row-level", markdown)


if __name__ == "__main__":
    unittest.main()
