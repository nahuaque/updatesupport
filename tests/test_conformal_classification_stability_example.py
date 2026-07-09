from __future__ import annotations

import unittest

import updatesupport as us
from examples import conformal_classification_stability as example


class ConformalClassificationStabilityExampleTests(unittest.TestCase):
    def test_classification_result_contains_prediction_set_targets(self):
        result = example.build_conformal_result()

        self.assertIsInstance(result, us.ConformalAdapterResult)
        self.assertEqual(result.source, "split_conformal_classification")
        self.assertEqual(result.source_rows, 18)
        self.assertEqual(result.prediction_set_size_column, "prediction_set_size")
        self.assertEqual(result.ambiguous_set_column, "ambiguous_set")
        self.assertTrue(any(row["ambiguous_set"] for row in result.rows))
        self.assertTrue(any(row["contains_positive_label"] for row in result.rows))

    def test_classification_report_audits_operational_set_burdens(self):
        report = example.build_stability_report()

        self.assertIsInstance(report, us.ConformalReportingStabilityReport)
        self.assertEqual(report.target_count, 5)
        self.assertEqual(report.status, "needs_refinement")

        tables = report.to_tables()
        target_rows = {row["target"]: row for row in tables["targets"]}
        self.assertGreater(target_rows["prediction_set_size"]["ambiguity"], 0.30)
        self.assertGreater(target_rows["ambiguous_set"]["ambiguity"], 0.20)
        self.assertGreater(target_rows["contains_positive_label"]["ambiguity"], 0.25)
        self.assertLessEqual(target_rows["miscovered"]["ambiguity"], 0.20)

    def test_classification_example_renders_markdown(self):
        markdown = example.render_report()

        self.assertIn(
            "# Conformal Classification Reporting Stability Example", markdown
        )
        self.assertIn("What The Conformal Layer Supplies", markdown)
        self.assertIn("Hidden-Composition Readout", markdown)
        self.assertIn("reject-label containment", markdown.lower())
        self.assertIn("prediction-set claims survive hidden", markdown)


if __name__ == "__main__":
    unittest.main()
