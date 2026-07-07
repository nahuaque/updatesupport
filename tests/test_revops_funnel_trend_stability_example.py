from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import updatesupport as us
from examples import revops_funnel_trend_stability as example


class RevOpsFunnelTrendStabilityExampleTests(unittest.TestCase):
    def test_synthetic_trend_report_crosses_zero(self):
        rows = example.synthetic_trend_rows()
        report = example.build_trend_report(rows)
        verdict = example.build_trend_claim(rows)
        comparison = example.build_quarter_comparison(rows)
        frontier = example.build_frontier(rows)

        self.assertEqual(len(rows), 18)
        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertGreater(report.observed_value, 0.0)
        self.assertLess(report.interval.lower, 0.0)
        self.assertGreater(report.interval.upper, 0.0)
        self.assertFalse(report.public_adequate)
        self.assertEqual(report.refinements[0].column, "lead_source")

        self.assertEqual(verdict.status, "fail")
        self.assertIsNotNone(verdict.decision)
        self.assertFalse(verdict.decision.invariant)
        self.assertIsNotNone(verdict.decision_repair_candidate)
        self.assertEqual(
            verdict.decision_repair_candidate.added_columns,
            ("rep_ramp_band",),
        )

        self.assertIsInstance(comparison, us.RobustComparisonReport)
        self.assertEqual(comparison.status, "ambiguous_winner")
        self.assertEqual(comparison.observed_winner, example.CURRENT_PERIOD)
        self.assertIsNone(comparison.certified_winner)
        self.assertFalse(comparison.pairwise_results[0].robust_order)

        self.assertIsInstance(frontier, us.PublicRepresentationFrontier)
        self.assertEqual(frontier.minimal_stable.added_columns, ("rep_ramp_band",))

    def test_synthetic_trend_report_renders_markdown(self):
        markdown = example.render_report(include_frontier=False)

        self.assertIn("# RevOps Funnel Trend Stability Example", markdown)
        self.assertIn("Observed Q/Q SQL conversion lift", markdown)
        self.assertIn("Trend Claim Audit", markdown)
        self.assertIn("Robust Quarter Comparison", markdown)
        self.assertIn("Decision invariant: no", markdown)
        self.assertIn("base + rep_ramp_band", markdown)

    def test_synthetic_trend_exports_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "revops_trend_review"

            written = example.export_review_artifacts(output_dir)

            self.assertGreater(len(written), 6)
            self.assertTrue((output_dir / "revops_funnel_trend_stability.md").exists())
            self.assertTrue((output_dir / "trend_report.json").exists())
            self.assertTrue((output_dir / "trend_claim.json").exists())
            self.assertTrue((output_dir / "quarter_comparison.json").exists())
            self.assertTrue((output_dir / "frontier.json").exists())
            self.assertTrue(
                (output_dir / "tables" / "trend_claim__summary.csv").exists()
            )
            self.assertTrue(
                (
                    output_dir / "tables" / "quarter_comparison__pairwise_margins.csv"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
