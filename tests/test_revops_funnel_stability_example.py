from __future__ import annotations

import unittest

import updatesupport as us
from examples import revops_funnel_stability as example


class RevOpsFunnelStabilityExampleTests(unittest.TestCase):
    def test_synthetic_funnel_report_crosses_health_threshold(self):
        rows = example.synthetic_funnel_rows()
        report = example.build_public_report(rows)
        verdict = example.build_funnel_claim(rows)
        frontier = example.build_frontier(rows)

        self.assertEqual(len(rows), 18)
        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertGreater(report.observed_value, 0.18)
        self.assertLess(report.interval.lower, 0.18)
        self.assertGreater(report.interval.upper, 0.18)
        self.assertFalse(report.public_adequate)
        self.assertEqual(report.refinements[0].column, "lead_source")

        self.assertEqual(verdict.status, "fail")
        self.assertIsNotNone(verdict.decision)
        self.assertFalse(verdict.decision.invariant)
        self.assertIsNotNone(verdict.decision_repair_candidate)
        self.assertEqual(
            verdict.decision_repair_candidate.added_columns,
            ("deal_size_band", "rep_ramp_band"),
        )
        self.assertEqual(
            verdict.refinement_recommendations[0].columns,
            ("deal_size_band", "rep_ramp_band"),
        )

        self.assertIsInstance(frontier, us.PublicRepresentationFrontier)
        self.assertEqual(
            frontier.minimal_stable.added_columns,
            ("deal_size_band", "rep_ramp_band"),
        )
        self.assertGreater(frontier.baseline.max_ambiguity, 0.1)

    def test_synthetic_funnel_report_renders_markdown(self):
        markdown = example.render_report(include_frontier=False)

        self.assertIn("# RevOps Funnel Analysis Stability Example", markdown)
        self.assertIn("Observed MQL-to-SQL conversion", markdown)
        self.assertIn("Funnel Health Claim Audit", markdown)
        self.assertIn("Decision invariant: no", markdown)
        self.assertIn("base + deal_size_band, rep_ramp_band", markdown)


if __name__ == "__main__":
    unittest.main()
