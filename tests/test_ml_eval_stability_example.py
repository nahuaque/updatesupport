from __future__ import annotations

import unittest

import updatesupport as us
from examples import ml_eval_stability as example


class MlEvalStabilityExampleTests(unittest.TestCase):
    def test_synthetic_eval_report_crosses_leaderboard_decision_threshold(self):
        rows = example.synthetic_eval_rows()
        report = example.build_public_report(rows)
        verdict = example.build_leaderboard_claim(rows)

        self.assertEqual(len(rows), 13)
        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertGreater(report.observed_value, 0.0)
        self.assertLess(report.interval.lower, 0.0)
        self.assertGreater(report.interval.upper, 0.0)
        self.assertFalse(report.public_adequate)
        self.assertEqual(report.refinements[0].column, "prompt_template")

        self.assertEqual(verdict.status, "fail")
        self.assertIsNotNone(verdict.decision)
        self.assertFalse(verdict.decision.invariant)
        self.assertIsNotNone(verdict.decision_repair_candidate)
        self.assertEqual(
            verdict.decision_repair_candidate.added_columns,
            ("source_dataset",),
        )

    def test_synthetic_eval_report_renders_markdown(self):
        markdown = example.render_report(include_frontier=False)

        self.assertIn("# AI / ML Evaluation Stability Example", markdown)
        self.assertIn("Observed challenger-minus-baseline margin", markdown)
        self.assertIn("Leaderboard Decision Audit", markdown)
        self.assertIn("Decision invariant: no", markdown)
        self.assertIn("base + source_dataset", markdown)


if __name__ == "__main__":
    unittest.main()
