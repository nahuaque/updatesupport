from __future__ import annotations

import unittest

import updatesupport as us


class PublicDescentReportTests(unittest.TestCase):
    def test_public_descent_report_summarizes_observed_law_ambiguity(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
        ]

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            candidate_refinements=["hidden"],
            q=us.q_bounded_shift(0.5),
            top=1,
            min_cell_weight=1,
            title="Demo Report",
            target_description="Pr(label is 1)",
        )

        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertAlmostEqual(report.observed_value, 0.5)
        self.assertEqual(report.grouped.q_name, "bounded_shift(radius=0.5)")
        self.assertAlmostEqual(report.interval.lower, 0.35)
        self.assertAlmostEqual(report.interval.upper, 0.65)
        self.assertAlmostEqual(report.interval.diameter, 0.3)
        self.assertFalse(report.public_adequate)
        self.assertEqual(report.fibers[0].public_value, ("A",))
        self.assertAlmostEqual(report.fibers[0].contribution, 0.3)
        self.assertEqual(report.refinements[0].column, "hidden")
        self.assertAlmostEqual(report.refinements[0].before_ambiguity, 0.3)
        self.assertAlmostEqual(report.refinements[0].after_ambiguity, 0.0)
        self.assertAlmostEqual(report.refinements[0].diameter, 0.0)
        self.assertAlmostEqual(report.refinements[0].reduction, 0.3)
        self.assertAlmostEqual(report.refinements[0].reduction_percent, 100.0)
        self.assertAlmostEqual(report.refinements[0].reduction_fraction, 1.0)

    def test_public_descent_report_renders_markdown(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
        ]

        markdown = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            candidate_refinements=["hidden"],
            top=1,
            title="Demo Report",
        ).to_markdown()

        self.assertIn("# Demo Report", markdown)
        self.assertIn("- Observed value: 0.5000", markdown)
        self.assertIn("Observed-law partial-ID interval: [0.2000, 0.8000]", markdown)
        self.assertIn("## Statistical Interpretation", markdown)
        self.assertIn("not a sampling confidence interval", markdown)
        self.assertIn("## Worst Public Fibers", markdown)
        self.assertIn("measurement-value table", markdown)
        self.assertIn("before=0.6000, after=0.0000", markdown)
        self.assertIn("reduction_pct=100.0%", markdown)
        self.assertIn("## Analyst Notes", markdown)

    def test_recommend_refinements_returns_before_after_and_percent_reduction(self):
        rows = [
            {"public": "A", "hidden": "x", "noise": "n", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "noise": "n", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "noise": "n", "target": 0.5, "weight": 40},
        ]

        candidates = us.recommend_refinements(
            rows,
            public=["public"],
            hidden=["public", "hidden", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "hidden"],
        )

        self.assertEqual(candidates[0].column, "hidden")
        self.assertAlmostEqual(candidates[0].before_ambiguity, 0.6)
        self.assertAlmostEqual(candidates[0].after_ambiguity, 0.0)
        self.assertAlmostEqual(candidates[0].reduction, 0.6)
        self.assertAlmostEqual(candidates[0].reduction_percent, 100.0)
        self.assertEqual(candidates[1].column, "noise")
        self.assertAlmostEqual(candidates[1].before_ambiguity, 0.6)
        self.assertAlmostEqual(candidates[1].after_ambiguity, 0.6)
        self.assertAlmostEqual(candidates[1].reduction, 0.0)
        self.assertAlmostEqual(candidates[1].reduction_percent, 0.0)

    def test_public_descent_report_accepts_precompiled_grouped_problem(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
        ]
        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
        )

        report = us.public_descent_report(grouped, source_data=rows, top=2)

        self.assertIs(report.grouped, grouped)
        self.assertEqual(len(report.fibers), 2)
        self.assertAlmostEqual(report.observed_value, 0.5)

    def test_precompiled_report_requires_source_data_for_refinements(self):
        grouped = us.from_dataframe(
            [{"public": "A", "hidden": "x", "target": 1.0}],
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
        )

        with self.assertRaisesRegex(ValueError, "source_data is required"):
            us.public_descent_report(
                grouped,
                candidate_refinements=["hidden"],
            )


if __name__ == "__main__":
    unittest.main()
