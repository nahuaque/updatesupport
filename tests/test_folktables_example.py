from __future__ import annotations

import unittest

import updatesupport as us

from examples.folktables_acs import (
    TARGET_COLUMN,
    build_problem_from_rows,
    fiber_diagnostics,
    refinement_candidates,
    render_report,
)
from examples.folktables_acs_causal import (
    EFFECT_COLUMN,
    build_stratified_effect_rows,
    render_causal_report,
    synthetic_causal_source_rows,
)


class FolktablesExampleTests(unittest.TestCase):
    def test_grouped_problem_computes_observed_law_ambiguity(self):
        rows = [
            {"public": "A", "hidden": "x", TARGET_COLUMN: 0.0, "weight": 30},
            {"public": "A", "hidden": "y", TARGET_COLUMN: 1.0, "weight": 30},
            {"public": "B", "hidden": "z", TARGET_COLUMN: 0.5, "weight": 40},
        ]

        grouped = build_problem_from_rows(
            rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden"],
            weight_column="weight",
        )

        interval = grouped.problem.global_transport_modulus()
        diagnostics = fiber_diagnostics(grouped, top=1)

        self.assertIsInstance(grouped, us.GroupedProblem)
        self.assertFalse(grouped.problem.is_public_adequate())
        self.assertAlmostEqual(interval.lower, 0.2)
        self.assertAlmostEqual(interval.upper, 0.8)
        self.assertAlmostEqual(interval.diameter, 0.6)
        self.assertEqual(diagnostics[0]["public_value"], ("A",))
        self.assertAlmostEqual(diagnostics[0]["contribution"], 0.6)

    def test_refinement_candidates_rank_hidden_split(self):
        rows = [
            {
                "public": "A",
                "hidden": "x",
                "noise": "n",
                TARGET_COLUMN: 0.0,
                "weight": 30,
            },
            {
                "public": "A",
                "hidden": "y",
                "noise": "n",
                TARGET_COLUMN: 1.0,
                "weight": 30,
            },
            {
                "public": "B",
                "hidden": "z",
                "noise": "n",
                TARGET_COLUMN: 0.5,
                "weight": 40,
            },
        ]

        candidates = refinement_candidates(
            rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden", "noise"],
            candidate_columns=["hidden", "noise"],
            weight_column="weight",
        )

        self.assertEqual(candidates[0]["column"], "hidden")
        self.assertAlmostEqual(candidates[0]["before_ambiguity"], 0.6)
        self.assertAlmostEqual(candidates[0]["after_ambiguity"], 0.0)
        self.assertAlmostEqual(candidates[0]["diameter"], 0.0)
        self.assertAlmostEqual(candidates[0]["reduction"], 0.6)
        self.assertAlmostEqual(candidates[0]["reduction_percent"], 100.0)

    def test_report_includes_stats_analyst_interpretation(self):
        rows = [
            {"public": "A", "hidden": "x", TARGET_COLUMN: 0.0, "weight": 30},
            {"public": "A", "hidden": "y", TARGET_COLUMN: 1.0, "weight": 30},
            {"public": "B", "hidden": "z", TARGET_COLUMN: 0.5, "weight": 40},
        ]
        grouped = build_problem_from_rows(
            rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden"],
            weight_column="weight",
        )

        report = render_report(
            task="income",
            grouped=grouped,
            rows=rows,
            candidate_columns=["hidden"],
            top=1,
            min_cell_weight=1,
        )

        self.assertIn("## Statistical Interpretation", report)
        self.assertIn("not a sampling confidence interval", report)
        self.assertIn("measurement-value table", report)
        self.assertIn("## Analyst Notes", report)

    def test_causal_example_builds_stratified_effect_targets(self):
        rows, public_columns, hidden_columns, _candidate_columns = (
            synthetic_causal_source_rows()
        )

        result = build_stratified_effect_rows(
            rows,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            min_arm_weight=1,
        )

        self.assertEqual(result.source_rows, 10)
        self.assertEqual(result.retained_strata, 5)
        self.assertEqual(result.dropped_strata, 0)
        by_key = {
            tuple(row[column] for column in hidden_columns): row for row in result.rows
        }
        self.assertAlmostEqual(
            by_key[("25_34", "1", "tech", "36_45", "1")][EFFECT_COLUMN],
            0.44,
        )
        self.assertAlmostEqual(
            by_key[("25_34", "1", "service", "21_35", "1")][EFFECT_COLUMN],
            0.06,
        )

    def test_causal_example_report_shows_handoff_to_updatesupport(self):
        rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_causal_source_rows()
        )
        effect_result = build_stratified_effect_rows(
            rows,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            min_arm_weight=1,
        )

        markdown = render_causal_report(
            effect_result=effect_result,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            candidate_columns=candidate_columns,
            treatment_label="BA or graduate degree versus less than BA",
            outcome_label="ACSIncome target label",
            min_cell_weight=1,
            q=us.q_bounded_shift(0.5),
            top=2,
        )

        self.assertIn("# Folktables ACS Causal-Effect Reporting Demo", markdown)
        self.assertIn("## Causal Estimation Step", markdown)
        self.assertIn("`__tau_hat__`", markdown)
        self.assertIn("## Update-Support Question", markdown)
        self.assertIn("# Representation Stability Audit", markdown)
        self.assertIn("Observed weighted effect estimate", markdown)


if __name__ == "__main__":
    unittest.main()
