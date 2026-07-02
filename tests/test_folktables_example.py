from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import updatesupport as us

from examples.benchmark_gallery import generate_benchmark_gallery
from examples.acic_2016 import (
    EFFECT_COLUMN as ACIC_EFFECT_COLUMN,
    attach_oracle_effects,
    estimate_acic_effects_with_econml,
    render_acic_report,
    synthetic_acic_2016_source_rows,
)
from examples.folktables_acs import (
    TARGET_COLUMN,
    build_problem_from_rows,
    fiber_diagnostics,
    refinement_candidates,
    render_frontier,
    render_report,
)
from examples.folktables_acs_causal import (
    EFFECT_COLUMN,
    estimate_effects_with_econml,
    render_causal_report,
    synthetic_causal_source_rows,
)


class FakeEconMLEstimator:
    def fit(self, y, treatment, *, X, sample_weight=None, inference="auto"):
        self.y = y
        self.treatment = treatment
        self.x_shape = X.shape
        self.sample_weight = sample_weight
        self.inference = inference
        return self

    def effect(self, X):
        return [0.05 + 0.001 * index for index in range(X.shape[0])]


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
        self.assertIn("# Folktables ACSIncome Public Representation Frontier", report)
        self.assertIn("## Selected Representation Explanation", report)

    def test_folktables_frontier_case_study_renders_choice_diagnostics(self):
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

        markdown = render_frontier(
            task="income",
            rows=rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden", "noise"],
            candidate_columns=["noise", "hidden"],
            weight_column="weight",
            ambiguity_limit=0.05,
        )

        self.assertIn("# Folktables ACSIncome Public Representation Frontier", markdown)
        self.assertIn("Search mode: beam", markdown)
        self.assertIn("base + hidden", markdown)
        self.assertIn("Selected vs baseline max ambiguity", markdown)
        self.assertIn("### Close Dominated Alternatives", markdown)

    def test_causal_example_builds_econml_effect_targets(self):
        rows, public_columns, hidden_columns, _candidate_columns = (
            synthetic_causal_source_rows()
        )
        estimator = FakeEconMLEstimator()

        result = estimate_effects_with_econml(
            rows,
            feature_columns=hidden_columns,
            estimator_factory=lambda _random_state: estimator,
        )

        self.assertEqual(result.source_rows, len(rows))
        self.assertEqual(len(result.rows), len(rows))
        self.assertEqual(result.feature_columns, hidden_columns)
        self.assertGreater(len(result.design_columns), 0)
        self.assertEqual(estimator.x_shape[0], len(rows))
        self.assertEqual(len(estimator.sample_weight), len(rows))
        self.assertIsNone(estimator.inference)
        self.assertAlmostEqual(result.rows[0][EFFECT_COLUMN], 0.05)
        self.assertAlmostEqual(result.rows[-1][EFFECT_COLUMN], 0.05 + 0.001 * 95)

    def test_causal_example_report_shows_handoff_to_updatesupport(self):
        rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_causal_source_rows()
        )
        effect_result = estimate_effects_with_econml(
            rows,
            feature_columns=hidden_columns,
            estimator_factory=lambda _random_state: FakeEconMLEstimator(),
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
        self.assertIn("Effect estimator: EconML FakeEconMLEstimator", markdown)
        self.assertIn("`__tau_hat__ = estimator.effect(X)`", markdown)
        self.assertIn("`__tau_hat__`", markdown)
        self.assertIn("## Update-Support Question", markdown)
        self.assertIn("# Representation Stability Audit", markdown)
        self.assertIn("Observed weighted effect estimate", markdown)

    def test_acic_synthetic_rows_support_oracle_effect_audit(self):
        rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_acic_2016_source_rows()
        )

        result = attach_oracle_effects(
            rows,
            feature_columns=hidden_columns,
        )
        markdown = render_acic_report(
            effect_result=result,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            candidate_columns=candidate_columns,
            min_cell_weight=1,
            q=us.q_bounded_shift(0.5),
            top=2,
        )

        self.assertGreater(len(rows), 0)
        self.assertGreater(len(public_columns), 0)
        self.assertGreater(len(candidate_columns), 0)
        self.assertEqual(result.effect_source, "oracle")
        self.assertIn(ACIC_EFFECT_COLUMN, result.rows[0])
        self.assertIn("# ACIC 2016 Causal Benchmark Demo", markdown)
        self.assertIn("Inferential group: treated", markdown)
        self.assertIn("# ACIC 2016 Representation Stability Audit", markdown)

    def test_acic_econml_path_builds_effect_targets_with_fake_estimator(self):
        rows, _public_columns, hidden_columns, _candidate_columns = (
            synthetic_acic_2016_source_rows()
        )
        estimator = FakeEconMLEstimator()

        result = estimate_acic_effects_with_econml(
            rows,
            feature_columns=hidden_columns,
            estimator_factory=lambda _random_state: estimator,
        )

        self.assertEqual(result.source_rows, len(rows))
        self.assertEqual(result.effect_source, "econml")
        self.assertEqual(result.estimator_name, "FakeEconMLEstimator")
        self.assertEqual(estimator.x_shape[0], len(rows))
        self.assertIsNone(estimator.inference)
        self.assertAlmostEqual(result.rows[0][ACIC_EFFECT_COLUMN], 0.05)

    def test_benchmark_gallery_generates_reproducible_markdown_index(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gallery"
            missing_acic = Path(directory) / "missing_acic.csv"

            reports = generate_benchmark_gallery(
                output_dir=output_dir,
                acic_csv=missing_acic,
                include_real_folktables=False,
                skip_acic_econml=True,
            )

            statuses = {report.slug: report.status for report in reports}
            self.assertEqual(
                statuses["folktables_acs_income_synthetic"],
                "generated",
            )
            self.assertEqual(
                statuses["folktables_acs_causal_synthetic"],
                "generated",
            )
            self.assertEqual(statuses["acic_2016_oracle"], "skipped")
            self.assertTrue((output_dir / "index.md").exists())
            self.assertTrue(
                (output_dir / "folktables_acs_income_synthetic.md").exists()
            )
            self.assertTrue(
                (output_dir / "folktables_acs_causal_synthetic.md").exists()
            )
            folktables_report = (
                output_dir / "folktables_acs_income_synthetic.md"
            ).read_text(encoding="utf-8")
            index = (output_dir / "index.md").read_text(encoding="utf-8")
            self.assertIn("Public Representation Frontier", folktables_report)
            self.assertIn("Selected Representation Explanation", folktables_report)
            self.assertIn("updatesupport Benchmark Gallery", index)
            self.assertIn("ACIC CSV not found", index)


if __name__ == "__main__":
    unittest.main()
