from __future__ import annotations

import unittest

import updatesupport as us


def _require_cvxpy_solver(name: str) -> None:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise unittest.SkipTest("cvxpy extra is not installed") from exc

    installed = {solver.upper() for solver in cp.installed_solvers()}
    if name.upper() not in installed:
        raise unittest.SkipTest(f"CVXPY solver {name!r} is not installed")


def _ratio_grouped_problem() -> us.GroupedProblem:
    states = (("A", "low"), ("A", "high"), ("B", "anchor"))
    public_map = {state: (state[0],) for state in states}
    target = us.RatioTarget(
        numerator={
            ("A", "low"): 1.0,
            ("A", "high"): 4.0,
            ("B", "anchor"): 10.0,
        },
        denominator={
            ("A", "low"): 1.0,
            ("A", "high"): 2.0,
            ("B", "anchor"): 5.0,
        },
        name="loss_ratio",
        description="loss divided by exposure",
    )
    public_law = {("A",): 0.5, ("B",): 0.5}
    cell_weights = {
        ("A", "low"): 0.25,
        ("A", "high"): 0.25,
        ("B", "anchor"): 0.5,
    }
    problem = us.FiniteProblem(
        states=states,
        public=public_map,
        estimand=target,
        environments=us.PublicFiberSaturated.fixed(public_law),
    )
    return us.GroupedProblem(
        problem=problem,
        public_law=public_law,
        public_columns=("segment",),
        hidden_columns=("segment", "risk"),
        target_column="loss_ratio",
        target_functional=target,
        total_weight=1.0,
        cell_weights=cell_weights,
        q_name="saturated",
        q_description="fixed public-law saturated ratio stress test",
    )


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

    def test_public_descent_report_adds_estimator_uncertainty_adjustment(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "se": 0.1, "weight": 30},
            {"public": "A", "hidden": "y", "target": 1.0, "se": 0.2, "weight": 30},
            {"public": "B", "hidden": "z", "target": 0.5, "se": 0.3, "weight": 40},
        ]

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            target_standard_error="se",
            weight="weight",
            target_confidence_multiplier=2.0,
            top=2,
        )
        uncertainty = report.estimator_uncertainty
        markdown = report.to_markdown()
        payload = report.as_dict()
        tables = report.to_tables()

        self.assertIsInstance(
            report.grouped.target_functional, us.UncertainLinearTarget
        )
        self.assertIsInstance(uncertainty, us.EstimatorUncertaintyAdjustment)
        self.assertAlmostEqual(report.interval.lower, 0.2)
        self.assertAlmostEqual(report.interval.upper, 0.8)
        self.assertAlmostEqual(uncertainty.base_diameter, 0.6)
        self.assertAlmostEqual(
            uncertainty.lower_endpoint_standard_error,
            ((0.6 * 0.1) ** 2 + (0.4 * 0.3) ** 2) ** 0.5,
        )
        self.assertAlmostEqual(
            uncertainty.upper_endpoint_standard_error,
            ((0.6 * 0.2) ** 2 + (0.4 * 0.3) ** 2) ** 0.5,
        )
        self.assertAlmostEqual(
            uncertainty.conservative_standard_error_bound,
            ((0.6 * 0.2) ** 2 + (0.4 * 0.3) ** 2) ** 0.5,
        )
        self.assertAlmostEqual(
            uncertainty.conservative_lower,
            0.2 - 2.0 * uncertainty.conservative_standard_error_bound,
        )
        self.assertAlmostEqual(
            uncertainty.conservative_upper,
            0.8 + 2.0 * uncertainty.conservative_standard_error_bound,
        )
        self.assertIn("## Estimator-Uncertainty-Aware Hidden Ambiguity", markdown)
        self.assertIn("Estimator-uncertainty-aware interval", markdown)
        self.assertEqual(
            payload["estimator_uncertainty"]["method"], "endpoint_and_conservative"
        )
        self.assertIn("estimator_uncertainty", tables)
        self.assertTrue(tables["summary"][0]["has_estimator_uncertainty"])

    def test_public_descent_report_includes_data_diagnostics(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 1},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 10},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 10},
        ]

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            candidate_refinements=["public", "hidden", "missing"],
            min_cell_weight=2,
        )
        markdown = report.to_markdown()
        payload = report.as_dict()

        codes = {row.code for row in report.diagnostics}
        self.assertIn("min_cell_weight_dropped_cells", codes)
        self.assertIn("candidate_refinement_already_public", codes)
        self.assertIn("candidate_refinement_not_hidden", codes)
        self.assertIn("## Data Diagnostics", markdown)
        self.assertIn("candidate_refinement_not_hidden", markdown)
        self.assertIn("diagnostics", payload)

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
        self.assertIn("## Target Contract", markdown)
        self.assertIn("Type: linear target", markdown)
        self.assertIn("psi(q) = sum_d h(d) q(d)", markdown)
        self.assertIn(
            "Hidden-cell target values fixed after compilation: yes", markdown
        )
        self.assertIn("Nonlinear, ratio, or representation-dependent targets", markdown)
        self.assertIn("## Statistical Interpretation", markdown)
        self.assertIn("## What This Report Separates", markdown)
        self.assertIn("Causal estimate / reported value", markdown)
        self.assertIn("Statistical uncertainty", markdown)
        self.assertIn("Hidden-composition ambiguity", markdown)
        self.assertIn("Public refinement recommendations", markdown)
        self.assertIn("not a sampling confidence interval", markdown)
        self.assertIn("## CVXPY Dual Diagnostics", markdown)
        self.assertIn("No CVXPY dual diagnostics are available", markdown)
        self.assertIn("## Limitations", markdown)
        self.assertIn("does not identify causal effects", markdown)
        self.assertIn("## Worst Public Fibers", markdown)
        self.assertIn("measurement-value table", markdown)
        self.assertIn("reporting and measurement recommendations", markdown)
        self.assertIn("before=0.6000, after=0.0000", markdown)
        self.assertIn("reduction_pct=100.0%", markdown)
        self.assertIn("## Analyst Notes", markdown)

    def test_witness_report_shows_endpoint_hidden_composition_shift(self):
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
            top=2,
        )
        witness = report.witness_report(top=2)
        raw_witness = us.witness_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
        )
        helper_witness = us.witness_report(report, top=1)
        markdown = witness.to_markdown()
        payload = witness.as_dict()
        tables = witness.to_tables()

        self.assertIsInstance(witness, us.WitnessReport)
        self.assertAlmostEqual(raw_witness.ambiguity, witness.ambiguity)
        self.assertEqual(helper_witness.top, 1)
        self.assertTrue(witness.public_law_match)
        self.assertTrue(witness.additive_contributions)
        self.assertAlmostEqual(witness.observed_value, 0.5)
        self.assertAlmostEqual(witness.lower_value, 0.2)
        self.assertAlmostEqual(witness.upper_value, 0.8)
        self.assertAlmostEqual(witness.ambiguity, 0.6)

        fiber_by_label = {row.public_label: row for row in witness.fibers}
        fiber_a = fiber_by_label["public=A"]
        self.assertEqual(fiber_a.hidden_cells, 2)
        self.assertAlmostEqual(fiber_a.lower_public_mass, 0.6)
        self.assertAlmostEqual(fiber_a.upper_public_mass, 0.6)
        self.assertAlmostEqual(fiber_a.public_mass_difference, 0.0)
        self.assertAlmostEqual(fiber_a.total_abs_mass_shift, 1.2)
        self.assertAlmostEqual(fiber_a.contribution_shift, 0.6)

        cells = {row.state_label: row for row in witness.cells}
        self.assertAlmostEqual(cells["public=A, hidden=x"].lower_mass, 0.6)
        self.assertAlmostEqual(cells["public=A, hidden=x"].upper_mass, 0.0)
        self.assertAlmostEqual(cells["public=A, hidden=x"].mass_shift, -0.6)
        self.assertAlmostEqual(cells["public=A, hidden=y"].lower_mass, 0.0)
        self.assertAlmostEqual(cells["public=A, hidden=y"].upper_mass, 0.6)
        self.assertAlmostEqual(cells["public=A, hidden=y"].mass_shift, 0.6)
        self.assertAlmostEqual(cells["public=A, hidden=y"].contribution_shift, 0.6)

        self.assertIn("# Adversarial Witness Report", markdown)
        self.assertIn("same public report cannot distinguish", markdown)
        self.assertIn("Same public distribution: yes", markdown)
        self.assertIn("## Public-Fiber Check", markdown)
        self.assertIn("## Largest Hidden-Cell Shifts", markdown)
        self.assertIn("| public=A | 2 | 0.6000 | 0.6000 | 0.0000 |", markdown)
        self.assertEqual(payload["ambiguity"], witness.ambiguity)
        self.assertIn("cell_shifts", payload)
        self.assertEqual(
            set(tables),
            {"summary", "fiber_shifts", "cell_shifts"},
        )
        self.assertEqual(tables["summary"][0]["title"], "Adversarial Witness Report")
        self.assertEqual(
            tables["cell_shifts"][0]["state_label"], witness.cells[0].state_label
        )

    def test_public_descent_report_explains_scip_mip_dual_absence(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {"public": "A", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "A", "hidden": "high", "target": 1.0, "weight": 25},
            {"public": "B", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "B", "hidden": "high", "target": 1.0, "weight": 25},
        ]

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q=us.q_fiber_support_floor(2, min_share=0.25),
        )
        markdown = report.to_markdown()

        self.assertAlmostEqual(report.interval.lower, 0.25, places=5)
        self.assertAlmostEqual(report.interval.upper, 0.75, places=5)
        self.assertIn("mixed-integer `fiber_support_floor`", markdown)
        self.assertIn("solver `SCIP`", markdown)
        self.assertIn("primal interval and witness distributions", markdown)

    def test_public_descent_report_includes_procedure_target_metadata(self):
        rows = [
            {"public": "A", "hidden": "x", "base": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "base": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "base": 0.5, "weight": 40},
        ]
        compiled_public = []

        def compiler(context):
            scale = len(context.public)
            compiled_public.append(context.public)
            return us.row_metric(
                f"procedure_value_x{scale}",
                lambda row, scale=scale: scale * float(row["base"]),
                columns=("base",),
                description=f"procedure value scaled by {scale}",
            )

        target = us.ProcedureTarget(
            "reporting_procedure",
            compiler,
            description="representation-dependent reporting target",
        )

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target=target,
            weight="weight",
            candidate_refinements=["hidden"],
            top=1,
        )
        markdown = report.to_markdown()
        payload = report.as_dict()

        self.assertIs(report.grouped.target_procedure, target)
        self.assertEqual(report.grouped.target_column.name, "procedure_value_x1")
        self.assertEqual(report.refinements[0].column, "hidden")
        self.assertEqual(
            compiled_public,
            [("public",), ("public",), ("public", "hidden")],
        )
        self.assertIn("## Procedure Target", markdown)
        self.assertIn("Procedure: `reporting_procedure`", markdown)
        self.assertIn("Compiled target: `procedure_value_x1`", markdown)
        self.assertIn("procedure-comparison sensitivity analyses", markdown)
        self.assertEqual(payload["target_procedure"]["name"], "reporting_procedure")
        self.assertEqual(payload["target"], "procedure_value_x1")

    def test_ratio_report_gates_nonadditive_decomposition(self):
        report = us.public_descent_report(_ratio_grouped_problem(), top=2)
        markdown = report.to_markdown()
        payload = report.as_dict()

        self.assertFalse(report.fiber_decomposition_available)
        self.assertEqual(report.fiber_diagnostic_kind, "point_range")
        self.assertIsNone(report.top_fiber_contribution)
        self.assertIsNone(report.top_fiber_contribution_share)
        self.assertIsNone(report.fibers[0].contribution)
        self.assertFalse(report.fibers[0].contribution_available)
        self.assertEqual(report.fibers[0].diagnostic_kind, "point_range")
        self.assertAlmostEqual(report.observed_value, 25 / 13)
        self.assertIn("Public-fiber contribution decomposition: not additive", markdown)
        self.assertIn("## Public Fiber Point Ranges", markdown)
        self.assertIn("point_range=", markdown)
        self.assertNotIn("contribution=0.0000", markdown)
        self.assertFalse(payload["fiber_decomposition_available"])
        self.assertEqual(payload["fiber_diagnostic_kind"], "point_range")
        self.assertIsNone(payload["top_fiber_contribution"])
        self.assertIsNone(payload["top_fiber_contribution_share"])
        self.assertIsNone(payload["top_fibers"][0]["contribution"])

    def test_causal_reporting_stability_suite_packages_standard_outputs(self):
        rows = [
            {"public": "A", "hidden": "x", "noise": "n1", "effect": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "noise": "n2", "effect": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "noise": "n1", "effect": 0.5, "weight": 40},
        ]

        suite = us.causal_reporting_stability(
            rows,
            public=["public"],
            hidden=["public", "hidden", "noise"],
            effect="effect",
            weight="weight",
            candidate_refinements=["hidden", "noise"],
            q=us.q_bounded_shift(0.5),
            min_cell_weight=1,
            sensitivity_min_cell_weights=[1],
            sensitivity_q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
            statistical_estimate=0.5,
            statistical_standard_error=0.1,
            statistical_interval=(0.3, 0.7),
            statistical_confidence_level=0.95,
            statistical_method="bootstrap",
            top=2,
        )
        markdown = suite.to_markdown()
        payload = suite.as_dict()

        self.assertIsInstance(suite, us.CausalReportingStabilitySuite)
        self.assertIsInstance(suite.statistical_uncertainty, us.StatisticalUncertainty)
        self.assertIsInstance(suite.primary, us.PublicDescentReport)
        self.assertIsInstance(suite.sensitivity, us.SensitivityReport)
        self.assertIsInstance(
            suite.refinement_sensitivity,
            us.RefinementSensitivityReport,
        )
        self.assertAlmostEqual(suite.primary.observed_value, 0.5)
        self.assertAlmostEqual(suite.primary.interval.diameter, 0.3)
        self.assertEqual(suite.sensitivity.summary.scenario_count, 3)
        self.assertEqual(suite.refinement_sensitivity.candidates[0].column, "hidden")
        self.assertEqual(payload["primary"]["q_name"], "bounded_shift(radius=0.5)")
        self.assertIn("# Causal Reporting Stability Suite", markdown)
        self.assertIn("## What This Suite Separates", markdown)
        self.assertIn("Causal estimate", markdown)
        self.assertIn("Statistical uncertainty", markdown)
        self.assertIn("Hidden-composition ambiguity", markdown)
        self.assertIn("Public refinement recommendations", markdown)
        self.assertIn("bootstrap", markdown)
        self.assertIn("## Causal Estimate", markdown)
        self.assertIn("## Statistical Uncertainty", markdown)
        self.assertIn("## Hidden-Composition Ambiguity", markdown)
        self.assertIn("## Sensitivity Scenarios", markdown)
        self.assertIn("## Refinement Recommendations", markdown)
        self.assertIn("## CVXPY Dual Diagnostics", markdown)
        self.assertIn("## Limitations", markdown)
        self.assertIn("No CVXPY dual diagnostics are available", markdown)
        self.assertIn("## Robustness Grid", markdown)
        self.assertIn("## Sensitivity-Aware Refinements", markdown)

    def test_sensitivity_report_compares_saturated_and_scip_support_floor(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {"public": "A", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "A", "hidden": "high", "target": 1.0, "weight": 25},
            {"public": "B", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "B", "hidden": "high", "target": 1.0, "weight": 25},
        ]

        report = us.sensitivity_report(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q_presets=[
                "saturated",
                us.q_fiber_support_floor(2, min_share=0.25),
            ],
        )
        markdown = report.to_markdown()

        self.assertEqual(len(report.rows), 2)
        self.assertAlmostEqual(report.rows[0].ambiguity, 1.0, places=5)
        self.assertAlmostEqual(report.rows[1].ambiguity, 0.5, places=5)
        self.assertEqual(
            report.rows[1].q_name,
            "fiber_support_floor(min_active=2, min_share=0.25)",
        )
        self.assertIn("fiber_support_floor(min_active=2, min_share=0.25)", markdown)

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

    def test_recommend_refinements_can_use_residopt_screening(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

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
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            screening_backend="residopt",
            ambiguity_limit=1.0,
        )

        self.assertEqual(candidates[0].column, "hidden")
        self.assertEqual(candidates[0].screening_backend, "residopt")
        self.assertEqual(candidates[0].screening_status, "screen_certified")
        self.assertTrue(candidates[0].screening_certified)
        self.assertTrue(candidates[0].screening_exact_solve_avoided)
        self.assertEqual(
            candidates[0].after_ambiguity_bound_type,
            "conservative_upper_bound",
        )

    def test_public_descent_report_renders_refinement_screening_metadata(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        rows = [
            {"public": "A", "hidden": "x", "noise": "n", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "noise": "n", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "noise": "n", "target": 0.5, "weight": 40},
        ]

        report = us.public_descent_report(
            rows,
            public=["public"],
            hidden=["public", "hidden", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "hidden"],
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            refinement_screening_backend="residopt",
            refinement_ambiguity_limit=1.0,
        )
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertIsNotNone(report.refinement_screening)
        self.assertIn("Refinement screening used", markdown)
        self.assertIn("screening_status=screen_certified", markdown)
        self.assertTrue(tables["summary"][0]["has_refinement_screening"])
        self.assertIn("refinement_screening_summary", tables)
        self.assertIn("refinement_screening_candidates", tables)

    def test_recommend_refinements_sensitivity_aggregates_grid(self):
        rows = [
            {"public": "A", "hidden": "x", "noise": "n", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "noise": "n", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "noise": "n", "target": 0.5, "weight": 40},
        ]

        report = us.recommend_refinements_sensitivity(
            rows,
            public=["public"],
            hidden=["public", "hidden", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "hidden"],
            q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
            top=None,
        )
        markdown = report.to_markdown()

        self.assertIsInstance(report, us.RefinementSensitivityReport)
        self.assertEqual(len(report.scenarios), 3)
        self.assertEqual(len(report.rows), 6)
        self.assertEqual(len(report.candidates), 2)
        self.assertEqual(report.candidates[0].column, "hidden")
        self.assertAlmostEqual(report.candidates[0].mean_reduction, 0.3)
        self.assertAlmostEqual(report.candidates[0].min_reduction, 0.0)
        self.assertAlmostEqual(report.candidates[0].max_reduction, 0.6)
        self.assertAlmostEqual(
            report.candidates[0].mean_reduction_percent,
            200.0 / 3.0,
        )
        self.assertEqual(report.candidates[0].positive_reduction_scenarios, 2)
        self.assertAlmostEqual(report.candidates[0].positive_reduction_share, 2 / 3)
        self.assertEqual(report.candidates[0].best_rank, 1)
        self.assertAlmostEqual(report.candidates[0].mean_rank, 4 / 3)
        self.assertEqual(report.candidates[0].worst_rank, 2)
        self.assertEqual(report.candidates[0].top_rank_count, 2)
        self.assertEqual(report.candidates[0].rank_range, 1)
        self.assertEqual(report.candidates[1].column, "noise")
        self.assertAlmostEqual(report.candidates[1].mean_reduction, 0.0)
        self.assertEqual(report.scenarios[0].best_column, "hidden")
        self.assertAlmostEqual(report.scenarios[0].baseline_ambiguity, 0.6)
        self.assertIn("# Public Refinement Sensitivity Report", markdown)
        self.assertIn("## Aggregate Summary", markdown)
        self.assertIn("Top aggregate refinement", markdown)
        self.assertIn("## Aggregate Refinement Ranking", markdown)
        self.assertIn("## Scenario Summary", markdown)
        self.assertIn("changes rank across scenarios", markdown)

    def test_recommend_refinements_sensitivity_reports_failed_scenarios(self):
        report = us.recommend_refinements_sensitivity(
            [{"public": "A", "hidden": "x", "target": 1.0}],
            public=["public"],
            hidden=["public", "hidden"],
            hidden_sets=[["hidden"]],
            target="target",
            candidate_refinements=["hidden"],
            q_presets=["saturated"],
        )
        markdown = report.to_markdown()

        self.assertEqual(len(report.successful_scenarios), 0)
        self.assertEqual(len(report.failed_scenarios), 1)
        self.assertEqual(len(report.candidates), 0)
        self.assertIn("No refinement scenario completed successfully", markdown)
        self.assertIn("error: public columns must also be hidden columns", markdown)

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

    def test_audit_effects_wraps_public_descent_with_effect_labels(self):
        rows = [
            {"public": "A", "hidden": "x", "tau_hat": -0.1, "weight": 30},
            {"public": "A", "hidden": "y", "tau_hat": 0.3, "weight": 30},
            {"public": "B", "hidden": "z", "tau_hat": 0.2, "weight": 40},
        ]

        report = us.audit_effects(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            candidate_refinements=["hidden"],
            q=us.q_bounded_shift(0.5),
            min_cell_weight=1,
            top=1,
        )
        markdown = report.to_markdown()

        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertEqual(report.title, "Causal Effect Representation Stability Audit")
        self.assertEqual(report.target_description, "estimated treatment effect")
        self.assertEqual(report.observed_label, "Observed effect estimate")
        self.assertAlmostEqual(report.observed_value, 0.14)
        self.assertAlmostEqual(report.interval.lower, 0.08)
        self.assertAlmostEqual(report.interval.upper, 0.20)
        self.assertIn("- Observed effect estimate: 0.1400", markdown)
        self.assertIn("aggregate estimated treatment effect", markdown)
        self.assertIn("where the causal estimator enters", markdown)
        self.assertIn("does not identify the causal graph", markdown)
        self.assertIn("not causal adjustment recommendations", markdown)

    def test_audit_effects_accepts_effect_column_alias(self):
        report = us.audit_effects(
            [{"public": "A", "hidden": "x", "effect": 1.0}],
            public=["public"],
            hidden=["public", "hidden"],
            effect_column="effect",
        )

        self.assertAlmostEqual(report.observed_value, 1.0)

    def test_audit_effects_accepts_effect_standard_error_alias(self):
        report = us.audit_effects(
            [
                {"public": "A", "hidden": "x", "effect": 0.0, "se": 0.1},
                {"public": "A", "hidden": "y", "effect": 1.0, "se": 0.2},
            ],
            public=["public"],
            hidden=["public", "hidden"],
            effect="effect",
            effect_standard_error_column="se",
            effect_confidence_multiplier=1.0,
        )

        self.assertIsInstance(
            report.estimator_uncertainty, us.EstimatorUncertaintyAdjustment
        )
        self.assertAlmostEqual(
            report.estimator_uncertainty.conservative_standard_error_bound,
            0.2,
        )

    def test_audit_effects_requires_effect_for_raw_data(self):
        with self.assertRaisesRegex(TypeError, "missing required keyword"):
            us.audit_effects(
                [{"public": "A", "hidden": "x", "effect": 1.0}],
                public=["public"],
                hidden=["public", "hidden"],
            )

    def test_audit_effects_rejects_conflicting_effect_aliases(self):
        with self.assertRaisesRegex(TypeError, "use either 'effect'"):
            us.audit_effects(
                [{"public": "A", "hidden": "x", "effect": 1.0}],
                public=["public"],
                hidden=["public", "hidden"],
                effect="effect",
                effect_column="other_effect",
            )


if __name__ == "__main__":
    unittest.main()
