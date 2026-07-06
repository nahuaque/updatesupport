from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

import updatesupport as us
import updatesupport_finance as usf


def _require_cvxpy() -> None:
    try:
        import cvxpy  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("cvxpy extra is not installed") from exc


def _portfolio_rows():
    return [
        {
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "channel": "broker",
            "employment": "salaried",
            "pd": 0.02,
            "lgd": 0.40,
            "ead": 100_000,
            "defaulted": 0,
        },
        {
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "channel": "direct",
            "employment": "self_employed",
            "pd": 0.05,
            "lgd": 0.50,
            "ead": 50_000,
            "defaulted": 1,
        },
        {
            "product": "card",
            "region": "south",
            "fico_band": "near_prime",
            "ltv_band": "na",
            "channel": "direct",
            "employment": "salaried",
            "pd": 0.08,
            "lgd": 0.70,
            "ead": 10_000,
            "defaulted": 0,
        },
    ]


def _factor_rows():
    return [
        {
            "product": "loan",
            "region": "north",
            "channel": "broker",
            "pd": 0.0,
            "lgd": 1.0,
            "ead": 30.0,
            "macro_factor": -1.0,
        },
        {
            "product": "loan",
            "region": "north",
            "channel": "direct",
            "pd": 1.0,
            "lgd": 1.0,
            "ead": 30.0,
            "macro_factor": 1.0,
        },
        {
            "product": "card",
            "region": "south",
            "channel": "direct",
            "pd": 0.5,
            "lgd": 1.0,
            "ead": 40.0,
            "macro_factor": 0.0,
        },
    ]


class FinancePluginTests(unittest.TestCase):
    def tearDown(self):
        us.unregister_plugin("finance")

    def test_from_portfolio_compiles_expected_loss_metric(self):
        grouped = usf.from_portfolio(
            _portfolio_rows(),
            public=["product", "region", "fico_band", "ltv_band"],
            hidden=[
                "product",
                "region",
                "fico_band",
                "ltv_band",
                "channel",
                "employment",
            ],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            exposure="ead",
        )

        self.assertIsInstance(grouped, us.GroupedProblem)
        self.assertEqual(grouped.target_column.name, "expected_loss_rate")
        self.assertAlmostEqual(
            grouped.problem.estimand_map[
                ("mortgage", "north", "prime", "low", "broker", "salaried")
            ],
            0.008,
        )
        self.assertAlmostEqual(sum(grouped.cell_weights.values()), 1.0)

    def test_disclosure_triangulation_uses_core_named_linear_solver(self):
        growth = usf.rounded_growth_constraints(
            "component_growth",
            current="component_current",
            previous="component_previous",
            growth_percent=55.0,
            rounding=5.0,
            provenance="Example rounded growth disclosure",
            verified=True,
        )
        spec = usf.disclosure_triangulation_spec(
            title="Generic Disclosure Triangulation",
            variables=[
                usf.disclosure_variable("component_previous"),
                usf.disclosure_variable("component_current"),
                usf.disclosure_variable("total_previous"),
                usf.disclosure_variable("total_current"),
            ],
            constraints=[
                usf.exact_disclosure_constraint(
                    "reported_total_previous",
                    "total_previous",
                    100.0,
                    provenance="Example total disclosure",
                    verified=True,
                ),
                usf.exact_disclosure_constraint(
                    "reported_total_current",
                    "total_current",
                    200.0,
                    provenance="Example total disclosure",
                    verified=True,
                ),
                usf.containment_constraint(
                    "previous_containment",
                    child="component_previous",
                    parent="total_previous",
                ),
                usf.containment_constraint(
                    "current_containment",
                    child="component_current",
                    parent="total_current",
                ),
                *growth,
                usf.interval_disclosure_constraint(
                    "current_anchor",
                    "component_current",
                    lower=90.0,
                    category="assumption",
                    provenance="Example analyst assumption",
                ),
            ],
            targets=[
                usf.disclosure_target(
                    "previous_component",
                    "component_previous",
                    label="Previous-period component",
                )
            ],
            tiers=[
                usf.disclosure_tier(
                    "T0 containment",
                    [
                        "reported_total_previous",
                        "reported_total_current",
                        "previous_containment",
                        "current_containment",
                    ],
                ),
                usf.disclosure_tier(
                    "T1 + growth + anchor",
                    [
                        "reported_total_previous",
                        "reported_total_current",
                        "previous_containment",
                        "current_containment",
                        "component_growth_lower",
                        "component_growth_upper",
                        "current_anchor",
                    ],
                ),
            ],
        )

        report = usf.triangulate_disclosure(spec)
        interval = report.interval(
            target="previous_component",
            scenario="T1 + growth + anchor",
        )
        tables = report.to_tables()

        self.assertIsInstance(report, us.NamedLinearFeasibilityReport)
        self.assertAlmostEqual(interval.lower, 56.25)
        self.assertAlmostEqual(interval.upper, 100.0)
        self.assertIn("Disclosure Triangulation", report.to_markdown())
        self.assertIn("triangulate_disclosure", usf.__all__)
        self.assertIn("disclosure_triangulation", usf.plugin.report_profiles)
        self.assertIn("disclosure_triangulation", usf.plugin.compilers)
        self.assertEqual(
            {
                row["kind"]
                for row in tables["active_constraints"]
                if row["scenario"] == "T1 + growth + anchor"
            },
            {
                "assumption",
                "containment",
                "disclosure",
                "rounded_growth_disclosure",
            },
        )

    def test_model_risk_report_uses_finance_labels(self):
        report = usf.model_risk_report(
            _portfolio_rows(),
            public=["product", "region", "fico_band", "ltv_band"],
            hidden=[
                "product",
                "region",
                "fico_band",
                "ltv_band",
                "channel",
                "employment",
            ],
            metric=usf.default_rate(default="defaulted"),
            exposure="ead",
            q=usf.q_portfolio_mix_shift(radius=0.25),
            top=2,
            model_id="PD_MORTGAGE_2026Q2",
            portfolio_name="Retail portfolio",
            as_of_date="2026-06-30",
            intended_use="Expected loss validation",
            ambiguity_limit=0.001,
            public_adequacy_required=True,
        )
        markdown = report.to_markdown()

        self.assertIsInstance(report, usf.ModelRiskReport)
        self.assertIsInstance(report.core, us.PublicDescentReport)
        self.assertEqual(report.review_status, "attention required")
        self.assertIn("transport ambiguity", report.review_reasons[0])
        self.assertIn("Financial Model-Risk Representation Stability Report", markdown)
        self.assertIn("## Model-Risk Context", markdown)
        self.assertIn("Model ID: PD_MORTGAGE_2026Q2", markdown)
        self.assertIn("## Review Status", markdown)
        self.assertIn("Status: attention required", markdown)
        self.assertIn("## Financial Model-Risk Interpretation", markdown)
        self.assertIn("## Reported Portfolio Risk Estimate", markdown)
        self.assertIn("## Supplied Statistical / Model Uncertainty", markdown)
        self.assertIn("## Hidden-Composition Ambiguity", markdown)
        self.assertIn("## Concentration-Stress Ambiguity", markdown)
        self.assertIn("## Public Refinement Recommendations", markdown)
        self.assertIn("## Dual Diagnostics", markdown)
        self.assertIn("## Data Diagnostics", markdown)
        self.assertIn("## Limitations / Reviewer Notes", markdown)
        self.assertIn("## Core Update-Support Audit", markdown)
        self.assertIn("Reported portfolio risk estimate", markdown)
        self.assertIn("bounded_shift(radius=0.25)", markdown)
        self.assertIn("not a concentration-balance preset", markdown)

        tables = report.to_tables()
        self.assertIn("finance_model_risk", tables)
        self.assertIn("finance_concentration_stress", tables)
        self.assertIn("finance_review_reasons", tables)
        self.assertIn("finance_refinement_recommendations", tables)
        self.assertIn("finance_dual_diagnostics", tables)
        self.assertIn("finance_data_diagnostics", tables)
        self.assertIn("finance_limitations", tables)
        self.assertIn("core_summary", tables)

    def test_model_risk_report_surfaces_uncertainty_and_reviewer_notes(self):
        _require_cvxpy()
        rows = [row | {"metric_se": 0.03} for row in _factor_rows()]
        report = usf.model_risk_report(
            rows,
            public=["product"],
            hidden=["product", "channel"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            metric_standard_error="metric_se",
            exposure="ead",
            q=usf.q_factor_exposure_shift(
                0.2,
                rows,
                hidden=["product", "channel"],
                factors={"macro": "macro_factor"},
                exposure="ead",
            ),
            candidate_refinements=["channel"],
            top=1,
            statistical_estimate=0.45,
            statistical_standard_error=0.02,
            statistical_interval=(0.40, 0.50),
            statistical_confidence_level=0.95,
            statistical_method="validation bootstrap",
            reviewer_notes=["Review concentration scenario with portfolio owners."],
            limitations=["Synthetic unit-test portfolio."],
        )
        markdown = report.to_markdown()
        tables = report.to_tables()
        payload = report.as_dict()

        self.assertIsNotNone(report.statistical_uncertainty)
        self.assertIsNotNone(report.core.estimator_uncertainty)
        self.assertEqual(
            payload["concentration_stress"]["stress_type"],
            "factor-exposure concentration stress",
        )
        self.assertIn("External interval: [0.4000, 0.5000]", markdown)
        self.assertIn("validation bootstrap", markdown)
        self.assertIn(
            "Estimator-uncertainty-aware conservative interval",
            markdown,
        )
        self.assertIn("factor-exposure concentration stress", markdown)
        self.assertIn("Reviewer notes:", markdown)
        self.assertIn("Review concentration scenario", markdown)
        self.assertIn("Synthetic unit-test portfolio.", markdown)
        self.assertIn("finance_statistical_uncertainty", tables)
        self.assertIn("finance_estimator_uncertainty", tables)
        self.assertIn("finance_concentration_stress", tables)
        self.assertIn("finance_refinement_recommendations", tables)
        self.assertIn("finance_dual_diagnostics", tables)
        self.assertIn("finance_reviewer_notes", tables)
        self.assertIn("core_estimator_uncertainty", tables)
        self.assertEqual(
            tables["finance_concentration_stress"][0]["stress_type"],
            "factor-exposure concentration stress",
        )

    def test_model_risk_report_can_include_model_assisted_portfolio_uncertainty(
        self,
    ):
        rows = [row | {"pd_se": 0.02, "lgd_se": 0.05} for row in _factor_rows()]
        report = usf.model_risk_report(
            rows,
            public=["product"],
            hidden=["product", "channel"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            metric_standard_error=usf.expected_loss_standard_error(
                pd="pd",
                lgd="lgd",
                pd_standard_error="pd_se",
                lgd_standard_error="lgd_se",
            ),
            exposure="ead",
            q=usf.q_portfolio_mix_shift(radius=0.25),
            ambiguity_limit=1.0,
            statistical_interval=(0.40, 0.60),
            statistical_method="external validation interval",
            composition_uncertainty_draws=8,
            composition_uncertainty_seed=123,
            composition_uncertainty_confidence_level=0.8,
        )
        markdown = report.to_markdown()
        payload = report.as_dict()
        tables = report.to_tables()

        self.assertIsNotNone(report.core.estimator_uncertainty)
        self.assertIsNotNone(report.composition_uncertainty)
        self.assertEqual(report.composition_uncertainty.draw_count, 8)
        self.assertEqual(
            report.composition_uncertainty.joint_model.method,
            "bayesian_bootstrap",
        )
        self.assertIn("## Model-Assisted Portfolio Uncertainty", markdown)
        self.assertIn("Posterior/bootstrap ambiguity", markdown)
        self.assertIn("Supplied statistical/model uncertainty", markdown)
        self.assertIn("Hidden-cell metric standard errors were supplied", markdown)
        self.assertIn("Separation: this section resamples hidden composition", markdown)
        self.assertIn("composition_uncertainty", payload)
        self.assertEqual(payload["composition_uncertainty"]["draw_count"], 8)
        self.assertIn("finance_model_assisted_summary", tables)
        self.assertIn("finance_model_assisted_metric_summaries", tables)
        self.assertIn("finance_model_assisted_draws", tables)
        self.assertIn("finance_model_assisted_joint_cells", tables)
        self.assertIn("finance_estimator_uncertainty", tables)

    def test_model_assisted_portfolio_uncertainty_wraps_core_report(self):
        uncertainty = usf.model_assisted_portfolio_uncertainty(
            _factor_rows(),
            public=["product"],
            hidden=["product", "channel"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            exposure="ead",
            draws=5,
            seed=7,
            q=usf.q_portfolio_mix_shift(radius=0.25),
            ambiguity_limit=1.0,
        )

        self.assertIsInstance(uncertainty, us.HiddenCompositionUncertaintyReport)
        self.assertEqual(uncertainty.draw_count, 5)
        self.assertEqual(uncertainty.target_name, "expected_loss_rate")
        self.assertEqual(uncertainty.q_name, "bounded_shift(radius=0.25)")

    def test_model_risk_report_structured_exports_are_finance_named(self):
        report = usf.model_risk_report(
            _portfolio_rows(),
            public=["product", "region", "fico_band", "ltv_band"],
            hidden=[
                "product",
                "region",
                "fico_band",
                "ltv_band",
                "channel",
                "employment",
            ],
            metric=usf.default_rate(default="defaulted"),
            exposure="ead",
            q=usf.q_portfolio_mix_shift(radius=0.25),
            top=2,
            model_id="PD_EXPORT_001",
            portfolio_name="Export test portfolio",
            as_of_date="2026-06-30",
            intended_use="Governance dashboard export",
            reviewer_notes=["Archive with quarterly validation evidence."],
        )

        payload = json.loads(report.to_json())
        tables = report.to_tables()

        self.assertEqual(payload["metadata"]["model_id"], "PD_EXPORT_001")
        self.assertEqual(payload["review_status"], "pass")
        self.assertIn("reported_portfolio_risk_estimate", payload)
        self.assertIn("concentration_stress", payload)
        self.assertIn("core", payload)
        self.assertEqual(
            tables["finance_model_risk"][0]["model_id"],
            "PD_EXPORT_001",
        )
        self.assertEqual(
            tables["finance_reviewer_notes"][0]["note"],
            "Archive with quarterly validation evidence.",
        )
        self.assertIn("finance_concentration_stress", tables)
        self.assertIn("finance_refinement_recommendations", tables)
        self.assertIn("finance_dual_diagnostics", tables)
        self.assertIn("finance_data_diagnostics", tables)
        self.assertIn("finance_limitations", tables)
        self.assertIn("core_summary", tables)

        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("pandas is not installed") from exc
        frames = report.to_dataframes()
        self.assertIn("finance_model_risk", frames)
        self.assertEqual(
            frames["finance_model_risk"].loc[0, "model_id"],
            "PD_EXPORT_001",
        )

    def test_review_thresholds_can_pass(self):
        report = usf.model_risk_report(
            _portfolio_rows(),
            public=["product", "region", "fico_band", "ltv_band"],
            hidden=[
                "product",
                "region",
                "fico_band",
                "ltv_band",
                "channel",
                "employment",
            ],
            metric=usf.default_rate(default="defaulted"),
            exposure="ead",
            q="observed",
            thresholds=usf.ReviewThresholds(ambiguity_limit=0.0),
        )

        self.assertEqual(report.review_status, "pass")
        self.assertEqual(report.review_reasons, ())

    def test_review_threshold_rejects_negative_ambiguity_limit(self):
        with self.assertRaisesRegex(ValueError, "ambiguity_limit must be non-negative"):
            usf.ReviewThresholds(ambiguity_limit=-0.01)

    def test_factor_exposure_shift_limits_portfolio_metric_movement(self):
        _require_cvxpy()
        rows = _factor_rows()
        moments = usf.portfolio_factor_moments(
            rows,
            hidden=["product", "channel"],
            factors={"macro": "macro_factor"},
            exposure="ead",
        )

        grouped = usf.from_portfolio(
            rows,
            public=["product"],
            hidden=["product", "channel"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            exposure="ead",
            q=usf.q_factor_exposure_shift(
                0.2,
                rows,
                hidden=["product", "channel"],
                factors={"macro": "macro_factor"},
                exposure="ead",
            ),
        )
        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(set(moments), {"factor:macro"})
        self.assertAlmostEqual(moments["factor:macro"][("loan", "broker")], -1.0)
        self.assertEqual(grouped.q_name, "covariate_balance(radius=0.2)")
        self.assertAlmostEqual(interval.lower, 0.42254033, places=5)
        self.assertAlmostEqual(interval.upper, 0.57745967, places=5)
        self.assertAlmostEqual(interval.diameter, 0.15491933, places=5)

    def test_regional_concentration_shift_can_pin_region_mix(self):
        _require_cvxpy()
        rows = [
            {
                "product": "loan",
                "region": "north",
                "pd": 0.0,
                "lgd": 1.0,
                "ead": 50.0,
            },
            {
                "product": "loan",
                "region": "south",
                "pd": 1.0,
                "lgd": 1.0,
                "ead": 50.0,
            },
        ]

        moments = usf.portfolio_concentration_moments(
            rows,
            hidden=["product", "region"],
            category="region",
            exposure="ead",
        )
        grouped = usf.from_portfolio(
            rows,
            public=["product"],
            hidden=["product", "region"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            exposure="ead",
            q=usf.q_regional_concentration_shift(
                0.0,
                rows,
                hidden=["product", "region"],
                region="region",
                exposure="ead",
            ),
        )
        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(set(moments), {"region:north", "region:south"})
        self.assertAlmostEqual(moments["region:north"][("loan", "north")], 1.0)
        self.assertAlmostEqual(moments["region:north"][("loan", "south")], 0.0)
        self.assertAlmostEqual(interval.lower, 0.5, places=5)
        self.assertAlmostEqual(interval.upper, 0.5, places=5)
        self.assertAlmostEqual(interval.diameter, 0.0, places=5)

    def test_finance_sensitivity_grid_builds_named_profile(self):
        _require_cvxpy()

        grid = usf.finance_sensitivity_grid(
            _factor_rows(),
            hidden=["product", "channel"],
            exposure="ead",
            factors={"macro": "macro_factor"},
        )

        self.assertEqual(grid[0], "saturated")
        self.assertEqual((grid[1].name, grid[1].radius), ("bounded_shift", 0.35))
        self.assertEqual((grid[2].name, grid[2].radius), ("tv_budget", 0.10))
        self.assertEqual((grid[3].name, grid[3].radius), ("covariate_balance", 0.20))
        self.assertEqual((grid[4].name, grid[4].radius), ("covariate_balance", 0.10))
        self.assertEqual(grid[5], "observed")

    def test_certify_portfolio_segmentation_returns_finance_certificate(self):
        _require_cvxpy()

        certificate = usf.certify_portfolio_segmentation(
            _factor_rows(),
            public=["product"],
            hidden=["product", "channel"],
            metric=usf.expected_loss(pd="pd", lgd="lgd"),
            exposure="ead",
            candidate_refinements=["channel"],
            factors={"macro": "macro_factor"},
            ambiguity_limit=0.16,
            bucket_budget=3,
            search="exhaustive",
            model_id="EL_TEST_001",
            portfolio_name="Test portfolio",
            intended_use="Expected-loss segmentation review",
        )
        markdown = certificate.to_markdown()
        payload = certificate.as_dict()
        tables = certificate.to_tables()

        self.assertIsInstance(certificate, usf.FinanceStabilityCertificate)
        self.assertIsInstance(certificate.core, us.RepresentationStabilityCertificate)
        self.assertTrue(certificate.passed)
        self.assertEqual(certificate.status, "pass")
        self.assertEqual(certificate.certified_candidate.added_columns, ("channel",))
        self.assertEqual(payload["metadata"]["model_id"], "EL_TEST_001")
        self.assertEqual(payload["core"]["status"], "pass")
        self.assertIn("Financial Model-Risk Segmentation", markdown)
        self.assertIn("Financial Certification Interpretation", markdown)
        self.assertIn("Core Certificate Evidence", markdown)
        self.assertIn("finance_certificate", tables)
        self.assertIn("core_summary", tables)

    def test_plugin_descriptor_can_be_registered(self):
        us.register_plugin(usf.plugin)

        report = us.validate_plugin(usf.plugin)
        self.assertTrue(report.ok)
        self.assertEqual(usf.plugin.metadata.package, "updatesupport-finance")
        self.assertEqual(usf.plugin.metadata.domain, "financial-model-risk")
        self.assertIs(us.plugin_metric("finance", "expected_loss"), usf.expected_loss)
        self.assertIs(
            us.plugin_metric("finance", "expected_loss_standard_error"),
            usf.expected_loss_standard_error,
        )
        self.assertIs(
            us.plugin_q_preset("finance", "portfolio_mix_shift"),
            usf.q_portfolio_mix_shift,
        )
        self.assertIs(
            us.plugin_q_preset("finance", "factor_exposure_shift"),
            usf.q_factor_exposure_shift,
        )
        self.assertIs(
            us.plugin_q_preset("finance", "regional_concentration_shift"),
            usf.q_regional_concentration_shift,
        )
        self.assertIs(
            us.plugin_report_profile("finance", "segmentation_certificate"),
            usf.certify_portfolio_segmentation,
        )
        self.assertIs(
            us.plugin_report_profile(
                "finance",
                "model_assisted_portfolio_uncertainty",
            ),
            usf.model_assisted_portfolio_uncertainty,
        )
        self.assertIs(
            us.plugin_report_profile("finance", "disclosure_triangulation"),
            usf.triangulate_disclosure,
        )
        self.assertIs(us.plugin_compiler("finance", "portfolio"), usf.from_portfolio)
        self.assertIs(
            us.plugin_compiler("finance", "disclosure_triangulation"),
            usf.disclosure_triangulation_spec,
        )

    def test_plugin_entry_point_can_be_discovered(self):
        discovered = us.discover_plugins()

        self.assertIn("finance", {plugin.name for plugin in discovered})
        self.assertIs(us.plugin_metric("finance", "expected_loss"), usf.expected_loss)

    def test_synthetic_portfolio_example_builds_model_risk_report(self):
        example_path = (
            Path(__file__).resolve().parents[1] / "examples" / "model_risk_portfolio.py"
        )
        namespace = runpy.run_path(str(example_path), run_name="updatesupport_example")

        rows = namespace["synthetic_portfolio_rows"]()
        report = namespace["build_report"]()
        frontier = namespace["build_frontier"]()
        markdown = report.to_markdown()
        combined_markdown = markdown + "\n\n" + frontier.to_markdown()

        self.assertEqual(len(rows), 10)
        self.assertIsInstance(report, usf.ModelRiskReport)
        self.assertIsInstance(frontier, us.PublicRepresentationFrontier)
        self.assertEqual(frontier.search_trace.search, "beam")
        self.assertIn(
            "Synthetic Finance Public Representation Frontier", combined_markdown
        )
        self.assertIn("Selected Representation Explanation", combined_markdown)
        self.assertIn("EL_SYNTHETIC_RETAIL_001", markdown)
        self.assertIn("hardship_history", markdown)
        self.assertIn("local_housing_market", markdown)

    def test_disclosure_triangulation_example_builds_tiered_report(self):
        example_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "disclosure_triangulation.py"
        )
        namespace = runpy.run_path(str(example_path), run_name="updatesupport_example")

        spec = namespace["build_spec"]()
        report = namespace["build_report"]()
        rows = namespace["width_reduction_rows"](report)
        markdown = namespace["render_markdown"](report)
        target_2022 = report.interval(
            target="component_2022",
            scenario="T2 + anchor disclosure",
        )
        target_2024 = report.interval(
            target="component_2024",
            scenario="T2 + anchor disclosure",
        )

        self.assertIsInstance(spec, us.NamedLinearFeasibilityProblem)
        self.assertIsInstance(report, us.NamedLinearFeasibilityReport)
        self.assertEqual(len(rows), 9)
        self.assertAlmostEqual(target_2022.lower, 100.0 / 1.475 / 1.325)
        self.assertAlmostEqual(target_2022.upper, 120.0 / 1.425 / 1.275)
        self.assertAlmostEqual(target_2024.lower, 100.0)
        self.assertAlmostEqual(target_2024.upper, 120.0)
        self.assertIn("Generic Disclosure Triangulation Worked Example", markdown)
        self.assertIn("T2 + anchor disclosure", markdown)
        self.assertIn("Width Reduction By Tier", markdown)
        self.assertIn("Binding Endpoint Constraints", markdown)
        self.assertIn("component_2024_anchor", markdown)

    def test_colab_demo_notebooks_are_valid_and_unexecuted(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
        notebooks = sorted(notebook_dir.glob("*.ipynb"))

        self.assertEqual(
            {path.name for path in notebooks},
            {
                "model_assisted_portfolio_uncertainty_colab.ipynb",
                "model_risk_portfolio_colab.ipynb",
            },
        )
        for path in notebooks:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            self.assertGreaterEqual(len(payload["cells"]), 8)
            source = "\n".join(
                "".join(cell.get("source", ())) for cell in payload["cells"]
            )
            self.assertIn("updatesupport_finance", source)
            self.assertIn("seaborn", source)
            self.assertIn("ipywidgets", source)
            self.assertIn("colab.research.google.com", source)
            self.assertNotIn(".legend_.remove()", source)
            if path.name == "model_risk_portfolio_colab.ipynb":
                self.assertIn("updatesupport[finance,cvxpy]", source)
                self.assertIn("q_factor_exposure_shift", source)
                self.assertIn("dual_summary", source)
            for cell in payload["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
