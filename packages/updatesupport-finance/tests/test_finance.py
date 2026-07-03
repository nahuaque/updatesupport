from __future__ import annotations

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
        self.assertIn("Reported portfolio risk estimate", markdown)
        self.assertIn("bounded_shift(radius=0.25)", markdown)

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
        self.assertIs(us.plugin_compiler("finance", "portfolio"), usf.from_portfolio)

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


if __name__ == "__main__":
    unittest.main()
