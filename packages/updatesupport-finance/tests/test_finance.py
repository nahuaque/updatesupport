from __future__ import annotations

import runpy
import unittest
from pathlib import Path

import updatesupport as us
import updatesupport_finance as usf


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

    def test_plugin_descriptor_can_be_registered(self):
        us.register_plugin(usf.plugin)

        self.assertIs(us.plugin_metric("finance", "expected_loss"), usf.expected_loss)
        self.assertIs(
            us.plugin_q_preset("finance", "portfolio_mix_shift"),
            usf.q_portfolio_mix_shift,
        )
        self.assertIs(us.plugin_compiler("finance", "portfolio"), usf.from_portfolio)

    def test_plugin_entry_point_can_be_discovered(self):
        discovered = us.discover_plugins()

        self.assertIn("finance", {plugin.name for plugin in discovered})
        self.assertIs(us.plugin_metric("finance", "expected_loss"), usf.expected_loss)

    def test_synthetic_portfolio_example_builds_model_risk_report(self):
        example_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "model_risk_portfolio.py"
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
        self.assertIn("Synthetic Finance Public Representation Frontier", combined_markdown)
        self.assertIn("Selected Representation Explanation", combined_markdown)
        self.assertIn("EL_SYNTHETIC_RETAIL_001", markdown)
        self.assertIn("hardship_history", markdown)
        self.assertIn("local_housing_market", markdown)


if __name__ == "__main__":
    unittest.main()
