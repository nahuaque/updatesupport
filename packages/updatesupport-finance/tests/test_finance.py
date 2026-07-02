from __future__ import annotations

import unittest

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
        )
        markdown = report.to_markdown()

        self.assertIn("Financial Model-Risk Representation Stability Report", markdown)
        self.assertIn("Reported portfolio risk estimate", markdown)
        self.assertIn("bounded_shift(radius=0.25)", markdown)

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


if __name__ == "__main__":
    unittest.main()
