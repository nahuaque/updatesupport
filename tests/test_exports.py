from __future__ import annotations

import json
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
        {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
        {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
    ]


class StructuredExportTests(unittest.TestCase):
    def test_public_descent_report_exports_json_and_tables(self):
        report = us.public_descent_report(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q=us.q_bounded_shift(0.5),
            title="Export Demo",
        )

        payload = json.loads(report.to_json())
        tables = report.to_tables()

        self.assertEqual(payload["title"], "Export Demo")
        self.assertEqual(payload["q_name"], "bounded_shift(radius=0.5)")
        self.assertEqual(payload["target_contract"]["kind"], "linear")
        self.assertEqual(payload["top_fibers"][0]["public_value"], ["A"])
        self.assertEqual(
            set(tables),
            {
                "summary",
                "worst_fibers",
                "refinements",
                "data_diagnostics",
                "dual_diagnostics",
            },
        )
        self.assertEqual(tables["summary"][0]["title"], "Export Demo")
        self.assertAlmostEqual(tables["summary"][0]["ambiguity"], 0.3)
        self.assertEqual(tables["summary"][0]["target_kind"], "linear")
        self.assertEqual(
            tables["summary"][0]["target_formula"],
            "psi(q) = sum_d h(d) q(d)",
        )
        self.assertEqual(tables["refinements"][0]["column"], "driver")
        self.assertIn("diagnostic_count", tables["summary"][0])
        self.assertEqual(
            json.loads(us.report_to_json(report))["q_name"],
            "bounded_shift(radius=0.5)",
        )

    def test_audit_run_exports_spec_and_prefixed_report_tables(self):
        run = us.AuditSpec(
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
            q={"name": "bounded_shift", "radius": 0.5},
        ).run(_rows())

        payload = json.loads(run.to_json())
        tables = run.to_tables()

        self.assertEqual(payload["spec"]["q"]["name"], "bounded_shift")
        self.assertEqual(payload["report_type"], "public_descent")
        self.assertIn("spec", tables)
        self.assertIn("report_summary", tables)
        self.assertIn("report_refinements", tables)
        self.assertEqual(tables["report_refinements"][0]["column"], "driver")

    def test_sensitivity_report_exports_scenario_tables(self):
        report = us.sensitivity_report(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
            min_cell_weights=[1, 35],
        )

        tables = report.to_tables()
        payload = json.loads(report.to_json())

        self.assertEqual(tables["summary"][0]["scenario_count"], 6)
        self.assertEqual(len(tables["scenarios"]), 6)
        self.assertEqual(payload["summary"]["scenario_count"], 6)

    def test_frontier_exports_candidate_and_scenario_tables(self):
        report = us.public_representation_frontier(
            _rows(),
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
            q_presets=["saturated", us.q_bounded_shift(0.5)],
            ambiguity_limit=0.05,
        )

        tables = report.to_tables()
        payload = json.loads(report.to_json())

        self.assertIn("frontier", tables)
        self.assertIn("candidate_scenarios", tables)
        self.assertEqual(tables["summary"][0]["minimal_stable"], ("driver",))
        self.assertEqual(payload["minimal_stable"]["added_columns"], ["driver"])
        self.assertEqual(
            tables["candidate_scenarios"][0]["candidate_label"],
            "base public representation",
        )

    def test_report_to_dataframes_uses_pandas_when_available(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        report = us.public_descent_report(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
        )

        frames = report.to_dataframes()

        self.assertIsInstance(frames["summary"], pd.DataFrame)
        self.assertEqual(frames["summary"].loc[0, "title"], "Public Descent Report")
        self.assertEqual(frames["refinements"].loc[0, "column"], "driver")

    def test_dowhy_wrapper_exports_underlying_report_tables(self):
        audit = us.audit_dowhy_effects(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            effect="target",
            candidate_refinements=["driver"],
        )

        tables = audit.to_tables()
        helper_tables = us.report_tables(audit)
        payload = json.loads(audit.to_json())

        self.assertIn("dowhy_refutation", tables)
        self.assertIn("dowhy_refutation", helper_tables)
        self.assertIn("refinements", tables)
        self.assertEqual(
            payload["report"]["title"], "DoWhy Effect Representation Stability Audit"
        )


if __name__ == "__main__":
    unittest.main()
