from __future__ import annotations

import unittest

import updatesupport as us


def _rows():
    return [
        {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
        {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
        {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
    ]


class QPresetTests(unittest.TestCase):
    def test_saturated_preset_matches_existing_observed_law_ambiguity(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q="saturated",
        )

        self.assertEqual(grouped.q_name, "saturated")
        self.assertAlmostEqual(grouped.problem.global_transport_modulus().diameter, 0.6)

    def test_observed_preset_has_zero_ambiguity(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q=us.q_observed(),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "observed")
        self.assertAlmostEqual(interval.lower, 0.5)
        self.assertAlmostEqual(interval.upper, 0.5)
        self.assertAlmostEqual(interval.diameter, 0.0)
        self.assertTrue(grouped.problem.is_public_adequate())

    def test_bounded_shift_preset_limits_hidden_reweighting(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q=us.q_bounded_shift(0.5),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "bounded_shift(radius=0.5)")
        self.assertAlmostEqual(interval.lower, 0.35)
        self.assertAlmostEqual(interval.upper, 0.65)
        self.assertAlmostEqual(interval.diameter, 0.3)

    def test_bounded_shift_radius_can_be_passed_separately(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q="bounded_shift",
            q_radius=0.0,
        )

        self.assertEqual(grouped.q_name, "bounded_shift(radius=0)")
        self.assertAlmostEqual(grouped.problem.global_transport_modulus().diameter, 0.0)

    def test_fiber_support_floor_preset_builds_scip_environment(self):
        rows = [
            {"public": "A", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "A", "hidden": "high", "target": 1.0, "weight": 25},
            {"public": "B", "hidden": "low", "target": 0.0, "weight": 25},
            {"public": "B", "hidden": "high", "target": 1.0, "weight": 25},
        ]

        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q=us.q_fiber_support_floor(2, min_share=0.25),
        )

        self.assertEqual(
            grouped.q_name,
            "fiber_support_floor(min_active=2, min_share=0.25)",
        )
        self.assertIn("at least 2 active hidden cells", grouped.q_description)
        self.assertIsInstance(grouped.problem.environments, us.CvxpyEnvironments)
        self.assertEqual(grouped.problem.environments.solver, "SCIP")

    def test_fiber_support_floor_rejects_sparse_public_fibers(self):
        with self.assertRaisesRegex(ValueError, "retained hidden cells"):
            us.from_dataframe(
                _rows(),
                public=["public"],
                hidden=["public", "hidden"],
                target="target",
                weight="weight",
                q=us.q_fiber_support_floor(2, min_share=0.25),
            )

    def test_cvxpy_q_preset_carries_solver_metadata(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            q=us.q_tv_budget(
                0.15,
                solver="SCIP",
                solver_options={"limits/time": 5},
            ),
        )

        env = grouped.problem.environments

        self.assertEqual(grouped.q_name, "tv_budget(radius=0.15)")
        self.assertIsInstance(env, us.CvxpyEnvironments)
        self.assertEqual(env.solver, "SCIP")
        self.assertEqual(env.solver_options, {"limits/time": 5})

    def test_sensitivity_report_runs_q_and_min_cell_weight_grid(self):
        report = us.sensitivity_report(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            min_cell_weights=[1, 35],
            q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
        )

        rows = report.rows
        markdown = report.to_markdown()

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0].q_name, "saturated")
        self.assertAlmostEqual(rows[0].ambiguity, 0.6)
        self.assertEqual(rows[1].q_name, "bounded_shift(radius=0.5)")
        self.assertAlmostEqual(rows[1].ambiguity, 0.3)
        self.assertEqual(rows[2].q_name, "observed")
        self.assertAlmostEqual(rows[2].ambiguity, 0.0)
        self.assertAlmostEqual(rows[3].ambiguity, 0.0)
        self.assertIsInstance(report.summary, us.SensitivitySummary)
        self.assertEqual(report.summary.scenario_count, 6)
        self.assertEqual(report.summary.successful_scenarios, 6)
        self.assertEqual(report.summary.failed_scenarios, 0)
        self.assertEqual(report.summary.baseline_scenario, "S1")
        self.assertEqual(report.summary.lowest_ambiguity_scenario, "S3")
        self.assertEqual(report.summary.highest_ambiguity_scenario, "S1")
        self.assertAlmostEqual(report.summary.min_ambiguity, 0.0)
        self.assertAlmostEqual(report.summary.max_ambiguity, 0.6)
        self.assertAlmostEqual(report.summary.ambiguity_span, 0.6)
        self.assertEqual(report.summary.public_adequacy_pattern, "mixed")
        self.assertAlmostEqual(report.summary.observed_span, 0.0)
        self.assertIn("# Public Descent Sensitivity Report", markdown)
        self.assertIn("## Scenario Summary", markdown)
        self.assertIn("- Scenarios: 6", markdown)
        self.assertIn("Ambiguity range across successful scenarios", markdown)
        self.assertIn("## Interpretation", markdown)
        self.assertIn("Public adequacy changes across successful scenarios", markdown)
        self.assertIn("## Scenario Table", markdown)
        self.assertIn("bounded_shift(radius=0.5)", markdown)

    def test_sensitivity_report_runs_hidden_set_grid(self):
        rows = [
            {
                "public": "A",
                "hidden": "x",
                "noise": "n1",
                "target": 0.0,
                "weight": 30,
            },
            {
                "public": "A",
                "hidden": "y",
                "noise": "n2",
                "target": 1.0,
                "weight": 30,
            },
            {
                "public": "B",
                "hidden": "z",
                "noise": "n1",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.sensitivity_report(
            rows,
            public=["public"],
            hidden=["public", "hidden", "noise"],
            target="target",
            weight="weight",
            hidden_sets=[
                ["public", "hidden"],
                ["public", "hidden", "noise"],
            ],
            q_presets=["saturated"],
        )

        self.assertEqual(len(report.rows), 2)
        self.assertEqual(report.rows[0].hidden_columns, ("public", "hidden"))
        self.assertEqual(
            report.rows[1].hidden_columns,
            ("public", "hidden", "noise"),
        )

    def test_sensitivity_report_summarizes_failed_scenarios(self):
        report = us.sensitivity_report(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            hidden_sets=[["hidden"]],
            q_presets=["saturated"],
        )

        markdown = report.to_markdown()

        self.assertEqual(report.summary.scenario_count, 1)
        self.assertEqual(report.summary.successful_scenarios, 0)
        self.assertEqual(report.summary.failed_scenarios, 1)
        self.assertIsNone(report.summary.baseline_scenario)
        self.assertEqual(report.summary.public_adequacy_pattern, "unavailable")
        self.assertIn("No scenario completed successfully", markdown)
        self.assertIn("error: public columns must also be hidden columns", markdown)


if __name__ == "__main__":
    unittest.main()
