from __future__ import annotations

import json
import unittest

import updatesupport as us


def _example_problem() -> us.NamedLinearFeasibilityProblem:
    constraints = [
        us.named_linear_constraint(
            "total_previous",
            "total_previous",
            lower=100.0,
            upper=100.0,
            kind="reported_total",
        ),
        us.named_linear_constraint(
            "total_current",
            "total_current",
            lower=200.0,
            upper=200.0,
            kind="reported_total",
        ),
        us.named_linear_constraint(
            "previous_containment",
            {"component_previous": 1.0, "total_previous": -1.0},
            upper=0.0,
            kind="containment",
        ),
        us.named_linear_constraint(
            "current_containment",
            {"component_current": 1.0, "total_current": -1.0},
            upper=0.0,
            kind="containment",
        ),
        us.named_linear_constraint(
            "growth_lower",
            {"component_current": 1.0, "component_previous": -1.5},
            lower=0.0,
            kind="rounded_growth",
        ),
        us.named_linear_constraint(
            "growth_upper",
            {"component_current": 1.0, "component_previous": -1.6},
            upper=0.0,
            kind="rounded_growth",
        ),
        us.named_linear_constraint(
            "current_anchor",
            "component_current",
            lower=90.0,
            kind="anchor",
        ),
    ]
    return us.named_linear_feasibility_problem(
        title="Generic Disclosure Triangulation",
        variables=[
            us.named_linear_variable("component_previous", lower=0.0),
            us.named_linear_variable("component_current", lower=0.0),
            us.named_linear_variable("total_previous", lower=0.0),
            us.named_linear_variable("total_current", lower=0.0),
        ],
        constraints=constraints,
        targets=[
            us.named_linear_target(
                "previous_component",
                "component_previous",
                label="Previous-period component",
            )
        ],
        scenarios=[
            us.named_linear_scenario(
                "T0 containment",
                [
                    "total_previous",
                    "total_current",
                    "previous_containment",
                    "current_containment",
                ],
            ),
            us.named_linear_scenario(
                "T1 + growth",
                [
                    "total_previous",
                    "total_current",
                    "previous_containment",
                    "current_containment",
                    "growth_lower",
                    "growth_upper",
                ],
            ),
            us.named_linear_scenario(
                "T2 + anchor",
                [
                    "total_previous",
                    "total_current",
                    "previous_containment",
                    "current_containment",
                    "growth_lower",
                    "growth_upper",
                    "current_anchor",
                ],
            ),
        ],
    )


class NamedLinearFeasibilityTests(unittest.TestCase):
    def test_solves_tiered_named_linear_intervals(self):
        report = us.solve_named_linear_feasibility(_example_problem())

        t0 = report.interval(
            target="previous_component",
            scenario="T0 containment",
        )
        t2 = report.interval(
            target="previous_component",
            scenario="T2 + anchor",
        )

        self.assertEqual(t0.status, "bounded")
        self.assertAlmostEqual(t0.lower, 0.0)
        self.assertAlmostEqual(t0.upper, 100.0)
        self.assertEqual(t2.status, "bounded")
        self.assertAlmostEqual(t2.lower, 56.25)
        self.assertAlmostEqual(t2.upper, 100.0)

        reduction = report.width_reduction(
            target="previous_component",
            baseline_scenario="T0 containment",
            comparison_scenario="T2 + anchor",
        )
        self.assertAlmostEqual(reduction["width_reduction"], 56.25)
        self.assertGreater(reduction["width_reduction_percent"], 50.0)

        markdown = report.to_markdown()
        self.assertIn("## Feasible Intervals", markdown)
        self.assertIn("Previous-period component", markdown)
        tables = report.to_tables()
        self.assertIn("intervals", tables)
        self.assertIn("endpoints", tables)
        self.assertIn("active_constraints", tables)
        self.assertIn("endpoint_constraint_diagnostics", tables)
        self.assertIn("Generic Disclosure Triangulation", report.to_json())
        self.assertEqual(
            json.loads(report.to_json())["backend"],
            "scipy-linprog",
        )

    def test_endpoint_constraint_diagnostics_include_binding_duals(self):
        report = us.solve_named_linear_feasibility(_example_problem())
        interval = report.interval(
            target="previous_component",
            scenario="T2 + anchor",
        )

        diagnostics = interval.lower_endpoint.constraint_diagnostics
        anchor_lower = next(
            row
            for row in diagnostics
            if row.constraint == "current_anchor" and row.side == "lower"
        )

        self.assertIsInstance(anchor_lower, us.NamedLinearConstraintDiagnostic)
        self.assertTrue(anchor_lower.binding)
        self.assertEqual(anchor_lower.kind, "anchor")
        self.assertEqual(anchor_lower.slack, 0.0)
        self.assertIsNotNone(anchor_lower.solver_marginal)
        self.assertIsNotNone(anchor_lower.target_marginal)
        self.assertGreater(anchor_lower.dual_magnitude, 0.0)
        self.assertIn("current_anchor:lower", interval.lower_endpoint.binding_constraint_sides)
        self.assertIn("Dual / Binding Constraint Diagnostics", report.to_markdown())

    def test_constraint_attribution_ranks_interval_tightening(self):
        report = us.solve_named_linear_feasibility(_example_problem())

        attribution = us.attribute_named_linear_constraints(
            report,
            target="previous_component",
            scenario="T2 + anchor",
        )
        anchor = next(row for row in attribution.rows if row.group == "current_anchor")

        self.assertIsInstance(
            attribution,
            us.NamedLinearConstraintAttributionReport,
        )
        self.assertEqual(attribution.rows[0].group, "current_anchor")
        self.assertAlmostEqual(anchor.relaxed_lower, 0.0)
        self.assertAlmostEqual(anchor.relaxed_upper, 100.0)
        self.assertAlmostEqual(anchor.width_increase, 56.25)
        self.assertAlmostEqual(anchor.lower_tightening, 56.25)
        self.assertAlmostEqual(anchor.upper_tightening, 0.0)
        self.assertIn("Ranked Constraint Values", attribution.to_markdown())
        self.assertIn("constraint_attribution", attribution.to_tables())
        self.assertIn("current_anchor", attribution.to_json())

    def test_constraint_attribution_can_group_by_kind(self):
        report = us.solve_named_linear_feasibility(_example_problem())

        attribution = report.attribute_constraints(
            target="previous_component",
            scenario="T2 + anchor",
            group_by="kind",
        )
        rows = {row.group: row for row in attribution.rows}

        self.assertEqual(rows["rounded_growth"].constraint_count, 2)
        self.assertEqual(rows["anchor"].constraint_count, 1)
        self.assertGreater(rows["anchor"].width_increase, 50.0)
        self.assertEqual(attribution.group_by, "kind")

    def test_problem_mapping_round_trip(self):
        problem = _example_problem()
        report = us.solve_named_linear_feasibility(problem.as_dict())

        interval = report.interval(
            target="previous_component",
            scenario="T2 + anchor",
        )
        self.assertAlmostEqual(interval.lower, 56.25)

    def test_unknown_expression_variables_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "unknown variables"):
            us.named_linear_feasibility_problem(
                variables=["x"],
                constraints=[
                    us.named_linear_constraint(
                        "bad_constraint",
                        {"x": 1.0, "missing": 1.0},
                        lower=0.0,
                    )
                ],
                targets=[us.named_linear_target("x", "x")],
                scenarios=[
                    us.named_linear_scenario("scenario", ["bad_constraint"])
                ],
            )

    def test_unbounded_and_infeasible_statuses_are_reported(self):
        unbounded = us.solve_named_linear_feasibility(
            us.named_linear_feasibility_problem(
                variables=["x"],
                constraints=[],
                targets=[us.named_linear_target("x", "x")],
                scenarios=[us.named_linear_scenario("open", [])],
            )
        ).interval(target="x", scenario="open")
        self.assertEqual(unbounded.status, "unbounded")

        infeasible = us.solve_named_linear_feasibility(
            us.named_linear_feasibility_problem(
                variables=[us.named_linear_variable("x", lower=0.0, upper=1.0)],
                constraints=[
                    us.named_linear_constraint("too_high", "x", lower=2.0)
                ],
                targets=[us.named_linear_target("x", "x")],
                scenarios=[us.named_linear_scenario("closed", ["too_high"])],
            )
        ).interval(target="x", scenario="closed")
        self.assertEqual(infeasible.status, "infeasible")


if __name__ == "__main__":
    unittest.main()
