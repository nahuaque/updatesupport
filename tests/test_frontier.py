from __future__ import annotations

import unittest
from typing import cast

import updatesupport as us


def _require_cvxpy_solver(name: str) -> None:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise unittest.SkipTest("CVXPY is not installed") from exc
    installed = {str(solver).upper() for solver in cp.installed_solvers()}
    if name.upper() not in installed:
        raise unittest.SkipTest(f"CVXPY solver {name!r} is not installed")


class PublicRepresentationFrontierTests(unittest.TestCase):
    def test_certify_public_representation_passes_with_exact_stable_candidate(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        certificate = us.certify_public_representation(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            bucket_budget=3,
        )
        payload = certificate.as_dict()
        markdown = certificate.to_markdown()
        tables = certificate.to_tables()

        self.assertIsInstance(certificate, us.RepresentationStabilityCertificate)
        self.assertTrue(certificate.passed)
        self.assertEqual(certificate.status, "pass")
        self.assertIsNotNone(certificate.certified_candidate)
        self.assertEqual(certificate.certified_candidate.added_columns, ("driver",))
        self.assertEqual(
            certificate.certified_candidate.public_columns,
            ("segment", "driver"),
        )
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["certified_candidate"]["added_columns"], ("driver",))
        self.assertIn("Certification status: **PASS**", markdown)
        self.assertIn("Certified representation", markdown)
        self.assertIn("## Certified Scenario Evidence", markdown)
        self.assertIn("summary", tables)
        self.assertIn("selected_scenarios", tables)
        self.assertIn("frontier_candidates", tables)

    def test_certify_public_representation_fails_when_budget_blocks_stable_choice(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        certificate = us.certify_public_representation(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            bucket_budget=2,
        )

        self.assertTrue(certificate.failed)
        self.assertIsNone(certificate.certified_candidate)
        self.assertEqual(certificate.selected_candidate, None)
        self.assertIn("No evaluated representation", certificate.reasons[0])

    def test_certify_public_representation_marks_required_heuristic_inconclusive(self):
        rows = [
            {"segment": "A", "driver": "low", "noise": "n", "target": 0.0},
            {"segment": "A", "driver": "high", "noise": "n", "target": 1.0},
            {"segment": "B", "driver": "flat", "noise": "n", "target": 0.5},
        ]

        certificate = us.certify_public_representation(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            search="greedy",
            max_added_columns=2,
        )
        relaxed = us.certify_public_representation(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            search="greedy",
            max_added_columns=2,
            exact_required=False,
        )

        self.assertTrue(certificate.inconclusive)
        self.assertFalse(certificate.search_exact)
        self.assertIsNotNone(certificate.selected_candidate)
        self.assertIn("heuristic", certificate.reasons[0])
        self.assertTrue(relaxed.passed)

    def test_public_representation_frontier_finds_nondominated_refinements(self):
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated", us.q_bounded_shift(0.5)],
            ambiguity_limit=0.05,
            bucket_budget=2,
        )
        payload = report.as_dict()

        self.assertIsInstance(report, us.PublicRepresentationFrontier)
        self.assertIsInstance(report.search_trace, us.FrontierSearchTrace)
        self.assertEqual(report.search_trace.search, "exhaustive")
        self.assertTrue(report.search_trace.exact)
        self.assertEqual(report.search_trace.candidate_space_size, 4)
        self.assertEqual(report.search_trace.scenario_count, 2)
        self.assertEqual(report.search_trace.evaluated_candidates, 4)
        self.assertEqual(len(report.candidates), 4)
        self.assertEqual(len(report.frontier), 2)
        self.assertEqual(len(report.dominated), 2)
        self.assertEqual(
            [row.added_columns for row in report.frontier],
            [(), ("driver",)],
        )
        self.assertEqual(
            [row.added_columns for row in report.dominated],
            [("noise",), ("noise", "driver")],
        )

        baseline = report.candidates[0]
        self.assertEqual(baseline.public_cells, 2)
        self.assertAlmostEqual(baseline.max_ambiguity, 0.6)
        self.assertAlmostEqual(baseline.mean_ambiguity, 0.45)
        self.assertFalse(baseline.passes_ambiguity_limit)
        self.assertFalse(baseline.public_adequate)
        self.assertEqual(baseline.ambiguity_by_scenario()["S1"], 0.6)
        self.assertAlmostEqual(baseline.ambiguity_by_scenario()["S2"], 0.3)

        minimal = report.minimal_stable
        self.assertIsNotNone(minimal)
        minimal = cast(us.PublicRepresentationCandidate, minimal)
        self.assertEqual(minimal.added_columns, ("driver",))
        self.assertEqual(minimal.public_cells, 3)
        self.assertAlmostEqual(minimal.max_ambiguity, 0.0)
        self.assertTrue(minimal.passes_ambiguity_limit)
        self.assertTrue(minimal.public_adequate)

        budget_best = report.best_under_bucket_budget()
        self.assertIsNotNone(budget_best)
        budget_best = cast(us.PublicRepresentationCandidate, budget_best)
        self.assertEqual(budget_best.added_columns, ())
        self.assertEqual(payload["minimal_stable"]["added_columns"], ("driver",))
        self.assertEqual(
            payload["best_under_bucket_budget"]["added_columns"],
            (),
        )

    def test_public_representation_frontier_renders_markdown(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        markdown = us.public_representation_frontier(
            rows,
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            bucket_budget=3,
            title="Demo Frontier",
        ).to_markdown()

        self.assertIn("# Demo Frontier", markdown)
        self.assertIn("## Interpretation", markdown)
        self.assertIn("## Pareto Frontier", markdown)
        self.assertIn("## Scenario Details", markdown)
        self.assertIn("Minimal stable representation", markdown)
        self.assertIn("Best representation within bucket budget", markdown)
        self.assertIn("Search mode: exhaustive (exact)", markdown)
        self.assertIn("base public representation", markdown)
        self.assertIn("base + driver", markdown)
        self.assertIn("max ambiguity", markdown)
        self.assertIn("Pareto-frontier", markdown)

    def test_public_representation_frontier_recompiles_procedure_targets(self):
        rows = [
            {"segment": "A", "driver": "low", "base": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "base": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "base": 0.5, "weight": 40},
        ]
        compiled_public = []

        def compiler(context):
            scale = len(context.public)
            compiled_public.append(context.public)
            return us.row_metric(
                f"frontier_value_x{scale}",
                lambda row, scale=scale: scale * float(row["base"]),
                columns=("base",),
            )

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target=us.ProcedureTarget("frontier_procedure", compiler),
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            include_base=True,
        )
        baseline = next(row for row in report.candidates if not row.added_columns)
        refined = next(
            row for row in report.candidates if row.added_columns == ("driver",)
        )

        self.assertEqual(compiled_public, [("segment",), ("segment", "driver")])
        self.assertAlmostEqual(baseline.observed_value, 0.5)
        self.assertAlmostEqual(refined.observed_value, 1.0)
        self.assertAlmostEqual(baseline.max_ambiguity, 0.6)
        self.assertAlmostEqual(refined.max_ambiguity, 0.0)

    def test_public_representation_frontier_supports_greedy_search(self):
        rows = [
            {"segment": "A", "driver": "low", "noise": "n", "target": 0.0},
            {"segment": "A", "driver": "high", "noise": "n", "target": 1.0},
            {"segment": "B", "driver": "flat", "noise": "n", "target": 0.5},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            search="greedy",
            max_added_columns=2,
        )

        self.assertFalse(report.search_trace.exact)
        self.assertEqual(report.search_trace.search, "greedy")
        self.assertEqual(report.search_trace.stopping_reason, "ambiguity_limit reached")
        self.assertEqual(report.search_trace.candidate_space_size, 4)
        self.assertEqual(report.minimal_stable.added_columns, ("driver",))
        self.assertIn(("driver",), [row.added_columns for row in report.candidates])

    def test_public_representation_frontier_supports_beam_search(self):
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n",
                "extra": "e1",
                "target": 0.0,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n",
                "extra": "e2",
                "target": 1.0,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n",
                "extra": "e1",
                "target": 0.5,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise", "extra"],
            target="target",
            candidate_refinements=["noise", "driver", "extra"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            search="beam",
            beam_width=1,
            max_added_columns=2,
        )

        self.assertFalse(report.search_trace.exact)
        self.assertEqual(report.search_trace.search, "beam")
        self.assertEqual(report.search_trace.beam_width, 1)
        self.assertGreater(report.search_trace.pruned_by_beam, 0)
        self.assertLess(
            report.search_trace.evaluated_candidates,
            report.search_trace.candidate_space_size,
        )
        self.assertIn(("driver",), [row.added_columns for row in report.frontier])

    def test_public_representation_frontier_scores_scalarized_objective(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            scalarized_weights={"max_ambiguity": 1.0, "public_cells": 1.0},
        )
        baseline = next(row for row in report.candidates if row.added_columns == ())
        refined = next(
            row for row in report.candidates if row.added_columns == ("driver",)
        )
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertEqual(
            report.scalarized_weights,
            {"max_ambiguity": 1.0, "public_cells": 1.0},
        )
        self.assertIsNotNone(report.best_scalarized)
        self.assertEqual(report.best_scalarized.added_columns, ())
        self.assertAlmostEqual(baseline.scalarized_score, 2.6)
        self.assertAlmostEqual(refined.scalarized_score, 3.0)
        self.assertEqual(baseline.scalarized_components["public_cells"], 2.0)
        self.assertIn("Scalarized objective weights", markdown)
        self.assertIn("Best scalarized representation", markdown)
        self.assertEqual(tables["summary"][0]["best_scalarized"], ())
        self.assertEqual(
            tables["summary"][0]["scalarized_weights"],
            {"max_ambiguity": 1.0, "public_cells": 1.0},
        )
        self.assertIn("scalarized_score", tables["candidates"][0])

    def test_public_representation_frontier_supports_scalarized_search(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            search="scalarized",
            max_added_columns=1,
            scalarized_weights={"max_ambiguity": 1.0, "public_cells": 1.0},
        )

        self.assertFalse(report.search_trace.exact)
        self.assertEqual(report.search_trace.search, "scalarized")
        self.assertEqual(
            report.search_trace.scalarized_weights,
            {"max_ambiguity": 1.0, "public_cells": 1.0},
        )
        self.assertEqual(
            report.search_trace.stopping_reason,
            "no scalarized improvement",
        )
        self.assertEqual(report.best_scalarized.added_columns, ())
        self.assertIn((), [row.added_columns for row in report.candidates])
        self.assertIn(("driver",), [row.added_columns for row in report.candidates])

    def test_public_representation_frontier_supports_mip_search(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n1",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n1",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n2",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated"],
            ambiguity_limit=0.05,
            search="mip",
            max_added_columns=1,
        )
        markdown = report.to_markdown()

        self.assertTrue(report.search_trace.exact)
        self.assertEqual(report.search_trace.search, "mip")
        self.assertEqual(report.search_trace.solver.upper(), "SCIP")
        self.assertIn("optimal", report.search_trace.solver_status)
        self.assertIn("mip", report.search_trace.stopping_reason)
        self.assertEqual(report.minimal_stable.added_columns, ("driver",))
        self.assertIn(("driver",), [row.added_columns for row in report.candidates])
        self.assertIn("Search solver: SCIP", markdown)
        self.assertIn("MIP search directly optimizes", markdown)

    def test_public_representation_frontier_mip_supports_scalarized_public_cells(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=["saturated"],
            search="mip",
            max_added_columns=1,
            scalarized_weights={"max_ambiguity": 1.0, "public_cells": 1.0},
        )

        self.assertTrue(report.search_trace.exact)
        self.assertEqual(report.best_scalarized.added_columns, ())
        self.assertEqual(report.search_trace.scalarized_weights["public_cells"], 1.0)

    def test_public_representation_frontier_mip_rejects_non_saturated_q(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0},
            {"segment": "A", "driver": "high", "target": 1.0},
        ]

        with self.assertRaisesRegex(ValueError, "only saturated Q presets"):
            us.public_representation_frontier(
                rows,
                base_public=["segment"],
                hidden=["segment", "driver"],
                target="target",
                candidate_refinements=["driver"],
                q_presets=[us.q_bounded_shift(0.5)],
                search="mip",
            )

    def test_public_representation_frontier_supports_mip_oracle_search(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
            {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
            {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            q_presets=[us.q_tv_budget(0.15)],
            ambiguity_limit=0.31,
            bucket_budget=2,
            search="mip_oracle",
            max_added_columns=1,
        )
        markdown = report.to_markdown()

        self.assertTrue(report.search_trace.exact)
        self.assertEqual(report.search_trace.search, "mip_oracle")
        self.assertEqual(report.search_trace.solver.upper(), "SCIP")
        self.assertEqual(report.search_trace.oracle_iterations, 1)
        self.assertEqual(report.search_trace.oracle_rejections, 0)
        self.assertEqual(report.minimal_stable.added_columns, ())
        self.assertAlmostEqual(report.minimal_stable.max_ambiguity, 0.3, places=4)
        self.assertEqual(report.search_trace.scalarized_weights, None)
        self.assertTrue(report.search_trace.enforce_bucket_budget)
        self.assertIn("support-function oracle", markdown)
        self.assertIn("Support-function oracle evaluations: 1", markdown)

    def test_public_representation_frontier_mip_oracle_adds_no_good_cuts(self):
        _require_cvxpy_solver("SCIP")
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "driver"],
            q_presets=[us.q_tv_budget(0.15)],
            ambiguity_limit=0.05,
            bucket_budget=3,
            search="mip_oracle",
            max_added_columns=1,
        )

        self.assertTrue(report.search_trace.exact)
        self.assertEqual(report.minimal_stable.added_columns, ("driver",))
        self.assertGreaterEqual(report.search_trace.oracle_iterations, 2)
        self.assertGreaterEqual(report.search_trace.oracle_rejections, 1)
        self.assertIn(("noise",), [row.added_columns for row in report.candidates])

    def test_public_representation_frontier_mip_oracle_rejects_unsupported_q(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0},
            {"segment": "A", "driver": "high", "target": 1.0},
        ]

        with self.assertRaisesRegex(ValueError, "support-function-compatible"):
            us.public_representation_frontier(
                rows,
                base_public=["segment"],
                hidden=["segment", "driver"],
                target="target",
                candidate_refinements=["driver"],
                q_presets=[us.q_bounded_shift(0.5)],
                ambiguity_limit=0.05,
                search="mip_oracle",
            )

    def test_public_representation_frontier_rejects_bad_scalarized_weights(self):
        rows = [
            {"segment": "A", "driver": "low", "target": 0.0},
            {"segment": "A", "driver": "high", "target": 1.0},
        ]

        with self.assertRaisesRegex(ValueError, "scalarized_weights keys"):
            us.public_representation_frontier(
                rows,
                base_public=["segment"],
                hidden=["segment", "driver"],
                target="target",
                candidate_refinements=["driver"],
                scalarized_weights={"unknown": 1.0},
            )

    def test_public_representation_frontier_tracks_max_evaluations(self):
        rows = [
            {"segment": "A", "driver": "low", "noise": "n", "target": 0.0},
            {"segment": "A", "driver": "high", "noise": "n", "target": 1.0},
            {"segment": "B", "driver": "flat", "noise": "n", "target": 0.5},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated"],
            max_evaluations=2,
        )

        self.assertFalse(report.search_trace.exact)
        self.assertEqual(report.search_trace.evaluated_candidates, 2)
        self.assertEqual(report.search_trace.candidate_space_size, 4)
        self.assertEqual(report.search_trace.stopping_reason, "max_evaluations reached")

    def test_public_representation_frontier_supports_column_constraints(self):
        rows = [
            {"segment": "A", "driver": "low", "noise": "n", "target": 0.0},
            {"segment": "A", "driver": "high", "noise": "n", "target": 1.0},
            {"segment": "B", "driver": "flat", "noise": "n", "target": 0.5},
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            candidate_refinements=["noise"],
            q_presets=["saturated"],
            must_include=["driver"],
            must_exclude=["noise"],
        )

        self.assertEqual(report.candidate_refinements, ("driver",))
        self.assertEqual(report.search_trace.candidate_space_size, 1)
        self.assertEqual(
            [row.added_columns for row in report.candidates], [("driver",)]
        )

    def test_public_representation_frontier_supports_sensitivity_grid(self):
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n",
                "extra": "e1",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n",
                "extra": "e2",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n",
                "extra": "e1",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise", "extra"],
            hidden_sets=[
                ["segment", "driver", "noise", "extra"],
                ["segment", "driver", "noise"],
            ],
            target="target",
            weight="weight",
            candidate_refinements=["driver", "extra", "noise"],
            q_presets=["saturated", us.q_bounded_shift(0.5)],
            min_cell_weights=[1, 35],
            ambiguity_limit=0.05,
        )
        markdown = report.to_markdown()

        self.assertEqual(report.hidden_sets[0], ("segment", "driver", "noise", "extra"))
        self.assertEqual(report.hidden_sets[1], ("segment", "driver", "noise"))
        self.assertEqual(report.min_cell_weights, (1.0, 35.0))
        self.assertEqual(report.candidate_refinements, ("driver", "noise"))
        self.assertEqual(report.search_trace.scenario_count, 8)
        self.assertEqual(len(report.candidates[0].scenarios), 8)
        self.assertEqual(report.candidates[0].min_public_cells, 1)
        self.assertEqual(report.candidates[0].max_public_cells, 2)
        self.assertEqual(report.candidates[0].min_hidden_cells, 1)
        self.assertEqual(report.candidates[0].max_hidden_cells, 3)
        self.assertEqual(report.minimal_stable.added_columns, ("driver",))
        self.assertEqual(report.minimal_stable.max_public_cells, 3)
        self.assertIn("Hidden-set scenarios: 2", markdown)
        self.assertIn("Minimum hidden-cell weights: 1, 35", markdown)
        self.assertIn("Stress-test scenarios per representation: 8", markdown)
        self.assertIn("sensitivity-aware", markdown)
        self.assertIn("min_cell_weight", markdown)

    def test_public_representation_frontier_explains_selected_candidate(self):
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "noise": "n",
                "extra": "e1",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "noise": "n",
                "extra": "e2",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "noise": "n",
                "extra": "e1",
                "target": 0.5,
                "weight": 40,
            },
        ]

        report = us.public_representation_frontier(
            rows,
            base_public=["segment"],
            hidden=["segment", "driver", "noise", "extra"],
            hidden_sets=[
                ["segment", "driver", "noise", "extra"],
                ["segment", "driver", "noise"],
            ],
            target="target",
            weight="weight",
            candidate_refinements=["segment", "driver", "extra", "missing", "noise"],
            q_presets=["saturated", us.q_bounded_shift(0.5)],
            min_cell_weights=[1, 35],
            ambiguity_limit=0.05,
        )

        explanation = report.explain_minimal_stable()
        self.assertIsInstance(explanation, us.FrontierCandidateExplanation)
        explanation = cast(us.FrontierCandidateExplanation, explanation)
        markdown = explanation.to_markdown()
        payload = explanation.as_dict()
        explicit = report.explain(["driver"])

        self.assertEqual(explanation.candidate.added_columns, ("driver",))
        self.assertEqual(explicit.candidate.added_columns, ("driver",))
        self.assertAlmostEqual(explanation.baseline_ambiguity, 0.6)
        self.assertAlmostEqual(explanation.selected_ambiguity, 0.0)
        self.assertAlmostEqual(explanation.ambiguity_reduction, 0.6)
        self.assertAlmostEqual(explanation.ambiguity_reduction_percent, 100.0)
        self.assertEqual(explanation.added_public_cells, 1)
        self.assertEqual(len(explanation.scenario_comparisons), 8)
        self.assertEqual(len(explanation.failing_scenarios), 0)
        self.assertTrue(explanation.close_dominated_alternatives)
        self.assertEqual(
            [row.reason for row in explanation.screened_refinements],
            [
                "already public",
                "unavailable across hidden sets",
                "not present in any hidden set",
            ],
        )
        self.assertEqual(payload["candidate"]["added_columns"], ("driver",))
        self.assertIn("## Selected Representation Explanation", markdown)
        self.assertIn("Selected vs baseline max ambiguity: 0.6000 -> 0.0000", markdown)
        self.assertIn("### Ambiguity Reduction by Scenario", markdown)
        self.assertIn("### Close Dominated Alternatives", markdown)
        self.assertIn("### Screened-Out Refinements", markdown)
        self.assertIn("unavailable across hidden sets", markdown)
        self.assertIn("Search provenance: exhaustive (exact)", markdown)
        self.assertIn("## Selected Representation Explanation", report.to_markdown())

    def test_public_representation_frontier_rejects_hidden_sets_missing_public(self):
        with self.assertRaisesRegex(ValueError, "base_public columns"):
            us.public_representation_frontier(
                [{"segment": "A", "driver": "x", "target": 1.0}],
                base_public=["segment"],
                hidden=["segment", "driver"],
                hidden_sets=[["driver"]],
                target="target",
            )

    def test_public_representation_frontier_rejects_conflicting_public_aliases(self):
        with self.assertRaisesRegex(TypeError, "use either 'base_public'"):
            us.public_representation_frontier(
                [{"segment": "A", "driver": "x", "target": 1.0}],
                base_public=["segment"],
                public=["other"],
                hidden=["segment", "driver"],
                target="target",
            )


if __name__ == "__main__":
    unittest.main()
