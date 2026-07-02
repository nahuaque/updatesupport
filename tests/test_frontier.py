from __future__ import annotations

import unittest

import updatesupport as us


class PublicRepresentationFrontierTests(unittest.TestCase):
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
        assert minimal is not None
        self.assertEqual(minimal.added_columns, ("driver",))
        self.assertEqual(minimal.public_cells, 3)
        self.assertAlmostEqual(minimal.max_ambiguity, 0.0)
        self.assertTrue(minimal.passes_ambiguity_limit)
        self.assertTrue(minimal.public_adequate)

        budget_best = report.best_under_bucket_budget()
        self.assertIsNotNone(budget_best)
        assert budget_best is not None
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
        self.assertEqual([row.added_columns for row in report.candidates], [("driver",)])

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
        assert explanation is not None
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
