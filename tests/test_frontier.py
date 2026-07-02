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
        self.assertIn("base public representation", markdown)
        self.assertIn("base + driver", markdown)
        self.assertIn("max ambiguity", markdown)
        self.assertIn("Pareto-frontier", markdown)

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
