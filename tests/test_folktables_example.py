from __future__ import annotations

import unittest

from examples.folktables_acs import (
    TARGET_COLUMN,
    build_problem_from_rows,
    fiber_diagnostics,
    refinement_candidates,
)


class FolktablesExampleTests(unittest.TestCase):
    def test_grouped_problem_computes_observed_law_ambiguity(self):
        rows = [
            {"public": "A", "hidden": "x", TARGET_COLUMN: 0.0, "weight": 30},
            {"public": "A", "hidden": "y", TARGET_COLUMN: 1.0, "weight": 30},
            {"public": "B", "hidden": "z", TARGET_COLUMN: 0.5, "weight": 40},
        ]

        grouped = build_problem_from_rows(
            rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden"],
            weight_column="weight",
        )

        interval = grouped.problem.global_transport_modulus()
        diagnostics = fiber_diagnostics(grouped, top=1)

        self.assertFalse(grouped.problem.is_public_adequate())
        self.assertAlmostEqual(interval.lower, 0.2)
        self.assertAlmostEqual(interval.upper, 0.8)
        self.assertAlmostEqual(interval.diameter, 0.6)
        self.assertEqual(diagnostics[0]["public_value"], ("A",))
        self.assertAlmostEqual(diagnostics[0]["contribution"], 0.6)

    def test_refinement_candidates_rank_hidden_split(self):
        rows = [
            {
                "public": "A",
                "hidden": "x",
                "noise": "n",
                TARGET_COLUMN: 0.0,
                "weight": 30,
            },
            {
                "public": "A",
                "hidden": "y",
                "noise": "n",
                TARGET_COLUMN: 1.0,
                "weight": 30,
            },
            {
                "public": "B",
                "hidden": "z",
                "noise": "n",
                TARGET_COLUMN: 0.5,
                "weight": 40,
            },
        ]

        candidates = refinement_candidates(
            rows,
            public_columns=["public"],
            hidden_columns=["public", "hidden", "noise"],
            candidate_columns=["hidden", "noise"],
            weight_column="weight",
        )

        self.assertEqual(candidates[0]["column"], "hidden")
        self.assertAlmostEqual(candidates[0]["diameter"], 0.0)
        self.assertAlmostEqual(candidates[0]["reduction"], 0.6)


if __name__ == "__main__":
    unittest.main()
