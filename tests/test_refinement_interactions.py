from __future__ import annotations

import json
import unittest

import updatesupport as us


def _xor_rows() -> list[dict[str, object]]:
    return [
        {"public": "all", "a": 0, "b": 0, "target": 1.0, "weight": 25},
        {"public": "all", "a": 0, "b": 1, "target": 0.0, "weight": 25},
        {"public": "all", "a": 1, "b": 0, "target": 0.0, "weight": 25},
        {"public": "all", "a": 1, "b": 1, "target": 1.0, "weight": 25},
    ]


class InteractionRefinementTests(unittest.TestCase):
    def test_interaction_search_finds_pair_missed_by_single_columns(self):
        report = us.recommend_refinement_interactions(
            _xor_rows(),
            public=["public"],
            hidden=["public", "a", "b"],
            target="target",
            weight="weight",
            candidate_refinements=["public", "a", "missing", "a", "b"],
            max_order=2,
            top=None,
            q="saturated",
        )

        self.assertIsInstance(report, us.InteractionRefinementReport)
        self.assertEqual(report.candidate_refinements, ("a", "b"))
        self.assertAlmostEqual(report.baseline_ambiguity, 1.0)
        self.assertEqual(len(report.singletons), 2)
        self.assertTrue(all(row.reduction == 0.0 for row in report.singletons))

        best = report.best
        if best is None:
            self.fail("expected an interaction refinement candidate")
        self.assertEqual(best.columns, ("a", "b"))
        self.assertEqual(best.order, 2)
        self.assertAlmostEqual(best.after_ambiguity, 0.0)
        self.assertAlmostEqual(best.reduction, 1.0)
        self.assertAlmostEqual(best.interaction_gain, 1.0)
        self.assertAlmostEqual(best.additive_synergy, 1.0)
        self.assertEqual(report.best_interaction, best)

        markdown = report.to_markdown()
        tables = report.to_tables()
        helper_tables = us.report_tables(report)
        payload = json.loads(report.to_json())

        self.assertIn("Interaction Candidates", markdown)
        self.assertIn("interaction_gain", markdown)
        self.assertIn("interaction_candidates", tables)
        self.assertIn("interaction_candidates", helper_tables)
        self.assertEqual(payload["best"]["columns"], ["a", "b"])

    def test_interaction_search_respects_evaluation_cap(self):
        rows = [{**row, "c": row["a"]} for row in _xor_rows()]

        report = us.recommend_refinement_interactions(
            rows,
            public=["public"],
            hidden=["public", "a", "b", "c"],
            target="target",
            weight="weight",
            candidate_refinements=["a", "b", "c"],
            max_order=2,
            max_evaluations=2,
            top=None,
        )

        self.assertEqual(report.evaluated_sets, 2)
        self.assertTrue(report.truncated)
        self.assertEqual(len(report.candidates), 2)

    def test_interaction_search_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "max_order"):
            us.recommend_refinement_interactions(
                _xor_rows(),
                public=["public"],
                hidden=["public", "a", "b"],
                target="target",
                candidate_refinements=["a", "b"],
                max_order=0,
            )


if __name__ == "__main__":
    unittest.main()
