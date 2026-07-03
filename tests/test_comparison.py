from __future__ import annotations

import json
import unittest

import updatesupport as us


def _stable_rows() -> list[dict[str, object]]:
    cells = [
        ("retail", "short", 30),
        ("retail", "long", 30),
        ("enterprise", "long", 40),
    ]
    scores = {
        "alpha": (0.85, 0.80, 0.78),
        "beta": (0.70, 0.65, 0.70),
        "gamma": (0.50, 0.55, 0.60),
    }
    return _rows_from_scores(cells, scores)


def _ambiguous_winner_rows() -> list[dict[str, object]]:
    cells = [
        ("retail", "short", 30),
        ("retail", "long", 30),
        ("enterprise", "long", 40),
    ]
    scores = {
        "alpha": (0.80, 0.50, 0.75),
        "beta": (0.60, 0.65, 0.70),
        "gamma": (0.40, 0.45, 0.50),
    }
    return _rows_from_scores(cells, scores)


def _rows_from_scores(
    cells: list[tuple[str, str, int]],
    scores: dict[str, tuple[float, ...]],
) -> list[dict[str, object]]:
    rows = []
    for model, values in scores.items():
        for (segment, driver, weight), value in zip(cells, values, strict=True):
            rows.append(
                {
                    "model": model,
                    "segment": segment,
                    "driver": driver,
                    "score": value,
                    "weight": weight,
                }
            )
    return rows


class RobustComparisonTests(unittest.TestCase):
    def test_full_ranking_is_stable_when_pairwise_margins_stay_positive(self):
        report = us.robust_comparison_report(
            _stable_rows(),
            item="model",
            public=["segment"],
            hidden=["segment", "driver"],
            target="score",
            weight="weight",
            q="saturated",
            title="Stable Model Ranking",
        )

        self.assertEqual(report.status, "full_ranking_stable")
        self.assertEqual(report.observed_winner, "alpha")
        self.assertEqual(report.certified_winner, "alpha")
        self.assertTrue(report.full_ranking_stable)
        self.assertEqual(report.observed_order, ("alpha", "beta", "gamma"))
        self.assertTrue(all(row.robust_order for row in report.pairwise_results))
        self.assertIn("Pairwise Margins", report.to_markdown())

        tables = report.to_tables()
        helper_tables = us.report_tables(report)
        payload = json.loads(report.to_json())

        self.assertIn("items", tables)
        self.assertIn("pairwise_margins", tables)
        self.assertIn("pairwise_margins", helper_tables)
        self.assertEqual(payload["status"], "full_ranking_stable")

    def test_observed_winner_can_be_ambiguous_under_recomposition(self):
        report = us.robust_comparison_report(
            _ambiguous_winner_rows(),
            item="model",
            public=["segment"],
            hidden=["segment", "driver"],
            target="score",
            weight="weight",
            q="saturated",
            title="Ambiguous Model Ranking",
        )

        self.assertEqual(report.status, "ambiguous_winner")
        self.assertEqual(report.observed_winner, "alpha")
        self.assertIsNone(report.certified_winner)
        alpha_beta = next(
            row
            for row in report.pairwise_results
            if row.preferred_item == "alpha" and row.compared_item == "beta"
        )
        self.assertGreater(alpha_beta.observed_margin, 0.0)
        self.assertLess(alpha_beta.lower, 0.0)
        self.assertFalse(alpha_beta.robust_order)

    def test_lower_is_better_rankings_use_reversed_margin(self):
        rows = [
            {"model": "alpha", "segment": "A", "driver": "x", "loss": 0.10, "n": 50},
            {"model": "alpha", "segment": "A", "driver": "y", "loss": 0.20, "n": 50},
            {"model": "beta", "segment": "A", "driver": "x", "loss": 0.30, "n": 50},
            {"model": "beta", "segment": "A", "driver": "y", "loss": 0.35, "n": 50},
        ]

        report = us.robust_ranking_report(
            rows,
            item="model",
            public=["segment"],
            hidden=["segment", "driver"],
            target="loss",
            weight="n",
            q="saturated",
            higher_is_better=False,
        )

        self.assertEqual(report.observed_order, ("alpha", "beta"))
        self.assertEqual(report.status, "full_ranking_stable")
        self.assertGreater(report.pairwise_results[0].lower, 0.0)

    def test_comparison_requires_balanced_hidden_cell_support(self):
        rows = _stable_rows()
        rows = [
            row
            for row in rows
            if not (row["model"] == "beta" and row["driver"] == "long")
        ]

        with self.assertRaisesRegex(ValueError, "same hidden-cell support"):
            us.robust_comparison_report(
                rows,
                item="model",
                public=["segment"],
                hidden=["segment", "driver"],
                target="score",
                weight="weight",
            )


if __name__ == "__main__":
    unittest.main()
