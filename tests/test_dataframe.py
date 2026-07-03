from __future__ import annotations

import unittest

import updatesupport as us


class FakeFrame:
    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        if orient != "records":
            raise AssertionError(f"unexpected orient: {orient!r}")
        return list(self.records)


class FromDataFrameTests(unittest.TestCase):
    def test_from_dataframe_compiles_weighted_grouped_problem(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 30},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 30},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 40},
        ]

        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertIsInstance(grouped, us.GroupedProblem)
        self.assertEqual(grouped.public_columns, ("public",))
        self.assertEqual(grouped.hidden_columns, ("public", "hidden"))
        self.assertEqual(grouped.target_column, "target")
        self.assertIsInstance(grouped.target_functional, us.LinearTarget)
        self.assertIs(grouped.problem.target_functional, grouped.target_functional)
        self.assertEqual(grouped.problem.target_contract.kind, "linear")
        self.assertTrue(grouped.problem.target_contract.fixed_after_compilation)
        self.assertEqual(grouped.problem.target_contract.name, "target")
        self.assertAlmostEqual(grouped.total_weight, 100.0)
        self.assertAlmostEqual(grouped.public_law[("A",)], 0.6)
        self.assertAlmostEqual(grouped.public_law[("B",)], 0.4)
        self.assertFalse(grouped.problem.is_public_adequate())
        self.assertAlmostEqual(interval.lower, 0.2)
        self.assertAlmostEqual(interval.upper, 0.8)
        self.assertAlmostEqual(interval.diameter, 0.6)

    def test_from_dataframe_accepts_dataframe_like_records(self):
        frame = FakeFrame(
            [
                {"public": "A", "hidden": "x", "target": 0.0},
                {"public": "A", "hidden": "y", "target": 1.0},
                {"public": "B", "hidden": "z", "target": 0.5},
            ]
        )

        grouped = us.from_dataframe(
            frame,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
        )

        self.assertAlmostEqual(grouped.total_weight, 3.0)
        self.assertEqual(len(grouped.problem.states), 3)

    def test_from_dataframe_filters_tiny_hidden_cells(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 0.0, "weight": 1},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 10},
            {"public": "B", "hidden": "z", "target": 0.5, "weight": 10},
        ]

        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
            min_cell_weight=2,
        )

        self.assertNotIn(("A", "x"), grouped.problem.states)
        self.assertAlmostEqual(grouped.total_weight, 20.0)
        self.assertAlmostEqual(grouped.public_law[("A",)], 0.5)
        self.assertAlmostEqual(grouped.public_law[("B",)], 0.5)
        self.assertIsInstance(grouped.diagnostics, us.DataDiagnostics)
        self.assertAlmostEqual(grouped.diagnostics.total_weight, 21.0)
        self.assertAlmostEqual(grouped.diagnostics.retained_weight, 20.0)
        self.assertAlmostEqual(grouped.diagnostics.dropped_weight, 1.0)
        self.assertAlmostEqual(grouped.diagnostics.dropped_weight_share, 1 / 21)
        self.assertEqual(grouped.diagnostics.hidden_cells, 3)
        self.assertEqual(grouped.diagnostics.retained_hidden_cells, 2)
        self.assertEqual(grouped.diagnostics.dropped_hidden_cells, 1)
        self.assertIn(
            "min_cell_weight_dropped_cells",
            {row.code for row in grouped.diagnostics.diagnostics},
        )
        self.assertIn(
            "singleton_public_fibers",
            {row.code for row in grouped.diagnostics.diagnostics},
        )

    def test_from_dataframe_diagnoses_missing_categories_and_constant_fibers(self):
        rows = [
            {"public": "A", "hidden": "x", "target": 1.0, "weight": 1},
            {"public": "A", "hidden": "y", "target": 1.0, "weight": 1},
            {"public": None, "hidden": "z", "target": 0.5, "weight": 1},
        ]

        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
        )

        codes = {row.code for row in grouped.diagnostics.diagnostics}
        self.assertIn("missing_category_values", codes)
        self.assertIn("constant_target_public_fibers", codes)
        self.assertIn(("NA", "z"), grouped.problem.states)

    def test_from_dataframe_requires_public_columns_to_refine_hidden_columns(self):
        with self.assertRaisesRegex(
            ValueError, "public columns must also be hidden columns"
        ):
            us.from_dataframe(
                [{"public": "A", "hidden": "x", "target": 1.0}],
                public=["public"],
                hidden=["hidden"],
                target="target",
            )

    def test_from_dataframe_rejects_negative_weights(self):
        with self.assertRaisesRegex(ValueError, "row weights must be non-negative"):
            us.from_dataframe(
                [{"public": "A", "hidden": "x", "target": 1.0, "weight": -1}],
                public=["public"],
                hidden=["public", "hidden"],
                target="target",
                weight="weight",
            )

    def test_from_dataframe_accepts_row_metric_target(self):
        rows = [
            {"public": "A", "hidden": "x", "pd": 0.02, "lgd": 0.5, "ead": 100},
            {"public": "A", "hidden": "y", "pd": 0.04, "lgd": 0.5, "ead": 100},
            {"public": "B", "hidden": "z", "pd": 0.01, "lgd": 0.4, "ead": 200},
        ]
        metric = us.row_metric(
            "expected_loss_rate",
            lambda row: row["pd"] * row["lgd"],
            columns=("pd", "lgd"),
            description="expected loss rate",
        )

        grouped = us.from_dataframe(
            rows,
            public=["public"],
            hidden=["public", "hidden"],
            target=metric,
            weight="ead",
        )

        self.assertIs(grouped.target_column, metric)
        self.assertEqual(grouped.problem.target_contract.name, "expected_loss_rate")
        self.assertEqual(
            grouped.problem.target_contract.description, "expected loss rate"
        )
        self.assertAlmostEqual(grouped.problem.estimand_map[("A", "x")], 0.01)
        self.assertAlmostEqual(grouped.problem.estimand_map[("A", "y")], 0.02)
        self.assertAlmostEqual(grouped.problem.estimand_map[("B", "z")], 0.004)

    def test_from_dataframe_rejects_unsupported_nonlinear_target(self):
        target = us.UnsupportedTarget(
            name="approval_quantile",
            kind="quantile",
            formula="Q_0.9(Y)",
            reason="Quantile targets need a dedicated target-functional backend.",
        )

        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "supports only fixed linear plug-in targets",
        ):
            us.from_dataframe(
                [{"public": "A", "hidden": "x", "target": 1.0}],
                public=["public"],
                hidden=["public", "hidden"],
                target=target,
            )


if __name__ == "__main__":
    unittest.main()
