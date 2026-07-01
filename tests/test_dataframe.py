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


if __name__ == "__main__":
    unittest.main()
