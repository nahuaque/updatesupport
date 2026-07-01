from __future__ import annotations

import unittest

import updatesupport as us


class FakeDataFrame:
    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        self.orient = orient
        return self.records


class FakeEconMLEstimator:
    def __init__(self):
        self.seen_x = None

    def effect(self, X):
        self.seen_x = X
        return [0.1 + 0.05 * index for index in range(len(X))]


class FakeDoWhyEstimate:
    value = 0.25


class FakeDoubleMLModel:
    coef = [0.4, 0.8]


class AdapterTests(unittest.TestCase):
    def test_dataframe_adapter_copies_existing_effect_column(self):
        result = us.adapt_dataframe_effects(
            FakeDataFrame(
                [
                    {"public": "A", "hidden": "x", "effect_estimate": 0.1},
                    {"public": "A", "hidden": "y", "effect_estimate": 0.3},
                ]
            ),
            effect="effect_estimate",
            effect_column="tau_hat",
        )

        self.assertIsInstance(result, us.EstimatorAdapterResult)
        self.assertEqual(result.source, "dataframe")
        self.assertEqual(result.effect_column, "tau_hat")
        self.assertEqual(result.source_rows, 2)
        self.assertAlmostEqual(result.rows[0]["tau_hat"], 0.1)
        self.assertAlmostEqual(result.rows[1]["tau_hat"], 0.3)

    def test_dataframe_adapter_accepts_separate_effect_values(self):
        result = us.adapt_dataframe_effects(
            _rows_without_effect(),
            effect_values=[[-0.1], [0.3], [0.2]],
            effect_column="tau_hat",
        )

        self.assertAlmostEqual(result.rows[0]["tau_hat"], -0.1)
        self.assertAlmostEqual(result.rows[1]["tau_hat"], 0.3)
        self.assertAlmostEqual(result.rows[2]["tau_hat"], 0.2)

        suite = result.causal_reporting_stability(
            public=["public"],
            hidden=["public", "hidden"],
            weight="weight",
            min_cell_weight=1,
            include_sensitivity=False,
            include_refinement_sensitivity=False,
        )
        self.assertIsInstance(suite, us.CausalReportingStabilitySuite)

    def test_dataframe_adapter_rejects_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "one value per row"):
            us.adapt_dataframe_effects(
                _rows_without_effect(),
                effect_values=[0.1],
            )

    def test_econml_adapter_calls_effect_and_can_audit(self):
        estimator = FakeEconMLEstimator()
        result = us.adapt_econml_effects(
            estimator,
            _rows_without_effect(),
            X=[["x"], ["y"], ["z"]],
        )

        self.assertEqual(estimator.seen_x, [["x"], ["y"], ["z"]])
        self.assertEqual(result.source, "econml")
        self.assertEqual(result.estimator_name, "FakeEconMLEstimator")
        self.assertAlmostEqual(result.rows[2]["tau_hat"], 0.2)

        report = result.audit_effects(
            public=["public"],
            hidden=["public", "hidden"],
            weight="weight",
            min_cell_weight=1,
        )
        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertEqual(report.row_count, 3)

    def test_dowhy_adapter_repeats_scalar_estimate(self):
        result = us.adapt_dowhy_effects(
            FakeDoWhyEstimate(),
            _rows_without_effect(),
        )

        self.assertEqual(result.source, "dowhy")
        self.assertEqual(result.effect_kind, "scalar causal estimate")
        self.assertAlmostEqual(result.metadata["estimated_effect"], 0.25)
        self.assertTrue(all(row["tau_hat"] == 0.25 for row in result.rows))

    def test_dowhy_adapter_can_require_explicit_effect_values(self):
        with self.assertRaisesRegex(ValueError, "effect_values"):
            us.adapt_dowhy_effects(
                FakeDoWhyEstimate(),
                _rows_without_effect(),
                allow_scalar=False,
            )

    def test_doubleml_adapter_repeats_selected_coefficient(self):
        result = us.adapt_doubleml_effects(
            FakeDoubleMLModel(),
            _rows_without_effect(),
            coef_index=1,
        )

        self.assertEqual(result.source, "doubleml")
        self.assertAlmostEqual(result.metadata["coef"], 0.8)
        self.assertTrue(all(row["tau_hat"] == 0.8 for row in result.rows))

    def test_doubleml_adapter_prefers_explicit_effect_values(self):
        result = us.adapt_doubleml_effects(
            FakeDoubleMLModel(),
            _rows_without_effect(),
            effect_values=[0.1, 0.2, 0.3],
        )

        self.assertEqual(result.effect_kind, "row-level causal effect")
        self.assertAlmostEqual(result.metadata["coef"], 0.4)
        self.assertAlmostEqual(result.rows[2]["tau_hat"], 0.3)


def _rows_without_effect():
    return [
        {"public": "A", "hidden": "x", "weight": 30},
        {"public": "A", "hidden": "y", "weight": 30},
        {"public": "B", "hidden": "z", "weight": 40},
    ]


if __name__ == "__main__":
    unittest.main()
