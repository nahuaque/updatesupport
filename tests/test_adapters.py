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

    def test_conformal_regression_adapter_adds_interval_targets(self):
        result = us.adapt_conformal_regression(
            _rows_without_effect(),
            prediction=[0.2, 0.6, 0.8],
            lower=[0.1, 0.4, 0.7],
            upper=[0.3, 0.9, 0.95],
            y_true=[0.25, 0.95, 0.75],
            threshold=0.5,
        )

        self.assertIsInstance(result, us.ConformalAdapterResult)
        self.assertEqual(result.source, "conformal_regression")
        self.assertEqual(result.source_rows, 3)
        self.assertAlmostEqual(result.rows[1]["interval_width"], 0.5)
        self.assertFalse(result.rows[1]["covered"])
        self.assertTrue(result.rows[1]["miscovered"])
        self.assertTrue(result.rows[1]["crosses_threshold"])

        claim = result.claim(
            "Threshold-crossing burden is stable",
            public=["public"],
            hidden=["public", "hidden"],
            target="crosses_threshold",
            weight="weight",
            candidate_refinements=["hidden"],
            ambiguity_limit=0.8,
        )
        design = result.design(claim)
        self.assertIsInstance(design, us.PublicReportDesign)

    def test_conformal_regression_adapter_accepts_interval_tensor(self):
        result = us.adapt_conformal_regression(
            _rows_without_effect(),
            interval=[
                [[0.1, 0.0], [0.3, 0.4]],
                [[0.4, 0.2], [0.9, 1.0]],
                [[0.7, 0.5], [0.95, 1.1]],
            ],
            interval_index=1,
        )

        self.assertAlmostEqual(result.rows[0]["y_lower"], 0.0)
        self.assertAlmostEqual(result.rows[0]["y_upper"], 0.4)
        self.assertIsNone(result.prediction_column)

    def test_conformal_classification_adapter_accepts_literal_sets(self):
        result = us.adapt_conformal_classification(
            _rows_without_effect(),
            prediction=["approve", "review", "reject"],
            prediction_sets=[
                {"approve"},
                {"approve", "review"},
                {"reject"},
            ],
            y_true=["approve", "reject", "reject"],
            positive_label="approve",
        )

        self.assertEqual(result.rows[0]["prediction_set"], ("approve",))
        self.assertEqual(result.rows[1]["prediction_set_size"], 2)
        self.assertTrue(result.rows[1]["ambiguous_set"])
        self.assertFalse(result.rows[1]["covered"])
        self.assertTrue(result.rows[1]["contains_positive_label"])

    def test_conformal_classification_adapter_accepts_class_masks(self):
        result = us.adapt_conformal_classification(
            _rows_without_effect(),
            classes=["approve", "review", "reject"],
            prediction_sets=[
                [True, False, False],
                [True, True, False],
                [False, False, True],
            ],
            positive_label="approve",
        )

        self.assertEqual(result.rows[0]["prediction_set"], ("approve",))
        self.assertEqual(result.rows[1]["prediction_set"], ("approve", "review"))
        self.assertEqual(result.metadata["classes"], ("approve", "review", "reject"))

    def test_conformal_regression_reporting_stability_discovers_targets(self):
        result = us.adapt_conformal_regression(
            _rows_without_effect(),
            prediction=[0.2, 0.6, 0.8],
            lower=[0.1, 0.4, 0.7],
            upper=[0.3, 0.9, 0.95],
            y_true=[0.25, 0.95, 0.75],
            threshold=0.5,
        )

        report = result.reporting_stability(
            public=["public"],
            hidden=["public", "hidden"],
            weight="weight",
            candidate_refinements=["hidden"],
            ambiguity_limits={
                "interval_width": 0.1,
                "crosses_threshold": 0.1,
            },
        )

        self.assertIsInstance(report, us.ConformalReportingStabilityReport)
        self.assertEqual(report.source, "conformal_regression")
        self.assertEqual(report.target_count, 7)
        self.assertIn("Conformal Reporting Stability", report.to_markdown())
        tables = report.to_tables()
        self.assertIn("targets", tables)
        self.assertIn("refinement_recommendations", tables)
        target_columns = {row["target"] for row in tables["targets"]}
        self.assertIn("interval_width", target_columns)
        self.assertIn("crosses_threshold", target_columns)

    def test_conformal_reporting_stability_accepts_explicit_classification_targets(
        self,
    ):
        result = us.adapt_conformal_classification(
            _rows_without_effect(),
            prediction_sets=[
                {"approve"},
                {"approve", "review"},
                {"reject"},
            ],
            y_true=["approve", "reject", "reject"],
            positive_label="approve",
        )

        report = us.conformal_reporting_stability(
            result,
            public=["public"],
            hidden=["public", "hidden"],
            weight="weight",
            targets=[
                "prediction_set_size",
                {
                    "column": "ambiguous_set",
                    "label": "ambiguous set rate",
                    "role": "manual_review",
                    "target_description": "share of non-singleton prediction sets",
                    "ambiguity_limit": 0.5,
                },
            ],
        )

        self.assertEqual(report.target_count, 2)
        self.assertEqual(report.target_audits[1].spec.label, "ambiguous set rate")
        self.assertIn("prediction_set_size", report.to_json())


def _rows_without_effect():
    return [
        {"public": "A", "hidden": "x", "weight": 30},
        {"public": "A", "hidden": "y", "weight": 30},
        {"public": "B", "hidden": "z", "weight": 40},
    ]


if __name__ == "__main__":
    unittest.main()
