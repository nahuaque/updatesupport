from __future__ import annotations

import unittest

import updatesupport as us


class FakeEstimate:
    value = 0.123


class FakeRefutation:
    def __init__(self, *, estimated_effect, new_effect, refutation_type):
        self.estimated_effect = estimated_effect
        self.new_effect = new_effect
        self.refutation_type = refutation_type


class DoWhySupportTests(unittest.TestCase):
    def test_audit_dowhy_effects_wraps_effect_report(self):
        audit = us.audit_dowhy_effects(
            _effect_rows(),
            estimate=FakeEstimate(),
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            candidate_refinements=["hidden"],
            q=us.q_bounded_shift(0.5),
            min_cell_weight=1,
            top=1,
        )

        self.assertIsInstance(audit, us.DoWhyRepresentationAudit)
        self.assertIsInstance(audit.report, us.PublicDescentReport)
        self.assertEqual(
            audit.report.title, "DoWhy Effect Representation Stability Audit"
        )
        self.assertAlmostEqual(audit.estimated_effect, 0.123)
        self.assertAlmostEqual(audit.new_effect[0], 0.08)
        self.assertAlmostEqual(audit.new_effect[1], 0.20)
        self.assertAlmostEqual(audit.ambiguity, 0.12)
        self.assertIn("DoWhy Effect Representation", audit.to_markdown())

    def test_audit_dowhy_effects_uses_report_observed_value_without_estimate(self):
        audit = us.audit_dowhy_effects(
            _effect_rows(),
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            min_cell_weight=1,
        )

        self.assertAlmostEqual(audit.estimated_effect, audit.report.observed_value)

    def test_to_refutation_attaches_updatesupport_metadata(self):
        audit = us.audit_dowhy_effects(
            _effect_rows(),
            estimate=FakeEstimate(),
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            min_cell_weight=1,
        )

        refutation = audit.to_refutation(refutation_class=FakeRefutation)

        self.assertIsInstance(refutation, FakeRefutation)
        self.assertAlmostEqual(refutation.estimated_effect, 0.123)
        self.assertEqual(
            refutation.new_effect,
            (audit.report.interval.lower, audit.report.interval.upper),
        )
        self.assertEqual(refutation.refutation_type, audit.refutation_type)
        self.assertIs(refutation.updatesupport_report, audit.report)
        self.assertEqual(refutation.updatesupport_interval, refutation.new_effect)
        self.assertAlmostEqual(
            refutation.updatesupport_ambiguity,
            audit.report.interval.diameter,
        )
        self.assertEqual(
            refutation.updatesupport_public_adequate,
            audit.report.public_adequate,
        )

    def test_refutation_converter_accepts_explicit_baseline(self):
        report = us.audit_effects(
            _effect_rows(),
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            min_cell_weight=1,
        )

        refutation = us.dowhy_refutation_from_report(
            report,
            estimate=FakeEstimate(),
            estimated_effect=0.456,
            refutation_class=FakeRefutation,
        )

        self.assertAlmostEqual(refutation.estimated_effect, 0.456)

    def test_non_numeric_estimate_is_rejected(self):
        report = us.audit_effects(
            _effect_rows(),
            public=["public"],
            hidden=["public", "hidden"],
            effect="tau_hat",
            weight="weight",
            min_cell_weight=1,
        )

        with self.assertRaisesRegex(ValueError, "estimate must be numeric"):
            us.dowhy_refutation_from_report(
                report,
                estimate=object(),
                refutation_class=FakeRefutation,
            )


def _effect_rows():
    return [
        {"public": "A", "hidden": "x", "tau_hat": -0.1, "weight": 30},
        {"public": "A", "hidden": "y", "tau_hat": 0.3, "weight": 30},
        {"public": "B", "hidden": "z", "tau_hat": 0.2, "weight": 40},
    ]


if __name__ == "__main__":
    unittest.main()
