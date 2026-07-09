from __future__ import annotations

import json
import math
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {"segment": "A", "driver": "low", "target": 0.0, "weight": 50},
        {"segment": "A", "driver": "high", "target": 1.0, "weight": 50},
    ]


def _claim(operator: str = ">=", threshold: float = 0.4) -> us.ClaimSpec:
    return us.claim(
        "target clears the review threshold",
        public=["segment"],
        hidden=["segment", "driver"],
        target="target",
        weight="weight",
        min_cell_weight=0.0,
        decision=us.threshold_decision(operator, threshold),
    )


class MinimumClaimBreakingWitnessTests(unittest.TestCase):
    def test_feature_is_on_claim_first_public_surface(self):
        self.assertIn("minimum_claim_breaking_witness", us.__all__)
        self.assertIn("MinimumClaimBreakingWitnessReport", us.__all__)
        self.assertTrue(hasattr(us.ClaimSpec, "breaking_witness"))
        self.assertTrue(hasattr(us.ClaimAudit, "breaking_witness"))

    def test_tv_witness_is_exact_minimum_mass_transfer(self):
        report = us.minimum_claim_breaking_witness(
            _rows(),
            _claim(),
            threshold_margin=1e-6,
        )

        self.assertIsInstance(report, us.MinimumClaimBreakingWitnessReport)
        self.assertEqual(report.status, "found")
        self.assertEqual(report.distance_metric, "tv")
        self.assertAlmostEqual(report.observed_value, 0.5)
        self.assertAlmostEqual(report.breaking_boundary, 0.399999)
        self.assertAlmostEqual(report.witness_value, 0.399999, places=7)
        self.assertEqual(report.witness_decision, "fail")
        self.assertAlmostEqual(report.distance, 0.100001, places=7)
        self.assertAlmostEqual(report.witness_tv_distance, report.distance, places=8)
        self.assertAlmostEqual(report.total_transferred_mass, report.distance, places=8)
        self.assertLess(report.public_law_error, 1e-9)
        self.assertEqual(len(report.transfers), 1)
        transfer = report.transfers[0]
        self.assertEqual(transfer.source_state, ("A", "high"))
        self.assertEqual(transfer.destination_state, ("A", "low"))
        self.assertAlmostEqual(transfer.target_change, -0.100001, places=7)

        tables = report.to_tables()
        payload = json.loads(report.to_json())
        self.assertEqual(tables["summary"][0]["status"], "found")
        self.assertEqual(len(tables["cell_shifts"]), 2)
        self.assertEqual(len(tables["transfers"]), 1)
        self.assertEqual(payload["witness_decision"], "fail")
        self.assertIn("closest decision-breaking", report.to_markdown().lower())

    def test_default_margin_produces_an_actual_decision_flip(self):
        report = _claim().breaking_witness(_rows())

        self.assertEqual(report.status, "found")
        self.assertEqual(report.witness_decision, "fail")
        self.assertLess(report.witness_value, 0.4)

    def test_claim_and_audit_methods_use_same_compiled_problem(self):
        claim = _claim()
        direct = claim.breaking_witness(_rows(), threshold_margin=1e-6)
        audit = claim.audit(_rows())
        from_audit = audit.breaking_witness(threshold_margin=1e-6)

        self.assertAlmostEqual(direct.distance, from_audit.distance)
        self.assertAlmostEqual(direct.witness_value, from_audit.witness_value)

    def test_lower_direction_rule_moves_mass_to_high_target_cell(self):
        report = _claim("<=", 0.6).breaking_witness(
            _rows(),
            threshold_margin=1e-6,
        )

        self.assertEqual(report.status, "found")
        self.assertAlmostEqual(report.witness_value, 0.600001, places=7)
        self.assertAlmostEqual(report.distance, 0.100001, places=7)
        self.assertEqual(report.transfers[0].source_state, ("A", "low"))
        self.assertEqual(report.transfers[0].destination_state, ("A", "high"))

    def test_infeasible_when_retained_support_cannot_break_claim(self):
        rows = [
            {"segment": "A", "driver": "x", "target": 1.0, "weight": 50},
            {"segment": "A", "driver": "y", "target": 1.0, "weight": 50},
        ]
        report = _claim(">=", 0.5).breaking_witness(rows)

        self.assertEqual(report.status, "infeasible")
        self.assertIsNone(report.distance)
        self.assertIsNone(report.witness_value)
        self.assertEqual(report.cells, ())
        self.assertEqual(report.transfers, ())

    def test_already_broken_claim_has_zero_distance(self):
        report = _claim(">=", 0.6).breaking_witness(_rows())

        self.assertEqual(report.status, "already_broken")
        self.assertEqual(report.distance, 0.0)
        self.assertEqual(report.witness_value, report.observed_value)
        self.assertEqual(report.transfers, ())

    def test_l2_and_identity_mahalanobis_match_expected_geometry(self):
        try:
            import cvxpy  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("cvxpy is not installed") from exc

        claim = _claim()
        l2 = claim.breaking_witness(
            _rows(),
            distance="l2",
            threshold_margin=1e-6,
            solver="CLARABEL",
        )
        mahalanobis = claim.breaking_witness(
            _rows(),
            distance="mahalanobis",
            covariance=[[1.0, 0.0], [0.0, 1.0]],
            threshold_margin=1e-6,
            solver="CLARABEL",
        )

        expected = math.sqrt(2.0) * 0.100001
        self.assertEqual(l2.status, "found")
        self.assertAlmostEqual(l2.distance, expected, places=6)
        self.assertAlmostEqual(l2.witness_tv_distance, 0.100001, places=6)
        self.assertAlmostEqual(mahalanobis.distance, l2.distance, places=6)

    def test_validation_requires_decision_and_valid_distance_configuration(self):
        no_decision = us.claim(
            "descriptive claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
        )
        with self.assertRaisesRegex(ValueError, "claim.decision"):
            no_decision.breaking_witness(_rows())
        with self.assertRaisesRegex(ValueError, "distance"):
            _claim().breaking_witness(_rows(), distance="kl")
        with self.assertRaisesRegex(ValueError, "covariance is required"):
            _claim().breaking_witness(_rows(), distance="mahalanobis")
        with self.assertRaisesRegex(ValueError, "positive"):
            _claim().breaking_witness(_rows(), threshold_margin=0.0)


if __name__ == "__main__":
    unittest.main()
