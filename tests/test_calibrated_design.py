from __future__ import annotations

import json
import unittest

import updatesupport as us


_BASE_MASS = {
    ("a", "new"): 0.175,
    ("a", "established"): 0.175,
    ("b", "new"): 0.175,
    ("b", "established"): 0.175,
    ("c", "new"): 0.150,
    ("c", "established"): 0.150,
}


def _rows_for_mass(
    mass: dict[tuple[str, str], float],
    *,
    period: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    for (channel, tenure), share in mass.items():
        row: dict[str, object] = {
            "segment": "all",
            "channel": channel,
            "tenure": tenure,
            "channel_metric": {"a": 0.0, "b": 0.1, "c": 1.0}[channel],
            "tenure_metric": 0.0 if tenure == "new" else 1.0,
            "weight": 1000.0 * share,
        }
        if period is not None:
            row["period"] = period
        rows.append(row)
    return rows


def _transfer(
    mass: dict[tuple[str, str], float],
    source: tuple[str, str],
    destination: tuple[str, str],
    amount: float = 0.05,
) -> dict[tuple[str, str], float]:
    updated = dict(mass)
    updated[source] -= amount
    updated[destination] += amount
    return updated


def _historical_rows() -> list[dict[str, object]]:
    p1 = dict(_BASE_MASS)
    p2 = _transfer(p1, ("a", "new"), ("a", "established"))
    p3 = _transfer(p2, ("b", "new"), ("b", "established"))
    p4 = _transfer(p3, ("c", "new"), ("c", "established"))
    rows = []
    for period, mass in zip(("P1", "P2", "P3", "P4"), (p1, p2, p3, p4)):
        rows.extend(_rows_for_mass(mass, period=period))
    return rows


def _current_rows() -> list[dict[str, object]]:
    return _rows_for_mass(dict(_BASE_MASS))


def _claims() -> tuple[us.ClaimSpec, us.ClaimSpec, us.ClaimSpec]:
    common = {
        "public": ["segment"],
        "hidden": ["segment", "channel", "tenure"],
        "weight": "weight",
        "candidate_refinements": ["channel", "tenure"],
        "min_cell_weight": 0.0,
    }
    return (
        us.claim(
            "Channel metric ambiguity stays below eight points",
            target="channel_metric",
            ambiguity_limit=0.08,
            **common,
        ),
        us.claim(
            "Channel metric remains above the decision floor",
            target="channel_metric",
            decision=us.threshold_decision(">=", 0.32),
            **common,
        ),
        us.claim(
            "Tenure metric ambiguity stays below eight points",
            target="tenure_metric",
            ambiguity_limit=0.08,
            **common,
        ),
    )


class CalibratedPublicReportDesignTests(unittest.TestCase):
    def _portfolio_design(self):
        portfolio = us.claim_portfolio(
            _claims(),
            name="Calibrated KPI report",
            candidate_refinements=["channel", "tenure"],
        )
        return portfolio.design_calibrated(
            _historical_rows(),
            _current_rows(),
            period="period",
            coverage=1.0,
            min_train_transitions=1,
            rollup_column="channel",
            rollup_max_groups=2,
            rollup_output_column="channel_group",
            max_added_columns=2,
            threshold_margin=1e-8,
        )

    def test_combines_calibration_rollup_shared_design_and_breaking_witness(self):
        design = self._portfolio_design()

        self.assertIsInstance(design, us.CalibratedPublicReportDesign)
        self.assertEqual(design.status, "calibrated_design_found")
        self.assertEqual(design.design_kind, "shared")
        self.assertTrue(design.all_claims_certified)
        self.assertEqual(design.certified_claim_count, 3)
        for radius in design.calibrated_radii:
            self.assertAlmostEqual(radius, 0.05)

        self.assertIsNotNone(design.rollup)
        self.assertTrue(design.rollup_applied)
        self.assertEqual(design.rollup.selected.groups, (("a", "b"), ("c",)))
        self.assertEqual(design.rollup_anchor_claim_index, 1)
        self.assertEqual(
            design.candidate_refinements,
            ("channel_group", "tenure"),
        )

        self.assertIsNotNone(design.shared_design)
        self.assertEqual(
            design.recommended_public,
            ("segment", "channel_group", "tenure"),
        )
        self.assertEqual(
            design.shared_design.selected.added_columns,
            ("channel_group", "tenure"),
        )

        decision_result = design.claim_results[1]
        self.assertTrue(decision_result.certified)
        self.assertIsNotNone(decision_result.breaking_witness)
        self.assertEqual(decision_result.breaking_witness.status, "found")
        self.assertAlmostEqual(
            decision_result.breaking_tv_distance,
            0.1500001,
            places=6,
        )
        self.assertAlmostEqual(
            decision_result.breaking_radius_multiple,
            3.000002,
            places=5,
        )
        self.assertIsNone(design.claim_results[0].breaking_witness)
        self.assertIsNone(design.claim_results[2].breaking_witness)

        tables = design.to_tables()
        payload = json.loads(design.to_json())
        markdown = design.to_markdown()
        self.assertIn("claim_outcomes", tables)
        self.assertIn("calibration_transitions", tables)
        self.assertIn("rollup_selected_groups", tables)
        self.assertIn("shared_design_candidates", tables)
        self.assertIn("breaking_witnesses", tables)
        self.assertIn("breaking_transfers", tables)
        self.assertEqual(payload["certified_claim_count"], 3)
        self.assertIn("3/3", markdown)
        self.assertIn("radius multiple", markdown)

    def test_single_claim_method_returns_single_claim_design(self):
        claim = _claims()[0]

        design = claim.design_calibrated(
            _historical_rows(),
            _current_rows(),
            period="period",
            coverage=1.0,
            min_train_transitions=1,
            candidate_refinements=["channel"],
            max_added_columns=1,
        )

        self.assertEqual(design.design_kind, "single_claim")
        self.assertIsNotNone(design.public_design)
        self.assertIsNone(design.shared_design)
        self.assertTrue(design.all_claims_certified)
        self.assertEqual(design.recommended_public, ("segment", "channel"))

    def test_functional_api_and_validation(self):
        claim = _claims()[0]
        design = us.design_calibrated_public_report(
            _historical_rows(),
            _current_rows(),
            claim,
            period="period",
            coverage=1.0,
            min_train_transitions=1,
            candidate_refinements=["channel"],
        )

        self.assertIn("design_calibrated_public_report", us.__all__)
        self.assertIn("CalibratedPublicReportDesign", us.__all__)
        self.assertEqual(design.claim_count, 1)

        with self.assertRaisesRegex(ValueError, "rollup_claim_index"):
            us.design_calibrated_public_report(
                _historical_rows(),
                _current_rows(),
                claim,
                period="period",
                rollup_claim_index=1,
                rollup_column="channel",
            )

    def test_freezes_and_audits_without_refitting(self):
        policy = self._portfolio_design().freeze()

        self.assertIsInstance(policy, us.FrozenPublicReportPolicy)
        self.assertIn("FrozenPublicReportPolicy", us.__all__)
        self.assertEqual(
            policy.recommended_public,
            ("segment", "channel_group", "tenure"),
        )
        self.assertEqual(len(policy.fingerprint), 16)
        self.assertEqual(
            policy.fingerprint, self._portfolio_design().freeze().fingerprint
        )
        self.assertTrue(
            all(not claim.claim.candidate_refinements for claim in policy.claims)
        )
        policy_tables = policy.to_tables()
        policy_payload = json.loads(policy.to_json())
        self.assertIn("reference_support", policy_tables)
        self.assertIn("rollup_mapping", policy_tables)
        self.assertEqual(policy_payload["fingerprint"], policy.fingerprint)

        audit = policy.audit(_current_rows(), period="P5")

        self.assertIsInstance(audit, us.FrozenPolicyAudit)
        self.assertEqual(audit.status, "pass")
        self.assertTrue(audit.support_compatible)
        self.assertEqual(audit.pass_count, 3)
        self.assertEqual(audit.radius_breach_count, 0)
        for result in audit.claim_results:
            self.assertEqual(result.status, "pass")
            self.assertAlmostEqual(result.drift.actual_tv_radius, 0.0)
            self.assertTrue(result.drift.within_calibrated_radius)
        self.assertIsNotNone(audit.claim_results[1].breaking_witness)

        tables = audit.to_tables()
        payload = json.loads(audit.to_json())
        self.assertIn("support_drift", tables)
        self.assertIn("breaking_witnesses", tables)
        self.assertEqual(payload["status"], "pass")
        self.assertIn("frozen design-period composition", audit.to_markdown())

    def test_frozen_rollup_policy_round_trips_and_audits(self):
        policy = self._portfolio_design().freeze()
        restored = us.FrozenPublicReportPolicy.from_json(policy.to_json())
        self.assertEqual(restored.fingerprint, policy.fingerprint)
        self.assertEqual(
            restored.rollup.selected_mapping, policy.rollup.selected_mapping
        )
        self.assertEqual(restored.as_dict(), policy.as_dict())
        self.assertEqual(
            restored.audit(_current_rows()).as_dict(),
            policy.audit(_current_rows()).as_dict(),
        )

    def test_compatible_radius_breach_requires_review(self):
        policy = self._portfolio_design().freeze()
        shifted = _transfer(
            dict(_BASE_MASS),
            ("a", "new"),
            ("b", "new"),
            amount=0.08,
        )

        audit = policy.audit(
            _rows_for_mass(shifted),
            include_breaking_witness=False,
        )

        self.assertEqual(audit.status, "review")
        self.assertTrue(audit.support_compatible)
        self.assertEqual(audit.radius_breach_count, 3)
        for result in audit.claim_results:
            self.assertEqual(result.status, "review")
            self.assertAlmostEqual(result.drift.actual_tv_radius, 0.08)
            self.assertFalse(result.drift.within_calibrated_radius)

    def test_unseen_rollup_category_is_inconclusive_not_an_exception(self):
        policy = self._portfolio_design().freeze()
        rows = _current_rows()
        rows.append(
            {
                "segment": "all",
                "channel": "d",
                "tenure": "new",
                "channel_metric": 0.4,
                "tenure_metric": 0.0,
                "weight": 10.0,
            }
        )

        audit = policy.audit(rows)

        self.assertEqual(audit.status, "inconclusive")
        self.assertFalse(audit.support_compatible)
        self.assertEqual(audit.unseen_rollup_categories, ("d",))
        self.assertIn("Unseen categories", audit.transform_error)
        self.assertTrue(all(row.audit is None for row in audit.claim_results))

    def test_backtest_applies_one_fingerprint_to_ordered_holdouts(self):
        policy = self._portfolio_design().freeze()
        shifted = _transfer(
            dict(_BASE_MASS),
            ("a", "new"),
            ("b", "new"),
            amount=0.08,
        )
        rows = [
            *_rows_for_mass(dict(_BASE_MASS), period="P5"),
            *_rows_for_mass(shifted, period="P6"),
        ]

        backtest = policy.backtest(rows, period="period")

        self.assertIsInstance(backtest, us.FrozenPolicyBacktest)
        self.assertEqual(backtest.period_order, ("P5", "P6"))
        self.assertEqual(backtest.pass_count, 1)
        self.assertEqual(backtest.review_count, 1)
        self.assertEqual(backtest.inconclusive_count, 0)
        self.assertAlmostEqual(backtest.pass_rate, 0.5)
        self.assertAlmostEqual(backtest.support_compatibility_rate, 1.0)
        self.assertEqual(backtest.radius_breach_count, 3)
        self.assertTrue(
            all(
                audit.policy_fingerprint == policy.fingerprint
                for audit in backtest.audits
            )
        )

        tables = backtest.to_tables()
        payload = json.loads(backtest.to_json())
        self.assertIn("claim_period_outcomes", tables)
        self.assertEqual(len(tables["periods"]), 2)
        self.assertEqual(payload["pass_rate"], 0.5)
        self.assertIn("No radius, rollup, schema", backtest.to_markdown())


if __name__ == "__main__":
    unittest.main()
