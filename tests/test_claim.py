from __future__ import annotations

import json
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
        {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
        {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
    ]


class ReportingClaimTests(unittest.TestCase):
    def test_claim_factory_and_audit_alias_return_claim_audit(self):
        claim = us.claim(
            "Factory claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        report = claim.audit(_rows())
        recommendations = report.recommend_refinements()
        tables = report.to_tables()

        self.assertIsInstance(claim, us.ClaimSpec)
        self.assertIsInstance(report, us.ClaimAudit)
        self.assertTrue(report.failed)
        self.assertEqual(report.observed_value, report.primary.observed_value)
        self.assertAlmostEqual(report.ambiguity, report.primary.interval.diameter)
        self.assertEqual(recommendations[0].columns, ("driver",))
        self.assertTrue(recommendations[0].selected_repair)
        self.assertTrue(recommendations[0].meets_ambiguity_limit)
        self.assertIsInstance(
            recommendations[0],
            us.ClaimRefinementRecommendation,
        )
        self.assertIn("claim_refinement_recommendations", tables)
        self.assertIn("claim-centered", report.to_markdown().lower())

    def test_grouped_problem_helpers_keep_low_level_workflow_available(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q=us.q_bounded_shift(0.5),
        )

        report = grouped.to_report(title="Grouped Evidence")
        claim = grouped.claim("Grouped claim")
        verdict = claim.verify(grouped)

        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertEqual(report.title, "Grouped Evidence")
        self.assertIsInstance(claim, us.ReportingClaim)
        self.assertTrue(verdict.inconclusive)
        self.assertEqual(verdict.claim.public, ("segment",))

    def test_verify_claim_passes_when_current_public_representation_is_stable(self):
        claim = us.ReportingClaim(
            estimate_name="Demo target rate",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=[us.q_bounded_shift(0.5)],
            candidate_refinements=["driver"],
            ambiguity_limit=0.31,
            statistical_interval=(0.45, 0.55),
        )

        report = us.verify_claim(_rows(), claim)
        markdown = report.to_markdown()
        payload = report.as_dict()

        self.assertIsInstance(report, us.ClaimVerificationReport)
        self.assertTrue(report.passed)
        self.assertEqual(report.status, "pass")
        self.assertAlmostEqual(report.primary.interval.diameter, 0.3)
        self.assertIsNotNone(report.certificate)
        self.assertEqual(report.certificate.status, "pass")
        self.assertIsNotNone(report.witness)
        self.assertEqual(payload["claim"]["estimate_name"], "Demo target rate")
        self.assertIn("## Verdict", markdown)
        self.assertIn("Statistical uncertainty", markdown)
        self.assertIn("Counterexample Witness", markdown)

    def test_verify_claim_fails_and_reports_repair_when_current_claim_breaks(self):
        claim = us.ReportingClaim(
            estimate_name="Strict target rate",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        report = us.verify_claim(_rows(), claim)
        tables = report.to_tables()

        self.assertTrue(report.failed)
        self.assertIsNotNone(report.witness)
        self.assertIsNotNone(report.certificate)
        self.assertEqual(report.certificate.status, "pass")
        self.assertEqual(report.repair_candidate.added_columns, ("driver",))
        self.assertIn("repair representation", report.reasons[-1])
        self.assertEqual(tables["summary"][0]["status"], "fail")
        self.assertEqual(tables["summary"][0]["repair_label"], "base + driver")
        self.assertIn("primary_summary", tables)
        self.assertIn("certificate_summary", tables)
        self.assertIn("witness_cell_shifts", tables)

    def test_verify_claim_is_inconclusive_without_ambiguity_limit(self):
        report = us.verify_claim(
            _rows(),
            {
                "estimate_name": "Exploratory target rate",
                "public": ["segment"],
                "hidden": ["segment", "driver"],
                "target": "target",
                "candidate_refinements": ["driver"],
            },
        )

        self.assertTrue(report.inconclusive)
        self.assertIn("No ambiguity limit", report.reasons[0])
        self.assertIsNotNone(report.certificate)
        self.assertIn("cannot issue a pass/fail", report.to_markdown())

    def test_claim_verification_exports_json_and_tables(self):
        claim = us.ReportingClaim(
            estimate_name="Exported claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )
        report = claim.verify(_rows())

        payload = json.loads(report.to_json())
        tables = us.report_tables(report)

        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["claim"]["estimate_name"], "Exported claim")
        self.assertEqual(tables["claim"][0]["estimate_name"], "Exported claim")
        self.assertIn("reasons", tables)
        self.assertIn("primary_refinements", tables)

    def test_verify_claim_passes_when_decision_is_invariant(self):
        claim = us.ReportingClaim(
            estimate_name="Decision-stable target",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            decision=us.threshold_decision("<=", 0.9, label="target within limit"),
        )

        report = us.verify_claim(_rows(), claim)
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertTrue(report.passed)
        self.assertIsInstance(report.decision, us.DecisionResult)
        self.assertTrue(report.decision.invariant)
        self.assertEqual(report.decision.certified_decision, "pass")
        self.assertIn("Decision Invariance", markdown)
        self.assertIn("Decision invariant: yes", markdown)
        self.assertEqual(tables["summary"][0]["decision_invariant"], True)
        self.assertEqual(tables["decision"][0]["certified_decision"], "pass")

    def test_verify_claim_fails_and_reports_decision_repair_when_threshold_crosses(
        self,
    ):
        claim = us.ReportingClaim(
            estimate_name="Decision-crossing target",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            candidate_refinements=["driver"],
            decision={
                "operator": "<=",
                "threshold": 0.6,
                "label": "target within tolerance",
            },
        )

        report = us.verify_claim(_rows(), claim)
        payload = json.loads(report.to_json())
        tables = report.to_tables()

        self.assertTrue(report.failed)
        self.assertIsNotNone(report.decision)
        self.assertFalse(report.decision.invariant)
        self.assertIsNotNone(report.decision_repair_candidate)
        self.assertEqual(report.decision_repair_candidate.added_columns, ("driver",))
        self.assertEqual(report.repair_candidate.added_columns, ("driver",))
        self.assertIn("decision-invariant repair", report.reasons[-1])
        self.assertEqual(payload["decision"]["threshold_crossed"], True)
        self.assertIn("decision_repair", tables)
        self.assertEqual(
            tables["decision_repair"][0]["added_columns"],
            ("driver",),
        )

    def test_verify_claim_supports_model_assisted_joint_draws(self):
        claim = us.ReportingClaim(
            estimate_name="Model-assisted claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
            min_cell_weight=0,
        )
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            effective_sample_size=10,
        )

        report = us.verify_claim(
            _rows(),
            claim,
            joint_model=joint,
            joint_draws=5,
            joint_seed=123,
        )
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertIsInstance(report.model_assisted, us.ModelAssistedStabilitySummary)
        self.assertEqual(report.model_assisted.draw_count, 5)
        self.assertEqual(report.model_assisted.successful_draws, 5)
        self.assertEqual(report.model_assisted.failed_draws, 5)
        self.assertEqual(report.model_assisted.failure_rate, 1.0)
        self.assertIn("Model-Assisted Joint Analysis", markdown)
        self.assertIn("model_assisted_summary", tables)
        self.assertIn("model_assisted_metric_summaries", tables)
        self.assertIn("model_assisted_draws", tables)
        self.assertEqual(len(tables["model_assisted_draws"]), 5)

    def test_verify_claim_can_fit_model_assisted_joint_model_implicitly(self):
        claim = us.ReportingClaim(
            estimate_name="Implicit joint claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            ambiguity_limit=0.05,
            min_cell_weight=0,
        )

        report = claim.verify(_rows(), joint_draws=3, joint_seed=99)

        self.assertIsNotNone(report.model_assisted)
        self.assertEqual(report.model_assisted.draw_count, 3)
        self.assertEqual(report.model_assisted.joint_model.cell_count, 3)


if __name__ == "__main__":
    unittest.main()
