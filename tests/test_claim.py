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


class ClaimSpecTests(unittest.TestCase):
    def test_old_claim_names_are_not_top_level_api(self):
        self.assertFalse(hasattr(us, "verify_claim"))
        self.assertFalse(hasattr(us, "ReportingClaim"))
        self.assertFalse(hasattr(us, "ClaimVerificationReport"))

    def test_curated_star_import_surface_uses_claim_names(self):
        self.assertIn("claim", us.__all__)
        self.assertIn("audit_claim", us.__all__)
        self.assertIn("claim_tree", us.__all__)
        self.assertIn("audit_claim_tree", us.__all__)
        self.assertIn("ClaimSpec", us.__all__)
        self.assertIn("ClaimAudit", us.__all__)
        self.assertIn("ClaimRepairPlan", us.__all__)
        self.assertIn("PublicReportDesign", us.__all__)
        self.assertIn("design_public_report", us.__all__)
        self.assertIn("ClaimTree", us.__all__)
        self.assertIn("ClaimTreeAudit", us.__all__)
        self.assertTrue(hasattr(us, "ClaimRepairOption"))
        self.assertTrue(hasattr(us, "plan_claim_repair"))
        self.assertTrue(hasattr(us, "ClaimScreeningResult"))
        self.assertNotIn("ClaimRepairOption", us.__all__)
        self.assertNotIn("plan_claim_repair", us.__all__)
        self.assertNotIn("ClaimScreeningResult", us.__all__)
        self.assertNotIn("verify_claim", us.__all__)
        self.assertNotIn("ReportingClaim", us.__all__)
        self.assertNotIn("ClaimVerificationReport", us.__all__)

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

    def test_claim_repair_plan_consolidates_repair_evidence(self):
        claim = us.claim(
            "Repairable claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )
        report = claim.audit(_rows())
        plan = report.repair_plan()
        helper_plan = us.plan_claim_repair(report)
        tables = us.report_tables(plan)
        payload = json.loads(plan.to_json())
        markdown = plan.to_markdown()

        self.assertIsInstance(plan, us.ClaimRepairPlan)
        self.assertIsInstance(plan.recommended, us.ClaimRepairOption)
        self.assertEqual(plan.status, "repair_found")
        self.assertEqual(plan.recommended.label, "driver")
        self.assertTrue(plan.recommended.certifies_claim)
        self.assertEqual(helper_plan.recommended.label, "driver")
        self.assertEqual(tables["summary"][0]["recommended_label"], "driver")
        self.assertIn("options", tables)
        self.assertEqual(payload["recommended"]["label"], "driver")
        self.assertIn("Candidate Repair Options", markdown)

    def test_claim_design_wraps_audit_repair_and_frontier(self):
        claim = us.claim(
            "Designable claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        design = claim.design(_rows())
        helper_design = us.design_public_report(claim, _rows())
        tables = design.to_tables()
        payload = json.loads(design.to_json())
        markdown = design.to_markdown()

        self.assertIsInstance(design, us.PublicReportDesign)
        self.assertEqual(design.status, "repair_available")
        self.assertEqual(design.recommended_public, ("segment", "driver"))
        self.assertIsInstance(design.audit, us.ClaimAudit)
        self.assertIsInstance(design.repair_plan, us.ClaimRepairPlan)
        self.assertIsNotNone(design.certificate)
        self.assertIsNotNone(design.frontier)
        self.assertEqual(helper_design.recommended_public, design.recommended_public)
        self.assertEqual(tables["summary"][0]["recommended_label"], "segment + driver")
        self.assertIn("repair_options", tables)
        self.assertIn("frontier_candidates", tables)
        self.assertEqual(payload["recommended_label"], "segment + driver")
        self.assertIn("Public Report Design", markdown)

    def test_claim_design_marks_passing_public_report_as_already_defensible(self):
        claim = us.claim(
            "Already defensible claim",
            public=["segment", "driver"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        design = claim.design(_rows())

        self.assertTrue(design.audit.passed)
        self.assertEqual(design.status, "already_defensible")
        self.assertEqual(design.recommended_public, ("segment", "driver"))
        self.assertIsNone(design.recommended_option)

    def test_claim_design_can_include_refinement_attribution(self):
        claim = us.claim(
            "Attribution design claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        design = claim.design(_rows(), include_attribution=True)

        self.assertIsNotNone(design.attribution)
        self.assertIn("refinement_attributions", design.to_tables())

    def test_plan_claim_repair_audits_claim_spec_and_uses_action_costs(self):
        rows = [
            {
                "segment": "A",
                "driver": "low",
                "driver_copy": "low",
                "target": 0.0,
                "weight": 30,
            },
            {
                "segment": "A",
                "driver": "high",
                "driver_copy": "high",
                "target": 1.0,
                "weight": 30,
            },
            {
                "segment": "B",
                "driver": "flat",
                "driver_copy": "flat",
                "target": 0.5,
                "weight": 40,
            },
        ]
        claim = us.claim(
            "Cost-aware repairable claim",
            public=["segment"],
            hidden=["segment", "driver", "driver_copy"],
            target="target",
            weight="weight",
            candidate_refinements=["driver", "driver_copy"],
            ambiguity_limit=0.05,
        )

        plan = us.plan_claim_repair(
            claim,
            rows,
            action_costs={"driver": 10.0, "driver_copy": 1.0},
        )

        self.assertEqual(plan.status, "repair_found")
        self.assertEqual(plan.recommended.label, "driver_copy")
        self.assertEqual(plan.recommended.cost, 1.0)
        self.assertTrue(plan.recommended.certifies_claim)
        self.assertEqual(plan.options[1].label, "driver")
        self.assertEqual(plan.options[1].cost, 10.0)

    def test_repair_plan_does_not_recommend_repair_for_passing_claim(self):
        claim = us.claim(
            "Already stable claim",
            public=["segment", "driver"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )
        report = claim.audit(_rows())
        plan = report.repair_plan()
        tables = plan.to_tables()

        self.assertTrue(report.passed)
        self.assertEqual(plan.status, "already_certified")
        self.assertIsNone(plan.recommended)
        self.assertIsNone(tables["summary"][0]["recommended_label"])

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
        verdict = claim.audit(grouped)

        self.assertIsInstance(report, us.PublicDescentReport)
        self.assertEqual(report.title, "Grouped Evidence")
        self.assertIsInstance(claim, us.ClaimSpec)
        self.assertTrue(verdict.inconclusive)
        self.assertEqual(verdict.claim.public, ("segment",))

    def test_residopt_screening_certifies_simple_l2_decision_when_available(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        claim = us.claim(
            "Screened launch decision",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            decision=us.threshold_decision(">=", -1.0),
            screening_backend="residopt",
            min_cell_weight=0.0,
        )

        report = us.audit_claim(_rows(), claim)
        tables = report.to_tables()

        self.assertTrue(report.passed)
        self.assertIsNotNone(report.screening)
        self.assertTrue(report.screening.used)
        self.assertTrue(report.screening.exact_solve_avoided)
        self.assertEqual(report.primary.interval.bound_type, "conservative")
        self.assertEqual(tables["screening"][0]["used"], True)
        self.assertIn("Endpoint Screening", report.to_markdown())

    def test_residopt_screening_falls_back_when_inconclusive(self):
        claim = us.claim(
            "Screened threshold decision",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            decision=us.threshold_decision(">=", 0.5),
            screening_backend="residopt",
            min_cell_weight=0.0,
        )

        report = us.audit_claim(_rows(), claim)

        self.assertIsNotNone(report.screening)
        self.assertFalse(report.screening.used)
        self.assertFalse(report.screening.exact_solve_avoided)
        self.assertEqual(report.primary.interval.bound_type, "exact")

    def test_claim_certificate_can_use_refinement_residopt_screening(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        claim = us.claim(
            "Screened refinement certificate",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=[us.q_l2_budget(0.05, solver="CLARABEL")],
            candidate_refinements=["driver"],
            ambiguity_limit=1.0,
            refinement_screening_backend="residopt",
            min_cell_weight=0.0,
        )

        report = us.audit_claim(_rows(), claim)
        payload = report.as_dict()
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertTrue(report.passed)
        self.assertIsNotNone(report.certificate)
        self.assertTrue(report.certificate.passed)
        self.assertIsNotNone(report.certificate.frontier.screening)
        self.assertEqual(
            report.certificate.frontier.screening.certified_count,
            report.certificate.frontier.screening.endpoint_count,
        )
        self.assertEqual(
            payload["certificate"]["screening"]["backend"],
            "residopt",
        )
        self.assertIn("Frontier screening", markdown)
        self.assertTrue(tables["summary"][0]["has_frontier_screening"])

    def test_audit_claim_passes_when_current_public_representation_is_stable(self):
        claim = us.ClaimSpec(
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

        report = us.audit_claim(_rows(), claim)
        markdown = report.to_markdown()
        payload = report.as_dict()

        self.assertIsInstance(report, us.ClaimAudit)
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

    def test_audit_claim_fails_and_reports_repair_when_current_claim_breaks(self):
        claim = us.ClaimSpec(
            estimate_name="Strict target rate",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        report = us.audit_claim(_rows(), claim)
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

    def test_audit_claim_is_inconclusive_without_ambiguity_limit(self):
        report = us.audit_claim(
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

    def test_claim_audit_exports_json_and_tables(self):
        claim = us.ClaimSpec(
            estimate_name="Exported claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )
        report = claim.audit(_rows())

        payload = json.loads(report.to_json())
        tables = us.report_tables(report)

        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["claim"]["estimate_name"], "Exported claim")
        self.assertEqual(tables["claim"][0]["estimate_name"], "Exported claim")
        self.assertIn("reasons", tables)
        self.assertIn("primary_refinements", tables)

    def test_claim_tree_audits_nested_claims_and_exports(self):
        root_claim = us.claim(
            "Overall posterior mean stable",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=[us.q_bounded_shift(0.5)],
            ambiguity_limit=0.31,
        )
        child_claim = us.claim(
            "Group-level posterior mean stable",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            candidate_refinements=["driver"],
            ambiguity_limit=0.05,
        )

        tree = us.claim_tree(
            us.ClaimNode(root_claim, role="overall"),
            children=[
                us.ClaimNode(
                    child_claim,
                    role="group",
                    metadata={"hierarchy_level": "region"},
                )
            ],
            name="Bayesian Hierarchy Claim Audit",
            description="Posterior summaries are supplied upstream.",
        )
        report = tree.audit(_rows())
        tables = report.to_tables()
        exported_tables = us.report_tables(report)
        payload = json.loads(report.to_json())
        markdown = report.to_markdown()

        self.assertIsInstance(tree, us.ClaimTree)
        self.assertIsInstance(report, us.ClaimTreeAudit)
        self.assertTrue(report.failed)
        self.assertEqual(report.root.status, "pass")
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.pass_count, 1)
        self.assertEqual(report.fail_count, 1)
        self.assertEqual(report.leaf_count, 1)
        self.assertEqual(
            report.worst_nodes()[0].label, "Group-level posterior mean stable"
        )
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(tables["summary"][0]["status"], "fail")
        self.assertEqual(len(tables["nodes"]), 2)
        self.assertEqual(len(tables["edges"]), 1)
        self.assertEqual(len(exported_tables["nodes"]), 2)
        self.assertIn("Bayesian hierarchical", markdown)
        self.assertIn("Group-level posterior mean stable", markdown)

    def test_audit_claim_tree_accepts_mapping_payload(self):
        payload = {
            "name": "Mapping Tree",
            "root": {
                "claim": {
                    "estimate_name": "Root mapping claim",
                    "public": ["segment"],
                    "hidden": ["segment", "driver"],
                    "target": "target",
                    "ambiguity_limit": 1.0,
                },
                "children": [
                    {
                        "claim": {
                            "estimate_name": "Child mapping claim",
                            "public": ["segment"],
                            "hidden": ["segment", "driver"],
                            "target": "target",
                            "ambiguity_limit": 0.05,
                        },
                        "role": "subgroup",
                    }
                ],
            },
        }

        report = us.audit_claim_tree(_rows(), payload)

        self.assertIsInstance(report, us.ClaimTreeAudit)
        self.assertEqual(report.title, "Mapping Tree")
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.nodes[1].node.role, "subgroup")

    def test_audit_claim_passes_when_decision_is_invariant(self):
        claim = us.ClaimSpec(
            estimate_name="Decision-stable target",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            q_presets=["saturated"],
            decision=us.threshold_decision("<=", 0.9, label="target within limit"),
        )

        report = us.audit_claim(_rows(), claim)
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

    def test_audit_claim_fails_and_reports_decision_repair_when_threshold_crosses(
        self,
    ):
        claim = us.ClaimSpec(
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

        report = us.audit_claim(_rows(), claim)
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

    def test_audit_claim_supports_model_assisted_joint_draws(self):
        claim = us.ClaimSpec(
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

        report = us.audit_claim(
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

    def test_audit_claim_can_fit_model_assisted_joint_model_implicitly(self):
        claim = us.ClaimSpec(
            estimate_name="Implicit joint claim",
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            ambiguity_limit=0.05,
            min_cell_weight=0,
        )

        report = claim.audit(_rows(), joint_draws=3, joint_seed=99)

        self.assertIsNotNone(report.model_assisted)
        self.assertEqual(report.model_assisted.draw_count, 3)
        self.assertEqual(report.model_assisted.joint_model.cell_count, 3)


if __name__ == "__main__":
    unittest.main()
