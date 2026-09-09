import unittest
from dataclasses import replace
import json
from unittest.mock import patch
import pytest
import updatesupport as us


def _rows():
    return [
        {"public": "all", "hidden": "kept", "target": 0.0, "weight": 10.0},
        *[
            {"public": "all", "hidden": f"sparse-{i}", "target": 1.0, "weight": 1.0}
            for i in range(100)
        ],
    ]


def _claim(**overrides):
    return us.claim(
        "Stable aggregate",
        public=["public"],
        hidden=["public", "hidden"],
        target="target",
        weight="weight",
        min_cell_weight=10,
        ambiguity_limit=0.01,
        **overrides,
    )


class ClaimCoverageTests(unittest.TestCase):
    def test_coverage_is_opt_in_and_visible_beside_verdict(self):
        legacy = _claim().audit(_rows())
        self.assertTrue(legacy.passed)
        guarded = _claim(max_dropped_weight_share=0.05).audit(_rows())
        self.assertTrue(guarded.inconclusive)
        self.assertEqual(guarded.interval, legacy.interval)
        self.assertAlmostEqual(guarded.coverage["dropped_weight_share"], 100 / 110)
        self.assertAlmostEqual(guarded.coverage["retained_weight_share"], 10 / 110)
        self.assertIs(guarded.coverage["coverage_requirement_met"], False)
        self.assertIn("Retained input weight: 9.1%", guarded.to_markdown())
        self.assertIn("exceeding the declared 5.0% limit", guarded.reasons[0])
        self.assertIs(
            guarded.to_tables()["summary"][0]["coverage_requirement_met"], False
        )
        self.assertEqual(
            json.loads(guarded.to_json())["coverage"]["max_dropped_weight_share"], 0.05
        )

    def test_invalid_coverage_limit_is_rejected(self):
        for limit in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(limit=limit):
                with pytest.raises(ValueError, match="max_dropped_weight_share"):
                    _claim(max_dropped_weight_share=limit)

    def test_limit_is_inclusive_and_claim_spec_round_trips(self):
        claim = _claim(max_dropped_weight_share=100 / 110)
        restored = us.ClaimSpec.from_dict(json.loads(json.dumps(claim.as_dict())))
        self.assertTrue(restored.audit(_rows()).passed)
        self.assertTrue(
            replace(
                claim, min_cell_weight=0, ambiguity_limit=1, max_dropped_weight_share=0
            )
            .audit(_rows())
            .passed
        )

    def test_bad_coverage_takes_precedence_over_threshold_failure(self):
        claim = _claim(
            max_dropped_weight_share=0.05, decision=us.threshold_decision(">=", 0.5)
        )
        rows = [
            *_rows(),
            {"public": "all", "hidden": "also-kept", "target": 1.0, "weight": 10},
        ]
        audit = claim.audit(rows)
        self.assertFalse(audit.decision.invariant)
        self.assertTrue(audit.inconclusive)

    def test_endpoint_screen_cannot_bypass_coverage_grid(self):
        claim = replace(
            _claim(max_dropped_weight_share=0.05, screening_backend="residopt"),
            min_cell_weight=0,
            min_cell_weights=[0, 10],
            ambiguity_limit=1,
        )
        with patch(
            "updatesupport.claim._try_claim_screen",
            side_effect=AssertionError("coverage bypass"),
        ):
            self.assertTrue(claim.audit(_rows()).inconclusive)

    def test_refinement_does_not_repair_discarded_population(self):
        claim = _claim(max_dropped_weight_share=0.05, candidate_refinements=["hidden"])
        design = claim.design(_rows())
        self.assertEqual(design.status, "inconclusive")
        self.assertIs(design.recommended_public, None)
        self.assertIs(design.audit.repair_candidate, None)
        self.assertFalse(
            any((row.certifies_claim for row in design.repair_plan.options))
        )

    def test_stress_grid_coverage_is_checked_as_well_as_primary(self):
        claim = replace(
            _claim(max_dropped_weight_share=0.05),
            min_cell_weight=0,
            min_cell_weights=[0, 10],
            ambiguity_limit=1,
        )
        audit = claim.audit(_rows())
        self.assertTrue(audit.inconclusive)
        self.assertEqual(audit.coverage["dropped_weight_share"], 0)
        self.assertAlmostEqual(audit.coverage["worst_dropped_weight_share"], 100 / 110)

    def test_missing_compiled_diagnostics_cannot_certify_coverage(self):
        grouped = us.from_dataframe(
            _rows(),
            public=["public"],
            hidden=["public", "hidden"],
            target="target",
            weight="weight",
        )
        claim = replace(
            _claim(max_dropped_weight_share=1),
            ambiguity_limit=None,
            decision=us.threshold_decision(">=", -1),
        )
        audit = claim.audit(replace(grouped, diagnostics=None))
        self.assertTrue(audit.inconclusive)
        self.assertIs(audit.coverage["coverage_requirement_met"], None)

    def test_shared_design_respects_primary_coverage_when_grid_omits_it(self):
        claim = replace(
            _claim(max_dropped_weight_share=0.05),
            min_cell_weights=[0],
            ambiguity_limit=1,
        )
        design = us.claim_portfolio(
            claim, replace(claim, estimate_name="Second claim")
        ).design(_rows())
        self.assertFalse(design.selected.all_claims_certified)
        self.assertIs(design.selected.claim_results[0].coverage_requirement_met, False)

    def test_rollup_cannot_certify_insufficient_coverage(self):
        rows = [
            {"public": "all", "hidden": "a", "target": 0.0, "weight": 10},
            {"public": "all", "hidden": "b", "target": 0.0, "weight": 10},
            {"public": "all", "hidden": "c", "target": 0.0, "weight": 9},
        ]
        design = _claim(max_dropped_weight_share=0.05).design_categorical_rollup(
            rows, column="hidden"
        )
        self.assertFalse(design.selected.certifies_claim)
        self.assertIs(design.selected.coverage_requirement_met, False)
