from __future__ import annotations

import json
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for channel in ("direct", "partner"):
        for tenure in ("new", "established"):
            rows.append(
                {
                    "segment": "all",
                    "channel": channel,
                    "tenure": tenure,
                    "channel_metric": 0.0 if channel == "direct" else 1.0,
                    "tenure_metric": 0.0 if tenure == "new" else 1.0,
                    "weight": 25.0,
                }
            )
    return rows


def _claims() -> tuple[us.ClaimSpec, us.ClaimSpec]:
    common = {
        "public": ["segment"],
        "hidden": ["segment", "channel", "tenure"],
        "weight": "weight",
        "candidate_refinements": ["channel", "tenure"],
        "ambiguity_limit": 0.01,
    }
    return (
        us.claim(
            "Channel-sensitive metric",
            target="channel_metric",
            **common,
        ),
        us.claim(
            "Tenure-sensitive metric",
            target="tenure_metric",
            **common,
        ),
    )


class SharedRepresentationDesignTests(unittest.TestCase):
    def test_exact_search_finds_representation_certifying_every_claim(self):
        channel_claim, tenure_claim = _claims()
        portfolio = us.claim_portfolio(
            channel_claim,
            tenure_claim,
            name="Shared KPI report",
            candidate_refinements=["channel", "tenure"],
        )

        design = portfolio.design(_rows(), max_added_columns=2)

        self.assertIsInstance(portfolio, us.ClaimPortfolio)
        self.assertIsInstance(design, us.SharedRepresentationDesign)
        self.assertIn("claim_portfolio", us.__all__)
        self.assertIn("design_shared_representation", us.__all__)
        self.assertEqual(design.status, "shared_representation_found")
        self.assertEqual(design.evaluated_representation_count, 4)
        self.assertEqual(design.selected.added_columns, ("channel", "tenure"))
        self.assertEqual(
            design.recommended_public,
            ("segment", "channel", "tenure"),
        )
        self.assertEqual(design.selected.max_public_cells, 4)
        self.assertTrue(design.selected.all_claims_certified)
        self.assertEqual(design.selected.certified_claim_count, 2)
        self.assertTrue(
            all(row.certifies_claim for row in design.selected.claim_results)
        )

        audits = design.audit(_rows())
        self.assertEqual(len(audits), 2)
        self.assertTrue(all(audit.passed for audit in audits))
        self.assertTrue(
            all(
                audit.claim.public == ("segment", "channel", "tenure")
                for audit in audits
            )
        )

        tables = design.to_tables()
        payload = json.loads(design.to_json())
        markdown = design.to_markdown()
        self.assertIn("candidate_claims", tables)
        self.assertIn("selected_scenarios", tables)
        self.assertEqual(payload["selected"]["certified_claim_count"], 2)
        self.assertIn("Selected Claim Outcomes", markdown)
        self.assertIn("2/2", markdown)

    def test_one_column_budget_returns_explicit_best_effort_design(self):
        portfolio = us.claim_portfolio(
            _claims(),
            candidate_refinements=["channel", "tenure"],
        )

        design = us.design_shared_representation(
            _rows(),
            portfolio,
            max_added_columns=1,
        )

        self.assertEqual(design.status, "no_shared_representation")
        self.assertEqual(design.selected.added_columns, ("channel",))
        self.assertEqual(design.selected.certified_claim_count, 1)
        self.assertFalse(design.selected.all_claims_certified)

    def test_public_cell_budget_is_applied_to_shared_design(self):
        design = us.claim_portfolio(
            _claims(),
            candidate_refinements=["channel", "tenure"],
        ).design(
            _rows(),
            max_added_columns=2,
            bucket_budget=2,
        )

        self.assertEqual(design.status, "no_shared_representation")
        self.assertLessEqual(design.selected.max_public_cells, 2)
        self.assertEqual(design.selected.certified_claim_count, 1)
        full = next(row for row in design.candidates if len(row.added_columns) == 2)
        self.assertTrue(full.all_claims_certified)
        self.assertFalse(full.passes_bucket_budget)

    def test_decision_only_claim_participates_in_shared_certificate(self):
        channel_claim, _tenure_claim = _claims()
        decision_claim = us.claim(
            "Tenure decision remains positive",
            public=["segment"],
            hidden=["segment", "channel", "tenure"],
            target="tenure_metric",
            weight="weight",
            candidate_refinements=["channel", "tenure"],
            decision=us.threshold_decision(">=", 0.25),
        )

        design = us.claim_portfolio(
            channel_claim,
            decision_claim,
            candidate_refinements=["channel", "tenure"],
        ).design(_rows())

        self.assertEqual(design.selected.added_columns, ("channel", "tenure"))
        decision_result = design.selected.claim_results[1]
        self.assertIsNone(decision_result.ambiguity_limit)
        self.assertTrue(decision_result.decision_certified)
        self.assertTrue(decision_result.certifies_claim)

    def test_each_claim_keeps_its_own_stress_scenario_grid(self):
        _channel_claim, tenure_claim = _claims()
        channel_claim = us.claim(
            "Channel-sensitive metric",
            public=["segment"],
            hidden=["segment", "channel", "tenure"],
            target="channel_metric",
            weight="weight",
            ambiguity_limit=0.01,
            candidate_refinements=["channel", "tenure"],
            q_presets=["observed", "saturated"],
        )

        design = us.claim_portfolio(channel_claim, tenure_claim).design(_rows())

        first_result = design.selected.claim_results[0]
        self.assertEqual(len(first_result.scenarios), 2)
        self.assertEqual(
            {row.q_name for row in first_result.scenarios},
            {"observed", "saturated"},
        )
        self.assertTrue(first_result.certifies_claim)

    def test_zero_bucket_budget_reports_no_feasible_candidate(self):
        design = us.claim_portfolio(
            _claims(),
            candidate_refinements=["channel", "tenure"],
        ).design(
            _rows(),
            bucket_budget=0,
        )

        self.assertEqual(design.status, "no_feasible_candidate")
        self.assertEqual(design.selected.added_columns, ())
        self.assertFalse(design.selected.eligible)

    def test_portfolio_rejects_claim_without_certification_requirement(self):
        first, _second = _claims()
        no_requirement = us.claim(
            "Descriptive claim",
            public=["segment"],
            hidden=["segment", "channel", "tenure"],
            target="tenure_metric",
            weight="weight",
            candidate_refinements=["channel", "tenure"],
        )

        with self.assertRaisesRegex(ValueError, "ambiguity_limit or decision"):
            us.claim_portfolio(first, no_requirement).design(_rows())

    def test_portfolio_rejects_mismatched_public_contract(self):
        first, _second = _claims()
        mismatched = us.claim(
            "Different public schema",
            public=["segment", "channel"],
            hidden=["segment", "channel", "tenure"],
            target="tenure_metric",
            weight="weight",
            ambiguity_limit=0.01,
        )

        with self.assertRaisesRegex(ValueError, "same public columns"):
            us.claim_portfolio(first, mismatched).design(_rows())

    def test_candidate_must_exist_in_every_hidden_set_scenario(self):
        _first, second = _claims()
        restricted = us.claim(
            "Restricted hidden scenario",
            public=["segment"],
            hidden=["segment", "channel", "tenure"],
            hidden_sets=[["segment", "channel"]],
            target="channel_metric",
            weight="weight",
            ambiguity_limit=0.01,
        )

        with self.assertRaisesRegex(ValueError, "every claim hidden-set"):
            us.claim_portfolio(
                restricted,
                second,
                candidate_refinements=["tenure"],
            ).design(_rows())

    def test_exact_search_guard_rejects_large_candidate_space(self):
        portfolio = us.claim_portfolio(
            _claims(),
            candidate_refinements=["channel", "tenure"],
        )

        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            portfolio.design(
                _rows(),
                max_added_columns=2,
                max_evaluations=3,
            )


if __name__ == "__main__":
    unittest.main()
