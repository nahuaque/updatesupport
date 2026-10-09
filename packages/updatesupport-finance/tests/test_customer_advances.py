"""Reconstruct source controls and preserve unknown advance movements.

Public SEC filing regression inputs; no vendor payloads or live source retrieval:
https://www.sec.gov/Archives/edgar/data/1513845/000110465926094844/nbis-20260812xex99d2.htm
https://www.sec.gov/Archives/edgar/data/1839341/000183934126000014/core-20260630.htm
"""

import json
import unittest
from dataclasses import replace

import pytest
import updatesupport as us
import updatesupport_finance as f


check = unittest.TestCase()


def scope():
    return f.DisclosureScope(
        "Example",
        "customer advances",
        "USD million",
        "2026-06-30",
        "H1",
        "all customers",
        period_start="2026-01-01",
        currency="USD",
    )


def bridge(*, complete=True, movements=None, **kwargs):
    return f.CustomerAdvanceBridge(
        "advances",
        scope(),
        "opening",
        "closing",
        movements
        if movements is not None
        else (
            f.AdvanceMovement("cash", "net_cash", "Net cash received"),
            f.AdvanceMovement("recognized", "recognition", "Billed/earned recognition"),
            f.AdvanceMovement("other", "noncash", "Unresolved noncash movements"),
        ),
        "Reviewed regime",
        "Billed/earned amounts, separately from noncash reductions",
        complete,
        "Reviewed movement classes; unknown amounts remain unrestricted",
        **kwargs,
    )


def exact(name, value):
    return f.exact_disclosure_constraint(
        f"fact:{name}", name, value, category="reported_fact"
    )


def interval(problem, target, tier="reported"):
    return us.solve_named_linear_feasibility(problem).interval(
        target=target, scenario=tier
    )


def check_witnesses(result):
    problem = result.problem
    for row in result.intervals:
        for endpoint in (row.lower_endpoint, row.upper_endpoint):
            if endpoint.assignment is None:
                continue
            active = next(
                s.constraints for s in problem.scenarios if s.name == row.scenario
            )
            for c in problem.constraints:
                if c.name in active:
                    value = c.expression.evaluate(endpoint.assignment)
                    if c.lower is not None:
                        check.assertGreaterEqual(value, c.lower - 1e-6)
                    if c.upper is not None:
                        check.assertLessEqual(value, c.upper + 1e-6)
            for v in problem.variables:
                value = endpoint.assignment[v.name]
                if v.lower is not None:
                    check.assertGreaterEqual(value, v.lower - 1e-6)
                if v.upper is not None:
                    check.assertLessEqual(value, v.upper + 1e-6)


def broad_problem(*, include_closing_cap=True):
    b = bridge(
        movements=(
            f.AdvanceMovement(
                "cash", "net_cash", "Net advance cash, including old bills"
            ),
            f.AdvanceMovement(
                "recognized", "recognition", "All recognition from advance liabilities"
            ),
            f.AdvanceMovement(
                "unpaid_open", "unpaid_opening", "Unpaid advance billings at opening"
            ),
            f.AdvanceMovement(
                "unpaid_close", "unpaid_closing", "Unpaid advance billings at closing"
            ),
            f.AdvanceMovement(
                "other", "noncash", "All other noncash additions and reductions"
            ),
        ),
        cash_flow_adjustment="cf_adjustment",
    )
    constraints = [
        exact("opening", 1577.5),
        exact("closing", 5975.2),
        exact("cf_adjustment", 4395),
        exact("cfo", 4504.1),
        f.interval_disclosure_constraint(
            "recognition_bounds", "recognized", lower=154.3, upper=981.3
        ),
        f.interval_disclosure_constraint(
            "opening_gross_ar", "unpaid_open", upper=724.3
        ),
    ]
    if include_closing_cap:
        constraints.append(
            f.interval_disclosure_constraint(
                "closing_gross_ar", "unpaid_close", upper=292.7
            )
        )
    return b, b.problem(
        constraints=constraints,
        variables=[f.disclosure_variable("cfo", lower=None, unit=b.scope.unit)],
    )


@pytest.mark.parametrize(
    "opening,closing,recognition,noncash_reduction,expected",
    [
        (553.878, 652.288, 58.710, 24.029, 181.149),
        (1.973, 2.155, 1.972, None, 2.154),
    ],
)
def test_exhaustive_control_reconstructs_withheld_cash(
    opening, closing, recognition, noncash_reduction, expected
):
    movements = [
        f.AdvanceMovement("cash", "net_cash", "Net cash supplied in exhaustive table"),
        f.AdvanceMovement("recognized", "recognition", "Billed/earned recognition"),
    ]
    constraints = [
        exact("opening", opening),
        exact("closing", closing),
        exact("recognized", recognition),
    ]
    if noncash_reduction is not None:
        movements.append(
            f.AdvanceMovement(
                "straight_line", "noncash", "ASC 842 straight-line reduction", effect=-1
            )
        )
        constraints.append(exact("straight_line", noncash_reduction))
    b = bridge(movements=movements)
    problem = b.problem(constraints=constraints)
    report = us.solve_named_linear_feasibility(problem)
    row = report.interval(target=b.net_cash, scenario="reported")
    check.assertAlmostEqual(row.lower, expected)
    check.assertAlmostEqual(row.upper, expected)
    check_witnesses(report)


def test_rounding_is_retained_and_snapshot_replays_offline():
    b = bridge(
        movements=(
            f.AdvanceMovement("cash", "net_cash", "Net cash"),
            f.AdvanceMovement("recognized", "recognition", "Recognition"),
            f.AdvanceMovement(
                "straight_line", "noncash", "Noncash reduction", effect=-1
            ),
        )
    )
    facts = [
        f.DisclosureFact(
            name,
            "Example",
            name,
            value,
            "USD",
            end,
            "2026-07-28T00:00:00+00:00",
            "filing",
            "https://example.org/filing",
            period_start=start,
            currency="USD",
            scale=1e6,
            decimals=-3,
        )
        for name, value, start, end in (
            ("opening", 553.878, None, "2025-12-31"),
            ("closing", 652.288, None, "2026-06-30"),
            ("recognized", 58.710, "2026-01-01", "2026-06-30"),
            ("straight_line", 24.029, "2026-01-01", "2026-06-30"),
        )
    ]
    # Fact amounts must use the modeled numerical unit; retain source precision.
    normalized = [
        f.normalize_disclosure_fact(
            x, unit=b.scope.unit, source_units_per_solver_unit=1e6
        )
        for x in facts
    ]
    modeled = [x.fact for x in normalized]
    problem = b.problem(
        constraints=[
            f.disclosure_fact_constraint(f"fact:{x.fact_id}", x.fact_id, x)
            for x in modeled
        ]
    )
    snapshot = f.capture_disclosure_snapshot(
        problem,
        facts=modeled,
        target=b.net_cash,
        tier="reported",
        as_of="2026-08-01T00:00:00+00:00",
        context={
            **b.snapshot_context(),
            "normalizations": [x.as_dict() for x in normalized],
        },
    )
    restored = f.DisclosureSnapshot.from_json(snapshot.to_json())
    row = restored.replay().report.interval(target=b.net_cash, scenario="reported")
    check.assertAlmostEqual(row.lower, 181.147)
    check.assertAlmostEqual(row.upper, 181.151)
    check.assertEqual(restored.fingerprint, snapshot.fingerprint)
    check.assertEqual(
        restored.as_dict()["context"]["customer_advance_bridge"], b.as_dict()
    )


def test_unknown_noncash_leaves_cash_unbounded_and_cf_gap_is_only_diagnostic():
    b, problem = broad_problem()
    cash = interval(problem, b.net_cash)
    check.assertIsNone(cash.lower)
    check.assertIsNone(cash.upper)
    gap = interval(problem, b.cash_flow_gap)
    check.assertAlmostEqual(gap.lower, 2.7)
    check.assertAlmostEqual(gap.upper, 2.7)
    check.assertIsNone(interval(problem, b.noncash).upper)


@pytest.mark.parametrize(
    "fraction,expected", [(0.5, 2007.25), (0.8, 656.02), (0.9, 205.61), (1, -244.8)]
)
def test_minimum_alternative_explanation_is_joint_and_conditional(fraction, expected):
    b, problem = broad_problem()
    analysis = f.minimum_advance_explanation(
        b,
        problem,
        cash_ceiling={"cfo": fraction},
        ceiling_description="Hypothetical ceiling as a fraction of reported CFO",
    )
    row = analysis.report.interval(target=b.noncash, scenario="advance_explanation")
    check.assertAlmostEqual(row.lower, expected)
    check.assertIsNone(row.upper)
    check.assertEqual(analysis.support["advance_explanation"]["basis"], "conditional")
    check.assertIn("analyst_policy", analysis.support["advance_explanation"]["roles"])
    check.assertEqual(
        json.loads(json.dumps(analysis.as_dict()))["support"],
        analysis.as_dict()["support"],
    )
    check_witnesses(analysis.report)


def test_absent_ar_cap_leaves_minimum_explanation_unbounded():
    b, problem = broad_problem(include_closing_cap=False)
    analysis = f.minimum_advance_explanation(
        b, problem, cash_ceiling=3603.28, ceiling_description="80% of fixed CFO"
    )
    check.assertIsNone(
        analysis.report.interval(target=b.noncash, scenario="advance_explanation").lower
    )


def test_joint_source_precision_and_cohort_recognition_reproduce_conservative_requirement():
    b, _ = broad_problem()
    b = replace(
        b,
        movements=[m for m in b.movements if m.name != "recognized"]
        + [
            f.AdvanceMovement(
                "opening_recognition",
                "recognition",
                "Reported opening-cohort recognition",
            ),
            f.AdvanceMovement(
                "current_recognition",
                "recognition",
                "Unknown recognition from current-period additions",
            ),
        ],
    )
    values = {
        "opening": (1577.5, "2025-12-31", None),
        "closing": (5975.2, "2026-06-30", None),
        "opening_recognition": (154.3, "2026-06-30", "2026-01-01"),
        "revenue": (981.3, "2026-06-30", "2026-01-01"),
        "ar_net": (288.6, "2026-06-30", None),
        "allowance": (4.1, "2026-06-30", None),
        "cfo": (4504.1, "2026-06-30", "2026-01-01"),
    }
    normalized = [
        f.normalize_disclosure_fact(
            f.DisclosureFact(
                n,
                "Example",
                n,
                value,
                "USD",
                end,
                "2026-08-12T00:00:00+00:00",
                "filing",
                "https://example.org/filing",
                period_start=start,
                currency="USD",
                scale=1e6,
                decimals=-5,
            ),
            unit=b.scope.unit,
            source_units_per_solver_unit=1e6,
        ).fact
        for n, (value, end, start) in values.items()
    ]
    relationships = [
        f.interval_disclosure_constraint(
            "recognition_cap",
            {b.recognition: 1, "revenue": -1},
            upper=0,
            category="reconciliation",
            description="Reviewed recognition is included in total revenue",
        ),
        f.interval_disclosure_constraint(
            "gross_ar_cap",
            {"unpaid_close": 1, "ar_net": -1, "allowance": -1},
            upper=0,
            category="reconciliation",
            description="Reviewed unpaid bills are contained in gross AR",
        ),
    ]
    problem = b.problem(
        constraints=[
            *[
                f.disclosure_fact_constraint(f"fact:{x.fact_id}", x.fact_id, x)
                for x in normalized
            ],
            *relationships,
        ],
        variables=[
            f.disclosure_variable(n, lower=None, unit=b.scope.unit)
            for n in ("revenue", "ar_net", "allowance", "cfo")
        ],
    )
    analysis = f.minimum_advance_explanation(
        b,
        problem,
        cash_ceiling={"cfo": 0.8},
        ceiling_description="Alternative at most 80% of CFO",
    )
    check.assertAlmostEqual(
        analysis.report.interval(
            target=b.noncash, scenario="advance_explanation"
        ).lower,
        655.730,
    )
    check_witnesses(analysis.report)


def test_amount_and_constant_expression_ceilings_agree():
    b, problem = broad_problem()
    for ceiling in (3703.28, us.NamedLinearExpression({"cfo": 0.8}, constant=100)):
        result = f.minimum_advance_explanation(
            b,
            problem,
            cash_ceiling=ceiling,
            ceiling_description="Alternative cash amount",
        )
        check.assertAlmostEqual(
            result.report.interval(
                target=b.noncash, scenario="advance_explanation"
            ).lower,
            556.02,
        )


def test_incomplete_mapping_keeps_signed_unclassified_residual():
    b = bridge(
        complete=False,
        movements=[f.AdvanceMovement("cash", "net_cash", "Supplied cash movement")],
    )
    problem = b.problem(
        constraints=[exact("opening", 10), exact("closing", 5), exact("cash", 0)]
    )
    check.assertEqual(interval(problem, b.unclassified).lower, -5)
    check.assertEqual(interval(problem, b.noncash).lower, 0)
    with pytest.raises(ValueError, match="complete mapping"):
        f.minimum_advance_explanation(
            b, problem, cash_ceiling=0, ceiling_description="Example"
        )


def test_gross_cash_refunds_and_unpaid_change_have_correct_signs():
    b = bridge(
        movements=[
            f.AdvanceMovement("receipts", "cash_receipt", "Gross receipts"),
            f.AdvanceMovement("refunds", "cash_refund", "Gross refunds"),
            f.AdvanceMovement("unpaid", "unpaid_change", "Signed unpaid change"),
        ]
    )
    problem = b.problem(
        constraints=[
            exact(n, v)
            for n, v in {
                "opening": 10,
                "closing": 6,
                "receipts": 5,
                "refunds": 7,
                "unpaid": -2,
            }.items()
        ]
    )
    check.assertEqual(interval(problem, b.net_cash).lower, -2)


def test_structural_constraints_survive_custom_tiers_and_cannot_be_released():
    b = bridge()
    policy = f.ConstraintPolicy(
        "noncash_cap",
        "analyst_policy",
        [f.interval_disclosure_constraint("cap", "other", upper=0)],
    )
    constraints = [
        exact("opening", 10),
        exact("closing", 20),
        exact("recognized", 0),
        *policy.constraints,
    ]
    problem = b.problem(
        constraints=constraints,
        tiers=[us.NamedLinearScenario("custom", [c.name for c in constraints])],
    )
    check.assertTrue(
        {c.name for c in b.constraints} <= set(problem.scenarios[0].constraints)
    )
    analysis = f.minimum_advance_explanation(
        b,
        problem,
        tier="custom",
        cash_ceiling=5,
        ceiling_description="Alternative cash limit",
        release_constraints=["cap"],
    )
    check.assertEqual(
        analysis.report.interval(
            target=b.noncash, scenario="advance_explanation"
        ).lower,
        5,
    )
    with pytest.raises(ValueError, match="only release"):
        f.minimum_advance_explanation(
            b,
            problem,
            tier="custom",
            cash_ceiling=5,
            ceiling_description="Alternative",
            release_constraints=[b.constraints[-1].name],
        )
    infeasible = f.minimum_advance_explanation(
        b, problem, tier="custom", cash_ceiling=5, ceiling_description="Alternative"
    )
    check.assertEqual(
        infeasible.report.interval(
            target=b.noncash, scenario="advance_explanation"
        ).status,
        "infeasible",
    )
    check.assertIn("advance_explanation", infeasible.conflicts)


@pytest.mark.parametrize(
    "kinds",
    [
        ("net_cash", "cash_receipt"),
        ("net_cash", "cash_refund"),
        ("net_cash", "unpaid_opening"),
        ("net_cash", "unpaid_change", "unpaid_opening", "unpaid_closing"),
        ("recognition",),
    ],
)
def test_ambiguous_or_missing_movements_are_rejected(kinds):
    with pytest.raises(ValueError):
        bridge(
            movements=[f.AdvanceMovement(f"v{i}", k, k) for i, k in enumerate(kinds)]
        )


def test_scope_names_and_threshold_guards():
    b, problem = broad_problem()
    for changes in (
        {"complete": None},
        {"scope": replace(scope(), currency=None)},
        {"scope": replace(scope(), window_kind="lifetime")},
        {"scope": replace(scope(), period_start=None)},
        {"opening": "closing"},
        {"closing": b.net_cash},
        {"completeness_basis": ""},
    ):
        with pytest.raises(ValueError):
            replace(b, **changes)
    with pytest.raises(ValueError):
        f.AdvanceMovement("x", "recognition", "Description", effect=-1)
    for ceiling in (float("inf"), {"unknown": 1}, b.net_cash):
        with pytest.raises(ValueError):
            f.minimum_advance_explanation(
                b, problem, cash_ceiling=ceiling, ceiling_description="Alternative"
            )
    with pytest.raises(ValueError, match="numerical unit"):
        b.problem(variables=[us.NamedLinearVariable("x", unit="EUR")])
    missing_identity = replace(
        problem,
        scenarios=[
            us.NamedLinearScenario(
                "reported",
                [
                    c.name
                    for c in problem.constraints
                    if c.name != b.constraints[-1].name
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="retain the bridge"):
        f.minimum_advance_explanation(
            b, missing_identity, cash_ceiling=0, ceiling_description="Alternative"
        )
    changed_variable = replace(
        problem,
        variables=[
            replace(v, lower=0) if v.name == "other" else v for v in problem.variables
        ],
    )
    with pytest.raises(ValueError, match="retain the bridge"):
        f.minimum_advance_explanation(
            b, changed_variable, cash_ceiling=0, ceiling_description="Alternative"
        )
