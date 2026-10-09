"""Economic alternatives, mapping dependence and non-vacuous evidence plans.

Selected public CoreWeave regression amounts from the Q1/H1 2026 SEC filings:
https://www.sec.gov/Archives/edgar/data/1769628/000176962826000222/crwv-20260331.htm
https://www.sec.gov/Archives/edgar/data/1769628/000176962826000366/crwv-20260630.htm
They exercise a partial model; the tests do not identify the company's cause.
"""

from dataclasses import replace
import json
import unittest

import pytest
import updatesupport as us
import updatesupport_finance as f

check = unittest.TestCase()


def row(name, terms, lower=None, upper=None, kind="analyst_policy"):
    return us.NamedLinearConstraint(name, terms, lower=lower, upper=upper, kind=kind)


def zero(name):
    return row(name, name, 0, 0)


def fixture(*, nominal=False, bounded_other=False):
    variables = [
        us.NamedLinearVariable(
            n,
            lower=0 if n in {"exit", "refund", "transfer"} else None,
            unit="USD million",
        )
        for n in [
            "exit",
            "refund",
            "transfer",
            "residual",
            "delta",
            "cf",
            "operating",
            "other",
        ]
    ]
    constraints = [
        row(
            "exit_amount",
            "exit",
            1469 if nominal else 1442.5,
            1469 if nominal else 1469.5,
            "reported_fact",
        ),
        row(
            "dr_delta",
            "delta",
            2169 if nominal else 2167,
            2169 if nominal else 2171,
            "reported_fact",
        ),
        row(
            "cf_amount",
            "cf",
            790 if nominal else 789,
            790 if nominal else 791,
            "reported_fact",
        ),
        row(
            "caption_bridge",
            {"exit": 1, "refund": -1, "transfer": -1, "residual": -1},
            0,
            0,
            "reconciliation",
        ),
        row(
            "dr_bridge",
            {"delta": 1, "operating": -1, "transfer": -1, "other": -1},
            0,
            0,
            "reconciliation",
        ),
    ]
    if bounded_other:
        constraints.append(row("selected_small_other", "other", -100, 100))
    problem = us.NamedLinearFeasibilityProblem(
        variables,
        constraints,
        [
            us.NamedLinearTarget(n, n, unit="USD million")
            for n in ["exit", "refund", "transfer", "other"]
        ],
        [us.NamedLinearScenario("reported", [c.name for c in constraints])],
    )
    mappings = [
        f.DisclosureAlternative(
            "dr_only",
            "CF line covers DR operating movement only",
            [row("scope", {"operating": 1, "cf": -1}, 0, 0)],
        ),
        f.DisclosureAlternative(
            "joint",
            "CF line includes customer-caption refunds",
            [row("scope", {"operating": 1, "refund": -1, "cf": -1}, 0, 0)],
        ),
    ]
    explanations = [
        f.DisclosureAlternative(
            "transfer",
            "Selected entire exit transfers to DR",
            [zero("refund"), zero("residual")],
        ),
        f.DisclosureAlternative(
            "refund",
            "Selected entire exit is refunded",
            [zero("transfer"), zero("residual")],
        ),
    ]
    return problem, mappings, explanations


def compare(**kwargs):
    fixture_args = {
        k: kwargs.pop(k) for k in ["nominal", "bounded_other"] if k in kwargs
    }
    p, m, e = fixture(**fixture_args)
    return f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=m, explanations=e, **kwargs
    )


def test_explanation_module_preserves_public_imports_and_planning_workflow():
    from updatesupport_finance import explanations as api

    for name in api.__all__:
        check.assertIs(getattr(api, name), getattr(f, name))
    p, mappings, explanations = fixture()
    comparison = api.compare_disclosure_explanations(
        p, base_tier="reported", mappings=mappings, explanations=explanations
    )
    request = api.DisclosureEvidenceRequest(
        "cap", "Hypothetical refund cap", [row("cap", "refund", upper=100)]
    )
    outcome = api.evaluate_disclosure_evidence(comparison, [request])
    plan = api.plan_disclosure_evidence(comparison, [request], eliminate=["refund"])
    check.assertEqual(outcome.status, "compatible")
    check.assertEqual([b.requests for b in plan.minimal_bundles], [("cap",)])
    check.assertEqual(
        outcome.comparison.summary(), plan.minimal_bundles[0].comparison.summary()
    )


def test_same_data_survives_opposing_explanations_and_scope_changes_residual():
    c = compare()
    check.assertTrue(all(case.status == "feasible" for case in c.cases))
    check.assertEqual(
        c.case("dr_only", "transfer").intervals["other"].upper, pytest.approx(-60.5)
    )
    check.assertEqual(
        c.case("dr_only", "refund").intervals["other"].lower, pytest.approx(1376)
    )
    for explanation in ["transfer", "refund"]:
        result = c.case("joint", explanation).intervals["other"]
        check.assertEqual((result.lower, result.upper), pytest.approx((-93.5, -60.5)))
    check.assertTrue(
        all(row["status"] == "possible_in_all_feasible_mappings" for row in c.summary())
    )
    for case in c.cases:
        check.assertTrue(
            all(
                x.passed
                for x in us.check_named_linear_assignment(
                    c.report.problem, case.witness, scenario=case.tier
                )
            )
        )
        a = case.witness
        check.assertEqual(
            a["exit"], pytest.approx(a["refund"] + a["transfer"] + a["residual"])
        )
        check.assertEqual(
            a["delta"], pytest.approx(a["operating"] + a["transfer"] + a["other"])
        )


def test_mapping_dependent_result_and_conflict_are_explicit():
    c = compare(bounded_other=True)
    check.assertEqual(c.case("dr_only", "refund").status, "infeasible")
    check.assertEqual(c.case("dr_only", "refund").conflict.status, "irreducible")
    check.assertEqual(c.case("joint", "refund").status, "feasible")
    summaries = {r["explanation"]: r for r in c.summary()}
    check.assertEqual(summaries["refund"]["status"], "mapping_dependent")
    check.assertEqual(summaries["refund"]["possible_mappings"], ["joint"])
    check.assertEqual(
        summaries["transfer"]["status"], "possible_in_all_feasible_mappings"
    )


def test_paired_gross_cash_histories_have_same_net_and_reported_totals():
    p, m, e = fixture(nominal=True)
    extra = [
        row("chosen_recognition", "recognition", 1000, 1000),
        row("chosen_other", "other", -90, -90),
        row(
            "cash_identity",
            {"operating": 1, "receipts": -1, "recognition": 1},
            0,
            0,
            "reconciliation",
        ),
    ]
    p = replace(
        p,
        variables=(
            *p.variables,
            us.NamedLinearVariable("receipts", lower=0),
            us.NamedLinearVariable("recognition", lower=0),
        ),
        constraints=(*p.constraints, *extra),
        targets=(
            *p.targets,
            us.NamedLinearTarget("net_cash", {"receipts": 1, "refund": -1}),
        ),
        scenarios=[
            us.NamedLinearScenario(
                "reported", [c.name for c in (*p.constraints, *extra)]
            )
        ],
    )
    result = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=[m[1]], explanations=e
    )
    a, b = [result.case("joint", name).witness for name in ["transfer", "refund"]]
    check.assertEqual(
        (a["receipts"], b["receipts"], b["refund"]), pytest.approx((1790, 3259, 1469))
    )
    check.assertTrue(
        a["receipts"] - a["refund"]
        == b["receipts"] - b["refund"]
        == pytest.approx(1790)
    )
    check.assertTrue(a["delta"] == b["delta"] == 2169)
    check.assertTrue(a["cf"] == b["cf"] == 790)


def test_evidence_bundles_require_amount_and_scope_and_find_direct_alternative():
    c = compare()
    requests = [
        f.DisclosureEvidenceRequest(
            "other_cap",
            "Hypothetical reviewed net other DR effects <=100",
            [row("cap", "other", upper=100)],
        ),
        f.DisclosureEvidenceRequest(
            "dr_scope",
            "Hypothetical confirmation of DR-only CF coverage",
            mapping_names=["dr_only"],
        ),
        f.DisclosureEvidenceRequest(
            "refund_cap",
            "Hypothetical gross refunds <=100",
            [row("cap", "refund", upper=100)],
        ),
    ]
    plan = f.plan_disclosure_evidence(c, requests, eliminate=["refund"])
    check.assertEqual(
        {b.requests for b in plan.minimal_bundles},
        {
            ("other_cap", "dr_scope"),
            ("refund_cap",),
        },
    )
    lookup = {b.requests: b for b in plan.bundles}
    check.assertEqual(lookup[()].status, "not_separated")
    check.assertFalse(lookup[("other_cap",)].sufficient)
    check.assertFalse(lookup[("dr_scope",)].sufficient)
    check.assertEqual(lookup[("other_cap", "dr_scope")].retained_mappings, ("dr_only",))
    check.assertEqual(
        {c.mapping for c in lookup[("other_cap", "dr_scope")].comparison.cases},
        {"dr_only"},
    )
    check.assertTrue(
        all(r.as_dict()["evidence_role"] == "hypothetical_outcome" for r in requests)
    )


def test_request_dependencies_and_contradictory_mapping_reviews():
    requests = [
        f.DisclosureEvidenceRequest(
            "cap",
            "Hypothetical bounded refund, only after scope review",
            [row("cap", "refund", upper=100)],
            requires=["scope"],
        ),
        f.DisclosureEvidenceRequest(
            "scope", "Retain DR-only mapping", mapping_names=["dr_only"]
        ),
        f.DisclosureEvidenceRequest(
            "joint_scope", "Retain joint mapping", mapping_names=["joint"]
        ),
    ]
    plan = f.plan_disclosure_evidence(compare(), requests, eliminate=["refund"])
    by = {b.requests: b for b in plan.bundles}
    check.assertEqual(by[("cap",)].status, "missing_dependencies")
    check.assertEqual(by[("scope", "joint_scope")].status, "no_retained_mappings")
    check.assertFalse(by[("scope", "joint_scope")].sufficient)
    check.assertEqual([b.requests for b in plan.minimal_bundles], [("cap", "scope")])


def test_inconsistent_evidence_is_not_a_successful_separator():
    request = f.DisclosureEvidenceRequest(
        "bad",
        "Hypothetical incompatible exit amount",
        [row("wrong_exit", "exit", upper=10)],
    )
    plan = f.plan_disclosure_evidence(compare(), [request], eliminate=["refund"])
    bundle = plan.bundles[1]
    check.assertEqual(bundle.status, "inconsistent_with_all_mappings")
    check.assertTrue(not bundle.sufficient and not plan.minimal_bundles)


def test_uncovered_mapping_is_not_an_actual_explanation_or_separator():
    p, m, e = fixture()
    c = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=m, explanations=e[1:]
    )
    plan = f.plan_disclosure_evidence(
        c,
        [
            f.DisclosureEvidenceRequest(
                "cap",
                "Refunds bounded below caption exit",
                [row("cap", "refund", upper=100)],
            )
        ],
        eliminate=["refund"],
    )
    check.assertEqual(plan.bundles[1].status, "uncovered_mapping")
    check.assertFalse(plan.bundles[1].sufficient)


def test_inconsistent_source_and_mapping_do_not_create_vacuous_robustness():
    p, m, e = fixture()
    bad = row("bad", "exit", upper=10)
    p = replace(
        p,
        constraints=(*p.constraints, bad),
        scenarios=[
            us.NamedLinearScenario("reported", [c.name for c in (*p.constraints, bad)])
        ],
    )
    c = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=m, explanations=e
    )
    check.assertEqual(c.source.status, "infeasible")
    check.assertTrue(all(s["status"] == "undetermined" for s in c.summary()))
    check.assertEqual(
        f.plan_disclosure_evidence(c, [], eliminate=["refund"]).bundles[0].status,
        "inconsistent_with_all_mappings",
    )


def test_all_unbounded_targets_still_have_checked_compatibility_witnesses():
    p = us.NamedLinearFeasibilityProblem(
        [us.NamedLinearVariable("x")],
        [],
        [us.NamedLinearTarget("x", "x")],
        [us.NamedLinearScenario("reported", [])],
    )
    c = f.compare_disclosure_explanations(
        p,
        base_tier="reported",
        mappings=[f.DisclosureAlternative("open", "Open mapping")],
        explanations=[f.DisclosureAlternative("open", "Open explanation")],
    )
    check.assertEqual(c.cases[0].status, "feasible")
    check.assertEqual(c.cases[0].intervals["x"].status, "unbounded")
    check.assertIsNot(c.cases[0].witness, None)
    check.assertEqual(c.summary()[0]["status"], "possible_in_all_feasible_mappings")


def test_numerical_failure_remains_undetermined_and_never_separates(monkeypatch):
    solve = us.solve_named_linear_feasibility

    def failed(p):
        result = solve(p)
        return replace(
            result,
            intervals=tuple(
                replace(
                    i,
                    status="numerical_error",
                    lower=None,
                    lower_endpoint=replace(
                        i.lower_endpoint, status="numerical_error", assignment=None
                    ),
                )
                for i in result.intervals
            ),
        )

    monkeypatch.setattr(us, "solve_named_linear_feasibility", failed)
    c = compare(include_conflicts=False)
    check.assertTrue(
        all(case.status == "undetermined" and case.witness is None for case in c.cases)
    )
    check.assertTrue(all(row["status"] == "undetermined" for row in c.summary()))
    check.assertEqual(
        f.plan_disclosure_evidence(c, [], eliminate=["refund"]).bundles[0].status,
        "undetermined",
    )


def test_base_history_and_constraint_collisions_are_preserved_and_snapshot_replays():
    p, mappings, explanations = fixture()
    collided = replace(p.constraints[0], name="mapping:0:scope")
    p = replace(
        p,
        constraints=(collided, *p.constraints[1:]),
        scenarios=[
            us.NamedLinearScenario(
                "reported", [collided.name, *p.scenarios[0].constraints[1:]]
            )
        ],
    )
    original = p.as_dict()
    c = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=mappings, explanations=explanations
    )
    check.assertEqual(p.as_dict(), original)
    check.assertEqual(c.report.problem.constraints[: len(p.constraints)], p.constraints)
    check.assertEqual(
        len({x.name for x in c.report.problem.constraints}),
        len(c.report.problem.constraints),
    )
    selected = c.case("joint", "refund")
    snap = f.capture_disclosure_snapshot(
        c.report.problem,
        facts=[],
        as_of="2026-10-07T00:00:00Z",
        target="refund",
        tier=selected.tier,
        context=c.snapshot_context(),
    )
    restored = f.DisclosureSnapshot.from_json(snap.to_json())
    check.assertEqual(restored.fingerprint, snap.fingerprint)
    check.assertEqual(
        json.loads(json.dumps(restored.replay().as_dict())), snap.as_dict()["result"]
    )
    check.assertTrue(json.loads(c.to_json())["summary"])
    check.assertTrue(c.to_tables()["explanation_cases"])
    check.assertIn("mapping", c.to_markdown().lower())


@pytest.mark.parametrize(
    "kind,meta",
    [
        ("reported_fact", None),
        ("derived_fact", None),
        ("other", {"evidence_role": "reported_fact"}),
        ("other", {"fact_variable": "x"}),
    ],
)
def test_source_measurements_cannot_be_smuggled_into_alternatives(kind, meta):
    c = us.NamedLinearConstraint(
        "fact", "x", lower=1, upper=1, kind=kind, metadata=meta
    )
    with pytest.raises(ValueError, match="source measurements"):
        f.DisclosureAlternative("choice", "Hypothetical", [c])
    with pytest.raises(ValueError, match="source measurements"):
        f.DisclosureEvidenceRequest("request", "Hypothetical", [c])


def test_catalog_bounds_and_unknown_dependencies_are_rejected():
    c = compare()
    with pytest.raises(ValueError, match="max_bundles"):
        f.plan_disclosure_evidence(
            c,
            [
                f.DisclosureEvidenceRequest(
                    "r", "Outcome", [row("x", "refund", upper=0)]
                )
            ],
            eliminate=["refund"],
            max_bundles=1,
        )
    for request in [
        f.DisclosureEvidenceRequest(
            "r", "Outcome", [row("x", "refund", upper=0)], requires=["missing"]
        ),
        f.DisclosureEvidenceRequest("r", "Outcome", mapping_names=["missing"]),
    ]:
        with pytest.raises(ValueError, match="unknown"):
            f.plan_disclosure_evidence(c, [request], eliminate=["refund"])
    a = f.DisclosureEvidenceRequest(
        "a", "Outcome", mapping_names=["joint"], requires=["b"]
    )
    b = f.DisclosureEvidenceRequest(
        "b", "Outcome", mapping_names=["joint"], requires=["a"]
    )
    with pytest.raises(ValueError, match="acyclic"):
        f.plan_disclosure_evidence(c, [a, b], eliminate=["refund"])


def test_empty_bundle_can_record_an_already_excluded_catalog_entry():
    p, m, e = fixture(bounded_other=True)
    c = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=m[:1], explanations=e
    )
    plan = f.plan_disclosure_evidence(c, [], eliminate=["refund"])
    check.assertEqual([b.requests for b in plan.minimal_bundles], [()])
    check.assertIn("No additional evidence", plan.to_markdown())
    check.assertEqual(json.loads(plan.to_json())["minimal_bundles"], [[]])


def test_plan_export_preserves_basis_and_outcome_witnesses():
    c = compare()
    plan = f.plan_disclosure_evidence(
        c,
        [
            f.DisclosureEvidenceRequest(
                "cap", "Refund at most 100", [row("cap", "refund", upper=100)]
            )
        ],
        eliminate=["refund"],
    )
    payload = json.loads(plan.to_json())
    check.assertEqual(
        payload["comparison"]["report"]["problem"], c.report.problem.as_dict()
    )
    outcome = payload["bundles"][1]["comparison"]
    for case in outcome["cases"]:
        if case["status"] == "feasible":
            check.assertTrue(case["witness"])
    check.assertTrue(
        any(
            (x["metadata"] or {}).get("comparison_role") == "request"
            for x in outcome["report"]["problem"]["constraints"]
        )
    )


def test_names_are_not_silently_split_into_characters():
    with pytest.raises(ValueError, match="not a string"):
        f.DisclosureEvidenceRequest("r", "Mapping outcome", mapping_names="joint")
    with pytest.raises(ValueError, match="not a string"):
        f.DisclosureEvidenceRequest(
            "r", "Mapping outcome", mapping_names=["joint"], requires="parent"
        )
    with pytest.raises(ValueError, match="not a string"):
        f.plan_disclosure_evidence(compare(), [], eliminate="refund")
    with pytest.raises(ValueError, match="not a string"):
        compare(targets="refund")


def test_direct_outcomes_show_either_account_or_neither_without_an_elimination_goal():
    p, mappings, explanations = fixture()
    requests = [
        f.DisclosureEvidenceRequest(
            "low", "Hypothetical refunds <=100", [row("cap", "refund", upper=100)]
        ),
        f.DisclosureEvidenceRequest(
            "middle",
            "Hypothetical refunds 490..510",
            [row("measurement", "refund", 490, 510)],
        ),
        f.DisclosureEvidenceRequest(
            "high", "Hypothetical refunds >=1400", [row("floor", "refund", lower=1400)]
        ),
    ]
    c = f.compare_disclosure_explanations(
        p, base_tier="reported", mappings=mappings, explanations=explanations
    )
    low = f.evaluate_disclosure_evidence(c, requests, selected=["low"], targets=None)
    high = f.evaluate_disclosure_evidence(c, requests, selected=["high"])
    check.assertTrue(low.status == high.status == "compatible")
    check.assertTrue(
        all(
            low.comparison.case(m.name, "refund").status == "infeasible"
            for m in mappings
        )
    )
    check.assertTrue(
        all(
            high.comparison.case(m.name, "transfer").status == "infeasible"
            for m in mappings
        )
    )
    check.assertEqual(
        low.comparison.case("joint", "transfer").intervals["refund"].upper, 0
    )
    check.assertTrue(high.comparison.case("joint", "refund").witness)
    middle = f.evaluate_disclosure_evidence(c, requests, selected=["middle"])
    check.assertEqual(middle.status, "uncovered_mapping")
    check.assertTrue(
        all(x.status == "feasible" for x in middle.comparison.mapping_checks)
    )
    check.assertTrue(all(x.status == "infeasible" for x in middle.comparison.cases))
    c = f.compare_disclosure_explanations(
        p,
        base_tier="reported",
        mappings=mappings,
        explanations=[
            *explanations,
            f.DisclosureAlternative("other", "Open mixed account"),
        ],
    )
    middle = f.evaluate_disclosure_evidence(c, requests, selected=["middle"])
    check.assertEqual(middle.status, "compatible")
    check.assertTrue(
        all(
            middle.comparison.case(m.name, "other").status == "feasible"
            for m in mappings
        )
    )
    contradictory = f.evaluate_disclosure_evidence(
        c, requests, selected=["low", "high"]
    )
    check.assertEqual(contradictory.status, "inconsistent_with_all_mappings")
    check.assertTrue(json.loads(middle.to_json())["comparison"]["cases"])
    check.assertIn("Hypothetical", middle.to_markdown())


def test_direct_outcome_dependencies_empty_selection_and_catalog_validation():
    c = compare()
    requests = [
        f.DisclosureEvidenceRequest(
            "scope", "Matched scope review", mapping_names=["dr_only"]
        ),
        f.DisclosureEvidenceRequest(
            "cap",
            "Conditional amount",
            [row("cap", "refund", upper=100)],
            requires=["scope"],
        ),
    ]
    before = c.source_problem.as_dict()
    missing = f.evaluate_disclosure_evidence(c, requests, selected=["cap"])
    check.assertTrue(
        missing.status == "missing_dependencies" and missing.comparison is None
    )
    complete = f.evaluate_disclosure_evidence(c, requests)
    check.assertEqual(complete.status, "compatible")
    check.assertEqual(complete.retained_mappings, ("dr_only",))
    payload = json.loads(complete.to_json())
    check.assertEqual(payload["request_definitions"][0]["mapping_names"], ["dr_only"])
    check.assertEqual(payload["request_definitions"][1]["requires"], ["scope"])
    empty = f.evaluate_disclosure_evidence(c, requests, selected=())
    check.assertTrue(empty.requests == () and empty.status == "compatible")
    check.assertEqual(empty.retained_mappings, ("dr_only", "joint"))
    check.assertEqual(c.source_problem.as_dict(), before)
    for selected in [["missing"], ["cap", "cap"], "cap"]:
        with pytest.raises(ValueError):
            f.evaluate_disclosure_evidence(c, requests, selected=selected)
    with pytest.raises(ValueError, match="unknown"):
        f.evaluate_disclosure_evidence(c, requests[1:])
    with pytest.raises(ValueError, match="targets"):
        f.evaluate_disclosure_evidence(c, requests, targets=["unknown"])
    with pytest.raises(ValueError, match="boolean"):
        f.evaluate_disclosure_evidence(
            c, requests, selected=["cap"], include_conflicts="yes"
        )
