"""Synthetic semantic and numerical regressions; no provider fixtures needed."""

import json
import unittest
from dataclasses import replace

import pytest
import updatesupport_finance as f

check = unittest.TestCase()
AS_OF = "2026-10-06T12:00:00Z"


def fact(name, value, **kw):
    defaults = dict(
        unit="USD",
        currency="USD",
        decimals=0,
        period_start="2026-06-01",
        period_end="2026-08-31",
        available_at="2026-09-01T00:00:00Z",
    )
    defaults.update(kw)
    return f.DisclosureFact(
        name,
        "issuer",
        name,
        value,
        source_id="source",
        source_url="https://example.org/filing",
        **defaults,
    )


def test_compiler_quarantines_partial_totals_and_does_not_fill_zero():
    definition = f.MeasurementDefinition(
        "borrowings", "stock", "USD", "USD", {"includes": "current and noncurrent"}
    )
    current = f.MeasurementObservation(
        fact("current", 10, period_start=None),
        replace(definition, name="current"),
        complete=True,
    )
    total = f.MeasurementObservation(
        fact("total", 10, period_start=None),
        definition,
        complete=True,
        components=("current",),
    )
    requirement = f.DisclosureRequirement(
        "total",
        definition,
        "issuer",
        required_components=("current", "noncurrent"),
        require_complete=True,
    )
    result = f.compile_disclosure_evidence(
        [total, current], requirements=[requirement], as_of=AS_OF
    )
    check.assertEqual(result.ledger[0]["status"], "incomplete")
    check.assertIn("component:noncurrent", result.ledger[0]["issues"])
    check.assertEqual(result.facts, ())
    with pytest.raises(ValueError, match="incomplete"):
        result.require_valid()
    with pytest.raises(ValueError, match="not satisfied"):
        result.fact_for("total")


def test_compiler_transitive_completeness_versions_and_future_schedule():
    definition = f.MeasurementDefinition("flow", "flow", "USD", "USD")
    parent = f.MeasurementObservation(
        fact("part", 5), replace(definition, name="part"), complete=False
    )
    total = f.MeasurementObservation(
        fact("sum", 5, reported=False, derivation="sum", derivation_inputs=("part",)),
        definition,
        complete=True,
    )
    req = f.DisclosureRequirement("flow", definition, "issuer")
    result = f.compile_disclosure_evidence(
        [parent, total], requirements=[req], as_of=AS_OF
    )
    check.assertEqual(result.ledger[0]["status"], "incomplete")
    old = f.MeasurementObservation(
        fact("old", 5, period_end="2025-08-31", period_start="2025-06-01"), definition
    )
    result = f.compile_disclosure_evidence(
        [old], requirements=[replace(req, max_age_days=100)], as_of=AS_OF
    )
    check.assertEqual(result.ledger[0]["status"], "stale")
    newer = f.MeasurementObservation(
        fact("new", 8, available_at="2027-01-01T00:00:00Z"), definition
    )
    result = f.compile_disclosure_evidence(
        [old, newer], requirements=[req], as_of=AS_OF
    ).require_valid()
    check.assertEqual(result.fact_for("flow").fact_id, "old")
    ambiguous = f.MeasurementObservation(
        replace(old.fact, fact_id="another"), definition
    )
    check.assertEqual(
        f.compile_disclosure_evidence(
            [old, ambiguous], requirements=[req], as_of=AS_OF
        ).ledger[0]["status"],
        "ambiguous",
    )
    schedule = replace(definition, kind="schedule")
    projected = f.MeasurementObservation(
        fact("future", 7, period_end="2027-05-31"), schedule
    )
    f.compile_disclosure_evidence(
        [projected], requirements=[replace(req, definition=schedule)], as_of=AS_OF
    ).require_valid()
    incompatible = f.compile_disclosure_evidence(
        [old],
        requirements=[replace(req, definition=replace(definition, kind="stock"))],
        as_of=AS_OF,
    )
    check.assertEqual(incompatible.ledger[0]["status"], "incompatible")


def test_normalization_preserves_source_sign_rounding_and_snapshot_checks():
    source = fact("outflow", 5_000_000, decimals=-6)
    norm = f.normalize_disclosure_fact(
        source, unit="USD million", source_units_per_solver_unit=1e6, cash_multiplier=-1
    )
    check.assertEqual(norm.source, source)
    check.assertEqual(norm.fact.base_value, -5)
    check.assertEqual(norm.fact.decimals, 0)
    spec = f.disclosure_triangulation_spec(
        variables=[f.disclosure_variable("cash", lower=None, unit="USD million")],
        constraints=[f.disclosure_fact_constraint("cash_fact", "cash", norm.fact)],
        targets=[f.disclosure_target("cash", "cash", unit="USD million")],
        tiers=[f.disclosure_tier("history", ["cash_fact"])],
    )
    snapshot = f.capture_disclosure_snapshot(
        spec,
        facts=[norm.fact],
        as_of=AS_OF,
        target="cash",
        tier="history",
        context={"normalizations": [norm.as_dict()]},
    )
    check.assertAlmostEqual(snapshot.replay().interval.lower, -5.5)
    raw_spec = replace(
        spec,
        variables=[f.disclosure_variable("cash", lower=None, unit="USD")],
        constraints=[f.disclosure_fact_constraint("cash_fact", "cash", source)],
        targets=[f.disclosure_target("cash", "cash", scale=1e6)],
    )
    raw = f.triangulate_disclosure(raw_spec).interval(target="cash", scenario="history")
    check.assertAlmostEqual(raw.scaled_lower, -snapshot.replay().interval.upper)
    check.assertAlmostEqual(raw.scaled_upper, -snapshot.replay().interval.lower)
    check.assertEqual(raw_spec.feasibility_tolerance, spec.feasibility_tolerance)
    payload = snapshot.as_dict()
    payload["context"]["normalizations"][0]["source"]["value"] = 7e6
    with pytest.raises(ValueError, match="normalization"):
        f.DisclosureSnapshot.from_json(json.dumps(payload))
    for factor in (3, 0):
        with pytest.raises(ValueError):
            f.normalize_disclosure_fact(
                source, unit="USD million", source_units_per_solver_unit=factor
            )
    with pytest.raises(ValueError):
        f.normalize_disclosure_fact(source, unit="USD", cash_multiplier=True)


def test_cash_measures_share_primitives_and_require_scope_bridge():
    measures = [
        f.CashMeasure("net", {"capex": 1, "prepay": -1}, "declared net capex"),
        f.CashMeasure("cfo_ex", {"cfo": 1, "prepay": -1}, "named advances excluded"),
        f.CashMeasure(
            "consistent", {"cfo_ex": 1, "net": -1}, "consistent cash surplus"
        ),
    ]
    constraints = f.cash_measure_constraints(
        measures, primitives=["cfo", "capex", "prepay"]
    )
    check.assertNotIn("prepay", constraints[-1].expression.coefficients)
    with pytest.raises(ValueError, match="cyclic"):
        f.cash_measure_constraints(
            [
                f.CashMeasure("a", {"b": 1}, "cash"),
                f.CashMeasure("b", {"a": 1}, "cash"),
            ],
            primitives=[],
        )
    with pytest.raises(ValueError, match="scope"):
        f.cash_bridge_constraint(
            "bridge",
            opening="opening",
            closing="closing",
            movements={"cfo": 1},
            opening_scope="all",
            closing_scope="unrestricted",
        )
    bridge = f.cash_bridge_constraint(
        "bridge",
        opening="opening",
        closing="closing",
        movements={"cfo": 1},
        opening_scope="all",
        closing_scope="unrestricted",
        scope_adjustment="minus_restricted",
    )
    check.assertEqual(bridge.expression.coefficients["minus_restricted"], -1)


def table():
    amounts = [
        replace(fact("total", 100), concept="revenue"),
        replace(
            fact("compute", 90, dimensions={"segment": "compute"}), concept="revenue"
        ),
        replace(fact("other", 10, dimensions={"segment": "other"}), concept="revenue"),
    ]
    share = fact(
        "buyer",
        0.4,
        unit="ratio",
        currency=None,
        decimals=2,
        dimensions={"customer": "anonymous"},
    )
    return f.allocation_table(
        rows=[f.AllocationMember("compute"), f.AllocationMember("other")],
        columns=[f.AllocationMember("buyer"), f.AllocationMember("remaining")],
        margins=[
            f.AllocationMargin("total", amounts[0]),
            f.AllocationMargin("compute", amounts[1], rows=["compute"]),
            f.AllocationMargin("other", amounts[2], rows=["other"]),
            f.AllocationShareMargin(
                "buyer", share, denominator="total", columns=["buyer"]
            ),
        ],
        measure="revenue",
        unit="USD",
        as_of=AS_OF,
        exhaustive=True,
    )


def test_share_table_bounds_threshold_and_custom_tier_positivity():
    t = table()
    amount = t.target("buyer_compute", rows=["compute"], columns=["buyer"])
    threshold = t.share_target(
        "one_third",
        denominator="compute",
        threshold=1 / 3,
        rows=["compute"],
        columns=["buyer"],
        label="Buyer supplies at least one third",
    )
    spec = t.problem([amount, threshold])
    report = f.triangulate_disclosure(spec)
    # Total also equals both segment margins; their jointly feasible endpoints
    # are tighter than independently combining total and other-segment extremes.
    check.assertAlmostEqual(
        report.interval(target="buyer_compute", scenario="reported").lower, 29
    )
    check.assertEqual(
        f.disclosure_claim(target="one_third", tier="reported", lower_at_least=0)
        .audit(report)
        .verdict,
        "inconclusive",
    )
    partial = t.problem([threshold], scenarios=[f.disclosure_tier("partial", [])])
    check.assertIn(threshold.positivity.name, partial.scenarios[0].constraints)
    check.assertEqual(
        f.triangulate_disclosure(partial)
        .interval(target="one_third", scenario="partial")
        .status,
        "unbounded",
    )
    with pytest.raises(ValueError, match="positive"):
        f.share_threshold_target(
            "bad",
            numerator="n",
            denominator="d",
            threshold=0.3,
            denominator_lower_bound=0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        f.percentage_constraints(
            "bad", numerator="n", denominator="d", fact=t.facts[-1]
        )
    with pytest.raises(ValueError, match="policy"):
        f.percentage_constraints(
            "bad",
            numerator="n",
            denominator="d",
            fact=t.facts[-1],
            bounds=(0.3, 0.5),
            denominator_nonnegative=True,
        )
    with pytest.raises(ValueError, match="contained"):
        t.share_target(
            "invalid", denominator="compute", threshold=0.3, columns=["buyer"]
        )
    with pytest.raises(ValueError, match="amount margin"):
        t.share_target("invalid", denominator="absent", threshold=0.3)


@pytest.mark.parametrize("value, expected", [(0, (0, 0.005)), (1, (0.995, 1))])
def test_percentage_rounding_intersects_the_share_domain(value, expected):
    share = fact("edge", value, unit="ratio", currency=None, decimals=2)
    constraints = f.percentage_constraints(
        "edge", numerator="n", denominator="d", fact=share, denominator_nonnegative=True
    )
    check.assertEqual(tuple(constraints[0].metadata["share_bounds"]), expected)


def test_percentage_margin_rejects_period_and_denominator_mismatch():
    amount = fact("revenue", 100)
    share = fact(
        "share", 0.4, unit="ratio", currency=None, decimals=2, period_end="2026-07-31"
    )
    for denominator in ("absent", "total"):
        with pytest.raises(ValueError):
            f.allocation_table(
                rows=[f.AllocationMember("r")],
                columns=[f.AllocationMember("buyer"), f.AllocationMember("other")],
                margins=[
                    f.AllocationMargin("total", amount),
                    f.AllocationShareMargin(
                        "share", share, denominator=denominator, columns=["buyer"]
                    ),
                ],
                measure="revenue",
                unit="USD",
                as_of=AS_OF,
                exhaustive=True,
            )


def test_windows_remain_unallocated_until_explicit_schedule_and_reject_lifetime():
    definition = f.MeasurementDefinition("recognition", "schedule", "USD")
    kwargs = dict(
        total="next12",
        source_window=f.DisclosureWindow("next12", "2026-09-01", "2027-08-31"),
        windows=[
            f.DisclosureWindow("fiscal9", "2026-09-01", "2027-05-31"),
            f.DisclosureWindow("last3", "2027-06-01", "2027-08-31"),
        ],
        unit="USD",
        cohort="reported backlog",
        cohort_as_of="2026-08-31",
        measurement=definition,
    )
    allocation = f.time_window_allocation("recognition", **kwargs)
    total = f.exact_disclosure_constraint("total", "next12", 100)
    pace = allocation.even_schedule_constraints(
        basis="months", policy="even monthly conversion"
    )
    constraints = [total, *allocation.constraints, *pace]
    spec = f.disclosure_triangulation_spec(
        variables=[f.disclosure_variable("next12"), *allocation.variables],
        constraints=constraints,
        targets=[f.disclosure_target("fiscal", "fiscal9")],
        tiers=[
            f.disclosure_tier("free", [total.name, allocation.constraints[0].name]),
            f.disclosure_tier("even", [c.name for c in constraints]),
        ],
    )
    report = f.triangulate_disclosure(spec)
    check.assertEqual(report.interval(target="fiscal", scenario="free").lower, 0)
    check.assertEqual(report.interval(target="fiscal", scenario="free").upper, 100)
    check.assertEqual(report.interval(target="fiscal", scenario="even").lower, 75)
    with pytest.raises(ValueError, match="flow/schedule"):
        f.time_window_allocation(
            "bad", **{**kwargs, "measurement": replace(definition, kind="lifetime")}
        )
    with pytest.raises(ValueError, match="contiguous"):
        f.time_window_allocation(
            "bad",
            **{
                **kwargs,
                "windows": [f.DisclosureWindow("a", "2026-09-02", "2027-08-31")],
            },
        )
    with pytest.raises(ValueError, match="whole"):
        f.DisclosureWindow("partial", "2026-09-02", "2026-09-30").duration("months")


def cash_model():
    history = f.ConstraintPolicy(
        "history",
        f.EvidenceRole.REPORTED_FACT,
        [f.exact_disclosure_constraint("opening", "opening", 10)],
    )
    accounting = f.ConstraintPolicy(
        "cash equation",
        "accounting_relationship",
        [
            f.stock_flow_constraint(
                "bridge",
                opening="opening",
                closing="cash",
                movements={"prepay": 1, "issuance": 1, "capex": -1},
            )
        ],
    )
    plan = f.ConstraintPolicy(
        "plan",
        "management_expectation",
        [f.exact_disclosure_constraint("capex", "capex", 20)],
    )
    policy = f.ConstraintPolicy(
        "funding",
        "analyst_policy",
        [
            f.exact_disclosure_constraint("prepay", "prepay", 5),
            f.exact_disclosure_constraint("issuance", "issuance", 10),
            f.interval_disclosure_constraint("floor", "cash", lower=5),
        ],
    )
    constraints = (
        *history.constraints,
        *accounting.constraints,
        *plan.constraints,
        *policy.constraints,
    )
    return f.disclosure_triangulation_spec(
        variables=[
            f.disclosure_variable(n, lower=None if n == "cash" else 0)
            for n in ("opening", "cash", "prepay", "issuance", "capex")
        ],
        constraints=constraints,
        targets=[
            f.disclosure_target("cash", "cash", label="Ending cash", unit="USD"),
            f.disclosure_target("issuance", "issuance"),
        ],
        tiers=[
            f.disclosure_tier("plan", [c.name for c in constraints]),
            f.disclosure_tier(
                "reported",
                [c.name for c in (*history.constraints, *accounting.constraints)],
            ),
        ],
    )


def test_stress_resolves_dependencies_and_reports_conflicts_and_break_even():
    spec = cash_model()
    bad = f.DisclosureStressCase(
        "delayed",
        "plan",
        {"prepay": f.exact_disclosure_constraint("delay", "prepay", 0)},
        "Prepayment delayed without replacement financing",
    )
    analysis = f.run_disclosure_scenarios(spec, [bad])
    check.assertEqual(
        analysis.report.interval(target="cash", scenario="delayed").status, "infeasible"
    )
    check.assertIn("delayed", analysis.conflicts)

    check.assertEqual(analysis.support["reported"]["basis"], "reported_evidence")
    check.assertEqual(analysis.support["plan"]["basis"], "conditional")
    for key in ("opening", "bridge"):
        with pytest.raises(ValueError, match="counterfactual"):
            f.compile_disclosure_scenarios(
                spec,
                [
                    f.DisclosureStressCase(
                        "bad", "plan", {key: None}, "illegal alteration"
                    )
                ],
            )
    counter = f.compile_disclosure_scenarios(
        spec,
        [
            f.DisclosureStressCase(
                "historical",
                "plan",
                {"opening": f.exact_disclosure_constraint("open", "opening", 15)},
                "deliberate history change",
                historical_counterfactual=True,
            )
        ],
    )
    check.assertEqual(
        f.triangulate_disclosure(counter)
        .interval(target="cash", scenario="historical")
        .lower,
        10,
    )
    breakeven = f.break_even_analysis(
        spec,
        target="issuance",
        tier="plan",
        release_constraints=["issuance"],
        conditions=[
            f.exact_disclosure_constraint("preserve", {"cash": 1, "opening": -1}, 0)
        ],
    )
    check.assertEqual(
        breakeven.report.interval(target="issuance", scenario="break_even").lower, 15
    )
    check.assertEqual(
        breakeven.report.interval(target="issuance", scenario="break_even").upper, 15
    )


def test_brief_distinguishes_infeasible_tiers_from_unbounded_tiers():
    spec = cash_model()
    bad = f.DisclosureStressCase(
        "delayed",
        "plan",
        {"prepay": f.exact_disclosure_constraint("delay", "prepay", 0)},
        "Delayed prepayment",
    )
    analysis = f.run_disclosure_scenarios(spec, [bad])
    pack = f.disclosure_audit_pack(analysis.report, target="cash", tier="delayed")
    scope = f.DisclosureScope(
        "issuer",
        "cash",
        "USD",
        "2027-05-31",
        "fiscal_year",
        "aggregate",
        period_start="2026-06-01",
    )
    brief = f.AnalystDecisionBrief(
        "Funding", [f.AnalystFinding("Can the plan be funded?", pack, scope)]
    )
    check.assertEqual(
        brief.as_dict()["findings"][0]["interval"]["status"], "infeasible"
    )
    markdown = brief.to_markdown()
    check.assertIn("| delayed | infeasible | — | — |", markdown)
    check.assertIn("**Ending cash: — to — USD** (infeasible)", markdown)
    # Genuine unbounded endpoints retain their meaning in another scenario.
    unbounded = f.disclosure_audit_pack(
        f.triangulate_disclosure(spec), target="cash", tier="reported"
    )
    check.assertIn(
        "unbounded",
        f.AnalystDecisionBrief(
            "Funding", [f.AnalystFinding("Range?", unbounded, scope)]
        ).to_markdown(),
    )


def test_brief_units_labels_scope_guards_and_legacy_replay():
    spec = cash_model()
    claim = f.disclosure_claim(
        target="cash", tier="plan", lower_at_least=0, label="Cash remains positive"
    )
    scope = f.DisclosureScope(
        "issuer",
        "cash",
        "USD",
        "2027-05-31",
        "fiscal_year",
        "aggregate",
        period_start="2026-06-01",
    )
    snap = f.capture_disclosure_snapshot(
        spec,
        facts=[],
        as_of=AS_OF,
        target="cash",
        tier="plan",
        claim=claim,
        context={"scope": scope.as_dict()},
    )
    brief = f.AnalystDecisionBrief(
        "Funding decision",
        [
            f.AnalystFinding(
                "Can cash cover the plan?",
                snap.replay(),
                scope,
                f.AnalystBaseline("nominal", 5, "USD"),
                missing_evidence=["dated prepayments"],
                unanswered_scope=["monthly liquidity"],
            )
        ],
    )
    markdown = brief.to_markdown()
    for text in (
        "Ending cash",
        "Cash remains positive",
        "conditional",
        "dated prepayments",
        "monthly liquidity",
    ):
        check.assertIn(text, markdown)
    with pytest.raises(ValueError, match="display unit"):
        f.AnalystFinding(
            "question", snap.replay(), scope, f.AnalystBaseline("mixed", 5, "$bn")
        )
    anonymous = replace(
        scope,
        measure="revenue",
        cohort="anonymous A",
        cohort_kind="anonymous",
        window_kind="fiscal_quarter",
    )
    check.assertFalse(
        f.compare_disclosure_scopes(
            anonymous, replace(anonymous, period_end="2027-08-31")
        )["comparable"]
    )
    check.assertFalse(
        f.compare_disclosure_scopes(scope, replace(scope, window_kind="fiscal_half"))[
            "comparable"
        ]
    )
    check.assertTrue(
        f.compare_disclosure_snapshots(snap, snap)["comparability"]["comparable"]
    )
    legacy = f.capture_disclosure_snapshot(
        spec, facts=[], as_of=AS_OF, target="cash", tier="plan"
    )
    check.assertNotIn("context", legacy.as_dict())
    check.assertEqual(
        f.compare_disclosure_snapshots(legacy, legacy)["comparability"]["status"],
        "unassessed",
    )
    check.assertEqual(
        f.DisclosureSnapshot.from_json(legacy.to_json()).replay().interval.lower, 5
    )
