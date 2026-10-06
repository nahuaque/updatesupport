"""Scope, evidence selection, headline logic, disclosure tables, and replay."""

import json
import unittest
from dataclasses import replace

import pytest
import updatesupport as us
import updatesupport_finance as uf

# Keep assertions active under optimized Python, matching the existing suite.
check = unittest.TestCase()


AS_OF = "2026-10-06T12:00:00+00:00"


def fact(
    issuer="A",
    value=10,
    *,
    fact_id=None,
    available_at="2026-02-01T00:00:00+00:00",
    period_end="2025-12-31",
    **kwargs,
):
    return uf.DisclosureFact(
        fact_id or issuer,
        issuer,
        "OCF",
        value,
        "currency",
        period_end,
        available_at,
        "filing",
        "https://example.org/filing",
        currency="USD",
        **kwargs,
    )


def compile_book(
    *,
    unknown=20,
    cash=10,
    observations=None,
    strict=False,
    transform="below",
    bounds=None,
):
    positions = [
        uf.PortfolioPosition("a1", 40, "A"),
        uf.PortfolioPosition("a2", 20, "A"),
        uf.PortfolioPosition("b", 20, "B"),
    ]
    if unknown:
        positions.append(uf.PortfolioPosition("u", unknown, "U"))
    if cash:
        positions.append(uf.PortfolioPosition("cash", cash, security_type="cash"))
    universe = uf.PortfolioUniverse(positions, AS_OF, "USD")
    observations = (
        observations
        if observations is not None
        else [
            uf.FundamentalObservation(fact(), "FY"),
            uf.FundamentalObservation(fact("B", -10), "FY"),
        ]
    )
    taxonomy = [
        uf.TaxonomyAssignment(i, {"sector": "Tech", "industry": i}, "test")
        for i in ("A", "B", "U")
    ]
    policy = uf.PortfolioMetricPolicy(
        "OCF",
        "currency",
        strict_taxonomy_as_of=strict,
        transform=transform,
        currency="USD",
        target_bounds=bounds,
    )
    return uf.compile_portfolio_evidence(
        universe, observations=observations, taxonomy=taxonomy, policy=policy
    )


def headline(book, **kwargs):
    config = dict(
        headline="Less than 30% negative OCF",
        decision=us.threshold_decision("<", 0.3),
        public=("sector",),
        hidden=("sector", "industry", "issuer_id", "target_status"),
        candidate_refinements=("industry", "target_status"),
    )
    config.update(kwargs)
    return uf.portfolio_headline_report(book, **config)


def test_scope_bounds_do_not_renormalize_unknowns_or_count_cash():
    book = compile_book()
    check.assertEqual(
        len(book.rows), 3
    )  # Both share classes retain their economic weights.
    check.assertEqual(book.coverage.covered_mean, 0.25)
    check.assertEqual(book.coverage.supplied_value, 110)
    check.assertEqual(book.coverage.eligible_value, 100)
    check.assertEqual(book.coverage.unknown_value, 20)
    check.assertEqual(book.coverage.observed_bounds, (0.2, 0.4))
    check.assertEqual(
        [e.status for e in book.coverage.entries],
        [
            "covered",
            "covered",
            "covered",
            "missing",
            "excluded",
        ],
    )
    check.assertEqual(book.coverage.as_dict()["covered_share_eligible"], 0.8)
    check.assertEqual(
        len([d for d in book.diagnostics if d["code"] == "taxonomy_as_of_unknown"]), 3
    )


def test_unknown_unbounded_metric_is_inconclusive():
    book = compile_book(transform="value")
    check.assertIs(book.coverage.observed_bounds, None)
    report = headline(book, scope="eligible", include_breaking=False)
    check.assertTrue(report.actual_headline == report.summary_support == "inconclusive")
    bounded = compile_book(transform="value", bounds=(-20, 20))
    check.assertEqual(bounded.coverage.observed_bounds, (0, 8))


def test_ledger_must_match_every_original_position():
    book = compile_book()
    with pytest.raises(ValueError, match="every position"):
        uf.PortfolioCoverageReport(book.universe, book.coverage.entries[:-1])
    entries = [
        replace(e, value=e.value + 1) if e.position_id == "b" else e
        for e in book.coverage.entries
    ]
    with pytest.raises(ValueError, match="original universe"):
        uf.PortfolioCoverageReport(book.universe, entries)
    with pytest.raises(ValueError, match="unique"):
        uf.PortfolioUniverse([book.universe.positions[0]] * 2, AS_OF, "USD")
    with pytest.raises(ValueError, match="nonnegative"):
        uf.PortfolioPosition("short", -1, "A")


def test_selection_respects_asof_amendments_and_fiscal_dates():
    observations = [
        uf.FundamentalObservation(fact(fact_id="old", value=10), "FY", "2025-12-30"),
        uf.FundamentalObservation(
            fact(
                fact_id="amendment", value=-10, available_at="2026-03-01T00:00:00+00:00"
            ),
            "FY",
            "2025-12-30",
        ),
        uf.FundamentalObservation(
            fact(fact_id="future", available_at="2026-10-07T00:00:00+00:00"), "FY"
        ),
        uf.FundamentalObservation(
            fact(fact_id="quarter", value=10, period_end="2026-06-30"), "Q"
        ),
        uf.FundamentalObservation(fact("B", -10), "FY"),
    ]
    book = compile_book(observations=observations)
    check.assertEqual(book.rows[0]["fact_id"], "amendment")
    check.assertEqual(book.coverage.covered_mean, 1)
    check.assertEqual(book.diagnostics[0]["reported"], "2025-12-31")
    check.assertEqual(book.diagnostics[0]["normalized"], "2025-12-30")
    tied = [
        *observations,
        uf.FundamentalObservation(
            replace(observations[1].fact, fact_id="tie", source_id="other"), "FY"
        ),
    ]
    check.assertEqual(compile_book(observations=tied).coverage.covered_value, 20)


def test_derived_evidence_requires_available_inputs_and_no_cycles():
    base = fact(fact_id="input", period_end="2024-12-31")
    derived = fact(
        fact_id="derived",
        reported=False,
        derivation="explicit-test-recipe",
        derivation_inputs=("input",),
    )
    inputs = [
        uf.FundamentalObservation(base, "FY"),
        uf.FundamentalObservation(derived, "FY"),
        uf.FundamentalObservation(fact("B", -10), "FY"),
    ]
    check.assertEqual(compile_book(observations=inputs).rows[0]["fact_id"], "derived")
    missing = [x for x in inputs if x.fact.fact_id != "input"]
    check.assertEqual(compile_book(observations=missing).coverage.covered_value, 20)
    cyclic = [
        replace(
            x,
            fact=replace(
                x.fact,
                reported=False,
                derivation="cycle-test",
                derivation_inputs=("derived",),
            ),
        )
        if x.fact.fact_id == "input"
        else x
        for x in inputs
    ]
    check.assertEqual(compile_book(observations=cyclic).coverage.covered_value, 20)


def test_context_and_taxonomy_requirements_quarantine_unusable_evidence():
    wrong = [
        uf.FundamentalObservation(fact(dimensions={"segment": "subsidiary"}), "FY"),
        uf.FundamentalObservation(fact("B", -10), "FY"),
    ]
    check.assertEqual(compile_book(observations=wrong).coverage.covered_value, 20)
    strict = compile_book(strict=True)
    check.assertEqual(strict.coverage.covered_value, 0)
    report = headline(strict, scope="eligible")
    check.assertTrue(report.actual_headline == report.summary_support == "inconclusive")
    check.assertTrue(report.observed_bounds == report.summary_bounds == (0, 1))
    check.assertIs(report.audit, None)
    replay = uf.capture_portfolio_snapshot(report).replay()
    check.assertEqual(replay.as_dict(), report.as_dict())


def test_dated_taxonomy_and_staleness_are_enforced():
    book = compile_book()
    dated = [
        replace(
            t,
            version="v1",
            available_at="2026-01-01T00:00:00Z",
            effective_at="2026-01-01T00:00:00Z",
        )
        for t in book.taxonomy
    ]
    strict = uf.compile_portfolio_evidence(
        book.universe,
        observations=book.observations,
        taxonomy=dated,
        policy=replace(book.policy, strict_taxonomy_as_of=True),
    )
    check.assertTrue(strict.coverage.covered_value == 80 and not strict.diagnostics)
    future = [replace(t, effective_at="2026-10-07T00:00:00Z") for t in dated]
    unavailable = uf.compile_portfolio_evidence(
        book.universe,
        observations=book.observations,
        taxonomy=future,
        policy=book.policy,
    )
    check.assertEqual(unavailable.coverage.covered_value, 0)
    stale = uf.compile_portfolio_evidence(
        book.universe,
        observations=book.observations,
        taxonomy=dated,
        policy=replace(book.policy, max_age_days=30),
    )
    check.assertEqual(stale.coverage.covered_value, 0)


def test_claim_configuration_is_validated_even_without_covered_rows():
    book = compile_book(strict=True)
    for kwargs in (
        {"scope": "whole"},
        {"hidden": ("sector",)},
        {"direct_target_refinements": ()},
        {"respect_q": "yes"},
        {"public": "sector"},
    ):
        with pytest.raises((TypeError, ValueError)):
            headline(book, **kwargs)


def test_headline_truth_and_summary_support_are_distinct():
    book = compile_book(unknown=0)
    report = headline(book)
    check.assertEqual(report.actual_headline, "supported")
    check.assertEqual(report.summary_support, "inconclusive")
    check.assertEqual(report.summary_bounds, (0, 1))
    check.assertEqual(report.breaking.distance, pytest.approx(0.050001))
    check.assertEqual(
        sum(t["amount"] for t in report.transfers), pytest.approx(4.00008)
    )
    kinds = {tuple(r["columns"]): r["kind"] for r in report.refinements}
    check.assertEqual(kinds[("industry",)], "independent")
    check.assertEqual(kinds[("target_status",)], "direct_target")
    false = headline(book, decision=us.threshold_decision("<", 0.2))
    check.assertEqual(false.actual_headline, "contradicted")
    check.assertEqual(false.breaking.status, "already_broken")
    # Core pass means invariant even if the headline is always false.
    contradicted = headline(
        book, public=("sector", "industry"), decision=us.threshold_decision("<", 0.2)
    )
    check.assertEqual(contradicted.audit.status, "pass")
    check.assertEqual(contradicted.summary_support, "contradicted")


def test_covered_refinement_cannot_repair_missing_whole_book_evidence():
    report = headline(compile_book(), scope="eligible")
    check.assertEqual(report.actual_headline, "inconclusive")
    check.assertTrue(
        all(r["headline_support"] == "inconclusive" for r in report.refinements)
    )
    check.assertEqual(report.summary_bounds, (0, 1))


def test_portfolio_snapshot_recompiles_replays_and_detects_corruption():
    report = headline(compile_book(), scope="eligible")
    snapshot = uf.capture_portfolio_snapshot(
        report, capture_references={"raw": "a" * 64}
    )
    restored = uf.PortfolioSnapshot.from_json(snapshot.to_json())
    check.assertEqual(restored.fingerprint, snapshot.fingerprint)
    check.assertEqual(
        json.loads(restored.replay().to_json()), json.loads(report.to_json())
    )
    check.assertTrue(not uf.compare_portfolio_snapshots(snapshot, restored)["changed"])
    check.assertTrue(not restored.runtime_differences())
    payload = snapshot.as_dict()
    payload["portfolio"]["rows"][0]["metric"] = 1
    with pytest.raises(ValueError, match="recompilation"):
        uf.PortfolioSnapshot(json.dumps(payload))
    payload = snapshot.as_dict()
    payload["result"]["actual_headline"] = "supported"
    with pytest.raises(ValueError, match="digest"):
        uf.PortfolioSnapshot(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA256"):
        uf.capture_portfolio_snapshot(report, capture_references={"raw": "invalid"})


def margin_fact(name, value, axis):
    return uf.DisclosureFact(
        name,
        "Company",
        "Revenue",
        value,
        "USD",
        "2025-12-31",
        "2026-02-01T00:00:00+00:00",
        "filing",
        "https://example.org/filing",
        currency="USD",
        decimals=0,
        dimensions=axis,
    )


def table(*, zeros=(), margins=None):
    rows = [
        uf.AllocationMember("Phone"),
        uf.AllocationMember("Other products"),
        uf.AllocationMember("Products", "subtotal", ("Phone", "Other products")),
        uf.AllocationMember("Services"),
    ]
    columns = [
        uf.AllocationMember("US"),
        uf.AllocationMember(
            "Outside US", qualifier="Includes all non-US jurisdictions"
        ),
    ]
    margins = margins or [
        uf.AllocationMargin("total", margin_fact("total", 100, {})),
        uf.AllocationMargin(
            "phone", margin_fact("phone", 60, {"product": "Phone"}), rows=("Phone",)
        ),
        uf.AllocationMargin(
            "us", margin_fact("us", 40, {"region": "US"}), columns=("US",)
        ),
    ]
    return uf.allocation_table(
        rows=rows,
        columns=columns,
        margins=margins,
        measure="Revenue",
        unit="USD",
        as_of=AS_OF,
        exhaustive=True,
        structural_zeros=zeros,
    )


def test_allocation_builder_bounds_rounding_and_snapshot_replay():
    compiled = table()
    check.assertEqual(len(compiled.cell_variables), 6)
    target = compiled.target("phone_outside", rows=("Phone",), columns=("Outside US",))
    problem = compiled.problem([target])
    result = uf.triangulate_disclosure(problem)
    interval = result.intervals[0]
    check.assertEqual(interval.lower, pytest.approx(19))
    check.assertEqual(interval.upper, pytest.approx(60.5))
    snapshot = uf.capture_disclosure_snapshot(
        problem,
        facts=compiled.facts,
        as_of=AS_OF,
        target=target.name,
        tier="reported",
        assumptions=compiled.assumptions,
    )
    check.assertEqual(
        json.loads(snapshot.replay().to_json()),
        json.loads(snapshot.to_json())["result"],
    )
    with pytest.raises(ValueError, match="overlap"):
        compiled.target("bad", rows=("Products", "Phone"))


def test_allocation_hierarchy_context_precision_and_exclusions():
    compiled = table(zeros=(("Phone", "US"),))
    interval = uf.triangulate_disclosure(
        compiled.problem(
            [compiled.target("outside", rows=("Phone",), columns=("Outside US",))]
        )
    ).intervals[0]
    check.assertEqual(interval.lower, pytest.approx(59.5))
    with pytest.raises(ValueError, match="leaf cells"):
        table(zeros=(("Products", "US"),))
    with pytest.raises(ValueError, match="precision"):
        table(
            margins=[
                uf.AllocationMargin(
                    "total", replace(margin_fact("t", 100, {}), decimals=None)
                )
            ]
        )
    with pytest.raises(ValueError, match="share entity"):
        table(
            margins=[
                uf.AllocationMargin("total", margin_fact("t", 100, {})),
                uf.AllocationMargin(
                    "bad",
                    replace(
                        margin_fact("b", 40, {"region": "US"}), period_end="2024-12-31"
                    ),
                ),
            ]
        )
    with pytest.raises(ValueError, match="cyclic"):
        uf.allocation_table(
            rows=[uf.AllocationMember("loop", "subtotal", ("loop",))],
            columns=[uf.AllocationMember("US")],
            margins=[],
            measure="Revenue",
            unit="USD",
            as_of=AS_OF,
            exhaustive=True,
        )


def test_allocation_measure_override_preserves_source_concept_and_qualifiers():
    source = replace(
        margin_fact("native", 60, {"product": "Phone"}), concept="NativeProductSales"
    )
    compiled = table(
        margins=[
            uf.AllocationMargin("phone", source, rows=("Phone",), measure="Revenue")
        ]
    )
    check.assertEqual(compiled.facts[0].concept, "NativeProductSales")
    relation = compiled.constraints[1]
    check.assertEqual(relation.metadata["declared_measure"], "Revenue")
    check.assertEqual(relation.metadata["source_concept"], "NativeProductSales")
    check.assertEqual(
        relation.metadata["column_qualifiers"]["Outside US"],
        "Includes all non-US jurisdictions",
    )
    check.assertIn(
        "Includes all non-US jurisdictions", compiled.variables[1].description
    )


def test_mandates_bound_forward_and_inverse_and_replay_tuple_keyed_q():
    pytest.importorskip("cvxpy")
    book = compile_book(unknown=0)
    hidden = ("sector", "industry", "issuer_id", "target_status")
    mandate = uf.PortfolioMandate(issuer_caps={"B": 0.28})
    q = uf.q_portfolio_mandate(book, hidden=hidden, mandate=mandate)
    report = headline(book, q=q)
    check.assertEqual(report.summary_support, "supported")
    check.assertEqual(report.summary_bounds[1], pytest.approx(0.28, abs=1e-7))
    check.assertEqual(report.breaking.status, "infeasible")
    check.assertTrue(report.breaking.respect_q)
    snapshot = uf.capture_portfolio_snapshot(report)
    replay = snapshot.replay()
    check.assertEqual(replay.summary_support, report.summary_support)
    check.assertEqual(replay.breaking.status, "infeasible")
    frozen = uf.q_portfolio_mandate(
        book, hidden=hidden, mandate=uf.PortfolioMandate(preserve_columns=("industry",))
    )
    report = headline(book, q=frozen)
    check.assertEqual(report.summary_bounds, pytest.approx((0.25, 0.25), abs=1e-7))
    with pytest.raises(ValueError, match="exceeds"):
        uf.q_portfolio_mandate(
            book, hidden=hidden, mandate=uf.PortfolioMandate(issuer_caps={"B": 0.2})
        )
    with pytest.raises(ValueError, match="absent"):
        uf.q_portfolio_mandate(
            book,
            hidden=hidden,
            mandate=uf.PortfolioMandate(locked_issuers=("missing",)),
        )
