"""Offline synthetic explanation comparison and hypothetical evidence planning.

Run: uv run python packages/updatesupport-finance/examples/disclosure_explanations.py
All amounts and source links are fictional; no provider connection is needed.
"""

import updatesupport as us
import updatesupport_finance as f


def condition(name, terms, lower=None, upper=None):
    return us.NamedLinearConstraint(
        name, terms, lower=lower, upper=upper, kind="analyst_policy"
    )


def example():
    unit = "USD million"
    facts = [
        f.DisclosureFact(
            "synthetic:" + name,
            "Synthetic Company",
            name,
            value,
            unit,
            "2026-06-30",
            "2026-08-01T00:00:00Z",
            "synthetic-disclosure",
            "https://example.invalid/synthetic-disclosure",
            decimals=0,
            currency="USD",
            period_start="2026-04-01",
        )
        for name, value in [("exit", 150), ("delta", 220), ("cf", 80)]
    ]
    rows = [
        f.disclosure_fact_constraint("fact:" + fact.concept, fact.concept, fact)
        for fact in facts
    ]
    rows += [
        us.NamedLinearConstraint(
            "customer_caption_bridge",
            {"exit": 1, "refund": -1, "transfer": -1, "residual": -1},
            lower=0,
            upper=0,
            kind="reconciliation",
        ),
        us.NamedLinearConstraint(
            "deferred_revenue_bridge",
            {"delta": 1, "operating": -1, "transfer": -1, "other": -1},
            lower=0,
            upper=0,
            kind="reconciliation",
        ),
    ]
    problem = us.NamedLinearFeasibilityProblem(
        variables=[
            us.NamedLinearVariable(
                name,
                lower=0 if name in {"exit", "refund", "transfer"} else None,
                unit=unit,
            )
            for name in [
                "exit",
                "refund",
                "transfer",
                "residual",
                "delta",
                "cf",
                "operating",
                "other",
            ]
        ],
        constraints=rows,
        targets=[
            us.NamedLinearTarget(name, name, unit=unit)
            for name in ["refund", "transfer", "other"]
        ],
        scenarios=[us.NamedLinearScenario("reported", [c.name for c in rows])],
        title="Synthetic customer-caption movement",
    )
    mappings = [
        f.DisclosureAlternative(
            "dr_only",
            "CF line covers deferred-revenue operating movement only",
            [condition("scope", {"operating": 1, "cf": -1}, 0, 0)],
        ),
        f.DisclosureAlternative(
            "joint",
            "CF line also includes customer-caption cash refunds",
            [condition("scope", {"operating": 1, "refund": -1, "cf": -1}, 0, 0)],
        ),
    ]
    explanations = [
        f.DisclosureAlternative(
            "transfer_only",
            "Entire exit transfers to deferred revenue",
            [
                condition("no_refund", "refund", 0, 0),
                condition("no_residual", "residual", 0, 0),
            ],
        ),
        f.DisclosureAlternative(
            "refund_only",
            "Entire exit is refunded",
            [
                condition("no_transfer", "transfer", 0, 0),
                condition("no_residual", "residual", 0, 0),
            ],
        ),
    ]
    comparison = f.compare_disclosure_explanations(
        problem,
        base_tier="reported",
        mappings=mappings,
        explanations=explanations,
        evidence=facts,
    )
    requests = [
        f.DisclosureEvidenceRequest(
            "other_DR_cap",
            "Hypothetical other net DR additions at most 20",
            [condition("cap", "other", upper=20)],
        ),
        f.DisclosureEvidenceRequest(
            "scope_review",
            "Hypothetical review confirms the dr_only mapping",
            mapping_names=["dr_only"],
        ),
        f.DisclosureEvidenceRequest(
            "refund_cap",
            "Hypothetical gross refunds at most 20",
            [condition("cap", "refund", upper=20)],
        ),
    ]
    plan = f.plan_disclosure_evidence(comparison, requests, eliminate=["refund_only"])
    if any(c.status != "feasible" for c in comparison.cases):
        raise RuntimeError("Unexpected synthetic explanation compatibility")
    if {b.requests for b in plan.minimal_bundles} != {
        ("refund_cap",),
        ("other_DR_cap", "scope_review"),
    }:
        raise RuntimeError("Unexpected synthetic minimal evidence bundles")
    selected = comparison.case("joint", "refund_only")
    snapshot = f.capture_disclosure_snapshot(
        comparison.report.problem,
        facts=facts,
        as_of="2026-08-01T00:00:00Z",
        target="refund",
        tier=selected.tier,
        context=comparison.snapshot_context(),
        assumptions=[
            "Synthetic illustration; hypotheses do not identify an actual cause."
        ],
    )
    restored = f.DisclosureSnapshot.from_json(snapshot.to_json())
    if restored.fingerprint != snapshot.fingerprint:
        raise RuntimeError("Snapshot round-trip changed the fingerprint")
    restored.replay()
    return comparison, plan


def outcome_example(comparison):
    basis = f.compare_disclosure_explanations(
        comparison.source_problem,
        base_tier=comparison.base_tier,
        mappings=comparison.mappings,
        explanations=[
            *comparison.explanations,
            f.DisclosureAlternative(
                "other_or_mixed", "Open mixed or unclassified account"
            ),
        ],
        evidence=comparison.evidence,
    )
    answers = [
        f.DisclosureEvidenceRequest(
            "low",
            "Hypothetical refunds <=20",
            [condition("amount", "refund", upper=20)],
        ),
        f.DisclosureEvidenceRequest(
            "middle",
            "Hypothetical refunds 50..60",
            [condition("amount", "refund", 50, 60)],
        ),
        f.DisclosureEvidenceRequest(
            "high",
            "Hypothetical refunds >=140",
            [condition("amount", "refund", lower=140)],
        ),
    ]
    outcomes = {
        answer.name: f.evaluate_disclosure_evidence(
            basis, answers, selected=[answer.name]
        )
        for answer in answers
    }
    if any(o.status != "compatible" for o in outcomes.values()):
        raise RuntimeError("Unexpected synthetic evidence outcome status")
    for name, excluded in [
        ("low", ["refund_only"]),
        ("middle", ["transfer_only", "refund_only"]),
        ("high", ["transfer_only"]),
    ]:
        summary = outcomes[name].comparison.summary()
        if {
            r["explanation"]
            for r in summary
            if r["status"] == "excluded_in_all_feasible_mappings"
        } != set(excluded):
            raise RuntimeError("Unexpected synthetic surviving explanations")
    return outcomes


if __name__ == "__main__":
    comparison, plan = example()
    print(comparison.to_markdown())
    print()
    print(plan.to_markdown())
    print("\n# What different answers leave standing\n")
    for name, outcome in outcome_example(comparison).items():
        possible = [
            r["explanation"]
            for r in outcome.comparison.summary()
            if r["possible_mappings"]
        ]
        print(f"- {name}: {', '.join(possible)}")
