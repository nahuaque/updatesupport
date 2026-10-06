"""Offline multi-metric diagnosis, local drill, research plan, and replay."""

import updatesupport as us
import updatesupport_finance as uf


def main():
    universe = uf.PortfolioUniverse(
        [uf.PortfolioPosition(i, v, i) for i, v in (("A", 40), ("B", 40), ("U", 20))],
        "2026-10-06T12:00:00Z",
        "USD",
    )
    observations = []
    for concept, values in (
        ("OCF", {"A": 10, "B": -10}),
        ("cash_gap", {"A": -5, "B": 5}),
    ):
        for issuer, value in values.items():
            fact = uf.DisclosureFact(
                f"{issuer}:{concept}",
                issuer,
                concept,
                value,
                "currency",
                "2025-12-31",
                "2026-02-01T00:00:00Z",
                "synthetic",
                "https://example.org/filing",
                currency="USD",
            )
            observations.append(uf.FundamentalObservation(fact, "FY"))
    taxonomy = [
        uf.TaxonomyAssignment(
            i, {"sector": "Tech", "industry": "Tools", "listing_age": age}, "synthetic"
        )
        for i, age in (("A", "young"), ("B", "old"), ("U", "old"))
    ]
    evidence = uf.compile_portfolio_metrics(
        universe,
        observations=observations,
        taxonomy=taxonomy,
        policies={
            "negative_ocf": uf.PortfolioMetricPolicy("OCF", "currency"),
            "negative_gap": uf.PortfolioMetricPolicy("cash_gap", "currency"),
        },
    )
    hidden = ("sector", "industry", "issuer_id", "listing_age")
    suggestions = us.suggest_conditional_refinements(
        evidence.rows,
        public=("sector",),
        targets=tuple(evidence.metrics),
        candidate_columns=("listing_age",),
        weight="weight",
    )
    report = uf.joint_portfolio_report(
        evidence,
        public=("sector",),
        hidden=hidden,
        candidate_refinements=("listing_age",),
        conditional_refinements=suggestions.candidates,
    )
    print(report.to_markdown())
    print(
        report.breaking_witness(
            "negative_ocf", decision=us.threshold_decision("<", 0.6)
        )["transferred_value"]
    )
    plan = uf.plan_portfolio_repairs(
        report,
        [
            uf.EvidenceAcquisition(
                "U annual cash package",
                ("U",),
                {"negative_ocf": 0, "negative_gap": 0},
                availability="unknown",
                source_requirements=("dated annual cash-flow filing",),
            )
        ],
        ambiguity_limits={"negative_ocf": 0.05, "negative_gap": 0.05},
    )
    print(plan.to_markdown())
    snapshot = uf.capture_joint_portfolio_snapshot(report)
    restored = uf.JointPortfolioSnapshot.from_json(snapshot.to_json())
    if not restored.replay_matches():
        raise RuntimeError("offline replay changed the result")
    frozen = uf.FrozenJointPortfolioContract.freeze(report, candidate_index=1)
    print(frozen.audit(evidence)["status"])  # Raw contract: evaluated_without_criteria.


if __name__ == "__main__":
    main()
