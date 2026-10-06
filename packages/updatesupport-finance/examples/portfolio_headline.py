"""Synthetic, provider-neutral scope/headline/replay workflow."""

from __future__ import annotations

import updatesupport as us
import updatesupport_finance as uf


def run():
    universe = uf.PortfolioUniverse(
        [
            uf.PortfolioPosition("a", 60, "A"),
            uf.PortfolioPosition("b", 20, "B"),
            uf.PortfolioPosition("missing", 20, "U"),
            uf.PortfolioPosition("cash", 10, security_type="cash"),
        ],
        as_of="2026-10-06T00:00:00Z",
        currency="USD",
        name="Synthetic portfolio",
    )
    observations = [
        uf.FundamentalObservation(
            uf.DisclosureFact(
                fact_id=issuer,
                entity=issuer,
                concept="OperatingCashFlow",
                value=value,
                unit="currency",
                currency="USD",
                period_end="2025-12-31",
                available_at="2026-02-01T00:00:00Z",
                source_id="synthetic-filing",
                source_url="https://example.org/synthetic-filing",
            ),
            "FY",
        )
        for issuer, value in (("A", 10), ("B", -10))
    ]
    taxonomy = [
        uf.TaxonomyAssignment(
            issuer,
            {"sector": "Tech", "industry": issuer},
            "synthetic",
            version="v1",
            available_at="2026-01-01T00:00:00Z",
            effective_at="2026-01-01T00:00:00Z",
        )
        for issuer in ("A", "B", "U")
    ]
    compiled = uf.compile_portfolio_evidence(
        universe,
        observations=observations,
        taxonomy=taxonomy,
        policy=uf.PortfolioMetricPolicy(
            "OperatingCashFlow", "currency", strict_taxonomy_as_of=True
        ),
    )
    hidden = ("sector", "industry", "issuer_id", "target_status")
    config = dict(
        headline="Less than 30% negative annual OCF",
        decision=us.threshold_decision("<", 0.30),
        public=("sector",),
        hidden=hidden,
        candidate_refinements=("industry", "target_status"),
    )
    report = uf.portfolio_headline_report(compiled, **config)
    print(report.to_markdown())
    # Covered book: 25%; eligible scope: [20%, 40%] from missing fundamentals.
    eligible = uf.portfolio_headline_report(compiled, scope="eligible", **config)
    if (
        report.actual_headline != "supported"
        or eligible.actual_headline != "inconclusive"
    ):
        raise RuntimeError("Unexpected headline verdicts")
    snapshot = uf.capture_portfolio_snapshot(eligible)
    if snapshot.replay().as_dict() != eligible.as_dict():
        raise RuntimeError("Offline replay changed the report")
    print("\nEligible observed bounds:", eligible.observed_bounds)
    print("Offline replay fingerprint:", snapshot.fingerprint)
    return compiled, config


if __name__ == "__main__":
    run()
