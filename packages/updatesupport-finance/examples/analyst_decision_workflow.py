"""Synthetic, provider-neutral customer exposure audit with an offline brief."""

from dataclasses import replace

import updatesupport_finance as f


def run():
    cutoff = "2026-10-06T00:00:00Z"
    definition = f.MeasurementDefinition("revenue", "flow", "USD", "USD")
    source = f.DisclosureFact(
        "revenue",
        "ExampleCo",
        "revenue",
        100_000_000,
        "USD",
        "2026-08-31",
        "2026-09-01T00:00:00Z",
        "synthetic",
        "https://example.org/synthetic",
        period_start="2026-06-01",
        currency="USD",
        decimals=-6,
    )
    evidence = f.compile_disclosure_evidence(
        [f.MeasurementObservation(source, definition, complete=True)],
        requirements=[
            f.DisclosureRequirement("revenue", definition, "ExampleCo", max_age_days=90)
        ],
        as_of=cutoff,
        unit_scales={"USD": ("USD million", 1e6)},
    ).require_valid()
    total = evidence.fact_for("revenue")
    segment = replace(
        total, fact_id="segment", value=90, dimensions={"segment": "compute"}
    )
    share = replace(
        total,
        fact_id="share",
        concept="direct buyer revenue share",
        value=0.4,
        unit="ratio",
        currency=None,
        decimals=2,
        dimensions={"customer": "anonymous A"},
    )
    table = f.allocation_table(
        rows=[f.AllocationMember("compute"), f.AllocationMember("other")],
        columns=[f.AllocationMember("buyer"), f.AllocationMember("remaining")],
        margins=[
            f.AllocationMargin("total", total),
            f.AllocationMargin("compute", segment, rows=["compute"]),
            f.AllocationShareMargin(
                "buyer", share, denominator="total", columns=["buyer"]
            ),
        ],
        measure="revenue",
        unit="USD million",
        as_of=cutoff,
        exhaustive=True,
    )
    target = replace(
        table.target("exposure", rows=["compute"], columns=["buyer"]),
        label="Buyer's compute revenue",
    )
    threshold = table.share_target(
        "one_third",
        denominator="compute",
        threshold=1 / 3,
        rows=["compute"],
        columns=["buyer"],
        label="Buyer contributes at least one third of compute revenue",
    )
    spec = table.problem([target, threshold])
    scope = f.DisclosureScope(
        "ExampleCo",
        "direct buyer compute revenue",
        "USD million",
        "2026-08-31",
        "fiscal_quarter",
        "anonymous A in this quarter",
        period_start="2026-06-01",
        currency="USD",
        cohort_kind="anonymous",
        customer_basis="direct buyer",
    )
    snapshot = f.capture_disclosure_snapshot(
        spec,
        facts=table.facts,
        as_of=cutoff,
        target="exposure",
        tier="reported",
        context={**evidence.snapshot_context(), "scope": scope.as_dict()},
        assumptions=[
            "The displayed 40% uses inclusive nearest-percentage-point rounding."
        ],
    )
    brief = f.AnalystDecisionBrief(
        "Customer exposure decision",
        [
            f.AnalystFinding(
                "How much compute revenue must belong to this buyer?",
                snapshot.replay(),
                scope,
                f.AnalystBaseline(
                    "40% total revenue less nominal other-segment capacity",
                    30,
                    "USD million",
                ),
                missing_evidence=["Customer-by-segment revenue"],
                unanswered_scope=["Ultimate end-buyer identity and profit impact"],
            )
        ],
    )
    print(brief.to_markdown())
    return snapshot, brief


if __name__ == "__main__":
    run()
