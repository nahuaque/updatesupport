"""Illustrative offline evidence, triangulation, and revision-replay workflow.

These are invented disclosures, not StockFit data or an actual issuer.
Run with ``uv run --package updatesupport-finance python
packages/updatesupport-finance/examples/disclosure_evidence_workflow.py``.
Pass ``--output DIRECTORY`` to save both snapshots and the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import updatesupport_finance as usf


def build_snapshot(*, revised: bool = False) -> usf.DisclosureSnapshot:
    version = "v2" if revised else "v1"
    available = "2026-03-01T12:00:00Z" if revised else "2026-02-01T12:00:00Z"
    common = {
        "entity": "illustrative-company",
        "unit": "USD",
        "currency": "USD",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "decimals": 0,
        "available_at": available,
        "source_id": f"illustrative-{version}",
        "source_url": f"https://example.com/illustrative-{version}",
    }
    facts = [
        usf.DisclosureFact(
            fact_id=f"category-total-{version}",
            concept="category_revenue",
            value=80 if revised else 100,
            **common,
        ),
        usf.DisclosureFact(
            fact_id=f"other-capacity-{version}",
            concept="segment_revenue",
            dimensions={"segment": "other"},
            value=35,
            **common,
        ),
    ]
    constraints = [
        usf.disclosure_fact_constraint("category_total", "total", facts[0]),
        usf.disclosure_fact_constraint("other_capacity", "capacity", facts[1]),
        usf.reconciliation_constraint(
            "allocation", total="total", components=["focus", "other"]
        ),
        usf.containment_constraint(
            "other_containment", child="other", parent="capacity"
        ),
    ]
    spec = usf.disclosure_triangulation_spec(
        title="Illustrative category-revenue allocation",
        variables=[
            usf.disclosure_variable(name, unit="USD")
            for name in ("total", "capacity", "focus", "other")
        ],
        constraints=constraints,
        targets=[usf.disclosure_target("focus_revenue", "focus", unit="USD")],
        tiers=[usf.disclosure_tier("disclosures", [row.name for row in constraints])],
    )
    claim = usf.disclosure_claim(
        target="focus_revenue", tier="disclosures", lower_at_least=60
    )
    return usf.capture_disclosure_snapshot(
        spec,
        facts=facts,
        as_of=available,
        target="focus_revenue",
        tier="disclosures",
        claim=claim,
        assumptions=[
            "The category revenue is nonnegative and entirely allocated to the focus and other segments.",
            "The other segment's category revenue cannot exceed its total segment revenue.",
            "Amounts use nearest-dollar rounding, encoded as inclusive intervals.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    before, after = build_snapshot(), build_snapshot(revised=True)
    comparison = usf.compare_disclosure_snapshots(before, after)
    print(before.replay().to_markdown())
    print("\n## After the illustrative revision\n")
    print(after.replay().to_markdown())
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "original.json").write_text(before.to_json(indent=2))
        (args.output / "revised.json").write_text(after.to_json(indent=2))
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        (args.output / "audit.md").write_text(after.replay().to_markdown())


if __name__ == "__main__":
    main()
