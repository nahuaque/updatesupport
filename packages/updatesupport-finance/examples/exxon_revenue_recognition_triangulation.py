"""Exxon Mobil disclosure-triangulation worked example.

Run from the repository root with:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_revenue_recognition_triangulation.py

Optionally write the Markdown report:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_revenue_recognition_triangulation.py \
        --output data/exxon_revenue_recognition_triangulation_report.md

The example uses values scraped from Exxon Mobil's Q1 2026 SEC 10-Q rendered
XBRL pages. It intentionally stores the values directly so the example is
reproducible without live SEC access.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import updatesupport as us
import updatesupport_finance as usf


FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/xom-20260331.htm"
)
SEGMENT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R26.htm"
)
REVENUE_RECOGNITION_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R27.htm"
)
GEOGRAPHY_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R28.htm"
)

OUTSIDE_ASC_TOTAL = 26_295.0
ASC606_TOTAL = 56_866.0
COMPANY_SALES_AND_OTHER_OPERATING_REVENUE = 83_161.0

SEGMENTS = (
    "upstream",
    "energy_products",
    "chemical_products",
    "specialty_products",
)
REGIONS = ("us", "non_us")
SEGMENT_REGION_SALES = {
    ("upstream", "us"): 7_265.0,
    ("upstream", "non_us"): 2_813.0,
    ("energy_products", "us"): 25_990.0,
    ("energy_products", "non_us"): 37_204.0,
    ("chemical_products", "us"): 1_970.0,
    ("chemical_products", "non_us"): 3_504.0,
    ("specialty_products", "us"): 1_372.0,
    ("specialty_products", "non_us"): 3_018.0,
}
GEOGRAPHY_SALES = {
    "us": 36_627.0,
    "non_us": 46_534.0,
}
OPERATING_SEGMENT_SALES = sum(SEGMENT_REGION_SALES.values())
RECONCILING_SALES = COMPANY_SALES_AND_OTHER_OPERATING_REVENUE - OPERATING_SEGMENT_SALES

BASELINE_TIER = "T0 outside-ASC total only"
CAPACITY_TIER = "T1 + disclosed segment/geography capacities"
GEOGRAPHY_TIER = "T2 + geographic sales capacities"


def build_spec() -> usf.DisclosureTriangulationSpec:
    """Build the Exxon revenue-recognition triangulation problem.

    Exxon discloses total sales and other operating revenue by segment and
    broad geography, but it discloses the ASC 606 / outside-ASC split only at
    the company level. The question is how much outside-ASC revenue could sit
    in each segment while respecting the disclosed segment/geography sales
    capacities.
    """

    variables = [
        usf.disclosure_variable(_cell_variable(segment, region), unit="$M")
        for segment, region in SEGMENT_REGION_SALES
    ]
    variables.append(
        usf.disclosure_variable(
            "outside_asc__reconciling__unallocated",
            unit="$M",
            description=(
                "Small company-level sales amount not assigned to the operating "
                "segment x geography cells in the rendered segment table."
            ),
        )
    )
    constraints = _constraints(variables)
    return usf.disclosure_triangulation_spec(
        title="Exxon Mobil Q1 2026 Revenue Recognition Triangulation",
        description=(
            "Bounds how Exxon Mobil's company-level revenue outside the scope "
            "of ASC 606 could be allocated across disclosed operating-segment "
            "and geography sales cells."
        ),
        variables=variables,
        constraints=constraints,
        targets=_targets(),
        tiers=[
            usf.disclosure_tier(
                BASELINE_TIER,
                ["outside_asc_total_q1_2026"],
                description=(
                    "Only the disclosed company-level outside-ASC revenue total "
                    "is active."
                ),
            ),
            usf.disclosure_tier(
                CAPACITY_TIER,
                [
                    constraint.name
                    for constraint in constraints
                    if constraint.kind != "geographic_sales_capacity"
                ],
                description=(
                    "Adds disclosed operating-segment/geography sales capacities. "
                    "Outside-ASC revenue assigned to a disclosed cell cannot "
                    "exceed that cell's total sales and other operating revenue."
                ),
            ),
            usf.disclosure_tier(
                GEOGRAPHY_TIER,
                [constraint.name for constraint in constraints],
                description=(
                    "Adds the separate U.S. and non-U.S. sales totals from the "
                    "geography disclosure."
                ),
            ),
        ],
    )


def build_report() -> usf.DisclosureTriangulationReport:
    """Solve the Exxon revenue-recognition triangulation problem."""

    return usf.triangulate_disclosure(build_spec())


def build_attribution_report(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "outside_asc_energy_products",
    tier: str = CAPACITY_TIER,
    group_by: str = "constraint",
) -> us.NamedLinearConstraintAttributionReport:
    """Rank which disclosures most narrow the headline interval."""

    if report is None:
        report = build_report()
    return usf.attribute_disclosure_constraints(
        report,
        target=target,
        tier=tier,
        group_by=group_by,
    )


def build_claim_audit(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "outside_asc_energy_products_share",
    tier: str = CAPACITY_TIER,
    lower_at_least: float = 20.0,
) -> us.NamedLinearClaimAudit:
    """Audit a bounded claim about Energy Products' outside-ASC share."""

    if report is None:
        report = build_report()
    claim = usf.disclosure_claim(
        target=target,
        tier=tier,
        lower_at_least=lower_at_least,
        label="Energy Products outside-ASC share lower-bound claim",
        description=(
            "This claim passes if every feasible allocation puts at least the "
            "specified percentage of outside-ASC revenue in Energy Products."
        ),
    )
    return claim.audit(report)


def build_audit_pack(
    report: us.NamedLinearFeasibilityReport | None = None,
) -> usf.DisclosureAuditPack:
    """Build the analyst-facing disclosure audit pack."""

    if report is None:
        report = build_report()
    claim = usf.disclosure_claim(
        target="outside_asc_energy_products_share",
        tier=CAPACITY_TIER,
        lower_at_least=20.0,
        label="Energy Products outside-ASC share lower-bound claim",
        description=(
            "This claim passes if every feasible allocation puts at least 20% "
            "of outside-ASC revenue in Energy Products."
        ),
        attribution_top=8,
        diagnostic_top=8,
    )
    return usf.disclosure_audit_pack(
        report,
        claim=claim,
        title="Exxon Mobil Q1 2026 Disclosure Audit Pack",
        sources=[
            {
                "label": "Sales and other operating revenue",
                "value": f"{COMPANY_SALES_AND_OTHER_OPERATING_REVENUE:,.0f} $M",
                "url": FILING_URL,
                "description": "Company-level sales and other operating revenue.",
            },
            {
                "label": "Revenue from contracts with customers",
                "value": f"{ASC606_TOTAL:,.0f} $M",
                "url": REVENUE_RECOGNITION_SOURCE_URL,
                "description": "Company-level ASC 606 revenue disclosure.",
            },
            {
                "label": "Revenue outside the scope of ASC 606",
                "value": f"{OUTSIDE_ASC_TOTAL:,.0f} $M",
                "url": REVENUE_RECOGNITION_SOURCE_URL,
                "description": "Company-level outside-ASC revenue disclosure.",
            },
            {
                "label": "Operating-segment x geography sales cells",
                "value": f"{OPERATING_SEGMENT_SALES:,.0f} $M",
                "url": SEGMENT_SOURCE_URL,
                "description": "Sales capacities by operating segment and geography.",
            },
            {
                "label": "U.S. / non-U.S. sales totals",
                "value": f"{sum(GEOGRAPHY_SALES.values()):,.0f} $M",
                "url": GEOGRAPHY_SOURCE_URL,
                "description": "Separate broad geography sales disclosure.",
            },
        ],
        assumptions=[
            "All modeled outside-ASC allocations are nonnegative.",
            "Outside-ASC revenue assigned to a disclosed cell cannot exceed "
            "that cell's total sales and other operating revenue.",
            "The small difference between company-level sales and operating-"
            "segment x geography sales is modeled as reconciling capacity.",
            "Values are in USD millions and are taken from Exxon Mobil's "
            "rendered SEC XBRL pages for the three months ended March 31, 2026.",
        ],
        reviewer_notes=[
            "Energy Products outside-ASC revenue: 6,328 $M to 26,295 $M.",
            "Energy Products share of outside-ASC revenue: 24.1% to 100.0%.",
            "The exact segment allocation remains unidentified; this is a "
            "filing-implied feasibility bound, not a point estimate.",
        ],
        attribution_top=8,
        diagnostic_top=8,
    )


def width_reduction_rows(
    report: us.NamedLinearFeasibilityReport,
    *,
    baseline_tier: str = BASELINE_TIER,
) -> list[dict[str, Any]]:
    """Summarize interval narrowing relative to the baseline tier."""

    rows: list[dict[str, Any]] = []
    for interval in report.intervals:
        baseline = report.interval(
            target=interval.target,
            scenario=baseline_tier,
        )
        reduction = None
        reduction_percent = None
        if baseline.width is not None and interval.width is not None:
            reduction = baseline.width - interval.width
            reduction_percent = (
                None if baseline.width <= 0.0 else 100.0 * reduction / baseline.width
            )
        rows.append(
            {
                "target": interval.target,
                "tier": interval.scenario,
                "lower": interval.scaled_lower,
                "upper": interval.scaled_upper,
                "width": interval.scaled_width,
                "reduction_vs_baseline": reduction,
                "reduction_percent_vs_baseline": reduction_percent,
                "status": interval.status,
            }
        )
    return rows


def render_markdown(report: us.NamedLinearFeasibilityReport | None = None) -> str:
    """Render a tutorial-style Markdown report for the example."""

    if report is None:
        report = build_report()
    return build_audit_pack(report).to_markdown()


def _constraints(
    variables: Iterable[usf.DisclosureVariable],
) -> list[usf.DisclosureConstraint]:
    all_variable_names = [variable.name for variable in variables]
    constraints = [
        usf.exact_disclosure_constraint(
            "outside_asc_total_q1_2026",
            {name: 1.0 for name in all_variable_names},
            OUTSIDE_ASC_TOTAL,
            category="revenue_recognition_disclosure",
            provenance=(
                "Exxon Mobil Q1 2026 10-Q revenue recognition table: "
                f"{REVENUE_RECOGNITION_SOURCE_URL}"
            ),
            description=(
                "Company-level revenue outside the scope of ASC 606 for the "
                "three months ended March 31, 2026."
            ),
            verified=True,
        )
    ]
    for (segment, region), sales in SEGMENT_REGION_SALES.items():
        constraints.append(
            usf.interval_disclosure_constraint(
                f"capacity_{segment}_{region}_sales",
                _cell_variable(segment, region),
                upper=sales,
                category="segment_geography_sales_capacity",
                provenance=(
                    f"Exxon Mobil Q1 2026 10-Q segment table: {SEGMENT_SOURCE_URL}"
                ),
                description=(
                    "Outside-ASC revenue allocated to this disclosed cell "
                    "cannot exceed total sales and other operating revenue for "
                    "the same cell."
                ),
                verified=True,
            )
        )
    constraints.append(
        usf.interval_disclosure_constraint(
            "capacity_reconciling_sales",
            "outside_asc__reconciling__unallocated",
            upper=RECONCILING_SALES,
            category="reconciling_sales_capacity",
            provenance=(
                "Company sales less operating-segment x geography sales in "
                f"Exxon Mobil Q1 2026 10-Q: {SEGMENT_SOURCE_URL}"
            ),
            description=(
                "Small reconciling capacity between company-level sales and "
                "the operating-segment x geography cells."
            ),
            verified=True,
        )
    )
    for region, sales in GEOGRAPHY_SALES.items():
        constraints.append(
            usf.interval_disclosure_constraint(
                f"capacity_{region}_geographic_sales",
                _region_coefficients(region),
                upper=sales,
                category="geographic_sales_capacity",
                provenance=(
                    f"Exxon Mobil Q1 2026 10-Q geography table: {GEOGRAPHY_SOURCE_URL}"
                ),
                description=(
                    "Outside-ASC revenue allocated to this broad geography "
                    "cannot exceed reported sales and other operating revenue "
                    "for the geography."
                ),
                verified=True,
            )
        )
    return constraints


def _targets() -> list[usf.DisclosureTarget]:
    targets: list[usf.DisclosureTarget] = []
    for segment in SEGMENTS:
        label = segment.replace("_", " ")
        targets.append(
            usf.disclosure_target(
                f"outside_asc_{segment}",
                _segment_coefficients(segment),
                label=f"Outside-ASC revenue: {label}",
                unit="$M",
            )
        )
    targets.append(
        usf.disclosure_target(
            "outside_asc_energy_products_share",
            {
                name: coefficient * 100.0 / OUTSIDE_ASC_TOTAL
                for name, coefficient in _segment_coefficients(
                    "energy_products"
                ).items()
            },
            label="Outside-ASC share: energy products",
            unit="%",
        )
    )
    for region in REGIONS:
        targets.append(
            usf.disclosure_target(
                f"outside_asc_{region}",
                _region_coefficients(region),
                label=f"Outside-ASC revenue: {region}",
                unit="$M",
            )
        )
    targets.append(
        usf.disclosure_target(
            "outside_asc_reconciling",
            "outside_asc__reconciling__unallocated",
            label="Outside-ASC reconciling revenue",
            unit="$M",
        )
    )
    return targets


def _cell_variable(segment: str, region: str) -> str:
    return f"outside_asc__{segment}__{region}"


def _segment_coefficients(segment: str) -> dict[str, float]:
    return {_cell_variable(segment, region): 1.0 for region in REGIONS}


def _region_coefficients(region: str) -> dict[str, float]:
    return {_cell_variable(segment, region): 1.0 for segment in SEGMENTS}


def _headline_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    targets = {
        "outside_asc_energy_products",
        "outside_asc_energy_products_share",
        "outside_asc_upstream",
        "outside_asc_chemical_products",
        "outside_asc_specialty_products",
    }
    return [row for row in rows if row["target"] in targets]


def _width_reduction_table(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Target | Tier | Lower | Upper | Width | Reduction vs T0 | Reduction % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["target"]),
                    str(row["tier"]),
                    _format_optional(row["lower"]),
                    _format_optional(row["upper"]),
                    _format_optional(row["width"]),
                    _format_optional(row["reduction_vs_baseline"]),
                    _format_optional(row["reduction_percent_vs_baseline"], suffix="%"),
                ]
            )
            + " |"
        )
    return lines


def _format_optional(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 1_000:
        return f"{numeric:,.0f}{suffix}"
    return f"{numeric:.4g}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Exxon revenue-recognition triangulation report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    args = parser.parse_args()

    markdown = render_markdown()
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
