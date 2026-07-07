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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import updatesupport as us
import updatesupport_finance as usf


Q1_2026_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/xom-20260331.htm"
)
Q1_2026_SEGMENT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R26.htm"
)
Q1_2026_REVENUE_RECOGNITION_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R27.htm"
)
Q1_2026_GEOGRAPHY_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R28.htm"
)
FY2025_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm"
)
FY2025_SEGMENT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R56.htm"
)
FY2025_REVENUE_RECOGNITION_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R57.htm"
)
FY2025_GEOGRAPHY_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R58.htm"
)

SEGMENTS = (
    "upstream",
    "energy_products",
    "chemical_products",
    "specialty_products",
)
REGIONS = ("us", "non_us")

BASELINE_TIER = "T0 outside-ASC total only"
CAPACITY_TIER = "T1 + disclosed segment/geography capacities"
GEOGRAPHY_TIER = "T2 + geographic sales capacities"
DEFAULT_PERIOD = "q1_2026"


@dataclass(frozen=True)
class ExxonRevenuePeriod:
    """One scraped Exxon disclosure period used by the example."""

    key: str
    label: str
    filing_url: str
    segment_url: str
    revenue_recognition_url: str
    geography_url: str
    outside_asc_total: float
    asc606_total: float
    company_sales: float
    segment_region_sales: Mapping[tuple[str, str], float]
    geography_sales: Mapping[str, float]

    @property
    def operating_segment_sales(self) -> float:
        return sum(self.segment_region_sales.values())

    @property
    def reconciling_sales(self) -> float:
        return self.company_sales - self.operating_segment_sales

    @property
    def energy_products_minimum(self) -> float:
        non_energy_capacity = sum(
            value
            for (segment, _region), value in self.segment_region_sales.items()
            if segment != "energy_products"
        )
        return max(
            0.0, self.outside_asc_total - non_energy_capacity - self.reconciling_sales
        )

    @property
    def energy_products_minimum_share(self) -> float:
        return 100.0 * self.energy_products_minimum / self.outside_asc_total


EXXON_REVENUE_PERIODS = {
    "fy2025": ExxonRevenuePeriod(
        key="fy2025",
        label="FY2025",
        filing_url=FY2025_FILING_URL,
        segment_url=FY2025_SEGMENT_SOURCE_URL,
        revenue_recognition_url=FY2025_REVENUE_RECOGNITION_SOURCE_URL,
        geography_url=FY2025_GEOGRAPHY_SOURCE_URL,
        outside_asc_total=96_996.0,
        asc606_total=226_909.0,
        company_sales=323_905.0,
        segment_region_sales={
            ("upstream", "us"): 25_396.0,
            ("upstream", "non_us"): 13_993.0,
            ("energy_products", "us"): 99_073.0,
            ("energy_products", "non_us"): 145_378.0,
            ("chemical_products", "us"): 7_594.0,
            ("chemical_products", "non_us"): 14_615.0,
            ("specialty_products", "us"): 5_502.0,
            ("specialty_products", "non_us"): 12_269.0,
        },
        geography_sales={
            "us": 137_639.0,
            "non_us": 186_266.0,
        },
    ),
    "q1_2026": ExxonRevenuePeriod(
        key="q1_2026",
        label="Q1 2026",
        filing_url=Q1_2026_FILING_URL,
        segment_url=Q1_2026_SEGMENT_SOURCE_URL,
        revenue_recognition_url=Q1_2026_REVENUE_RECOGNITION_SOURCE_URL,
        geography_url=Q1_2026_GEOGRAPHY_SOURCE_URL,
        outside_asc_total=26_295.0,
        asc606_total=56_866.0,
        company_sales=83_161.0,
        segment_region_sales={
            ("upstream", "us"): 7_265.0,
            ("upstream", "non_us"): 2_813.0,
            ("energy_products", "us"): 25_990.0,
            ("energy_products", "non_us"): 37_204.0,
            ("chemical_products", "us"): 1_970.0,
            ("chemical_products", "non_us"): 3_504.0,
            ("specialty_products", "us"): 1_372.0,
            ("specialty_products", "non_us"): 3_018.0,
        },
        geography_sales={
            "us": 36_627.0,
            "non_us": 46_534.0,
        },
    ),
}


# Backward-compatible constants for the original Q1 2026 single-period example.
OUTSIDE_ASC_TOTAL = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].outside_asc_total
ASC606_TOTAL = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].asc606_total
COMPANY_SALES_AND_OTHER_OPERATING_REVENUE = EXXON_REVENUE_PERIODS[
    DEFAULT_PERIOD
].company_sales
SEGMENT_REGION_SALES = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].segment_region_sales
GEOGRAPHY_SALES = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].geography_sales
OPERATING_SEGMENT_SALES = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].operating_segment_sales
RECONCILING_SALES = EXXON_REVENUE_PERIODS[DEFAULT_PERIOD].reconciling_sales


def build_spec(
    period: str | ExxonRevenuePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationSpec:
    """Build the Exxon revenue-recognition triangulation problem.

    Exxon discloses total sales and other operating revenue by segment and
    broad geography, but it discloses the ASC 606 / outside-ASC split only at
    the company level. The question is how much outside-ASC revenue could sit
    in each segment while respecting the disclosed segment/geography sales
    capacities.
    """

    period_row = _period(period)
    variables = [
        usf.disclosure_variable(_cell_variable(segment, region), unit="$M")
        for segment, region in period_row.segment_region_sales
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
    constraints = _constraints(variables, period=period_row)
    return usf.disclosure_triangulation_spec(
        title=f"Exxon Mobil {period_row.label} Revenue Recognition Triangulation",
        description=(
            "Bounds how Exxon Mobil's company-level revenue outside the scope "
            "of ASC 606 could be allocated across disclosed operating-segment "
            "and geography sales cells."
        ),
        variables=variables,
        constraints=constraints,
        targets=_targets(period_row),
        tiers=[
            usf.disclosure_tier(
                BASELINE_TIER,
                [_outside_asc_total_constraint(period_row)],
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


def build_report(
    period: str | ExxonRevenuePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationReport:
    """Solve the Exxon revenue-recognition triangulation problem."""

    return usf.triangulate_disclosure(build_spec(period))


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
    *,
    period: str | ExxonRevenuePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureAuditPack:
    """Build the analyst-facing disclosure audit pack."""

    period_row = _period(period)
    if report is None:
        report = build_report(period_row)
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
        title=f"Exxon Mobil {period_row.label} Disclosure Audit Pack",
        sources=[
            {
                "label": "Sales and other operating revenue",
                "value": f"{period_row.company_sales:,.0f} $M",
                "url": period_row.filing_url,
                "description": "Company-level sales and other operating revenue.",
            },
            {
                "label": "Revenue from contracts with customers",
                "value": f"{period_row.asc606_total:,.0f} $M",
                "url": period_row.revenue_recognition_url,
                "description": "Company-level ASC 606 revenue disclosure.",
            },
            {
                "label": "Revenue outside the scope of ASC 606",
                "value": f"{period_row.outside_asc_total:,.0f} $M",
                "url": period_row.revenue_recognition_url,
                "description": "Company-level outside-ASC revenue disclosure.",
            },
            {
                "label": "Operating-segment x geography sales cells",
                "value": f"{period_row.operating_segment_sales:,.0f} $M",
                "url": period_row.segment_url,
                "description": "Sales capacities by operating segment and geography.",
            },
            {
                "label": "U.S. / non-U.S. sales totals",
                "value": f"{sum(period_row.geography_sales.values()):,.0f} $M",
                "url": period_row.geography_url,
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
            f"rendered SEC XBRL pages for {period_row.label}.",
        ],
        reviewer_notes=[
            "Energy Products outside-ASC revenue: "
            f"{period_row.energy_products_minimum:,.0f} $M to "
            f"{period_row.outside_asc_total:,.0f} $M.",
            "Energy Products share of outside-ASC revenue: "
            f"{period_row.energy_products_minimum_share:.1f}% to 100.0%.",
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


def render_panel_markdown(
    periods: Iterable[str | ExxonRevenuePeriod] = ("fy2025", "q1_2026"),
) -> str:
    """Render a cross-period Exxon disclosure-triangulation panel."""

    period_rows = [_period(period) for period in periods]
    packs = [build_audit_pack(period=period) for period in period_rows]
    lines = [
        "# Exxon Mobil Revenue Recognition Disclosure Triangulation Panel",
        "",
        "This panel repeats the same disclosure-audit question across Exxon "
        "Mobil's FY2025 10-K and Q1 2026 10-Q:",
        "",
        "> The company discloses revenue outside the scope of ASC 606 at the "
        "company level. Given disclosed operating-segment and geography sales "
        "capacities, how much must be Energy Products revenue?",
        "",
        "## Cross-Period Readout",
        "",
    ]
    lines.extend(_period_readout_table(period_rows, packs))
    lines.extend(
        [
            "",
            "The exact segment allocation remains unidentified in both periods. "
            "The useful result is the lower bound: the disclosed non-Energy-"
            "Products cells are too small to absorb all outside-ASC revenue.",
        ]
    )
    for pack in packs:
        lines.extend(["", f"## {pack.audit_title}", ""])
        pack_lines = pack.to_markdown().splitlines()
        if pack_lines and pack_lines[0].startswith("# "):
            pack_lines = pack_lines[2:]
        lines.extend(pack_lines)
    return "\n".join(lines)


def _constraints(
    variables: Iterable[usf.DisclosureVariable],
    *,
    period: ExxonRevenuePeriod,
) -> list[usf.DisclosureConstraint]:
    all_variable_names = [variable.name for variable in variables]
    constraints = [
        usf.exact_disclosure_constraint(
            _outside_asc_total_constraint(period),
            {name: 1.0 for name in all_variable_names},
            period.outside_asc_total,
            category="revenue_recognition_disclosure",
            provenance=(
                f"Exxon Mobil {period.label} revenue recognition table: "
                f"{period.revenue_recognition_url}"
            ),
            description=(
                "Company-level revenue outside the scope of ASC 606 for the "
                f"period {period.label}."
            ),
            verified=True,
        )
    ]
    for (segment, region), sales in period.segment_region_sales.items():
        constraints.append(
            usf.interval_disclosure_constraint(
                f"capacity_{segment}_{region}_sales",
                _cell_variable(segment, region),
                upper=sales,
                category="segment_geography_sales_capacity",
                provenance=(
                    f"Exxon Mobil {period.label} segment table: {period.segment_url}"
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
            upper=period.reconciling_sales,
            category="reconciling_sales_capacity",
            provenance=(
                "Company sales less operating-segment x geography sales in "
                f"Exxon Mobil {period.label}: {period.segment_url}"
            ),
            description=(
                "Small reconciling capacity between company-level sales and "
                "the operating-segment x geography cells."
            ),
            verified=True,
        )
    )
    for region, sales in period.geography_sales.items():
        constraints.append(
            usf.interval_disclosure_constraint(
                f"capacity_{region}_geographic_sales",
                _region_coefficients(region),
                upper=sales,
                category="geographic_sales_capacity",
                provenance=(
                    f"Exxon Mobil {period.label} geography table: {period.geography_url}"
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


def _targets(period: ExxonRevenuePeriod) -> list[usf.DisclosureTarget]:
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
                name: coefficient * 100.0 / period.outside_asc_total
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


def _outside_asc_total_constraint(period: ExxonRevenuePeriod) -> str:
    return f"outside_asc_total_{period.key}"


def _segment_coefficients(segment: str) -> dict[str, float]:
    return {_cell_variable(segment, region): 1.0 for region in REGIONS}


def _region_coefficients(region: str) -> dict[str, float]:
    return {_cell_variable(segment, region): 1.0 for segment in SEGMENTS}


def _period(period: str | ExxonRevenuePeriod) -> ExxonRevenuePeriod:
    if isinstance(period, ExxonRevenuePeriod):
        return period
    try:
        return EXXON_REVENUE_PERIODS[period]
    except KeyError as exc:
        raise KeyError(f"unknown Exxon revenue period: {period!r}") from exc


def _period_readout_table(
    periods: Iterable[ExxonRevenuePeriod],
    packs: Iterable[usf.DisclosureAuditPack],
) -> list[str]:
    lines = [
        "| Period | Outside-ASC total | Energy Products lower bound | Lower-bound share | Claim verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for period, pack in zip(periods, packs, strict=True):
        interval = pack.interval
        verdict = "n/a" if pack.claim_audit is None else pack.claim_audit.verdict
        lines.append(
            "| "
            + " | ".join(
                [
                    period.label,
                    f"{period.outside_asc_total:,.0f} $M",
                    f"{period.energy_products_minimum:,.0f} $M",
                    f"{interval.scaled_lower:.1f}%",
                    verdict,
                ]
            )
            + " |"
        )
    return lines


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

    markdown = render_panel_markdown()
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
