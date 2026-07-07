"""Exxon Mobil capex-capacity disclosure-triangulation worked example.

Run from the repository root with:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_capex_capacity_triangulation.py

Optionally write the Markdown report:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_capex_capacity_triangulation.py \
        --output data/exxon_capex_capacity_triangulation_report.md

This is a generic disclosure-capacity model. The cash-flow statement reports a
company-level cash outflow for additions to property, plant and equipment. The
segment note reports additions to property, plant and equipment by operating
segment and geography. The example asks how much of the company-level cash PP&E
additions must be attributable to Upstream if each segment/geography cell can
absorb no more than its disclosed PP&E additions.

Values are in USD millions and are stored directly so the example is
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


FY2025_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm"
)
FY2025_SEGMENT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R56.htm"
)
FY2025_CASH_FLOW_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R7.htm"
)
Q1_2026_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/xom-20260331.htm"
)
Q1_2026_SEGMENT_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R26.htm"
)
Q1_2026_CASH_FLOW_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R6.htm"
)

SEGMENTS = (
    "upstream",
    "energy_products",
    "chemical_products",
    "specialty_products",
    "corporate",
)
OPERATING_SEGMENTS = SEGMENTS[:-1]
REGIONS = ("us", "non_us")

BASELINE_TIER = "T0 cash PP&E additions total only"
CAPACITY_TIER = "T1 + segment/geography PP&E additions capacities"
DEFAULT_PERIOD = "fy2025"


@dataclass(frozen=True)
class ExxonCapexPeriod:
    """One scraped Exxon capex-capacity disclosure period."""

    key: str
    label: str
    filing_url: str
    segment_url: str
    cash_flow_url: str
    cash_ppe_additions: float
    segment_region_ppe_additions: Mapping[tuple[str, str], float]
    corporate_ppe_additions: float

    @property
    def operating_segment_ppe_additions(self) -> float:
        return sum(self.segment_region_ppe_additions.values())

    @property
    def segment_note_ppe_additions(self) -> float:
        return self.operating_segment_ppe_additions + self.corporate_ppe_additions

    @property
    def segment_note_cushion(self) -> float:
        return self.segment_note_ppe_additions - self.cash_ppe_additions

    @property
    def upstream_capacity(self) -> float:
        return sum(
            value
            for (segment, _region), value in self.segment_region_ppe_additions.items()
            if segment == "upstream"
        )

    @property
    def non_upstream_capacity(self) -> float:
        return self.segment_note_ppe_additions - self.upstream_capacity

    @property
    def upstream_minimum(self) -> float:
        return max(0.0, self.cash_ppe_additions - self.non_upstream_capacity)

    @property
    def upstream_minimum_share(self) -> float:
        return 100.0 * self.upstream_minimum / self.cash_ppe_additions

    @property
    def upstream_maximum_share(self) -> float:
        return 100.0 * self.upstream_capacity / self.cash_ppe_additions


EXXON_CAPEX_PERIODS = {
    "fy2025": ExxonCapexPeriod(
        key="fy2025",
        label="FY2025",
        filing_url=FY2025_FILING_URL,
        segment_url=FY2025_SEGMENT_SOURCE_URL,
        cash_flow_url=FY2025_CASH_FLOW_URL,
        cash_ppe_additions=28_358.0,
        segment_region_ppe_additions={
            ("upstream", "us"): 15_872.0,
            ("upstream", "non_us"): 9_490.0,
            ("energy_products", "us"): 703.0,
            ("energy_products", "non_us"): 1_251.0,
            ("chemical_products", "us"): 800.0,
            ("chemical_products", "non_us"): 522.0,
            ("specialty_products", "us"): 368.0,
            ("specialty_products", "non_us"): 227.0,
        },
        corporate_ppe_additions=2_243.0,
    ),
    "q1_2026": ExxonCapexPeriod(
        key="q1_2026",
        label="Q1 2026",
        filing_url=Q1_2026_FILING_URL,
        segment_url=Q1_2026_SEGMENT_SOURCE_URL,
        cash_flow_url=Q1_2026_CASH_FLOW_URL,
        cash_ppe_additions=6_470.0,
        segment_region_ppe_additions={
            ("upstream", "us"): 3_275.0,
            ("upstream", "non_us"): 1_885.0,
            ("energy_products", "us"): 851.0,
            ("energy_products", "non_us"): 166.0,
            ("chemical_products", "us"): 152.0,
            ("chemical_products", "non_us"): 25.0,
            ("specialty_products", "us"): 34.0,
            ("specialty_products", "non_us"): 19.0,
        },
        corporate_ppe_additions=347.0,
    ),
}


def build_spec(
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationSpec:
    """Build the Exxon capex-capacity triangulation problem."""

    period_row = _period(period)
    variables = [
        usf.disclosure_variable(_cell_variable(segment, region), unit="$M")
        for segment, region in period_row.segment_region_ppe_additions
    ]
    variables.append(
        usf.disclosure_variable(
            "cash_ppe_additions__corporate__unallocated",
            unit="$M",
            description=(
                "Cash PP&E additions allocated to disclosed corporate "
                "nonsegment PP&E additions capacity."
            ),
        )
    )
    constraints = _constraints(variables, period=period_row)
    return usf.disclosure_triangulation_spec(
        title=f"Exxon Mobil {period_row.label} Capex Capacity Triangulation",
        description=(
            "Bounds how Exxon Mobil's company-level cash additions to "
            "property, plant and equipment could be allocated across disclosed "
            "segment and geography PP&E-addition capacity cells."
        ),
        variables=variables,
        constraints=constraints,
        targets=_targets(period_row),
        tiers=[
            usf.disclosure_tier(
                BASELINE_TIER,
                [_cash_ppe_total_constraint(period_row)],
                description=(
                    "Only the disclosed company-level cash PP&E additions "
                    "total is active."
                ),
            ),
            usf.disclosure_tier(
                CAPACITY_TIER,
                [constraint.name for constraint in constraints],
                description=(
                    "Adds disclosed segment/geography PP&E additions as cell "
                    "capacities."
                ),
            ),
        ],
    )


def build_report(
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationReport:
    """Solve the Exxon capex-capacity triangulation problem."""

    return usf.triangulate_disclosure(build_spec(period))


def build_attribution_report(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "cash_ppe_additions_upstream_share",
    tier: str = CAPACITY_TIER,
    group_by: str = "constraint",
) -> us.NamedLinearConstraintAttributionReport:
    """Rank which disclosures narrow the headline capex interval."""

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
    target: str = "cash_ppe_additions_upstream_share",
    tier: str = CAPACITY_TIER,
    lower_at_least: float = 75.0,
) -> us.NamedLinearClaimAudit:
    """Audit whether Upstream must be at least 75% of cash PP&E additions."""

    if report is None:
        report = build_report()
    claim = usf.disclosure_claim(
        target=target,
        tier=tier,
        lower_at_least=lower_at_least,
        label="Upstream cash PP&E additions share lower-bound claim",
        description=(
            "This claim passes if every feasible allocation puts at least the "
            "specified percentage of cash PP&E additions in Upstream."
        ),
        attribution_top=8,
        diagnostic_top=8,
    )
    return claim.audit(report)


def build_audit_pack(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
) -> usf.DisclosureAuditPack:
    """Build the analyst-facing capex-capacity audit pack."""

    period_row = _period(period)
    if report is None:
        report = build_report(period_row)
    claim = usf.disclosure_claim(
        target="cash_ppe_additions_upstream_share",
        tier=CAPACITY_TIER,
        lower_at_least=75.0,
        label="Upstream cash PP&E additions share lower-bound claim",
        description=(
            "This claim passes if every feasible allocation puts at least 75% "
            "of company-level cash PP&E additions in Upstream."
        ),
        attribution_top=8,
        diagnostic_top=8,
    )
    return usf.disclosure_audit_pack(
        report,
        claim=claim,
        title=f"Exxon Mobil {period_row.label} Capex Capacity Disclosure Audit Pack",
        sources=[
            {
                "label": "Cash additions to PP&E",
                "value": f"{period_row.cash_ppe_additions:,.0f} $M",
                "url": period_row.cash_flow_url,
                "description": "Company-level cash-flow investing disclosure.",
            },
            {
                "label": "Operating-segment PP&E additions capacity",
                "value": f"{period_row.operating_segment_ppe_additions:,.0f} $M",
                "url": period_row.segment_url,
                "description": (
                    "Disclosed PP&E additions by operating segment and geography."
                ),
            },
            {
                "label": "Corporate PP&E additions capacity",
                "value": f"{period_row.corporate_ppe_additions:,.0f} $M",
                "url": period_row.segment_url,
                "description": "Disclosed corporate nonsegment PP&E additions.",
            },
        ],
        assumptions=[
            "Cash PP&E additions assigned to a disclosed cell are nonnegative.",
            "Cash PP&E additions assigned to a disclosed cell cannot exceed "
            "that cell's disclosed PP&E additions capacity.",
            "The segment-note PP&E additions total is treated as a capacity "
            "for the cash-flow PP&E additions total, not as an exact cash-flow "
            "classification.",
            "Values are in USD millions and are taken from Exxon Mobil's "
            f"rendered SEC XBRL pages for {period_row.label}.",
        ],
        reviewer_notes=[
            "Upstream cash PP&E additions: "
            f"{period_row.upstream_minimum:,.0f} $M to "
            f"{period_row.upstream_capacity:,.0f} $M.",
            "Upstream share of cash PP&E additions: "
            f"{period_row.upstream_minimum_share:.1f}% to "
            f"{period_row.upstream_maximum_share:.1f}%.",
            "The exact allocation remains unidentified; this is a filing-"
            "implied feasibility bound under the capacity assumption.",
        ],
        attribution_top=8,
        diagnostic_top=8,
    )


def frontier_rows(
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
) -> list[dict[str, Any]]:
    """Build neutral hidden-cell rows for public-representation frontier search.

    The exact cash allocation is unidentified, so the frontier uses a neutral
    capacity-proportional allocation as its observed hidden mix. The frontier
    then asks which public disclosure refinements make the Upstream-share
    target stable under recomposition inside the public buckets.
    """

    period_row = _period(period)
    rows: list[dict[str, Any]] = []
    for (segment, region), capacity in period_row.segment_region_ppe_additions.items():
        allocation = (
            period_row.cash_ppe_additions
            * capacity
            / period_row.segment_note_ppe_additions
        )
        rows.append(
            {
                "period": period_row.label,
                "segment": segment,
                "region": region,
                "disclosure_cell": f"{segment}:{region}",
                "neutral_allocation": allocation,
                "is_upstream": 1.0 if segment == "upstream" else 0.0,
            }
        )
    rows.append(
        {
            "period": period_row.label,
            "segment": "corporate",
            "region": "unallocated",
            "disclosure_cell": "corporate:unallocated",
            "neutral_allocation": (
                period_row.cash_ppe_additions
                * period_row.corporate_ppe_additions
                / period_row.segment_note_ppe_additions
            ),
            "is_upstream": 0.0,
        }
    )
    return rows


def build_frontier(
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
    *,
    q_presets: Iterable[Any] = ("saturated",),
    ambiguity_limit: float = 0.05,
) -> us.PublicRepresentationFrontier:
    """Search disclosure refinements for stable Upstream-share reporting."""

    period_row = _period(period)
    return us.public_representation_frontier(
        frontier_rows(period_row),
        base_public=["period"],
        hidden=["period", "segment", "region"],
        target="is_upstream",
        weight="neutral_allocation",
        candidate_refinements=["region", "segment"],
        q_presets=tuple(q_presets),
        ambiguity_limit=ambiguity_limit,
        min_cell_weight=0.0,
        title=f"Exxon Mobil {period_row.label} Capex Disclosure Refinement Frontier",
    )


def width_reduction_rows(
    report: us.NamedLinearFeasibilityReport,
    *,
    baseline_tier: str = BASELINE_TIER,
) -> list[dict[str, Any]]:
    """Summarize interval narrowing relative to the baseline tier."""

    rows: list[dict[str, Any]] = []
    for interval in report.intervals:
        baseline = report.interval(target=interval.target, scenario=baseline_tier)
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
    """Render a Markdown audit pack for the default FY2025 example."""

    if report is None:
        report = build_report()
    return build_audit_pack(report).to_markdown()


def render_panel_markdown(
    periods: Iterable[str | ExxonCapexPeriod] = ("fy2025", "q1_2026"),
) -> str:
    """Render a cross-period Exxon capex-capacity panel."""

    period_rows = [_period(period) for period in periods]
    packs = [build_audit_pack(period=period) for period in period_rows]
    lines = [
        "# Exxon Mobil Capex Capacity Disclosure Triangulation Panel",
        "",
        "This panel repeats the same disclosure-audit question across Exxon "
        "Mobil's FY2025 10-K and Q1 2026 10-Q:",
        "",
        "> Given company-level cash additions to property, plant and equipment "
        "and disclosed segment/geography PP&E-addition capacities, how much "
        "cash PP&E spending must be Upstream?",
        "",
        "This is not a direct subtraction from a single table. The lower bound "
        "comes from the fact that non-Upstream and corporate capacity cells are "
        "too small to absorb all company-level cash PP&E additions.",
        "",
        "## Cross-Period Readout",
        "",
    ]
    lines.extend(_period_readout_table(period_rows, packs))
    lines.extend(
        [
            "",
            "The exact allocation remains unidentified in both periods. The "
            "filings nevertheless force a high Upstream floor under the stated "
            "capacity interpretation.",
        ]
    )
    lines.extend(["", "## Disclosure Refinement Frontier", ""])
    lines.extend(_frontier_summary_table(period_rows))
    for pack in packs:
        lines.extend(["", f"## {pack.audit_title}", ""])
        pack_lines = pack.to_markdown().splitlines()
        if pack_lines and pack_lines[0].startswith("# "):
            pack_lines = pack_lines[2:]
        lines.extend(pack_lines)
    return "\n".join(lines)


def render_frontier_markdown(
    frontier: us.PublicRepresentationFrontier | None = None,
    *,
    period: str | ExxonCapexPeriod = DEFAULT_PERIOD,
) -> str:
    """Render a standalone disclosure-refinement frontier report."""

    if frontier is None:
        frontier = build_frontier(period)
    lines = [
        frontier.to_markdown(),
        "",
        "## Disclosure Interpretation",
        "",
        "The frontier treats the capacity-proportional allocation as a neutral "
        "baseline because the exact cash PP&E allocation is not disclosed. "
        "It then asks whether a coarser public report would still determine "
        "the Upstream-share target under hidden recomposition.",
        "",
        "A segment refinement is the useful public disclosure: once segment is "
        "public, the Upstream indicator is constant inside each public fiber. "
        "Region alone does not settle the target because each region still "
        "contains both Upstream and non-Upstream cells.",
    ]
    return "\n".join(lines)


def _constraints(
    variables: Iterable[usf.DisclosureVariable],
    *,
    period: ExxonCapexPeriod,
) -> list[usf.DisclosureConstraint]:
    all_variable_names = [variable.name for variable in variables]
    constraints = [
        usf.exact_disclosure_constraint(
            _cash_ppe_total_constraint(period),
            {name: 1.0 for name in all_variable_names},
            period.cash_ppe_additions,
            category="cash_flow_disclosure",
            provenance=(
                f"Exxon Mobil {period.label} cash-flow statement: "
                f"{period.cash_flow_url}"
            ),
            description=(
                "Company-level cash additions to property, plant and equipment."
            ),
            verified=True,
        )
    ]
    for (segment, region), capacity in period.segment_region_ppe_additions.items():
        constraints.append(
            usf.interval_disclosure_constraint(
                f"capacity_{segment}_{region}_ppe_additions",
                _cell_variable(segment, region),
                upper=capacity,
                category="segment_geography_ppe_additions_capacity",
                provenance=(
                    f"Exxon Mobil {period.label} segment table: {period.segment_url}"
                ),
                description=(
                    "Cash PP&E additions allocated to this disclosed cell "
                    "cannot exceed total PP&E additions disclosed for the same "
                    "cell."
                ),
                verified=True,
            )
        )
    constraints.append(
        usf.interval_disclosure_constraint(
            "capacity_corporate_ppe_additions",
            "cash_ppe_additions__corporate__unallocated",
            upper=period.corporate_ppe_additions,
            category="corporate_ppe_additions_capacity",
            provenance=(
                f"Exxon Mobil {period.label} segment table: {period.segment_url}"
            ),
            description=(
                "Cash PP&E additions allocated to corporate nonsegment cannot "
                "exceed disclosed corporate PP&E additions."
            ),
            verified=True,
        )
    )
    return constraints


def _targets(period: ExxonCapexPeriod) -> list[usf.DisclosureTarget]:
    targets: list[usf.DisclosureTarget] = []
    for segment in SEGMENTS:
        label = segment.replace("_", " ")
        targets.append(
            usf.disclosure_target(
                f"cash_ppe_additions_{segment}",
                _segment_coefficients(segment),
                label=f"Cash PP&E additions: {label}",
                unit="$M",
            )
        )
        targets.append(
            usf.disclosure_target(
                f"cash_ppe_additions_{segment}_share",
                {
                    name: coefficient * 100.0 / period.cash_ppe_additions
                    for name, coefficient in _segment_coefficients(segment).items()
                },
                label=f"Cash PP&E additions share: {label}",
                unit="%",
            )
        )
    for region in REGIONS:
        targets.append(
            usf.disclosure_target(
                f"cash_ppe_additions_{region}",
                _region_coefficients(region),
                label=f"Cash PP&E additions: {region}",
                unit="$M",
            )
        )
    return targets


def _cell_variable(segment: str, region: str) -> str:
    return f"cash_ppe_additions__{segment}__{region}"


def _cash_ppe_total_constraint(period: ExxonCapexPeriod) -> str:
    return f"cash_ppe_additions_total_{period.key}"


def _segment_coefficients(segment: str) -> dict[str, float]:
    if segment == "corporate":
        return {"cash_ppe_additions__corporate__unallocated": 1.0}
    return {_cell_variable(segment, region): 1.0 for region in REGIONS}


def _region_coefficients(region: str) -> dict[str, float]:
    return {_cell_variable(segment, region): 1.0 for segment in OPERATING_SEGMENTS}


def _period(period: str | ExxonCapexPeriod) -> ExxonCapexPeriod:
    if isinstance(period, ExxonCapexPeriod):
        return period
    try:
        return EXXON_CAPEX_PERIODS[period]
    except KeyError as exc:
        raise KeyError(f"unknown Exxon capex period: {period!r}") from exc


def _period_readout_table(
    periods: Iterable[ExxonCapexPeriod],
    packs: Iterable[usf.DisclosureAuditPack],
) -> list[str]:
    lines = [
        "| Period | Cash PP&E additions | Segment-note capacity | Upstream lower bound | Upstream share interval | Claim verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for period, pack in zip(periods, packs, strict=True):
        interval = pack.interval
        verdict = "n/a" if pack.claim_audit is None else pack.claim_audit.verdict
        lines.append(
            "| "
            + " | ".join(
                [
                    period.label,
                    f"{period.cash_ppe_additions:,.0f} $M",
                    f"{period.segment_note_ppe_additions:,.0f} $M",
                    f"{period.upstream_minimum:,.0f} $M",
                    f"{interval.scaled_lower:.1f}% to {interval.scaled_upper:.1f}%",
                    verdict,
                ]
            )
            + " |"
        )
    return lines


def _frontier_summary_table(periods: Iterable[ExxonCapexPeriod]) -> list[str]:
    lines = [
        "| Period | Baseline ambiguity | Region-only ambiguity | Segment ambiguity | Minimal stable refinement |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for period in periods:
        frontier = build_frontier(period)
        baseline = frontier.baseline
        region = frontier.explain(["region"]).candidate
        segment = frontier.explain(["segment"]).candidate
        minimal = frontier.minimal_stable
        minimal_label = "none" if minimal is None else f"`{minimal.label}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    period.label,
                    _format_percentage_points(
                        0.0 if baseline is None else baseline.max_ambiguity
                    ),
                    _format_percentage_points(region.max_ambiguity),
                    _format_percentage_points(segment.max_ambiguity),
                    minimal_label,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The frontier uses a neutral capacity-proportional allocation as "
            "the observed hidden mix. It is a disclosure-design check, not a "
            "replacement for the exact capacity LP above.",
        ]
    )
    return lines


def _format_percentage_points(value: float) -> str:
    return f"{100.0 * value:.1f} pp"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Exxon capex-capacity triangulation report.",
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
