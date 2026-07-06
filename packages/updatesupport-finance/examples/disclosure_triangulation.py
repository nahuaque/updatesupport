"""Generic disclosure triangulation worked example.

Run from the repository root with:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/disclosure_triangulation.py

Optionally write the Markdown report:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/disclosure_triangulation.py \
        --output data/disclosure_triangulation_report.md
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import updatesupport as us
import updatesupport_finance as usf


TARGETS = ("component_2022", "component_2023", "component_2024")
BASELINE_TIER = "T0 totals + containment"


def build_spec() -> usf.DisclosureTriangulationSpec:
    """Build a neutral disclosure feasibility problem.

    The unknown component is constrained by disclosed totals, containment inside
    those totals, rounded growth-rate statements, and a later anchor interval.
    Names are deliberately generic so the example documents the modeling
    pattern rather than an issuer-specific case.
    """

    growth_2023 = usf.rounded_growth_constraints(
        "growth_2023",
        current="component_2023",
        previous="component_2022",
        growth_percent=30.0,
        rounding=2.5,
        provenance="Example rounded growth disclosure for 2023",
        verified=True,
    )
    growth_2024 = usf.rounded_growth_constraints(
        "growth_2024",
        current="component_2024",
        previous="component_2023",
        growth_percent=45.0,
        rounding=2.5,
        provenance="Example rounded growth disclosure for 2024",
        verified=True,
    )
    constraints = [
        usf.exact_disclosure_constraint(
            "reported_total_2022",
            "total_2022",
            120.0,
            provenance="Example annual report table",
            verified=True,
        ),
        usf.exact_disclosure_constraint(
            "reported_total_2023",
            "total_2023",
            165.0,
            provenance="Example annual report table",
            verified=True,
        ),
        usf.exact_disclosure_constraint(
            "reported_total_2024",
            "total_2024",
            230.0,
            provenance="Example annual report table",
            verified=True,
        ),
        usf.containment_constraint(
            "component_within_total_2022",
            child="component_2022",
            parent="total_2022",
            provenance="Component is part of reported total",
            verified=True,
        ),
        usf.containment_constraint(
            "component_within_total_2023",
            child="component_2023",
            parent="total_2023",
            provenance="Component is part of reported total",
            verified=True,
        ),
        usf.containment_constraint(
            "component_within_total_2024",
            child="component_2024",
            parent="total_2024",
            provenance="Component is part of reported total",
            verified=True,
        ),
        *growth_2023,
        *growth_2024,
        usf.interval_disclosure_constraint(
            "component_2024_anchor",
            "component_2024",
            lower=100.0,
            upper=120.0,
            category="anchor_disclosure",
            provenance="Example later direct component disclosure",
            description="Later anchor interval for the same component.",
            verified=True,
        ),
    ]
    return usf.disclosure_triangulation_spec(
        title="Generic Disclosure Triangulation",
        description=(
            "A neutral feasibility example for overlapping disclosures: totals, "
            "containment, rounded growth rates, and a later anchor interval."
        ),
        variables=[
            usf.disclosure_variable("component_2022", unit="units"),
            usf.disclosure_variable("component_2023", unit="units"),
            usf.disclosure_variable("component_2024", unit="units"),
            usf.disclosure_variable("total_2022", unit="units"),
            usf.disclosure_variable("total_2023", unit="units"),
            usf.disclosure_variable("total_2024", unit="units"),
        ],
        constraints=constraints,
        targets=[
            usf.disclosure_target(
                target,
                target,
                label=target.replace("_", " "),
                unit="units",
            )
            for target in TARGETS
        ],
        tiers=[
            usf.disclosure_tier(
                BASELINE_TIER,
                [
                    "reported_total_2022",
                    "reported_total_2023",
                    "reported_total_2024",
                    "component_within_total_2022",
                    "component_within_total_2023",
                    "component_within_total_2024",
                ],
                description=(
                    "Only exact reported totals and component-within-total "
                    "containment are active."
                ),
            ),
            usf.disclosure_tier(
                "T1 + rounded growth",
                [
                    "reported_total_2022",
                    "reported_total_2023",
                    "reported_total_2024",
                    "component_within_total_2022",
                    "component_within_total_2023",
                    "component_within_total_2024",
                    "growth_2023_lower",
                    "growth_2023_upper",
                    "growth_2024_lower",
                    "growth_2024_upper",
                ],
                description=(
                    "Adds rounded growth-rate disclosures. These ratios connect "
                    "years but do not by themselves identify the component level."
                ),
            ),
            usf.disclosure_tier(
                "T2 + anchor disclosure",
                [
                    "reported_total_2022",
                    "reported_total_2023",
                    "reported_total_2024",
                    "component_within_total_2022",
                    "component_within_total_2023",
                    "component_within_total_2024",
                    "growth_2023_lower",
                    "growth_2023_upper",
                    "growth_2024_lower",
                    "growth_2024_upper",
                    "component_2024_anchor",
                ],
                description=(
                    "Adds a later direct component interval. Combined with the "
                    "growth chain, this anchor propagates backward into earlier "
                    "years."
                ),
            ),
        ],
    )


def build_report() -> usf.DisclosureTriangulationReport:
    """Solve the example disclosure feasibility problem."""

    return usf.triangulate_disclosure(build_spec())


def build_attribution_report(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "component_2022",
    tier: str = "T2 + anchor disclosure",
    group_by: str = "constraint",
) -> us.NamedLinearConstraintAttributionReport:
    """Rank which active disclosures most narrow one target interval."""

    if report is None:
        report = build_report()
    return usf.attribute_disclosure_constraints(
        report,
        target=target,
        tier=tier,
        group_by=group_by,
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
                None
                if baseline.width <= 0.0
                else 100.0 * reduction / baseline.width
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
    lines = [
        "# Generic Disclosure Triangulation Worked Example",
        "",
        "This example asks what an overlapping set of disclosures pins down about "
        "an undisclosed component. It is not issuer-specific: the variables are "
        "generic component and total amounts.",
        "",
        "The tiers separate what each layer of information buys:",
        "",
        "- `T0`: exact totals plus component containment",
        "- `T1`: rounded growth-rate disclosures",
        "- `T2`: a later direct component anchor interval",
        "",
        "## Width Reduction By Tier",
        "",
    ]
    lines.extend(_width_reduction_table(width_reduction_rows(report)))
    lines.extend(
        [
            "",
            "## Constraint Value Attribution",
            "",
            build_attribution_report(report).to_markdown(),
            "",
            "## Solver Report",
            "",
            report.to_markdown(),
        ]
    )
    return "\n".join(lines)


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
    return f"{float(value):.4g}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a generic disclosure triangulation report.",
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
