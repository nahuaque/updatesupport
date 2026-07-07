"""RevOps funnel analysis stability example.

The example audits whether a reported marketing-to-sales funnel conversion
claim survives hidden pipeline-mix recomposition. It is synthetic and
dependency-free: each row is a retained funnel cell with an MQL count and an
MQL-to-SQL conversion rate.

Run from the repository root with:

    uv run python examples/revops_funnel_stability.py

Optionally write the Markdown report:

    uv run python examples/revops_funnel_stability.py \
        --output data/revops_funnel_stability.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us


PUBLIC_COLUMNS = ("quarter", "reported_segment", "region")
HIDDEN_REFINEMENTS = (
    "lead_source",
    "campaign_type",
    "industry",
    "deal_size_band",
    "rep_ramp_band",
)
HIDDEN_COLUMNS = PUBLIC_COLUMNS + HIDDEN_REFINEMENTS
CANDIDATE_REFINEMENTS = (
    "lead_source",
    "campaign_type",
    "industry",
    "deal_size_band",
    "rep_ramp_band",
)
CONVERSION_COLUMN = "sql_conversion_rate"
WEIGHT_COLUMN = "mql_count"


def synthetic_funnel_rows() -> list[dict[str, Any]]:
    """Return synthetic retained RevOps funnel cells."""

    rows = [
        (
            "FY26Q2",
            "enterprise",
            "north_america",
            "demo_request",
            "high_intent",
            "software",
            "250k_plus",
            "tenured",
            0.38,
            250,
        ),
        (
            "FY26Q2",
            "enterprise",
            "north_america",
            "outbound",
            "target_account",
            "manufacturing",
            "250k_plus",
            "ramping",
            0.21,
            180,
        ),
        (
            "FY26Q2",
            "enterprise",
            "north_america",
            "partner",
            "co_sell",
            "software",
            "100k_250k",
            "tenured",
            0.31,
            90,
        ),
        (
            "FY26Q2",
            "enterprise",
            "emea",
            "field_event",
            "event_followup",
            "financial_services",
            "250k_plus",
            "tenured",
            0.33,
            120,
        ),
        (
            "FY26Q2",
            "enterprise",
            "emea",
            "outbound",
            "target_account",
            "manufacturing",
            "250k_plus",
            "ramping",
            0.15,
            160,
        ),
        (
            "FY26Q2",
            "enterprise",
            "emea",
            "intent_data",
            "high_intent",
            "software",
            "100k_250k",
            "tenured",
            0.25,
            100,
        ),
        (
            "FY26Q2",
            "mid_market",
            "north_america",
            "paid_search",
            "demand_gen",
            "software",
            "50k_100k",
            "tenured",
            0.19,
            300,
        ),
        (
            "FY26Q2",
            "mid_market",
            "north_america",
            "organic",
            "content",
            "software",
            "50k_100k",
            "tenured",
            0.27,
            220,
        ),
        (
            "FY26Q2",
            "mid_market",
            "north_america",
            "webinar",
            "nurture",
            "healthcare",
            "25k_50k",
            "ramping",
            0.23,
            180,
        ),
        (
            "FY26Q2",
            "mid_market",
            "emea",
            "paid_social",
            "demand_gen",
            "retail",
            "25k_50k",
            "ramping",
            0.13,
            220,
        ),
        (
            "FY26Q2",
            "mid_market",
            "emea",
            "organic",
            "content",
            "software",
            "50k_100k",
            "tenured",
            0.25,
            150,
        ),
        (
            "FY26Q2",
            "mid_market",
            "emea",
            "partner",
            "co_sell",
            "manufacturing",
            "50k_100k",
            "tenured",
            0.21,
            130,
        ),
        (
            "FY26Q2",
            "smb",
            "north_america",
            "paid_search",
            "demand_gen",
            "horizontal",
            "under_25k",
            "ramping",
            0.11,
            400,
        ),
        (
            "FY26Q2",
            "smb",
            "north_america",
            "product_signup",
            "product_led",
            "horizontal",
            "under_25k",
            "tenured",
            0.21,
            350,
        ),
        (
            "FY26Q2",
            "smb",
            "north_america",
            "content",
            "nurture",
            "horizontal",
            "under_25k",
            "ramping",
            0.15,
            250,
        ),
        (
            "FY26Q2",
            "smb",
            "emea",
            "paid_social",
            "demand_gen",
            "horizontal",
            "under_25k",
            "ramping",
            0.09,
            300,
        ),
        (
            "FY26Q2",
            "smb",
            "emea",
            "product_signup",
            "product_led",
            "horizontal",
            "under_25k",
            "tenured",
            0.19,
            220,
        ),
        (
            "FY26Q2",
            "smb",
            "emea",
            "content",
            "nurture",
            "horizontal",
            "under_25k",
            "ramping",
            0.14,
            180,
        ),
    ]
    columns = HIDDEN_COLUMNS + (CONVERSION_COLUMN, WEIGHT_COLUMN)
    return [dict(zip(columns, row, strict=True)) for row in rows]


def build_public_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> us.PublicDescentReport:
    """Build the public-descent audit for the synthetic funnel report."""

    rows = synthetic_funnel_rows() if rows is None else rows
    return us.public_descent_report(
        rows,
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=CONVERSION_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q=q,
        q_radius=q_radius,
        top=top,
        title="RevOps Funnel Public-Segment Stability Audit",
        target_description="MQL-to-SQL conversion rate",
        observed_label="Observed SQL conversion rate",
        row_count=len(rows),
        row_count_label="Retained funnel cells",
        weight_column=WEIGHT_COLUMN,
        min_cell_weight=0.0,
    )


def build_funnel_claim(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    threshold: float = 0.18,
    ambiguity_limit: float = 0.03,
    max_added_columns: int = 2,
) -> us.ClaimAudit:
    """Audit whether the reported funnel-health threshold is invariant."""

    rows = synthetic_funnel_rows() if rows is None else rows
    healthy_label = "funnel_healthy"
    risk_label = "conversion_risk"
    claim = us.claim(
        estimate_name="SQL conversion funnel-health claim",
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=CONVERSION_COLUMN,
        weight=WEIGHT_COLUMN,
        q_presets=("saturated",),
        candidate_refinements=CANDIDATE_REFINEMENTS,
        ambiguity_limit=ambiguity_limit,
        decision=us.threshold_decision(
            ">=",
            threshold,
            label=f"SQL conversion is at least {100.0 * threshold:.0f}%",
            pass_label=healthy_label,
            fail_label=risk_label,
        ),
        search="beam",
        beam_width=8,
        max_added_columns=max_added_columns,
        max_evaluations=96,
        exact_required=False,
        target_description="MQL-to-SQL conversion rate",
        observed_label="Observed SQL conversion rate",
        title="RevOps Funnel Health Claim Audit",
    )
    return claim.audit(rows)


def build_frontier(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.03,
    q_radius: float = 0.5,
) -> us.PublicRepresentationFrontier:
    """Search for small public segmentations that stabilize conversion."""

    rows = synthetic_funnel_rows() if rows is None else rows
    return us.public_representation_frontier(
        rows,
        base_public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=CONVERSION_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
        min_cell_weights=(0.0,),
        ambiguity_limit=ambiguity_limit,
        search="beam",
        beam_width=8,
        max_added_columns=3,
        max_evaluations=96,
        title="RevOps Funnel Public Representation Frontier",
    )


def build_review_artifacts(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.03,
) -> dict[str, Any]:
    """Build the report objects a RevOps analyst would review or export."""

    rows = synthetic_funnel_rows() if rows is None else rows
    artifacts: dict[str, Any] = {
        "public_report": build_public_report(rows),
        "claim_audit": build_funnel_claim(rows),
    }
    if include_frontier:
        artifacts["frontier"] = build_frontier(
            rows,
            ambiguity_limit=frontier_ambiguity_limit,
        )
    return artifacts


def render_report(
    *,
    include_public_report: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.03,
) -> str:
    """Render the synthetic RevOps funnel stability report as Markdown."""

    rows = synthetic_funnel_rows()
    public_report = build_public_report(rows)
    claim_report = build_funnel_claim(rows)
    observed = public_report.observed_value
    lower = public_report.interval.lower
    upper = public_report.interval.upper
    ambiguity = public_report.interval.diameter

    lines = [
        "# RevOps Funnel Analysis Stability Example",
        "",
        "This synthetic demo audits a RevOps funnel report. The public report "
        "shows MQL-to-SQL conversion by quarter, reported segment, and region. "
        "The retained cells also track lead source, campaign type, industry, "
        "deal-size band, and rep ramp.",
        "",
        "The headline funnel metric looks healthy:",
        "",
        f"- Observed MQL-to-SQL conversion: {_percent(observed)}",
        "",
        "But if the public funnel mix is held fixed and hidden pipeline "
        "composition is allowed to shift inside those public buckets, the "
        "compatible conversion rate ranges from:",
        "",
        f"```text\n{_percent(lower)} to {_percent(upper)}\n```",
        "",
        f"The ambiguity width is {_percent(ambiguity)}. Because the interval "
        "crosses the 18% funnel-health threshold, the apparent healthy-funnel "
        "conclusion is not invariant to the declared hidden-composition stress "
        "test.",
        "",
        "Plain English: the reported RevOps cuts are too coarse to certify that "
        "the funnel is healthy. A different retained mix of lead sources, "
        "campaign types, industries, deal sizes, or rep-ramp bands could "
        "preserve the same public segment mix while pulling SQL conversion "
        "below the operating threshold.",
        "",
        "This is not a confidence interval and not a replacement for funnel "
        "forecast uncertainty. It is a representation-stability audit for "
        "retained funnel cells and the selected Q stress test.",
    ]

    if include_public_report:
        lines.extend(["", "## Public-Descent Audit", ""])
        lines.extend(_without_title(public_report.to_markdown()))

    lines.extend(["", "## Funnel Health Claim Audit", ""])
    lines.extend(_without_title(claim_report.to_markdown()))

    if include_frontier:
        lines.extend(["", "## Public Representation Frontier", ""])
        lines.extend(
            _without_title(
                build_frontier(
                    rows,
                    ambiguity_limit=frontier_ambiguity_limit,
                ).to_markdown()
            )
        )
    return "\n".join(lines)


def export_review_artifacts(
    output_dir: Path,
    *,
    include_public_report: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.03,
) -> tuple[Path, ...]:
    """Write Markdown, JSON, and CSV artifacts for a RevOps review packet."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    markdown = render_report(
        include_public_report=include_public_report,
        include_frontier=include_frontier,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
    )
    markdown_path = output_dir / "revops_funnel_stability.md"
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    written.append(markdown_path)

    artifacts = build_review_artifacts(
        include_frontier=include_frontier,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
    )
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for artifact_name, report in artifacts.items():
        json_path = output_dir / f"{artifact_name}.json"
        json_path.write_text(report.to_json() + "\n", encoding="utf-8")
        written.append(json_path)
        for table_name, rows in report.to_tables().items():
            csv_path = tables_dir / f"{artifact_name}__{table_name}.csv"
            _write_csv_table(csv_path, rows)
            written.append(csv_path)
    return tuple(written)


def _write_csv_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, bool | int | float):
        return value
    return json.dumps(value, sort_keys=True)


def _without_title(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0]:
            lines = lines[1:]
    return lines


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help=("Optional directory for Markdown, JSON, and CSV review artifacts."),
    )
    parser.add_argument(
        "--no-public-report",
        action="store_true",
        help="Omit the detailed public-descent section.",
    )
    parser.add_argument(
        "--no-frontier",
        action="store_true",
        help="Omit public-representation frontier search.",
    )
    parser.add_argument(
        "--frontier-ambiguity-limit",
        type=float,
        default=0.03,
        help="Frontier stability limit for SQL-conversion ambiguity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = render_report(
        include_public_report=not args.no_public_report,
        include_frontier=not args.no_frontier,
        frontier_ambiguity_limit=args.frontier_ambiguity_limit,
    )
    if args.output is None and args.export_dir is None:
        print(markdown)
        return
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.export_dir is not None:
        written = export_review_artifacts(
            args.export_dir,
            include_public_report=not args.no_public_report,
            include_frontier=not args.no_frontier,
            frontier_ambiguity_limit=args.frontier_ambiguity_limit,
        )
        print(f"Wrote {len(written)} review artifacts to {args.export_dir}")


if __name__ == "__main__":
    main()
