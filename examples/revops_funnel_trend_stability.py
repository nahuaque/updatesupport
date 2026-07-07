"""RevOps funnel trend stability example.

The example audits whether a reported quarter-over-quarter MQL-to-SQL
conversion improvement survives hidden pipeline-mix recomposition. It is
synthetic and dependency-free.

Run from the repository root with:

    uv run python examples/revops_funnel_trend_stability.py

Optionally write review artifacts:

    uv run python examples/revops_funnel_trend_stability.py \
        --export-dir data/revops_funnel_trend_review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us

from examples.revops_funnel_stability import _write_csv_table


PUBLIC_COLUMNS = ("reported_segment", "region")
HIDDEN_REFINEMENTS = (
    "lead_source",
    "campaign_type",
    "industry",
    "deal_size_band",
    "rep_ramp_band",
)
HIDDEN_COLUMNS = PUBLIC_COLUMNS + HIDDEN_REFINEMENTS
CANDIDATE_REFINEMENTS = HIDDEN_REFINEMENTS
PERIOD_COLUMN = "quarter"
PRIOR_PERIOD = "FY26Q1"
CURRENT_PERIOD = "FY26Q2"
CONVERSION_COLUMN = "sql_conversion_rate"
LIFT_COLUMN = "qoq_sql_conversion_lift"
WEIGHT_COLUMN = "comparison_weight"


def synthetic_trend_rows() -> list[dict[str, Any]]:
    """Return paired hidden cells for a synthetic RevOps trend audit."""

    rows = [
        (
            "enterprise",
            "north_america",
            "demo_request",
            "high_intent",
            "software",
            "250k_plus",
            "tenured",
            0.38,
            250,
            0.05,
        ),
        (
            "enterprise",
            "north_america",
            "outbound",
            "target_account",
            "manufacturing",
            "250k_plus",
            "ramping",
            0.21,
            180,
            -0.02,
        ),
        (
            "enterprise",
            "north_america",
            "partner",
            "co_sell",
            "software",
            "100k_250k",
            "tenured",
            0.31,
            90,
            0.02,
        ),
        (
            "enterprise",
            "emea",
            "field_event",
            "event_followup",
            "financial_services",
            "250k_plus",
            "tenured",
            0.33,
            120,
            0.03,
        ),
        (
            "enterprise",
            "emea",
            "outbound",
            "target_account",
            "manufacturing",
            "250k_plus",
            "ramping",
            0.15,
            160,
            -0.04,
        ),
        (
            "enterprise",
            "emea",
            "intent_data",
            "high_intent",
            "software",
            "100k_250k",
            "tenured",
            0.25,
            100,
            0.02,
        ),
        (
            "mid_market",
            "north_america",
            "paid_search",
            "demand_gen",
            "software",
            "50k_100k",
            "tenured",
            0.19,
            300,
            -0.01,
        ),
        (
            "mid_market",
            "north_america",
            "organic",
            "content",
            "software",
            "50k_100k",
            "tenured",
            0.27,
            220,
            0.04,
        ),
        (
            "mid_market",
            "north_america",
            "webinar",
            "nurture",
            "healthcare",
            "25k_50k",
            "ramping",
            0.23,
            180,
            0.01,
        ),
        (
            "mid_market",
            "emea",
            "paid_social",
            "demand_gen",
            "retail",
            "25k_50k",
            "ramping",
            0.13,
            220,
            -0.03,
        ),
        (
            "mid_market",
            "emea",
            "organic",
            "content",
            "software",
            "50k_100k",
            "tenured",
            0.25,
            150,
            0.05,
        ),
        (
            "mid_market",
            "emea",
            "partner",
            "co_sell",
            "manufacturing",
            "50k_100k",
            "tenured",
            0.21,
            130,
            0.01,
        ),
        (
            "smb",
            "north_america",
            "paid_search",
            "demand_gen",
            "horizontal",
            "under_25k",
            "ramping",
            0.11,
            400,
            -0.01,
        ),
        (
            "smb",
            "north_america",
            "product_signup",
            "product_led",
            "horizontal",
            "under_25k",
            "tenured",
            0.21,
            350,
            0.06,
        ),
        (
            "smb",
            "north_america",
            "content",
            "nurture",
            "horizontal",
            "under_25k",
            "ramping",
            0.15,
            250,
            0.01,
        ),
        (
            "smb",
            "emea",
            "paid_social",
            "demand_gen",
            "horizontal",
            "under_25k",
            "ramping",
            0.09,
            300,
            -0.02,
        ),
        (
            "smb",
            "emea",
            "product_signup",
            "product_led",
            "horizontal",
            "under_25k",
            "tenured",
            0.19,
            220,
            0.05,
        ),
        (
            "smb",
            "emea",
            "content",
            "nurture",
            "horizontal",
            "under_25k",
            "ramping",
            0.14,
            180,
            0.01,
        ),
    ]
    output: list[dict[str, Any]] = []
    for row in rows:
        hidden = dict(zip(HIDDEN_COLUMNS, row[: len(HIDDEN_COLUMNS)], strict=True))
        current_rate = row[7]
        weight = row[8]
        lift = row[9]
        output.append(
            {
                **hidden,
                f"{PRIOR_PERIOD}_sql_conversion_rate": current_rate - lift,
                f"{CURRENT_PERIOD}_sql_conversion_rate": current_rate,
                LIFT_COLUMN: lift,
                WEIGHT_COLUMN: weight,
            }
        )
    return output


def synthetic_quarter_rows(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return long-form quarter rows for the robust comparison API."""

    rows = synthetic_trend_rows() if rows is None else rows
    output: list[dict[str, Any]] = []
    for row in rows:
        hidden = {column: row[column] for column in HIDDEN_COLUMNS}
        output.append(
            {
                **hidden,
                PERIOD_COLUMN: PRIOR_PERIOD,
                CONVERSION_COLUMN: row[f"{PRIOR_PERIOD}_sql_conversion_rate"],
                WEIGHT_COLUMN: row[WEIGHT_COLUMN],
            }
        )
        output.append(
            {
                **hidden,
                PERIOD_COLUMN: CURRENT_PERIOD,
                CONVERSION_COLUMN: row[f"{CURRENT_PERIOD}_sql_conversion_rate"],
                WEIGHT_COLUMN: row[WEIGHT_COLUMN],
            }
        )
    return output


def build_trend_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> us.PublicDescentReport:
    """Build the public-descent audit for the Q/Q trend target."""

    rows = synthetic_trend_rows() if rows is None else rows
    return us.public_descent_report(
        rows,
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=LIFT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q=q,
        q_radius=q_radius,
        top=top,
        title="RevOps Funnel Trend Public-Segment Stability Audit",
        target_description="current-quarter minus prior-quarter SQL conversion",
        observed_label="Observed Q/Q SQL conversion lift",
        row_count=len(rows),
        row_count_label="Paired retained funnel cells",
        weight_column=WEIGHT_COLUMN,
        min_cell_weight=0.0,
    )


def build_trend_claim(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.025,
    max_added_columns: int = 2,
) -> us.ClaimAudit:
    """Audit whether the positive Q/Q trend claim is invariant."""

    rows = synthetic_trend_rows() if rows is None else rows
    improved_label = "trend_improved"
    not_certified_label = "trend_not_certified"
    claim = us.claim(
        estimate_name="Positive RevOps Q/Q conversion trend claim",
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=LIFT_COLUMN,
        weight=WEIGHT_COLUMN,
        q_presets=("saturated",),
        candidate_refinements=CANDIDATE_REFINEMENTS,
        ambiguity_limit=ambiguity_limit,
        decision=us.threshold_decision(
            ">=",
            0.0,
            label="Q/Q SQL conversion lift is nonnegative",
            pass_label=improved_label,
            fail_label=not_certified_label,
        ),
        search="beam",
        beam_width=8,
        max_added_columns=max_added_columns,
        max_evaluations=96,
        exact_required=False,
        target_description="current-quarter minus prior-quarter SQL conversion",
        observed_label="Observed Q/Q SQL conversion lift",
        title="RevOps Funnel Trend Claim Audit",
    )
    return claim.audit(rows)


def build_quarter_comparison(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> us.RobustComparisonReport:
    """Compare current-quarter and prior-quarter conversion robustly."""

    rows = synthetic_trend_rows() if rows is None else rows
    return us.robust_comparison_report(
        synthetic_quarter_rows(rows),
        item=PERIOD_COLUMN,
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=CONVERSION_COLUMN,
        weight=WEIGHT_COLUMN,
        items=(CURRENT_PERIOD, PRIOR_PERIOD),
        q="saturated",
        min_cell_weight=0.0,
        title="RevOps Q/Q Funnel Conversion Robust Comparison",
        target_description="MQL-to-SQL conversion rate",
        observed_label="Observed SQL conversion rate",
    )


def build_frontier(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.025,
    q_radius: float = 0.5,
) -> us.PublicRepresentationFrontier:
    """Search for small public segmentations that stabilize the trend claim."""

    rows = synthetic_trend_rows() if rows is None else rows
    return us.public_representation_frontier(
        rows,
        base_public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=LIFT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
        min_cell_weights=(0.0,),
        ambiguity_limit=ambiguity_limit,
        search="beam",
        beam_width=8,
        max_added_columns=3,
        max_evaluations=96,
        title="RevOps Funnel Trend Public Representation Frontier",
    )


def build_review_artifacts(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.025,
) -> dict[str, Any]:
    """Build the report objects for a RevOps trend review packet."""

    rows = synthetic_trend_rows() if rows is None else rows
    artifacts: dict[str, Any] = {
        "trend_report": build_trend_report(rows),
        "trend_claim": build_trend_claim(rows),
        "quarter_comparison": build_quarter_comparison(rows),
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
    include_comparison: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.025,
) -> str:
    """Render the synthetic RevOps trend stability report as Markdown."""

    rows = synthetic_trend_rows()
    trend_report = build_trend_report(rows)
    claim_report = build_trend_claim(rows)
    comparison = build_quarter_comparison(rows)
    observed = trend_report.observed_value
    lower = trend_report.interval.lower
    upper = trend_report.interval.upper
    ambiguity = trend_report.interval.diameter

    lines = [
        "# RevOps Funnel Trend Stability Example",
        "",
        "This synthetic demo audits a quarter-over-quarter RevOps trend. The "
        "public report compares MQL-to-SQL conversion across reported segment "
        "and region. The retained paired cells also track lead source, campaign "
        "type, industry, deal-size band, and rep ramp.",
        "",
        "The current-quarter weighted trend looks positive:",
        "",
        f"- Observed Q/Q SQL conversion lift: {_percent(observed)}",
        "",
        "But if the public segment mix is held fixed and hidden pipeline "
        "composition is allowed to shift inside those public buckets, the "
        "compatible Q/Q lift ranges from:",
        "",
        f"```text\n{_percent(lower)} to {_percent(upper)}\n```",
        "",
        f"The ambiguity width is {_percent(ambiguity)}. Because the interval "
        "crosses zero, the apparent improvement is not invariant to the "
        "declared hidden-composition stress test.",
        "",
        "Plain English: the trend may be a pipeline-mix artifact. The same "
        "segment and region mix can hide different lead-source, campaign, "
        "industry, deal-size, or rep-ramp compositions that erase or reverse "
        "the observed improvement.",
        "",
        "This is not a confidence interval. It is a representation-stability "
        "audit for a supplied trend target and a declared Q stress test.",
    ]

    if include_public_report:
        lines.extend(["", "## Public-Descent Trend Audit", ""])
        lines.extend(_without_title(trend_report.to_markdown()))

    lines.extend(["", "## Trend Claim Audit", ""])
    lines.extend(_without_title(claim_report.to_markdown()))

    if include_comparison:
        lines.extend(["", "## Robust Quarter Comparison", ""])
        lines.extend(_without_title(comparison.to_markdown()))

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
    include_comparison: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.025,
) -> tuple[Path, ...]:
    """Write Markdown, JSON, and CSV artifacts for a RevOps trend review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    markdown = render_report(
        include_public_report=include_public_report,
        include_comparison=include_comparison,
        include_frontier=include_frontier,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
    )
    markdown_path = output_dir / "revops_funnel_trend_stability.md"
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    written.append(markdown_path)

    artifacts = build_review_artifacts(
        include_frontier=include_frontier,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
    )
    if not include_comparison:
        artifacts.pop("quarter_comparison", None)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for artifact_name, report in artifacts.items():
        json_path = output_dir / f"{artifact_name}.json"
        json_path.write_text(report.to_json() + "\n", encoding="utf-8")
        written.append(json_path)
        for table_name, table_rows in report.to_tables().items():
            csv_path = tables_dir / f"{artifact_name}__{table_name}.csv"
            _write_csv_table(csv_path, table_rows)
            written.append(csv_path)
    return tuple(written)


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
        "--no-comparison",
        action="store_true",
        help="Omit the robust quarter-comparison section.",
    )
    parser.add_argument(
        "--no-frontier",
        action="store_true",
        help="Omit public-representation frontier search.",
    )
    parser.add_argument(
        "--frontier-ambiguity-limit",
        type=float,
        default=0.025,
        help="Frontier stability limit for Q/Q SQL-conversion lift ambiguity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = render_report(
        include_public_report=not args.no_public_report,
        include_comparison=not args.no_comparison,
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
            include_comparison=not args.no_comparison,
            include_frontier=not args.no_frontier,
            frontier_ambiguity_limit=args.frontier_ambiguity_limit,
        )
        print(f"Wrote {len(written)} review artifacts to {args.export_dir}")


if __name__ == "__main__":
    main()
