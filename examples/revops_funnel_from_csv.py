"""RevOps funnel stability recipe for analyst CSV exports.

This recipe lets a RevOps analyst point ``updatesupport`` at a retained
funnel-cell CSV exported from a warehouse, Salesforce, HubSpot, or a BI tool.
It supports two modes:

* ``level``: audit a threshold claim such as "SQL conversion is at least 18%".
* ``trend``: audit a current-minus-prior trend claim such as "Q/Q conversion
  improved".

Example level claim:

    uv run python examples/revops_funnel_from_csv.py \
      --input funnel_cells.csv \
      --mode level \
      --public quarter reported_segment region \
      --hidden quarter reported_segment region lead_source campaign_type \
        industry deal_size_band rep_ramp_band \
      --target sql_conversion_rate \
      --weight mql_count \
      --threshold 0.18 \
      --output-dir data/revops_review

Example trend claim:

    uv run python examples/revops_funnel_from_csv.py \
      --input paired_funnel_cells.csv \
      --mode trend \
      --public reported_segment region \
      --hidden reported_segment region lead_source campaign_type industry \
        deal_size_band rep_ramp_band \
      --current-target FY26Q2_sql_conversion_rate \
      --prior-target FY26Q1_sql_conversion_rate \
      --weight comparison_weight \
      --output-dir data/revops_trend_review
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us

from examples.revops_funnel_stability import _write_csv_table


TREND_TARGET_COLUMN = "__revops_qoq_lift__"
TREND_PERIOD_COLUMN = "__revops_period__"
TREND_VALUE_COLUMN = "__revops_period_value__"


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Load an analyst-exported CSV as row dictionaries."""

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} does not contain any data rows")
    return rows


def build_level_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: str,
    weight: str,
    threshold: float,
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.03,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
    top: int = 8,
) -> dict[str, Any]:
    """Build report objects for a threshold claim over a CSV funnel export."""

    public_columns = tuple(public)
    hidden_columns = tuple(hidden)
    candidates = _candidate_refinements(
        public=public_columns,
        hidden=hidden_columns,
        candidate_refinements=candidate_refinements,
    )
    numeric_rows = _numeric_rows(rows, (target, weight))
    _validate_columns(
        numeric_rows,
        public_columns + hidden_columns + (target, weight),
    )
    healthy_label = "claim_passes"
    risk_label = "claim_fails"
    public_report = us.public_descent_report(
        numeric_rows,
        public=public_columns,
        hidden=hidden_columns,
        target=target,
        weight=weight,
        candidate_refinements=candidates,
        q="saturated",
        top=top,
        title="RevOps CSV Public-Segment Stability Audit",
        target_description=target,
        observed_label=f"Observed {target}",
        row_count=len(numeric_rows),
        row_count_label="Retained funnel CSV cells",
        weight_column=weight,
        min_cell_weight=0.0,
    )
    claim = us.claim(
        estimate_name=f"{target} threshold claim",
        public=public_columns,
        hidden=hidden_columns,
        target=target,
        weight=weight,
        q_presets=("saturated",),
        candidate_refinements=candidates,
        ambiguity_limit=ambiguity_limit,
        decision=us.threshold_decision(
            ">=",
            threshold,
            label=f"{target} >= {threshold:g}",
            pass_label=healthy_label,
            fail_label=risk_label,
        ),
        search="beam",
        beam_width=8,
        max_added_columns=max_added_columns,
        max_evaluations=96,
        exact_required=False,
        target_description=target,
        observed_label=f"Observed {target}",
        title="RevOps CSV Threshold Claim Audit",
    ).audit(numeric_rows)
    artifacts: dict[str, Any] = {
        "public_report": public_report,
        "claim_audit": claim,
    }
    if include_frontier:
        artifacts["frontier"] = us.public_representation_frontier(
            numeric_rows,
            base_public=public_columns,
            hidden=hidden_columns,
            target=target,
            weight=weight,
            candidate_refinements=candidates,
            q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
            min_cell_weights=(0.0,),
            ambiguity_limit=(
                ambiguity_limit
                if frontier_ambiguity_limit is None
                else frontier_ambiguity_limit
            ),
            search="beam",
            beam_width=8,
            max_added_columns=3,
            max_evaluations=96,
            title="RevOps CSV Public Representation Frontier",
        )
    return artifacts


def render_level_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: str,
    weight: str,
    threshold: float,
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.03,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
) -> str:
    """Render a Markdown review packet for a CSV level claim."""

    artifacts = build_level_artifacts(
        rows,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        threshold=threshold,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    report = artifacts["public_report"]
    claim = artifacts["claim_audit"]
    lines = [
        "# RevOps Funnel CSV Level Claim Audit",
        "",
        "This report was generated from an analyst-supplied retained funnel-cell "
        "CSV. It audits whether the reported level claim survives hidden "
        "pipeline-mix recomposition inside the declared public buckets.",
        "",
        f"- Target: `{target}`",
        f"- Weight: `{weight}`",
        f"- Public columns: `{', '.join(public)}`",
        f"- Hidden columns: `{', '.join(hidden)}`",
        f"- Threshold: `{threshold:g}`",
        "",
        f"Observed `{target}` is {_format_float(report.observed_value)}. Under "
        "the declared hidden-composition stress test, the compatible interval "
        f"is [{_format_float(report.interval.lower)}, "
        f"{_format_float(report.interval.upper)}].",
        "",
        "This is not a confidence interval. It is a representation-stability "
        "audit relative to the retained CSV cells and the selected Q stress "
        "test.",
        "",
        "## Public-Descent Audit",
        "",
    ]
    lines.extend(_without_title(report.to_markdown()))
    lines.extend(["", "## Claim Audit", ""])
    lines.extend(_without_title(claim.to_markdown()))
    if include_frontier and "frontier" in artifacts:
        lines.extend(["", "## Public Representation Frontier", ""])
        lines.extend(_without_title(artifacts["frontier"].to_markdown()))
    return "\n".join(lines)


def export_level_review_artifacts(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: str,
    weight: str,
    threshold: float,
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.03,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
) -> tuple[Path, ...]:
    """Write Markdown, JSON, and CSV artifacts for a CSV level claim."""

    markdown = render_level_report(
        rows,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        threshold=threshold,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    artifacts = build_level_artifacts(
        rows,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        threshold=threshold,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    return _export_artifacts(
        output_dir,
        markdown_name="revops_csv_level_claim.md",
        markdown=markdown,
        artifacts=artifacts,
    )


def build_trend_artifacts(
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    current_target: str,
    prior_target: str,
    weight: str,
    threshold: float = 0.0,
    current_label: str = "current",
    prior_label: str = "prior",
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.025,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
    top: int = 8,
) -> dict[str, Any]:
    """Build report objects for a current-minus-prior CSV trend claim."""

    public_columns = tuple(public)
    hidden_columns = tuple(hidden)
    candidates = _candidate_refinements(
        public=public_columns,
        hidden=hidden_columns,
        candidate_refinements=candidate_refinements,
    )
    numeric_rows = _numeric_rows(rows, (current_target, prior_target, weight))
    _validate_columns(
        numeric_rows,
        public_columns + hidden_columns + (current_target, prior_target, weight),
    )
    trend_rows = _trend_rows(
        numeric_rows,
        hidden=hidden_columns,
        current_target=current_target,
        prior_target=prior_target,
        weight=weight,
    )
    comparison_rows = _comparison_rows(
        numeric_rows,
        hidden=hidden_columns,
        current_target=current_target,
        prior_target=prior_target,
        weight=weight,
        current_label=current_label,
        prior_label=prior_label,
    )
    improved_label = "trend_improved"
    not_certified_label = "trend_not_certified"
    trend_report = us.public_descent_report(
        trend_rows,
        public=public_columns,
        hidden=hidden_columns,
        target=TREND_TARGET_COLUMN,
        weight=weight,
        candidate_refinements=candidates,
        q="saturated",
        top=top,
        title="RevOps CSV Trend Public-Segment Stability Audit",
        target_description=f"{current_target} minus {prior_target}",
        observed_label="Observed current-minus-prior lift",
        row_count=len(trend_rows),
        row_count_label="Paired retained funnel CSV cells",
        weight_column=weight,
        min_cell_weight=0.0,
    )
    claim = us.claim(
        estimate_name="RevOps CSV current-minus-prior trend claim",
        public=public_columns,
        hidden=hidden_columns,
        target=TREND_TARGET_COLUMN,
        weight=weight,
        q_presets=("saturated",),
        candidate_refinements=candidates,
        ambiguity_limit=ambiguity_limit,
        decision=us.threshold_decision(
            ">=",
            threshold,
            label=f"{current_target} - {prior_target} >= {threshold:g}",
            pass_label=improved_label,
            fail_label=not_certified_label,
        ),
        search="beam",
        beam_width=8,
        max_added_columns=max_added_columns,
        max_evaluations=96,
        exact_required=False,
        target_description=f"{current_target} minus {prior_target}",
        observed_label="Observed current-minus-prior lift",
        title="RevOps CSV Trend Claim Audit",
    ).audit(trend_rows)
    comparison = us.robust_comparison_report(
        comparison_rows,
        item=TREND_PERIOD_COLUMN,
        public=public_columns,
        hidden=hidden_columns,
        target=TREND_VALUE_COLUMN,
        weight=weight,
        items=(current_label, prior_label),
        q="saturated",
        min_cell_weight=0.0,
        title="RevOps CSV Current-vs-Prior Robust Comparison",
        target_description="funnel metric value",
        observed_label="Observed funnel metric value",
    )
    artifacts: dict[str, Any] = {
        "trend_report": trend_report,
        "trend_claim": claim,
        "period_comparison": comparison,
    }
    if include_frontier:
        artifacts["frontier"] = us.public_representation_frontier(
            trend_rows,
            base_public=public_columns,
            hidden=hidden_columns,
            target=TREND_TARGET_COLUMN,
            weight=weight,
            candidate_refinements=candidates,
            q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
            min_cell_weights=(0.0,),
            ambiguity_limit=(
                ambiguity_limit
                if frontier_ambiguity_limit is None
                else frontier_ambiguity_limit
            ),
            search="beam",
            beam_width=8,
            max_added_columns=3,
            max_evaluations=96,
            title="RevOps CSV Trend Public Representation Frontier",
        )
    return artifacts


def render_trend_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    current_target: str,
    prior_target: str,
    weight: str,
    threshold: float = 0.0,
    current_label: str = "current",
    prior_label: str = "prior",
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.025,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
) -> str:
    """Render a Markdown review packet for a CSV trend claim."""

    artifacts = build_trend_artifacts(
        rows,
        public=public,
        hidden=hidden,
        current_target=current_target,
        prior_target=prior_target,
        weight=weight,
        threshold=threshold,
        current_label=current_label,
        prior_label=prior_label,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    report = artifacts["trend_report"]
    claim = artifacts["trend_claim"]
    comparison = artifacts["period_comparison"]
    lines = [
        "# RevOps Funnel CSV Trend Claim Audit",
        "",
        "This report was generated from an analyst-supplied retained funnel-cell "
        "CSV. It audits whether a current-minus-prior trend claim survives "
        "hidden pipeline-mix recomposition inside the declared public buckets.",
        "",
        f"- Current target: `{current_target}`",
        f"- Prior target: `{prior_target}`",
        f"- Weight: `{weight}`",
        f"- Public columns: `{', '.join(public)}`",
        f"- Hidden columns: `{', '.join(hidden)}`",
        f"- Threshold: `{threshold:g}`",
        "",
        "Observed current-minus-prior lift is "
        f"{_format_float(report.observed_value)}. Under the declared "
        "hidden-composition stress test, the compatible interval is "
        f"[{_format_float(report.interval.lower)}, "
        f"{_format_float(report.interval.upper)}].",
        "",
        "This is not a confidence interval. It is a representation-stability "
        "audit relative to the retained CSV cells and the selected Q stress "
        "test.",
        "",
        "## Trend Public-Descent Audit",
        "",
    ]
    lines.extend(_without_title(report.to_markdown()))
    lines.extend(["", "## Trend Claim Audit", ""])
    lines.extend(_without_title(claim.to_markdown()))
    lines.extend(["", "## Robust Current-vs-Prior Comparison", ""])
    lines.extend(_without_title(comparison.to_markdown()))
    if include_frontier and "frontier" in artifacts:
        lines.extend(["", "## Public Representation Frontier", ""])
        lines.extend(_without_title(artifacts["frontier"].to_markdown()))
    return "\n".join(lines)


def export_trend_review_artifacts(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    current_target: str,
    prior_target: str,
    weight: str,
    threshold: float = 0.0,
    current_label: str = "current",
    prior_label: str = "prior",
    candidate_refinements: Sequence[str] | None = None,
    ambiguity_limit: float = 0.025,
    frontier_ambiguity_limit: float | None = None,
    include_frontier: bool = True,
    q_radius: float = 0.5,
    max_added_columns: int = 2,
) -> tuple[Path, ...]:
    """Write Markdown, JSON, and CSV artifacts for a CSV trend claim."""

    markdown = render_trend_report(
        rows,
        public=public,
        hidden=hidden,
        current_target=current_target,
        prior_target=prior_target,
        weight=weight,
        threshold=threshold,
        current_label=current_label,
        prior_label=prior_label,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    artifacts = build_trend_artifacts(
        rows,
        public=public,
        hidden=hidden,
        current_target=current_target,
        prior_target=prior_target,
        weight=weight,
        threshold=threshold,
        current_label=current_label,
        prior_label=prior_label,
        candidate_refinements=candidate_refinements,
        ambiguity_limit=ambiguity_limit,
        frontier_ambiguity_limit=frontier_ambiguity_limit,
        include_frontier=include_frontier,
        q_radius=q_radius,
        max_added_columns=max_added_columns,
    )
    return _export_artifacts(
        output_dir,
        markdown_name="revops_csv_trend_claim.md",
        markdown=markdown,
        artifacts=artifacts,
    )


def _export_artifacts(
    output_dir: Path,
    *,
    markdown_name: str,
    markdown: str,
    artifacts: Mapping[str, Any],
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    markdown_path = output_dir / markdown_name
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    written.append(markdown_path)
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


def _trend_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden: Sequence[str],
    current_target: str,
    prior_target: str,
    weight: str,
) -> tuple[dict[str, Any], ...]:
    output = []
    for row in rows:
        output.append(
            {
                **{column: row[column] for column in hidden},
                TREND_TARGET_COLUMN: row[current_target] - row[prior_target],
                weight: row[weight],
            }
        )
    return tuple(output)


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden: Sequence[str],
    current_target: str,
    prior_target: str,
    weight: str,
    current_label: str,
    prior_label: str,
) -> tuple[dict[str, Any], ...]:
    output = []
    for row in rows:
        hidden_values = {column: row[column] for column in hidden}
        output.append(
            {
                **hidden_values,
                TREND_PERIOD_COLUMN: prior_label,
                TREND_VALUE_COLUMN: row[prior_target],
                weight: row[weight],
            }
        )
        output.append(
            {
                **hidden_values,
                TREND_PERIOD_COLUMN: current_label,
                TREND_VALUE_COLUMN: row[current_target],
                weight: row[weight],
            }
        )
    return tuple(output)


def _numeric_rows(
    rows: Sequence[Mapping[str, Any]],
    numeric_columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    output = []
    for row_number, row in enumerate(rows, start=1):
        converted = dict(row)
        for column in numeric_columns:
            if column not in row:
                raise ValueError(f"row {row_number} is missing column {column!r}")
            converted[column] = _coerce_float(
                row[column],
                column=column,
                row_number=row_number,
            )
        output.append(converted)
    return tuple(output)


def _coerce_float(value: Any, *, column: str, row_number: int) -> float:
    if value is None or value == "":
        raise ValueError(f"row {row_number} has an empty value for {column!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_number} has nonnumeric value {value!r} for {column!r}"
        ) from exc


def _candidate_refinements(
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    candidate_refinements: Sequence[str] | None,
) -> tuple[str, ...]:
    public_set = set(public)
    candidates = (
        tuple(candidate_refinements)
        if candidate_refinements is not None
        else tuple(column for column in hidden if column not in public_set)
    )
    if not candidates:
        raise ValueError(
            "candidate refinements cannot be empty; pass at least one hidden "
            "column that is not already public"
        )
    return candidates


def _validate_columns(
    rows: Sequence[Mapping[str, Any]],
    required_columns: Sequence[str],
) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    missing = sorted(
        {
            column
            for column in required_columns
            if any(column not in row for row in rows)
        }
    )
    if missing:
        raise ValueError(f"missing required columns: {missing!r}")


def _without_title(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0]:
            lines = lines[1:]
    return lines


def _format_float(value: float) -> str:
    return f"{float(value):.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("level", "trend"), default="level")
    parser.add_argument("--public", nargs="+", required=True)
    parser.add_argument("--hidden", nargs="+", required=True)
    parser.add_argument("--candidate-refinements", nargs="*")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--target", help="Target column for --mode level.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold. Required for level mode; defaults to 0 for trend.",
    )
    parser.add_argument(
        "--current-target", help="Current-period target for trend mode."
    )
    parser.add_argument("--prior-target", help="Prior-period target for trend mode.")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--prior-label", default="prior")
    parser.add_argument("--ambiguity-limit", type=float)
    parser.add_argument("--frontier-ambiguity-limit", type=float)
    parser.add_argument("--q-radius", type=float, default=0.5)
    parser.add_argument("--max-added-columns", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-frontier", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "level":
        if args.target is None:
            parser.error("--target is required in level mode")
        if args.threshold is None:
            parser.error("--threshold is required in level mode")
    if args.mode == "trend":
        if args.current_target is None or args.prior_target is None:
            parser.error(
                "--current-target and --prior-target are required in trend mode"
            )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_csv_rows(args.input)
    if args.mode == "level":
        markdown = render_level_report(
            rows,
            public=args.public,
            hidden=args.hidden,
            target=args.target,
            weight=args.weight,
            threshold=args.threshold,
            candidate_refinements=args.candidate_refinements,
            ambiguity_limit=(
                0.03 if args.ambiguity_limit is None else args.ambiguity_limit
            ),
            frontier_ambiguity_limit=args.frontier_ambiguity_limit,
            include_frontier=not args.no_frontier,
            q_radius=args.q_radius,
            max_added_columns=args.max_added_columns,
        )
        if args.output_dir is not None:
            written = export_level_review_artifacts(
                args.output_dir,
                rows,
                public=args.public,
                hidden=args.hidden,
                target=args.target,
                weight=args.weight,
                threshold=args.threshold,
                candidate_refinements=args.candidate_refinements,
                ambiguity_limit=(
                    0.03 if args.ambiguity_limit is None else args.ambiguity_limit
                ),
                frontier_ambiguity_limit=args.frontier_ambiguity_limit,
                include_frontier=not args.no_frontier,
                q_radius=args.q_radius,
                max_added_columns=args.max_added_columns,
            )
            print(f"Wrote {len(written)} review artifacts to {args.output_dir}")
    else:
        trend_threshold = 0.0 if args.threshold is None else args.threshold
        trend_ambiguity_limit = (
            0.025 if args.ambiguity_limit is None else args.ambiguity_limit
        )
        markdown = render_trend_report(
            rows,
            public=args.public,
            hidden=args.hidden,
            current_target=args.current_target,
            prior_target=args.prior_target,
            weight=args.weight,
            threshold=trend_threshold,
            current_label=args.current_label,
            prior_label=args.prior_label,
            candidate_refinements=args.candidate_refinements,
            ambiguity_limit=trend_ambiguity_limit,
            frontier_ambiguity_limit=args.frontier_ambiguity_limit,
            include_frontier=not args.no_frontier,
            q_radius=args.q_radius,
            max_added_columns=args.max_added_columns,
        )
        if args.output_dir is not None:
            written = export_trend_review_artifacts(
                args.output_dir,
                rows,
                public=args.public,
                hidden=args.hidden,
                current_target=args.current_target,
                prior_target=args.prior_target,
                weight=args.weight,
                threshold=trend_threshold,
                current_label=args.current_label,
                prior_label=args.prior_label,
                candidate_refinements=args.candidate_refinements,
                ambiguity_limit=trend_ambiguity_limit,
                frontier_ambiguity_limit=args.frontier_ambiguity_limit,
                include_frontier=not args.no_frontier,
                q_radius=args.q_radius,
                max_added_columns=args.max_added_columns,
            )
            print(f"Wrote {len(written)} review artifacts to {args.output_dir}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.output is None and args.output_dir is None:
        print(markdown)


if __name__ == "__main__":
    main()
