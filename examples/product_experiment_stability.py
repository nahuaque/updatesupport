"""Product experimentation / A/B testing stability example.

The example audits whether a reported product-experiment lift survives hidden
segment recomposition. It is synthetic and dependency-free: each row is a
retained experiment cell with a user count and a treatment-minus-control lift.

Run from the repository root with:

    uv run python examples/product_experiment_stability.py

Optionally write the Markdown report:

    uv run python examples/product_experiment_stability.py \
        --output data/product_experiment_stability.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us


PUBLIC_COLUMNS = ("surface", "reported_segment", "platform")
HIDDEN_REFINEMENTS = (
    "acquisition_channel",
    "tenure_band",
    "geo_market",
    "plan_type",
    "device_type",
)
HIDDEN_COLUMNS = PUBLIC_COLUMNS + HIDDEN_REFINEMENTS
CANDIDATE_REFINEMENTS = (
    "acquisition_channel",
    "tenure_band",
    "geo_market",
    "plan_type",
    "device_type",
)
LIFT_COLUMN = "treatment_lift"
WEIGHT_COLUMN = "users"


def synthetic_experiment_rows() -> list[dict[str, Any]]:
    """Return synthetic retained cells for a product A/B-test audit."""

    rows = [
        (
            "onboarding",
            "new_users",
            "mobile",
            "paid_search",
            "0_30d",
            "north_america",
            "free",
            "ios",
            0.045,
            1200,
        ),
        (
            "onboarding",
            "new_users",
            "mobile",
            "paid_social",
            "0_30d",
            "europe",
            "free",
            "android",
            -0.018,
            900,
        ),
        (
            "onboarding",
            "new_users",
            "mobile",
            "organic",
            "31_90d",
            "north_america",
            "free",
            "ios",
            0.012,
            700,
        ),
        (
            "onboarding",
            "new_users",
            "desktop",
            "organic",
            "0_30d",
            "north_america",
            "free",
            "web",
            0.030,
            850,
        ),
        (
            "onboarding",
            "new_users",
            "desktop",
            "partner",
            "31_90d",
            "europe",
            "free",
            "web",
            -0.012,
            650,
        ),
        (
            "checkout",
            "returning",
            "mobile",
            "email",
            "90d_plus",
            "north_america",
            "plus",
            "ios",
            0.022,
            1000,
        ),
        (
            "checkout",
            "returning",
            "mobile",
            "paid_social",
            "31_90d",
            "europe",
            "free",
            "android",
            -0.026,
            600,
        ),
        (
            "checkout",
            "returning",
            "desktop",
            "email",
            "90d_plus",
            "north_america",
            "pro",
            "web",
            0.018,
            900,
        ),
        (
            "checkout",
            "returning",
            "desktop",
            "partner",
            "31_90d",
            "apac",
            "plus",
            "web",
            -0.008,
            500,
        ),
        (
            "retention",
            "returning",
            "mobile",
            "in_app",
            "90d_plus",
            "north_america",
            "plus",
            "ios",
            0.026,
            750,
        ),
        (
            "retention",
            "returning",
            "mobile",
            "push",
            "31_90d",
            "latin_america",
            "free",
            "android",
            -0.020,
            550,
        ),
        (
            "retention",
            "returning",
            "desktop",
            "email",
            "90d_plus",
            "europe",
            "pro",
            "web",
            0.014,
            650,
        ),
        (
            "retention",
            "returning",
            "desktop",
            "partner",
            "31_90d",
            "apac",
            "plus",
            "web",
            -0.006,
            450,
        ),
    ]
    columns = HIDDEN_COLUMNS + (LIFT_COLUMN, WEIGHT_COLUMN)
    return [dict(zip(columns, row, strict=True)) for row in rows]


def build_public_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> us.PublicDescentReport:
    """Build the public-descent audit for the synthetic experiment."""

    rows = synthetic_experiment_rows() if rows is None else rows
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
        title="Product Experiment Public-Segment Stability Audit",
        target_description="treatment minus control conversion lift",
        observed_label="Observed experiment lift",
        row_count=len(rows),
        row_count_label="Retained experiment cells",
        weight_column=WEIGHT_COLUMN,
    )


def build_launch_claim(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.01,
    max_added_columns: int = 1,
) -> us.ClaimVerificationReport:
    """Verify whether the positive-lift launch decision is invariant."""

    rows = synthetic_experiment_rows() if rows is None else rows
    claim = us.ReportingClaim(
        estimate_name="Positive product-experiment lift claim",
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
            label="treatment lift is nonnegative",
            pass_label="launch_or_continue",
            fail_label="hold_or_roll_back",
        ),
        max_added_columns=max_added_columns,
        target_description="treatment minus control conversion lift",
        observed_label="Observed experiment lift",
        title="Product Experiment Launch Claim Verification",
    )
    return claim.verify(rows)


def build_frontier(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.01,
    q_radius: float = 0.5,
) -> us.PublicRepresentationFrontier:
    """Search for small public segmentations that stabilize experiment lift."""

    rows = synthetic_experiment_rows() if rows is None else rows
    return us.public_representation_frontier(
        rows,
        base_public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=LIFT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
        min_cell_weights=(1.0,),
        ambiguity_limit=ambiguity_limit,
        search="beam",
        beam_width=8,
        max_added_columns=3,
        max_evaluations=96,
        title="Product Experiment Public Representation Frontier",
    )


def render_report(
    *,
    include_public_report: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.01,
) -> str:
    """Render the synthetic product-experiment stability report as Markdown."""

    rows = synthetic_experiment_rows()
    public_report = build_public_report(rows)
    claim_report = build_launch_claim(rows)
    observed = public_report.observed_value
    lower = public_report.interval.lower
    upper = public_report.interval.upper
    ambiguity = public_report.interval.diameter

    lines = [
        "# Product Experimentation / A/B Test Stability Example",
        "",
        "This synthetic demo audits an A/B-test report. The public report shows "
        "lift by product surface, reported user segment, and platform. The "
        "retained cells also track acquisition channel, tenure band, geography, "
        "plan type, and device type.",
        "",
        "The headline result looks launchable:",
        "",
        f"- Observed treatment-minus-control lift: {_percent(observed)}",
        "",
        "But if the public experiment mix is held fixed and hidden segment "
        "composition is allowed to shift inside those public buckets, the "
        "compatible lift ranges from:",
        "",
        f"```text\n{_percent(lower)} to {_percent(upper)}\n```",
        "",
        f"The ambiguity width is {_percent(ambiguity)}. Because the interval "
        "crosses zero, the apparent positive-lift launch conclusion is not "
        "invariant to the declared hidden-composition stress test.",
        "",
        "Plain English: the public experiment segments are too coarse to certify "
        "the launch decision. A different retained mix of acquisition channels, "
        "tenure bands, geographies, plans, or devices could preserve the same "
        "public segment mix while making the treatment look neutral or harmful.",
        "",
        "This is not a confidence interval and not a replacement for experiment "
        "standard errors. It is a representation-stability audit for retained "
        "experiment cells and the selected Q stress test.",
    ]

    if include_public_report:
        lines.extend(["", "## Public-Descent Audit", ""])
        lines.extend(_without_title(public_report.to_markdown()))

    lines.extend(["", "## Launch Decision Audit", ""])
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


def _without_title(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0]:
            lines = lines[1:]
    return lines


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f} percentage points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
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
        default=0.01,
        help="Frontier stability limit for experiment-lift ambiguity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = render_report(
        include_public_report=not args.no_public_report,
        include_frontier=not args.no_frontier,
        frontier_ambiguity_limit=args.frontier_ambiguity_limit,
    )
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
