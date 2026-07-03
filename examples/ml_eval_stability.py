"""AI / ML evaluation stability example.

The example audits whether a headline model-comparison benchmark result is
stable to hidden task-composition shifts. It is synthetic and dependency-free:
each row is a retained benchmark cell with an item count and a challenger-minus-
baseline score margin.

Run from the repository root with:

    uv run python examples/ml_eval_stability.py

Optionally write the Markdown report:

    uv run python examples/ml_eval_stability.py --output data/ml_eval_stability.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us


PUBLIC_COLUMNS = ("task_family", "difficulty", "language")
HIDDEN_REFINEMENTS = (
    "source_dataset",
    "prompt_template",
    "topic",
    "evaluator_group",
    "failure_mode",
)
HIDDEN_COLUMNS = PUBLIC_COLUMNS + HIDDEN_REFINEMENTS
CANDIDATE_REFINEMENTS = (
    "source_dataset",
    "prompt_template",
    "topic",
    "failure_mode",
    "evaluator_group",
)
MARGIN_COLUMN = "challenger_margin"
WEIGHT_COLUMN = "item_count"


def synthetic_eval_rows() -> list[dict[str, Any]]:
    """Return synthetic retained benchmark cells for a model-comparison audit."""

    rows = [
        (
            "qa",
            "easy",
            "en",
            "curated",
            "concise",
            "factual",
            "expert",
            "none",
            0.08,
            120,
        ),
        (
            "qa",
            "easy",
            "en",
            "web",
            "cot",
            "retrieval",
            "crowd",
            "retrieval",
            -0.02,
            80,
        ),
        (
            "qa",
            "hard",
            "en",
            "curated",
            "cot",
            "math",
            "expert",
            "reasoning",
            0.10,
            70,
        ),
        (
            "qa",
            "hard",
            "en",
            "adversarial",
            "concise",
            "math",
            "crowd",
            "reasoning",
            -0.06,
            60,
        ),
        (
            "qa",
            "hard",
            "en",
            "web",
            "verbose",
            "science",
            "crowd",
            "format",
            0.02,
            50,
        ),
        (
            "coding",
            "hard",
            "en",
            "unit_tests",
            "cot",
            "algorithm",
            "expert",
            "none",
            0.07,
            90,
        ),
        (
            "coding",
            "hard",
            "en",
            "notebooks",
            "concise",
            "data",
            "crowd",
            "tool_use",
            -0.04,
            70,
        ),
        (
            "coding",
            "hard",
            "en",
            "unit_tests",
            "debug",
            "debugging",
            "expert",
            "reasoning",
            0.03,
            40,
        ),
        (
            "safety",
            "hard",
            "en",
            "redteam",
            "adversarial",
            "jailbreak",
            "expert",
            "refusal",
            -0.08,
            60,
        ),
        (
            "safety",
            "hard",
            "en",
            "policy",
            "concise",
            "policy",
            "expert",
            "none",
            0.04,
            90,
        ),
        (
            "safety",
            "hard",
            "en",
            "redteam",
            "scenario",
            "self_harm",
            "crowd",
            "refusal",
            -0.03,
            50,
        ),
        (
            "qa",
            "easy",
            "es",
            "translated",
            "concise",
            "culture",
            "crowd",
            "translation",
            0.06,
            70,
        ),
        (
            "qa",
            "easy",
            "es",
            "translated",
            "cot",
            "culture",
            "expert",
            "reasoning",
            -0.01,
            50,
        ),
    ]
    columns = HIDDEN_COLUMNS + (MARGIN_COLUMN, WEIGHT_COLUMN)
    return [dict(zip(columns, row, strict=True)) for row in rows]


def build_public_report(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> us.PublicDescentReport:
    """Build the public-descent audit for the synthetic benchmark."""

    rows = synthetic_eval_rows() if rows is None else rows
    return us.public_descent_report(
        rows,
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=MARGIN_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q=q,
        q_radius=q_radius,
        top=top,
        title="AI Benchmark Public-Bucket Stability Audit",
        target_description="challenger minus baseline benchmark score",
        observed_label="Observed benchmark margin",
        row_count=len(rows),
        row_count_label="Retained benchmark cells",
        weight_column=WEIGHT_COLUMN,
    )


def build_leaderboard_claim(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    q_presets: Sequence[Any] = ("saturated",),
    ambiguity_limit: float = 0.025,
    max_added_columns: int = 2,
    max_evaluations: int = 64,
) -> us.ClaimVerificationReport:
    """Verify whether the benchmark winner is invariant to recomposition."""

    rows = synthetic_eval_rows() if rows is None else rows
    claim = us.ReportingClaim(
        estimate_name="Challenger beats baseline benchmark claim",
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=MARGIN_COLUMN,
        weight=WEIGHT_COLUMN,
        q_presets=q_presets,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        ambiguity_limit=ambiguity_limit,
        decision=us.threshold_decision(
            ">=",
            0.0,
            label="challenger margin is nonnegative",
            pass_label="challenger_wins_or_ties",
            fail_label="baseline_can_win",
        ),
        search="beam",
        beam_width=8,
        max_added_columns=max_added_columns,
        max_evaluations=max_evaluations,
        exact_required=False,
        target_description="challenger minus baseline benchmark score",
        observed_label="Observed benchmark margin",
        title="AI Benchmark Leaderboard Claim Verification",
    )
    return claim.verify(rows)


def build_frontier(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    ambiguity_limit: float = 0.02,
    q_radius: float = 0.5,
) -> us.PublicRepresentationFrontier:
    """Search for small public benchmark segmentations that stabilize the margin."""

    rows = synthetic_eval_rows() if rows is None else rows
    return us.public_representation_frontier(
        rows,
        base_public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=MARGIN_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q_presets=("saturated", us.q_bounded_shift(q_radius), "observed"),
        min_cell_weights=(1.0,),
        ambiguity_limit=ambiguity_limit,
        search="beam",
        beam_width=8,
        max_added_columns=3,
        max_evaluations=96,
        title="AI Benchmark Public Representation Frontier",
    )


def render_report(
    *,
    include_public_report: bool = True,
    include_frontier: bool = True,
    frontier_ambiguity_limit: float = 0.02,
) -> str:
    """Render the synthetic AI benchmark stability report as Markdown."""

    rows = synthetic_eval_rows()
    public_report = build_public_report(rows)
    claim_report = build_leaderboard_claim(rows)
    observed = public_report.observed_value
    lower = public_report.interval.lower
    upper = public_report.interval.upper
    ambiguity = public_report.interval.diameter

    lines = [
        "# AI / ML Evaluation Stability Example",
        "",
        "This synthetic demo audits a model-comparison benchmark. The public "
        "report shows results by task family, difficulty, and language. The "
        "retained cells also track source dataset, prompt template, topic, "
        "evaluator group, and failure mode.",
        "",
        "The headline margin says the challenger model beats the baseline:",
        "",
        f"- Observed challenger-minus-baseline margin: {_percent(observed)}",
        "",
        "But if the public benchmark mix is held fixed and hidden task "
        "composition is allowed to shift inside those public buckets, the "
        "compatible margin ranges from:",
        "",
        f"```text\n{_percent(lower)} to {_percent(upper)}\n```",
        "",
        f"The ambiguity width is {_percent(ambiguity)}. Because the interval "
        "crosses zero, the apparent leaderboard conclusion is not invariant to "
        "the declared hidden-composition stress test.",
        "",
        "Plain English: the public benchmark categories are too coarse to "
        "certify the model ranking. A different retained mix of prompt sources, "
        "templates, topics, evaluator groups, or failure modes could preserve "
        "the same public task mix while making the baseline competitive or "
        "better.",
        "",
        "This is not a confidence interval and not a claim about future benchmark "
        "sampling. It is a representation-stability audit for the retained "
        "benchmark cells and the selected Q stress test.",
    ]

    if include_public_report:
        lines.extend(["", "## Public-Descent Audit", ""])
        lines.extend(_without_title(public_report.to_markdown()))

    lines.extend(["", "## Leaderboard Decision Audit", ""])
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
        default=0.02,
        help="Frontier stability limit for benchmark-margin ambiguity.",
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
