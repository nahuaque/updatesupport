"""Generate reproducible benchmark-gallery reports.

The gallery writes Markdown reports to ``data/benchmark_gallery/`` by default.
That directory is intentionally gitignored: the source code and documentation
are tracked, while local benchmark outputs can be regenerated from the scripts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us

from examples.acic_2016 import (
    attach_oracle_effects,
    estimate_acic_effects_with_econml,
    load_acic_2016_source_rows,
    render_acic_report,
)
from examples.folktables_acs import TARGET_COLUMN, build_problem_from_rows
from examples.folktables_acs import load_folktables_rows, render_report
from examples.folktables_acs import synthetic_rows
from examples.folktables_acs_causal import (
    estimate_effects_with_econml as estimate_folktables_effects_with_econml,
)
from examples.folktables_acs_causal import (
    render_causal_report,
    synthetic_causal_source_rows,
)
from examples.conformal_reporting_stability import (
    render_report as render_conformal_report,
)
from examples.conformal_reporting_stability import synthetic_conformal_audit_rows
from examples.ml_eval_stability import render_report as render_ml_eval_report
from examples.ml_eval_stability import synthetic_eval_rows
from examples.product_experiment_stability import (
    render_report as render_product_experiment_report,
)
from examples.product_experiment_stability import synthetic_experiment_rows
from examples.revops_funnel_stability import render_report as render_revops_report
from examples.revops_funnel_stability import synthetic_funnel_rows
from examples.revops_funnel_trend_stability import (
    render_report as render_revops_trend_report,
)
from examples.revops_funnel_trend_stability import synthetic_trend_rows


DEFAULT_OUTPUT_DIR = Path("data/benchmark_gallery")
DEFAULT_ACIC_CSV = Path("data/acic_2016_p1_s1.csv")
FOLKTABLES_REAL_MIN_CELL_WEIGHT = 5.0


@dataclass(frozen=True)
class GalleryReport:
    """Status for one generated or skipped gallery report."""

    slug: str
    title: str
    description: str
    status: str
    path: Path | None = None
    row_count: int | None = None
    detail: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "generated"


class DeterministicEffectEstimator:
    """Small estimator used only for no-download deterministic gallery output."""

    def fit(self, y, treatment, *, X, sample_weight=None, inference=None):
        self.x_shape = X.shape
        return self

    def effect(self, X):
        return [0.05 + 0.001 * index for index in range(X.shape[0])]


def generate_benchmark_gallery(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    acic_csv: Path = DEFAULT_ACIC_CSV,
    include_real_folktables: bool = True,
    folktables_task: str = "income",
    folktables_states: Sequence[str] = ("CA",),
    folktables_year: int = 2018,
    folktables_horizon: str = "1-Year",
    folktables_sample: int | None = 10000,
    folktables_download: bool = False,
    skip_acic_econml: bool = False,
    acic_econml_sample: int | None = 1000,
    random_state: int = 0,
) -> tuple[GalleryReport, ...]:
    """Generate the benchmark gallery and return per-report statuses."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[GalleryReport] = []

    reports.append(_generate_ml_eval_report(output_dir))
    reports.append(_generate_conformal_report(output_dir))
    reports.append(_generate_product_experiment_report(output_dir))
    reports.append(_generate_revops_report(output_dir))
    reports.append(_generate_revops_trend_report(output_dir))
    reports.append(_generate_folktables_label_report(output_dir))
    if include_real_folktables:
        reports.append(
            _generate_folktables_real_report(
                output_dir=output_dir,
                task=folktables_task,
                states=folktables_states,
                year=folktables_year,
                horizon=folktables_horizon,
                sample_size=folktables_sample,
                download=folktables_download,
                random_state=random_state,
            )
        )
    reports.append(_generate_folktables_causal_report(output_dir, random_state))
    reports.extend(
        _generate_acic_reports(
            output_dir=output_dir,
            acic_csv=acic_csv,
            skip_acic_econml=skip_acic_econml,
            acic_econml_sample=acic_econml_sample,
            random_state=random_state,
        )
    )
    _write_index(output_dir, reports)
    return tuple(reports)


def _generate_ml_eval_report(output_dir: Path) -> GalleryReport:
    markdown = render_ml_eval_report()
    path = output_dir / "ml_eval_stability_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="ml_eval_stability_synthetic",
        title="AI / ML Evaluation Stability Synthetic Audit",
        description=(
            "No-download model-comparison benchmark example that audits whether "
            "a headline leaderboard margin survives hidden task-composition "
            "shifts inside public benchmark buckets."
        ),
        status="generated",
        path=path,
        row_count=len(synthetic_eval_rows()),
    )


def _generate_conformal_report(output_dir: Path) -> GalleryReport:
    markdown = render_conformal_report()
    path = output_dir / "conformal_reporting_stability_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="conformal_reporting_stability_synthetic",
        title="Conformal Prediction Reporting Stability Synthetic Audit",
        description=(
            "No-download split-conformal regression example that audits whether "
            "aggregate interval width, miscoverage, and threshold-crossing "
            "burden survive hidden customer-mix recomposition."
        ),
        status="generated",
        path=path,
        row_count=len(synthetic_conformal_audit_rows()),
    )


def _generate_product_experiment_report(output_dir: Path) -> GalleryReport:
    markdown = render_product_experiment_report()
    path = output_dir / "product_experiment_stability_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="product_experiment_stability_synthetic",
        title="Product Experimentation Stability Synthetic Audit",
        description=(
            "No-download A/B-test example that audits whether a positive "
            "reported lift survives hidden segment recomposition inside public "
            "experiment buckets."
        ),
        status="generated",
        path=path,
        row_count=len(synthetic_experiment_rows()),
    )


def _generate_revops_report(output_dir: Path) -> GalleryReport:
    markdown = render_revops_report()
    path = output_dir / "revops_funnel_stability_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="revops_funnel_stability_synthetic",
        title="RevOps Funnel Stability Synthetic Audit",
        description=(
            "No-download revenue-operations funnel example that audits whether "
            "a reported MQL-to-SQL health claim survives hidden pipeline-mix "
            "recomposition inside public segment buckets."
        ),
        status="generated",
        path=path,
        row_count=len(synthetic_funnel_rows()),
    )


def _generate_revops_trend_report(output_dir: Path) -> GalleryReport:
    markdown = render_revops_trend_report()
    path = output_dir / "revops_funnel_trend_stability_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="revops_funnel_trend_stability_synthetic",
        title="RevOps Funnel Trend Stability Synthetic Audit",
        description=(
            "No-download revenue-operations trend example that audits whether "
            "a reported Q/Q MQL-to-SQL improvement survives hidden pipeline-mix "
            "recomposition inside public segment buckets."
        ),
        status="generated",
        path=path,
        row_count=len(synthetic_trend_rows()),
    )


def _generate_folktables_label_report(output_dir: Path) -> GalleryReport:
    rows, public_columns, hidden_columns, candidate_columns = synthetic_rows()
    grouped = build_problem_from_rows(
        rows,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        target_column=TARGET_COLUMN,
        weight_column="weight",
    )
    markdown = render_report(
        task="income",
        grouped=grouped,
        rows=rows,
        candidate_columns=candidate_columns,
        top=8,
        min_cell_weight=1.0,
    )
    path = output_dir / "folktables_acs_income_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="folktables_acs_income_synthetic",
        title="Folktables ACSIncome Synthetic Label-Rate Audit",
        description=(
            "No-download ACSIncome-shaped example that audits target-rate "
            "stability under coarse public categories."
        ),
        status="generated",
        path=path,
        row_count=len(rows),
    )


def _generate_folktables_real_report(
    *,
    output_dir: Path,
    task: str,
    states: Sequence[str],
    year: int,
    horizon: str,
    sample_size: int | None,
    download: bool,
    random_state: int,
) -> GalleryReport:
    title = f"Folktables ACS{task.title()} Real Sample Label-Rate Audit"
    try:
        rows, public_columns, hidden_columns, candidate_columns = load_folktables_rows(
            task=task,
            states=states,
            year=year,
            horizon=horizon,
            download=download,
            sample_size=sample_size,
            random_state=random_state,
        )
        grouped = build_problem_from_rows(
            rows,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            target_column=TARGET_COLUMN,
            weight_column="weight" if rows and "weight" in rows[0] else None,
            min_cell_weight=FOLKTABLES_REAL_MIN_CELL_WEIGHT,
        )
        hidden_cells = len(grouped.problem.states)
        public_cells = len(grouped.problem.public_values)
        if hidden_cells < 4 or public_cells < 2:
            raise ValueError(
                "real Folktables data retained only "
                f"{hidden_cells} hidden cells across {public_cells} public cells"
            )
        markdown = render_report(
            task=task,
            grouped=grouped,
            rows=rows,
            candidate_columns=candidate_columns,
            top=8,
            min_cell_weight=FOLKTABLES_REAL_MIN_CELL_WEIGHT,
        )
    except (Exception, SystemExit) as exc:
        detail = str(exc)
        if not download:
            detail = f"{detail}; pass --folktables-download to fetch ACS data"
        return GalleryReport(
            slug=f"folktables_acs_{task}_real",
            title=title,
            description="Real sampled Folktables ACS label-rate report.",
            status="skipped",
            detail=detail,
        )

    state_slug = "_".join(str(state).lower() for state in states)
    path = output_dir / f"folktables_acs_{task}_{state_slug}_{year}_real.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug=f"folktables_acs_{task}_real",
        title=title,
        description=(
            "Real Folktables ACS sample that audits target-rate stability under "
            "coarse public reporting categories."
        ),
        status="generated",
        path=path,
        row_count=len(rows),
        detail=(
            f"states={','.join(states)}, year={year}, sample_size={sample_size}"
            if sample_size is not None
            else f"states={','.join(states)}, year={year}, full sample"
        ),
    )


def _generate_folktables_causal_report(
    output_dir: Path, random_state: int
) -> GalleryReport:
    rows, public_columns, hidden_columns, candidate_columns = (
        synthetic_causal_source_rows()
    )
    effect_result = estimate_folktables_effects_with_econml(
        rows,
        feature_columns=hidden_columns,
        random_state=random_state,
        estimator_factory=lambda _random_state: DeterministicEffectEstimator(),
    )
    markdown = render_causal_report(
        effect_result=effect_result,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        candidate_columns=candidate_columns,
        treatment_label="BA or graduate degree versus less than BA",
        outcome_label="ACSIncome target label",
        estimator_label="Deterministic test estimator",
        min_cell_weight=1.0,
        q=us.q_bounded_shift(0.5),
        q_radius=0.5,
        top=8,
    )
    path = output_dir / "folktables_acs_causal_synthetic.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="folktables_acs_causal_synthetic",
        title="Folktables ACS Synthetic Causal-Effect Audit",
        description=(
            "No-download causal handoff that computes row-level effects and "
            "audits effect-reporting stability."
        ),
        status="generated",
        path=path,
        row_count=effect_result.source_rows,
    )


def _generate_acic_reports(
    *,
    output_dir: Path,
    acic_csv: Path,
    skip_acic_econml: bool,
    acic_econml_sample: int | None,
    random_state: int,
) -> tuple[GalleryReport, ...]:
    if not acic_csv.exists():
        detail = f"ACIC CSV not found at {acic_csv}"
        return (
            GalleryReport(
                slug="acic_2016_oracle",
                title="ACIC 2016 Oracle SATT-Style Audit",
                description="Real ACIC report using simulated potential outcomes.",
                status="skipped",
                detail=detail,
            ),
            GalleryReport(
                slug="acic_2016_econml_estimated",
                title="ACIC 2016 EconML Estimated-Effect Audit",
                description="Real ACIC report using estimated row-level effects.",
                status="skipped",
                detail=detail,
            ),
        )

    source_rows, public_columns, hidden_columns, candidate_columns = (
        load_acic_2016_source_rows(
            acic_csv,
            weight_column=None,
            public_count=3,
            candidate_count=5,
        )
    )
    oracle_result = attach_oracle_effects(
        source_rows,
        feature_columns=hidden_columns,
    )
    oracle_markdown = render_acic_report(
        effect_result=oracle_result,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        candidate_columns=candidate_columns,
        min_cell_weight=5.0,
        q=us.q_bounded_shift(0.5),
        q_radius=0.5,
        top=8,
    )
    oracle_path = output_dir / "acic_2016_oracle.md"
    _write_markdown(oracle_path, oracle_markdown)
    reports = [
        GalleryReport(
            slug="acic_2016_oracle",
            title="ACIC 2016 Oracle SATT-Style Audit",
            description=(
                "Real ACIC report using simulated potential outcomes, so the "
                "audited effect is the oracle row-level effect."
            ),
            status="generated",
            path=oracle_path,
            row_count=oracle_result.source_rows,
        )
    ]

    if skip_acic_econml:
        reports.append(
            GalleryReport(
                slug="acic_2016_econml_estimated",
                title="ACIC 2016 EconML Estimated-Effect Audit",
                description="Real ACIC report using estimated row-level effects.",
                status="skipped",
                detail="--skip-acic-econml was passed",
            )
        )
        return tuple(reports)

    reports.append(
        _generate_acic_econml_report(
            output_dir=output_dir,
            acic_csv=acic_csv,
            acic_econml_sample=acic_econml_sample,
            random_state=random_state,
        )
    )
    return tuple(reports)


def _generate_acic_econml_report(
    *,
    output_dir: Path,
    acic_csv: Path,
    acic_econml_sample: int | None,
    random_state: int,
) -> GalleryReport:
    try:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            load_acic_2016_source_rows(
                acic_csv,
                weight_column=None,
                sample_size=acic_econml_sample,
                random_state=random_state,
                public_count=3,
                candidate_count=5,
            )
        )
        effect_result = estimate_acic_effects_with_econml(
            source_rows,
            feature_columns=hidden_columns,
            random_state=random_state,
        )
        markdown = render_acic_report(
            effect_result=effect_result,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            candidate_columns=candidate_columns,
            min_cell_weight=5.0,
            q=us.q_bounded_shift(0.5),
            q_radius=0.5,
            top=8,
        )
    except (ImportError, SystemExit) as exc:
        return GalleryReport(
            slug="acic_2016_econml_estimated",
            title="ACIC 2016 EconML Estimated-Effect Audit",
            description="Real ACIC report using estimated row-level effects.",
            status="skipped",
            detail=str(exc),
        )

    path = output_dir / "acic_2016_econml_estimated.md"
    _write_markdown(path, markdown)
    return GalleryReport(
        slug="acic_2016_econml_estimated",
        title="ACIC 2016 EconML Estimated-Effect Audit",
        description=(
            "Real ACIC report using EconML-estimated row-level effects instead "
            "of the oracle potential-outcome contrast."
        ),
        status="generated",
        path=path,
        row_count=effect_result.source_rows,
        detail=(
            f"sample_size={acic_econml_sample}"
            if acic_econml_sample is not None
            else "full CSV"
        ),
    )


def _write_index(output_dir: Path, reports: Sequence[GalleryReport]) -> None:
    lines = [
        "# updatesupport Benchmark Gallery",
        "",
        "These reports are generated artifacts. Regenerate them with:",
        "",
        "```bash",
        "uv run --extra examples --extra causal python examples/benchmark_gallery.py",
        "```",
        "",
        "| Case study | Status | Rows | Report | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for report in reports:
        rows = "" if report.row_count is None else str(report.row_count)
        link = (
            "" if report.path is None else f"[{report.path.name}]({report.path.name})"
        )
        notes = report.detail or report.description
        lines.append(
            f"| {report.title} | {report.status} | {rows} | {link} | {notes} |"
        )
    lines.extend(
        [
            "",
            "The ACIC oracle and estimated-effect reports are intentionally separate. "
            "The oracle report audits the simulated potential-outcome contrast; "
            "the estimated-effect report audits `tau_hat = estimator.effect(X)` "
            "from the causal first stage.",
        ]
    )
    _write_markdown(output_dir / "index.md", "\n".join(lines))


def _write_markdown(path: Path, markdown: str) -> None:
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--acic-csv", type=Path, default=DEFAULT_ACIC_CSV)
    parser.add_argument(
        "--include-real-folktables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt a real Folktables ACS report from cached or downloaded data.",
    )
    parser.add_argument(
        "--folktables-task",
        choices=["income", "employment"],
        default="income",
    )
    parser.add_argument("--folktables-states", nargs="+", default=["CA"])
    parser.add_argument("--folktables-year", type=int, default=2018)
    parser.add_argument("--folktables-horizon", default="1-Year")
    parser.add_argument("--folktables-sample", type=int, default=10000)
    parser.add_argument(
        "--folktables-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Folktables to download ACS data if it is not cached.",
    )
    parser.add_argument(
        "--skip-acic-econml",
        action="store_true",
        help="Generate the oracle ACIC report but skip the EconML first stage.",
    )
    parser.add_argument(
        "--acic-econml-sample",
        type=int,
        default=1000,
        help="Optional row sample for the ACIC EconML report; use 0 for full CSV.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sample = None if args.acic_econml_sample == 0 else args.acic_econml_sample
    reports = generate_benchmark_gallery(
        output_dir=args.output_dir,
        acic_csv=args.acic_csv,
        include_real_folktables=args.include_real_folktables,
        folktables_task=args.folktables_task,
        folktables_states=args.folktables_states,
        folktables_year=args.folktables_year,
        folktables_horizon=args.folktables_horizon,
        folktables_sample=args.folktables_sample,
        folktables_download=args.folktables_download,
        skip_acic_econml=args.skip_acic_econml,
        acic_econml_sample=sample,
        random_state=args.random_state,
    )
    for report in reports:
        location = "" if report.path is None else f" -> {report.path}"
        detail = "" if report.detail is None else f" ({report.detail})"
        print(f"{report.status}: {report.slug}{location}{detail}")
    print(f"index: {args.output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
