"""Analyst-facing public-descent reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

from .data import DataDiagnostic, GroupedProblem, TabularTarget, from_dataframe
from .environments import BatchedCvxpyEnvironments, CvxpyEnvironments
from .metrics import target_description as describe_target
from .problem import FiniteProblem
from .presets import (
    QPreset,
    normalize_q_preset,
    q_bounded_shift,
    q_description,
    q_name,
    resolve_q_environment,
)
from .results import ConstraintDual, TransportResult
from .targets import ProcedureTarget


_PARAMETERIZED_SENSITIVITY_PRESETS = frozenset(
    {"tv_budget", "chi_square_budget", "kl_budget", "wasserstein"}
)


@dataclass(frozen=True)
class PublicFiberDiagnostic:
    """Public-fiber contribution or point-range diagnostic."""

    public_value: tuple[Hashable, ...]
    public_mass: float
    hidden_cells: int
    fiber_range: float
    contribution: float | None
    min_state: tuple[Hashable, ...]
    min_value: float
    max_state: tuple[Hashable, ...]
    max_value: float
    decomposition_available: bool = True

    @property
    def contribution_available(self) -> bool:
        return self.contribution is not None

    @property
    def diagnostic_kind(self) -> str:
        if self.decomposition_available:
            return "additive_contribution"
        return "point_range"

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_value": self.public_value,
            "public_mass": self.public_mass,
            "hidden_cells": self.hidden_cells,
            "range": self.fiber_range,
            "contribution": self.contribution,
            "contribution_available": self.contribution_available,
            "decomposition_available": self.decomposition_available,
            "diagnostic_kind": self.diagnostic_kind,
            "min_state": self.min_state,
            "min_value": self.min_value,
            "max_state": self.max_state,
            "max_value": self.max_value,
        }


@dataclass(frozen=True)
class RefinementCandidate:
    """One-column public refinement ranked by ambiguity reduction."""

    column: str
    before_ambiguity: float
    after_ambiguity: float
    reduction: float
    reduction_percent: float
    public_cells: int

    @property
    def diameter(self) -> float:
        """Backward-compatible alias for the after-refinement ambiguity."""

        return self.after_ambiguity

    @property
    def reduction_fraction(self) -> float:
        return self.reduction_percent / 100.0

    @property
    def percent_reduction(self) -> float:
        """Alias for callers who prefer noun-first naming."""

        return self.reduction_percent

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "before_ambiguity": self.before_ambiguity,
            "after_ambiguity": self.after_ambiguity,
            "diameter": self.after_ambiguity,
            "reduction": self.reduction,
            "reduction_fraction": self.reduction_fraction,
            "reduction_percent": self.reduction_percent,
            "percent_reduction": self.reduction_percent,
            "public_cells": self.public_cells,
        }


@dataclass
class _RefinementSensitivityCache:
    baseline: GroupedProblem
    refined: dict[str, GroupedProblem]


@dataclass(frozen=True)
class RefinementSensitivityCandidate:
    """One refinement candidate aggregated over a sensitivity grid."""

    column: str
    evaluated_scenarios: int
    mean_before_ambiguity: float
    mean_after_ambiguity: float
    mean_reduction: float
    min_reduction: float
    max_reduction: float
    mean_reduction_percent: float
    min_reduction_percent: float
    max_reduction_percent: float
    positive_reduction_scenarios: int
    best_rank: int
    mean_rank: float
    worst_rank: int
    top_rank_count: int
    min_public_cells: int
    max_public_cells: int

    @property
    def rank_range(self) -> int:
        return self.worst_rank - self.best_rank

    @property
    def positive_reduction_share(self) -> float:
        if self.evaluated_scenarios == 0:
            return 0.0
        return self.positive_reduction_scenarios / self.evaluated_scenarios

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "evaluated_scenarios": self.evaluated_scenarios,
            "mean_before_ambiguity": self.mean_before_ambiguity,
            "mean_after_ambiguity": self.mean_after_ambiguity,
            "mean_reduction": self.mean_reduction,
            "min_reduction": self.min_reduction,
            "max_reduction": self.max_reduction,
            "mean_reduction_percent": self.mean_reduction_percent,
            "min_reduction_percent": self.min_reduction_percent,
            "max_reduction_percent": self.max_reduction_percent,
            "positive_reduction_scenarios": self.positive_reduction_scenarios,
            "positive_reduction_share": self.positive_reduction_share,
            "best_rank": self.best_rank,
            "mean_rank": self.mean_rank,
            "worst_rank": self.worst_rank,
            "rank_range": self.rank_range,
            "top_rank_count": self.top_rank_count,
            "min_public_cells": self.min_public_cells,
            "max_public_cells": self.max_public_cells,
        }


@dataclass(frozen=True)
class RefinementSensitivityRow:
    """One candidate's refinement score in one sensitivity scenario."""

    scenario: str
    column: str
    rank: int
    q_name: str
    q_description: str
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    before_ambiguity: float
    after_ambiguity: float
    reduction: float
    reduction_percent: float
    public_cells: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "column": self.column,
            "rank": self.rank,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "before_ambiguity": self.before_ambiguity,
            "after_ambiguity": self.after_ambiguity,
            "reduction": self.reduction,
            "reduction_percent": self.reduction_percent,
            "public_cells": self.public_cells,
        }


@dataclass(frozen=True)
class RefinementSensitivityScenario:
    """Scenario-level status for refinement sensitivity aggregation."""

    scenario: str
    q_name: str
    q_description: str
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    candidate_count: int = 0
    best_column: str | None = None
    best_reduction: float | None = None
    baseline_ambiguity: float | None = None
    status: str = "ok"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "candidate_count": self.candidate_count,
            "best_column": self.best_column,
            "best_reduction": self.best_reduction,
            "baseline_ambiguity": self.baseline_ambiguity,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class RefinementSensitivityReport:
    """Refinement recommendations aggregated over a sensitivity grid."""

    candidates: tuple[RefinementSensitivityCandidate, ...]
    scenarios: tuple[RefinementSensitivityScenario, ...]
    rows: tuple[RefinementSensitivityRow, ...]
    title: str = "Public Refinement Sensitivity Report"
    row_count: int | None = None

    @property
    def successful_scenarios(self) -> tuple[RefinementSensitivityScenario, ...]:
        return tuple(row for row in self.scenarios if row.status == "ok")

    @property
    def failed_scenarios(self) -> tuple[RefinementSensitivityScenario, ...]:
        return tuple(row for row in self.scenarios if row.status != "ok")

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "row_count": self.row_count,
            "candidates": [row.as_dict() for row in self.candidates],
            "scenarios": [row.as_dict() for row in self.scenarios],
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.row_count is not None:
            lines.extend([f"- Rows: {self.row_count}", ""])
        lines.extend(
            [
                "## Aggregate Summary",
                "",
                f"- Scenarios: {len(self.scenarios)}",
                f"- Successful scenarios: {len(self.successful_scenarios)}",
                f"- Failed scenarios: {len(self.failed_scenarios)}",
                f"- Candidate refinements evaluated: {len(self.candidates)}",
            ]
        )
        if self.candidates:
            top = self.candidates[0]
            lines.append(
                "- Top aggregate refinement: "
                f"`{top.column}` with mean reduction {top.mean_reduction:.4f}, "
                f"worst-case reduction {top.min_reduction:.4f}, "
                f"and mean rank {top.mean_rank:.2f}."
            )

        lines.extend(["", "## Interpretation", ""])
        lines.extend(_refinement_sensitivity_interpretation(self))
        lines.extend(
            [
                "",
                "## Aggregate Refinement Ranking",
                "",
                "| column | scenarios | mean reduction | worst reduction | best reduction | mean reduction pct | best rank | mean rank | worst rank | top rank count | public cells |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in self.candidates:
            public_cells = (
                f"{row.min_public_cells}"
                if row.min_public_cells == row.max_public_cells
                else f"{row.min_public_cells}-{row.max_public_cells}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row.column),
                        str(row.evaluated_scenarios),
                        _format_optional_float(row.mean_reduction),
                        _format_optional_float(row.min_reduction),
                        _format_optional_float(row.max_reduction),
                        f"{row.mean_reduction_percent:.1f}%",
                        str(row.best_rank),
                        f"{row.mean_rank:.2f}",
                        str(row.worst_rank),
                        str(row.top_rank_count),
                        public_cells,
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Scenario Summary",
                "",
                "| scenario | Q | min_cell_weight | hidden columns | candidates | best column | best reduction | baseline ambiguity | status |",
                "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in self.scenarios:
            hidden_columns = ", ".join(row.hidden_columns)
            status = row.status if row.error is None else f"error: {row.error}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row.scenario),
                        _escape_table(row.q_name),
                        f"{row.min_cell_weight:g}",
                        _escape_table(hidden_columns),
                        str(row.candidate_count),
                        _escape_table(row.best_column or ""),
                        _format_optional_float(row.best_reduction),
                        _format_optional_float(row.baseline_ambiguity),
                        _escape_table(status),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class StatisticalUncertainty:
    """Optional statistical uncertainty metadata supplied by an external workflow."""

    estimate: float | None = None
    standard_error: float | None = None
    lower: float | None = None
    upper: float | None = None
    confidence_level: float | None = None
    method: str | None = None
    label: str = "Statistical uncertainty"

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "label": self.label,
        }


@dataclass(frozen=True)
class CausalReportingStabilitySuite:
    """One-stop report object for causal-effect reporting stability."""

    primary: "PublicDescentReport"
    sensitivity: "SensitivityReport | None" = None
    refinement_sensitivity: RefinementSensitivityReport | None = None
    statistical_uncertainty: StatisticalUncertainty | None = None
    title: str = "Causal Reporting Stability Suite"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "primary": _public_descent_summary_dict(self.primary),
            "statistical_uncertainty": None
            if self.statistical_uncertainty is None
            else self.statistical_uncertainty.as_dict(),
            "sensitivity": None
            if self.sensitivity is None
            else {
                "summary": self.sensitivity.summary.as_dict(),
                "rows": [row.as_dict() for row in self.sensitivity.rows],
            },
            "refinement_sensitivity": None
            if self.refinement_sensitivity is None
            else {
                "candidates": [
                    row.as_dict() for row in self.refinement_sensitivity.candidates
                ],
                "scenarios": [
                    row.as_dict() for row in self.refinement_sensitivity.scenarios
                ],
                "rows": [row.as_dict() for row in self.refinement_sensitivity.rows],
            },
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Executive Summary",
            "",
            f"- Causal estimate / reported value: {self.primary.observed_value:.4f}",
            "- Hidden-composition ambiguity: "
            f"{self.primary.interval.diameter:.4f} "
            f"([{self.primary.interval.lower:.4f}, {self.primary.interval.upper:.4f}])",
            f"- Public adequate: {'yes' if self.primary.public_adequate else 'no'}",
            f"- Primary Q preset: {self.primary.grouped.q_name}",
        ]
        if self.statistical_uncertainty is not None:
            lines.append(
                "- Statistical uncertainty: "
                f"{_format_statistical_uncertainty(self.statistical_uncertainty)}"
            )
        if self.sensitivity is not None:
            summary = self.sensitivity.summary
            lines.append(
                "- Sensitivity grid: "
                f"{summary.successful_scenarios}/{summary.scenario_count} "
                "successful scenarios"
            )
            if summary.min_ambiguity is not None and summary.max_ambiguity is not None:
                lines.append(
                    "- Sensitivity ambiguity range: "
                    f"{summary.min_ambiguity:.4f} to {summary.max_ambiguity:.4f}"
                )
        if self.refinement_sensitivity is not None:
            top = (
                self.refinement_sensitivity.candidates[0]
                if self.refinement_sensitivity.candidates
                else None
            )
            if top is not None:
                lines.append(
                    "- Top robust public refinement: "
                    f"`{top.column}` with mean reduction {top.mean_reduction:.4f}"
                )

        lines.extend(
            [
                "",
                "## What This Suite Separates",
                "",
                "- Causal estimate: the supplied row-level or hidden-cell effect target, "
                "usually produced by a causal estimator outside `updatesupport`.",
                "- Statistical uncertainty: standard errors or confidence intervals "
                "supplied by that external estimator, when provided.",
                "- Hidden-composition ambiguity: the update-support partial-ID "
                "interval holding the public distribution fixed under the chosen Q.",
                "- Public refinement recommendations: variables that make the public "
                "reporting representation more stable, not causal adjustment advice.",
            ]
        )

        lines.extend(["", "## Causal Estimate", ""])
        lines.extend(_suite_causal_estimate_markdown(self.primary))

        if self.statistical_uncertainty is not None:
            lines.extend(
                [
                    "",
                    "## Statistical Uncertainty",
                    "",
                    _format_statistical_uncertainty(self.statistical_uncertainty),
                ]
            )

        lines.extend(["", "## Hidden-Composition Ambiguity", ""])
        lines.extend(_suite_hidden_ambiguity_markdown(self.primary))

        lines.extend(["", "## Sensitivity Scenarios", ""])
        lines.extend(_suite_sensitivity_review_markdown(self.sensitivity))

        lines.extend(["", "## Refinement Recommendations", ""])
        lines.extend(
            _suite_refinement_review_markdown(
                self.primary,
                self.refinement_sensitivity,
            )
        )

        lines.extend(
            _dual_diagnostics_markdown(
                self.primary.interval,
                grouped=self.primary.grouped,
            )
        )

        lines.extend(["", "## Limitations", ""])
        lines.extend(_model_review_limitations())

        lines.extend(["", "## Primary Reporting Audit", "", self.primary.to_markdown()])

        if self.sensitivity is not None:
            lines.extend(
                [
                    "",
                    "## Robustness Grid",
                    "",
                    self.sensitivity.to_markdown(),
                ]
            )

        if self.refinement_sensitivity is not None:
            lines.extend(
                [
                    "",
                    "## Sensitivity-Aware Refinements",
                    "",
                    self.refinement_sensitivity.to_markdown(),
                ]
            )

        return "\n".join(lines)


@dataclass(frozen=True)
class PublicDescentReport:
    """Structured public-descent audit with Markdown rendering."""

    grouped: GroupedProblem
    observed_value: float
    interval: TransportResult
    public_adequate: bool
    fibers: tuple[PublicFiberDiagnostic, ...]
    refinements: tuple[RefinementCandidate, ...]
    title: str = "Public Descent Report"
    target_description: str = "target value"
    observed_label: str = "Observed value"
    row_count: int | None = None
    row_count_label: str = "Rows"
    min_cell_weight: float | None = None
    diagnostics: tuple[DataDiagnostic, ...] = ()

    @property
    def fiber_decomposition_available(self) -> bool:
        return self.grouped.problem.target_contract.supports_fiber_decomposition

    @property
    def fiber_diagnostic_kind(self) -> str:
        if self.fiber_decomposition_available:
            return "additive_contribution"
        return "point_range"

    @property
    def top_fiber_contribution(self) -> float | None:
        if not self.fiber_decomposition_available:
            return None
        return sum(
            row.contribution for row in self.fibers if row.contribution is not None
        )

    @property
    def top_fiber_contribution_share(self) -> float | None:
        if not self.fiber_decomposition_available:
            return None
        if self.interval.diameter <= self.grouped.problem.tol:
            return 0.0
        if self.top_fiber_contribution is None:
            return None
        return self.top_fiber_contribution / self.interval.diameter

    @property
    def interval_contains_observed(self) -> bool:
        tol = self.grouped.problem.tol
        return (
            self.interval.lower - tol
            <= self.observed_value
            <= self.interval.upper + tol
        )

    def as_dict(self) -> dict[str, Any]:
        return _public_descent_summary_dict(self)

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)

    def to_markdown(self) -> str:
        grouped = self.grouped
        problem = grouped.problem
        lines = [
            f"# {self.title}",
            "",
        ]
        if self.row_count is not None:
            lines.append(f"- {self.row_count_label}: {self.row_count}")
        lines.extend(
            [
                f"- Hidden cells: {len(problem.states)}",
                f"- Public cells: {len(problem.public_values)}",
                f"- Public columns: {', '.join(grouped.public_columns)}",
                f"- Hidden columns: {', '.join(grouped.hidden_columns)}",
                f"- Q preset: {grouped.q_name}",
            ]
        )
        if self.min_cell_weight is not None:
            lines.append(f"- Minimum hidden-cell weight: {self.min_cell_weight:g}")
        lines.extend(
            [
                f"- {self.observed_label}: {self.observed_value:.4f}",
                f"- Public adequate: {'yes' if self.public_adequate else 'no'}",
                f"- Observed-law partial-ID interval: [{self.interval.lower:.4f}, {self.interval.upper:.4f}]",
                f"- Observed-law transport ambiguity: {self.interval.diameter:.4f}",
            ]
        )
        if self.fiber_decomposition_available:
            lines.append(
                f"- Top {len(self.fibers)} fiber contribution share: "
                f"{_percent(self.top_fiber_contribution_share)}"
            )
        else:
            lines.append(
                "- Public-fiber contribution decomposition: not additive for this target"
            )

        lines.extend(_target_contract_markdown(grouped))
        lines.extend(_procedure_target_markdown(grouped))

        lines.extend(
            [
                "",
                "## Statistical Interpretation",
                "",
                f"The estimand is the aggregate {self.target_description}. Each hidden cell "
                "gets its own empirical target value, and the current observed value "
                "is computed over the observed hidden-cell mix.",
                "",
                "The partial-ID interval fixes the observed public distribution and "
                "then applies the selected Q stress test: "
                f"{grouped.q_description}. Under that stress test, the aggregate "
                f"value can range from {self.interval.lower:.4f} to {self.interval.upper:.4f}. "
                f"The observed value {self.observed_value:.4f} "
                f"{'falls inside' if self.interval_contains_observed else 'does not fall inside'} "
                "that interval.",
                "",
                f"The transport ambiguity is the interval width, {self.interval.diameter:.4f}. "
                "It is a sensitivity / partial-identification diameter, not a "
                "sampling confidence interval. It does not include binomial standard "
                "errors, design weights, survey design uncertainty, model "
                "uncertainty, or uncertainty in the hidden-cell target values.",
                "",
                "Public adequacy asks whether the public categories alone determine "
                "the estimand under the chosen hidden-reweighting class. If public "
                "adequacy is `no`, then at least one public cell contains hidden cells "
                "with different target values, so the aggregate can move even when "
                "public cell shares are held fixed.",
            ]
        )
        if self.fiber_decomposition_available:
            lines.extend(
                [
                    "",
                    "For each public fiber below, `range` is the max-minus-min hidden-cell "
                    "target value inside that public cell. `contribution` is the fiber's "
                    "difference between the upper and lower transport witnesses; under "
                    "the saturated preset this equals `mass * range`. The listed top "
                    "fibers account for "
                    f"{_percent(self.top_fiber_contribution_share)} of total transport "
                    "ambiguity.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "For this target, public-fiber ambiguity is not additively "
                    "decomposable. The table below reports hidden-cell point-value "
                    "ranges for orientation, but its `contribution` column should "
                    "not be read as an additive share of the total interval.",
                ]
            )
        lines.extend(
            [
                "",
                "## What This Report Separates",
                "",
                f"- Causal estimate / reported value: `{self.observed_label}` is the "
                f"aggregate {self.target_description} supplied to `updatesupport`. "
                "For causal workflows, this is where the causal estimator enters. "
                "`updatesupport` does not identify the causal graph, fit the effect "
                "model, or change the supplied hidden-cell target values.",
                "- Statistical uncertainty: this report does not estimate standard "
                "errors, confidence intervals, bootstrap intervals, survey-design "
                "uncertainty, or uncertainty in the supplied causal/model estimates. "
                "Report those separately using the causal, statistical, or survey "
                "workflow that produced the target values.",
                "- Hidden-composition ambiguity: the partial-ID interval and its "
                "width are computed by holding the public distribution fixed and "
                f"applying the selected Q stress test. Here that ambiguity is "
                f"{self.interval.diameter:.4f}.",
                "- Public refinement recommendations: candidate refinements are "
                "variables that would make the public representation more stable "
                "for this supplied target. They are reporting and measurement "
                "recommendations, not causal adjustment recommendations by "
                "themselves.",
            ]
        )

        lines.extend(_data_diagnostics_markdown(self.diagnostics))

        lines.extend(_dual_diagnostics_markdown(self.interval, grouped=grouped))

        if self.fiber_decomposition_available:
            lines.extend(["", "## Worst Public Fibers"])
        else:
            lines.extend(["", "## Public Fiber Point Ranges"])

        for row in self.fibers:
            lines.append(f"- {_format_key(grouped.public_columns, row.public_value)}")
            if row.contribution is None:
                lines.append(
                    f"  mass={row.public_mass:.4f}, hidden_cells={row.hidden_cells}, "
                    f"point_range={row.fiber_range:.4f}"
                )
            else:
                lines.append(
                    f"  mass={row.public_mass:.4f}, hidden_cells={row.hidden_cells}, "
                    f"range={row.fiber_range:.4f}, contribution={row.contribution:.4f}"
                )
            lines.extend(
                [
                    f"  min: {row.min_value:.4f} at "
                    f"{_format_key(grouped.hidden_columns, row.min_state)}",
                    f"  max: {row.max_value:.4f} at "
                    f"{_format_key(grouped.hidden_columns, row.max_state)}",
                ]
            )

        if self.refinements:
            lines.extend(
                [
                    "",
                    "## One-Column Refinement Candidates",
                    "",
                    "Each row asks: what if this hidden column were promoted into the "
                    "public representation? `reduction` is the drop in transport "
                    "ambiguity from `before` to `after`, `reduction_pct` is the "
                    "percentage of baseline ambiguity removed, and `public_cells` is "
                    "the resulting number of public strata. This is a "
                    "measurement-value table: large reductions identify variables "
                    "that make the coarse public representation more stable, with "
                    "the usual tradeoff that more strata may increase sparsity. "
                    "These recommendations address hidden-composition ambiguity in "
                    "the reporting representation; they do not by themselves choose "
                    "causal adjustment variables or account for statistical "
                    "uncertainty.",
                ]
            )
            for row in self.refinements:
                lines.append(
                    f"- add {row.column}: before={row.before_ambiguity:.4f}, "
                    f"after={row.after_ambiguity:.4f}, "
                    f"reduction={row.reduction:.4f}, "
                    f"reduction_pct={row.reduction_percent:.1f}%, "
                    f"public_cells={row.public_cells}"
                )

        lines.extend(
            [
                "",
                "## Limitations",
                "",
                *_model_review_limitations(),
                "",
                "## Analyst Notes",
                "",
                "- Treat very small hidden cells cautiously. Raising `min_cell_weight` "
                "shrinks the state space and reduces noisy one-off hidden-cell "
                "target values, but it also changes the admissible hidden support.",
                "- A wide interval means the chosen public categories are not stable "
                "for this estimand under within-public-cell composition shift.",
                "- A narrow interval does not prove causal validity; it says this "
                "specific support/reweighting stress test leaves little residual "
                "ambiguity.",
            ]
        )
        return "\n".join(lines)


def public_descent_report(
    data: Any | GroupedProblem,
    *,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    target: TabularTarget | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    target_column: TabularTarget | None = None,
    weight_column: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    top: int = 10,
    min_cell_weight: float = 1.0,
    title: str = "Public Descent Report",
    target_description: str = "target value",
    observed_label: str = "Observed value",
    row_count: int | None = None,
    row_count_label: str = "Rows",
    q: Any | None = None,
    q_radius: float | None = None,
) -> PublicDescentReport:
    """Build an analyst-facing public-descent report.

    ``data`` may be a raw dataframe/row iterable or a precompiled
    :class:`GroupedProblem`. When ``data`` is precompiled, pass ``source_data``
    to compute one-column refinement candidates.
    """

    if top < 0:
        raise ValueError("top must be non-negative")

    candidate_refinements = _resolve_sequence_arg(
        candidate_refinements,
        candidate_columns,
        primary_name="candidate_refinements",
        alias_name="candidate_columns",
    )
    if isinstance(data, GroupedProblem):
        grouped = data
        refinement_data = source_data
        effective_q = q if q is not None else grouped.q
        if effective_q is None:
            effective_q = "saturated"
    else:
        compile_data, inferred_row_count = _repeatable_data(data)
        if row_count is None:
            row_count = inferred_row_count
        effective_q = "saturated" if q is None else q
        resolved_target = _resolve_scalar_arg(
            target,
            target_column,
            primary_name="target",
            alias_name="target_column",
        )
        grouped = from_dataframe(
            compile_data,
            public=public,
            hidden=hidden,
            target=resolved_target,
            weight=weight,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            weight_column=weight_column,
            min_cell_weight=min_cell_weight,
            q=effective_q,
            q_radius=q_radius,
        )
        refinement_data = source_data if source_data is not None else compile_data

    if (
        target_description == "target value"
        and grouped.target_procedure is not None
        and grouped.target_procedure.description
    ):
        target_description = grouped.target_procedure.description
    elif target_description == "target value" and grouped.target_column is not None:
        target_description = describe_target(grouped.target_column)

    interval = grouped.problem.global_transport_modulus()
    observed_value = _observed_value(grouped)
    fibers = public_fiber_diagnostics(grouped, top=top)
    diagnostics = (
        tuple(grouped.diagnostics.diagnostics)
        if grouped.diagnostics is not None
        else ()
    )
    if candidate_refinements:
        diagnostics = diagnostics + _candidate_refinement_diagnostics(
            candidate_refinements,
            public=grouped.public_columns,
            hidden=grouped.hidden_columns,
        )
    refinements: tuple[RefinementCandidate, ...] = ()
    if candidate_refinements:
        if refinement_data is None:
            raise ValueError(
                "source_data is required to compute refinements for a GroupedProblem"
            )
        refinement_target: TabularTarget = (
            grouped.target_procedure
            if grouped.target_procedure is not None
            else grouped.target_column
        )
        refinements = recommend_refinements(
            refinement_data,
            public=grouped.public_columns,
            hidden=grouped.hidden_columns,
            target=refinement_target,
            weight=weight if weight is not None else weight_column,
            candidate_refinements=candidate_refinements,
            min_cell_weight=min_cell_weight,
            q=effective_q,
            q_radius=q_radius,
            top=top,
        )

    return PublicDescentReport(
        grouped=grouped,
        observed_value=observed_value,
        interval=interval,
        public_adequate=grouped.problem.is_public_adequate(),
        fibers=fibers,
        refinements=refinements,
        title=title,
        target_description=target_description,
        observed_label=observed_label,
        row_count=row_count,
        row_count_label=row_count_label,
        min_cell_weight=min_cell_weight,
        diagnostics=diagnostics,
    )


def audit_effects(
    data: Any | GroupedProblem,
    *,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    effect: str | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    effect_column: str | None = None,
    weight_column: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    top: int = 10,
    min_cell_weight: float = 1.0,
    title: str = "Causal Effect Representation Stability Audit",
    effect_description: str = "estimated treatment effect",
    observed_label: str = "Observed effect estimate",
    row_count: int | None = None,
    row_count_label: str = "Rows",
    q: Any | None = None,
    q_radius: float | None = None,
) -> PublicDescentReport:
    """Audit whether public categories stably report estimated effects.

    This is a convenience wrapper around :func:`public_descent_report` for causal
    or uplift workflows. A causal library should produce the row-level,
    subgroup-level, or hidden-cell-level effect target; this function audits the
    reporting representation for that supplied target.
    """

    effect = _resolve_scalar_arg(
        effect,
        effect_column,
        primary_name="effect",
        alias_name="effect_column",
    )
    if effect is None and not isinstance(data, GroupedProblem):
        raise TypeError("audit_effects() missing required keyword argument: 'effect'")

    return public_descent_report(
        data,
        source_data=source_data,
        public=public,
        hidden=hidden,
        target=effect,
        weight=weight,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        weight_column=weight_column,
        candidate_refinements=candidate_refinements,
        candidate_columns=candidate_columns,
        top=top,
        min_cell_weight=min_cell_weight,
        title=title,
        target_description=effect_description,
        observed_label=observed_label,
        row_count=row_count,
        row_count_label=row_count_label,
        q=q,
        q_radius=q_radius,
    )


def causal_reporting_stability(
    data: Any | GroupedProblem,
    *,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    effect: str | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    effect_column: str | None = None,
    weight_column: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    min_cell_weight: float = 1.0,
    q: Any | None = None,
    q_radius: float | None = None,
    top: int = 10,
    include_sensitivity: bool = True,
    include_refinement_sensitivity: bool = True,
    sensitivity_min_cell_weights: Sequence[float] | None = None,
    sensitivity_hidden_sets: Sequence[Sequence[str]] | None = None,
    sensitivity_q_presets: Sequence[Any] | None = None,
    statistical_estimate: float | None = None,
    statistical_standard_error: float | None = None,
    statistical_interval: tuple[float, float] | None = None,
    statistical_confidence_level: float | None = None,
    statistical_method: str | None = None,
    statistical_label: str = "Statistical uncertainty",
    title: str = "Causal Reporting Stability Suite",
    primary_title: str = "Causal Effect Representation Stability Audit",
    raise_errors: bool = False,
) -> CausalReportingStabilitySuite:
    """Run the standard causal reporting-stability workflow.

    This packages the main effect audit, optional Q/min-cell/hidden-set
    sensitivity grid, optional sensitivity-aware refinement ranking, and
    externally supplied statistical uncertainty metadata into one report object.
    """

    effect = _resolve_scalar_arg(
        effect,
        effect_column,
        primary_name="effect",
        alias_name="effect_column",
    )
    weight = _resolve_scalar_arg(
        weight,
        weight_column,
        primary_name="weight",
        alias_name="weight_column",
    )
    candidate_refinements = _resolve_sequence_arg(
        candidate_refinements,
        candidate_columns,
        primary_name="candidate_refinements",
        alias_name="candidate_columns",
    )
    if candidate_refinements is None:
        candidate_refinements = ()

    raw_data = source_data
    row_count = None
    primary_data: Any | GroupedProblem = data
    if not isinstance(data, GroupedProblem):
        primary_data, row_count = _repeatable_data(data)
        if raw_data is None:
            raw_data = primary_data
        else:
            raw_data, _source_row_count = _repeatable_data(raw_data)
    elif raw_data is not None:
        raw_data, row_count = _repeatable_data(raw_data)

    primary = audit_effects(
        primary_data,
        source_data=raw_data,
        public=public,
        hidden=hidden,
        effect=effect,
        weight=weight,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        candidate_refinements=candidate_refinements,
        top=top,
        min_cell_weight=min_cell_weight,
        title=primary_title,
        effect_description="estimated treatment effect",
        observed_label="Observed effect estimate",
        row_count=row_count,
        q=q,
        q_radius=q_radius,
    )

    public_tuple = primary.grouped.public_columns
    hidden_tuple = primary.grouped.hidden_columns
    effect_name: TabularTarget = (
        primary.grouped.target_procedure
        if primary.grouped.target_procedure is not None
        else primary.grouped.target_column
    )
    if effect is not None:
        effect_name = effect
    grid_min_cell_weights = tuple(
        sensitivity_min_cell_weights
        if sensitivity_min_cell_weights is not None
        else (min_cell_weight,)
    )

    sensitivity = None
    if include_sensitivity and raw_data is not None:
        sensitivity = sensitivity_report(
            raw_data,
            public=public_tuple,
            hidden=hidden_tuple,
            target=effect_name,
            weight=weight,
            min_cell_weights=grid_min_cell_weights,
            hidden_sets=sensitivity_hidden_sets,
            q_presets=sensitivity_q_presets,
            title="Causal Effect Reporting Sensitivity Report",
            raise_errors=raise_errors,
        )

    refinement_sensitivity = None
    if (
        include_refinement_sensitivity
        and raw_data is not None
        and candidate_refinements
    ):
        refinement_sensitivity = recommend_refinements_sensitivity(
            raw_data,
            public=public_tuple,
            hidden=hidden_tuple,
            target=effect_name,
            candidate_refinements=candidate_refinements,
            weight=weight,
            min_cell_weights=grid_min_cell_weights,
            hidden_sets=sensitivity_hidden_sets,
            q_presets=sensitivity_q_presets,
            top=top,
            title="Causal Effect Public Refinement Sensitivity Report",
            raise_errors=raise_errors,
        )

    return CausalReportingStabilitySuite(
        primary=primary,
        sensitivity=sensitivity,
        refinement_sensitivity=refinement_sensitivity,
        statistical_uncertainty=_build_statistical_uncertainty(
            estimate=statistical_estimate,
            standard_error=statistical_standard_error,
            interval=statistical_interval,
            confidence_level=statistical_confidence_level,
            method=statistical_method,
            label=statistical_label,
        ),
        title=title,
    )


def public_fiber_diagnostics(
    grouped: GroupedProblem, *, top: int | None = 10
) -> tuple[PublicFiberDiagnostic, ...]:
    """Return public-fiber contribution or point-range diagnostics."""

    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    problem = grouped.problem
    additive = problem.target_contract.supports_fiber_decomposition
    interval = problem.global_transport_modulus() if additive else None
    rows = []
    for public_value in problem.public_values:
        states = problem.public_fibers[public_value]
        ordered_states = sorted(states, key=lambda state: problem.estimand_map[state])
        min_state = ordered_states[0]
        max_state = ordered_states[-1]
        fiber_range = problem.estimand_map[max_state] - problem.estimand_map[min_state]
        public_mass = grouped.public_law[public_value]
        contribution = public_mass * fiber_range if additive else None
        if (
            additive
            and interval is not None
            and interval.q_lower is not None
            and interval.q_upper is not None
        ):
            lower_value = sum(
                interval.q_lower[state] * problem.estimand_map[state]
                for state in states
            )
            upper_value = sum(
                interval.q_upper[state] * problem.estimand_map[state]
                for state in states
            )
            contribution = max(0.0, upper_value - lower_value)
        rows.append(
            PublicFiberDiagnostic(
                public_value=public_value,
                public_mass=public_mass,
                hidden_cells=len(states),
                fiber_range=fiber_range,
                contribution=contribution,
                min_state=min_state,
                min_value=problem.estimand_map[min_state],
                max_state=max_state,
                max_value=problem.estimand_map[max_state],
                decomposition_available=additive,
            )
        )
    if additive:
        rows.sort(
            key=lambda row: (
                0.0 if row.contribution is None else row.contribution,
                row.fiber_range,
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: (row.fiber_range, row.public_mass), reverse=True)
    return tuple(rows if top is None else rows[:top])


def recommend_refinements(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    weight: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int | None = 8,
) -> tuple[RefinementCandidate, ...]:
    """Rank one-column public refinements by transport-ambiguity reduction."""

    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    candidate_refinements = _resolve_sequence_arg(
        candidate_refinements,
        candidate_columns,
        primary_name="candidate_refinements",
        alias_name="candidate_columns",
    )
    if candidate_refinements is None:
        candidate_refinements = ()

    repeatable_data, _row_count = _repeatable_data(data)
    baseline = from_dataframe(
        repeatable_data,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        min_cell_weight=min_cell_weight,
        q=q,
        q_radius=q_radius,
    )
    baseline_diameter = baseline.problem.global_transport_modulus().diameter

    scores = []
    for column in candidate_refinements:
        if column in public:
            continue
        if column not in hidden:
            continue
        refined_public = tuple(public) + (column,)
        refined = from_dataframe(
            repeatable_data,
            public=refined_public,
            hidden=hidden,
            target=target,
            weight=weight,
            min_cell_weight=min_cell_weight,
            q=q,
            q_radius=q_radius,
        )
        diameter = refined.problem.global_transport_modulus().diameter
        reduction = baseline_diameter - diameter
        reduction_percent = (
            100.0 * reduction / baseline_diameter if baseline_diameter > 0 else 0.0
        )
        scores.append(
            RefinementCandidate(
                column=column,
                before_ambiguity=baseline_diameter,
                after_ambiguity=diameter,
                reduction=reduction,
                reduction_percent=reduction_percent,
                public_cells=len(refined.problem.public_values),
            )
        )

    scores.sort(key=lambda row: row.reduction, reverse=True)
    return tuple(scores if top is None else scores[:top])


def _parameterized_sensitivity_q(q: Any) -> QPreset | None:
    preset = normalize_q_preset(q)
    if preset is None:
        return None
    if preset.name not in _PARAMETERIZED_SENSITIVITY_PRESETS:
        return None
    backend = (preset.backend or "cvxpy").strip().lower()
    if backend not in {"cvxpy", "parameterized_cvxpy"}:
        return None
    if preset.name in {"tv_budget", "chi_square_budget", "kl_budget"}:
        if preset.radius is None or float(preset.radius) == 0.0:
            return None
    return QPreset(
        name=preset.name,
        radius=preset.radius,
        cost=preset.cost,
        backend="parameterized_cvxpy",
        solver=preset.solver,
        solver_options=preset.solver_options,
        settings=preset.settings,
    )


def _can_reuse_parameterized_problem(target: TabularTarget) -> bool:
    return not isinstance(target, ProcedureTarget)


def _parameterized_sensitivity_key(preset: QPreset) -> tuple[Any, ...]:
    cost_key = id(preset.cost) if preset.name == "wasserstein" else None
    return (
        preset.name,
        cost_key,
        preset.solver,
        _solver_options_key(preset.solver_options),
    )


def _solver_options_key(options: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    if options is None:
        return None
    return tuple(sorted((str(key), repr(value)) for key, value in options.items()))


def _batched_sensitivity_key(q: Any) -> tuple[Any, ...] | None:
    preset = normalize_q_preset(q)
    if preset is None or preset.name not in _PARAMETERIZED_SENSITIVITY_PRESETS:
        return None
    backend = (preset.backend or "cvxpy").strip().lower()
    if backend != "batched_cvxpy":
        return None
    if preset.name in {"tv_budget", "chi_square_budget", "kl_budget"}:
        if preset.radius is None or float(preset.radius) == 0.0:
            return None
    return _parameterized_sensitivity_key(preset)


def _batched_sensitivity_rows(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    weight: str | None,
    min_cell_weight: float,
    q_presets: Sequence[Any],
    scenarios: Sequence[str],
) -> tuple[SensitivityRow, ...]:
    grouped = from_dataframe(
        data,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        min_cell_weight=min_cell_weight,
        q="saturated",
    )
    scenario_builders = []
    first_preset: QPreset | None = None
    for q_preset in q_presets:
        preset = normalize_q_preset(q_preset)
        if preset is None:
            raise TypeError("batched sensitivity requires built-in Q presets")
        if first_preset is None:
            first_preset = preset
        runtime_preset = QPreset(
            name=preset.name,
            radius=preset.radius,
            cost=preset.cost,
            backend="cvxpy",
            solver=preset.solver,
            solver_options=preset.solver_options,
            settings=preset.settings,
        )
        q_environment = resolve_q_environment(
            runtime_preset,
            public_law=grouped.public_law,
            public_map=grouped.problem.public_map,
            cell_weights=grouped.cell_weights,
        )
        if not isinstance(q_environment.environment, CvxpyEnvironments):
            raise TypeError("batched sensitivity requires CVXPY-compatible Q presets")
        scenario_builders.append(tuple(q_environment.environment.constraint_builders))

    batched_env = BatchedCvxpyEnvironments(
        fixed_public_law=grouped.public_law,
        scenario_constraint_builders=tuple(scenario_builders),
        scenario_names=tuple(scenarios),
        solver=None if first_preset is None else first_preset.solver,
        solver_options=None if first_preset is None else first_preset.solver_options,
        name="batched sensitivity cvxpy",
    )
    problem = FiniteProblem(
        states=grouped.problem.states,
        public=grouped.problem.public_map,
        estimand=grouped.problem.target_functional,
        environments=batched_env,
        tol=grouped.problem.tol,
    )
    intervals = batched_env.batched_local_transport(
        problem,
        [grouped.public_law for _ in q_presets],
    )
    observed = _observed_value(grouped)
    rows = []
    for scenario, q_preset, interval in zip(
        scenarios,
        q_presets,
        intervals,
        strict=True,
    ):
        rows.append(
            SensitivityRow(
                scenario=scenario,
                q_name=q_name(q_preset),
                q_description=q_description(q_preset),
                min_cell_weight=float(min_cell_weight),
                hidden_columns=tuple(hidden),
                hidden_cells=len(problem.states),
                public_cells=len(problem.public_values),
                observed_value=observed,
                lower=interval.lower,
                upper=interval.upper,
                ambiguity=interval.diameter,
                public_adequate=interval.diameter <= problem.tol,
            )
        )
    return tuple(rows)


def _set_parameterized_radius(grouped: GroupedProblem, preset: QPreset) -> None:
    if preset.radius is None:
        raise ValueError(f"{preset.name} radius is required")
    set_parameter = getattr(grouped.problem.environments, "set_parameter", None)
    if set_parameter is None:
        raise TypeError("compiled Q environment does not support CVXPY parameters")
    set_parameter("radius", float(preset.radius))


def _sensitivity_row_from_grouped(
    grouped: GroupedProblem,
    *,
    scenario: str,
    q: Any,
    min_cell_weight: float,
    hidden_columns: Sequence[str],
) -> SensitivityRow:
    interval = grouped.problem.global_transport_modulus()
    return SensitivityRow(
        scenario=scenario,
        q_name=q_name(q),
        q_description=q_description(q),
        min_cell_weight=float(min_cell_weight),
        hidden_columns=tuple(hidden_columns),
        hidden_cells=len(grouped.problem.states),
        public_cells=len(grouped.problem.public_values),
        observed_value=_observed_value(grouped),
        lower=interval.lower,
        upper=interval.upper,
        ambiguity=interval.diameter,
        public_adequate=grouped.problem.is_public_adequate(),
    )


def _parameterized_refinement_candidates(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    candidate_refinements: Sequence[str],
    weight: str | None,
    min_cell_weight: float,
    q: QPreset,
    cache: _RefinementSensitivityCache | None,
) -> tuple[_RefinementSensitivityCache, tuple[RefinementCandidate, ...]]:
    if cache is None:
        baseline = from_dataframe(
            data,
            public=public,
            hidden=hidden,
            target=target,
            weight=weight,
            min_cell_weight=min_cell_weight,
            q=q,
        )
        refined: dict[str, GroupedProblem] = {}
        for column in candidate_refinements:
            if column in public:
                continue
            if column not in hidden:
                continue
            refined[column] = from_dataframe(
                data,
                public=tuple(public) + (column,),
                hidden=hidden,
                target=target,
                weight=weight,
                min_cell_weight=min_cell_weight,
                q=q,
            )
        cache = _RefinementSensitivityCache(baseline=baseline, refined=refined)

    _set_parameterized_radius(cache.baseline, q)
    baseline_diameter = cache.baseline.problem.global_transport_modulus().diameter

    scores = []
    for column, refined in cache.refined.items():
        _set_parameterized_radius(refined, q)
        diameter = refined.problem.global_transport_modulus().diameter
        reduction = baseline_diameter - diameter
        reduction_percent = (
            100.0 * reduction / baseline_diameter if baseline_diameter > 0 else 0.0
        )
        scores.append(
            RefinementCandidate(
                column=column,
                before_ambiguity=baseline_diameter,
                after_ambiguity=diameter,
                reduction=reduction,
                reduction_percent=reduction_percent,
                public_cells=len(refined.problem.public_values),
            )
        )

    scores.sort(key=lambda row: row.reduction, reverse=True)
    return cache, tuple(scores)


def recommend_refinements_sensitivity(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    weight: str | None = None,
    min_cell_weights: Sequence[float] = (1.0,),
    hidden_sets: Sequence[Sequence[str]] | None = None,
    q_presets: Sequence[Any] | None = None,
    top: int | None = 8,
    title: str = "Public Refinement Sensitivity Report",
    raise_errors: bool = False,
) -> RefinementSensitivityReport:
    """Aggregate one-column refinement rankings over a sensitivity grid."""

    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    candidate_refinements = _resolve_sequence_arg(
        candidate_refinements,
        candidate_columns,
        primary_name="candidate_refinements",
        alias_name="candidate_columns",
    )
    if candidate_refinements is None:
        candidate_refinements = ()

    repeatable_data, row_count = _repeatable_data(data)
    hidden_grid = tuple(hidden_sets) if hidden_sets is not None else (tuple(hidden),)
    q_grid = (
        tuple(q_presets)
        if q_presets is not None
        else (
            "saturated",
            q_bounded_shift(0.5),
            "observed",
        )
    )

    scenario_rows: list[RefinementSensitivityScenario] = []
    candidate_rows: list[RefinementSensitivityRow] = []
    scenario_index = 0
    for hidden_columns in hidden_grid:
        hidden_columns_tuple = tuple(hidden_columns)
        for min_cell_weight in min_cell_weights:
            parameterized_refinement_cache: dict[
                tuple[Any, ...], _RefinementSensitivityCache
            ] = {}
            for q_preset in q_grid:
                scenario_index += 1
                scenario = f"R{scenario_index}"
                try:
                    parameterized_q = _parameterized_sensitivity_q(q_preset)
                    if parameterized_q is None or not _can_reuse_parameterized_problem(
                        target
                    ):
                        candidates = recommend_refinements(
                            repeatable_data,
                            public=public,
                            hidden=hidden_columns_tuple,
                            target=target,
                            candidate_refinements=candidate_refinements,
                            weight=weight,
                            min_cell_weight=min_cell_weight,
                            q=q_preset,
                            top=None,
                        )
                    else:
                        cache_key = _parameterized_sensitivity_key(parameterized_q)
                        cache, candidates = _parameterized_refinement_candidates(
                            repeatable_data,
                            public=public,
                            hidden=hidden_columns_tuple,
                            target=target,
                            candidate_refinements=candidate_refinements,
                            weight=weight,
                            min_cell_weight=min_cell_weight,
                            q=parameterized_q,
                            cache=parameterized_refinement_cache.get(cache_key),
                        )
                        parameterized_refinement_cache[cache_key] = cache
                except Exception as exc:
                    if raise_errors:
                        raise
                    scenario_rows.append(
                        RefinementSensitivityScenario(
                            scenario=scenario,
                            q_name=q_name(q_preset),
                            q_description=q_description(q_preset),
                            min_cell_weight=float(min_cell_weight),
                            hidden_columns=hidden_columns_tuple,
                            status="error",
                            error=str(exc),
                        )
                    )
                    continue

                q_label = q_name(q_preset)
                q_details = q_description(q_preset)
                best = candidates[0] if candidates else None
                scenario_rows.append(
                    RefinementSensitivityScenario(
                        scenario=scenario,
                        q_name=q_label,
                        q_description=q_details,
                        min_cell_weight=float(min_cell_weight),
                        hidden_columns=hidden_columns_tuple,
                        candidate_count=len(candidates),
                        best_column=None if best is None else best.column,
                        best_reduction=None if best is None else best.reduction,
                        baseline_ambiguity=(
                            None if best is None else best.before_ambiguity
                        ),
                    )
                )
                for rank, row in enumerate(candidates, start=1):
                    candidate_rows.append(
                        RefinementSensitivityRow(
                            scenario=scenario,
                            column=row.column,
                            rank=rank,
                            q_name=q_label,
                            q_description=q_details,
                            min_cell_weight=float(min_cell_weight),
                            hidden_columns=hidden_columns_tuple,
                            before_ambiguity=row.before_ambiguity,
                            after_ambiguity=row.after_ambiguity,
                            reduction=row.reduction,
                            reduction_percent=row.reduction_percent,
                            public_cells=row.public_cells,
                        )
                    )

    aggregated = _aggregate_refinement_sensitivity(candidate_rows)
    return RefinementSensitivityReport(
        candidates=tuple(aggregated if top is None else aggregated[:top]),
        scenarios=tuple(scenario_rows),
        rows=tuple(candidate_rows),
        title=title,
        row_count=row_count,
    )


@dataclass(frozen=True)
class SensitivityRow:
    """One robustness scenario in a sensitivity report."""

    scenario: str
    q_name: str
    q_description: str
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    hidden_cells: int | None = None
    public_cells: int | None = None
    observed_value: float | None = None
    lower: float | None = None
    upper: float | None = None
    ambiguity: float | None = None
    public_adequate: bool | None = None
    status: str = "ok"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "hidden_cells": self.hidden_cells,
            "public_cells": self.public_cells,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class SensitivitySummary:
    """Aggregate summary of a sensitivity-report scenario grid."""

    scenario_count: int
    successful_scenarios: int
    failed_scenarios: int
    baseline_scenario: str | None
    lowest_ambiguity_scenario: str | None
    highest_ambiguity_scenario: str | None
    min_ambiguity: float | None
    max_ambiguity: float | None
    ambiguity_span: float | None
    public_adequacy_pattern: str
    observed_min: float | None
    observed_max: float | None
    observed_span: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_count": self.scenario_count,
            "successful_scenarios": self.successful_scenarios,
            "failed_scenarios": self.failed_scenarios,
            "baseline_scenario": self.baseline_scenario,
            "lowest_ambiguity_scenario": self.lowest_ambiguity_scenario,
            "highest_ambiguity_scenario": self.highest_ambiguity_scenario,
            "min_ambiguity": self.min_ambiguity,
            "max_ambiguity": self.max_ambiguity,
            "ambiguity_span": self.ambiguity_span,
            "public_adequacy_pattern": self.public_adequacy_pattern,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "observed_span": self.observed_span,
        }


@dataclass(frozen=True)
class SensitivityReport:
    """Robustness grid over Q presets, hidden sets, and min-cell thresholds."""

    rows: tuple[SensitivityRow, ...]
    title: str = "Public Descent Sensitivity Report"
    row_count: int | None = None

    @property
    def successful_rows(self) -> tuple[SensitivityRow, ...]:
        return tuple(
            row for row in self.rows if row.status == "ok" and row.ambiguity is not None
        )

    @property
    def failed_rows(self) -> tuple[SensitivityRow, ...]:
        return tuple(
            row for row in self.rows if row.status != "ok" or row.ambiguity is None
        )

    @property
    def summary(self) -> SensitivitySummary:
        return _sensitivity_summary(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "row_count": self.row_count,
            "summary": self.summary.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.row_count is not None:
            lines.extend([f"- Rows: {self.row_count}", ""])
        lines.extend(_sensitivity_summary_markdown(self.summary, self.rows))
        lines.extend(["", "## Interpretation", ""])
        lines.extend(_sensitivity_interpretation_markdown(self.summary, self.rows))
        lines.extend(
            [
                "",
                "## Scenario Table",
                "",
            ]
        )
        lines.extend(
            [
                "| scenario | Q | min_cell_weight | hidden columns | hidden cells | public cells | observed | lower | upper | ambiguity | public adequate | status |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in self.rows:
            hidden_columns = ", ".join(row.hidden_columns)
            adequate = (
                ""
                if row.public_adequate is None
                else ("yes" if row.public_adequate else "no")
            )
            status = row.status if row.error is None else f"error: {row.error}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row.scenario),
                        _escape_table(row.q_name),
                        f"{row.min_cell_weight:g}",
                        _escape_table(hidden_columns),
                        _format_optional_int(row.hidden_cells),
                        _format_optional_int(row.public_cells),
                        _format_optional_float(row.observed_value),
                        _format_optional_float(row.lower),
                        _format_optional_float(row.upper),
                        _format_optional_float(row.ambiguity),
                        adequate,
                        _escape_table(status),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


def sensitivity_report(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    weight: str | None = None,
    min_cell_weights: Sequence[float] = (1.0,),
    hidden_sets: Sequence[Sequence[str]] | None = None,
    q_presets: Sequence[Any] | None = None,
    title: str = "Public Descent Sensitivity Report",
    raise_errors: bool = False,
) -> SensitivityReport:
    """Run robustness checks over Q presets, hidden sets, and cell thresholds."""

    repeatable_data, row_count = _repeatable_data(data)
    hidden_grid = tuple(hidden_sets) if hidden_sets is not None else (tuple(hidden),)
    q_grid = (
        tuple(q_presets)
        if q_presets is not None
        else (
            "saturated",
            q_bounded_shift(0.5),
            "observed",
        )
    )

    rows: list[SensitivityRow] = []
    scenario_index = 0
    for hidden_columns in hidden_grid:
        hidden_columns_tuple = tuple(hidden_columns)
        for min_cell_weight in min_cell_weights:
            parameterized_grouped_cache: dict[tuple[Any, ...], GroupedProblem] = {}
            q_index = 0
            while q_index < len(q_grid):
                batch_key = _batched_sensitivity_key(q_grid[q_index])
                if batch_key is not None and _can_reuse_parameterized_problem(target):
                    batch_end = q_index + 1
                    while (
                        batch_end < len(q_grid)
                        and _batched_sensitivity_key(q_grid[batch_end]) == batch_key
                    ):
                        batch_end += 1
                    if batch_end - q_index > 1:
                        scenario_names = tuple(
                            f"S{scenario_number}"
                            for scenario_number in range(
                                scenario_index + 1,
                                scenario_index + 1 + (batch_end - q_index),
                            )
                        )
                        try:
                            rows.extend(
                                _batched_sensitivity_rows(
                                    repeatable_data,
                                    public=public,
                                    hidden=hidden_columns_tuple,
                                    target=target,
                                    weight=weight,
                                    min_cell_weight=min_cell_weight,
                                    q_presets=q_grid[q_index:batch_end],
                                    scenarios=scenario_names,
                                )
                            )
                            scenario_index += batch_end - q_index
                            q_index = batch_end
                            continue
                        except Exception as exc:
                            if raise_errors:
                                raise
                            for q_preset, scenario in zip(
                                q_grid[q_index:batch_end],
                                scenario_names,
                                strict=True,
                            ):
                                rows.append(
                                    SensitivityRow(
                                        scenario=scenario,
                                        q_name=q_name(q_preset),
                                        q_description=q_description(q_preset),
                                        min_cell_weight=float(min_cell_weight),
                                        hidden_columns=hidden_columns_tuple,
                                        status="error",
                                        error=str(exc),
                                    )
                                )
                            scenario_index += batch_end - q_index
                            q_index = batch_end
                            continue

                q_preset = q_grid[q_index]
                scenario_index += 1
                scenario = f"S{scenario_index}"
                try:
                    parameterized_q = _parameterized_sensitivity_q(q_preset)
                    if parameterized_q is None or not _can_reuse_parameterized_problem(
                        target
                    ):
                        report = public_descent_report(
                            repeatable_data,
                            public=public,
                            hidden=hidden_columns_tuple,
                            target=target,
                            weight=weight,
                            min_cell_weight=min_cell_weight,
                            q=q_preset,
                            top=0,
                        )
                    else:
                        cache_key = _parameterized_sensitivity_key(parameterized_q)
                        grouped = parameterized_grouped_cache.get(cache_key)
                        if grouped is None:
                            grouped = from_dataframe(
                                repeatable_data,
                                public=public,
                                hidden=hidden_columns_tuple,
                                target=target,
                                weight=weight,
                                min_cell_weight=min_cell_weight,
                                q=parameterized_q,
                            )
                            parameterized_grouped_cache[cache_key] = grouped
                        _set_parameterized_radius(grouped, parameterized_q)
                        rows.append(
                            _sensitivity_row_from_grouped(
                                grouped,
                                scenario=scenario,
                                q=q_preset,
                                min_cell_weight=min_cell_weight,
                                hidden_columns=hidden_columns_tuple,
                            )
                        )
                        q_index += 1
                        continue
                except Exception as exc:
                    if raise_errors:
                        raise
                    rows.append(
                        SensitivityRow(
                            scenario=scenario,
                            q_name=q_name(q_preset),
                            q_description=q_description(q_preset),
                            min_cell_weight=float(min_cell_weight),
                            hidden_columns=hidden_columns_tuple,
                            status="error",
                            error=str(exc),
                        )
                    )
                    q_index += 1
                    continue

                grouped = report.grouped
                rows.append(
                    SensitivityRow(
                        scenario=scenario,
                        q_name=grouped.q_name,
                        q_description=grouped.q_description,
                        min_cell_weight=float(min_cell_weight),
                        hidden_columns=hidden_columns_tuple,
                        hidden_cells=len(grouped.problem.states),
                        public_cells=len(grouped.problem.public_values),
                        observed_value=report.observed_value,
                        lower=report.interval.lower,
                        upper=report.interval.upper,
                        ambiguity=report.interval.diameter,
                        public_adequate=report.public_adequate,
                    )
                )
                q_index += 1

    return SensitivityReport(rows=tuple(rows), title=title, row_count=row_count)


def _aggregate_refinement_sensitivity(
    rows: Sequence[RefinementSensitivityRow],
) -> tuple[RefinementSensitivityCandidate, ...]:
    by_column: dict[str, list[RefinementSensitivityRow]] = {}
    for row in rows:
        by_column.setdefault(row.column, []).append(row)

    candidates = []
    for column, column_rows in by_column.items():
        reductions = [row.reduction for row in column_rows]
        reduction_percents = [row.reduction_percent for row in column_rows]
        ranks = [row.rank for row in column_rows]
        before_values = [row.before_ambiguity for row in column_rows]
        after_values = [row.after_ambiguity for row in column_rows]
        public_cells = [row.public_cells for row in column_rows]
        candidates.append(
            RefinementSensitivityCandidate(
                column=column,
                evaluated_scenarios=len(column_rows),
                mean_before_ambiguity=_mean(before_values),
                mean_after_ambiguity=_mean(after_values),
                mean_reduction=_mean(reductions),
                min_reduction=min(reductions),
                max_reduction=max(reductions),
                mean_reduction_percent=_mean(reduction_percents),
                min_reduction_percent=min(reduction_percents),
                max_reduction_percent=max(reduction_percents),
                positive_reduction_scenarios=sum(
                    1 for reduction in reductions if reduction > 1e-12
                ),
                best_rank=min(ranks),
                mean_rank=_mean([float(rank) for rank in ranks]),
                worst_rank=max(ranks),
                top_rank_count=sum(1 for rank in ranks if rank == 1),
                min_public_cells=min(public_cells),
                max_public_cells=max(public_cells),
            )
        )

    candidates.sort(
        key=lambda row: (
            row.mean_reduction,
            row.min_reduction,
            row.top_rank_count,
            -row.mean_rank,
        ),
        reverse=True,
    )
    return tuple(candidates)


def _refinement_sensitivity_interpretation(
    report: RefinementSensitivityReport,
) -> list[str]:
    if not report.successful_scenarios:
        return [
            "- No refinement scenario completed successfully. Inspect the scenario "
            "summary table before interpreting candidate rankings.",
            "- Failed scenarios do not imply that a candidate is useless; they mean "
            "the requested refinement grid could not be evaluated.",
        ]

    lines = [
        "- Candidates are ranked by average ambiguity reduction across successful "
        "scenarios, with worst-case reduction and rank stability shown beside the "
        "average. A robust refinement should have positive average reduction, "
        "nonnegative worst-case reduction, and a stable rank.",
    ]
    if report.failed_scenarios:
        lines.append(
            "- Some scenarios failed. The aggregate ranking uses only successful "
            "candidate-scenario evaluations."
        )
    if not report.candidates:
        lines.append(
            "- No candidate refinement was evaluated successfully. Check that "
            "`candidate_refinements` are included in the tested hidden columns and "
            "are not already public columns."
        )
        return lines

    top = report.candidates[0]
    if top.min_reduction > 1e-12:
        lines.append(
            f"- `{top.column}` reduces ambiguity in every evaluated scenario where "
            "it appears, making it the strongest robust refinement in this grid."
        )
    elif top.mean_reduction > 1e-12:
        lines.append(
            f"- `{top.column}` has the largest average reduction, but its worst-case "
            "reduction is zero or negative. Treat it as scenario-dependent rather "
            "than uniformly robust."
        )
    else:
        lines.append(
            "- No evaluated refinement has positive average ambiguity reduction in "
            "this grid."
        )

    unstable = [
        row
        for row in report.candidates
        if row.evaluated_scenarios > 1 and row.rank_range > 0
    ]
    if unstable:
        lines.append(
            "- At least one candidate changes rank across scenarios. Use the "
            "`best rank`, `mean rank`, and `worst rank` columns to distinguish "
            "broadly useful refinements from preset- or threshold-specific ones."
        )
    else:
        lines.append(
            "- Candidate ranks are stable across the evaluated scenarios in this grid."
        )
    return lines


def _sensitivity_summary(rows: Sequence[SensitivityRow]) -> SensitivitySummary:
    successful = tuple(
        row for row in rows if row.status == "ok" and row.ambiguity is not None
    )
    failed = tuple(row for row in rows if row.status != "ok" or row.ambiguity is None)

    if not successful:
        return SensitivitySummary(
            scenario_count=len(rows),
            successful_scenarios=0,
            failed_scenarios=len(failed),
            baseline_scenario=None,
            lowest_ambiguity_scenario=None,
            highest_ambiguity_scenario=None,
            min_ambiguity=None,
            max_ambiguity=None,
            ambiguity_span=None,
            public_adequacy_pattern="unavailable",
            observed_min=None,
            observed_max=None,
            observed_span=None,
        )

    lowest = min(successful, key=lambda row: row.ambiguity or 0.0)
    highest = max(successful, key=lambda row: row.ambiguity or 0.0)
    observed_values = [
        row.observed_value for row in successful if row.observed_value is not None
    ]
    observed_min = min(observed_values) if observed_values else None
    observed_max = max(observed_values) if observed_values else None
    adequacy_values = {
        row.public_adequate for row in successful if row.public_adequate is not None
    }

    return SensitivitySummary(
        scenario_count=len(rows),
        successful_scenarios=len(successful),
        failed_scenarios=len(failed),
        baseline_scenario=successful[0].scenario,
        lowest_ambiguity_scenario=lowest.scenario,
        highest_ambiguity_scenario=highest.scenario,
        min_ambiguity=lowest.ambiguity,
        max_ambiguity=highest.ambiguity,
        ambiguity_span=(highest.ambiguity or 0.0) - (lowest.ambiguity or 0.0),
        public_adequacy_pattern=_public_adequacy_pattern(adequacy_values),
        observed_min=observed_min,
        observed_max=observed_max,
        observed_span=(
            None
            if observed_min is None or observed_max is None
            else observed_max - observed_min
        ),
    )


def _sensitivity_summary_markdown(
    summary: SensitivitySummary,
    rows: Sequence[SensitivityRow],
) -> list[str]:
    lines = [
        "## Scenario Summary",
        "",
        f"- Scenarios: {summary.scenario_count}",
        f"- Successful scenarios: {summary.successful_scenarios}",
        f"- Failed scenarios: {summary.failed_scenarios}",
        f"- Public adequacy pattern: {summary.public_adequacy_pattern}",
    ]

    if summary.baseline_scenario is not None:
        baseline = _row_by_scenario(rows, summary.baseline_scenario)
        lines.append(f"- Reference scenario: {_scenario_label(baseline)}")

    if summary.min_ambiguity is not None and summary.max_ambiguity is not None:
        low = _row_by_scenario(rows, summary.lowest_ambiguity_scenario)
        high = _row_by_scenario(rows, summary.highest_ambiguity_scenario)
        lines.extend(
            [
                "- Ambiguity range across successful scenarios: "
                f"{summary.min_ambiguity:.4f} to {summary.max_ambiguity:.4f} "
                f"(span {_format_optional_float(summary.ambiguity_span)})",
                f"- Lowest-ambiguity scenario: {_scenario_label(low)}",
                f"- Highest-ambiguity scenario: {_scenario_label(high)}",
            ]
        )

    if summary.observed_min is not None and summary.observed_max is not None:
        lines.append(
            "- Observed-value range across successful scenarios: "
            f"{summary.observed_min:.4f} to {summary.observed_max:.4f} "
            f"(span {_format_optional_float(summary.observed_span)})"
        )

    return lines


def _sensitivity_interpretation_markdown(
    summary: SensitivitySummary,
    rows: Sequence[SensitivityRow],
) -> list[str]:
    if summary.successful_scenarios == 0:
        return [
            "- No scenario completed successfully. Inspect the `status` column for "
            "compile or solver errors before interpreting the sensitivity grid.",
            "- The failed rows do not imply public adequacy or inadequacy; they mean "
            "the requested scenario was not evaluated.",
        ]

    lines = [
        "- The sensitivity grid varies the Q preset, the retained hidden state "
        "space, and/or `min_cell_weight`. The ambiguity values are "
        "hidden-composition sensitivity results, not confidence intervals.",
    ]

    if summary.failed_scenarios:
        lines.append(
            "- Some scenarios failed. Treat the successful rows as evaluated "
            "stress tests and inspect failed rows before comparing the full grid."
        )

    if summary.public_adequacy_pattern == "mixed":
        lines.append(
            "- Public adequacy changes across successful scenarios. The conclusion "
            "depends on the tested Q preset, hidden state space, or sparsity "
            "threshold."
        )
    elif summary.public_adequacy_pattern == "all adequate":
        lines.append(
            "- Public adequacy is `yes` in every successful scenario, so the tested "
            "public representation determines the supplied target under this grid."
        )
    elif summary.public_adequacy_pattern == "all inadequate":
        lines.append(
            "- Public adequacy is `no` in every successful scenario, so the tested "
            "public representation leaves residual hidden-composition ambiguity "
            "throughout this grid."
        )

    if summary.ambiguity_span is not None:
        if summary.ambiguity_span <= 1e-12:
            lines.append(
                "- Ambiguity is unchanged across successful scenarios. In this grid, "
                "the headline ambiguity is insensitive to the tested knobs."
            )
        else:
            high = _row_by_scenario(rows, summary.highest_ambiguity_scenario)
            lines.append(
                "- The highest-ambiguity scenario is the conservative row to cite "
                f"when stress-test robustness is the priority: {_scenario_label(high)}."
            )

    if summary.observed_span is not None and summary.observed_span > 1e-12:
        lines.append(
            "- The observed value changes across successful scenarios. This usually "
            "means a hidden-set choice or `min_cell_weight` threshold changed the "
            "retained support, so compare those rows as different retained-support "
            "analyses."
        )
    else:
        lines.append(
            "- The observed value is unchanged across successful scenarios, so the "
            "grid is isolating hidden-composition stress-test choices rather than "
            "changing the retained observed estimand."
        )

    return lines


def _public_adequacy_pattern(values: set[bool]) -> str:
    if values == {True}:
        return "all adequate"
    if values == {False}:
        return "all inadequate"
    if values == {True, False}:
        return "mixed"
    return "unavailable"


def _row_by_scenario(
    rows: Sequence[SensitivityRow], scenario: str | None
) -> SensitivityRow | None:
    if scenario is None:
        return None
    for row in rows:
        if row.scenario == scenario:
            return row
    return None


def _scenario_label(row: SensitivityRow | None) -> str:
    if row is None:
        return ""
    hidden_columns = ", ".join(row.hidden_columns)
    return (
        f"{row.scenario} (`{row.q_name}`, min_cell_weight={row.min_cell_weight:g}, "
        f"hidden={hidden_columns})"
    )


def _observed_value(grouped: GroupedProblem) -> float:
    return grouped.problem.psi(grouped.cell_weights)


def _build_statistical_uncertainty(
    *,
    estimate: float | None,
    standard_error: float | None,
    interval: tuple[float, float] | None,
    confidence_level: float | None,
    method: str | None,
    label: str,
) -> StatisticalUncertainty | None:
    lower = None
    upper = None
    if interval is not None:
        lower, upper = float(interval[0]), float(interval[1])
    if (
        estimate is None
        and standard_error is None
        and lower is None
        and upper is None
        and confidence_level is None
        and method is None
    ):
        return None
    return StatisticalUncertainty(
        estimate=None if estimate is None else float(estimate),
        standard_error=None if standard_error is None else float(standard_error),
        lower=lower,
        upper=upper,
        confidence_level=None if confidence_level is None else float(confidence_level),
        method=method,
        label=label,
    )


def _format_statistical_uncertainty(row: StatisticalUncertainty) -> str:
    parts = []
    if row.estimate is not None:
        parts.append(f"estimate={row.estimate:.4f}")
    if row.standard_error is not None:
        parts.append(f"se={row.standard_error:.4f}")
    if row.lower is not None and row.upper is not None:
        parts.append(f"interval=[{row.lower:.4f}, {row.upper:.4f}]")
    if row.confidence_level is not None:
        level = row.confidence_level
        label = f"{100.0 * level:.1f}%" if level <= 1.0 else f"{level:.1f}%"
        parts.append(f"confidence={label}")
    if row.method:
        parts.append(f"method={row.method}")
    detail = "; ".join(parts) if parts else "supplied"
    return f"{row.label}: {detail}"


def _suite_causal_estimate_markdown(report: PublicDescentReport) -> list[str]:
    return [
        f"- Reported value: {report.observed_value:.4f} (`{report.observed_label}`).",
        f"- Target: aggregate {report.target_description}.",
        f"- Public representation: {', '.join(report.grouped.public_columns)}.",
        f"- Hidden state space: {len(report.grouped.problem.states)} retained "
        f"hidden cells across {len(report.grouped.problem.public_values)} public "
        "cells.",
        f"- Stress-test class: {report.grouped.q_name} "
        f"({report.grouped.q_description}).",
    ]


def _suite_hidden_ambiguity_markdown(report: PublicDescentReport) -> list[str]:
    adequate = "yes" if report.public_adequate else "no"
    lines = [
        "- Partial-ID interval under fixed public law: "
        f"[{report.interval.lower:.4f}, {report.interval.upper:.4f}].",
        f"- Hidden-composition ambiguity: {report.interval.diameter:.4f}.",
        f"- Public adequate under this stress test: {adequate}.",
        "- Observed value position: "
        f"{'inside' if report.interval_contains_observed else 'outside'} the "
        "partial-ID interval.",
    ]
    if report.fibers:
        top = report.fibers[0]
        if report.fiber_decomposition_available and top.contribution is not None:
            lines.append(
                "- Largest listed public-fiber contributor: "
                f"`{_format_key(report.grouped.public_columns, top.public_value)}` "
                f"with contribution {top.contribution:.4f}."
            )
        else:
            lines.append(
                "- Largest listed public-fiber point range: "
                f"`{_format_key(report.grouped.public_columns, top.public_value)}` "
                f"with range {top.fiber_range:.4f}; contributions are not "
                "additively decomposable for this target."
            )
    return lines


def _suite_sensitivity_review_markdown(
    sensitivity: SensitivityReport | None,
) -> list[str]:
    if sensitivity is None:
        return [
            "- No sensitivity grid was supplied for this suite.",
            "- Reviewers should treat the primary ambiguity as scenario-specific "
            "unless a robustness grid is added.",
        ]

    summary = sensitivity.summary
    lines = [
        f"- Evaluated scenarios: {summary.successful_scenarios}/"
        f"{summary.scenario_count} successful.",
        f"- Public adequacy pattern: {summary.public_adequacy_pattern}.",
    ]
    if summary.min_ambiguity is not None and summary.max_ambiguity is not None:
        lines.append(
            "- Ambiguity range across successful scenarios: "
            f"{summary.min_ambiguity:.4f} to {summary.max_ambiguity:.4f}."
        )
    if summary.highest_ambiguity_scenario is not None:
        lines.append(
            f"- Highest-ambiguity scenario: `{summary.highest_ambiguity_scenario}`."
        )
    if summary.failed_scenarios:
        lines.append(
            f"- Failed scenarios: {summary.failed_scenarios}; inspect the scenario "
            "table before using the grid as a robustness claim."
        )
    return lines


def _suite_refinement_review_markdown(
    primary: PublicDescentReport,
    refinement_sensitivity: RefinementSensitivityReport | None,
) -> list[str]:
    if refinement_sensitivity is not None and refinement_sensitivity.candidates:
        lines = [
            "- Sensitivity-aware refinement ranking is available. Top candidates:",
        ]
        for row in refinement_sensitivity.candidates[:5]:
            lines.append(
                f"- `{row.column}`: mean reduction {row.mean_reduction:.4f}, "
                f"worst reduction {row.min_reduction:.4f}, "
                f"mean reduction pct {row.mean_reduction_percent:.1f}%."
            )
        return lines

    if primary.refinements:
        lines = ["- Primary-scenario refinement ranking is available. Top candidates:"]
        for row in primary.refinements[:5]:
            lines.append(
                f"- `{row.column}`: ambiguity {row.before_ambiguity:.4f} -> "
                f"{row.after_ambiguity:.4f}; reduction {row.reduction:.4f} "
                f"({row.reduction_percent:.1f}%)."
            )
        return lines

    return [
        "- No refinement recommendations were supplied or no candidate columns "
        "reduced hidden-composition ambiguity.",
    ]


def _candidate_refinement_diagnostics(
    candidate_refinements: Sequence[str],
    *,
    public: Sequence[str],
    hidden: Sequence[str],
) -> tuple[DataDiagnostic, ...]:
    diagnostics: list[DataDiagnostic] = []
    public_set = set(public)
    hidden_set = set(hidden)
    already_public = tuple(
        column for column in candidate_refinements if column in public_set
    )
    not_hidden = tuple(
        column
        for column in candidate_refinements
        if column not in hidden_set and column not in public_set
    )
    if already_public:
        diagnostics.append(
            DataDiagnostic(
                code="candidate_refinement_already_public",
                severity="info",
                message=(
                    "Some candidate refinements are already public columns and "
                    "were skipped by refinement ranking."
                ),
                count=len(already_public),
                columns=already_public,
            )
        )
    if not_hidden:
        diagnostics.append(
            DataDiagnostic(
                code="candidate_refinement_not_hidden",
                severity="warning",
                message=(
                    "Some candidate refinements are not present in the hidden "
                    "state space and were skipped by refinement ranking."
                ),
                count=len(not_hidden),
                columns=not_hidden,
            )
        )
    return tuple(diagnostics)


def _data_diagnostics_markdown(
    diagnostics: Sequence[DataDiagnostic],
) -> list[str]:
    lines = ["", "## Data Diagnostics", ""]
    if not diagnostics:
        lines.append("- No pre-solve data diagnostics were raised.")
        return lines

    for row in diagnostics:
        columns = f" columns={', '.join(row.columns)}" if row.columns else ""
        count = "" if row.count is None else f" count={row.count}"
        lines.append(f"- {row.severity}: `{row.code}`{count}{columns} - {row.message}")
    return lines


def _target_contract_markdown(grouped: GroupedProblem) -> list[str]:
    contract = grouped.problem.target_contract
    fixed = "yes" if contract.fixed_after_compilation else "no"
    fiber_decomposition = "yes" if contract.supports_fiber_decomposition else "no"
    if grouped.target_procedure is None:
        interpretation = (
            "- Current interpretation: this report transports the declared fixed "
            "target functional over admissible hidden distributions. Public "
            "refinements reuse the same target contract; they do not refit a "
            "model or recompute a representation-dependent target inside "
            "`updatesupport`."
        )
    else:
        interpretation = (
            "- Current interpretation: this report transports the compiled target "
            "generated by the procedure for the current representation. "
            "Procedure-aware workflows recompile the target for each public "
            "representation or sensitivity scenario, so those comparisons are "
            "procedure-comparison sensitivity analyses rather than transport of "
            "one unchanged target."
        )
    lines = [
        "",
        "## Target Contract",
        "",
        f"- Type: {contract.kind} target.",
        f"- Name: `{contract.name}`.",
        f"- Formula: `{contract.formula}`.",
        f"- Description: {contract.description}.",
        f"- Hidden-cell target values fixed after compilation: {fixed}.",
        f"- Supports public adequacy checks: "
        f"{'yes' if contract.supports_adequacy else 'no'}.",
        f"- Supports interval solving: "
        f"{'yes' if contract.supports_interval else 'no'}.",
        f"- Supports public-fiber decomposition: {fiber_decomposition}.",
        interpretation,
    ]
    if contract.limitations:
        lines.append("- Target limitations:")
        lines.extend(f"  - {limitation}" for limitation in contract.limitations)
    return lines


def _procedure_target_markdown(grouped: GroupedProblem) -> list[str]:
    procedure = grouped.target_procedure
    if procedure is None:
        return []
    context = grouped.target_procedure_context
    lines = [
        "",
        "## Procedure Target",
        "",
        f"- Procedure: `{procedure.name}`.",
        f"- Compiled target: `{_target_label(grouped.target_column)}`.",
        f"- Formula: `{procedure.formula}`.",
    ]
    if procedure.description:
        lines.append(f"- Description: {procedure.description}.")
    if context is not None:
        lines.extend(
            [
                f"- Compiled public columns: {', '.join(context.public)}.",
                f"- Compiled hidden columns: {', '.join(context.hidden)}.",
            ]
        )
    lines.extend(
        [
            "- Interpretation: the target values were recomputed or selected by "
            "the procedure for this representation before solving the finite "
            "transport problem.",
            "- Refinement, sensitivity, and frontier workflows that receive this "
            "ProcedureTarget re-run the compiler for each candidate "
            "representation or scenario.",
        ]
    )
    if procedure.limitations:
        lines.append("- Procedure limitations:")
        lines.extend(f"  - {limitation}" for limitation in procedure.limitations)
    return lines


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return str(getattr(target, "name", type(target).__name__))


def _dual_diagnostics_markdown(
    interval: TransportResult,
    *,
    grouped: GroupedProblem | None = None,
) -> list[str]:
    lines = [
        "",
        "## CVXPY Dual Diagnostics",
        "",
    ]
    if not interval.duals:
        solver = (
            None
            if grouped is None
            else getattr(grouped.problem.environments, "solver", None)
        )
        q_name_value = None if grouped is None else getattr(grouped.q, "name", None)
        lines.extend(
            [
                "- No CVXPY dual diagnostics are available for this interval.",
                "- Duals are produced by CVXPY-backed Q presets and custom CVXPY "
                "constraints. SciPy/simplex-backed intervals still report the "
                "primal ambiguity interval and witnesses, but not solver duals.",
            ]
        )
        if q_name_value == "fiber_support_floor":
            solver_text = "" if solver is None else f" with solver `{solver}`"
            lines.append(
                "- This run used the mixed-integer `fiber_support_floor` preset"
                f"{solver_text}. Mixed-integer solves generally do not expose "
                "KKT-style dual multipliers for the solved integer model; use "
                "the primal interval and witness distributions as the diagnostic "
                "evidence."
            )
        return lines

    lines.append(
        "The largest CVXPY dual multipliers identify constraints that are locally "
        "influential for the solved transport interval. Treat them as solver-scale "
        "sensitivity diagnostics: relaxing a large active constraint is where the "
        "interval is most likely to move."
    )
    for row in interval.dual_summary(top=8, min_magnitude=1e-8):
        lines.append(f"- {_format_dual(row)}")
    return lines


def _model_review_limitations() -> list[str]:
    return [
        "- `updatesupport` audits representation stability for a supplied target; "
        "it does not identify causal effects, fit the causal model, or validate "
        "unconfoundedness, exclusion restrictions, overlap, or graph assumptions.",
        "- The partial-ID interval is hidden-composition ambiguity under the stated "
        "Q stress test. It is not a confidence interval and should be reported "
        "separately from statistical uncertainty.",
        "- Statistical uncertainty is included only when supplied by the upstream "
        "statistical or causal workflow.",
        "- Refinement recommendations are measurement/reporting recommendations. "
        "They are not automatically valid causal adjustment variables.",
        "- Conclusions depend on the retained hidden state space, sparse-cell "
        "filtering, weights, and the chosen admissible-shift preset.",
        "- CVXPY dual diagnostics, when present, are local solver diagnostics, not "
        "global guarantees about every possible relaxation of the problem.",
    ]


def _public_descent_summary_dict(report: PublicDescentReport) -> dict[str, Any]:
    return {
        "title": report.title,
        "observed_value": report.observed_value,
        "lower": report.interval.lower,
        "upper": report.interval.upper,
        "ambiguity": report.interval.diameter,
        "public_adequate": report.public_adequate,
        "q_name": report.grouped.q_name,
        "q_description": report.grouped.q_description,
        "target_contract": report.grouped.problem.target_contract.as_dict(),
        "target": _target_label(report.grouped.target_column),
        "target_procedure": None
        if report.grouped.target_procedure is None
        else report.grouped.target_procedure.as_dict(),
        "target_procedure_context": None
        if report.grouped.target_procedure_context is None
        else report.grouped.target_procedure_context.as_dict(),
        "fiber_decomposition_available": report.fiber_decomposition_available,
        "fiber_diagnostic_kind": report.fiber_diagnostic_kind,
        "top_fiber_contribution": report.top_fiber_contribution,
        "top_fiber_contribution_share": report.top_fiber_contribution_share,
        "public_columns": report.grouped.public_columns,
        "hidden_columns": report.grouped.hidden_columns,
        "hidden_cells": len(report.grouped.problem.states),
        "public_cells": len(report.grouped.problem.public_values),
        "top_fibers": [row.as_dict() for row in report.fibers],
        "refinements": [row.as_dict() for row in report.refinements],
        "diagnostics": [row.as_dict() for row in report.diagnostics],
        "duals": [row.as_dict() for row in report.interval.dual_summary(top=20)],
    }


def _repeatable_data(data: Any) -> tuple[Any, int | None]:
    if hasattr(data, "to_dict"):
        try:
            return data, len(data)
        except TypeError:
            return data, None
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return data, len(data)
    records = tuple(data)
    return records, len(records)


def _resolve_sequence_arg(
    primary: Sequence[str] | None,
    alias: Sequence[str] | None,
    *,
    primary_name: str,
    alias_name: str,
) -> Sequence[str] | None:
    if primary is not None and alias is not None and tuple(primary) != tuple(alias):
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _resolve_scalar_arg(
    primary: str | None,
    alias: str | None,
    *,
    primary_name: str,
    alias_name: str,
) -> str | None:
    if primary is not None and alias is not None and primary != alias:
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _format_key(columns: Sequence[str], key: tuple[Hashable, ...]) -> str:
    return ", ".join(
        f"{column}={value}" for column, value in zip(columns, key, strict=True)
    )


def _format_dual(row: ConstraintDual) -> str:
    parts = [f"{row.solve}: {row.name}"]
    if row.variable is not None:
        parts.append(f"variable={row.variable}")
    if row.public_value is not None:
        parts.append(f"public={row.public_value!r}")
    if row.state is not None:
        parts.append(f"state={row.state!r}")
    if row.index is not None and row.state is None and row.public_value is None:
        parts.append(f"index={row.index}")
    parts.append(f"dual={row.magnitude:.4g}")
    if row.residual is not None:
        parts.append(f"residual={row.residual:.2g}")
    return ", ".join(parts)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
