"""Analyst-facing public-descent reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

from .data import GroupedProblem, from_dataframe
from .presets import q_bounded_shift, q_description, q_name
from .results import TransportResult


@dataclass(frozen=True)
class PublicFiberDiagnostic:
    """Contribution of one public fiber to transport ambiguity."""

    public_value: tuple[Hashable, ...]
    public_mass: float
    hidden_cells: int
    fiber_range: float
    contribution: float
    min_state: tuple[Hashable, ...]
    min_value: float
    max_state: tuple[Hashable, ...]
    max_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_value": self.public_value,
            "public_mass": self.public_mass,
            "hidden_cells": self.hidden_cells,
            "range": self.fiber_range,
            "contribution": self.contribution,
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

    @property
    def top_fiber_contribution(self) -> float:
        return sum(row.contribution for row in self.fibers)

    @property
    def top_fiber_contribution_share(self) -> float:
        if self.interval.diameter <= self.grouped.problem.tol:
            return 0.0
        return self.top_fiber_contribution / self.interval.diameter

    @property
    def interval_contains_observed(self) -> bool:
        tol = self.grouped.problem.tol
        return (
            self.interval.lower - tol
            <= self.observed_value
            <= self.interval.upper + tol
        )

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
                f"- Top {len(self.fibers)} fiber contribution share: "
                f"{_percent(self.top_fiber_contribution_share)}",
                "",
                "## Statistical Interpretation",
                "",
                f"The estimand is the aggregate {self.target_description}. Each hidden cell "
                "gets its own empirical target value, and the current observed value "
                "is the sample-weighted average over the observed hidden-cell mix.",
                "",
                "The partial-ID interval fixes the observed public distribution and "
                "then allows arbitrary reweighting among retained hidden cells inside "
                "each public cell. Under that stress test, the aggregate value can "
                f"range from {self.interval.lower:.4f} to {self.interval.upper:.4f}. "
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
                "",
                "For each public fiber below, `range` is the max-minus-min hidden-cell "
                "target value inside that public cell. `contribution` is the fiber's "
                "difference between the upper and lower transport witnesses; under "
                "the saturated preset this equals `mass * range`. The listed top "
                "fibers account for "
                f"{_percent(self.top_fiber_contribution_share)} of total transport "
                "ambiguity.",
                "",
                "## Worst Public Fibers",
            ]
        )

        for row in self.fibers:
            lines.extend(
                [
                    f"- {_format_key(grouped.public_columns, row.public_value)}",
                    f"  mass={row.public_mass:.4f}, hidden_cells={row.hidden_cells}, "
                    f"range={row.fiber_range:.4f}, contribution={row.contribution:.4f}",
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
                    "the usual tradeoff that more strata may increase sparsity.",
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
    target: str | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    target_column: str | None = None,
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
        grouped = from_dataframe(
            compile_data,
            public=public,
            hidden=hidden,
            target=target,
            weight=weight,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            target_column=target_column,
            weight_column=weight_column,
            min_cell_weight=min_cell_weight,
            q=effective_q,
            q_radius=q_radius,
        )
        refinement_data = source_data if source_data is not None else compile_data

    interval = grouped.problem.global_transport_modulus()
    observed_value = _observed_value(grouped)
    fibers = public_fiber_diagnostics(grouped, top=top)
    refinements: tuple[RefinementCandidate, ...] = ()
    if candidate_refinements:
        if refinement_data is None:
            raise ValueError(
                "source_data is required to compute refinements for a GroupedProblem"
            )
        refinements = recommend_refinements(
            refinement_data,
            public=grouped.public_columns,
            hidden=grouped.hidden_columns,
            target=grouped.target_column,
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
    )


def public_fiber_diagnostics(
    grouped: GroupedProblem, *, top: int | None = 10
) -> tuple[PublicFiberDiagnostic, ...]:
    """Return public fibers ranked by ambiguity contribution."""

    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    problem = grouped.problem
    interval = problem.global_transport_modulus()
    rows = []
    for public_value in problem.public_values:
        states = problem.public_fibers[public_value]
        ordered_states = sorted(states, key=lambda state: problem.estimand_map[state])
        min_state = ordered_states[0]
        max_state = ordered_states[-1]
        fiber_range = problem.estimand_map[max_state] - problem.estimand_map[min_state]
        public_mass = grouped.public_law[public_value]
        contribution = public_mass * fiber_range
        if interval.q_lower is not None and interval.q_upper is not None:
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
            )
        )
    rows.sort(key=lambda row: row.contribution, reverse=True)
    return tuple(rows if top is None else rows[:top])


def recommend_refinements(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: str,
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
class SensitivityReport:
    """Robustness grid over Q presets, hidden sets, and min-cell thresholds."""

    rows: tuple[SensitivityRow, ...]
    title: str = "Public Descent Sensitivity Report"
    row_count: int | None = None

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.row_count is not None:
            lines.extend([f"- Rows: {self.row_count}", ""])
        lines.extend(
            [
                "| scenario | Q | min_cell_weight | hidden columns | hidden cells | public cells | observed | lower | upper | ambiguity | public adequate | status |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in self.rows:
            hidden_columns = ", ".join(row.hidden_columns)
            adequate = "" if row.public_adequate is None else (
                "yes" if row.public_adequate else "no"
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
    target: str,
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
    q_grid = tuple(q_presets) if q_presets is not None else (
        "saturated",
        q_bounded_shift(0.5),
        "observed",
    )

    rows: list[SensitivityRow] = []
    scenario_index = 0
    for hidden_columns in hidden_grid:
        hidden_columns_tuple = tuple(hidden_columns)
        for min_cell_weight in min_cell_weights:
            for q_preset in q_grid:
                scenario_index += 1
                scenario = f"S{scenario_index}"
                try:
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

    return SensitivityReport(rows=tuple(rows), title=title, row_count=row_count)


def _observed_value(grouped: GroupedProblem) -> float:
    problem = grouped.problem
    return sum(
        grouped.cell_weights[state] * problem.estimand_map[state]
        for state in problem.states
    )


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


def _format_key(columns: Sequence[str], key: tuple[Hashable, ...]) -> str:
    return ", ".join(
        f"{column}={value}" for column, value in zip(columns, key, strict=True)
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
