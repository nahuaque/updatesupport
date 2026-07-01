"""Analyst-facing public-descent reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

from .data import GroupedProblem, from_dataframe
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
    diameter: float
    reduction: float
    public_cells: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "diameter": self.diameter,
            "reduction": self.reduction,
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
                "target value inside that public cell. `contribution = mass * range`, "
                "so it is the amount that fiber contributes to the overall interval "
                "width. The listed top fibers account for "
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
                    "ambiguity, and `public_cells` is the resulting number of public "
                    "strata. This is a measurement-value table: large reductions "
                    "identify variables that make the coarse public representation "
                    "more stable, with the usual tradeoff that more strata may "
                    "increase sparsity.",
                ]
            )
            for row in self.refinements:
                lines.append(
                    f"- add {row.column}: ambiguity={row.diameter:.4f}, "
                    f"reduction={row.reduction:.4f}, public_cells={row.public_cells}"
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
    compile_data = data
    if isinstance(data, GroupedProblem):
        grouped = data
        refinement_data = source_data
    else:
        compile_data, inferred_row_count = _repeatable_data(data)
        if row_count is None:
            row_count = inferred_row_count
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
    rows = []
    for public_value in problem.public_values:
        states = problem.public_fibers[public_value]
        ordered_states = sorted(states, key=lambda state: problem.estimand_map[state])
        min_state = ordered_states[0]
        max_state = ordered_states[-1]
        fiber_range = problem.estimand_map[max_state] - problem.estimand_map[min_state]
        public_mass = grouped.public_law[public_value]
        rows.append(
            PublicFiberDiagnostic(
                public_value=public_value,
                public_mass=public_mass,
                hidden_cells=len(states),
                fiber_range=fiber_range,
                contribution=public_mass * fiber_range,
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
        )
        diameter = refined.problem.global_transport_modulus().diameter
        scores.append(
            RefinementCandidate(
                column=column,
                diameter=diameter,
                reduction=baseline_diameter - diameter,
                public_cells=len(refined.problem.public_values),
            )
        )

    scores.sort(key=lambda row: row.reduction, reverse=True)
    return tuple(scores if top is None else scores[:top])


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
