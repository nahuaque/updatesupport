"""Pareto frontier search for public reporting representations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from .data import from_dataframe
from .metrics import RowMetric
from .presets import q_description, q_name


@dataclass(frozen=True)
class FrontierScenarioResult:
    """One stress-test result for one public-representation candidate."""

    scenario: str
    q_name: str
    q_description: str
    lower: float
    upper: float
    ambiguity: float
    observed_value: float
    public_adequate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "observed_value": self.observed_value,
            "public_adequate": self.public_adequate,
        }


@dataclass(frozen=True)
class PublicRepresentationCandidate:
    """A public column set evaluated over a stress-test grid."""

    added_columns: tuple[str, ...]
    public_columns: tuple[str, ...]
    public_cells: int
    hidden_cells: int
    scenarios: tuple[FrontierScenarioResult, ...]
    max_ambiguity: float
    mean_ambiguity: float
    passes_ambiguity_limit: bool | None = None

    @property
    def added_column_count(self) -> int:
        return len(self.added_columns)

    @property
    def observed_value(self) -> float:
        if not self.scenarios:
            return 0.0
        return self.scenarios[0].observed_value

    @property
    def public_adequate(self) -> bool:
        return all(row.public_adequate for row in self.scenarios)

    @property
    def label(self) -> str:
        if not self.added_columns:
            return "base public representation"
        return "base + " + ", ".join(self.added_columns)

    def ambiguity_by_scenario(self) -> dict[str, float]:
        return {row.scenario: row.ambiguity for row in self.scenarios}

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_columns": self.added_columns,
            "added_column_count": self.added_column_count,
            "public_columns": self.public_columns,
            "public_cells": self.public_cells,
            "hidden_cells": self.hidden_cells,
            "observed_value": self.observed_value,
            "max_ambiguity": self.max_ambiguity,
            "mean_ambiguity": self.mean_ambiguity,
            "public_adequate": self.public_adequate,
            "passes_ambiguity_limit": self.passes_ambiguity_limit,
            "scenarios": [row.as_dict() for row in self.scenarios],
        }


@dataclass(frozen=True)
class PublicRepresentationFrontier:
    """Pareto frontier over public-cell complexity and transport stability."""

    candidates: tuple[PublicRepresentationCandidate, ...]
    frontier: tuple[PublicRepresentationCandidate, ...]
    dominated: tuple[PublicRepresentationCandidate, ...]
    base_public: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    candidate_refinements: tuple[str, ...]
    ambiguity_limit: float | None = None
    bucket_budget: int | None = None
    title: str = "Public Representation Frontier"
    row_count: int | None = None

    @property
    def minimal_stable(self) -> PublicRepresentationCandidate | None:
        """Smallest public representation satisfying ``ambiguity_limit``."""

        if self.ambiguity_limit is None:
            return None
        stable = [
            row for row in self.candidates if row.max_ambiguity <= self.ambiguity_limit
        ]
        if not stable:
            return None
        return min(
            stable,
            key=lambda row: (
                row.public_cells,
                row.added_column_count,
                row.max_ambiguity,
                row.added_columns,
            ),
        )

    def best_under_bucket_budget(
        self, budget: int | None = None
    ) -> PublicRepresentationCandidate | None:
        """Most stable candidate with no more than ``budget`` public cells."""

        effective_budget = self.bucket_budget if budget is None else budget
        if effective_budget is None:
            return None
        if effective_budget < 0:
            raise ValueError("budget must be non-negative")
        feasible = [row for row in self.candidates if row.public_cells <= effective_budget]
        if not feasible:
            return None
        return min(
            feasible,
            key=lambda row: (
                row.max_ambiguity,
                row.mean_ambiguity,
                row.public_cells,
                row.added_column_count,
                row.added_columns,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "row_count": self.row_count,
            "base_public": self.base_public,
            "hidden_columns": self.hidden_columns,
            "candidate_refinements": self.candidate_refinements,
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
            "minimal_stable": None
            if self.minimal_stable is None
            else self.minimal_stable.as_dict(),
            "best_under_bucket_budget": None
            if self.best_under_bucket_budget() is None
            else self.best_under_bucket_budget().as_dict(),
            "frontier": [row.as_dict() for row in self.frontier],
            "dominated": [row.as_dict() for row in self.dominated],
            "candidates": [row.as_dict() for row in self.candidates],
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.row_count is not None:
            lines.append(f"- Rows: {self.row_count}")
        lines.extend(
            [
                f"- Base public columns: {', '.join(self.base_public)}",
                f"- Hidden columns: {', '.join(self.hidden_columns)}",
                f"- Candidate refinements: {', '.join(self.candidate_refinements)}",
                f"- Evaluated representations: {len(self.candidates)}",
                f"- Pareto frontier representations: {len(self.frontier)}",
            ]
        )
        if self.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.ambiguity_limit:.4f}")
        if self.bucket_budget is not None:
            lines.append(f"- Public-cell budget: {self.bucket_budget}")

        minimal = self.minimal_stable
        if minimal is not None:
            lines.append(
                "- Minimal stable representation: "
                f"`{minimal.label}` with {minimal.public_cells} public cells "
                f"and max ambiguity {minimal.max_ambiguity:.4f}."
            )
        elif self.ambiguity_limit is not None:
            lines.append(
                "- Minimal stable representation: none of the evaluated candidates "
                "met the ambiguity limit."
            )

        best_budget = self.best_under_bucket_budget()
        if best_budget is not None:
            lines.append(
                "- Best representation within bucket budget: "
                f"`{best_budget.label}` with max ambiguity "
                f"{best_budget.max_ambiguity:.4f}."
            )

        lines.extend(["", "## Interpretation", ""])
        lines.extend(_frontier_interpretation(self))
        lines.extend(["", "## Pareto Frontier", ""])
        lines.extend(_candidate_table(self.frontier))
        if self.dominated:
            lines.extend(["", "## Dominated Candidates", ""])
            lines.extend(_candidate_table(self.dominated))
        lines.extend(["", "## Scenario Details", ""])
        lines.extend(_scenario_table(self.frontier))
        return "\n".join(lines)


def public_representation_frontier(
    data: Any,
    *,
    base_public: Sequence[str] | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str],
    target: str | RowMetric,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    weight: str | None = None,
    min_cell_weight: float = 1.0,
    q_presets: Sequence[Any] = ("saturated",),
    ambiguity_limit: float | None = None,
    bucket_budget: int | None = None,
    max_refinements: int | None = None,
    include_base: bool = True,
    title: str = "Public Representation Frontier",
) -> PublicRepresentationFrontier:
    """Search public-column refinements and return the Pareto frontier.

    The first implementation searches over subsets of ``candidate_refinements``.
    It does not learn arbitrary partitions: every candidate is a concrete public
    representation formed by adding zero or more hidden columns to
    ``base_public``. Pareto dominance compares public-cell count, added-column
    count, and ambiguity under every supplied Q stress test.
    """

    base_public = _resolve_sequence_arg(
        base_public,
        public,
        primary_name="base_public",
        alias_name="public",
    )
    if base_public is None:
        raise TypeError(
            "public_representation_frontier() missing required keyword argument: "
            "'base_public'"
        )
    candidate_refinements = _resolve_sequence_arg(
        candidate_refinements,
        candidate_columns,
        primary_name="candidate_refinements",
        alias_name="candidate_columns",
    )
    if candidate_refinements is None:
        candidate_refinements = ()
    if min_cell_weight < 0:
        raise ValueError("min_cell_weight must be non-negative")
    if ambiguity_limit is not None and ambiguity_limit < 0:
        raise ValueError("ambiguity_limit must be non-negative")
    if bucket_budget is not None and bucket_budget < 0:
        raise ValueError("bucket_budget must be non-negative")
    if max_refinements is not None and max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")
    if not q_presets:
        raise ValueError("q_presets must contain at least one preset")

    base_public_tuple = tuple(base_public)
    hidden_tuple = tuple(hidden)
    candidate_tuple = _valid_candidate_refinements(
        base_public_tuple,
        hidden_tuple,
        candidate_refinements,
    )
    max_refinements = (
        len(candidate_tuple)
        if max_refinements is None
        else min(max_refinements, len(candidate_tuple))
    )

    repeatable_data, row_count = _repeatable_data(data)
    q_grid = tuple(q_presets)
    candidates = []
    for added_columns in _candidate_subsets(
        candidate_tuple,
        max_refinements=max_refinements,
        include_base=include_base,
    ):
        candidates.append(
            _evaluate_candidate(
                repeatable_data,
                base_public=base_public_tuple,
                hidden=hidden_tuple,
                target=target,
                added_columns=added_columns,
                q_presets=q_grid,
                weight=weight,
                min_cell_weight=min_cell_weight,
                ambiguity_limit=ambiguity_limit,
            )
        )

    candidates_tuple = tuple(sorted(candidates, key=_candidate_sort_key))
    frontier = tuple(
        row
        for row in candidates_tuple
        if not any(
            _dominates(other, row)
            for other in candidates_tuple
            if other is not row
        )
    )
    dominated = tuple(row for row in candidates_tuple if row not in frontier)
    return PublicRepresentationFrontier(
        candidates=candidates_tuple,
        frontier=frontier,
        dominated=dominated,
        base_public=base_public_tuple,
        hidden_columns=hidden_tuple,
        candidate_refinements=candidate_tuple,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        title=title,
        row_count=row_count,
    )


def _evaluate_candidate(
    data: Any,
    *,
    base_public: tuple[str, ...],
    hidden: tuple[str, ...],
    target: str | RowMetric,
    added_columns: tuple[str, ...],
    q_presets: tuple[Any, ...],
    weight: str | None,
    min_cell_weight: float,
    ambiguity_limit: float | None,
) -> PublicRepresentationCandidate:
    public_columns = base_public + added_columns
    scenarios = []
    public_cells = 0
    hidden_cells = 0
    for index, q_preset in enumerate(q_presets, start=1):
        grouped = from_dataframe(
            data,
            public=public_columns,
            hidden=hidden,
            target=target,
            weight=weight,
            min_cell_weight=min_cell_weight,
            q=q_preset,
        )
        interval = grouped.problem.global_transport_modulus()
        public_cells = len(grouped.problem.public_values)
        hidden_cells = len(grouped.problem.states)
        scenarios.append(
            FrontierScenarioResult(
                scenario=f"S{index}",
                q_name=q_name(q_preset),
                q_description=q_description(q_preset),
                lower=interval.lower,
                upper=interval.upper,
                ambiguity=interval.diameter,
                observed_value=_observed_value(grouped),
                public_adequate=grouped.problem.is_public_adequate(),
            )
        )

    scenario_tuple = tuple(scenarios)
    ambiguities = [row.ambiguity for row in scenario_tuple]
    max_ambiguity = max(ambiguities)
    mean_ambiguity = sum(ambiguities) / len(ambiguities)
    return PublicRepresentationCandidate(
        added_columns=added_columns,
        public_columns=public_columns,
        public_cells=public_cells,
        hidden_cells=hidden_cells,
        scenarios=scenario_tuple,
        max_ambiguity=max_ambiguity,
        mean_ambiguity=mean_ambiguity,
        passes_ambiguity_limit=None
        if ambiguity_limit is None
        else max_ambiguity <= ambiguity_limit,
    )


def _dominates(
    left: PublicRepresentationCandidate,
    right: PublicRepresentationCandidate,
    *,
    tol: float = 1e-12,
) -> bool:
    left_ambiguities = [row.ambiguity for row in left.scenarios]
    right_ambiguities = [row.ambiguity for row in right.scenarios]
    no_worse = (
        left.public_cells <= right.public_cells
        and left.added_column_count <= right.added_column_count
        and all(
            a <= b + tol
            for a, b in zip(left_ambiguities, right_ambiguities, strict=True)
        )
    )
    strictly_better = (
        left.public_cells < right.public_cells
        or left.added_column_count < right.added_column_count
        or any(
            a < b - tol
            for a, b in zip(left_ambiguities, right_ambiguities, strict=True)
        )
    )
    return no_worse and strictly_better


def _candidate_subsets(
    columns: tuple[str, ...],
    *,
    max_refinements: int,
    include_base: bool,
) -> tuple[tuple[str, ...], ...]:
    start = 0 if include_base else 1
    subsets = []
    for size in range(start, max_refinements + 1):
        subsets.extend(combinations(columns, size))
    return tuple(tuple(subset) for subset in subsets)


def _valid_candidate_refinements(
    base_public: tuple[str, ...],
    hidden: tuple[str, ...],
    candidate_refinements: Sequence[str],
) -> tuple[str, ...]:
    hidden_set = set(hidden)
    base_set = set(base_public)
    candidates = []
    seen = set()
    for column in candidate_refinements:
        if column in seen:
            continue
        seen.add(column)
        if column in base_set:
            continue
        if column not in hidden_set:
            continue
        candidates.append(column)
    return tuple(candidates)


def _candidate_sort_key(
    row: PublicRepresentationCandidate,
) -> tuple[float, float, int, int, tuple[str, ...]]:
    return (
        row.public_cells,
        row.max_ambiguity,
        row.mean_ambiguity,
        row.added_column_count,
        row.added_columns,
    )


def _observed_value(grouped: Any) -> float:
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


def _frontier_interpretation(report: PublicRepresentationFrontier) -> list[str]:
    lines = [
        "- Each candidate is a concrete public representation obtained by adding "
        "zero or more candidate hidden columns to the base public columns.",
        "- A candidate is Pareto-frontier if no other evaluated representation has "
        "no more public cells, no more added columns, and no larger ambiguity "
        "under every listed stress test, with at least one strict improvement.",
        "- `max ambiguity` is the conservative scalar summary across stress tests. "
        "`mean ambiguity` is useful for ranking, but the frontier itself checks "
        "every stress-test scenario.",
    ]
    if report.ambiguity_limit is not None:
        if report.minimal_stable is None:
            lines.append(
                "- No evaluated representation met the ambiguity limit. Consider "
                "raising the limit, adding candidate refinements, or changing the "
                "stress-test grid."
            )
        else:
            lines.append(
                "- The minimal stable representation is the smallest evaluated "
                "public representation whose worst-case ambiguity stays within "
                "the supplied limit."
            )
    if report.bucket_budget is not None:
        lines.append(
            "- The bucket-budget recommendation chooses the lowest worst-case "
            "ambiguity among representations with no more than the supplied "
            "number of public cells."
        )
    return lines


def _candidate_table(candidates: Sequence[PublicRepresentationCandidate]) -> list[str]:
    lines = [
        "| representation | public cells | added columns | max ambiguity | mean ambiguity | public adequate | stable |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in candidates:
        stable = (
            ""
            if row.passes_ambiguity_limit is None
            else ("yes" if row.passes_ambiguity_limit else "no")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.label),
                    str(row.public_cells),
                    str(row.added_column_count),
                    f"{row.max_ambiguity:.4f}",
                    f"{row.mean_ambiguity:.4f}",
                    "yes" if row.public_adequate else "no",
                    stable,
                ]
            )
            + " |"
        )
    return lines


def _scenario_table(candidates: Sequence[PublicRepresentationCandidate]) -> list[str]:
    lines = [
        "| representation | scenario | Q | observed | lower | upper | ambiguity | public adequate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in candidates:
        for row in candidate.scenarios:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(candidate.label),
                        row.scenario,
                        _escape_table(row.q_name),
                        f"{row.observed_value:.4f}",
                        f"{row.lower:.4f}",
                        f"{row.upper:.4f}",
                        f"{row.ambiguity:.4f}",
                        "yes" if row.public_adequate else "no",
                    ]
                )
                + " |"
            )
    return lines


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
