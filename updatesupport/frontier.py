"""Pareto frontier search for public reporting representations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
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
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    public_cells: int
    hidden_cells: int
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
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "public_cells": self.public_cells,
            "hidden_cells": self.hidden_cells,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "observed_value": self.observed_value,
            "public_adequate": self.public_adequate,
        }


@dataclass(frozen=True)
class FrontierSearchTrace:
    """Metadata describing how the frontier search was run."""

    search: str
    exact: bool
    evaluated_candidates: int
    candidate_space_size: int
    scenario_count: int
    max_added_columns: int
    max_evaluations: int | None = None
    beam_width: int | None = None
    enforce_bucket_budget: bool = False
    skipped_by_budget: int = 0
    pruned_by_dominance: int = 0
    pruned_by_beam: int = 0
    stopping_reason: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "search": self.search,
            "exact": self.exact,
            "evaluated_candidates": self.evaluated_candidates,
            "candidate_space_size": self.candidate_space_size,
            "scenario_count": self.scenario_count,
            "max_added_columns": self.max_added_columns,
            "max_evaluations": self.max_evaluations,
            "beam_width": self.beam_width,
            "enforce_bucket_budget": self.enforce_bucket_budget,
            "skipped_by_budget": self.skipped_by_budget,
            "pruned_by_dominance": self.pruned_by_dominance,
            "pruned_by_beam": self.pruned_by_beam,
            "stopping_reason": self.stopping_reason,
        }


@dataclass(frozen=True)
class _ScenarioSpec:
    scenario: str
    hidden_columns: tuple[str, ...]
    min_cell_weight: float
    q: Any


@dataclass(frozen=True)
class PublicRepresentationCandidate:
    """A public column set evaluated over a stress-test grid."""

    added_columns: tuple[str, ...]
    public_columns: tuple[str, ...]
    public_cells: int
    hidden_cells: int
    min_public_cells: int
    max_public_cells: int
    min_hidden_cells: int
    max_hidden_cells: int
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
            "min_public_cells": self.min_public_cells,
            "max_public_cells": self.max_public_cells,
            "min_hidden_cells": self.min_hidden_cells,
            "max_hidden_cells": self.max_hidden_cells,
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
    hidden_sets: tuple[tuple[str, ...], ...]
    min_cell_weights: tuple[float, ...]
    candidate_refinements: tuple[str, ...]
    ambiguity_limit: float | None = None
    bucket_budget: int | None = None
    title: str = "Public Representation Frontier"
    row_count: int | None = None
    search_trace: FrontierSearchTrace | None = None

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
            "hidden_sets": self.hidden_sets,
            "min_cell_weights": self.min_cell_weights,
            "candidate_refinements": self.candidate_refinements,
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
            "search_trace": None
            if self.search_trace is None
            else self.search_trace.as_dict(),
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
                f"- Hidden-set scenarios: {len(self.hidden_sets)}",
                "- Minimum hidden-cell weights: "
                f"{', '.join(f'{value:g}' for value in self.min_cell_weights)}",
                f"- Candidate refinements: {', '.join(self.candidate_refinements)}",
                f"- Evaluated representations: {len(self.candidates)}",
                f"- Pareto frontier representations: {len(self.frontier)}",
            ]
        )
        if self.search_trace is not None:
            exact = "exact" if self.search_trace.exact else "heuristic"
            lines.extend(
                [
                    f"- Search mode: {self.search_trace.search} ({exact})",
                    "- Search evaluations: "
                    f"{self.search_trace.evaluated_candidates}/"
                    f"{self.search_trace.candidate_space_size}",
                    f"- Stress-test scenarios per representation: "
                    f"{self.search_trace.scenario_count}",
                    f"- Search stopping reason: {self.search_trace.stopping_reason}",
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
    min_cell_weights: Sequence[float] | None = None,
    hidden_sets: Sequence[Sequence[str]] | None = None,
    q_presets: Sequence[Any] = ("saturated",),
    ambiguity_limit: float | None = None,
    bucket_budget: int | None = None,
    max_refinements: int | None = None,
    max_added_columns: int | None = None,
    search: str = "exhaustive",
    beam_width: int = 12,
    max_evaluations: int | None = None,
    must_include: Sequence[str] | None = None,
    must_exclude: Sequence[str] | None = None,
    enforce_bucket_budget: bool = False,
    include_base: bool = True,
    title: str = "Public Representation Frontier",
) -> PublicRepresentationFrontier:
    """Search public-column refinements and return the Pareto frontier.

    The search runs over subsets of ``candidate_refinements``. It does not learn
    arbitrary partitions: every candidate is a concrete public representation
    formed by adding zero or more hidden columns to ``base_public``. Pareto
    dominance compares public-cell count, added-column count, and ambiguity
    under every supplied Q stress test.
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
    min_cell_weight_grid = _resolve_min_cell_weights(
        min_cell_weight,
        min_cell_weights,
    )
    if ambiguity_limit is not None and ambiguity_limit < 0:
        raise ValueError("ambiguity_limit must be non-negative")
    if bucket_budget is not None and bucket_budget < 0:
        raise ValueError("bucket_budget must be non-negative")
    max_added_columns = _resolve_optional_int_arg(
        max_added_columns,
        max_refinements,
        primary_name="max_added_columns",
        alias_name="max_refinements",
    )
    if max_added_columns is not None and max_added_columns < 0:
        raise ValueError("max_added_columns must be non-negative")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_evaluations is not None and max_evaluations < 0:
        raise ValueError("max_evaluations must be non-negative")
    if not q_presets:
        raise ValueError("q_presets must contain at least one preset")

    base_public_tuple = tuple(base_public)
    hidden_tuple = tuple(hidden)
    hidden_grid = _resolve_hidden_sets(hidden_tuple, hidden_sets)
    _validate_hidden_grid(base_public_tuple, hidden_grid)
    common_hidden = _common_hidden_columns(hidden_grid)
    must_include_tuple = _unique_tuple(must_include or ())
    must_exclude_tuple = _unique_tuple(must_exclude or ())
    candidate_tuple, required_columns = _valid_candidate_refinements(
        base_public_tuple,
        common_hidden,
        candidate_refinements,
        must_include=must_include_tuple,
        must_exclude=must_exclude_tuple,
    )
    max_added_columns = (
        len(candidate_tuple)
        if max_added_columns is None
        else min(max_added_columns, len(candidate_tuple))
    )
    if len(required_columns) > max_added_columns:
        raise ValueError("must_include contains more columns than max_added_columns")

    repeatable_data, row_count = _repeatable_data(data)
    q_grid = tuple(q_presets)
    scenario_specs = _frontier_scenarios(
        hidden_sets=hidden_grid,
        min_cell_weights=min_cell_weight_grid,
        q_presets=q_grid,
    )
    search_mode = _normalize_search(search)
    candidate_space_size = _candidate_space_size(
        candidate_tuple,
        max_added_columns=max_added_columns,
        include_base=include_base,
        required_columns=required_columns,
    )
    search_result = _search_candidates(
        repeatable_data,
        base_public=base_public_tuple,
        target=target,
        candidate_refinements=candidate_tuple,
        required_columns=required_columns,
        scenario_specs=scenario_specs,
        weight=weight,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        max_added_columns=max_added_columns,
        include_base=include_base,
        search=search_mode,
        beam_width=beam_width,
        max_evaluations=max_evaluations,
        enforce_bucket_budget=enforce_bucket_budget,
        candidate_space_size=candidate_space_size,
    )

    candidates_tuple = tuple(sorted(search_result.candidates, key=_candidate_sort_key))
    frontier, dominated = _split_frontier(candidates_tuple)
    return PublicRepresentationFrontier(
        candidates=candidates_tuple,
        frontier=frontier,
        dominated=dominated,
        base_public=base_public_tuple,
        hidden_columns=hidden_tuple,
        hidden_sets=hidden_grid,
        min_cell_weights=min_cell_weight_grid,
        candidate_refinements=candidate_tuple,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        title=title,
        row_count=row_count,
        search_trace=search_result.trace,
    )


@dataclass
class _SearchResult:
    candidates: tuple[PublicRepresentationCandidate, ...]
    trace: FrontierSearchTrace


@dataclass
class _EvaluationState:
    data: Any
    base_public: tuple[str, ...]
    target: str | RowMetric
    candidate_refinements: tuple[str, ...]
    scenario_specs: tuple[_ScenarioSpec, ...]
    weight: str | None
    ambiguity_limit: float | None
    bucket_budget: int | None
    max_evaluations: int | None
    enforce_bucket_budget: bool
    candidates_by_subset: dict[
        tuple[str, ...],
        PublicRepresentationCandidate,
    ]
    result_subsets: list[tuple[str, ...]]
    result_subset_set: set[tuple[str, ...]]
    skipped_by_budget: int = 0
    skipped_budget_subsets: set[tuple[str, ...]] | None = None
    evaluation_limit_hit: bool = False

    def __post_init__(self) -> None:
        if self.skipped_budget_subsets is None:
            self.skipped_budget_subsets = set()

    @property
    def evaluated_count(self) -> int:
        return len(self.candidates_by_subset)

    def evaluate(
        self,
        added_columns: Sequence[str],
        *,
        include_result: bool = True,
    ) -> PublicRepresentationCandidate | None:
        subset = _ordered_subset(self.candidate_refinements, added_columns)
        candidate = self.candidates_by_subset.get(subset)
        if candidate is None:
            if (
                self.max_evaluations is not None
                and self.evaluated_count >= self.max_evaluations
            ):
                self.evaluation_limit_hit = True
                return None
            candidate = _evaluate_candidate(
                self.data,
                base_public=self.base_public,
                target=self.target,
                added_columns=subset,
                scenario_specs=self.scenario_specs,
                weight=self.weight,
                ambiguity_limit=self.ambiguity_limit,
            )
            self.candidates_by_subset[subset] = candidate

        if include_result:
            if self.is_allowed(candidate):
                if subset not in self.result_subset_set:
                    self.result_subsets.append(subset)
                    self.result_subset_set.add(subset)
            elif subset not in self.skipped_budget_subsets:
                self.skipped_by_budget += 1
                self.skipped_budget_subsets.add(subset)
        return candidate

    def is_allowed(self, candidate: PublicRepresentationCandidate) -> bool:
        return (
            not self.enforce_bucket_budget
            or self.bucket_budget is None
            or candidate.public_cells <= self.bucket_budget
        )

    def result_candidates(self) -> tuple[PublicRepresentationCandidate, ...]:
        return tuple(self.candidates_by_subset[subset] for subset in self.result_subsets)


def _search_candidates(
    data: Any,
    *,
    base_public: tuple[str, ...],
    target: str | RowMetric,
    candidate_refinements: tuple[str, ...],
    required_columns: tuple[str, ...],
    scenario_specs: tuple[_ScenarioSpec, ...],
    weight: str | None,
    ambiguity_limit: float | None,
    bucket_budget: int | None,
    max_added_columns: int,
    include_base: bool,
    search: str,
    beam_width: int,
    max_evaluations: int | None,
    enforce_bucket_budget: bool,
    candidate_space_size: int,
) -> _SearchResult:
    state = _EvaluationState(
        data=data,
        base_public=base_public,
        target=target,
        candidate_refinements=candidate_refinements,
        scenario_specs=scenario_specs,
        weight=weight,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        max_evaluations=max_evaluations,
        enforce_bucket_budget=enforce_bucket_budget,
        candidates_by_subset={},
        result_subsets=[],
        result_subset_set=set(),
    )
    pruned_by_dominance = 0
    pruned_by_beam = 0
    if search == "exhaustive":
        stopping_reason = _exhaustive_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
        )
    elif search == "greedy":
        stopping_reason = _greedy_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
        )
    else:
        stopping_reason, pruned_by_dominance, pruned_by_beam = _beam_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
            beam_width=beam_width,
        )

    exact = (
        search == "exhaustive"
        and not state.evaluation_limit_hit
        and not enforce_bucket_budget
    )
    if state.evaluation_limit_hit:
        stopping_reason = "max_evaluations reached"
    return _SearchResult(
        candidates=state.result_candidates(),
        trace=FrontierSearchTrace(
            search=search,
            exact=exact,
            evaluated_candidates=state.evaluated_count,
            candidate_space_size=candidate_space_size,
            scenario_count=len(scenario_specs),
            max_added_columns=max_added_columns,
            max_evaluations=max_evaluations,
            beam_width=beam_width if search == "beam" else None,
            enforce_bucket_budget=enforce_bucket_budget,
            skipped_by_budget=state.skipped_by_budget,
            pruned_by_dominance=pruned_by_dominance,
            pruned_by_beam=pruned_by_beam,
            stopping_reason=stopping_reason,
        ),
    )


def _exhaustive_search(
    state: _EvaluationState,
    *,
    required_columns: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
) -> str:
    for added_columns in _candidate_subsets(
        state.candidate_refinements,
        max_added_columns=max_added_columns,
        include_base=include_base,
        required_columns=required_columns,
    ):
        if state.evaluate(added_columns) is None:
            return "max_evaluations reached"
    return "completed"


def _greedy_search(
    state: _EvaluationState,
    *,
    required_columns: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
) -> str:
    current = state.evaluate(
        required_columns,
        include_result=include_base or bool(required_columns),
    )
    if current is None:
        return "max_evaluations reached"
    if not state.is_allowed(current):
        return "bucket_budget reached"

    while len(current.added_columns) < max_added_columns:
        if _is_stable(current, state.ambiguity_limit):
            return "ambiguity_limit reached"

        proposals: list[PublicRepresentationCandidate] = []
        for column in state.candidate_refinements:
            if column in current.added_columns:
                continue
            candidate = state.evaluate((*current.added_columns, column))
            if candidate is None:
                return "max_evaluations reached"
            if state.is_allowed(candidate):
                proposals.append(candidate)

        if not proposals:
            return "no candidates"

        best = min(proposals, key=_candidate_search_rank)
        if not _improves(best, current):
            return "no improvement"
        current = best

    if _is_stable(current, state.ambiguity_limit):
        return "ambiguity_limit reached"
    return "max_added_columns reached"


def _beam_search(
    state: _EvaluationState,
    *,
    required_columns: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
    beam_width: int,
) -> tuple[str, int, int]:
    start = state.evaluate(
        required_columns,
        include_result=include_base or bool(required_columns),
    )
    if start is None:
        return "max_evaluations reached", 0, 0
    if not state.is_allowed(start):
        return "bucket_budget reached", 0, 0

    beam = [start]
    pruned_by_dominance = 0
    pruned_by_beam = 0
    while beam and len(beam[0].added_columns) < max_added_columns:
        child_subsets: set[tuple[str, ...]] = set()
        for candidate in beam:
            for column in state.candidate_refinements:
                if column in candidate.added_columns:
                    continue
                child_subsets.add(
                    _ordered_subset(
                        state.candidate_refinements,
                        (*candidate.added_columns, column),
                    )
                )

        if not child_subsets:
            return "no candidates", pruned_by_dominance, pruned_by_beam

        level_candidates = []
        for subset in sorted(child_subsets, key=lambda row: (len(row), row)):
            candidate = state.evaluate(subset)
            if candidate is None:
                return "max_evaluations reached", pruned_by_dominance, pruned_by_beam
            if state.is_allowed(candidate):
                level_candidates.append(candidate)

        if not level_candidates:
            return "no candidates", pruned_by_dominance, pruned_by_beam

        nondominated = list(_nondominated(level_candidates))
        pruned_by_dominance += len(level_candidates) - len(nondominated)
        ranked = sorted(nondominated, key=_candidate_search_rank)
        if len(ranked) > beam_width:
            pruned_by_beam += len(ranked) - beam_width
        beam = ranked[:beam_width]

    return "completed", pruned_by_dominance, pruned_by_beam


def _evaluate_candidate(
    data: Any,
    *,
    base_public: tuple[str, ...],
    target: str | RowMetric,
    added_columns: tuple[str, ...],
    scenario_specs: tuple[_ScenarioSpec, ...],
    weight: str | None,
    ambiguity_limit: float | None,
) -> PublicRepresentationCandidate:
    public_columns = base_public + added_columns
    scenarios = []
    public_cell_counts = []
    hidden_cell_counts = []
    for spec in scenario_specs:
        grouped = from_dataframe(
            data,
            public=public_columns,
            hidden=spec.hidden_columns,
            target=target,
            weight=weight,
            min_cell_weight=spec.min_cell_weight,
            q=spec.q,
        )
        interval = grouped.problem.global_transport_modulus()
        public_cells = len(grouped.problem.public_values)
        hidden_cells = len(grouped.problem.states)
        public_cell_counts.append(public_cells)
        hidden_cell_counts.append(hidden_cells)
        scenarios.append(
            FrontierScenarioResult(
                scenario=spec.scenario,
                q_name=q_name(spec.q),
                q_description=q_description(spec.q),
                min_cell_weight=spec.min_cell_weight,
                hidden_columns=spec.hidden_columns,
                public_cells=public_cells,
                hidden_cells=hidden_cells,
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
        public_cells=max(public_cell_counts),
        hidden_cells=max(hidden_cell_counts),
        min_public_cells=min(public_cell_counts),
        max_public_cells=max(public_cell_counts),
        min_hidden_cells=min(hidden_cell_counts),
        max_hidden_cells=max(hidden_cell_counts),
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


def _split_frontier(
    candidates: Sequence[PublicRepresentationCandidate],
) -> tuple[
    tuple[PublicRepresentationCandidate, ...],
    tuple[PublicRepresentationCandidate, ...],
]:
    frontier = _nondominated(candidates)
    dominated = tuple(row for row in candidates if row not in frontier)
    return frontier, dominated


def _nondominated(
    candidates: Sequence[PublicRepresentationCandidate],
) -> tuple[PublicRepresentationCandidate, ...]:
    return tuple(
        row
        for row in candidates
        if not any(_dominates(other, row) for other in candidates if other is not row)
    )


def _candidate_search_rank(
    row: PublicRepresentationCandidate,
) -> tuple[float, float, int, int, tuple[str, ...]]:
    return (
        row.max_ambiguity,
        row.mean_ambiguity,
        row.public_cells,
        row.added_column_count,
        row.added_columns,
    )


def _improves(
    candidate: PublicRepresentationCandidate,
    current: PublicRepresentationCandidate,
    *,
    tol: float = 1e-12,
) -> bool:
    return (
        candidate.max_ambiguity < current.max_ambiguity - tol
        or candidate.mean_ambiguity < current.mean_ambiguity - tol
    )


def _is_stable(
    candidate: PublicRepresentationCandidate,
    ambiguity_limit: float | None,
) -> bool:
    return ambiguity_limit is not None and candidate.max_ambiguity <= ambiguity_limit


def _candidate_space_size(
    columns: tuple[str, ...],
    *,
    max_added_columns: int,
    include_base: bool,
    required_columns: tuple[str, ...],
) -> int:
    required_count = len(required_columns)
    optional_count = len(columns) - required_count
    total = 0
    for added_count in range(required_count, max_added_columns + 1):
        if added_count == 0 and not include_base:
            continue
        optional_added = added_count - required_count
        if 0 <= optional_added <= optional_count:
            total += comb(optional_count, optional_added)
    return total


def _candidate_subsets(
    columns: tuple[str, ...],
    *,
    max_added_columns: int,
    include_base: bool,
    required_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    start = len(required_columns)
    subsets = []
    required_set = set(required_columns)
    optional_columns = tuple(column for column in columns if column not in required_set)
    for size in range(start, max_added_columns + 1):
        if size == 0 and not include_base:
            continue
        optional_size = size - len(required_columns)
        for optional_subset in combinations(optional_columns, optional_size):
            subsets.append(
                _ordered_subset(columns, (*required_columns, *optional_subset))
            )
    return tuple(subsets)


def _resolve_min_cell_weights(
    min_cell_weight: float,
    min_cell_weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if min_cell_weights is None:
        return (float(min_cell_weight),)
    weights = tuple(float(value) for value in min_cell_weights)
    if not weights:
        raise ValueError("min_cell_weights must contain at least one value")
    if any(value < 0 for value in weights):
        raise ValueError("min_cell_weights must be non-negative")
    return weights


def _resolve_hidden_sets(
    hidden: tuple[str, ...],
    hidden_sets: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    if hidden_sets is None:
        return (hidden,)
    hidden_grid = tuple(tuple(columns) for columns in hidden_sets)
    if not hidden_grid:
        raise ValueError("hidden_sets must contain at least one hidden-column set")
    if any(not columns for columns in hidden_grid):
        raise ValueError("hidden_sets cannot contain an empty hidden-column set")
    return hidden_grid


def _validate_hidden_grid(
    base_public: tuple[str, ...],
    hidden_sets: tuple[tuple[str, ...], ...],
) -> None:
    for index, hidden_columns in enumerate(hidden_sets, start=1):
        missing_public = [
            column for column in base_public if column not in hidden_columns
        ]
        if missing_public:
            raise ValueError(
                "base_public columns must appear in every hidden set; "
                f"hidden set {index} is missing {missing_public!r}"
            )


def _common_hidden_columns(
    hidden_sets: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    common = set(hidden_sets[0])
    for hidden_columns in hidden_sets[1:]:
        common &= set(hidden_columns)
    return tuple(column for column in hidden_sets[0] if column in common)


def _frontier_scenarios(
    *,
    hidden_sets: tuple[tuple[str, ...], ...],
    min_cell_weights: tuple[float, ...],
    q_presets: tuple[Any, ...],
) -> tuple[_ScenarioSpec, ...]:
    scenarios = []
    index = 0
    for hidden_columns in hidden_sets:
        for min_cell_weight in min_cell_weights:
            for q_preset in q_presets:
                index += 1
                scenarios.append(
                    _ScenarioSpec(
                        scenario=f"S{index}",
                        hidden_columns=hidden_columns,
                        min_cell_weight=min_cell_weight,
                        q=q_preset,
                    )
                )
    return tuple(scenarios)


def _valid_candidate_refinements(
    base_public: tuple[str, ...],
    hidden: tuple[str, ...],
    candidate_refinements: Sequence[str],
    *,
    must_include: tuple[str, ...],
    must_exclude: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hidden_set = set(hidden)
    base_set = set(base_public)
    _validate_column_constraints(
        base_set=base_set,
        hidden_set=hidden_set,
        must_include=must_include,
        must_exclude=must_exclude,
    )

    excluded = set(must_exclude)
    candidates = []
    seen = set()
    for column in candidate_refinements:
        if column in seen:
            continue
        seen.add(column)
        if column in base_set:
            continue
        if column in excluded:
            continue
        if column not in hidden_set:
            continue
        candidates.append(column)

    for column in must_include:
        if column in base_set:
            continue
        if column not in seen:
            candidates.append(column)
            seen.add(column)

    candidate_tuple = tuple(candidates)
    required_columns = tuple(
        column
        for column in candidate_tuple
        if column in must_include and column not in base_set
    )
    return candidate_tuple, required_columns


def _validate_column_constraints(
    *,
    base_set: set[str],
    hidden_set: set[str],
    must_include: tuple[str, ...],
    must_exclude: tuple[str, ...],
) -> None:
    overlap = sorted(set(must_include) & set(must_exclude))
    if overlap:
        raise ValueError(f"columns cannot be both required and excluded: {overlap!r}")

    allowed = base_set | hidden_set
    missing_required = [column for column in must_include if column not in allowed]
    if missing_required:
        raise ValueError(f"must_include columns are not in hidden: {missing_required!r}")
    missing_excluded = [column for column in must_exclude if column not in allowed]
    if missing_excluded:
        raise ValueError(f"must_exclude columns are not in hidden: {missing_excluded!r}")


def _ordered_subset(
    columns: tuple[str, ...],
    selected_columns: Sequence[str],
) -> tuple[str, ...]:
    selected = set(selected_columns)
    return tuple(column for column in columns if column in selected)


def _candidate_sort_key(
    row: PublicRepresentationCandidate,
) -> tuple[int, float, float, int, tuple[str, ...]]:
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


def _resolve_optional_int_arg(
    primary: int | None,
    alias: int | None,
    *,
    primary_name: str,
    alias_name: str,
) -> int | None:
    if primary is not None and alias is not None and primary != alias:
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _normalize_search(search: str) -> str:
    search_mode = search.strip().lower()
    if search_mode not in {"exhaustive", "greedy", "beam"}:
        raise ValueError("search must be one of: 'exhaustive', 'greedy', 'beam'")
    return search_mode


def _unique_tuple(values: Sequence[str]) -> tuple[str, ...]:
    output = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)


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
    if len(report.hidden_sets) > 1 or len(report.min_cell_weights) > 1:
        lines.append(
            "- This frontier is sensitivity-aware: every evaluated representation "
            "is scored across the requested hidden-column sets, min-cell "
            "thresholds, and Q presets. Candidate complexity uses the maximum "
            "public-cell count seen across those scenarios."
        )
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
        "| representation | public cells | hidden cells | added columns | max ambiguity | mean ambiguity | public adequate | stable |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
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
                    _format_range(row.min_public_cells, row.max_public_cells),
                    _format_range(row.min_hidden_cells, row.max_hidden_cells),
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
        "| representation | scenario | Q | min_cell_weight | hidden columns | public cells | hidden cells | observed | lower | upper | ambiguity | public adequate |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                        f"{row.min_cell_weight:g}",
                        _escape_table(", ".join(row.hidden_columns)),
                        str(row.public_cells),
                        str(row.hidden_cells),
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


def _format_range(lower: int, upper: int) -> str:
    if lower == upper:
        return str(upper)
    return f"{lower}-{upper}"
