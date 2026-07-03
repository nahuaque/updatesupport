"""Pareto frontier search for public reporting representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any, Mapping, Sequence

from .data import TabularTarget, from_dataframe
from .environments import CvxpyError
from .presets import q_description, q_name
from .targets import ProcedureTarget


_SCALARIZED_COMPONENT_ORDER = (
    "max_ambiguity",
    "mean_ambiguity",
    "public_cells",
    "hidden_cells",
    "added_columns",
)

_SCALARIZED_WEIGHT_ALIASES = {
    "max_ambiguity": "max_ambiguity",
    "ambiguity": "max_ambiguity",
    "worst_ambiguity": "max_ambiguity",
    "mean_ambiguity": "mean_ambiguity",
    "average_ambiguity": "mean_ambiguity",
    "avg_ambiguity": "mean_ambiguity",
    "public_cells": "public_cells",
    "public_cell_count": "public_cells",
    "hidden_cells": "hidden_cells",
    "hidden_cell_count": "hidden_cells",
    "added_columns": "added_columns",
    "added_column_count": "added_columns",
}


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
    scalarized_weights: dict[str, float] | None = None
    solver: str | None = None
    solver_status: str | None = None
    objective_value: float | None = None
    optimization_guarantee: str | None = None
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
            "scalarized_weights": self.scalarized_weights,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "optimization_guarantee": self.optimization_guarantee,
            "stopping_reason": self.stopping_reason,
        }


@dataclass(frozen=True)
class FrontierScreenedRefinement:
    """A requested refinement column that was not evaluated."""

    column: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "reason": self.reason,
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
    scalarized_score: float | None = None
    scalarized_components: dict[str, float] = field(default_factory=dict)

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
            "scalarized_score": self.scalarized_score,
            "scalarized_components": dict(self.scalarized_components),
            "scenarios": [row.as_dict() for row in self.scenarios],
        }


@dataclass(frozen=True)
class FrontierScenarioComparison:
    """Baseline-vs-selected ambiguity comparison for one scenario."""

    scenario: str
    q_name: str
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    baseline_ambiguity: float | None
    selected_ambiguity: float
    reduction: float | None
    reduction_percent: float | None
    passes_ambiguity_limit: bool | None
    public_adequate: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "q_name": self.q_name,
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "baseline_ambiguity": self.baseline_ambiguity,
            "selected_ambiguity": self.selected_ambiguity,
            "reduction": self.reduction,
            "reduction_percent": self.reduction_percent,
            "passes_ambiguity_limit": self.passes_ambiguity_limit,
            "public_adequate": self.public_adequate,
        }


@dataclass(frozen=True)
class FrontierCloseAlternative:
    """Nearby dominated alternative shown in an explanation."""

    added_columns: tuple[str, ...]
    label: str
    public_cells: int
    max_ambiguity: float
    delta_public_cells: int
    delta_max_ambiguity: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_columns": self.added_columns,
            "label": self.label,
            "public_cells": self.public_cells,
            "max_ambiguity": self.max_ambiguity,
            "delta_public_cells": self.delta_public_cells,
            "delta_max_ambiguity": self.delta_max_ambiguity,
        }


@dataclass(frozen=True)
class FrontierCandidateExplanation:
    """Review-oriented explanation for one frontier candidate."""

    candidate: PublicRepresentationCandidate
    baseline: PublicRepresentationCandidate | None
    scenario_comparisons: tuple[FrontierScenarioComparison, ...]
    close_dominated_alternatives: tuple[FrontierCloseAlternative, ...]
    screened_refinements: tuple[FrontierScreenedRefinement, ...]
    search_trace: FrontierSearchTrace | None
    ambiguity_limit: float | None
    bucket_budget: int | None

    @property
    def baseline_ambiguity(self) -> float | None:
        return None if self.baseline is None else self.baseline.max_ambiguity

    @property
    def selected_ambiguity(self) -> float:
        return self.candidate.max_ambiguity

    @property
    def ambiguity_reduction(self) -> float | None:
        if self.baseline is None:
            return None
        return self.baseline.max_ambiguity - self.candidate.max_ambiguity

    @property
    def ambiguity_reduction_percent(self) -> float | None:
        if self.baseline is None or self.baseline.max_ambiguity <= 0:
            return None
        return 100.0 * self.ambiguity_reduction / self.baseline.max_ambiguity

    @property
    def added_public_cells(self) -> int | None:
        if self.baseline is None:
            return None
        return self.candidate.public_cells - self.baseline.public_cells

    @property
    def failing_scenarios(self) -> tuple[FrontierScenarioComparison, ...]:
        return tuple(
            row
            for row in self.scenario_comparisons
            if row.passes_ambiguity_limit is False
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "baseline": None if self.baseline is None else self.baseline.as_dict(),
            "baseline_ambiguity": self.baseline_ambiguity,
            "selected_ambiguity": self.selected_ambiguity,
            "ambiguity_reduction": self.ambiguity_reduction,
            "ambiguity_reduction_percent": self.ambiguity_reduction_percent,
            "added_public_cells": self.added_public_cells,
            "failing_scenarios": [row.as_dict() for row in self.failing_scenarios],
            "scenario_comparisons": [
                row.as_dict() for row in self.scenario_comparisons
            ],
            "close_dominated_alternatives": [
                row.as_dict() for row in self.close_dominated_alternatives
            ],
            "screened_refinements": [
                row.as_dict() for row in self.screened_refinements
            ],
            "search_trace": None
            if self.search_trace is None
            else self.search_trace.as_dict(),
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
        }

    def to_markdown(
        self, *, heading: str = "## Selected Representation Explanation"
    ) -> str:
        return "\n".join(_candidate_explanation_markdown(self, heading=heading))

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)


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
    requested_refinements: tuple[str, ...] = ()
    screened_refinements: tuple[FrontierScreenedRefinement, ...] = ()
    ambiguity_limit: float | None = None
    bucket_budget: int | None = None
    scalarized_weights: dict[str, float] | None = None
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
        feasible = [
            row for row in self.candidates if row.public_cells <= effective_budget
        ]
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

    @property
    def best_scalarized(self) -> PublicRepresentationCandidate | None:
        """Lowest scalarized-score candidate, if scalarization was requested."""

        scored = [row for row in self.candidates if row.scalarized_score is not None]
        if not scored:
            return None
        return min(
            scored,
            key=lambda row: (
                row.scalarized_score,
                row.max_ambiguity,
                row.public_cells,
                row.added_column_count,
                row.added_columns,
            ),
        )

    @property
    def baseline(self) -> PublicRepresentationCandidate | None:
        for row in self.candidates:
            if not row.added_columns:
                return row
        return None

    def explain(
        self,
        candidate: PublicRepresentationCandidate | Sequence[str],
        *,
        top_close: int = 5,
    ) -> FrontierCandidateExplanation:
        """Explain why one representation is or is not attractive."""

        if top_close < 0:
            raise ValueError("top_close must be non-negative")
        selected = self._resolve_candidate(candidate)
        return FrontierCandidateExplanation(
            candidate=selected,
            baseline=self.baseline,
            scenario_comparisons=_scenario_comparisons(
                baseline=self.baseline,
                selected=selected,
                ambiguity_limit=self.ambiguity_limit,
            ),
            close_dominated_alternatives=_close_dominated_alternatives(
                selected,
                self.dominated,
                top=top_close,
            ),
            screened_refinements=self.screened_refinements,
            search_trace=self.search_trace,
            ambiguity_limit=self.ambiguity_limit,
            bucket_budget=self.bucket_budget,
        )

    def explain_minimal_stable(
        self,
        *,
        top_close: int = 5,
    ) -> FrontierCandidateExplanation | None:
        """Explain the minimal stable representation, if one exists."""

        if self.minimal_stable is None:
            return None
        return self.explain(self.minimal_stable, top_close=top_close)

    def _resolve_candidate(
        self,
        candidate: PublicRepresentationCandidate | Sequence[str],
    ) -> PublicRepresentationCandidate:
        if isinstance(candidate, PublicRepresentationCandidate):
            selected_columns = candidate.added_columns
        else:
            selected_columns = _ordered_subset(
                self.candidate_refinements,
                candidate,
            )
        for row in self.candidates:
            if row.added_columns == selected_columns:
                return row
        raise ValueError(f"candidate was not evaluated: {tuple(selected_columns)!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "row_count": self.row_count,
            "base_public": self.base_public,
            "hidden_columns": self.hidden_columns,
            "hidden_sets": self.hidden_sets,
            "min_cell_weights": self.min_cell_weights,
            "candidate_refinements": self.candidate_refinements,
            "requested_refinements": self.requested_refinements,
            "screened_refinements": [
                row.as_dict() for row in self.screened_refinements
            ],
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
            "scalarized_weights": self.scalarized_weights,
            "search_trace": None
            if self.search_trace is None
            else self.search_trace.as_dict(),
            "minimal_stable": None
            if self.minimal_stable is None
            else self.minimal_stable.as_dict(),
            "best_under_bucket_budget": None
            if self.best_under_bucket_budget() is None
            else self.best_under_bucket_budget().as_dict(),
            "best_scalarized": None
            if self.best_scalarized is None
            else self.best_scalarized.as_dict(),
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
            if self.search_trace.solver is not None:
                lines.append(f"- Search solver: {self.search_trace.solver}")
            if self.search_trace.solver_status is not None:
                lines.append(
                    f"- Search solver status: {self.search_trace.solver_status}"
                )
            if self.search_trace.optimization_guarantee is not None:
                lines.append(
                    "- Search optimization guarantee: "
                    f"{self.search_trace.optimization_guarantee}"
                )
        if self.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.ambiguity_limit:.4f}")
        if self.bucket_budget is not None:
            lines.append(f"- Public-cell budget: {self.bucket_budget}")
        if self.scalarized_weights is not None:
            weights = ", ".join(
                f"{name}={value:g}"
                for name, value in sorted(self.scalarized_weights.items())
            )
            lines.append(f"- Scalarized objective weights: {weights}")

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
        best_scalarized = self.best_scalarized
        if best_scalarized is not None:
            lines.append(
                "- Best scalarized representation: "
                f"`{best_scalarized.label}` with score "
                f"{best_scalarized.scalarized_score:.4f}."
            )

        lines.extend(["", "## Interpretation", ""])
        lines.extend(_frontier_interpretation(self))
        explanation = self.explain_minimal_stable()
        if explanation is None and best_budget is not None:
            explanation = self.explain(best_budget)
        if explanation is None and self.frontier:
            explanation = self.explain(self.frontier[0])
        if explanation is not None:
            lines.extend([""])
            lines.extend(_candidate_explanation_markdown(explanation))
        lines.extend(["", "## Pareto Frontier", ""])
        lines.extend(
            _candidate_table(
                self.frontier,
                include_scalarized=self.scalarized_weights is not None,
            )
        )
        if self.dominated:
            lines.extend(["", "## Dominated Candidates", ""])
            lines.extend(
                _candidate_table(
                    self.dominated,
                    include_scalarized=self.scalarized_weights is not None,
                )
            )
        lines.extend(["", "## Scenario Details", ""])
        lines.extend(_scenario_table(self.frontier))
        return "\n".join(lines)

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)


def public_representation_frontier(
    data: Any,
    *,
    base_public: Sequence[str] | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str],
    target: TabularTarget,
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
    scalarized_weights: Mapping[str, float] | None = None,
    mip_solver: str | None = "SCIP",
    mip_solver_options: Mapping[str, Any] | None = None,
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
    if mip_solver_options is not None and not isinstance(
        mip_solver_options,
        Mapping,
    ):
        raise TypeError("mip_solver_options must be a mapping or None")
    if not q_presets:
        raise ValueError("q_presets must contain at least one preset")

    base_public_tuple = tuple(base_public)
    hidden_tuple = tuple(hidden)
    hidden_grid = _resolve_hidden_sets(hidden_tuple, hidden_sets)
    _validate_hidden_grid(base_public_tuple, hidden_grid)
    common_hidden = _common_hidden_columns(hidden_grid)
    must_include_tuple = _unique_tuple(must_include or ())
    must_exclude_tuple = _unique_tuple(must_exclude or ())
    requested_refinements = _unique_tuple(candidate_refinements)
    screened_refinements = _screened_refinements(
        requested_refinements,
        base_public=base_public_tuple,
        hidden_sets=hidden_grid,
        common_hidden=common_hidden,
        must_exclude=must_exclude_tuple,
    )
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
    normalized_scalarized_weights = _normalize_scalarized_weights(
        scalarized_weights,
        search=search_mode,
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
        scalarized_weights=normalized_scalarized_weights,
        mip_solver=mip_solver,
        mip_solver_options=mip_solver_options,
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
        requested_refinements=requested_refinements,
        screened_refinements=screened_refinements,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        scalarized_weights=normalized_scalarized_weights,
        title=title,
        row_count=row_count,
        search_trace=search_result.trace,
    )


@dataclass
class _SearchResult:
    candidates: tuple[PublicRepresentationCandidate, ...]
    trace: FrontierSearchTrace


@dataclass(frozen=True)
class _MipSearchOutcome:
    stopping_reason: str
    exact: bool
    solver: str | None
    solver_status: str | None
    objective_value: float | None
    optimization_guarantee: str | None


@dataclass(frozen=True)
class _MipScenarioData:
    hidden_columns: tuple[str, ...]
    states: tuple[tuple[Any, ...], ...]
    public_values: tuple[tuple[Any, ...], ...]
    targets: tuple[float, ...]
    weights: tuple[float, ...]
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True)
class _MipSolveResult:
    added_columns: tuple[str, ...] | None
    status: str
    objective_value: float | None


@dataclass
class _EvaluationState:
    data: Any
    base_public: tuple[str, ...]
    target: TabularTarget
    candidate_refinements: tuple[str, ...]
    scenario_specs: tuple[_ScenarioSpec, ...]
    weight: str | None
    ambiguity_limit: float | None
    bucket_budget: int | None
    max_evaluations: int | None
    scalarized_weights: dict[str, float] | None
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
                scalarized_weights=self.scalarized_weights,
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
        return tuple(
            self.candidates_by_subset[subset] for subset in self.result_subsets
        )


def _search_candidates(
    data: Any,
    *,
    base_public: tuple[str, ...],
    target: TabularTarget,
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
    scalarized_weights: dict[str, float] | None,
    mip_solver: str | None,
    mip_solver_options: Mapping[str, Any] | None,
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
        scalarized_weights=scalarized_weights,
        enforce_bucket_budget=enforce_bucket_budget,
        candidates_by_subset={},
        result_subsets=[],
        result_subset_set=set(),
    )
    pruned_by_dominance = 0
    pruned_by_beam = 0
    mip_outcome: _MipSearchOutcome | None = None
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
            scalarized_weights=None,
        )
    elif search == "scalarized":
        stopping_reason = _greedy_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
            scalarized_weights=scalarized_weights,
        )
    elif search == "mip":
        mip_outcome = _mip_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
            solver=mip_solver,
            solver_options=mip_solver_options,
            scalarized_weights=scalarized_weights,
        )
        stopping_reason = mip_outcome.stopping_reason
    else:
        stopping_reason, pruned_by_dominance, pruned_by_beam = _beam_search(
            state,
            required_columns=required_columns,
            max_added_columns=max_added_columns,
            include_base=include_base,
            beam_width=beam_width,
            scalarized_weights=scalarized_weights,
        )

    exact = (
        search == "exhaustive"
        and not state.evaluation_limit_hit
        and not enforce_bucket_budget
    ) or (
        search == "mip"
        and mip_outcome is not None
        and mip_outcome.exact
        and not state.evaluation_limit_hit
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
            scalarized_weights=scalarized_weights,
            solver=None if mip_outcome is None else mip_outcome.solver,
            solver_status=None if mip_outcome is None else mip_outcome.solver_status,
            objective_value=None
            if mip_outcome is None
            else mip_outcome.objective_value,
            optimization_guarantee=None
            if mip_outcome is None
            else mip_outcome.optimization_guarantee,
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
    scalarized_weights: dict[str, float] | None,
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

        best = min(
            proposals,
            key=lambda row: _candidate_search_rank(
                row,
                scalarized_weights=scalarized_weights,
            ),
        )
        if not _improves(best, current, scalarized_weights=scalarized_weights):
            if scalarized_weights is not None:
                return "no scalarized improvement"
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
    scalarized_weights: dict[str, float] | None,
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
        ranked = sorted(
            nondominated,
            key=lambda row: _candidate_search_rank(
                row,
                scalarized_weights=scalarized_weights,
            ),
        )
        if len(ranked) > beam_width:
            pruned_by_beam += len(ranked) - beam_width
        beam = ranked[:beam_width]

    return "completed", pruned_by_dominance, pruned_by_beam


def _mip_search(
    state: _EvaluationState,
    *,
    required_columns: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
    scalarized_weights: dict[str, float] | None,
) -> _MipSearchOutcome:
    _validate_mip_search(
        state,
        required_columns=required_columns,
        include_base=include_base,
    )
    if not state.candidate_refinements:
        candidate = state.evaluate(
            required_columns,
            include_result=include_base or bool(required_columns),
        )
        reason = "completed" if candidate is not None else "max_evaluations reached"
        return _MipSearchOutcome(
            stopping_reason=reason,
            exact=candidate is not None,
            solver=None,
            solver_status=None,
            objective_value=None,
            optimization_guarantee="no candidate refinements required a MIP solve",
        )
    if max_added_columns == 0 and not include_base and not required_columns:
        return _MipSearchOutcome(
            stopping_reason="no candidates",
            exact=True,
            solver=None,
            solver_status=None,
            objective_value=None,
            optimization_guarantee="no feasible candidate subsets under search bounds",
        )

    scenario_data = _mip_scenario_data(state)
    solver_name = _normalized_mip_solver(solver)
    stable_limit = state.ambiguity_limit
    solve = _solve_mip_column_selection(
        scenario_data,
        candidate_refinements=state.candidate_refinements,
        required_columns=required_columns,
        max_added_columns=max_added_columns,
        include_base=include_base,
        ambiguity_limit=stable_limit,
        bucket_budget=state.bucket_budget if state.enforce_bucket_budget else None,
        scalarized_weights=scalarized_weights,
        solver=solver_name,
        solver_options=solver_options,
    )
    stopping_reason = "mip optimal"
    if solve.added_columns is None and _is_infeasible_status(solve.status):
        if stable_limit is not None:
            fallback = _solve_mip_column_selection(
                scenario_data,
                candidate_refinements=state.candidate_refinements,
                required_columns=required_columns,
                max_added_columns=max_added_columns,
                include_base=include_base,
                ambiguity_limit=None,
                bucket_budget=state.bucket_budget
                if state.enforce_bucket_budget
                else None,
                scalarized_weights=scalarized_weights,
                solver=solver_name,
                solver_options=solver_options,
            )
            if fallback.added_columns is None:
                return _MipSearchOutcome(
                    stopping_reason="no candidates",
                    exact=True,
                    solver=solver_name,
                    solver_status=fallback.status,
                    objective_value=fallback.objective_value,
                    optimization_guarantee=_mip_guarantee(state),
                )
            solve = fallback
            stopping_reason = "no stable representation; mip best alternative optimal"
        else:
            return _MipSearchOutcome(
                stopping_reason="no candidates",
                exact=True,
                solver=solver_name,
                solver_status=solve.status,
                objective_value=solve.objective_value,
                optimization_guarantee=_mip_guarantee(state),
            )
    elif solve.added_columns is None:
        raise CvxpyError(
            "MIP frontier search did not converge to an optimal solution; "
            f"solver status was {solve.status!r}."
        )

    if include_base or required_columns:
        if (
            state.evaluate(
                required_columns,
                include_result=include_base or bool(required_columns),
            )
            is None
        ):
            return _MipSearchOutcome(
                stopping_reason="max_evaluations reached",
                exact=False,
                solver=solver_name,
                solver_status=solve.status,
                objective_value=solve.objective_value,
                optimization_guarantee=_mip_guarantee(state),
            )
    if state.evaluate(solve.added_columns) is None:
        return _MipSearchOutcome(
            stopping_reason="max_evaluations reached",
            exact=False,
            solver=solver_name,
            solver_status=solve.status,
            objective_value=solve.objective_value,
            optimization_guarantee=_mip_guarantee(state),
        )

    exact = state.bucket_budget is None or state.enforce_bucket_budget
    return _MipSearchOutcome(
        stopping_reason=stopping_reason,
        exact=exact,
        solver=solver_name,
        solver_status=solve.status,
        objective_value=solve.objective_value,
        optimization_guarantee=_mip_guarantee(state),
    )


def _validate_mip_search(
    state: _EvaluationState,
    *,
    required_columns: tuple[str, ...],
    include_base: bool,
) -> None:
    if isinstance(state.target, ProcedureTarget):
        raise ValueError(
            "search='mip' currently supports fixed row/column targets only; "
            "ProcedureTarget depends on the public representation and must be "
            "evaluated by exhaustive, greedy, beam, or scalarized search."
        )
    non_saturated = [
        q_name(spec.q) for spec in state.scenario_specs if q_name(spec.q) != "saturated"
    ]
    if non_saturated:
        raise ValueError(
            "search='mip' currently supports only saturated Q presets; "
            f"unsupported presets: {tuple(non_saturated)!r}"
        )
    if not include_base and not required_columns and not state.candidate_refinements:
        raise ValueError("search='mip' has no candidate representation to evaluate")


def _mip_scenario_data(state: _EvaluationState) -> tuple[_MipScenarioData, ...]:
    scenarios = []
    for spec in state.scenario_specs:
        grouped = from_dataframe(
            state.data,
            public=state.base_public,
            hidden=spec.hidden_columns,
            target=state.target,
            weight=state.weight,
            min_cell_weight=spec.min_cell_weight,
            q="saturated",
        )
        states = tuple(grouped.problem.states)
        scenarios.append(
            _MipScenarioData(
                hidden_columns=spec.hidden_columns,
                states=states,
                public_values=tuple(
                    grouped.problem.public_map[state] for state in states
                ),
                targets=tuple(
                    float(grouped.problem.estimand_map[state]) for state in states
                ),
                weights=tuple(float(grouped.cell_weights[state]) for state in states),
                candidate_indices=tuple(
                    spec.hidden_columns.index(column)
                    for column in state.candidate_refinements
                ),
            )
        )
    return tuple(scenarios)


def _solve_mip_column_selection(
    scenario_data: tuple[_MipScenarioData, ...],
    *,
    candidate_refinements: tuple[str, ...],
    required_columns: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
    ambiguity_limit: float | None,
    bucket_budget: int | None,
    scalarized_weights: dict[str, float] | None,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
) -> _MipSolveResult:
    cp = _import_cvxpy_for_mip()
    column_count = len(candidate_refinements)
    x = cp.Variable(column_count, boolean=True)
    constraints = [cp.sum(x) <= max_added_columns]
    for column in required_columns:
        constraints.append(x[candidate_refinements.index(column)] == 1)
    if not include_base and not required_columns:
        constraints.append(cp.sum(x) >= 1)

    max_ambiguity = cp.Variable(nonneg=True)
    max_public_cells = cp.Variable(nonneg=True)
    scenario_ambiguities = []
    target_scale = 1.0
    for scenario in scenario_data:
        scenario_range = max(scenario.targets) - min(scenario.targets)
        target_scale = max(target_scale, abs(scenario_range))
        ambiguity, public_cells, scenario_constraints = _mip_scenario_expressions(
            cp,
            x=x,
            scenario=scenario,
        )
        constraints.extend(scenario_constraints)
        constraints.append(max_ambiguity >= ambiguity)
        constraints.append(max_public_cells >= public_cells)
        scenario_ambiguities.append(ambiguity)

    if ambiguity_limit is not None:
        constraints.append(max_ambiguity <= ambiguity_limit)
    if bucket_budget is not None:
        constraints.append(max_public_cells <= bucket_budget)

    selected_count = cp.sum(x)
    mean_ambiguity = sum(scenario_ambiguities) / len(scenario_ambiguities)
    objective = _mip_objective(
        max_ambiguity=max_ambiguity,
        mean_ambiguity=mean_ambiguity,
        max_public_cells=max_public_cells,
        hidden_cells=max(len(scenario.states) for scenario in scenario_data),
        selected_count=selected_count,
        target_scale=target_scale,
        ambiguity_limit=ambiguity_limit,
        scalarized_weights=scalarized_weights,
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    options = dict(solver_options or {})
    if solver is None:
        problem.solve(**options)
    else:
        problem.solve(solver=solver, **options)

    status = str(problem.status)
    if not _is_optimal_status(status):
        return _MipSolveResult(
            added_columns=None,
            status=status,
            objective_value=None if problem.value is None else float(problem.value),
        )
    selected = tuple(
        column
        for column, value in zip(candidate_refinements, x.value, strict=True)
        if float(value) >= 0.5
    )
    return _MipSolveResult(
        added_columns=selected,
        status=status,
        objective_value=None if problem.value is None else float(problem.value),
    )


def _mip_scenario_expressions(
    cp: Any,
    *,
    x: Any,
    scenario: _MipScenarioData,
) -> tuple[Any, Any, list[Any]]:
    state_count = len(scenario.states)
    lower_bound = min(scenario.targets)
    upper_bound = max(scenario.targets)
    big_m = max(upper_bound - lower_bound, 1.0)
    high = cp.Variable(state_count)
    low = cp.Variable(state_count)
    representatives = cp.Variable(state_count, boolean=True)
    constraints = [
        high >= lower_bound,
        high <= upper_bound,
        low >= lower_bound,
        low <= upper_bound,
        high >= low,
    ]
    same_indicators: dict[tuple[int, int], Any | None] = {}
    for i, target_i in enumerate(scenario.targets):
        constraints.extend([high[i] >= target_i, low[i] <= target_i])

    for i in range(state_count):
        for k in range(i + 1, state_count):
            if scenario.public_values[i] != scenario.public_values[k]:
                continue
            indicator = _mip_same_fiber_indicator(
                cp,
                x=x,
                scenario=scenario,
                left=i,
                right=k,
                constraints=constraints,
            )
            same_indicators[(i, k)] = indicator
            target_i = scenario.targets[i]
            target_k = scenario.targets[k]
            if indicator is None:
                constraints.extend(
                    [
                        high[i] >= target_k,
                        low[i] <= target_k,
                        high[k] >= target_i,
                        low[k] <= target_i,
                    ]
                )
            else:
                constraints.extend(
                    [
                        high[i] >= target_k - big_m * (1 - indicator),
                        low[i] <= target_k + big_m * (1 - indicator),
                        high[k] >= target_i - big_m * (1 - indicator),
                        low[k] <= target_i + big_m * (1 - indicator),
                    ]
                )

    for i in range(state_count):
        previous = [
            same_indicators[(k, i)]
            for k in range(i)
            if scenario.public_values[k] == scenario.public_values[i]
        ]
        if not previous:
            constraints.append(representatives[i] == 1)
        elif any(indicator is None for indicator in previous):
            constraints.append(representatives[i] == 0)
        else:
            for indicator in previous:
                constraints.append(representatives[i] <= 1 - indicator)
            constraints.append(representatives[i] >= 1 - sum(previous))

    ambiguity = cp.sum(cp.multiply(scenario.weights, high - low))
    public_cells = cp.sum(representatives)
    return ambiguity, public_cells, constraints


def _mip_same_fiber_indicator(
    cp: Any,
    *,
    x: Any,
    scenario: _MipScenarioData,
    left: int,
    right: int,
    constraints: list[Any],
) -> Any | None:
    differing_columns = tuple(
        index
        for index, hidden_index in enumerate(scenario.candidate_indices)
        if scenario.states[left][hidden_index] != scenario.states[right][hidden_index]
    )
    if not differing_columns:
        return None
    indicator = cp.Variable(boolean=True)
    for index in differing_columns:
        constraints.append(indicator <= 1 - x[index])
    constraints.append(indicator >= 1 - sum(x[index] for index in differing_columns))
    return indicator


def _mip_objective(
    *,
    max_ambiguity: Any,
    mean_ambiguity: Any,
    max_public_cells: Any,
    hidden_cells: int,
    selected_count: Any,
    target_scale: float,
    ambiguity_limit: float | None,
    scalarized_weights: dict[str, float] | None,
) -> Any:
    if scalarized_weights is not None:
        return (
            scalarized_weights.get("max_ambiguity", 0.0) * max_ambiguity
            + scalarized_weights.get("mean_ambiguity", 0.0) * mean_ambiguity
            + scalarized_weights.get("public_cells", 0.0) * max_public_cells
            + scalarized_weights.get("hidden_cells", 0.0) * float(hidden_cells)
            + scalarized_weights.get("added_columns", 0.0) * selected_count
        )
    scaled_ambiguity = max_ambiguity / max(target_scale, 1.0)
    if ambiguity_limit is not None:
        return max_public_cells + 1e-6 * selected_count + 1e-9 * scaled_ambiguity
    return scaled_ambiguity + 1e-6 * max_public_cells + 1e-9 * selected_count


def _normalized_mip_solver(solver: str | None) -> str | None:
    if solver is None:
        return None
    cp = _import_cvxpy_for_mip()
    requested = str(solver)
    installed = {str(name).upper(): str(name) for name in cp.installed_solvers()}
    if requested.upper() in installed:
        return installed[requested.upper()]
    hint = ""
    if requested.upper() == "SCIP":
        hint = (
            " Install SCIP support with `pip install updatesupport[scip]` "
            "or `uv add updatesupport[scip]`."
        )
    raise CvxpyError(f"CVXPY solver {requested!r} is not installed.{hint}")


def _import_cvxpy_for_mip() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise CvxpyError(
            "search='mip' requires CVXPY and a MIP-capable solver. "
            "Install SCIP support with `pip install updatesupport[scip]` "
            "or `uv add updatesupport[scip]`."
        ) from exc
    return cp


def _is_optimal_status(status: str) -> bool:
    return status in {"optimal", "optimal_inaccurate"}


def _is_infeasible_status(status: str) -> bool:
    return status in {"infeasible", "infeasible_inaccurate"}


def _mip_guarantee(state: _EvaluationState) -> str:
    if state.bucket_budget is not None and not state.enforce_bucket_budget:
        return (
            "SCIP solved the saturated column-selection MIP objective, but the "
            "public-cell budget was reporting-only rather than a hard MIP "
            "constraint."
        )
    return (
        "SCIP solved the saturated column-selection MIP over the declared "
        "candidate space and search bounds."
    )


def _evaluate_candidate(
    data: Any,
    *,
    base_public: tuple[str, ...],
    target: TabularTarget,
    added_columns: tuple[str, ...],
    scenario_specs: tuple[_ScenarioSpec, ...],
    weight: str | None,
    ambiguity_limit: float | None,
    scalarized_weights: dict[str, float] | None,
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
    scalarized_components = _scalarized_components(
        public_cells=max(public_cell_counts),
        hidden_cells=max(hidden_cell_counts),
        added_column_count=len(added_columns),
        max_ambiguity=max_ambiguity,
        mean_ambiguity=mean_ambiguity,
    )
    scalarized_score = (
        None
        if scalarized_weights is None
        else _scalarized_score(scalarized_components, scalarized_weights)
    )
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
        scalarized_score=scalarized_score,
        scalarized_components=scalarized_components
        if scalarized_weights is not None
        else {},
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
    *,
    scalarized_weights: dict[str, float] | None = None,
) -> tuple[Any, ...]:
    if scalarized_weights is not None and row.scalarized_score is not None:
        return (
            row.scalarized_score,
            row.max_ambiguity,
            row.mean_ambiguity,
            row.public_cells,
            row.added_column_count,
            row.added_columns,
        )
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
    scalarized_weights: dict[str, float] | None = None,
    tol: float = 1e-12,
) -> bool:
    if (
        scalarized_weights is not None
        and candidate.scalarized_score is not None
        and current.scalarized_score is not None
    ):
        return candidate.scalarized_score < current.scalarized_score - tol
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


def _scenario_comparisons(
    *,
    baseline: PublicRepresentationCandidate | None,
    selected: PublicRepresentationCandidate,
    ambiguity_limit: float | None,
) -> tuple[FrontierScenarioComparison, ...]:
    baseline_by_scenario = (
        {} if baseline is None else {row.scenario: row for row in baseline.scenarios}
    )
    rows = []
    for selected_row in selected.scenarios:
        baseline_row = baseline_by_scenario.get(selected_row.scenario)
        baseline_ambiguity = None if baseline_row is None else baseline_row.ambiguity
        reduction = (
            None
            if baseline_ambiguity is None
            else baseline_ambiguity - selected_row.ambiguity
        )
        reduction_percent = (
            None
            if baseline_ambiguity is None or baseline_ambiguity <= 0
            else 100.0 * reduction / baseline_ambiguity
        )
        passes_ambiguity_limit = (
            None
            if ambiguity_limit is None
            else selected_row.ambiguity <= ambiguity_limit
        )
        rows.append(
            FrontierScenarioComparison(
                scenario=selected_row.scenario,
                q_name=selected_row.q_name,
                min_cell_weight=selected_row.min_cell_weight,
                hidden_columns=selected_row.hidden_columns,
                baseline_ambiguity=baseline_ambiguity,
                selected_ambiguity=selected_row.ambiguity,
                reduction=reduction,
                reduction_percent=reduction_percent,
                passes_ambiguity_limit=passes_ambiguity_limit,
                public_adequate=selected_row.public_adequate,
            )
        )
    return tuple(rows)


def _close_dominated_alternatives(
    selected: PublicRepresentationCandidate,
    dominated: Sequence[PublicRepresentationCandidate],
    *,
    top: int,
) -> tuple[FrontierCloseAlternative, ...]:
    if top == 0:
        return ()
    alternatives = []
    for row in dominated:
        if row.added_columns == selected.added_columns:
            continue
        alternatives.append(
            FrontierCloseAlternative(
                added_columns=row.added_columns,
                label=row.label,
                public_cells=row.public_cells,
                max_ambiguity=row.max_ambiguity,
                delta_public_cells=row.public_cells - selected.public_cells,
                delta_max_ambiguity=row.max_ambiguity - selected.max_ambiguity,
            )
        )
    alternatives.sort(
        key=lambda row: (
            abs(row.delta_max_ambiguity),
            abs(row.delta_public_cells),
            row.public_cells,
            row.label,
        )
    )
    return tuple(alternatives[:top])


def _screened_refinements(
    requested_refinements: tuple[str, ...],
    *,
    base_public: tuple[str, ...],
    hidden_sets: tuple[tuple[str, ...], ...],
    common_hidden: tuple[str, ...],
    must_exclude: tuple[str, ...],
) -> tuple[FrontierScreenedRefinement, ...]:
    base_set = set(base_public)
    excluded = set(must_exclude)
    union_hidden = set().union(*(set(columns) for columns in hidden_sets))
    common = set(common_hidden)
    screened = []
    for column in requested_refinements:
        if column in base_set:
            reason = "already public"
        elif column in excluded:
            reason = "excluded by must_exclude"
        elif column not in union_hidden:
            reason = "not present in any hidden set"
        elif column not in common:
            reason = "unavailable across hidden sets"
        else:
            continue
        screened.append(FrontierScreenedRefinement(column=column, reason=reason))
    return tuple(screened)


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
        raise ValueError(
            f"must_include columns are not in hidden: {missing_required!r}"
        )
    missing_excluded = [column for column in must_exclude if column not in allowed]
    if missing_excluded:
        raise ValueError(
            f"must_exclude columns are not in hidden: {missing_excluded!r}"
        )


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
    return grouped.problem.psi(grouped.cell_weights)


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
    if search_mode not in {"exhaustive", "greedy", "beam", "scalarized", "mip"}:
        raise ValueError(
            "search must be one of: 'exhaustive', 'greedy', 'beam', 'scalarized', 'mip'"
        )
    return search_mode


def _normalize_scalarized_weights(
    weights: Mapping[str, float] | None,
    *,
    search: str,
) -> dict[str, float] | None:
    if weights is None:
        if search == "scalarized":
            return {"max_ambiguity": 1.0}
        return None

    normalized = {name: 0.0 for name in _SCALARIZED_COMPONENT_ORDER}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip().lower().replace("-", "_")
        if name not in _SCALARIZED_WEIGHT_ALIASES:
            allowed = ", ".join(_SCALARIZED_COMPONENT_ORDER)
            raise ValueError(f"scalarized_weights keys must be among: {allowed}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("scalarized_weights must be non-negative")
        normalized[_SCALARIZED_WEIGHT_ALIASES[name]] += weight

    normalized = {name: weight for name, weight in normalized.items() if weight != 0.0}
    if not normalized:
        raise ValueError("scalarized_weights must contain at least one positive weight")
    return normalized


def _scalarized_components(
    *,
    public_cells: int,
    hidden_cells: int,
    added_column_count: int,
    max_ambiguity: float,
    mean_ambiguity: float,
) -> dict[str, float]:
    return {
        "max_ambiguity": float(max_ambiguity),
        "mean_ambiguity": float(mean_ambiguity),
        "public_cells": float(public_cells),
        "hidden_cells": float(hidden_cells),
        "added_columns": float(added_column_count),
    }


def _scalarized_score(
    components: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    return sum(
        float(weights.get(name, 0.0)) * float(components[name])
        for name in _SCALARIZED_COMPONENT_ORDER
    )


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
    if report.search_trace is not None and report.search_trace.search == "mip":
        lines.append(
            "- MIP search directly optimizes the saturated column-selection "
            "problem over the declared candidate columns. The candidate table "
            "contains evaluated evidence candidates, not an exhaustive Pareto "
            "frontier unless every candidate was also evaluated."
        )
    if report.scalarized_weights is not None:
        lines.append(
            "- The scalarized objective is an explicit weighted review score over "
            "ambiguity and representation complexity. It is useful for choosing "
            "one candidate from a frontier, but it does not replace the Pareto "
            "frontier or the scenario-level ambiguity evidence."
        )
    return lines


def _candidate_table(
    candidates: Sequence[PublicRepresentationCandidate],
    *,
    include_scalarized: bool = False,
) -> list[str]:
    headers = [
        "representation",
        "public cells",
        "hidden cells",
        "added columns",
        "max ambiguity",
        "mean ambiguity",
    ]
    alignments = ["---", "---", "---", "---:", "---:", "---:"]
    if include_scalarized:
        headers.append("scalarized score")
        alignments.append("---:")
    headers.extend(["public adequate", "stable"])
    alignments.extend(["---", "---"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in candidates:
        stable = (
            ""
            if row.passes_ambiguity_limit is None
            else ("yes" if row.passes_ambiguity_limit else "no")
        )
        cells = [
            _escape_table(row.label),
            _format_range(row.min_public_cells, row.max_public_cells),
            _format_range(row.min_hidden_cells, row.max_hidden_cells),
            str(row.added_column_count),
            f"{row.max_ambiguity:.4f}",
            f"{row.mean_ambiguity:.4f}",
        ]
        if include_scalarized:
            cells.append(
                "" if row.scalarized_score is None else f"{row.scalarized_score:.4f}"
            )
        cells.extend(
            [
                "yes" if row.public_adequate else "no",
                stable,
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
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


def _candidate_explanation_markdown(
    explanation: FrontierCandidateExplanation,
    *,
    heading: str = "## Selected Representation Explanation",
) -> list[str]:
    candidate = explanation.candidate
    lines = [
        heading,
        "",
        f"- Selected representation: `{candidate.label}`",
        f"- Added columns: {_format_columns(candidate.added_columns)}",
        f"- Public cells: {_format_range(candidate.min_public_cells, candidate.max_public_cells)}",
        f"- Max ambiguity: {candidate.max_ambiguity:.4f}",
        f"- Mean ambiguity: {candidate.mean_ambiguity:.4f}",
        f"- Public adequate in all scenarios: {'yes' if candidate.public_adequate else 'no'}",
    ]
    if explanation.baseline is not None:
        reduction = explanation.ambiguity_reduction
        reduction_percent = explanation.ambiguity_reduction_percent
        lines.extend(
            [
                f"- Baseline max ambiguity: {explanation.baseline.max_ambiguity:.4f}",
                "- Selected vs baseline max ambiguity: "
                f"{explanation.baseline.max_ambiguity:.4f} -> "
                f"{candidate.max_ambiguity:.4f}",
                "- Max-ambiguity reduction: "
                f"{_format_optional_float(reduction)} "
                f"({_format_optional_percent(reduction_percent)})",
                "- Added public cells vs baseline: "
                f"{_format_optional_int(explanation.added_public_cells)}",
            ]
        )
    if explanation.ambiguity_limit is not None:
        lines.append(
            "- Ambiguity limit result: "
            f"{'pass' if candidate.passes_ambiguity_limit else 'fail'} "
            f"(limit {explanation.ambiguity_limit:.4f})"
        )
    if explanation.search_trace is not None:
        exact = "exact" if explanation.search_trace.exact else "heuristic"
        lines.append(
            "- Search provenance: "
            f"{explanation.search_trace.search} ({exact}), "
            f"{explanation.search_trace.evaluated_candidates}/"
            f"{explanation.search_trace.candidate_space_size} candidates evaluated, "
            f"stopping reason `{explanation.search_trace.stopping_reason}`."
        )

    lines.extend(["", "### Ambiguity Reduction by Scenario", ""])
    lines.extend(_scenario_comparison_table(explanation.scenario_comparisons))

    lines.extend(["", "### Failing Scenarios", ""])
    if explanation.failing_scenarios:
        lines.extend(_scenario_comparison_table(explanation.failing_scenarios))
    elif explanation.ambiguity_limit is None:
        lines.append(
            "- No ambiguity limit was supplied, so no pass/fail scenario test was applied."
        )
    else:
        lines.append(
            "- No selected-representation scenario exceeds the ambiguity limit."
        )

    lines.extend(["", "### Close Dominated Alternatives", ""])
    if explanation.close_dominated_alternatives:
        lines.extend(_close_alternative_table(explanation.close_dominated_alternatives))
    else:
        lines.append("- No dominated alternatives were close enough to list.")

    lines.extend(["", "### Screened-Out Refinements", ""])
    if explanation.screened_refinements:
        for row in explanation.screened_refinements:
            lines.append(f"- `{row.column}`: {row.reason}.")
    else:
        lines.append("- No requested refinement columns were screened out.")
    return lines


def _scenario_comparison_table(
    rows: Sequence[FrontierScenarioComparison],
) -> list[str]:
    lines = [
        "| scenario | Q | min_cell_weight | hidden columns | baseline ambiguity | selected ambiguity | reduction | reduction pct | public adequate | stable |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        stable = (
            ""
            if row.passes_ambiguity_limit is None
            else ("yes" if row.passes_ambiguity_limit else "no")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.scenario,
                    _escape_table(row.q_name),
                    f"{row.min_cell_weight:g}",
                    _escape_table(", ".join(row.hidden_columns)),
                    _format_optional_float(row.baseline_ambiguity),
                    f"{row.selected_ambiguity:.4f}",
                    _format_optional_float(row.reduction),
                    _format_optional_percent(row.reduction_percent),
                    "yes" if row.public_adequate else "no",
                    stable,
                ]
            )
            + " |"
        )
    return lines


def _close_alternative_table(
    rows: Sequence[FrontierCloseAlternative],
) -> list[str]:
    lines = [
        "| representation | public cells | max ambiguity | delta public cells | delta max ambiguity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(row.label),
                    str(row.public_cells),
                    f"{row.max_ambiguity:.4f}",
                    f"{row.delta_public_cells:+d}",
                    f"{row.delta_max_ambiguity:+.4f}",
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


def _format_columns(columns: Sequence[str]) -> str:
    if not columns:
        return "none"
    return ", ".join(f"`{column}`" for column in columns)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}%"
