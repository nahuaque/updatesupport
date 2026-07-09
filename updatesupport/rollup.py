"""Exact one-column categorical rollup design under saturated Q."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .claim import ClaimAudit, ClaimSpec
from .data import (
    GroupedProblem,
    _hashable_category,
    _iter_records,
    _record_value,
    from_dataframe,
)
from .partition import all_partitions
from .presets import normalize_q_preset, q_saturated


@dataclass(frozen=True)
class CategoricalRollupCandidate:
    """One exact categorical grouping evaluated under saturated Q."""

    groups: tuple[tuple[Hashable, ...], ...]
    public_cells: int
    hidden_cells: int
    observed_value: float
    lower: float
    upper: float
    ambiguity: float
    base_ambiguity: float
    passes_bucket_budget: bool
    meets_ambiguity_limit: bool | None
    decision_invariant: bool | None
    decision_certified: bool | None
    certifies_claim: bool

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def ambiguity_reduction(self) -> float:
        return max(0.0, self.base_ambiguity - self.ambiguity)

    @property
    def ambiguity_reduction_percent(self) -> float:
        if self.base_ambiguity <= 0.0:
            return 0.0
        return 100.0 * self.ambiguity_reduction / self.base_ambiguity

    @property
    def category_mapping(self) -> dict[Hashable, int]:
        return {
            category: group_index
            for group_index, group in enumerate(self.groups, start=1)
            for category in group
        }

    @property
    def label(self) -> str:
        return "; ".join(
            f"G{index}={{{', '.join(str(value) for value in group)}}}"
            for index, group in enumerate(self.groups, start=1)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "group_count": self.group_count,
            "label": self.label,
            "category_mapping": tuple(
                {"category": category, "group": group}
                for category, group in self.category_mapping.items()
            ),
            "public_cells": self.public_cells,
            "hidden_cells": self.hidden_cells,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "base_ambiguity": self.base_ambiguity,
            "ambiguity_reduction": self.ambiguity_reduction,
            "ambiguity_reduction_percent": self.ambiguity_reduction_percent,
            "passes_bucket_budget": self.passes_bucket_budget,
            "meets_ambiguity_limit": self.meets_ambiguity_limit,
            "decision_invariant": self.decision_invariant,
            "decision_certified": self.decision_certified,
            "certifies_claim": self.certifies_claim,
        }


@dataclass(frozen=True)
class CategoricalRollupDesign(ReportArtifactMixin):
    """Exact reporting design over rollups of one categorical column."""

    claim: ClaimSpec
    column: str
    categories: tuple[Hashable, ...]
    output_column: str
    max_groups: int
    max_categories: int
    bucket_budget: int | None
    evaluated_partition_count: int
    selected: CategoricalRollupCandidate
    base: CategoricalRollupCandidate
    best_by_group_count: tuple[CategoricalRollupCandidate, ...]
    frontier: tuple[CategoricalRollupCandidate, ...]
    status: str
    title: str = "Categorical Rollup Design"
    limitations: tuple[str, ...] = ()

    @property
    def category_count(self) -> int:
        return len(self.categories)

    @property
    def uses_rollup_column(self) -> bool:
        return 1 < self.selected.group_count < self.category_count

    @property
    def recommended_public(self) -> tuple[str, ...]:
        if self.selected.group_count == 1:
            return tuple(self.claim.public)
        if self.selected.group_count == self.category_count:
            return (*self.claim.public, self.column)
        return (*self.claim.public, self.output_column)

    @property
    def selected_mapping(self) -> dict[Hashable, int]:
        return self.selected.category_mapping

    @property
    def selected_claim(self) -> ClaimSpec:
        """Return the claim configured for the selected rollup."""

        saturated = q_saturated()
        if self.selected.group_count == 1:
            return replace(
                self.claim,
                q=saturated,
                q_presets=(saturated,),
            )
        if self.selected.group_count == self.category_count:
            return replace(
                self.claim,
                public=(*self.claim.public, self.column),
                hidden_sets=_hidden_sets_with_public_column(
                    self.claim.hidden_sets,
                    self.column,
                ),
                q=saturated,
                q_presets=(saturated,),
            )
        return replace(
            self.claim,
            public=(*self.claim.public, self.output_column),
            hidden=(*self.claim.hidden, self.output_column),
            hidden_sets=_hidden_sets_with_public_column(
                self.claim.hidden_sets,
                self.output_column,
            ),
            q=saturated,
            q_presets=(saturated,),
        )

    def transform(self, data: Any) -> tuple[dict[str, Any], ...]:
        """Apply the selected intermediate rollup to row records."""

        records = tuple(_iter_records(data))
        if not self.uses_rollup_column:
            return tuple(dict(row) for row in records)
        mapping = self.selected_mapping
        transformed: list[dict[str, Any]] = []
        for row_number, row in enumerate(records, start=1):
            if self.output_column in row:
                raise ValueError(
                    f"row {row_number} already contains rollup output column "
                    f"{self.output_column!r}"
                )
            category = _hashable_category(
                _record_value(row, self.column, row_number=row_number)
            )
            if category not in mapping:
                raise ValueError(
                    f"row {row_number} contains unseen rollup category "
                    f"{category!r} in column {self.column!r}"
                )
            transformed_row = dict(row)
            transformed_row[self.output_column] = mapping[category]
            transformed.append(transformed_row)
        return tuple(transformed)

    def audit(self, data: Any, **kwargs: Any) -> ClaimAudit:
        """Audit data using the selected categorical rollup."""

        return self.selected_claim.audit(self.transform(data), **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "claim": self.claim.as_dict(),
            "column": self.column,
            "categories": self.categories,
            "category_count": self.category_count,
            "output_column": self.output_column,
            "max_groups": self.max_groups,
            "max_categories": self.max_categories,
            "bucket_budget": self.bucket_budget,
            "evaluated_partition_count": self.evaluated_partition_count,
            "recommended_public": self.recommended_public,
            "uses_rollup_column": self.uses_rollup_column,
            "selected": self.selected.as_dict(),
            "base": self.base.as_dict(),
            "best_by_group_count": [row.as_dict() for row in self.best_by_group_count],
            "frontier": [row.as_dict() for row in self.frontier],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    "title": self.title,
                    "status": self.status,
                    "column": self.column,
                    "category_count": self.category_count,
                    "max_groups": self.max_groups,
                    "max_categories": self.max_categories,
                    "bucket_budget": self.bucket_budget,
                    "evaluated_partition_count": self.evaluated_partition_count,
                    "recommended_public": self.recommended_public,
                    "uses_rollup_column": self.uses_rollup_column,
                    "selected_group_count": self.selected.group_count,
                    "selected_public_cells": self.selected.public_cells,
                    "selected_ambiguity": self.selected.ambiguity,
                    "selected_certifies_claim": self.selected.certifies_claim,
                },
            ),
            "claim": (self.claim.as_dict(),),
            "selected_groups": tuple(
                {
                    "group": group_index,
                    "categories": group,
                    "category_count": len(group),
                }
                for group_index, group in enumerate(self.selected.groups, start=1)
            ),
            "best_by_group_count": tuple(
                row.as_dict() for row in self.best_by_group_count
            ),
            "frontier": tuple(row.as_dict() for row in self.frontier),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }

    def to_markdown(self, *, max_frontier_rows: int = 30) -> str:
        if max_frontier_rows < 0:
            raise ValueError("max_frontier_rows must be non-negative")
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Claim: {self.claim.estimate_name}",
            f"- Rolled-up column: `{self.column}`",
            f"- Retained categories: {self.category_count}",
            f"- Maximum groups searched: {self.max_groups}",
            f"- Exact partitions evaluated: {self.evaluated_partition_count}",
            f"- Status: `{self.status}`",
            f"- Selected groups: {self.selected.group_count}",
            f"- Selected public cells: {self.selected.public_cells}",
            f"- Base ambiguity: {self.base.ambiguity:.4f}",
            f"- Selected ambiguity: {self.selected.ambiguity:.4f}",
            "- Ambiguity reduction: "
            f"{self.selected.ambiguity_reduction:.4f} "
            f"({self.selected.ambiguity_reduction_percent:.1f}%)",
            "- Selected rollup certifies claim: "
            f"{_yes_no(self.selected.certifies_claim)}",
            "- Recommended public representation: "
            f"`{' + '.join(self.recommended_public)}`",
            "",
            "## Interpretation",
            "",
            _interpretation(self),
            "",
            "## Selected Category Groups",
            "",
            "| group | categories |",
            "| ---: | --- |",
        ]
        for group_index, group in enumerate(self.selected.groups, start=1):
            lines.append(
                f"| {group_index} | "
                + ", ".join(f"`{category}`" for category in group)
                + " |"
            )

        lines.extend(
            [
                "",
                "## Best Design By Group Count",
                "",
                "| groups | public cells | ambiguity | reduction | certifies | grouping |",
                "| ---: | ---: | ---: | ---: | :---: | --- |",
            ]
        )
        for row in self.best_by_group_count:
            lines.append(_candidate_markdown_row(row))

        lines.extend(
            [
                "",
                "## Pareto Frontier",
                "",
                "| groups | public cells | ambiguity | reduction | certifies | grouping |",
                "| ---: | ---: | ---: | ---: | :---: | --- |",
            ]
        )
        for row in self.frontier[:max_frontier_rows]:
            lines.append(_candidate_markdown_row(row))
        hidden = len(self.frontier) - min(len(self.frontier), max_frontier_rows)
        if hidden > 0:
            lines.append(f"| ... | ... | ... | ... | ... | {hidden} more rows |")

        lines.extend(["", "## Assumptions And Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


def design_categorical_rollup(
    data: Any,
    claim: ClaimSpec | Mapping[str, Any],
    *,
    column: str,
    max_groups: int | None = None,
    max_categories: int = 9,
    bucket_budget: int | None = None,
    output_column: str | None = None,
    tolerance: float = 1e-9,
    title: str = "Categorical Rollup Design",
) -> CategoricalRollupDesign:
    """Design an exact global rollup of one categorical column.

    Every set partition of the retained category levels up to ``max_groups`` is
    evaluated with the saturated public-fiber range formula. The selected
    design is the smallest budget-feasible rollup that certifies the claim, or
    the lowest-ambiguity budget-feasible design when no certifying rollup exists.
    """

    if isinstance(claim, Mapping):
        claim = ClaimSpec.from_dict(claim)
    if not isinstance(claim, ClaimSpec):
        raise TypeError("claim must be a ClaimSpec or mapping")
    if not isinstance(column, str) or not column:
        raise ValueError("column must be a non-empty string")
    if column not in claim.hidden:
        raise ValueError("rollup column must be present in claim.hidden")
    if column in claim.public:
        raise ValueError("rollup column must not already be public")
    preset = normalize_q_preset(claim.primary_q)
    if preset is None or preset.name != "saturated":
        raise ValueError(
            "categorical rollup design currently requires the claim's primary "
            "Q preset to be saturated"
        )
    if max_categories <= 0:
        raise ValueError("max_categories must be positive")
    if max_groups is not None and max_groups <= 0:
        raise ValueError("max_groups must be positive")
    if bucket_budget is not None and bucket_budget <= 0:
        raise ValueError("bucket_budget must be positive")
    if output_column is not None and (
        not isinstance(output_column, str) or not output_column
    ):
        raise ValueError("output_column must be a non-empty string or None")
    if output_column in claim.hidden:
        raise ValueError("output_column must not already be a claim column")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    records = tuple(_iter_records(data))
    if not records:
        raise ValueError("data must contain at least one row")
    grouped = from_dataframe(
        records,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        min_cell_weight=claim.min_cell_weight,
        q="saturated",
    )
    if not grouped.problem.has_linear_target:
        raise TypeError(
            "categorical rollup design currently requires a target that compiles "
            "to a fixed linear functional"
        )

    column_index = tuple(claim.hidden).index(column)
    categories = tuple(
        sorted(
            {state[column_index] for state in grouped.problem.states},
            key=str,
        )
    )
    if len(categories) > max_categories:
        raise ValueError(
            "exact categorical rollup search grows by Bell numbers; got "
            f"{len(categories)} retained categories with "
            f"max_categories={max_categories}"
        )
    effective_max_groups = (
        len(categories) if max_groups is None else min(max_groups, len(categories))
    )
    effective_bucket_budget = (
        claim.bucket_budget if bucket_budget is None else bucket_budget
    )

    raw_partitions = all_partitions(categories)
    partitions = tuple(
        _canonical_groups(partition, categories)
        for partition in raw_partitions
        if len(partition) <= effective_max_groups
    )
    base_groups = (categories,)
    base = _evaluate_candidate(
        grouped,
        claim,
        column_index=column_index,
        groups=base_groups,
        base_ambiguity=None,
        bucket_budget=effective_bucket_budget,
        tolerance=tolerance,
    )
    candidates = tuple(
        _evaluate_candidate(
            grouped,
            claim,
            column_index=column_index,
            groups=groups,
            base_ambiguity=base.ambiguity,
            bucket_budget=effective_bucket_budget,
            tolerance=tolerance,
        )
        for groups in partitions
    )
    base = next(row for row in candidates if row.group_count == 1)
    selected, status = _select_candidate(
        candidates,
        base=base,
        has_claim_requirement=(
            claim.ambiguity_limit is not None or claim.decision is not None
        ),
    )
    best_by_group_count = tuple(
        min(
            (row for row in candidates if row.group_count == group_count),
            key=_stability_sort_key,
        )
        for group_count in sorted({row.group_count for row in candidates})
    )
    frontier = tuple(
        sorted(
            _pareto_frontier(candidates, tolerance=tolerance),
            key=lambda row: (
                row.group_count,
                row.public_cells,
                row.ambiguity,
                row.label,
            ),
        )
    )
    output_column = _resolve_output_column(records, column, output_column)
    limitations = (
        "This first slice searches one categorical column and applies one global "
        "category mapping across all base public cells.",
        "The result is exact only over retained categories, the requested maximum "
        "group count, and saturated Q.",
        "The search does not learn different groupings by public cell, ordered "
        "numeric cutpoints, or constrained category hierarchies.",
        "Categories removed by minimum-cell filtering are outside the searched "
        "retained support.",
        "Exact enumeration grows by Bell numbers and is guarded by max_categories.",
    )
    return CategoricalRollupDesign(
        claim=claim,
        column=column,
        categories=categories,
        output_column=output_column,
        max_groups=effective_max_groups,
        max_categories=max_categories,
        bucket_budget=effective_bucket_budget,
        evaluated_partition_count=len(candidates),
        selected=selected,
        base=base,
        best_by_group_count=best_by_group_count,
        frontier=frontier,
        status=status,
        title=title,
        limitations=limitations,
    )


def _evaluate_candidate(
    grouped: GroupedProblem,
    claim: ClaimSpec,
    *,
    column_index: int,
    groups: tuple[tuple[Hashable, ...], ...],
    base_ambiguity: float | None,
    bucket_budget: int | None,
    tolerance: float,
) -> CategoricalRollupCandidate:
    mapping = {
        category: group_index
        for group_index, group in enumerate(groups, start=1)
        for category in group
    }
    aggregates: dict[tuple[tuple[Hashable, ...], int], list[float]] = defaultdict(
        lambda: [0.0, float("inf"), float("-inf")]
    )
    target_values = grouped.problem.estimand_map
    for state, mass in grouped.cell_weights.items():
        key = (grouped.problem.public_map[state], mapping[state[column_index]])
        row = aggregates[key]
        target = target_values[state]
        row[0] += mass
        row[1] = min(row[1], target)
        row[2] = max(row[2], target)

    lower = sum(mass * minimum for mass, minimum, _maximum in aggregates.values())
    upper = sum(mass * maximum for mass, _minimum, maximum in aggregates.values())
    ambiguity = max(0.0, upper - lower)
    observed_value = sum(
        grouped.cell_weights[state] * target_values[state]
        for state in grouped.problem.states
    )
    effective_base = ambiguity if base_ambiguity is None else base_ambiguity
    meets_ambiguity_limit = (
        None
        if claim.ambiguity_limit is None
        else ambiguity <= claim.ambiguity_limit + tolerance
    )
    decision_invariant = None
    decision_certified = None
    if claim.decision is not None:
        decision = claim.decision.interval_result(
            observed_value=observed_value,
            lower=lower,
            upper=upper,
        )
        decision_invariant = decision.invariant
        decision_certified = (
            decision.invariant
            and decision.certified_decision == decision.observed_decision
        )
    has_requirement = claim.ambiguity_limit is not None or claim.decision is not None
    certifies_claim = bool(
        has_requirement
        and (meets_ambiguity_limit is not False)
        and (decision_certified is not False)
    )
    public_cells = len(aggregates)
    return CategoricalRollupCandidate(
        groups=groups,
        public_cells=public_cells,
        hidden_cells=len(grouped.problem.states),
        observed_value=observed_value,
        lower=lower,
        upper=upper,
        ambiguity=ambiguity,
        base_ambiguity=effective_base,
        passes_bucket_budget=(bucket_budget is None or public_cells <= bucket_budget),
        meets_ambiguity_limit=meets_ambiguity_limit,
        decision_invariant=decision_invariant,
        decision_certified=decision_certified,
        certifies_claim=certifies_claim,
    )


def _select_candidate(
    candidates: Sequence[CategoricalRollupCandidate],
    *,
    base: CategoricalRollupCandidate,
    has_claim_requirement: bool,
) -> tuple[CategoricalRollupCandidate, str]:
    budget_feasible = [row for row in candidates if row.passes_bucket_budget]
    if not budget_feasible:
        return base, "no_budget_feasible"
    if has_claim_requirement:
        certifying = [row for row in budget_feasible if row.certifies_claim]
        if certifying:
            selected = min(
                certifying,
                key=lambda row: (
                    row.group_count,
                    row.public_cells,
                    row.ambiguity,
                    row.label,
                ),
            )
            status = (
                "already_certified"
                if selected.group_count == 1
                else "certifying_rollup_found"
            )
            return selected, status
        return min(budget_feasible, key=_stability_sort_key), "no_certifying_rollup"
    return min(budget_feasible, key=_stability_sort_key), "optimized"


def _stability_sort_key(
    row: CategoricalRollupCandidate,
) -> tuple[float, int, int, str]:
    return (row.ambiguity, row.group_count, row.public_cells, row.label)


def _pareto_frontier(
    candidates: Sequence[CategoricalRollupCandidate],
    *,
    tolerance: float,
) -> tuple[CategoricalRollupCandidate, ...]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            row.group_count,
            row.public_cells,
            row.ambiguity,
            row.label,
        ),
    )
    best_by_shape: dict[tuple[int, int], float] = {}
    frontier: list[CategoricalRollupCandidate] = []
    for candidate in ordered:
        dominated = any(
            ambiguity <= candidate.ambiguity + tolerance
            and (
                group_count < candidate.group_count
                or public_cells < candidate.public_cells
                or ambiguity < candidate.ambiguity - tolerance
            )
            for (group_count, public_cells), ambiguity in best_by_shape.items()
            if group_count <= candidate.group_count
            and public_cells <= candidate.public_cells
        )
        if not dominated:
            frontier.append(candidate)
        shape = (candidate.group_count, candidate.public_cells)
        best_by_shape[shape] = min(
            candidate.ambiguity,
            best_by_shape.get(shape, float("inf")),
        )
    return tuple(frontier)


def _canonical_groups(
    groups: Sequence[Sequence[Hashable]],
    categories: Sequence[Hashable],
) -> tuple[tuple[Hashable, ...], ...]:
    order = {category: index for index, category in enumerate(categories)}
    normalized = [
        tuple(sorted(group, key=lambda category: order[category])) for group in groups
    ]
    normalized.sort(key=lambda group: min(order[category] for category in group))
    return tuple(normalized)


def _resolve_output_column(
    records: Sequence[Mapping[str, Any]],
    column: str,
    requested: str | None,
) -> str:
    if requested is not None:
        if any(requested in row for row in records):
            raise ValueError(
                f"rollup output column {requested!r} already exists in the data"
            )
        return requested
    base = f"__updatesupport_rollup_{column}__"
    candidate = base
    suffix = 2
    while any(candidate in row for row in records):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _hidden_sets_with_public_column(
    hidden_sets: Sequence[Sequence[str]] | None,
    column: str,
) -> tuple[tuple[str, ...], ...] | None:
    if hidden_sets is None:
        return None
    return tuple(
        (*hidden_set, column) if column not in hidden_set else tuple(hidden_set)
        for hidden_set in hidden_sets
    )


def _candidate_markdown_row(row: CategoricalRollupCandidate) -> str:
    return (
        f"| {row.group_count} | {row.public_cells} | {row.ambiguity:.4f} | "
        f"{row.ambiguity_reduction:.4f} | {_yes_no(row.certifies_claim)} | "
        f"{row.label} |"
    )


def _interpretation(report: CategoricalRollupDesign) -> str:
    selected = report.selected
    if report.status == "already_certified":
        return (
            "The base public representation already certifies the claim under "
            "saturated recomposition. Publishing a rollup of the candidate column "
            "is not required for this claim."
        )
    if report.status == "certifying_rollup_found":
        return (
            f"A {selected.group_count}-group rollup of `{report.column}` is the "
            "smallest exact searched design that certifies the claim while "
            "respecting the public-cell budget. The grouping reduces saturated "
            f"ambiguity by {selected.ambiguity_reduction:.4f}."
        )
    if report.status == "no_budget_feasible":
        return (
            "Even the base public representation exceeds the supplied public-cell "
            "budget, so no searched rollup is budget-feasible."
        )
    if report.status == "no_certifying_rollup":
        return (
            "No searched budget-feasible rollup certifies the claim. The selected "
            "design is the lowest-ambiguity available grouping, not a certificate."
        )
    return (
        "No claim threshold was supplied. The selected design minimizes saturated "
        "ambiguity within the requested group and public-cell budgets."
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
