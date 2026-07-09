"""Shared public-representation design across a portfolio of claims."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .calibrated_design import CalibratedPublicReportDesign

from .artifacts import ReportArtifactMixin
from .claim import ClaimAudit, ClaimSpec
from .data import _iter_records
from .frontier import (
    FrontierScenarioResult,
    PublicRepresentationCandidate,
    PublicRepresentationFrontier,
    public_representation_frontier,
)


@dataclass(frozen=True)
class ClaimPortfolio:
    """Claims that must share one public reporting representation."""

    claims: Sequence[ClaimSpec | Mapping[str, Any]]
    name: str = "Claim Portfolio"
    description: str | None = None
    candidate_refinements: Sequence[str] | None = None

    def __post_init__(self) -> None:
        normalized = tuple(_coerce_claim(value) for value in self.claims)
        if len(normalized) < 2:
            raise ValueError("a claim portfolio requires at least two claims")
        if not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "claims", normalized)
        if self.candidate_refinements is not None:
            object.__setattr__(
                self,
                "candidate_refinements",
                _unique_strings(
                    self.candidate_refinements,
                    name="candidate_refinements",
                ),
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimPortfolio":
        """Build a claim portfolio from a JSON-compatible mapping."""

        return cls(
            claims=payload.get("claims", ()),
            name=payload.get("name", "Claim Portfolio"),
            description=payload.get("description"),
            candidate_refinements=payload.get("candidate_refinements"),
        )

    def design(self, data: Any, **kwargs: Any) -> "SharedRepresentationDesign":
        """Search for one public representation supporting every claim."""

        return design_shared_representation(data, self, **kwargs)

    def design_calibrated(
        self,
        historical_data: Any,
        current_data: Any,
        **kwargs: Any,
    ) -> "CalibratedPublicReportDesign":
        """Design one shared report under historically calibrated TV stress."""

        from .calibrated_design import design_calibrated_public_report

        return design_calibrated_public_report(
            historical_data,
            current_data,
            self,
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "candidate_refinements": self.candidate_refinements,
            "claims": [claim.as_dict() for claim in self.claims],
        }


@dataclass(frozen=True)
class PortfolioClaimScenarioResult:
    """One claim/scenario interval for a shared representation candidate."""

    scenario: str
    q_name: str
    min_cell_weight: float
    hidden_columns: tuple[str, ...]
    observed_value: float
    lower: float
    upper: float
    ambiguity: float
    ambiguity_limit_met: bool | None
    decision_invariant: bool | None
    decision_certified: bool | None
    certifies_scenario: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "q_name": self.q_name,
            "min_cell_weight": self.min_cell_weight,
            "hidden_columns": self.hidden_columns,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "ambiguity_limit_met": self.ambiguity_limit_met,
            "decision_invariant": self.decision_invariant,
            "decision_certified": self.decision_certified,
            "certifies_scenario": self.certifies_scenario,
        }


@dataclass(frozen=True)
class PortfolioClaimCandidateResult:
    """One claim's outcome for a shared representation candidate."""

    claim_index: int
    estimate_name: str
    target: str
    ambiguity_limit: float | None
    has_decision: bool
    scenarios: tuple[PortfolioClaimScenarioResult, ...]
    max_ambiguity: float
    mean_ambiguity: float
    ambiguity_limit_met: bool | None
    decision_certified: bool | None
    certifies_claim: bool
    normalized_ambiguity: float | None
    violation_score: float

    @property
    def observed_min(self) -> float:
        return min(row.observed_value for row in self.scenarios)

    @property
    def observed_max(self) -> float:
        return max(row.observed_value for row in self.scenarios)

    @property
    def lower(self) -> float:
        return min(row.lower for row in self.scenarios)

    @property
    def upper(self) -> float:
        return max(row.upper for row in self.scenarios)

    @property
    def worst_scenario(self) -> str:
        row = max(
            self.scenarios,
            key=lambda scenario: (
                not scenario.certifies_scenario,
                scenario.ambiguity,
                scenario.scenario,
            ),
        )
        return row.scenario

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "estimate_name": self.estimate_name,
            "target": self.target,
            "ambiguity_limit": self.ambiguity_limit,
            "has_decision": self.has_decision,
            "scenario_count": len(self.scenarios),
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "lower": self.lower,
            "upper": self.upper,
            "max_ambiguity": self.max_ambiguity,
            "mean_ambiguity": self.mean_ambiguity,
            "ambiguity_limit_met": self.ambiguity_limit_met,
            "decision_certified": self.decision_certified,
            "certifies_claim": self.certifies_claim,
            "normalized_ambiguity": self.normalized_ambiguity,
            "violation_score": self.violation_score,
            "worst_scenario": self.worst_scenario,
            "scenarios": [row.as_dict() for row in self.scenarios],
        }


@dataclass(frozen=True)
class SharedRepresentationCandidate:
    """One public representation evaluated across every portfolio claim."""

    added_columns: tuple[str, ...]
    public_columns: tuple[str, ...]
    min_public_cells: int
    max_public_cells: int
    claim_results: tuple[PortfolioClaimCandidateResult, ...]
    passes_bucket_budget: bool
    passes_required_columns: bool

    @property
    def added_column_count(self) -> int:
        return len(self.added_columns)

    @property
    def claim_count(self) -> int:
        return len(self.claim_results)

    @property
    def certified_claim_count(self) -> int:
        return sum(row.certifies_claim for row in self.claim_results)

    @property
    def uncertified_claim_count(self) -> int:
        return self.claim_count - self.certified_claim_count

    @property
    def certification_rate(self) -> float:
        return self.certified_claim_count / self.claim_count

    @property
    def all_claims_certified(self) -> bool:
        return self.certified_claim_count == self.claim_count

    @property
    def max_violation_score(self) -> float:
        return max(row.violation_score for row in self.claim_results)

    @property
    def max_normalized_ambiguity(self) -> float:
        values = [
            row.normalized_ambiguity
            for row in self.claim_results
            if row.normalized_ambiguity is not None
        ]
        return 0.0 if not values else max(values)

    @property
    def total_max_ambiguity(self) -> float:
        """Descriptive total; never used to rank claims on different scales."""

        return sum(row.max_ambiguity for row in self.claim_results)

    @property
    def eligible(self) -> bool:
        return self.passes_bucket_budget and self.passes_required_columns

    @property
    def label(self) -> str:
        if not self.added_columns:
            return "base public representation"
        return "base + " + ", ".join(self.added_columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_columns": self.added_columns,
            "added_column_count": self.added_column_count,
            "public_columns": self.public_columns,
            "min_public_cells": self.min_public_cells,
            "max_public_cells": self.max_public_cells,
            "claim_count": self.claim_count,
            "certified_claim_count": self.certified_claim_count,
            "uncertified_claim_count": self.uncertified_claim_count,
            "certification_rate": self.certification_rate,
            "all_claims_certified": self.all_claims_certified,
            "max_violation_score": self.max_violation_score,
            "max_normalized_ambiguity": self.max_normalized_ambiguity,
            "total_max_ambiguity": self.total_max_ambiguity,
            "passes_bucket_budget": self.passes_bucket_budget,
            "passes_required_columns": self.passes_required_columns,
            "eligible": self.eligible,
            "label": self.label,
            "claim_results": [row.as_dict() for row in self.claim_results],
        }


@dataclass(frozen=True)
class SharedRepresentationDesign(ReportArtifactMixin):
    """Exact shared public-representation search for a claim portfolio."""

    portfolio: ClaimPortfolio
    base_public: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    candidate_refinements: tuple[str, ...]
    required_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    max_added_columns: int
    bucket_budget: int | None
    evaluated_representation_count: int
    selected: SharedRepresentationCandidate
    base: SharedRepresentationCandidate
    candidates: tuple[SharedRepresentationCandidate, ...]
    frontier: tuple[SharedRepresentationCandidate, ...]
    status: str
    title: str = "Shared Representation Design"
    limitations: tuple[str, ...] = ()

    @property
    def recommended_public(self) -> tuple[str, ...]:
        return self.selected.public_columns

    @property
    def selected_claims(self) -> tuple[ClaimSpec, ...]:
        return tuple(
            replace(claim, public=self.selected.public_columns)
            for claim in self.portfolio.claims
        )

    def audit(self, data: Any, **kwargs: Any) -> tuple[ClaimAudit, ...]:
        """Audit every claim using the selected shared representation."""

        records = tuple(_iter_records(data))
        return tuple(claim.audit(records, **kwargs) for claim in self.selected_claims)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "portfolio": self.portfolio.as_dict(),
            "base_public": self.base_public,
            "hidden_columns": self.hidden_columns,
            "candidate_refinements": self.candidate_refinements,
            "required_columns": self.required_columns,
            "excluded_columns": self.excluded_columns,
            "max_added_columns": self.max_added_columns,
            "bucket_budget": self.bucket_budget,
            "evaluated_representation_count": self.evaluated_representation_count,
            "recommended_public": self.recommended_public,
            "selected": self.selected.as_dict(),
            "base": self.base.as_dict(),
            "frontier": [row.as_dict() for row in self.frontier],
            "candidates": [row.as_dict() for row in self.candidates],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        candidate_rows = tuple(_candidate_table_row(row) for row in self.candidates)
        selected_scenarios = tuple(
            {
                "claim_index": claim_result.claim_index,
                "estimate_name": claim_result.estimate_name,
                **scenario.as_dict(),
            }
            for claim_result in self.selected.claim_results
            for scenario in claim_result.scenarios
        )
        return {
            "summary": (
                {
                    "title": self.title,
                    "status": self.status,
                    "portfolio": self.portfolio.name,
                    "claim_count": len(self.portfolio.claims),
                    "base_public": self.base_public,
                    "candidate_refinements": self.candidate_refinements,
                    "required_columns": self.required_columns,
                    "excluded_columns": self.excluded_columns,
                    "max_added_columns": self.max_added_columns,
                    "bucket_budget": self.bucket_budget,
                    "evaluated_representation_count": (
                        self.evaluated_representation_count
                    ),
                    "recommended_public": self.recommended_public,
                    "selected_added_columns": self.selected.added_columns,
                    "selected_public_cells": self.selected.max_public_cells,
                    "selected_certified_claim_count": (
                        self.selected.certified_claim_count
                    ),
                    "selected_all_claims_certified": (
                        self.selected.all_claims_certified
                    ),
                },
            ),
            "claims": tuple(
                {"claim_index": index, **claim.as_dict()}
                for index, claim in enumerate(self.portfolio.claims, start=1)
            ),
            "candidates": candidate_rows,
            "frontier": tuple(_candidate_table_row(row) for row in self.frontier),
            "selected_claims": tuple(
                _claim_result_table_row(row) for row in self.selected.claim_results
            ),
            "selected_scenarios": selected_scenarios,
            "candidate_claims": tuple(
                {
                    "added_columns": candidate.added_columns,
                    "public_columns": candidate.public_columns,
                    **_claim_result_table_row(claim_result),
                }
                for candidate in self.candidates
                for claim_result in candidate.claim_results
            ),
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
            f"- Portfolio: {self.portfolio.name}",
            f"- Claims: {len(self.portfolio.claims)}",
            f"- Base public representation: `{' + '.join(self.base_public)}`",
            "- Candidate refinements: "
            + (", ".join(self.candidate_refinements) or "none"),
            f"- Exact representations evaluated: {self.evaluated_representation_count}",
            f"- Status: `{self.status}`",
            f"- Recommended public representation: `{' + '.join(self.recommended_public)}`",
            f"- Selected public cells: {self.selected.max_public_cells}",
            "- Claims certified by selected representation: "
            f"{self.selected.certified_claim_count}/{self.selected.claim_count}",
        ]
        if self.bucket_budget is not None:
            lines.append(f"- Public-cell budget: {self.bucket_budget}")
        if self.required_columns:
            lines.append(f"- Required refinements: {', '.join(self.required_columns)}")

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                _interpretation(self),
                "",
                "## Selected Claim Outcomes",
                "",
                "| claim | max ambiguity | limit | decision certified | claim certified | worst scenario |",
                "| --- | ---: | ---: | :---: | :---: | --- |",
            ]
        )
        for row in self.selected.claim_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.estimate_name,
                        f"{row.max_ambiguity:.4f}",
                        _format_optional(row.ambiguity_limit),
                        _yes_no_optional(row.decision_certified),
                        _yes_no(row.certifies_claim),
                        row.worst_scenario,
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Shared Representation Frontier",
                "",
                "| representation | public cells | certified | max violation | eligible |",
                "| --- | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in self.frontier[:max_frontier_rows]:
            lines.append(
                f"| {row.label} | {row.max_public_cells} | "
                f"{row.certified_claim_count}/{row.claim_count} | "
                f"{row.max_violation_score:.4f} | {_yes_no(row.eligible)} |"
            )
        hidden = len(self.frontier) - min(len(self.frontier), max_frontier_rows)
        if hidden > 0:
            lines.append(f"| ... | ... | ... | ... | {hidden} more rows |")

        lines.extend(["", "## Assumptions And Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


def claim_portfolio(
    *claims: ClaimSpec | Mapping[str, Any] | Sequence[ClaimSpec | Mapping[str, Any]],
    name: str = "Claim Portfolio",
    description: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
) -> ClaimPortfolio:
    """Create a claim portfolio from positional claims or one claim sequence."""

    normalized_claims: Sequence[ClaimSpec | Mapping[str, Any]]
    if len(claims) == 1 and isinstance(claims[0], (list, tuple)):
        normalized_claims = claims[0]
    else:
        normalized_claims = claims
    return ClaimPortfolio(
        claims=normalized_claims,
        name=name,
        description=description,
        candidate_refinements=candidate_refinements,
    )


def design_shared_representation(
    data: Any,
    portfolio: ClaimPortfolio
    | Sequence[ClaimSpec | Mapping[str, Any]]
    | Mapping[str, Any],
    *,
    candidate_refinements: Sequence[str] | None = None,
    max_added_columns: int | None = None,
    bucket_budget: int | None = None,
    max_evaluations: int | None = 4096,
    tolerance: float = 1e-9,
    title: str = "Shared Representation Design",
) -> SharedRepresentationDesign:
    """Search exact column subsets for one representation supporting all claims."""

    if isinstance(portfolio, Mapping):
        portfolio = ClaimPortfolio.from_dict(portfolio)
    elif not isinstance(portfolio, ClaimPortfolio):
        portfolio = ClaimPortfolio(claims=portfolio)
    claims = tuple(portfolio.claims)
    base_public, hidden_columns, weight = _validate_shared_claim_contract(claims)
    for claim in claims:
        if claim.ambiguity_limit is None and claim.decision is None:
            raise ValueError(
                "every portfolio claim must declare an ambiguity_limit or "
                f"decision rule; missing on {claim.estimate_name!r}"
            )

    requested = _resolve_candidate_refinements(
        portfolio,
        claims,
        candidate_refinements,
    )
    required_columns = _unique_strings(
        (
            column
            for claim in claims
            for column in claim.must_include
            if column not in base_public
        ),
        name="must_include",
    )
    excluded_columns = _unique_strings(
        (
            column
            for claim in claims
            for column in claim.must_exclude
            if column not in base_public
        ),
        name="must_exclude",
    )
    conflict = set(required_columns) & set(excluded_columns)
    if conflict:
        raise ValueError(
            "portfolio claims conflict on required/excluded refinements: "
            f"{sorted(conflict)!r}"
        )
    requested = _unique_strings(
        (*requested, *required_columns),
        name="candidate_refinements",
    )
    common_hidden = _common_hidden_scenario_columns(claims)
    invalid = [
        column
        for column in requested
        if column in base_public or column not in common_hidden
    ]
    if invalid:
        raise ValueError(
            "shared candidate refinements must be non-public columns present in "
            f"every claim hidden-set scenario; invalid={invalid!r}"
        )
    candidate_columns = tuple(
        column for column in requested if column not in excluded_columns
    )
    effective_max_added = (
        len(candidate_columns) if max_added_columns is None else int(max_added_columns)
    )
    if effective_max_added < 0:
        raise ValueError("max_added_columns must be non-negative")
    effective_max_added = min(effective_max_added, len(candidate_columns))
    if len(required_columns) > effective_max_added:
        raise ValueError("required portfolio refinements exceed max_added_columns")
    if bucket_budget is not None and bucket_budget < 0:
        raise ValueError("bucket_budget must be non-negative")
    if max_evaluations is not None and max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive or None")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    candidate_space_size = sum(
        comb(len(candidate_columns), count) for count in range(effective_max_added + 1)
    )
    if max_evaluations is not None and candidate_space_size > max_evaluations:
        raise ValueError(
            "exact shared representation search would evaluate "
            f"{candidate_space_size} candidates, exceeding "
            f"max_evaluations={max_evaluations}"
        )
    effective_bucket_budget = _effective_bucket_budget(claims, bucket_budget)
    records = tuple(_iter_records(data))
    if not records:
        raise ValueError("data must contain at least one row")

    frontiers = tuple(
        _claim_frontier(
            records,
            claim,
            candidate_refinements=candidate_columns,
            max_added_columns=effective_max_added,
        )
        for claim in claims
    )
    candidate_maps = tuple(
        {row.added_columns: row for row in frontier.candidates}
        for frontier in frontiers
    )
    shared_keys = tuple(candidate_maps[0])
    expected_keys = set(shared_keys)
    if any(set(mapping) != expected_keys for mapping in candidate_maps[1:]):
        raise RuntimeError(
            "claim frontier searches evaluated different representation sets"
        )

    candidates = tuple(
        _shared_candidate(
            added_columns,
            claims=claims,
            per_claim=tuple(mapping[added_columns] for mapping in candidate_maps),
            base_public=base_public,
            required_columns=required_columns,
            bucket_budget=effective_bucket_budget,
            tolerance=tolerance,
        )
        for added_columns in shared_keys
    )
    base = next(row for row in candidates if not row.added_columns)
    selected, status = _select_shared_candidate(candidates, base=base)
    frontier = tuple(
        sorted(
            _shared_frontier(candidates, tolerance=tolerance),
            key=lambda row: (
                row.max_public_cells,
                row.added_column_count,
                row.uncertified_claim_count,
                row.max_violation_score,
                row.added_columns,
            ),
        )
    )
    limitations = (
        "This first slice searches exact subsets of existing categorical or "
        "pre-binned refinement columns; it does not jointly learn category rollups.",
        "All claims must share the same base public columns, retained hidden "
        "columns, and weight column.",
        "Each claim is evaluated under its own declared Q, hidden-set, and "
        "minimum-cell-weight scenarios.",
        "The shared public-cell count is the maximum realized count across claim "
        "scenario grids, so the budget is conservative when filtering differs.",
        "Best-effort ranking prioritizes the number of certified claims and "
        "normalized threshold violations; it does not add target values measured "
        "on incomparable scales.",
    )
    return SharedRepresentationDesign(
        portfolio=portfolio,
        base_public=base_public,
        hidden_columns=hidden_columns,
        candidate_refinements=candidate_columns,
        required_columns=required_columns,
        excluded_columns=excluded_columns,
        max_added_columns=effective_max_added,
        bucket_budget=effective_bucket_budget,
        evaluated_representation_count=len(candidates),
        selected=selected,
        base=base,
        candidates=candidates,
        frontier=frontier,
        status=status,
        title=title,
        limitations=limitations,
    )


def _claim_frontier(
    records: Sequence[Mapping[str, Any]],
    claim: ClaimSpec,
    *,
    candidate_refinements: tuple[str, ...],
    max_added_columns: int,
) -> PublicRepresentationFrontier:
    q_presets = (
        (claim.primary_q, *claim.q_presets)
        if claim.q is not None
        else tuple(claim.q_presets)
    )
    frontier = public_representation_frontier(
        records,
        base_public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        candidate_refinements=candidate_refinements,
        weight=claim.weight,
        min_cell_weight=claim.min_cell_weight,
        min_cell_weights=claim.min_cell_weights,
        hidden_sets=claim.hidden_sets,
        q_presets=q_presets,
        ambiguity_limit=claim.ambiguity_limit,
        max_added_columns=max_added_columns,
        search="exhaustive",
        include_base=True,
        title=f"{claim.estimate_name} Shared Representation Frontier",
    )
    if frontier.search_trace is None or not frontier.search_trace.exact:
        raise RuntimeError("shared representation search requires exact frontiers")
    return frontier


def _shared_candidate(
    added_columns: tuple[str, ...],
    *,
    claims: tuple[ClaimSpec, ...],
    per_claim: tuple[PublicRepresentationCandidate, ...],
    base_public: tuple[str, ...],
    required_columns: tuple[str, ...],
    bucket_budget: int | None,
    tolerance: float,
) -> SharedRepresentationCandidate:
    claim_results = tuple(
        _claim_candidate_result(
            index,
            claim,
            candidate,
            tolerance=tolerance,
        )
        for index, (claim, candidate) in enumerate(
            zip(claims, per_claim, strict=True),
            start=1,
        )
    )
    min_public_cells = min(row.min_public_cells for row in per_claim)
    max_public_cells = max(row.max_public_cells for row in per_claim)
    return SharedRepresentationCandidate(
        added_columns=added_columns,
        public_columns=(*base_public, *added_columns),
        min_public_cells=min_public_cells,
        max_public_cells=max_public_cells,
        claim_results=claim_results,
        passes_bucket_budget=(
            bucket_budget is None or max_public_cells <= bucket_budget
        ),
        passes_required_columns=set(required_columns) <= set(added_columns),
    )


def _claim_candidate_result(
    claim_index: int,
    claim: ClaimSpec,
    candidate: PublicRepresentationCandidate,
    *,
    tolerance: float,
) -> PortfolioClaimCandidateResult:
    scenarios = tuple(
        _claim_scenario_result(claim, row, tolerance=tolerance)
        for row in candidate.scenarios
    )
    max_ambiguity = max(row.ambiguity for row in scenarios)
    mean_ambiguity = sum(row.ambiguity for row in scenarios) / len(scenarios)
    ambiguity_limit_met = (
        None
        if claim.ambiguity_limit is None
        else all(row.ambiguity_limit_met for row in scenarios)
    )
    decision_certified = (
        None
        if claim.decision is None
        else all(row.decision_certified for row in scenarios)
    )
    certifies_claim = all(row.certifies_scenario for row in scenarios)
    normalized_ambiguity = _normalized_ambiguity(
        max_ambiguity,
        claim.ambiguity_limit,
        tolerance=tolerance,
    )
    ambiguity_violation = (
        0.0 if normalized_ambiguity is None else max(0.0, normalized_ambiguity - 1.0)
    )
    decision_violation = 0.0 if decision_certified is not False else 1.0
    return PortfolioClaimCandidateResult(
        claim_index=claim_index,
        estimate_name=claim.estimate_name,
        target=_target_label(claim.target),
        ambiguity_limit=claim.ambiguity_limit,
        has_decision=claim.decision is not None,
        scenarios=scenarios,
        max_ambiguity=max_ambiguity,
        mean_ambiguity=mean_ambiguity,
        ambiguity_limit_met=ambiguity_limit_met,
        decision_certified=decision_certified,
        certifies_claim=certifies_claim,
        normalized_ambiguity=normalized_ambiguity,
        violation_score=max(ambiguity_violation, decision_violation),
    )


def _claim_scenario_result(
    claim: ClaimSpec,
    row: FrontierScenarioResult,
    *,
    tolerance: float,
) -> PortfolioClaimScenarioResult:
    ambiguity_limit_met = (
        None
        if claim.ambiguity_limit is None
        else row.ambiguity <= claim.ambiguity_limit + tolerance
    )
    decision_invariant = None
    decision_certified = None
    if claim.decision is not None:
        decision = claim.decision.interval_result(
            observed_value=row.observed_value,
            lower=row.lower,
            upper=row.upper,
        )
        decision_invariant = decision.invariant
        decision_certified = (
            decision.invariant
            and decision.certified_decision == decision.observed_decision
        )
    certifies_scenario = bool(
        (ambiguity_limit_met is not False) and (decision_certified is not False)
    )
    return PortfolioClaimScenarioResult(
        scenario=row.scenario,
        q_name=row.q_name,
        min_cell_weight=row.min_cell_weight,
        hidden_columns=row.hidden_columns,
        observed_value=row.observed_value,
        lower=row.lower,
        upper=row.upper,
        ambiguity=row.ambiguity,
        ambiguity_limit_met=ambiguity_limit_met,
        decision_invariant=decision_invariant,
        decision_certified=decision_certified,
        certifies_scenario=certifies_scenario,
    )


def _select_shared_candidate(
    candidates: Sequence[SharedRepresentationCandidate],
    *,
    base: SharedRepresentationCandidate,
) -> tuple[SharedRepresentationCandidate, str]:
    feasible = [row for row in candidates if row.eligible]
    if not feasible:
        return base, "no_feasible_candidate"
    shared = [row for row in feasible if row.all_claims_certified]
    if shared:
        selected = min(
            shared,
            key=lambda row: (
                row.max_public_cells,
                row.added_column_count,
                row.max_normalized_ambiguity,
                row.added_columns,
            ),
        )
        status = (
            "already_certified"
            if not selected.added_columns
            else "shared_representation_found"
        )
        return selected, status
    selected = min(
        feasible,
        key=lambda row: (
            -row.certified_claim_count,
            row.max_violation_score,
            row.max_public_cells,
            row.added_column_count,
            row.added_columns,
        ),
    )
    return selected, "no_shared_representation"


def _shared_frontier(
    candidates: Sequence[SharedRepresentationCandidate],
    *,
    tolerance: float,
) -> tuple[SharedRepresentationCandidate, ...]:
    frontier = []
    for candidate in candidates:
        if any(
            _shared_dominates(other, candidate, tolerance=tolerance)
            for other in candidates
            if other is not candidate
        ):
            continue
        frontier.append(candidate)
    return tuple(frontier)


def _shared_dominates(
    left: SharedRepresentationCandidate,
    right: SharedRepresentationCandidate,
    *,
    tolerance: float,
) -> bool:
    no_worse = (
        left.max_public_cells <= right.max_public_cells
        and left.added_column_count <= right.added_column_count
        and left.uncertified_claim_count <= right.uncertified_claim_count
        and left.max_violation_score <= right.max_violation_score + tolerance
    )
    strictly_better = (
        left.max_public_cells < right.max_public_cells
        or left.added_column_count < right.added_column_count
        or left.uncertified_claim_count < right.uncertified_claim_count
        or left.max_violation_score < right.max_violation_score - tolerance
    )
    return no_worse and strictly_better


def _validate_shared_claim_contract(
    claims: tuple[ClaimSpec, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    first = claims[0]
    base_public = tuple(first.public)
    hidden = tuple(first.hidden)
    weight = first.weight
    for claim in claims[1:]:
        if tuple(claim.public) != base_public:
            raise ValueError("portfolio claims must share the same public columns")
        if tuple(claim.hidden) != hidden:
            raise ValueError("portfolio claims must share the same hidden columns")
        if claim.weight != weight:
            raise ValueError("portfolio claims must share the same weight column")
    for claim in claims:
        for hidden_set in claim.hidden_sets or (claim.hidden,):
            missing = [column for column in base_public if column not in hidden_set]
            if missing:
                raise ValueError(
                    f"claim {claim.estimate_name!r} hidden-set scenario is missing "
                    f"public columns: {missing!r}"
                )
    return base_public, hidden, weight


def _common_hidden_scenario_columns(claims: Sequence[ClaimSpec]) -> set[str]:
    common: set[str] | None = None
    for claim in claims:
        for hidden_set in claim.hidden_sets or (claim.hidden,):
            values = set(hidden_set)
            common = values if common is None else common & values
    return set() if common is None else common


def _resolve_candidate_refinements(
    portfolio: ClaimPortfolio,
    claims: Sequence[ClaimSpec],
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    if explicit is not None:
        return _unique_strings(explicit, name="candidate_refinements")
    if portfolio.candidate_refinements is not None:
        return tuple(portfolio.candidate_refinements)
    return _unique_strings(
        (column for claim in claims for column in claim.candidate_refinements),
        name="candidate_refinements",
    )


def _effective_bucket_budget(
    claims: Sequence[ClaimSpec],
    explicit: int | None,
) -> int | None:
    if explicit is not None:
        return explicit
    budgets = [
        claim.bucket_budget for claim in claims if claim.bucket_budget is not None
    ]
    return None if not budgets else min(budgets)


def _normalized_ambiguity(
    ambiguity: float,
    limit: float | None,
    *,
    tolerance: float,
) -> float | None:
    if limit is None:
        return None
    if limit > tolerance:
        return ambiguity / limit
    if ambiguity <= tolerance:
        return 0.0
    return 1.0 + ambiguity / max(tolerance, 1e-12)


def _coerce_claim(value: ClaimSpec | Mapping[str, Any]) -> ClaimSpec:
    if isinstance(value, ClaimSpec):
        return value
    if isinstance(value, Mapping):
        return ClaimSpec.from_dict(value)
    raise TypeError("portfolio claims must be ClaimSpec objects or mappings")


def _unique_strings(values: Sequence[str] | Any, *, name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain non-empty strings")
        if value not in result:
            result.append(value)
    return tuple(result)


def _target_label(target: Any) -> str:
    return str(getattr(target, "name", target))


def _candidate_table_row(row: SharedRepresentationCandidate) -> dict[str, Any]:
    return {
        key: value for key, value in row.as_dict().items() if key != "claim_results"
    }


def _claim_result_table_row(
    row: PortfolioClaimCandidateResult,
) -> dict[str, Any]:
    return {key: value for key, value in row.as_dict().items() if key != "scenarios"}


def _interpretation(report: SharedRepresentationDesign) -> str:
    if report.status == "already_certified":
        return (
            "The base public representation already certifies every portfolio "
            "claim across its declared stress-test scenarios."
        )
    if report.status == "shared_representation_found":
        return (
            "The selected public representation is the smallest exact searched "
            "design that certifies every portfolio claim while satisfying the "
            "shared reporting constraints."
        )
    if report.status == "no_feasible_candidate":
        return (
            "No evaluated representation satisfies the shared public-cell and "
            "required-column constraints. The displayed base representation is "
            "diagnostic, not a recommendation."
        )
    return (
        "No budget-feasible evaluated representation certifies every claim. The "
        "selected best-effort design certifies the largest number of claims, then "
        "minimizes the worst normalized threshold violation."
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _yes_no_optional(value: bool | None) -> str:
    return "n/a" if value is None else _yes_no(value)
