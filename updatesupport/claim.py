"""Claim-level reporting-stability audits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .calibration import HistoricalTVCalibrationReport

from .artifacts import ReportArtifactMixin
from .certificate import (
    RepresentationStabilityCertificate,
    certify_public_representation,
)
from .data import GroupedProblem, TabularTarget, from_dataframe
from .frontier import PublicRepresentationCandidate, public_representation_frontier
from .joint import (
    HiddenCompositionUncertaintyReport,
    NonparametricJointDistribution,
    hidden_composition_uncertainty,
)
from .presets import QPreset
from .report import (
    PublicDescentReport,
    RefinementAttributionReport,
    RefinementCandidate,
    StatisticalUncertainty,
    WitnessReport,
    _repeatable_data,
    attribute_refinement_ambiguity,
    public_fiber_diagnostics,
    public_descent_report,
)
from .results import TransportResult
from .spec import QSpec


@dataclass(frozen=True)
class DecisionRule:
    """Threshold decision rule for a reported scalar estimate."""

    operator: str
    threshold: float
    label: str | None = None
    pass_label: str = "pass"
    fail_label: str = "fail"

    def __post_init__(self) -> None:
        operator = _normalize_decision_operator(self.operator)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "threshold", float(self.threshold))
        if not self.pass_label:
            raise ValueError("pass_label cannot be empty")
        if not self.fail_label:
            raise ValueError("fail_label cannot be empty")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"value {self.operator} {self.threshold:g}"

    def evaluate(self, value: float) -> str:
        return self.pass_label if self._passes(float(value)) else self.fail_label

    def interval_result(
        self,
        *,
        observed_value: float,
        lower: float,
        upper: float,
    ) -> "DecisionResult":
        lower_value = float(lower)
        upper_value = float(upper)
        if lower_value > upper_value:
            raise ValueError("decision interval lower bound exceeds upper bound")
        observed_decision = self.evaluate(observed_value)
        lower_decision = self.evaluate(lower_value)
        upper_decision = self.evaluate(upper_value)
        invariant = lower_decision == upper_decision
        certified_decision = lower_decision if invariant else None
        threshold_crossed = not invariant
        reason = (
            f"All admissible values imply decision {certified_decision!r}."
            if invariant
            else (
                f"The admissible interval crosses the decision threshold "
                f"{self.threshold:g}."
            )
        )
        return DecisionResult(
            rule=self,
            observed_value=float(observed_value),
            lower=lower_value,
            upper=upper_value,
            observed_decision=observed_decision,
            lower_decision=lower_decision,
            upper_decision=upper_decision,
            invariant=invariant,
            certified_decision=certified_decision,
            threshold_crossed=threshold_crossed,
            reason=reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "threshold": self.threshold,
            "label": self.label,
            "pass_label": self.pass_label,
            "fail_label": self.fail_label,
        }

    @classmethod
    def from_value(cls, value: "DecisionRule | Mapping[str, Any]") -> "DecisionRule":
        if isinstance(value, DecisionRule):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("decision must be a DecisionRule, mapping, or None")

    def _passes(self, value: float) -> bool:
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == ">":
            return value > self.threshold
        raise AssertionError(f"unsupported decision operator: {self.operator!r}")


@dataclass(frozen=True)
class DecisionResult:
    """Decision-rule evaluation over a hidden-composition interval."""

    rule: DecisionRule
    observed_value: float
    lower: float
    upper: float
    observed_decision: str
    lower_decision: str
    upper_decision: str
    invariant: bool
    certified_decision: str | None
    threshold_crossed: bool
    reason: str

    @property
    def status(self) -> str:
        return "pass" if self.invariant else "fail"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.as_dict(),
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "observed_decision": self.observed_decision,
            "lower_decision": self.lower_decision,
            "upper_decision": self.upper_decision,
            "invariant": self.invariant,
            "certified_decision": self.certified_decision,
            "threshold_crossed": self.threshold_crossed,
            "status": self.status,
            "reason": self.reason,
        }


def threshold_decision(
    operator: str | None = None,
    threshold: float | None = None,
    *,
    pass_if: str | None = None,
    label: str | None = None,
    pass_label: str = "pass",
    fail_label: str = "fail",
) -> DecisionRule:
    """Create a threshold decision rule for claim audits."""

    if pass_if is not None:
        if operator is not None and _normalize_decision_operator(operator) != (
            _normalize_decision_operator(pass_if)
        ):
            raise TypeError("use either operator or pass_if, not both")
        operator = pass_if
    if operator is None:
        raise TypeError("threshold_decision() missing required operator/pass_if")
    if threshold is None:
        raise TypeError("threshold_decision() missing required threshold")
    return DecisionRule(
        operator=operator,
        threshold=float(threshold),
        label=label,
        pass_label=pass_label,
        fail_label=fail_label,
    )


@dataclass(frozen=True)
class ModelAssistedDrawResult:
    """One model-assisted joint-composition draw evaluated for stability."""

    draw_index: int
    observed_value: float | None
    lower: float | None
    upper: float | None
    ambiguity: float | None
    public_adequate: bool | None
    status: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class ModelAssistedStabilitySummary:
    """Stability summary over fitted-joint distribution draws."""

    joint_model: NonparametricJointDistribution
    rows: tuple[ModelAssistedDrawResult, ...]
    ambiguity_limit: float | None = None
    seed: int | None = None
    uncertainty_report: HiddenCompositionUncertaintyReport | None = None

    @property
    def draw_count(self) -> int:
        return len(self.rows)

    @property
    def successful_draws(self) -> int:
        return sum(row.error is None for row in self.rows)

    @property
    def failed_draws(self) -> int:
        return sum(row.status == "fail" for row in self.rows)

    @property
    def error_count(self) -> int:
        return sum(row.error is not None for row in self.rows)

    @property
    def failure_rate(self) -> float | None:
        if self.ambiguity_limit is None:
            return None
        if self.successful_draws == 0:
            return None
        return self.failed_draws / self.successful_draws

    @property
    def public_adequate_rate(self) -> float | None:
        evaluated = [row for row in self.rows if row.public_adequate is not None]
        if not evaluated:
            return None
        return sum(bool(row.public_adequate) for row in evaluated) / len(evaluated)

    @property
    def ambiguities(self) -> tuple[float, ...]:
        return tuple(row.ambiguity for row in self.rows if row.ambiguity is not None)

    @property
    def ambiguity_min(self) -> float | None:
        values = self.ambiguities
        return None if not values else min(values)

    @property
    def ambiguity_max(self) -> float | None:
        values = self.ambiguities
        return None if not values else max(values)

    @property
    def ambiguity_mean(self) -> float | None:
        values = self.ambiguities
        return None if not values else sum(values) / len(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_count": self.draw_count,
            "successful_draws": self.successful_draws,
            "failed_draws": self.failed_draws,
            "error_count": self.error_count,
            "failure_rate": self.failure_rate,
            "public_adequate_rate": self.public_adequate_rate,
            "ambiguity_min": self.ambiguity_min,
            "ambiguity_max": self.ambiguity_max,
            "ambiguity_mean": self.ambiguity_mean,
            "ambiguity_limit": self.ambiguity_limit,
            "seed": self.seed,
            "metric_summaries": []
            if self.uncertainty_report is None
            else [row.as_dict() for row in self.uncertainty_report.metric_summaries],
            "joint_model": self.joint_model.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class ClaimRefinementRecommendation:
    """Claim-centered public-refinement recommendation."""

    columns: tuple[str, ...]
    source: str
    before_ambiguity: float | None
    after_ambiguity: float
    reduction: float | None
    reduction_percent: float | None
    public_cells: int
    meets_ambiguity_limit: bool | None = None
    selected_repair: bool = False
    decision_repair: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        return _column_label(self.columns)

    @property
    def column(self) -> str:
        """Compatibility label for one-column recommendation tables."""

        return self.label

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "column": self.column,
            "label": self.label,
            "source": self.source,
            "before_ambiguity": self.before_ambiguity,
            "after_ambiguity": self.after_ambiguity,
            "reduction": self.reduction,
            "reduction_percent": self.reduction_percent,
            "public_cells": self.public_cells,
            "meets_ambiguity_limit": self.meets_ambiguity_limit,
            "selected_repair": self.selected_repair,
            "decision_repair": self.decision_repair,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClaimRepairOption:
    """One candidate public-representation repair for a claim."""

    rank: int
    columns: tuple[str, ...]
    source: str
    cost: float
    before_ambiguity: float | None
    after_ambiguity: float
    reduction: float | None
    reduction_percent: float | None
    public_cells: int
    satisfies_ambiguity_limit: bool | None = None
    certifies_claim: bool = False
    selected_repair: bool = False
    decision_repair: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        return _column_label(self.columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "columns": self.columns,
            "label": self.label,
            "source": self.source,
            "cost": self.cost,
            "before_ambiguity": self.before_ambiguity,
            "after_ambiguity": self.after_ambiguity,
            "reduction": self.reduction,
            "reduction_percent": self.reduction_percent,
            "public_cells": self.public_cells,
            "satisfies_ambiguity_limit": self.satisfies_ambiguity_limit,
            "certifies_claim": self.certifies_claim,
            "selected_repair": self.selected_repair,
            "decision_repair": self.decision_repair,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClaimRepairPlan(ReportArtifactMixin):
    """Cost-aware consolidation of claim repair and refinement evidence."""

    audit: "ClaimAudit"
    options: tuple[ClaimRepairOption, ...]
    action_costs: Mapping[str, float] = field(default_factory=dict)
    title: str = "Claim Repair Plan"

    @property
    def recommended(self) -> ClaimRepairOption | None:
        if self.audit.passed:
            return None
        for option in self.options:
            if option.certifies_claim:
                return option
        return None

    @property
    def certifying_options(self) -> tuple[ClaimRepairOption, ...]:
        return tuple(option for option in self.options if option.certifies_claim)

    @property
    def non_certifying_options(self) -> tuple[ClaimRepairOption, ...]:
        return tuple(option for option in self.options if not option.certifies_claim)

    @property
    def status(self) -> str:
        if self.audit.passed:
            return "already_certified"
        if self.recommended is not None:
            return "repair_found"
        if self.options:
            return "no_certifying_repair"
        return "no_repair_candidates"

    @property
    def recommended_label(self) -> str | None:
        recommended = self.recommended
        return None if recommended is None else recommended.label

    def as_dict(self) -> dict[str, Any]:
        recommended = self.recommended
        return {
            "title": self.title,
            "status": self.status,
            "claim_status": self.audit.status,
            "claim": self.audit.claim.as_dict(),
            "observed_value": self.audit.observed_value,
            "lower": self.audit.interval.lower,
            "upper": self.audit.interval.upper,
            "ambiguity": self.audit.ambiguity,
            "ambiguity_limit": self.audit.claim.ambiguity_limit,
            "has_decision": self.audit.decision is not None,
            "decision_invariant": None
            if self.audit.decision is None
            else self.audit.decision.invariant,
            "recommended": None if recommended is None else recommended.as_dict(),
            "options": [option.as_dict() for option in self.options],
            "action_costs": dict(self.action_costs),
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return _claim_repair_plan_tables(self)

    def to_markdown(self) -> str:
        recommended = self.recommended
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Claim: {self.audit.claim.estimate_name}",
            f"- Claim status: **{self.audit.status.upper()}**",
            f"- Plan status: `{self.status}`",
            f"- Current ambiguity: {self.audit.ambiguity:.4f}",
        ]
        if self.audit.claim.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.audit.claim.ambiguity_limit:.4f}")
        if self.audit.decision is not None:
            lines.extend(
                [
                    f"- Decision rule: {self.audit.decision.rule.name}",
                    "- Decision invariant: "
                    f"{'yes' if self.audit.decision.invariant else 'no'}",
                ]
            )
        if self.audit.passed:
            lines.append("- Recommended repair: none required")
        elif recommended is None:
            lines.append("- Recommended repair: none found")
        else:
            lines.extend(
                [
                    f"- Recommended repair: `{recommended.label}`",
                    f"- Recommended repair cost: {recommended.cost:g}",
                    f"- Ambiguity after repair: {recommended.after_ambiguity:.4f}",
                    f"- Public cells after repair: {recommended.public_cells}",
                    f"- Repair signal: {recommended.reason}",
                ]
            )

        lines.extend(["", "## Interpretation", ""])
        lines.append(_repair_plan_interpretation(self))

        if self.options:
            lines.extend(
                [
                    "",
                    "## Candidate Repair Options",
                    "",
                    "| rank | refinement | cost | certifies | after | reduction | public cells | signal |",
                    "| ---: | --- | ---: | :---: | ---: | ---: | ---: | --- |",
                ]
            )
            for option in self.options:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(option.rank),
                            f"`{option.label}`",
                            f"{option.cost:g}",
                            "yes" if option.certifies_claim else "no",
                            f"{option.after_ambiguity:.4f}",
                            _format_optional_float(option.reduction),
                            str(option.public_cells),
                            _escape_table(option.reason),
                        ]
                    )
                    + " |"
                )
        else:
            lines.extend(
                [
                    "",
                    "## Candidate Repair Options",
                    "",
                    "No candidate public-refinement actions were available.",
                ]
            )

        lines.extend(
            [
                "",
                "## Scope",
                "",
                "- This plan does not run a separate optimization subsystem; it "
                "consolidates the claim audit, decision repair, certificate, "
                "frontier, and refinement-recommendation outputs already "
                "computed for the claim.",
                "- Costs are analyst-supplied action costs for public refinements. "
                "They are not inferred from the data.",
                "- A certifying option means the existing claim audit evidence says "
                "the repair satisfies the declared ambiguity or decision criterion. "
                "Other refinements may still be useful diagnostics.",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class PublicReportDesign(ReportArtifactMixin):
    """One-stop design artifact for the smallest defensible public report."""

    audit: "ClaimAudit"
    repair_plan: ClaimRepairPlan
    certificate: RepresentationStabilityCertificate | None = None
    frontier: Any | None = None
    attribution: RefinementAttributionReport | None = None
    title: str = "Public Report Design"

    @property
    def status(self) -> str:
        if self.audit.passed:
            return "already_defensible"
        if self.repair_plan.recommended is not None:
            return "repair_available"
        if (
            self.certificate is not None
            and self.certificate.selected_candidate is not None
        ):
            return "representation_available"
        if self.audit.failed:
            return "no_defensible_representation_found"
        return "inconclusive"

    @property
    def recommended_option(self) -> ClaimRepairOption | None:
        return self.repair_plan.recommended

    @property
    def selected_candidate(self) -> PublicRepresentationCandidate | None:
        if self.certificate is None:
            return None
        return self.certificate.selected_candidate

    @property
    def recommended_public(self) -> tuple[str, ...] | None:
        if self.audit.passed:
            return tuple(self.audit.claim.public)
        recommended = self.repair_plan.recommended
        if recommended is not None:
            return (*self.audit.claim.public, *recommended.columns)
        candidate = self.selected_candidate
        if candidate is not None:
            return (*self.audit.claim.public, *candidate.added_columns)
        return None

    @property
    def recommended_label(self) -> str | None:
        public = self.recommended_public
        return None if public is None else " + ".join(public)

    def as_dict(self) -> dict[str, Any]:
        recommended = self.recommended_option
        return {
            "title": self.title,
            "status": self.status,
            "claim_status": self.audit.status,
            "estimate_name": self.audit.claim.estimate_name,
            "current_public": tuple(self.audit.claim.public),
            "recommended_public": self.recommended_public,
            "recommended_label": self.recommended_label,
            "observed_value": self.audit.observed_value,
            "lower": self.audit.interval.lower,
            "upper": self.audit.interval.upper,
            "ambiguity": self.audit.ambiguity,
            "ambiguity_limit": self.audit.claim.ambiguity_limit,
            "repair": None if recommended is None else recommended.as_dict(),
            "certificate_status": None
            if self.certificate is None
            else self.certificate.status,
            "frontier_minimal_stable": None
            if self.frontier is None or self.frontier.minimal_stable is None
            else self.frontier.minimal_stable.as_dict(),
            "attribution": None
            if self.attribution is None
            else self.attribution.as_dict(),
            "audit": self.audit.as_dict(),
            "repair_plan": self.repair_plan.as_dict(),
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        recommended = self.recommended_option
        tables: dict[str, tuple[dict[str, Any], ...]] = {
            "summary": (
                {
                    "title": self.title,
                    "status": self.status,
                    "claim_status": self.audit.status,
                    "estimate_name": self.audit.claim.estimate_name,
                    "current_public": tuple(self.audit.claim.public),
                    "recommended_public": self.recommended_public,
                    "recommended_label": self.recommended_label,
                    "observed_value": self.audit.observed_value,
                    "lower": self.audit.interval.lower,
                    "upper": self.audit.interval.upper,
                    "ambiguity": self.audit.ambiguity,
                    "ambiguity_limit": self.audit.claim.ambiguity_limit,
                    "repair_label": None if recommended is None else recommended.label,
                    "certificate_status": None
                    if self.certificate is None
                    else self.certificate.status,
                    "frontier_candidate_count": None
                    if self.frontier is None
                    else len(self.frontier.candidates),
                },
            ),
            "repair_options": tuple(
                option.as_dict() for option in self.repair_plan.options
            ),
            "reasons": tuple({"reason": reason} for reason in self.audit.reasons),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.audit.limitations
            ),
        }
        if self.frontier is not None:
            tables["frontier"] = tuple(
                candidate.as_dict() for candidate in self.frontier.frontier
            )
            tables["frontier_candidates"] = tuple(
                candidate.as_dict() for candidate in self.frontier.candidates
            )
        if self.certificate is not None:
            tables["certificate_reasons"] = tuple(
                {"reason": reason} for reason in self.certificate.reasons
            )
            tables["certificate_limitations"] = tuple(
                {"limitation": limitation}
                for limitation in self.certificate.limitations
            )
        if self.attribution is not None:
            tables["refinement_attributions"] = tuple(
                row.as_dict() for row in self.attribution.attributions
            )
        return tables

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Recommendation",
            "",
            f"- Claim: {self.audit.claim.estimate_name}",
            f"- Design status: `{self.status}`",
            f"- Current public representation: `{_column_label(self.audit.claim.public)}`",
        ]
        if self.recommended_public is None:
            lines.append("- Recommended public representation: none found")
        else:
            lines.append(
                "- Recommended public representation: "
                f"`{_column_label(self.recommended_public)}`"
            )
        lines.extend(
            [
                f"- Current claim status: **{self.audit.status.upper()}**",
                f"- Current ambiguity: {self.audit.ambiguity:.4f}",
            ]
        )
        if self.audit.claim.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.audit.claim.ambiguity_limit:.4f}")
        if self.audit.decision is not None:
            lines.append(
                "- Decision invariant: "
                f"{'yes' if self.audit.decision.invariant else 'no'}"
            )

        lines.extend(["", "## Interpretation", ""])
        lines.append(_design_interpretation(self))

        if self.repair_plan.options:
            lines.extend(
                [
                    "",
                    "## Candidate Repairs",
                    "",
                    "| rank | refinement | cost | certifies | after ambiguity | public cells | reason |",
                    "| ---: | --- | ---: | :---: | ---: | ---: | --- |",
                ]
            )
            for option in self.repair_plan.options[:10]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(option.rank),
                            f"`{option.label}`",
                            f"{option.cost:g}",
                            "yes" if option.certifies_claim else "no",
                            f"{option.after_ambiguity:.4f}",
                            str(option.public_cells),
                            _escape_table(option.reason),
                        ]
                    )
                    + " |"
                )

        if self.frontier is not None and self.frontier.minimal_stable is not None:
            minimal = self.frontier.minimal_stable
            lines.extend(
                [
                    "",
                    "## Minimal Stable Frontier Candidate",
                    "",
                    f"- Added columns: `{_column_label(minimal.added_columns)}`",
                    f"- Public cells: {minimal.public_cells}",
                    f"- Max ambiguity: {minimal.max_ambiguity:.4f}",
                ]
            )

        if self.attribution is not None and self.attribution.attributions:
            lines.extend(["", "## Refinement Attribution", ""])
            lines.extend(
                [
                    "| refinement | Shapley value | share |",
                    "| --- | ---: | ---: |",
                ]
            )
            for row in self.attribution.attributions[:10]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{row.column}`",
                            f"{row.shapley_value:.4f}",
                            f"{row.shapley_share:.1%}",
                        ]
                    )
                    + " |"
                )

        if self.audit.limitations:
            lines.extend(["", "## Limitations", ""])
            lines.extend(f"- {limitation}" for limitation in self.audit.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class ClaimScreeningResult:
    """Optional conservative pre-screen used before an exact claim audit."""

    backend: str
    attempted: bool
    used: bool
    certified: bool
    fallback_required: bool
    exact_solve_avoided: bool
    reason: str
    observed_value: float | None = None
    lower: float | None = None
    upper: float | None = None
    ambiguity: float | None = None
    decision_invariant: bool | None = None
    ambiguity_limit_met: bool | None = None
    q_name: str | None = None
    conservative: bool = True
    compiled_templates_built: int = 0
    support_solves: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "attempted": self.attempted,
            "used": self.used,
            "certified": self.certified,
            "fallback_required": self.fallback_required,
            "exact_solve_avoided": self.exact_solve_avoided,
            "reason": self.reason,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "decision_invariant": self.decision_invariant,
            "ambiguity_limit_met": self.ambiguity_limit_met,
            "q_name": self.q_name,
            "conservative": self.conservative,
            "compiled_templates_built": self.compiled_templates_built,
            "support_solves": self.support_solves,
            "error": self.error,
        }


@dataclass(frozen=True)
class ClaimSpec:
    """Declarative claim that a reported aggregate is stable enough to defend."""

    estimate_name: str
    public: Sequence[str]
    hidden: Sequence[str]
    target: TabularTarget
    weight: str | None = None
    q: Any | None = None
    q_presets: Sequence[Any] = ("saturated",)
    candidate_refinements: Sequence[str] = ()
    ambiguity_limit: float | None = None
    bucket_budget: int | None = None
    decision: DecisionRule | Mapping[str, Any] | None = None
    statistical_interval: tuple[float, float] | None = None
    statistical_uncertainty: StatisticalUncertainty | Mapping[str, Any] | None = None
    min_cell_weight: float = 1.0
    min_cell_weights: Sequence[float] | None = None
    hidden_sets: Sequence[Sequence[str]] | None = None
    top: int = 10
    witness_top: int = 20
    search: str = "exhaustive"
    max_added_columns: int | None = None
    beam_width: int = 12
    max_evaluations: int | None = None
    must_include: Sequence[str] = ()
    must_exclude: Sequence[str] = ()
    enforce_bucket_budget: bool = False
    include_base: bool = True
    exact_required: bool = True
    title: str | None = None
    target_description: str | None = None
    observed_label: str = "Reported estimate"
    screening_backend: str | None = None
    refinement_screening_backend: str | None = None
    refinement_screening_exact_fallback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.estimate_name, str) or not self.estimate_name:
            raise ValueError("estimate_name must be a non-empty string")
        object.__setattr__(self, "public", _string_tuple(self.public, "public"))
        object.__setattr__(self, "hidden", _string_tuple(self.hidden, "hidden"))
        object.__setattr__(
            self,
            "candidate_refinements",
            _string_tuple(self.candidate_refinements, "candidate_refinements"),
        )
        object.__setattr__(
            self,
            "must_include",
            _string_tuple(self.must_include, "must_include"),
        )
        object.__setattr__(
            self,
            "must_exclude",
            _string_tuple(self.must_exclude, "must_exclude"),
        )
        object.__setattr__(
            self,
            "q_presets",
            tuple(QSpec.from_value(value).to_preset() for value in self.q_presets),
        )
        if not self.q_presets:
            raise ValueError("q_presets must contain at least one preset")
        if self.q is not None:
            object.__setattr__(self, "q", _normalize_optional_q(self.q))
        if self.weight is not None and not isinstance(self.weight, str):
            raise TypeError("weight must be a column name or None")
        if self.min_cell_weight < 0:
            raise ValueError("min_cell_weight must be non-negative")
        if self.min_cell_weights is not None:
            object.__setattr__(
                self,
                "min_cell_weights",
                _float_tuple(self.min_cell_weights, "min_cell_weights"),
            )
        if self.hidden_sets is not None:
            object.__setattr__(
                self,
                "hidden_sets",
                tuple(
                    _string_tuple(hidden_set, "hidden_sets")
                    for hidden_set in self.hidden_sets
                ),
            )
        if self.ambiguity_limit is not None and self.ambiguity_limit < 0:
            raise ValueError("ambiguity_limit must be non-negative")
        if self.bucket_budget is not None and self.bucket_budget < 0:
            raise ValueError("bucket_budget must be non-negative")
        if self.decision is not None:
            object.__setattr__(self, "decision", DecisionRule.from_value(self.decision))
        if self.max_added_columns is not None and self.max_added_columns < 0:
            raise ValueError("max_added_columns must be non-negative")
        if self.top < 0:
            raise ValueError("top must be non-negative")
        if self.witness_top < 0:
            raise ValueError("witness_top must be non-negative")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_evaluations is not None and self.max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        if self.statistical_interval is not None:
            if len(self.statistical_interval) != 2:
                raise ValueError("statistical_interval must contain lower and upper")
            lower, upper = (float(value) for value in self.statistical_interval)
            if lower > upper:
                raise ValueError("statistical_interval lower cannot exceed upper")
            object.__setattr__(self, "statistical_interval", (lower, upper))
        if self.screening_backend is not None:
            screening_backend = str(self.screening_backend).lower()
            if screening_backend not in {"residopt", "residopt_l2"}:
                raise ValueError(
                    "screening_backend must be None, 'residopt', or 'residopt_l2'"
                )
            object.__setattr__(self, "screening_backend", screening_backend)
        if self.refinement_screening_backend is not None:
            refinement_screening_backend = str(
                self.refinement_screening_backend
            ).lower()
            if refinement_screening_backend not in {"residopt", "residopt_l2"}:
                raise ValueError(
                    "refinement_screening_backend must be None, 'residopt', "
                    "or 'residopt_l2'"
                )
            object.__setattr__(
                self,
                "refinement_screening_backend",
                refinement_screening_backend,
            )
        object.__setattr__(
            self,
            "statistical_uncertainty",
            _normalize_statistical_uncertainty(
                self.statistical_uncertainty,
                interval=self.statistical_interval,
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimSpec":
        """Build a claim from a JSON-compatible mapping."""

        return cls(**dict(payload))

    @property
    def primary_q(self) -> Any:
        """Q preset used for the primary public-descent report."""

        return self.q if self.q is not None else self.q_presets[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate_name": self.estimate_name,
            "public": list(self.public),
            "hidden": list(self.hidden),
            "target": _target_label(self.target),
            "weight": self.weight,
            "q": _q_payload(self.primary_q),
            "q_presets": [_q_payload(q) for q in self.q_presets],
            "candidate_refinements": list(self.candidate_refinements),
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
            "decision": None if self.decision is None else self.decision.as_dict(),
            "statistical_interval": self.statistical_interval,
            "statistical_uncertainty": None
            if self.statistical_uncertainty is None
            else self.statistical_uncertainty.as_dict(),
            "min_cell_weight": self.min_cell_weight,
            "min_cell_weights": None
            if self.min_cell_weights is None
            else list(self.min_cell_weights),
            "hidden_sets": None
            if self.hidden_sets is None
            else [list(hidden_set) for hidden_set in self.hidden_sets],
            "top": self.top,
            "witness_top": self.witness_top,
            "search": self.search,
            "max_added_columns": self.max_added_columns,
            "beam_width": self.beam_width,
            "max_evaluations": self.max_evaluations,
            "must_include": list(self.must_include),
            "must_exclude": list(self.must_exclude),
            "enforce_bucket_budget": self.enforce_bucket_budget,
            "include_base": self.include_base,
            "exact_required": self.exact_required,
            "title": self.title,
            "target_description": self.target_description,
            "observed_label": self.observed_label,
            "screening_backend": self.screening_backend,
            "refinement_screening_backend": self.refinement_screening_backend,
            "refinement_screening_exact_fallback": (
                self.refinement_screening_exact_fallback
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for as_dict()."""

        return self.as_dict()

    def audit(self, data: Any, **kwargs: Any) -> "ClaimAudit":
        """Audit this claim against tabular data."""

        return audit_claim(data, self, **kwargs)

    def design(self, data: Any, **kwargs: Any) -> "PublicReportDesign":
        """Design the smallest defensible public report for this claim."""

        return design_public_report(self, data, **kwargs)

    def calibrate_tv(
        self,
        data: Any,
        **kwargs: Any,
    ) -> "HistoricalTVCalibrationReport":
        """Calibrate a TV stress radius from historical period transitions."""

        from .calibration import calibrate_tv_radius

        return calibrate_tv_radius(data, self, **kwargs)


@dataclass(frozen=True)
class ClaimAudit(ReportArtifactMixin):
    """Review artifact that certifies, breaks, or repairs a reporting claim."""

    claim: ClaimSpec
    primary: PublicDescentReport
    certificate: RepresentationStabilityCertificate | None = None
    witness: WitnessReport | None = None
    model_assisted: ModelAssistedStabilitySummary | None = None
    decision: DecisionResult | None = None
    decision_repair_candidate: PublicRepresentationCandidate | None = None
    decision_repair_search_exact: bool | None = None
    screening: ClaimScreeningResult | None = None
    status: str = "inconclusive"
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    title: str = "Claim Audit"

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "inconclusive"}:
            raise ValueError("status must be 'pass', 'fail', or 'inconclusive'")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def inconclusive(self) -> bool:
        return self.status == "inconclusive"

    @property
    def repair_candidate(self) -> PublicRepresentationCandidate | None:
        if self.decision_repair_candidate is not None:
            return self.decision_repair_candidate
        if self.certificate is None:
            return None
        return self.certificate.selected_candidate

    @property
    def observed_value(self) -> float:
        """Reported value audited by this claim."""

        return self.primary.observed_value

    @property
    def interval(self):
        """Primary hidden-composition interval."""

        return self.primary.interval

    @property
    def ambiguity(self) -> float:
        """Primary hidden-composition ambiguity width."""

        return self.primary.interval.diameter

    @property
    def refinement_recommendations(
        self,
    ) -> tuple[ClaimRefinementRecommendation, ...]:
        """Rank refinements by their role in stabilizing this claim."""

        return _claim_refinement_recommendations(self)

    def recommend_refinements(
        self,
        *,
        top: int | None = None,
    ) -> tuple[ClaimRefinementRecommendation, ...]:
        """Return claim-centered public-refinement recommendations.

        This is intentionally different from the lower-level
        :func:`updatesupport.recommend_refinements`: it annotates each candidate
        with claim-specific repair signals such as whether it satisfies the
        ambiguity limit or is the selected decision-invariant repair.
        """

        if top is not None and top < 0:
            raise ValueError("top must be non-negative")
        rows = self.refinement_recommendations
        return rows if top is None else rows[:top]

    def repair_plan(
        self,
        *,
        action_costs: Mapping[str, float] | None = None,
        top: int | None = None,
        title: str = "Claim Repair Plan",
    ) -> ClaimRepairPlan:
        """Return a cost-aware repair plan for this claim audit."""

        return _claim_repair_plan(
            self,
            action_costs=action_costs,
            top=top,
            title=title,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "inconclusive": self.inconclusive,
            "claim": self.claim.as_dict(),
            "primary": self.primary.as_dict(),
            "certificate": None
            if self.certificate is None
            else self.certificate.as_dict(),
            "witness": None if self.witness is None else self.witness.as_dict(),
            "model_assisted": None
            if self.model_assisted is None
            else self.model_assisted.as_dict(),
            "decision": None if self.decision is None else self.decision.as_dict(),
            "decision_repair_candidate": None
            if self.decision_repair_candidate is None
            else self.decision_repair_candidate.as_dict(),
            "decision_repair_search_exact": self.decision_repair_search_exact,
            "screening": None if self.screening is None else self.screening.as_dict(),
            "repair_candidate": None
            if self.repair_candidate is None
            else self.repair_candidate.as_dict(),
            "refinement_recommendations": [
                row.as_dict() for row in self.refinement_recommendations
            ],
            "reasons": self.reasons,
            "limitations": self.limitations,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Verdict",
            "",
            f"- Status: **{self.status.upper()}**",
            f"- Claim: {self.claim.estimate_name}",
            f"- Reported estimate: {self.primary.observed_value:.4f}",
            "- Hidden-composition interval: "
            f"[{self.primary.interval.lower:.4f}, {self.primary.interval.upper:.4f}]",
            f"- Hidden-composition ambiguity: {self.primary.interval.diameter:.4f}",
            f"- Public adequate: {'yes' if self.primary.public_adequate else 'no'}",
        ]
        if self.claim.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.claim.ambiguity_limit:.4f}")
        if self.claim.statistical_uncertainty is not None:
            lines.append(
                "- Statistical uncertainty: "
                f"{_format_statistical_uncertainty(self.claim.statistical_uncertainty)}"
            )
        if self.decision is not None:
            lines.extend(
                [
                    f"- Decision rule: {self.decision.rule.name}",
                    f"- Observed decision: {self.decision.observed_decision}",
                    "- Decision invariant: "
                    f"{'yes' if self.decision.invariant else 'no'}",
                ]
            )
            if self.decision.certified_decision is not None:
                lines.append(
                    f"- Certified decision: {self.decision.certified_decision}"
                )
        if self.certificate is not None:
            lines.append(
                f"- Representation certificate: {self.certificate.status.upper()}"
            )
            if self.certificate.frontier.screening is not None:
                screening = self.certificate.frontier.screening
                lines.append(
                    "- Frontier screening: "
                    f"{screening.certified_count}/{screening.endpoint_count} "
                    f"endpoints certified via {screening.backend}"
                )
        if self.model_assisted is not None:
            failure_rate = _format_optional_rate(self.model_assisted.failure_rate)
            lines.append(
                "- Model-assisted joint draws: "
                f"{self.model_assisted.successful_draws}/"
                f"{self.model_assisted.draw_count} successful"
                f"; failure rate {failure_rate}"
            )
        if self.screening is not None:
            lines.append(
                "- Endpoint screening: "
                f"{'used' if self.screening.used else 'fallback'}"
                f" via {self.screening.backend}"
            )

        lines.extend(["", "## Decision Basis", ""])
        lines.extend(f"- {reason}" for reason in self.reasons)

        if self.screening is not None:
            lines.extend(["", "## Endpoint Screening", ""])
            lines.extend(_screening_markdown(self.screening))

        lines.extend(["", "## Claim", ""])
        lines.extend(
            [
                f"- Estimate name: {self.claim.estimate_name}",
                f"- Public columns: `{_column_label(tuple(self.claim.public))}`",
                f"- Hidden columns: `{_column_label(tuple(self.claim.hidden))}`",
                f"- Target: `{_target_label(self.claim.target)}`",
                f"- Primary Q preset: {_q_label(self.claim.primary_q)}",
                "- Stress-test Q presets: "
                f"{', '.join(_q_label(q) for q in self.claim.q_presets)}",
            ]
        )
        if self.claim.decision is not None:
            lines.append(f"- Decision rule: {self.claim.decision.name}")
        if self.claim.candidate_refinements:
            lines.append(
                "- Candidate public refinements: "
                f"`{_column_label(tuple(self.claim.candidate_refinements))}`"
            )

        lines.extend(["", "## Causal Or Statistical Estimate", ""])
        lines.extend(
            [
                "The reported estimate is treated as the target functional "
                "supplied to `updatesupport`. The auditor does not refit the "
                "causal or statistical model; it audits whether the public "
                "representation supports the reported aggregate under hidden "
                "composition shifts.",
                "",
                f"- Reported value: {self.primary.observed_value:.4f}",
            ]
        )

        lines.extend(["", "## Statistical Uncertainty", ""])
        if self.claim.statistical_uncertainty is None:
            lines.append(
                "No statistical uncertainty was supplied. The hidden-composition "
                "ambiguity below is not a confidence interval."
            )
        else:
            lines.append(
                _format_statistical_uncertainty(self.claim.statistical_uncertainty)
            )

        if self.decision is not None:
            lines.extend(["", "## Decision Invariance", ""])
            lines.extend(_decision_markdown(self))

        lines.extend(["", "## Hidden-Composition Ambiguity", ""])
        lines.extend(
            [
                "The ambiguity interval holds the public distribution fixed and "
                "allows hidden composition to vary within the declared Q stress "
                "test.",
                "",
                f"- Lower endpoint: {self.primary.interval.lower:.4f}",
                f"- Upper endpoint: {self.primary.interval.upper:.4f}",
                f"- Ambiguity: {self.primary.interval.diameter:.4f}",
            ]
        )

        if self.certificate is not None:
            lines.extend(["", "## Repair Or Certification", ""])
            lines.extend(_certificate_summary_markdown(self.certificate))
        if self.decision_repair_candidate is not None:
            if self.certificate is None:
                lines.extend(["", "## Repair Or Certification", ""])
            else:
                lines.extend(["", "### Decision-Invariant Repair", ""])
            lines.extend(_decision_repair_markdown(self))

        if self.witness is not None:
            lines.extend(["", "## Counterexample Witness", ""])
            lines.extend(
                [
                    "The witness shows two admissible hidden-composition worlds "
                    "that keep the public report fixed but move the target value.",
                    "",
                    f"- Lower witness value: {self.witness.lower_value:.4f}",
                    f"- Upper witness value: {self.witness.upper_value:.4f}",
                    f"- Witness gap: {self.witness.ambiguity:.4f}",
                    f"- Same public distribution: {'yes' if self.witness.public_law_match else 'no'}",
                ]
            )
            lines.extend(["", "### Largest Hidden-Cell Shifts", ""])
            lines.extend(_witness_shift_table(self.witness))

        if self.model_assisted is not None:
            lines.extend(["", "## Model-Assisted Joint Analysis", ""])
            lines.extend(_model_assisted_markdown(self.model_assisted))

        lines.extend(["", "## Refinement Recommendations", ""])
        lines.extend(_refinement_markdown(self))

        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)

        return "\n".join(lines)


@dataclass(frozen=True)
class ClaimNode:
    """One node in a nested claim tree.

    A node wraps an ordinary :class:`ClaimSpec` and optional child nodes. This
    keeps hierarchical audits as an orchestration/reporting concern: each node
    is still audited by the same single-claim machinery.
    """

    claim: ClaimSpec | Mapping[str, Any]
    children: Sequence["ClaimNode | ClaimSpec | Mapping[str, Any]"] = ()
    name: str | None = None
    role: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim = self.claim
        if not isinstance(claim, ClaimSpec):
            claim = ClaimSpec.from_dict(claim)
        children = tuple(_coerce_claim_node(child) for child in self.children)
        metadata = dict(self.metadata)
        if self.name is not None and not str(self.name):
            raise ValueError("name cannot be empty")
        if self.role is not None and not str(self.role):
            raise ValueError("role cannot be empty")
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimNode":
        """Build a claim node from a JSON-compatible mapping."""

        if "claim" not in payload:
            return cls(claim=ClaimSpec.from_dict(payload))
        return cls(
            claim=payload["claim"],
            children=payload.get("children", ()),
            name=payload.get("name"),
            role=payload.get("role"),
            metadata=payload.get("metadata", {}),
        )

    @property
    def label(self) -> str:
        return self.name or self.claim.estimate_name

    def audit(self, data: Any, **kwargs: Any) -> "ClaimNodeAudit":
        """Audit this node and its descendants against tabular data."""

        return _audit_claim_node(data, self, path=(self.label,), depth=0, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "name": self.name,
            "role": self.role,
            "metadata": dict(self.metadata),
            "claim": self.claim.as_dict(),
            "children": [child.as_dict() for child in self.children],
        }


@dataclass(frozen=True)
class ClaimNodeAudit:
    """Audit result for one claim-tree node and its descendants."""

    node: ClaimNode
    audit: ClaimAudit
    children: Sequence["ClaimNodeAudit"] = ()
    path: Sequence[str] = ()
    depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "path", tuple(str(part) for part in self.path))
        if self.depth < 0:
            raise ValueError("depth must be non-negative")

    @property
    def label(self) -> str:
        return self.node.label

    @property
    def branch(self) -> str:
        return " / ".join(self.path)

    @property
    def status(self) -> str:
        return self.audit.status

    @property
    def ambiguity(self) -> float:
        return self.audit.ambiguity

    @property
    def passed(self) -> bool:
        return self.audit.passed

    @property
    def failed(self) -> bool:
        return self.audit.failed

    @property
    def inconclusive(self) -> bool:
        return self.audit.inconclusive

    def walk(self) -> tuple["ClaimNodeAudit", ...]:
        rows: list[ClaimNodeAudit] = [self]
        for child in self.children:
            rows.extend(child.walk())
        return tuple(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": list(self.path),
            "branch": self.branch,
            "depth": self.depth,
            "role": self.node.role,
            "metadata": dict(self.node.metadata),
            "status": self.status,
            "observed_value": self.audit.observed_value,
            "interval": {
                "lower": self.audit.interval.lower,
                "upper": self.audit.interval.upper,
                "diameter": self.audit.interval.diameter,
            },
            "ambiguity": self.ambiguity,
            "public_adequate": self.audit.primary.public_adequate,
            "decision_invariant": None
            if self.audit.decision is None
            else self.audit.decision.invariant,
            "claim": self.node.claim.as_dict(),
            "audit": self.audit.as_dict(),
            "children": [child.as_dict() for child in self.children],
        }


@dataclass(frozen=True)
class ClaimTree:
    """Nested collection of claims audited as one hierarchy."""

    root: ClaimNode | ClaimSpec | Mapping[str, Any]
    name: str = "Nested Claim Audit"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "root", _coerce_claim_node(self.root))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimTree":
        """Build a claim tree from a JSON-compatible mapping."""

        root = payload.get("root")
        if root is None:
            root = payload.get("claim")
        if root is None:
            raise ValueError("claim tree payload must contain 'root' or 'claim'")
        return cls(
            root=root,
            name=payload.get("name", "Nested Claim Audit"),
            description=payload.get("description"),
        )

    def audit(self, data: Any, **kwargs: Any) -> "ClaimTreeAudit":
        """Audit every node in the claim tree against tabular data."""

        return audit_claim_tree(data, self, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "root": self.root.as_dict(),
        }


@dataclass(frozen=True)
class ClaimTreeAudit(ReportArtifactMixin):
    """Nested claim report for hierarchical or multi-level review workflows."""

    tree: ClaimTree
    root: ClaimNodeAudit
    title: str = "Nested Claim Audit"

    @property
    def nodes(self) -> tuple[ClaimNodeAudit, ...]:
        return self.root.walk()

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def leaf_count(self) -> int:
        return sum(1 for node in self.nodes if not node.children)

    @property
    def pass_count(self) -> int:
        return sum(1 for node in self.nodes if node.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for node in self.nodes if node.failed)

    @property
    def inconclusive_count(self) -> int:
        return sum(1 for node in self.nodes if node.inconclusive)

    @property
    def status(self) -> str:
        if self.fail_count:
            return "fail"
        if self.inconclusive_count:
            return "inconclusive"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def inconclusive(self) -> bool:
        return self.status == "inconclusive"

    @property
    def max_ambiguity(self) -> float:
        if not self.nodes:
            return 0.0
        return max(node.ambiguity for node in self.nodes)

    def worst_nodes(
        self,
        *,
        top: int = 5,
        statuses: Sequence[str] = ("fail", "inconclusive"),
    ) -> tuple[ClaimNodeAudit, ...]:
        """Return the highest-risk claim nodes by status and ambiguity."""

        if top < 0:
            raise ValueError("top must be non-negative")
        allowed = set(statuses)
        selected = [node for node in self.nodes if node.status in allowed]
        if not selected:
            selected = list(self.nodes)
        selected.sort(
            key=lambda node: (
                _claim_tree_status_rank(node.status),
                -node.ambiguity,
                node.branch,
            )
        )
        return tuple(selected[:top])

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "inconclusive": self.inconclusive,
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "inconclusive_count": self.inconclusive_count,
            "max_ambiguity": self.max_ambiguity,
            "tree": self.tree.as_dict(),
            "root": self.root.as_dict(),
            "worst_nodes": [node.as_dict() for node in self.worst_nodes()],
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        summary = (
            {
                "title": self.title,
                "status": self.status,
                "node_count": self.node_count,
                "leaf_count": self.leaf_count,
                "pass_count": self.pass_count,
                "fail_count": self.fail_count,
                "inconclusive_count": self.inconclusive_count,
                "root_status": self.root.status,
                "max_ambiguity": self.max_ambiguity,
            },
        )
        node_rows = tuple(_claim_tree_node_row(node) for node in self.nodes)
        edge_rows = tuple(
            {
                "parent_path": node.branch,
                "child_path": child.branch,
                "parent_label": node.label,
                "child_label": child.label,
            }
            for node in self.nodes
            for child in node.children
        )
        reason_rows = tuple(
            {"path": node.branch, "label": node.label, "reason": reason}
            for node in self.nodes
            for reason in node.audit.reasons
        )
        limitation_rows = tuple(
            {"path": node.branch, "label": node.label, "limitation": limitation}
            for node in self.nodes
            for limitation in node.audit.limitations
        )
        worst_rows = tuple(
            {
                "path": node.branch,
                "label": node.label,
                "status": node.status,
                "ambiguity": node.ambiguity,
                "reason_count": len(node.audit.reasons),
            }
            for node in self.worst_nodes()
        )
        return {
            "summary": summary,
            "nodes": node_rows,
            "edges": edge_rows,
            "reasons": reason_rows,
            "limitations": limitation_rows,
            "worst_nodes": worst_rows,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Hierarchy Verdict",
            "",
            f"- Status: **{self.status.upper()}**",
            f"- Root claim: {self.root.label} ({self.root.status})",
            f"- Claim nodes: {self.node_count}",
            f"- Leaf claims: {self.leaf_count}",
            (
                f"- Node outcomes: {self.pass_count} pass, {self.fail_count} fail, "
                f"{self.inconclusive_count} inconclusive"
            ),
            f"- Maximum hidden-composition ambiguity: {self.max_ambiguity:.4f}",
        ]
        if self.tree.description:
            lines.extend(["", self.tree.description])

        lines.extend(["", "## Claim Tree", ""])
        lines.extend(_claim_tree_markdown_table(self.nodes))

        worst = self.worst_nodes(top=5)
        if worst:
            lines.extend(["", "## Highest-Risk Branches", ""])
            lines.extend(_claim_tree_worst_markdown(worst))

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Each row is an ordinary `ClaimAudit`; the tree only organizes "
                "those audits into a hierarchy. For Bayesian hierarchical "
                "workflows, pass posterior summaries, posterior-draw summaries, "
                "or draw-specific cell targets into the claim data before "
                "auditing. `updatesupport` does not fit or validate the "
                "Bayesian model; it separates supplied posterior/statistical "
                "uncertainty from hidden-composition ambiguity.",
                "",
                "A failed child means that level of the hierarchy is not stable "
                "under its declared public representation and Q stress test, "
                "even if the root aggregate is stable.",
            ]
        )
        return "\n".join(lines)


def audit_claim(
    data: Any,
    claim: ClaimSpec | Mapping[str, Any],
    *,
    joint_model: NonparametricJointDistribution | None = None,
    joint_draws: int = 0,
    joint_seed: int | None = None,
    **overrides: Any,
) -> ClaimAudit:
    """Audit a declared reporting claim against tabular data."""

    if not isinstance(claim, ClaimSpec):
        claim = ClaimSpec.from_dict(claim)
    if overrides:
        claim = replace(claim, **overrides)
    if joint_draws < 0:
        raise ValueError("joint_draws must be non-negative")

    audit_data = data
    row_count: int | None = None
    screening: ClaimScreeningResult | None = None
    if claim.screening_backend is not None:
        if joint_draws:
            screening = ClaimScreeningResult(
                backend=claim.screening_backend,
                attempted=False,
                used=False,
                certified=False,
                fallback_required=True,
                exact_solve_avoided=False,
                reason=(
                    "Endpoint screening is disabled when model-assisted joint "
                    "draws are requested."
                ),
            )
        else:
            if isinstance(data, GroupedProblem):
                audit_data = data
            else:
                audit_data, row_count = _repeatable_data(data)
            screening, screened_primary, screened_decision = _try_claim_screen(
                audit_data,
                claim=claim,
                row_count=row_count,
            )
            if screened_primary is not None:
                status, reasons = _claim_status(
                    claim,
                    primary=screened_primary,
                    certificate=None,
                    decision=screened_decision,
                    decision_repair_candidate=None,
                    decision_repair_search_exact=None,
                )
                return ClaimAudit(
                    claim=claim,
                    primary=screened_primary,
                    certificate=None,
                    witness=None,
                    model_assisted=None,
                    decision=screened_decision,
                    decision_repair_candidate=None,
                    decision_repair_search_exact=None,
                    screening=screening,
                    status=status,
                    reasons=(
                        *reasons,
                        "A conservative residopt endpoint screen certified the "
                        "claim, so the exact CVXPY endpoint solve was avoided.",
                    ),
                    limitations=_claim_limitations(
                        claim,
                        screened_primary,
                        None,
                        screening=screening,
                    ),
                    title=claim.title or "Claim Audit",
                )

    primary = public_descent_report(
        audit_data,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        candidate_refinements=claim.candidate_refinements,
        top=claim.top,
        min_cell_weight=claim.min_cell_weight,
        title=f"{claim.estimate_name} Public-Descent Evidence",
        target_description=claim.target_description or "target value",
        observed_label=claim.observed_label,
        q=claim.primary_q,
        row_count=row_count,
    )
    decision = (
        None
        if claim.decision is None
        else claim.decision.interval_result(
            observed_value=primary.observed_value,
            lower=primary.interval.lower,
            upper=primary.interval.upper,
        )
    )

    certificate: RepresentationStabilityCertificate | None = None
    if claim.ambiguity_limit is not None or claim.candidate_refinements:
        if claim.ambiguity_limit is None:
            certificate_limit = primary.interval.diameter
        else:
            certificate_limit = claim.ambiguity_limit
        certificate = certify_public_representation(
            data,
            base_public=claim.public,
            hidden=claim.hidden,
            target=claim.target,
            weight=claim.weight,
            candidate_refinements=claim.candidate_refinements,
            min_cell_weight=claim.min_cell_weight,
            min_cell_weights=claim.min_cell_weights,
            hidden_sets=claim.hidden_sets,
            q_presets=claim.q_presets,
            ambiguity_limit=certificate_limit,
            bucket_budget=claim.bucket_budget,
            max_added_columns=claim.max_added_columns,
            search=claim.search,
            beam_width=claim.beam_width,
            max_evaluations=claim.max_evaluations,
            must_include=claim.must_include,
            must_exclude=claim.must_exclude,
            enforce_bucket_budget=claim.enforce_bucket_budget,
            include_base=claim.include_base,
            exact_required=claim.exact_required,
            screening_backend=claim.refinement_screening_backend,
            screening_exact_fallback=claim.refinement_screening_exact_fallback,
            title=f"{claim.estimate_name} Representation Certificate",
        )

    decision_repair_candidate = None
    decision_repair_search_exact = None
    if decision is not None and not decision.invariant and claim.candidate_refinements:
        decision_repair_candidate, decision_repair_search_exact = (
            _decision_repair_candidate(
                data,
                claim=claim,
                expected_decision=decision.observed_decision,
            )
        )

    status, reasons = _claim_status(
        claim,
        primary=primary,
        certificate=certificate,
        decision=decision,
        decision_repair_candidate=decision_repair_candidate,
        decision_repair_search_exact=decision_repair_search_exact,
    )
    witness = None
    if status == "fail" or not primary.public_adequate:
        witness = primary.witness_report(
            title=f"{claim.estimate_name} Counterexample Witness",
            top=claim.witness_top,
        )
    model_assisted = None
    if joint_draws:
        model_assisted = _model_assisted_summary(
            data,
            claim=claim,
            joint_model=joint_model,
            draw_count=joint_draws,
            seed=joint_seed,
        )

    return ClaimAudit(
        claim=claim,
        primary=primary,
        certificate=certificate,
        witness=witness,
        model_assisted=model_assisted,
        decision=decision,
        decision_repair_candidate=decision_repair_candidate,
        decision_repair_search_exact=decision_repair_search_exact,
        screening=screening,
        status=status,
        reasons=reasons,
        limitations=_claim_limitations(
            claim,
            primary,
            certificate,
            screening=screening,
        ),
        title=claim.title or "Claim Audit",
    )


def plan_claim_repair(
    claim_or_audit: ClaimSpec | ClaimAudit | Mapping[str, Any],
    data: Any | None = None,
    *,
    action_costs: Mapping[str, float] | None = None,
    top: int | None = None,
    title: str = "Claim Repair Plan",
    **audit_overrides: Any,
) -> ClaimRepairPlan:
    """Build a cost-aware repair plan from a claim audit or claim spec.

    This is a consolidation layer over existing claim evidence. If a
    :class:`ClaimAudit` is supplied, no additional solve is run. If a
    :class:`ClaimSpec` or mapping is supplied, ``data`` is audited first and the
    repair plan is built from that audit.
    """

    if isinstance(claim_or_audit, ClaimAudit):
        if data is not None:
            raise ValueError("data must be None when planning from a ClaimAudit")
        if audit_overrides:
            raise ValueError(
                "audit overrides cannot be used when planning from a ClaimAudit"
            )
        audit = claim_or_audit
    else:
        if data is None:
            raise ValueError("data is required when planning from a claim spec")
        audit = audit_claim(data, claim_or_audit, **audit_overrides)
    return audit.repair_plan(
        action_costs=action_costs,
        top=top,
        title=title,
    )


def design_public_report(
    claim_or_audit: ClaimSpec | ClaimAudit | Mapping[str, Any],
    data: Any | None = None,
    *,
    action_costs: Mapping[str, float] | None = None,
    top: int | None = None,
    include_attribution: bool = False,
    attribution_max_exact_columns: int = 8,
    attribution_permutations: int | None = None,
    attribution_seed: int | None = None,
    title: str = "Public Report Design",
    **audit_overrides: Any,
) -> PublicReportDesign:
    """Design a defensible public representation for a declared claim.

    This is the claim-first orchestration layer. It audits the claim, reuses the
    embedded certificate/frontier search, packages a cost-aware repair plan, and
    optionally adds Shapley-style refinement attribution.
    """

    if isinstance(claim_or_audit, ClaimAudit):
        if audit_overrides:
            raise ValueError(
                "audit overrides cannot be used when designing from a ClaimAudit"
            )
        audit = claim_or_audit
    else:
        if data is None:
            raise ValueError("data is required when designing from a claim spec")
        audit = audit_claim(data, claim_or_audit, **audit_overrides)

    repair = audit.repair_plan(
        action_costs=action_costs,
        top=top,
        title=f"{audit.claim.estimate_name} Repair Plan",
    )
    certificate = audit.certificate
    frontier = None if certificate is None else certificate.frontier
    attribution = None
    if include_attribution:
        if data is None:
            raise ValueError("data is required when include_attribution=True")
        if audit.claim.candidate_refinements:
            attribution = attribute_refinement_ambiguity(
                data,
                public=audit.claim.public,
                hidden=audit.claim.hidden,
                target=audit.claim.target,
                weight=audit.claim.weight,
                candidate_refinements=audit.claim.candidate_refinements,
                min_cell_weight=audit.claim.min_cell_weight,
                q=audit.claim.primary_q,
                max_exact_columns=attribution_max_exact_columns,
                n_permutations=attribution_permutations,
                seed=attribution_seed,
                title=f"{audit.claim.estimate_name} Refinement Attribution",
            )

    return PublicReportDesign(
        audit=audit,
        repair_plan=repair,
        certificate=certificate,
        frontier=frontier,
        attribution=attribution,
        title=title,
    )


def claim(
    estimate_name: str,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    **kwargs: Any,
) -> ClaimSpec:
    """Create a claim spec using the simplified claim-first API."""

    return ClaimSpec(
        estimate_name=estimate_name,
        public=public,
        hidden=hidden,
        target=target,
        **kwargs,
    )


def claim_tree(
    root: ClaimNode | ClaimSpec | Mapping[str, Any],
    *,
    children: Sequence[ClaimNode | ClaimSpec | Mapping[str, Any]] = (),
    name: str = "Nested Claim Audit",
    description: str | None = None,
) -> ClaimTree:
    """Create a nested claim tree from a root claim and optional children."""

    root_node = _coerce_claim_node(root)
    if children:
        root_node = ClaimNode(
            claim=root_node.claim,
            children=(*root_node.children, *children),
            name=root_node.name,
            role=root_node.role,
            metadata=root_node.metadata,
        )
    return ClaimTree(root=root_node, name=name, description=description)


def audit_claim_tree(
    data: Any,
    tree: ClaimTree | ClaimNode | ClaimSpec | Mapping[str, Any],
    **kwargs: Any,
) -> ClaimTreeAudit:
    """Audit a nested claim tree against tabular data."""

    if not isinstance(tree, ClaimTree):
        if isinstance(tree, Mapping) and ("root" in tree or "claim" in tree):
            tree = ClaimTree.from_dict(tree)
        else:
            tree = ClaimTree(root=tree)
    root_audit = _audit_claim_node(
        data,
        tree.root,
        path=(tree.root.label,),
        depth=0,
        **kwargs,
    )
    return ClaimTreeAudit(tree=tree, root=root_audit, title=tree.name)


def _coerce_claim_node(value: ClaimNode | ClaimSpec | Mapping[str, Any]) -> ClaimNode:
    if isinstance(value, ClaimNode):
        return value
    if isinstance(value, ClaimSpec):
        return ClaimNode(claim=value)
    if isinstance(value, Mapping):
        return ClaimNode.from_dict(value)
    raise TypeError("claim tree nodes must be ClaimNode, ClaimSpec, or mapping")


def _audit_claim_node(
    data: Any,
    node: ClaimNode,
    *,
    path: tuple[str, ...],
    depth: int,
    **kwargs: Any,
) -> ClaimNodeAudit:
    audit = audit_claim(data, node.claim, **kwargs)
    child_audits = tuple(
        _audit_claim_node(
            data,
            child,
            path=path + (child.label,),
            depth=depth + 1,
            **kwargs,
        )
        for child in node.children
    )
    return ClaimNodeAudit(
        node=node,
        audit=audit,
        children=child_audits,
        path=path,
        depth=depth,
    )


def _claim_tree_node_row(node: ClaimNodeAudit) -> dict[str, Any]:
    decision = node.audit.decision
    return {
        "path": node.branch,
        "parent_path": " / ".join(node.path[:-1]),
        "depth": node.depth,
        "label": node.label,
        "role": node.node.role,
        "estimate_name": node.node.claim.estimate_name,
        "status": node.status,
        "observed_value": node.audit.observed_value,
        "lower": node.audit.interval.lower,
        "upper": node.audit.interval.upper,
        "ambiguity": node.ambiguity,
        "ambiguity_limit": node.node.claim.ambiguity_limit,
        "public_adequate": node.audit.primary.public_adequate,
        "decision_invariant": None if decision is None else decision.invariant,
        "certified_decision": None if decision is None else decision.certified_decision,
        "public_columns": node.node.claim.public,
        "hidden_columns": node.node.claim.hidden,
        "child_count": len(node.children),
        "reason_count": len(node.audit.reasons),
        "limitation_count": len(node.audit.limitations),
    }


def _claim_tree_status_rank(status: str) -> int:
    if status == "fail":
        return 0
    if status == "inconclusive":
        return 1
    if status == "pass":
        return 2
    return 99


def _claim_tree_markdown_table(nodes: Sequence[ClaimNodeAudit]) -> list[str]:
    lines = [
        "| Branch | Status | Ambiguity | Public adequate | Decision invariant |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for node in nodes:
        decision_invariant = (
            ""
            if node.audit.decision is None
            else ("yes" if node.audit.decision.invariant else "no")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(node.branch),
                    node.status,
                    f"{node.ambiguity:.4f}",
                    "yes" if node.audit.primary.public_adequate else "no",
                    decision_invariant,
                ]
            )
            + " |"
        )
    return lines


def _claim_tree_worst_markdown(nodes: Sequence[ClaimNodeAudit]) -> list[str]:
    lines = [
        "| Branch | Status | Ambiguity | First reason |",
        "| --- | ---: | ---: | --- |",
    ]
    for node in nodes:
        reason = node.audit.reasons[0] if node.audit.reasons else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table(node.branch),
                    node.status,
                    f"{node.ambiguity:.4f}",
                    _escape_table(reason),
                ]
            )
            + " |"
        )
    return lines


def _claim_status(
    claim: ClaimSpec,
    *,
    primary: PublicDescentReport,
    certificate: RepresentationStabilityCertificate | None,
    decision: DecisionResult | None,
    decision_repair_candidate: PublicRepresentationCandidate | None,
    decision_repair_search_exact: bool | None,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if decision is not None:
        if decision.invariant:
            reasons.append(
                "The decision rule is invariant over the primary "
                "hidden-composition interval."
            )
        else:
            reasons.append(
                "The decision rule is not invariant over the primary "
                "hidden-composition interval."
            )
            if decision_repair_candidate is None:
                if claim.exact_required and decision_repair_search_exact is False:
                    reasons.append(
                        "Decision-invariant repair search was heuristic, so the "
                        "absence of a repair is inconclusive under the requested "
                        "exactness standard."
                    )
                    return "inconclusive", tuple(reasons)
                reasons.append(
                    "No decision-invariant repair representation was found within "
                    "the declared refinement and bucket constraints."
                )
                return "fail", tuple(reasons)
            if claim.exact_required and decision_repair_search_exact is False:
                reasons.append(
                    "A decision-invariant repair representation was found, but "
                    "the repair search was inconclusive under the requested "
                    "exactness standard."
                )
                return "inconclusive", tuple(reasons)
            reasons.append(
                "A decision-invariant repair representation makes the observed "
                "decision invariant under the declared stress tests."
            )
            return "fail", tuple(reasons)

    if claim.ambiguity_limit is not None:
        if primary.interval.diameter <= claim.ambiguity_limit:
            reasons.append(
                "The current public representation satisfies the declared "
                "hidden-composition ambiguity limit."
            )
        else:
            reasons.append(
                "The current public representation exceeds the declared "
                "hidden-composition ambiguity limit."
            )
            if certificate is None or certificate.failed:
                reasons.append(
                    "No certified repair representation was found within the "
                    "declared refinement and bucket constraints."
                )
                return "fail", tuple(reasons)
            if certificate.inconclusive:
                reasons.append(
                    "A candidate repair was found, but the representation search "
                    "was inconclusive under the requested exactness standard."
                )
                return "inconclusive", tuple(reasons)
            reasons.append(
                "A repair representation satisfies the ambiguity limit under "
                "the declared certificate search."
            )
            return "fail", tuple(reasons)
    else:
        if decision is None:
            reasons.append(
                "No ambiguity limit or decision rule was supplied, so the "
                "auditor reports evidence but cannot issue a pass/fail "
                "stability verdict."
            )
            return "inconclusive", tuple(reasons)
        reasons.append(
            "No ambiguity limit was supplied; the pass/fail verdict is based on "
            "decision invariance."
        )

    if certificate is not None and claim.ambiguity_limit is not None:
        if certificate.failed:
            reasons.append(
                "The representation certificate did not find any candidate "
                "satisfying the declared certificate requirements."
            )
            return "fail", tuple(reasons)
        if certificate.inconclusive:
            reasons.append(
                "The representation certificate found only provisional evidence."
            )
            return "inconclusive", tuple(reasons)
        reasons.append("The representation certificate passed.")

    if not primary.public_adequate:
        reasons.append(
            "The public representation is not perfectly adequate, but its "
            "ambiguity is within the declared tolerance."
        )
    else:
        reasons.append("The public representation is adequate under the primary Q.")
    return "pass", tuple(reasons)


def _claim_limitations(
    claim: ClaimSpec,
    primary: PublicDescentReport,
    certificate: RepresentationStabilityCertificate | None,
    *,
    screening: ClaimScreeningResult | None = None,
) -> tuple[str, ...]:
    limitations = [
        "This verifies reporting stability, not whether the upstream causal or "
        "statistical estimator is correctly specified.",
        "Hidden-composition ambiguity is not statistical uncertainty and does "
        "not include sampling error unless supplied separately.",
        "The result is conditional on the retained support, target definition, "
        "public columns, hidden columns, and declared Q stress tests.",
    ]
    if claim.statistical_uncertainty is None:
        limitations.append(
            "No standard error, confidence interval, or survey-design "
            "uncertainty was supplied."
        )
    if not primary.interval.dual_summary(top=1):
        limitations.append(
            "Dual diagnostics are unavailable for this primary solve or backend."
        )
    if (
        certificate is not None
        and not certificate.search_exact
        and (certificate.frontier.screening is None or certificate.inconclusive)
    ):
        limitations.append(
            "The repair/certificate search was not exact over the full declared "
            "candidate space."
        )
    if certificate is not None and certificate.frontier.screening is not None:
        frontier_screening = certificate.frontier.screening
        limitations.append(
            "The repair/certificate frontier used experimental residopt "
            "screening. Conservative endpoints certify upper bounds for "
            f"{frontier_screening.certified_count} of "
            f"{frontier_screening.endpoint_count} "
            "evaluated scenario endpoints; inconclusive endpoints use exact "
            "fallback when enabled."
        )
    limitations.append(
        "Model-assisted joint analysis, when supplied, is conditional on the "
        "fitted joint-cell model and should be read separately from adversarial "
        "Q-based hidden-composition ambiguity."
    )
    if claim.decision is not None:
        limitations.append(
            "Decision invariance verifies the supplied threshold rule only; it "
            "does not validate whether the threshold itself is appropriate."
        )
    if screening is not None:
        if screening.used:
            limitations.append(
                "The primary interval was certified by an experimental residopt "
                "screening backend. It is conservative for the original Q set, "
                "so a pass is sound but the reported ambiguity may be wider than "
                "the exact simplex-constrained endpoint."
            )
        else:
            limitations.append(
                "Experimental residopt endpoint screening was attempted but did "
                "not certify the claim, so the audit fell back to the exact "
                "primary solver."
            )
    return tuple(limitations)


def _try_claim_screen(
    data: Any,
    *,
    claim: ClaimSpec,
    row_count: int | None,
) -> tuple[ClaimScreeningResult, PublicDescentReport | None, DecisionResult | None]:
    if claim.screening_backend not in {"residopt", "residopt_l2"}:
        raise ValueError(f"unsupported screening backend: {claim.screening_backend!r}")
    if claim.decision is None and claim.ambiguity_limit is None:
        return (
            ClaimScreeningResult(
                backend=claim.screening_backend,
                attempted=False,
                used=False,
                certified=False,
                fallback_required=True,
                exact_solve_avoided=False,
                reason=(
                    "Endpoint screening requires a decision rule or ambiguity "
                    "limit to certify."
                ),
            ),
            None,
            None,
        )
    q = claim.primary_q
    if not isinstance(q, QPreset) or q.name != "l2_budget":
        return (
            ClaimScreeningResult(
                backend=claim.screening_backend,
                attempted=False,
                used=False,
                certified=False,
                fallback_required=True,
                exact_solve_avoided=False,
                reason="Residopt screening currently supports q_l2_budget only.",
            ),
            None,
            None,
        )

    try:
        from .residopt_backend import ResidOptL2EndpointCompiler

        grouped = (
            data
            if isinstance(data, GroupedProblem)
            else from_dataframe(
                data,
                public=claim.public,
                hidden=claim.hidden,
                target=claim.target,
                weight=claim.weight,
                min_cell_weight=claim.min_cell_weight,
                q=q,
            )
        )
        compiler = ResidOptL2EndpointCompiler.from_grouped(grouped)
        screen_report = compiler.interval(
            title=f"{claim.estimate_name} ResidOpt Endpoint Screen"
        )
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        return (
            ClaimScreeningResult(
                backend=claim.screening_backend,
                attempted=True,
                used=False,
                certified=False,
                fallback_required=True,
                exact_solve_avoided=False,
                reason="Residopt screening could not certify this claim.",
                error=str(exc),
            ),
            None,
            None,
        )

    decision = (
        None
        if claim.decision is None
        else claim.decision.interval_result(
            observed_value=screen_report.observed_value,
            lower=screen_report.lower,
            upper=screen_report.upper,
        )
    )
    decision_certified = decision is None or decision.invariant
    ambiguity_limit_met = (
        None
        if claim.ambiguity_limit is None
        else screen_report.ambiguity <= claim.ambiguity_limit
    )
    ambiguity_certified = ambiguity_limit_met is not False
    certified = decision_certified and ambiguity_certified

    if not certified:
        reason_parts = ["Conservative residopt screening was inconclusive."]
        if decision is not None and not decision.invariant:
            reason_parts.append("The conservative interval crosses the decision rule.")
        if ambiguity_limit_met is False:
            reason_parts.append(
                "The conservative interval exceeds the ambiguity limit."
            )
        return (
            ClaimScreeningResult(
                backend=claim.screening_backend,
                attempted=True,
                used=False,
                certified=False,
                fallback_required=True,
                exact_solve_avoided=False,
                reason=" ".join(reason_parts),
                observed_value=screen_report.observed_value,
                lower=screen_report.lower,
                upper=screen_report.upper,
                ambiguity=screen_report.ambiguity,
                decision_invariant=None if decision is None else decision.invariant,
                ambiguity_limit_met=ambiguity_limit_met,
                q_name=screen_report.q_name,
                conservative=screen_report.conservative_for_updatesupport_q,
                compiled_templates_built=screen_report.compiled_templates_built,
                support_solves=screen_report.support_solves,
            ),
            None,
            decision,
        )

    primary = _public_descent_report_from_residopt_screen(
        grouped,
        screen_report=screen_report,
        claim=claim,
        row_count=row_count,
    )
    return (
        ClaimScreeningResult(
            backend=claim.screening_backend,
            attempted=True,
            used=True,
            certified=True,
            fallback_required=False,
            exact_solve_avoided=True,
            reason=(
                "A conservative residopt endpoint interval certified the claim "
                "without an exact endpoint solve."
            ),
            observed_value=screen_report.observed_value,
            lower=screen_report.lower,
            upper=screen_report.upper,
            ambiguity=screen_report.ambiguity,
            decision_invariant=None if decision is None else decision.invariant,
            ambiguity_limit_met=ambiguity_limit_met,
            q_name=screen_report.q_name,
            conservative=screen_report.conservative_for_updatesupport_q,
            compiled_templates_built=screen_report.compiled_templates_built,
            support_solves=screen_report.support_solves,
        ),
        primary,
        decision,
    )


def _public_descent_report_from_residopt_screen(
    grouped: GroupedProblem,
    *,
    screen_report: Any,
    claim: ClaimSpec,
    row_count: int | None,
) -> PublicDescentReport:
    interval = TransportResult(
        lower=screen_report.lower,
        upper=screen_report.upper,
        diameter=screen_report.ambiguity,
        public_law=grouped.public_law,
        bound_type="conservative",
        lower_bound_type="conservative",
        upper_bound_type="conservative",
        notes=tuple(screen_report.notes),
    )
    diagnostics = (
        tuple(grouped.diagnostics.diagnostics)
        if grouped.diagnostics is not None
        else ()
    )
    return PublicDescentReport(
        grouped=grouped,
        observed_value=screen_report.observed_value,
        interval=interval,
        public_adequate=screen_report.ambiguity <= grouped.problem.tol,
        fibers=public_fiber_diagnostics(grouped, top=claim.top),
        refinements=(),
        title=f"{claim.estimate_name} Public-Descent Evidence",
        target_description=claim.target_description or "target value",
        observed_label=claim.observed_label,
        row_count=row_count,
        row_count_label="Rows",
        min_cell_weight=claim.min_cell_weight,
        diagnostics=diagnostics,
        estimator_uncertainty=None,
    )


def _decision_repair_candidate(
    data: Any,
    *,
    claim: ClaimSpec,
    expected_decision: str,
) -> tuple[PublicRepresentationCandidate | None, bool | None]:
    search = claim.search
    if search in {"mip", "mip_oracle", "mip_minimum", "mip_exact"}:
        search = "exhaustive"
    frontier = public_representation_frontier(
        data,
        base_public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        candidate_refinements=claim.candidate_refinements,
        min_cell_weight=claim.min_cell_weight,
        min_cell_weights=claim.min_cell_weights,
        hidden_sets=claim.hidden_sets,
        q_presets=claim.q_presets,
        ambiguity_limit=None,
        bucket_budget=claim.bucket_budget,
        max_added_columns=claim.max_added_columns,
        search=search,
        beam_width=claim.beam_width,
        max_evaluations=claim.max_evaluations,
        must_include=claim.must_include,
        must_exclude=claim.must_exclude,
        enforce_bucket_budget=claim.enforce_bucket_budget,
        include_base=claim.include_base,
        screening_backend=claim.refinement_screening_backend,
        screening_exact_fallback=claim.refinement_screening_exact_fallback,
        title=f"{claim.estimate_name} Decision-Invariant Repair Search",
    )
    candidates = [
        candidate
        for candidate in frontier.candidates
        if _candidate_decision_invariant(
            candidate,
            decision=claim.decision,
            expected_decision=expected_decision,
        )
    ]
    exact = None if frontier.search_trace is None else frontier.search_trace.exact
    if not candidates:
        return None, exact
    return min(
        candidates,
        key=lambda candidate: (
            candidate.public_cells,
            candidate.added_column_count,
            candidate.max_ambiguity,
            candidate.added_columns,
        ),
    ), exact


def _candidate_decision_invariant(
    candidate: PublicRepresentationCandidate,
    *,
    decision: DecisionRule | None,
    expected_decision: str,
) -> bool:
    if decision is None:
        return False
    for scenario in candidate.scenarios:
        result = decision.interval_result(
            observed_value=scenario.observed_value,
            lower=scenario.lower,
            upper=scenario.upper,
        )
        if not result.invariant or result.certified_decision != expected_decision:
            return False
    return True


def _model_assisted_summary(
    data: Any,
    *,
    claim: ClaimSpec,
    joint_model: NonparametricJointDistribution | None,
    draw_count: int,
    seed: int | None,
) -> ModelAssistedStabilitySummary:
    uncertainty = hidden_composition_uncertainty(
        data,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        joint_model=joint_model,
        draws=draw_count,
        seed=seed,
        min_cell_weight=claim.min_cell_weight,
        q=claim.primary_q,
        ambiguity_limit=claim.ambiguity_limit,
        title=f"{claim.estimate_name} Model-Assisted Joint Uncertainty",
    )
    rows = tuple(
        _model_assisted_draw_result(row, claim=claim) for row in uncertainty.rows
    )
    return ModelAssistedStabilitySummary(
        joint_model=uncertainty.joint_model,
        rows=rows,
        ambiguity_limit=claim.ambiguity_limit,
        seed=seed,
        uncertainty_report=uncertainty,
    )


def _model_assisted_draw_result(
    row,
    *,
    claim: ClaimSpec,
) -> ModelAssistedDrawResult:
    status = row.status
    if (
        claim.decision is not None
        and row.error is None
        and row.observed_value is not None
        and row.lower is not None
        and row.upper is not None
    ):
        decision = claim.decision.interval_result(
            observed_value=row.observed_value,
            lower=row.lower,
            upper=row.upper,
        )
        status = "pass" if decision.invariant else "fail"
    return ModelAssistedDrawResult(
        draw_index=row.draw_index,
        observed_value=row.observed_value,
        lower=row.lower,
        upper=row.upper,
        ambiguity=row.ambiguity,
        public_adequate=row.public_adequate,
        status=status,
        error=row.error,
    )


def _normalize_statistical_uncertainty(
    value: StatisticalUncertainty | Mapping[str, Any] | None,
    *,
    interval: tuple[float, float] | None,
) -> StatisticalUncertainty | None:
    if isinstance(value, StatisticalUncertainty):
        return value
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError(
                "statistical_uncertainty must be StatisticalUncertainty, mapping, or None"
            )
        return StatisticalUncertainty(**dict(value))
    if interval is None:
        return None
    lower, upper = interval
    return StatisticalUncertainty(lower=lower, upper=upper)


def _normalize_optional_q(value: Any) -> Any:
    try:
        return QSpec.from_value(value).to_preset()
    except (TypeError, ValueError):
        return value


def _decision_markdown(report: ClaimAudit) -> list[str]:
    if report.decision is None:
        return []
    decision = report.decision
    lines = [
        f"- Rule: {decision.rule.name}",
        f"- Observed value: {decision.observed_value:.4f}",
        f"- Observed decision: {decision.observed_decision}",
        "- Hidden-composition decision interval: "
        f"[{decision.lower:.4f}, {decision.upper:.4f}]",
        f"- Lower-endpoint decision: {decision.lower_decision}",
        f"- Upper-endpoint decision: {decision.upper_decision}",
        f"- Decision invariant: {'yes' if decision.invariant else 'no'}",
        f"- Reason: {decision.reason}",
    ]
    if decision.certified_decision is not None:
        lines.append(f"- Certified decision: {decision.certified_decision}")
    if report.decision_repair_candidate is not None:
        candidate = report.decision_repair_candidate
        lines.append(
            "- Decision-invariant repair candidate: "
            f"`{candidate.label}` with {candidate.public_cells} public cells"
        )
    return lines


def _screening_markdown(screening: ClaimScreeningResult) -> list[str]:
    lines = [
        f"- Backend: {screening.backend}",
        f"- Attempted: {'yes' if screening.attempted else 'no'}",
        f"- Used: {'yes' if screening.used else 'no'}",
        f"- Exact solve avoided: {'yes' if screening.exact_solve_avoided else 'no'}",
        f"- Reason: {screening.reason}",
    ]
    if screening.lower is not None and screening.upper is not None:
        lines.extend(
            [
                "- Conservative screening interval: "
                f"[{screening.lower:.4f}, {screening.upper:.4f}]",
                f"- Conservative ambiguity: {screening.ambiguity:.4f}",
            ]
        )
    if screening.decision_invariant is not None:
        lines.append(
            "- Screening decision invariant: "
            f"{'yes' if screening.decision_invariant else 'no'}"
        )
    if screening.ambiguity_limit_met is not None:
        lines.append(
            "- Screening ambiguity limit met: "
            f"{'yes' if screening.ambiguity_limit_met else 'no'}"
        )
    if screening.compiled_templates_built or screening.support_solves:
        lines.extend(
            [
                f"- Compiled templates built: {screening.compiled_templates_built}",
                f"- Support solves: {screening.support_solves}",
            ]
        )
    if screening.error:
        lines.append(f"- Error: `{screening.error}`")
    return lines


def _decision_repair_markdown(report: ClaimAudit) -> list[str]:
    candidate = report.decision_repair_candidate
    if candidate is None:
        return []
    lines = [
        "The decision-specific repair promotes enough hidden information into "
        "the public representation to keep the observed decision fixed under "
        "all evaluated stress-test scenarios.",
        "",
        f"- Selected repair: `{candidate.label}`",
        f"- Added columns: `{_column_label(candidate.added_columns)}`",
        f"- Public cells: {candidate.public_cells}",
        f"- Max ambiguity after repair: {candidate.max_ambiguity:.4f}",
    ]
    if report.decision_repair_search_exact is not None:
        lines.append(
            "- Repair search exact: "
            f"{'yes' if report.decision_repair_search_exact else 'no'}"
        )
    return lines


def _claim_repair_plan(
    audit: ClaimAudit,
    *,
    action_costs: Mapping[str, float] | None,
    top: int | None,
    title: str,
) -> ClaimRepairPlan:
    if top is not None and top < 0:
        raise ValueError("top must be non-negative")
    costs = _normalize_action_costs(action_costs)
    options = [
        _repair_option_from_recommendation(
            row,
            audit=audit,
            action_costs=costs,
        )
        for row in audit.refinement_recommendations
    ]
    options.sort(key=_repair_option_sort_key)
    ranked = tuple(
        replace(option, rank=index) for index, option in enumerate(options, start=1)
    )
    if top is not None:
        ranked = ranked[:top]
    return ClaimRepairPlan(
        audit=audit,
        options=ranked,
        action_costs=costs,
        title=title,
    )


def _repair_option_from_recommendation(
    row: ClaimRefinementRecommendation,
    *,
    audit: ClaimAudit,
    action_costs: Mapping[str, float],
) -> ClaimRepairOption:
    certifies = _recommendation_certifies_claim(row, audit)
    return ClaimRepairOption(
        rank=0,
        columns=row.columns,
        source=row.source,
        cost=sum(action_costs.get(column, 1.0) for column in row.columns),
        before_ambiguity=row.before_ambiguity,
        after_ambiguity=row.after_ambiguity,
        reduction=row.reduction,
        reduction_percent=row.reduction_percent,
        public_cells=row.public_cells,
        satisfies_ambiguity_limit=row.meets_ambiguity_limit,
        certifies_claim=certifies,
        selected_repair=row.selected_repair,
        decision_repair=row.decision_repair,
        reason=row.reason,
    )


def _recommendation_certifies_claim(
    row: ClaimRefinementRecommendation,
    audit: ClaimAudit,
) -> bool:
    if row.selected_repair or row.decision_repair:
        return True
    if audit.decision is not None:
        return False
    return row.meets_ambiguity_limit is True


def _repair_option_sort_key(option: ClaimRepairOption) -> tuple[Any, ...]:
    return (
        0 if option.certifies_claim else 1,
        option.cost,
        option.public_cells,
        option.after_ambiguity,
        len(option.columns),
        option.label,
    )


def _normalize_action_costs(
    action_costs: Mapping[str, float] | None,
) -> dict[str, float]:
    if action_costs is None:
        return {}
    normalized: dict[str, float] = {}
    for column, cost in action_costs.items():
        if not isinstance(column, str) or not column:
            raise ValueError("action cost keys must be non-empty column names")
        numeric_cost = float(cost)
        if numeric_cost < 0 or not isfinite(numeric_cost):
            raise ValueError("action costs must be finite non-negative numbers")
        normalized[column] = numeric_cost
    return normalized


def _claim_repair_plan_tables(
    plan: ClaimRepairPlan,
) -> dict[str, tuple[dict[str, Any], ...]]:
    recommended = plan.recommended
    return {
        "summary": (
            {
                "title": plan.title,
                "status": plan.status,
                "claim_status": plan.audit.status,
                "estimate_name": plan.audit.claim.estimate_name,
                "observed_value": plan.audit.observed_value,
                "lower": plan.audit.interval.lower,
                "upper": plan.audit.interval.upper,
                "ambiguity": plan.audit.ambiguity,
                "ambiguity_limit": plan.audit.claim.ambiguity_limit,
                "has_decision": plan.audit.decision is not None,
                "decision_invariant": None
                if plan.audit.decision is None
                else plan.audit.decision.invariant,
                "recommended_label": None if recommended is None else recommended.label,
                "recommended_cost": None if recommended is None else recommended.cost,
                "recommended_after_ambiguity": None
                if recommended is None
                else recommended.after_ambiguity,
                "recommended_public_cells": None
                if recommended is None
                else recommended.public_cells,
                "option_count": len(plan.options),
                "certifying_option_count": len(plan.certifying_options),
            },
        ),
        "recommended": () if recommended is None else (recommended.as_dict(),),
        "options": tuple(option.as_dict() for option in plan.options),
        "certifying_options": tuple(
            option.as_dict() for option in plan.certifying_options
        ),
        "non_certifying_options": tuple(
            option.as_dict() for option in plan.non_certifying_options
        ),
        "action_costs": tuple(
            {"column": column, "cost": cost}
            for column, cost in sorted(plan.action_costs.items())
        ),
    }


def _repair_plan_interpretation(plan: ClaimRepairPlan) -> str:
    recommended = plan.recommended
    if plan.audit.passed:
        return (
            "The claim is already certified under the declared public "
            "representation and stress tests, so no public-representation "
            "repair is required. The options below remain diagnostic "
            "refinements, not required actions."
        )
    if recommended is None:
        if not plan.options:
            return (
                "No candidate refinements were supplied, so the planner cannot "
                "propose a public-representation repair."
            )
        return (
            "The audit found candidate refinements, but none is certified by the "
            "existing claim evidence. Treat the options as diagnostics or widen "
            "the repair search before relying on the claim."
        )
    return (
        f"The cheapest certifying repair under the supplied action costs is "
        f"`{recommended.label}`. It lowers hidden-composition ambiguity from "
        f"{plan.audit.ambiguity:.4f} to {recommended.after_ambiguity:.4f} "
        "while satisfying the claim's declared repair criterion."
    )


def _design_interpretation(design: PublicReportDesign) -> str:
    if design.audit.passed:
        return (
            "The current public representation already supports the declared "
            "claim under the chosen stress tests. No additional public columns "
            "are required for this claim."
        )
    recommended = design.recommended_option
    if recommended is not None:
        return (
            "The current public representation does not support the claim, but "
            f"adding `{recommended.label}` produces a certifying repair under "
            "the existing audit evidence. This is the recommended public-report "
            "design under the supplied action costs."
        )
    if design.frontier is not None and design.frontier.minimal_stable is not None:
        minimal = design.frontier.minimal_stable
        return (
            "The repair-plan ranking did not identify a preferred action, but "
            "the frontier contains a stable representation. The minimal stable "
            f"candidate adds `{_column_label(minimal.added_columns)}`."
        )
    if design.audit.failed:
        return (
            "The current public representation does not support the claim, and "
            "no evaluated candidate refinement certifies it under the declared "
            "limits. Broaden the candidate refinements, relax the reporting "
            "constraints, or report the claim as unstable."
        )
    return (
        "The design is inconclusive under the supplied claim settings. Add an "
        "ambiguity limit, a decision rule, or candidate refinements to turn the "
        "audit into an actionable public-report design."
    )


def _claim_refinement_recommendations(
    report: ClaimAudit,
) -> tuple[ClaimRefinementRecommendation, ...]:
    rows: list[ClaimRefinementRecommendation] = []
    seen: set[tuple[str, ...]] = set()
    repair = report.repair_candidate
    baseline = report.primary.interval.diameter

    if repair is not None and repair.added_columns:
        rows.append(
            ClaimRefinementRecommendation(
                columns=repair.added_columns,
                source=(
                    "decision_repair"
                    if repair is report.decision_repair_candidate
                    else "certificate_repair"
                ),
                before_ambiguity=baseline,
                after_ambiguity=repair.max_ambiguity,
                reduction=baseline - repair.max_ambiguity,
                reduction_percent=_percent_reduction(
                    before=baseline,
                    after=repair.max_ambiguity,
                ),
                public_cells=repair.public_cells,
                meets_ambiguity_limit=_meets_ambiguity_limit(
                    repair.max_ambiguity,
                    report.claim.ambiguity_limit,
                ),
                selected_repair=True,
                decision_repair=repair is report.decision_repair_candidate,
                reason=_repair_reason(report, repair),
            )
        )
        seen.add(repair.added_columns)

    for row in report.primary.refinements:
        columns = (row.column,)
        if columns in seen:
            continue
        rows.append(
            _one_column_claim_recommendation(
                row,
                report=report,
            )
        )
        seen.add(columns)

    return tuple(rows)


def _one_column_claim_recommendation(
    row: RefinementCandidate,
    *,
    report: ClaimAudit,
) -> ClaimRefinementRecommendation:
    columns = (row.column,)
    repair = report.repair_candidate
    decision_repair = (
        report.decision_repair_candidate is not None
        and report.decision_repair_candidate.added_columns == columns
    )
    selected_repair = repair is not None and repair.added_columns == columns
    meets_limit = _meets_ambiguity_limit(
        row.after_ambiguity,
        report.claim.ambiguity_limit,
    )
    return ClaimRefinementRecommendation(
        columns=columns,
        source="one_column_screen",
        before_ambiguity=row.before_ambiguity,
        after_ambiguity=row.after_ambiguity,
        reduction=row.reduction,
        reduction_percent=row.reduction_percent,
        public_cells=row.public_cells,
        meets_ambiguity_limit=meets_limit,
        selected_repair=selected_repair,
        decision_repair=decision_repair,
        reason=_refinement_reason(
            selected_repair=selected_repair,
            decision_repair=decision_repair,
            meets_ambiguity_limit=meets_limit,
            reduction=row.reduction,
        ),
    )


def _repair_reason(
    report: ClaimAudit,
    repair: PublicRepresentationCandidate,
) -> str:
    if repair is report.decision_repair_candidate:
        return "selected repair; makes the observed decision invariant"
    if report.claim.ambiguity_limit is not None:
        return "selected repair; satisfies the declared ambiguity limit"
    return "selected repair from the representation certificate"


def _refinement_reason(
    *,
    selected_repair: bool,
    decision_repair: bool,
    meets_ambiguity_limit: bool | None,
    reduction: float,
) -> str:
    if decision_repair:
        return "makes the observed decision invariant"
    if selected_repair:
        return "selected repair for this claim"
    if meets_ambiguity_limit is True:
        return "satisfies the declared ambiguity limit"
    if meets_ambiguity_limit is False:
        return "reduces ambiguity but does not satisfy the declared limit"
    if reduction > 0:
        return "reduces hidden-composition ambiguity"
    return "does not reduce hidden-composition ambiguity"


def _meets_ambiguity_limit(
    ambiguity: float,
    limit: float | None,
) -> bool | None:
    if limit is None:
        return None
    return ambiguity <= limit


def _percent_reduction(*, before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return 100.0 * (before - after) / before


def _certificate_summary_markdown(
    certificate: RepresentationStabilityCertificate,
) -> list[str]:
    lines = [
        f"- Certificate status: **{certificate.status.upper()}**",
        f"- Exact search required: {'yes' if certificate.exact_required else 'no'}",
        f"- Search exact: {'yes' if certificate.search_exact else 'no'}",
    ]
    if certificate.selected_candidate is not None:
        candidate = certificate.selected_candidate
        lines.extend(
            [
                f"- Selected representation: `{candidate.label}`",
                f"- Public cells: {candidate.public_cells}",
                f"- Max ambiguity: {candidate.max_ambiguity:.4f}",
            ]
        )
    if certificate.frontier.screening is not None:
        screening = certificate.frontier.screening
        lines.extend(
            [
                f"- Frontier screening backend: {screening.backend}",
                "- Frontier screening endpoints certified: "
                f"{screening.certified_count}/{screening.endpoint_count}",
                f"- Frontier exact fallbacks run: {screening.exact_solve_count}",
                "- Frontier exact solves avoided: "
                f"{screening.exact_solve_avoided_count}",
                "- Frontier conservative endpoints used: "
                f"{screening.conservative_endpoint_count}",
            ]
        )
    lines.extend(f"- {reason}" for reason in certificate.reasons)
    return lines


def _refinement_markdown(report: ClaimAudit) -> list[str]:
    recommendations = report.refinement_recommendations
    if recommendations:
        lines = [
            "Claim-centered refinement recommendations rank public "
            "representation changes by whether they repair the declared claim "
            "and how much hidden-composition ambiguity they remove.",
            "",
            "| rank | refinement | role | after | reduction | public cells | claim signal |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for index, row in enumerate(recommendations[: report.claim.top], start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        f"`{row.label}`",
                        row.source,
                        f"{row.after_ambiguity:.4f}",
                        _format_optional_float(row.reduction),
                        str(row.public_cells),
                        _escape_table(row.reason),
                    ]
                )
                + " |"
            )
        return lines
    return [
        "No candidate refinements were supplied, so the auditor cannot propose "
        "a public-representation repair."
    ]


def _model_assisted_markdown(
    summary: ModelAssistedStabilitySummary,
) -> list[str]:
    lines = [
        "This section samples plausible public/hidden joint compositions from "
        "the fitted nonparametric joint model, then reruns the primary "
        "hidden-composition ambiguity audit on each sampled composition.",
        "",
        f"- Joint model method: {summary.joint_model.method}",
        f"- Joint cells: {summary.joint_model.cell_count}",
        f"- Draws: {summary.successful_draws}/{summary.draw_count} successful",
        f"- Draw failures: {summary.error_count}",
        f"- Claim failure rate across successful draws: "
        f"{_format_optional_rate(summary.failure_rate)}",
        f"- Public adequacy rate: "
        f"{_format_optional_rate(summary.public_adequate_rate)}",
    ]
    if summary.ambiguity_min is not None and summary.ambiguity_max is not None:
        lines.extend(
            [
                f"- Ambiguity range: {summary.ambiguity_min:.4f} to "
                f"{summary.ambiguity_max:.4f}",
                f"- Mean ambiguity: {summary.ambiguity_mean:.4f}",
            ]
        )
    if summary.uncertainty_report is not None:
        ambiguity = summary.uncertainty_report.ambiguity_summary
        lines.append(
            "- Posterior/bootstrap ambiguity interval: "
            f"[{_format_optional_float(ambiguity.lower)}, "
            f"{_format_optional_float(ambiguity.upper)}]"
        )
    lines.extend(
        [
            "",
            "| draw | status | observed | ambiguity | lower | upper | adequate |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.rows[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.draw_index),
                    row.status,
                    _format_optional_float(row.observed_value),
                    _format_optional_float(row.ambiguity),
                    _format_optional_float(row.lower),
                    _format_optional_float(row.upper),
                    ""
                    if row.public_adequate is None
                    else ("yes" if row.public_adequate else "no"),
                ]
            )
            + " |"
        )
    return lines


def _witness_shift_table(witness: WitnessReport) -> list[str]:
    rows = [
        "| hidden cell | public cell | target | lower mass | upper mass | shift |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in witness.cells[: witness.top]:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{row.state_label}`",
                    f"`{row.public_label}`",
                    f"{row.target_value:.4f}",
                    f"{row.lower_mass:.4f}",
                    f"{row.upper_mass:.4f}",
                    f"{row.mass_shift:.4f}",
                ]
            )
            + " |"
        )
    return rows


def _format_statistical_uncertainty(value: StatisticalUncertainty) -> str:
    parts = []
    if value.estimate is not None:
        parts.append(f"estimate={value.estimate:.4f}")
    if value.standard_error is not None:
        parts.append(f"SE={value.standard_error:.4f}")
    if value.lower is not None and value.upper is not None:
        level = (
            ""
            if value.confidence_level is None
            else f" {100.0 * value.confidence_level:g}%"
        )
        parts.append(f"{level} interval=[{value.lower:.4f}, {value.upper:.4f}]")
    if value.method:
        parts.append(value.method)
    return "; ".join(parts) if parts else value.label


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_rate(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.1f}%"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def _normalize_decision_operator(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "<=": "<=",
        "le": "<=",
        "leq": "<=",
        "at_most": "<=",
        "less_equal": "<=",
        "less_than_or_equal": "<=",
        "<": "<",
        "lt": "<",
        "below": "<",
        "less_than": "<",
        ">=": ">=",
        "ge": ">=",
        "geq": ">=",
        "at_least": ">=",
        "greater_equal": ">=",
        "greater_than_or_equal": ">=",
        ">": ">",
        "gt": ">",
        "above": ">",
        "greater_than": ">",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "decision operator must be one of '<=', '<', '>=', or '>'"
        ) from exc


def _q_payload(value: Any) -> Any:
    try:
        return QSpec.from_value(value).as_dict()
    except (TypeError, ValueError):
        return _q_label(value)


def _q_label(value: Any) -> str:
    if isinstance(value, QPreset):
        if value.radius is None:
            return value.name
        return f"{value.name}(radius={value.radius:g})"
    return str(value)


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return str(getattr(target, "name", type(target).__name__))


def _column_label(columns: tuple[str, ...]) -> str:
    if not columns:
        return "none"
    return ", ".join(columns)


def _string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    try:
        result = tuple(str(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    if not all(result):
        raise ValueError(f"{name} cannot contain empty strings")
    return result


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of numbers") from exc
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must be non-negative")
    return result


__all__ = [
    "ClaimAudit",
    "ClaimNode",
    "ClaimNodeAudit",
    "ClaimRepairOption",
    "ClaimRepairPlan",
    "ClaimRefinementRecommendation",
    "ClaimScreeningResult",
    "ClaimSpec",
    "ClaimTree",
    "ClaimTreeAudit",
    "DecisionResult",
    "DecisionRule",
    "ModelAssistedDrawResult",
    "ModelAssistedStabilitySummary",
    "PublicReportDesign",
    "audit_claim",
    "audit_claim_tree",
    "claim",
    "claim_tree",
    "design_public_report",
    "plan_claim_repair",
    "threshold_decision",
]
