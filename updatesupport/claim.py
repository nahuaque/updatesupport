"""Claim-level reporting-stability verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .certificate import (
    RepresentationStabilityCertificate,
    certify_public_representation,
)
from .data import TabularTarget
from .frontier import PublicRepresentationCandidate, public_representation_frontier
from .joint import (
    HiddenCompositionUncertaintyReport,
    NonparametricJointDistribution,
    hidden_composition_uncertainty,
)
from .presets import QPreset
from .report import (
    PublicDescentReport,
    StatisticalUncertainty,
    WitnessReport,
    public_descent_report,
)
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
    """Create a threshold decision rule for claim verification."""

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
class ReportingClaim:
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
        object.__setattr__(
            self,
            "statistical_uncertainty",
            _normalize_statistical_uncertainty(
                self.statistical_uncertainty,
                interval=self.statistical_interval,
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReportingClaim":
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
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for as_dict()."""

        return self.as_dict()

    def verify(self, data: Any, **kwargs: Any) -> "ClaimVerificationReport":
        """Verify this claim against tabular data."""

        return verify_claim(data, self, **kwargs)


@dataclass(frozen=True)
class ClaimVerificationReport:
    """Review artifact that certifies, breaks, or repairs a reporting claim."""

    claim: ReportingClaim
    primary: PublicDescentReport
    certificate: RepresentationStabilityCertificate | None = None
    witness: WitnessReport | None = None
    model_assisted: ModelAssistedStabilitySummary | None = None
    decision: DecisionResult | None = None
    decision_repair_candidate: PublicRepresentationCandidate | None = None
    decision_repair_search_exact: bool | None = None
    status: str = "inconclusive"
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    title: str = "Reporting Claim Verification"

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
            "repair_candidate": None
            if self.repair_candidate is None
            else self.repair_candidate.as_dict(),
            "reasons": self.reasons,
            "limitations": self.limitations,
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
        if self.model_assisted is not None:
            failure_rate = _format_optional_rate(self.model_assisted.failure_rate)
            lines.append(
                "- Model-assisted joint draws: "
                f"{self.model_assisted.successful_draws}/"
                f"{self.model_assisted.draw_count} successful"
                f"; failure rate {failure_rate}"
            )

        lines.extend(["", "## Decision Basis", ""])
        lines.extend(f"- {reason}" for reason in self.reasons)

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
                "supplied to `updatesupport`. The verifier does not refit the "
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


def verify_claim(
    data: Any,
    claim: ReportingClaim | Mapping[str, Any],
    *,
    joint_model: NonparametricJointDistribution | None = None,
    joint_draws: int = 0,
    joint_seed: int | None = None,
    **overrides: Any,
) -> ClaimVerificationReport:
    """Verify a declared reporting claim against tabular data."""

    if not isinstance(claim, ReportingClaim):
        claim = ReportingClaim.from_dict(claim)
    if overrides:
        claim = replace(claim, **overrides)
    if joint_draws < 0:
        raise ValueError("joint_draws must be non-negative")

    primary = public_descent_report(
        data,
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

    return ClaimVerificationReport(
        claim=claim,
        primary=primary,
        certificate=certificate,
        witness=witness,
        model_assisted=model_assisted,
        decision=decision,
        decision_repair_candidate=decision_repair_candidate,
        decision_repair_search_exact=decision_repair_search_exact,
        status=status,
        reasons=reasons,
        limitations=_claim_limitations(claim, primary, certificate),
        title=claim.title or "Reporting Claim Verification",
    )


def _claim_status(
    claim: ReportingClaim,
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
                "verifier reports evidence but cannot issue a pass/fail "
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
    claim: ReportingClaim,
    primary: PublicDescentReport,
    certificate: RepresentationStabilityCertificate | None,
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
    if certificate is not None and not certificate.search_exact:
        limitations.append(
            "The repair/certificate search was not exact over the full declared "
            "candidate space."
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
    return tuple(limitations)


def _decision_repair_candidate(
    data: Any,
    *,
    claim: ReportingClaim,
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
    claim: ReportingClaim,
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
    claim: ReportingClaim,
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


def _decision_markdown(report: ClaimVerificationReport) -> list[str]:
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


def _decision_repair_markdown(report: ClaimVerificationReport) -> list[str]:
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
    lines.extend(f"- {reason}" for reason in certificate.reasons)
    return lines


def _refinement_markdown(report: ClaimVerificationReport) -> list[str]:
    if report.decision_repair_candidate is not None or (
        report.certificate is not None
        and report.certificate.selected_candidate is not None
    ):
        candidate = report.repair_candidate
        if candidate.added_columns:
            return [
                "The selected repair promotes these hidden variables "
                "into the public representation:",
                "",
                f"- `{_column_label(candidate.added_columns)}`",
            ]
    if report.primary.refinements:
        lines = [
            "One-column refinement screening ranks hidden variables by ambiguity "
            "reduction under the primary Q preset:",
            "",
            "| rank | column | before | after | reduction |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
        for index, row in enumerate(
            report.primary.refinements[: report.claim.top], start=1
        ):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(index),
                        f"`{row.column}`",
                        f"{row.before_ambiguity:.4f}",
                        f"{row.after_ambiguity:.4f}",
                        f"{row.reduction:.4f}",
                    ]
                )
                + " |"
            )
        return lines
    return [
        "No candidate refinements were supplied, so the verifier cannot propose "
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
    "ClaimVerificationReport",
    "DecisionResult",
    "DecisionRule",
    "ModelAssistedDrawResult",
    "ModelAssistedStabilitySummary",
    "ReportingClaim",
    "threshold_decision",
    "verify_claim",
]
