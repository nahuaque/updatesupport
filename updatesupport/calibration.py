"""Historical calibration of total-variation recomposition stress tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil
from typing import Any, Hashable, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .claim import ClaimAudit, ClaimSpec, PublicReportDesign
from .data import (
    GroupedProblem,
    _hashable_category,
    _iter_records,
    _record_value,
    from_dataframe,
)
from .presets import QPreset, q_tv_budget
from .report import public_descent_report


@dataclass(frozen=True)
class HistoricalTVTransition:
    """One consecutive-period hidden-composition transition."""

    reference_period: Hashable
    evaluation_period: Hashable
    tv_radius: float | None
    calibration_eligible: bool
    support_compatible: bool
    reference_observed_value: float
    recomposed_value: float | None
    composition_target_change: float | None
    reference_weight: float
    evaluation_weight: float
    new_hidden_cells: tuple[tuple[Hashable, ...], ...] = ()
    missing_reference_public_cells: tuple[tuple[Hashable, ...], ...] = ()
    missing_reference_public_mass: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_period": self.reference_period,
            "evaluation_period": self.evaluation_period,
            "tv_radius": self.tv_radius,
            "calibration_eligible": self.calibration_eligible,
            "support_compatible": self.support_compatible,
            "reference_observed_value": self.reference_observed_value,
            "recomposed_value": self.recomposed_value,
            "composition_target_change": self.composition_target_change,
            "reference_weight": self.reference_weight,
            "evaluation_weight": self.evaluation_weight,
            "new_hidden_cell_count": len(self.new_hidden_cells),
            "new_hidden_cells": self.new_hidden_cells,
            "missing_reference_public_cell_count": len(
                self.missing_reference_public_cells
            ),
            "missing_reference_public_cells": self.missing_reference_public_cells,
            "missing_reference_public_mass": self.missing_reference_public_mass,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RollingTVBacktest:
    """One rolling one-step TV-radius backtest."""

    reference_period: Hashable
    evaluation_period: Hashable
    training_transition_count: int
    calibrated_radius: float
    actual_tv_radius: float | None
    status: str
    support_compatible: bool
    shift_covered: bool | None
    reference_observed_value: float
    recomposed_value: float | None
    lower: float
    upper: float
    ambiguity: float
    target_covered: bool | None
    ambiguity_limit_met: bool | None
    decision_invariant: bool | None
    decision_certified: bool | None
    reference_decision: str | None
    realized_decision: str | None
    realized_decision_matches_reference: bool | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"covered", "miss", "unsupported_support"}:
            raise ValueError(
                "status must be 'covered', 'miss', or 'unsupported_support'"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_period": self.reference_period,
            "evaluation_period": self.evaluation_period,
            "training_transition_count": self.training_transition_count,
            "calibrated_radius": self.calibrated_radius,
            "actual_tv_radius": self.actual_tv_radius,
            "status": self.status,
            "support_compatible": self.support_compatible,
            "shift_covered": self.shift_covered,
            "reference_observed_value": self.reference_observed_value,
            "recomposed_value": self.recomposed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "target_covered": self.target_covered,
            "ambiguity_limit_met": self.ambiguity_limit_met,
            "decision_invariant": self.decision_invariant,
            "decision_certified": self.decision_certified,
            "reference_decision": self.reference_decision,
            "realized_decision": self.realized_decision,
            "realized_decision_matches_reference": (
                self.realized_decision_matches_reference
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HistoricalTVCalibrationReport(ReportArtifactMixin):
    """Historical TV-radius calibration with rolling one-step backtests."""

    claim: ClaimSpec
    period_column: str
    period_order: tuple[Hashable, ...]
    coverage: float
    min_train_transitions: int
    calibrated_radius: float
    transitions: tuple[HistoricalTVTransition, ...]
    backtests: tuple[RollingTVBacktest, ...]
    backend: str = "cvxpy"
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None
    title: str = "Historical TV-Radius Calibration"
    limitations: tuple[str, ...] = ()

    @property
    def eligible_transition_count(self) -> int:
        return sum(row.calibration_eligible for row in self.transitions)

    @property
    def unsupported_transition_count(self) -> int:
        return sum(not row.calibration_eligible for row in self.transitions)

    @property
    def backtest_count(self) -> int:
        return len(self.backtests)

    @property
    def evaluable_backtest_count(self) -> int:
        return sum(row.shift_covered is not None for row in self.backtests)

    @property
    def rolling_shift_coverage(self) -> float | None:
        rows = [
            row.shift_covered for row in self.backtests if row.shift_covered is not None
        ]
        if not rows:
            return None
        return sum(bool(value) for value in rows) / len(rows)

    @property
    def rolling_target_coverage(self) -> float | None:
        rows = [
            row.target_covered
            for row in self.backtests
            if row.target_covered is not None
        ]
        if not rows:
            return None
        return sum(bool(value) for value in rows) / len(rows)

    @property
    def rolling_decision_preservation(self) -> float | None:
        rows = [
            row.realized_decision_matches_reference
            for row in self.backtests
            if row.realized_decision_matches_reference is not None
        ]
        if not rows:
            return None
        return sum(bool(value) for value in rows) / len(rows)

    @property
    def q(self) -> QPreset:
        """Return the TV preset calibrated on all eligible transitions."""

        return q_tv_budget(
            self.calibrated_radius,
            backend=self.backend,
            solver=self.solver,
            solver_options=self.solver_options,
        )

    @property
    def calibrated_claim(self) -> ClaimSpec:
        """Return the source claim with the calibrated TV preset installed."""

        return replace(self.claim, q=self.q, q_presets=(self.q,))

    def audit(self, data: Any, **kwargs: Any) -> ClaimAudit:
        """Audit new data using the calibrated TV radius."""

        return self.calibrated_claim.audit(data, **kwargs)

    def design(self, data: Any, **kwargs: Any) -> PublicReportDesign:
        """Design a public report using the calibrated TV radius."""

        return self.calibrated_claim.design(data, **kwargs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "claim": self.claim.as_dict(),
            "period_column": self.period_column,
            "period_order": self.period_order,
            "coverage": self.coverage,
            "quantile_method": "higher",
            "min_train_transitions": self.min_train_transitions,
            "calibrated_radius": self.calibrated_radius,
            "q": {
                "name": "tv_budget",
                "radius": self.calibrated_radius,
                "backend": self.backend,
                "solver": self.solver,
                "solver_options": self.solver_options,
            },
            "transition_count": len(self.transitions),
            "eligible_transition_count": self.eligible_transition_count,
            "unsupported_transition_count": self.unsupported_transition_count,
            "backtest_count": self.backtest_count,
            "evaluable_backtest_count": self.evaluable_backtest_count,
            "rolling_shift_coverage": self.rolling_shift_coverage,
            "rolling_target_coverage": self.rolling_target_coverage,
            "rolling_decision_preservation": self.rolling_decision_preservation,
            "transitions": [row.as_dict() for row in self.transitions],
            "backtests": [row.as_dict() for row in self.backtests],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    key: value
                    for key, value in self.as_dict().items()
                    if key not in {"claim", "transitions", "backtests", "limitations"}
                },
            ),
            "claim": (self.claim.as_dict(),),
            "transitions": tuple(row.as_dict() for row in self.transitions),
            "backtests": tuple(row.as_dict() for row in self.backtests),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Claim: {self.claim.estimate_name}",
            f"- Period column: `{self.period_column}`",
            f"- Periods: {len(self.period_order)}",
            f"- Eligible historical transitions: {self.eligible_transition_count}",
            f"- Unsupported support-drift transitions: {self.unsupported_transition_count}",
            f"- Target historical coverage: {self.coverage:.1%}",
            f"- Calibrated TV radius: {self.calibrated_radius:.4f}",
            f"- Rolling backtests: {self.backtest_count}",
            f"- Rolling shift coverage: {_format_percent(self.rolling_shift_coverage)}",
            "- Rolling target-interval coverage: "
            f"{_format_percent(self.rolling_target_coverage)}",
        ]
        if self.claim.decision is not None:
            lines.append(
                "- Rolling decision preservation: "
                f"{_format_percent(self.rolling_decision_preservation)}"
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    "The calibrated radius is the higher empirical quantile of "
                    "eligible consecutive-period TV distances. Each distance "
                    "compares the later hidden mix after restandardizing it to "
                    "the earlier period's public law. The rolling rows use only "
                    "transitions available before the evaluation period."
                ),
                "",
                (
                    "Shift coverage asks whether the realized restandardized "
                    "composition fell inside the calibrated TV ball. Target "
                    "coverage asks whether the target value obtained by applying "
                    "the later composition to the earlier retained-cell target "
                    "values fell inside the resulting audit interval."
                ),
                "",
                "## Historical Transitions",
                "",
                "| reference | evaluation | TV radius | eligible | support compatible | target change |",
                "| --- | --- | ---: | :---: | :---: | ---: |",
            ]
        )
        for row in self.transitions:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.reference_period),
                        str(row.evaluation_period),
                        _format_float(row.tv_radius),
                        _yes_no(row.calibration_eligible),
                        _yes_no(row.support_compatible),
                        _format_float(row.composition_target_change),
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "## Rolling Backtests",
                "",
            ]
        )
        if not self.backtests:
            lines.append(
                "No rolling rows were available after the requested training warmup."
            )
        else:
            lines.extend(
                [
                    "| reference | evaluation | trained on | calibrated | actual | shift covered | target covered |",
                    "| --- | --- | ---: | ---: | ---: | :---: | :---: |",
                ]
            )
            for row in self.backtests:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.reference_period),
                            str(row.evaluation_period),
                            str(row.training_transition_count),
                            f"{row.calibrated_radius:.4f}",
                            _format_float(row.actual_tv_radius),
                            _yes_no_optional(row.shift_covered),
                            _yes_no_optional(row.target_covered),
                        ]
                    )
                    + " |"
                )

        lines.extend(["", "## Assumptions And Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class _PeriodProfile:
    period: Hashable
    rows: tuple[Mapping[str, Any], ...]
    grouped: GroupedProblem
    cell_weights: Mapping[tuple[Hashable, ...], float]
    public_law: Mapping[tuple[Hashable, ...], float]
    public_map: Mapping[tuple[Hashable, ...], tuple[Hashable, ...]]
    target_values: Mapping[tuple[Hashable, ...], float]
    observed_value: float


def calibrate_tv_radius(
    data: Any,
    claim: ClaimSpec | Mapping[str, Any],
    *,
    period: str,
    period_order: Sequence[Hashable] | None = None,
    coverage: float = 0.90,
    min_train_transitions: int = 3,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
    tolerance: float = 1e-9,
    title: str = "Historical TV-Radius Calibration",
) -> HistoricalTVCalibrationReport:
    """Calibrate a TV radius from history and run rolling one-step backtests.

    The later period's hidden composition is restandardized to the earlier
    period's public law before TV distance is measured. This isolates
    within-public-cell recomposition from changes in public bucket shares.
    """

    if isinstance(claim, Mapping):
        claim = ClaimSpec.from_dict(claim)
    if not isinstance(claim, ClaimSpec):
        raise TypeError("claim must be a ClaimSpec or mapping")
    if not isinstance(period, str) or not period:
        raise ValueError("period must be a non-empty column name")
    if period in claim.hidden:
        raise ValueError("period must not be part of the hidden/public representation")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if min_train_transitions <= 0:
        raise ValueError("min_train_transitions must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    records = tuple(_iter_records(data))
    if not records:
        raise ValueError("data must contain at least one row")
    rows_by_period: dict[Hashable, list[Mapping[str, Any]]] = defaultdict(list)
    observed_order: list[Hashable] = []
    for row_number, row in enumerate(records, start=1):
        period_value = _hashable_category(
            _record_value(row, period, row_number=row_number)
        )
        if period_value not in rows_by_period:
            observed_order.append(period_value)
        rows_by_period[period_value].append(row)

    ordered_periods = _resolve_period_order(observed_order, period_order)
    if len(ordered_periods) < 2:
        raise ValueError("historical TV calibration requires at least two periods")

    profiles = tuple(
        _period_profile(
            period_value,
            tuple(rows_by_period[period_value]),
            claim,
        )
        for period_value in ordered_periods
    )
    transitions = tuple(
        _transition(profiles[index - 1], profiles[index], tolerance=tolerance)
        for index in range(1, len(profiles))
    )
    eligible_distances = tuple(
        float(row.tv_radius)
        for row in transitions
        if row.calibration_eligible and row.tv_radius is not None
    )
    if not eligible_distances:
        raise ValueError(
            "no support-compatible historical transitions are available for "
            "TV-radius calibration"
        )

    backtests: list[RollingTVBacktest] = []
    for index, transition in enumerate(transitions):
        prior_distances = tuple(
            float(row.tv_radius)
            for row in transitions[:index]
            if row.calibration_eligible and row.tv_radius is not None
        )
        if len(prior_distances) < min_train_transitions:
            continue
        radius = _higher_quantile(prior_distances, coverage)
        backtests.append(
            _rolling_backtest(
                transition,
                reference=profiles[index],
                claim=claim,
                radius=radius,
                training_transition_count=len(prior_distances),
                backend=backend,
                solver=solver,
                solver_options=solver_options,
                tolerance=tolerance,
            )
        )

    limitations = (
        "The radius is an empirical historical stress calibration, not a "
        "guarantee against future regime changes.",
        "Calibration is conditional on the retained hidden columns, period-level "
        "minimum-cell filtering, and support compatibility.",
        "Later hidden composition is restandardized to the earlier public law, so "
        "the TV distance excludes changes in public bucket shares.",
        "Target backtests hold the earlier retained-cell target values fixed; they "
        "measure composition sensitivity, not target or model drift.",
        "Transitions with new retained hidden cells or missing reference public "
        "fibers are reported as support drift and excluded from radius calibration.",
    )
    return HistoricalTVCalibrationReport(
        claim=claim,
        period_column=period,
        period_order=ordered_periods,
        coverage=float(coverage),
        min_train_transitions=int(min_train_transitions),
        calibrated_radius=_higher_quantile(eligible_distances, coverage),
        transitions=transitions,
        backtests=tuple(backtests),
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
        title=title,
        limitations=limitations,
    )


def _period_profile(
    period: Hashable,
    rows: tuple[Mapping[str, Any], ...],
    claim: ClaimSpec,
) -> _PeriodProfile:
    grouped = from_dataframe(
        rows,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        min_cell_weight=claim.min_cell_weight,
        q="observed",
    )
    if not grouped.problem.has_linear_target:
        raise TypeError(
            "historical TV calibration currently requires a target that compiles "
            "to a fixed linear functional"
        )
    cell_weights = dict(grouped.cell_weights)
    target_values = dict(grouped.problem.estimand_map)
    observed_value = sum(
        cell_weights[state] * target_values[state] for state in grouped.problem.states
    )
    return _PeriodProfile(
        period=period,
        rows=rows,
        grouped=grouped,
        cell_weights=cell_weights,
        public_law=dict(grouped.public_law),
        public_map=dict(grouped.problem.public_map),
        target_values=target_values,
        observed_value=observed_value,
    )


def _transition(
    reference: _PeriodProfile,
    evaluation: _PeriodProfile,
    *,
    tolerance: float,
) -> HistoricalTVTransition:
    missing_public = tuple(
        public_value
        for public_value, mass in reference.public_law.items()
        if mass > tolerance
        and evaluation.public_law.get(public_value, 0.0) <= tolerance
    )
    missing_public_mass = sum(reference.public_law[value] for value in missing_public)
    if missing_public:
        return HistoricalTVTransition(
            reference_period=reference.period,
            evaluation_period=evaluation.period,
            tv_radius=None,
            calibration_eligible=False,
            support_compatible=False,
            reference_observed_value=reference.observed_value,
            recomposed_value=None,
            composition_target_change=None,
            reference_weight=reference.grouped.total_weight,
            evaluation_weight=evaluation.grouped.total_weight,
            missing_reference_public_cells=missing_public,
            missing_reference_public_mass=missing_public_mass,
            reason=(
                "the evaluation period has no hidden-composition observation for "
                "one or more positive-mass reference public cells"
            ),
        )

    restandardized: dict[tuple[Hashable, ...], float] = defaultdict(float)
    for state, mass in evaluation.cell_weights.items():
        public_value = evaluation.public_map[state]
        reference_public_mass = reference.public_law.get(public_value, 0.0)
        if reference_public_mass <= tolerance:
            continue
        evaluation_public_mass = evaluation.public_law[public_value]
        restandardized[state] += reference_public_mass * mass / evaluation_public_mass

    state_union = set(reference.cell_weights) | set(restandardized)
    tv_radius = 0.5 * sum(
        abs(reference.cell_weights.get(state, 0.0) - restandardized.get(state, 0.0))
        for state in state_union
    )
    new_hidden_cells = tuple(
        sorted(
            (
                state
                for state, mass in restandardized.items()
                if mass > tolerance and state not in reference.cell_weights
            ),
            key=str,
        )
    )
    support_compatible = not new_hidden_cells
    if support_compatible:
        recomposed_value = sum(
            restandardized.get(state, 0.0) * reference.target_values[state]
            for state in reference.cell_weights
        )
        composition_target_change = recomposed_value - reference.observed_value
        reason = "support-compatible within-public recomposition"
    else:
        recomposed_value = None
        composition_target_change = None
        reason = (
            "the evaluation period assigns positive standardized mass to hidden "
            "cells absent from the reference retained support"
        )
    return HistoricalTVTransition(
        reference_period=reference.period,
        evaluation_period=evaluation.period,
        tv_radius=tv_radius,
        calibration_eligible=support_compatible,
        support_compatible=support_compatible,
        reference_observed_value=reference.observed_value,
        recomposed_value=recomposed_value,
        composition_target_change=composition_target_change,
        reference_weight=reference.grouped.total_weight,
        evaluation_weight=evaluation.grouped.total_weight,
        new_hidden_cells=new_hidden_cells,
        reason=reason,
    )


def _rolling_backtest(
    transition: HistoricalTVTransition,
    *,
    reference: _PeriodProfile,
    claim: ClaimSpec,
    radius: float,
    training_transition_count: int,
    backend: str,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
    tolerance: float,
) -> RollingTVBacktest:
    report = public_descent_report(
        reference.rows,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        q=q_tv_budget(
            radius,
            backend=backend,
            solver=solver,
            solver_options=solver_options,
        ),
        min_cell_weight=claim.min_cell_weight,
        candidate_refinements=(),
        top=claim.top,
        title=f"TV Backtest Reference {transition.reference_period}",
        target_description=claim.target_description or claim.estimate_name,
        observed_label=claim.observed_label,
    )
    support_compatible = transition.support_compatible
    if not support_compatible:
        status = "unsupported_support"
        shift_covered = None
        target_covered = None
    else:
        shift_covered = bool(
            transition.tv_radius is not None
            and transition.tv_radius <= radius + tolerance
        )
        status = "covered" if shift_covered else "miss"
        target_covered = bool(
            transition.recomposed_value is not None
            and report.interval.lower - tolerance
            <= transition.recomposed_value
            <= report.interval.upper + tolerance
        )

    ambiguity_limit_met = (
        None
        if claim.ambiguity_limit is None
        else report.interval.diameter <= claim.ambiguity_limit + tolerance
    )
    decision_invariant = None
    decision_certified = None
    reference_decision = None
    realized_decision = None
    realized_decision_matches_reference = None
    if claim.decision is not None:
        decision = claim.decision.interval_result(
            observed_value=report.observed_value,
            lower=report.interval.lower,
            upper=report.interval.upper,
        )
        decision_invariant = decision.invariant
        decision_certified = (
            decision.invariant
            and decision.certified_decision == decision.observed_decision
        )
        reference_decision = decision.observed_decision
        if transition.recomposed_value is not None:
            realized_decision = claim.decision.evaluate(transition.recomposed_value)
            realized_decision_matches_reference = (
                realized_decision == reference_decision
            )

    return RollingTVBacktest(
        reference_period=transition.reference_period,
        evaluation_period=transition.evaluation_period,
        training_transition_count=training_transition_count,
        calibrated_radius=radius,
        actual_tv_radius=transition.tv_radius,
        status=status,
        support_compatible=support_compatible,
        shift_covered=shift_covered,
        reference_observed_value=report.observed_value,
        recomposed_value=transition.recomposed_value,
        lower=report.interval.lower,
        upper=report.interval.upper,
        ambiguity=report.interval.diameter,
        target_covered=target_covered,
        ambiguity_limit_met=ambiguity_limit_met,
        decision_invariant=decision_invariant,
        decision_certified=decision_certified,
        reference_decision=reference_decision,
        realized_decision=realized_decision,
        realized_decision_matches_reference=realized_decision_matches_reference,
        reason=transition.reason,
    )


def _resolve_period_order(
    observed_order: Sequence[Hashable],
    requested: Sequence[Hashable] | None,
) -> tuple[Hashable, ...]:
    observed = tuple(observed_order)
    if requested is None:
        try:
            return tuple(sorted(observed))
        except TypeError:
            return tuple(sorted(observed, key=str))
    normalized = tuple(_hashable_category(value) for value in requested)
    if len(set(normalized)) != len(normalized):
        raise ValueError("period_order must not contain duplicates")
    observed_set = set(observed)
    requested_set = set(normalized)
    if requested_set != observed_set:
        missing = observed_set - requested_set
        extra = requested_set - observed_set
        raise ValueError(
            "period_order must contain every observed period exactly once; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return normalized


def _higher_quantile(values: Sequence[float], coverage: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile values must be non-empty")
    rank = min(len(ordered), max(1, ceil(float(coverage) * len(ordered))))
    return ordered[rank - 1]


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _yes_no_optional(value: bool | None) -> str:
    return "n/a" if value is None else _yes_no(value)
