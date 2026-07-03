"""Breakdown-point analysis for reporting-stability decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

from .claim import DecisionResult, DecisionRule
from .data import TabularStandardError, TabularTarget
from .presets import (
    q_bounded_shift,
    q_chi_square_budget,
    q_kl_budget,
    q_l2_budget,
    q_tv_budget,
)
from .report import public_descent_report

QFamily = Callable[[float], Any]


@dataclass(frozen=True)
class BreakdownCurvePoint:
    """One radius evaluation in a breakdown-point analysis."""

    radius: float
    observed_value: float
    lower: float
    upper: float
    ambiguity: float
    public_adequate: bool
    q_name: str
    q_description: str
    decision: DecisionResult

    @property
    def observed_decision(self) -> str:
        return self.decision.observed_decision

    @property
    def lower_decision(self) -> str:
        return self.decision.lower_decision

    @property
    def upper_decision(self) -> str:
        return self.decision.upper_decision

    @property
    def decision_invariant(self) -> bool:
        return self.decision.invariant

    @property
    def decision_stable(self) -> bool:
        return (
            self.decision.invariant
            and self.decision.certified_decision == self.decision.observed_decision
        )

    @property
    def threshold_crossed(self) -> bool:
        return self.decision.threshold_crossed

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "observed_decision": self.observed_decision,
            "lower_decision": self.lower_decision,
            "upper_decision": self.upper_decision,
            "decision_invariant": self.decision_invariant,
            "decision_stable": self.decision_stable,
            "threshold_crossed": self.threshold_crossed,
            "decision_result": self.decision.as_dict(),
        }


@dataclass(frozen=True)
class BreakdownPointReport:
    """Report for the smallest stress radius that breaks a decision claim."""

    title: str
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target: str | None
    target_description: str
    observed_label: str
    decision: DecisionRule
    q_family_name: str
    q_family_description: str
    radius_min: float
    radius_max: float
    tolerance: float
    max_iterations: int
    status: str
    breakdown_radius: float | None
    stable_radius: float | None
    broken_radius: float | None
    observed_value: float
    observed_decision: str
    curve: tuple[BreakdownCurvePoint, ...]

    @property
    def found(self) -> bool:
        return self.status == "found"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target": self.target,
            "target_description": self.target_description,
            "observed_label": self.observed_label,
            "decision": self.decision.as_dict(),
            "q_family_name": self.q_family_name,
            "q_family_description": self.q_family_description,
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "breakdown_radius": self.breakdown_radius,
            "stable_radius": self.stable_radius,
            "broken_radius": self.broken_radius,
            "observed_value": self.observed_value,
            "observed_decision": self.observed_decision,
            "curve": [row.as_dict() for row in self.curve],
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the report to JSON."""

        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return named tables for structured export."""

        return {
            "summary": (
                {
                    "title": self.title,
                    "status": self.status,
                    "breakdown_radius": self.breakdown_radius,
                    "stable_radius": self.stable_radius,
                    "broken_radius": self.broken_radius,
                    "radius_min": self.radius_min,
                    "radius_max": self.radius_max,
                    "tolerance": self.tolerance,
                    "max_iterations": self.max_iterations,
                    "observed_value": self.observed_value,
                    "observed_decision": self.observed_decision,
                    "decision_rule": self.decision.name,
                    "q_family_name": self.q_family_name,
                    "q_family_description": self.q_family_description,
                    "target": self.target,
                    "target_description": self.target_description,
                    "public_columns": self.public_columns,
                    "hidden_columns": self.hidden_columns,
                },
            ),
            "curve": tuple(row.as_dict() for row in self.curve),
        }

    def to_dataframes(self) -> dict[str, Any]:
        """Return named pandas DataFrames for the report tables."""

        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        """Render an analyst-facing Markdown interpretation."""

        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Decision rule: `{self.decision.name}`",
            f"- Observed {self.observed_label}: {_format_float(self.observed_value)} "
            f"({self.observed_decision})",
            f"- Q family: `{self.q_family_name}`",
            f"- Search range: {_format_float(self.radius_min)} to "
            f"{_format_float(self.radius_max)}",
            f"- Status: `{self.status}`",
        ]
        if self.status == "found":
            lines.extend(
                [
                    f"- Breakdown radius: about {_format_float(self.breakdown_radius)}",
                    f"- Stable below: {_format_float(self.stable_radius)}",
                    f"- Broken by: {_format_float(self.broken_radius)}",
                ]
            )
        elif self.status == "not_found":
            lines.append(
                "- Breakdown radius: not found inside the requested search range."
            )
        else:
            lines.append(
                "- Breakdown radius: the decision is already broken at the "
                "minimum radius."
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                _interpretation(self),
                "",
                "This is a sensitivity threshold, not a confidence interval. It "
                "depends on the chosen finer refinement and on the chosen nested "
                "admissible-shift family `Q(radius)`. It does not claim absolute "
                "robustness to all possible composition changes.",
                "",
                "## Radius Curve",
                "",
                "| radius | lower | upper | ambiguity | invariant | stable |",
                "|---:|---:|---:|---:|:---:|:---:|",
            ]
        )
        for row in self.curve:
            lines.append(
                "| "
                f"{_format_float(row.radius)} | "
                f"{_format_float(row.lower)} | "
                f"{_format_float(row.upper)} | "
                f"{_format_float(row.ambiguity)} | "
                f"{_yes_no(row.decision_invariant)} | "
                f"{_yes_no(row.decision_stable)} |"
            )
        lines.extend(
            [
                "",
                "## Assumptions And Limitations",
                "",
                "- The radius family is assumed to be nested: larger radii allow "
                "at least the shifts allowed by smaller radii.",
                "- `hidden` columns are observed by the analyst but not part of the "
                "public/reporting representation being stress-tested.",
                "- The breakdown point is relative to this refinement, this target, "
                "and this Q family; it is not an all-purpose guarantee.",
            ]
        )
        return "\n".join(lines)


def breakdown_point(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    decision: DecisionRule | Mapping[str, Any],
    weight: str | None = None,
    target_standard_error: TabularStandardError | None = None,
    q_family: str | QFamily = "bounded_shift",
    radius_min: float = 0.0,
    radius_max: float = 1.0,
    tolerance: float = 1e-4,
    max_iterations: int = 30,
    grid_size: int = 11,
    min_cell_weight: float = 1.0,
    top: int = 5,
    title: str = "Breakdown Point Analysis",
    target_description: str = "target value",
    observed_label: str = "Observed value",
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> BreakdownPointReport:
    """Find the first radius where a reporting decision stops being stable.

    The search assumes that ``q_family(radius)`` is nested in ``radius``. A
    decision is stable when every admissible endpoint implies the same decision
    as the observed aggregate value.
    """

    _validate_search(radius_min, radius_max, tolerance, max_iterations, grid_size)
    decision_rule = DecisionRule.from_value(decision)
    q_name, q_description, q_factory = _resolve_q_family(
        q_family,
        backend=backend,
        solver=solver,
        solver_options=solver_options,
    )

    def evaluate(radius: float) -> BreakdownCurvePoint:
        q = q_factory(radius)
        report = public_descent_report(
            data,
            public=public,
            hidden=hidden,
            target=target,
            target_standard_error=target_standard_error,
            weight=weight,
            q=q,
            min_cell_weight=min_cell_weight,
            top=top,
            title=f"{title} radius={radius:g}",
            target_description=target_description,
            observed_label=observed_label,
        )
        decision_result = decision_rule.interval_result(
            observed_value=report.observed_value,
            lower=report.interval.lower,
            upper=report.interval.upper,
        )
        return BreakdownCurvePoint(
            radius=float(radius),
            observed_value=report.observed_value,
            lower=report.interval.lower,
            upper=report.interval.upper,
            ambiguity=report.interval.diameter,
            public_adequate=report.public_adequate,
            q_name=report.grouped.q_name,
            q_description=report.grouped.q_description,
            decision=decision_result,
        )

    min_point = evaluate(radius_min)
    if not min_point.decision_stable:
        status = "already_broken"
        breakdown_radius = radius_min
        stable_radius = None
        broken_radius = radius_min
    else:
        max_point = evaluate(radius_max)
        if max_point.decision_stable:
            status = "not_found"
            breakdown_radius = None
            stable_radius = radius_max
            broken_radius = None
        else:
            low = radius_min
            high = radius_max
            for _ in range(max_iterations):
                if high - low <= tolerance:
                    break
                mid = (low + high) / 2.0
                mid_point = evaluate(mid)
                if mid_point.decision_stable:
                    low = mid
                else:
                    high = mid
            status = "found"
            breakdown_radius = high
            stable_radius = low
            broken_radius = high

    curve = tuple(
        evaluate(radius) for radius in _radius_grid(radius_min, radius_max, grid_size)
    )
    if target is None:
        target_label = None
    elif isinstance(target, str):
        target_label = target
    else:
        target_label = getattr(target, "name", type(target).__name__)

    return BreakdownPointReport(
        title=title,
        public_columns=tuple(public),
        hidden_columns=tuple(hidden),
        target=target_label,
        target_description=target_description,
        observed_label=observed_label,
        decision=decision_rule,
        q_family_name=q_name,
        q_family_description=q_description,
        radius_min=float(radius_min),
        radius_max=float(radius_max),
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
        status=status,
        breakdown_radius=breakdown_radius,
        stable_radius=stable_radius,
        broken_radius=broken_radius,
        observed_value=min_point.observed_value,
        observed_decision=min_point.observed_decision,
        curve=curve,
    )


def _resolve_q_family(
    q_family: str | QFamily,
    *,
    backend: str,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
) -> tuple[str, str, QFamily]:
    if callable(q_family) and not isinstance(q_family, str):
        return (
            getattr(q_family, "__name__", "custom_q_family"),
            "Custom one-parameter admissible-shift family.",
            q_family,
        )
    if not isinstance(q_family, str):
        raise TypeError("q_family must be a string preset name or callable")

    key = q_family.strip().lower().replace("-", "_")
    if key in {"bounded", "bounded_shift"}:
        return (
            "bounded_shift",
            "Per-fiber bounded shift away from the observed hidden-cell law.",
            q_bounded_shift,
        )
    if key in {"tv", "total_variation", "tv_budget"}:
        return (
            "tv_budget",
            "Total-variation budget around the observed hidden-cell law.",
            lambda radius: q_tv_budget(
                radius,
                backend=backend,
                solver=solver,
                solver_options=solver_options,
            ),
        )
    if key in {"chi_square", "chi2", "chi_square_budget"}:
        return (
            "chi_square_budget",
            "Chi-square divergence budget around the observed hidden-cell law.",
            lambda radius: q_chi_square_budget(
                radius,
                backend=backend,
                solver=solver,
                solver_options=solver_options,
            ),
        )
    if key in {"kl", "kullback_leibler", "kl_budget"}:
        return (
            "kl_budget",
            "KL-divergence budget around the observed hidden-cell law.",
            lambda radius: q_kl_budget(
                radius,
                backend=backend,
                solver=solver,
                solver_options=solver_options,
            ),
        )
    if key in {"l2", "l2_budget"}:
        return (
            "l2_budget",
            "L2 budget around the observed hidden-cell law.",
            lambda radius: q_l2_budget(
                radius,
                backend=backend,
                solver=solver,
                solver_options=solver_options,
            ),
        )
    raise ValueError(
        "unsupported q_family. Expected one of: bounded_shift, tv_budget, "
        "chi_square_budget, kl_budget, l2_budget, or a callable."
    )


def _validate_search(
    radius_min: float,
    radius_max: float,
    tolerance: float,
    max_iterations: int,
    grid_size: int,
) -> None:
    for name, value in {
        "radius_min": radius_min,
        "radius_max": radius_max,
        "tolerance": tolerance,
    }.items():
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if radius_min < 0:
        raise ValueError("radius_min must be non-negative")
    if radius_max < radius_min:
        raise ValueError("radius_max must be greater than or equal to radius_min")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")


def _radius_grid(
    radius_min: float, radius_max: float, grid_size: int
) -> tuple[float, ...]:
    if grid_size == 2:
        return (float(radius_min), float(radius_max))
    step = (radius_max - radius_min) / float(grid_size - 1)
    return tuple(float(radius_min + i * step) for i in range(grid_size))


def _interpretation(report: BreakdownPointReport) -> str:
    if report.status == "found":
        return (
            "The reported decision remains stable for admissible composition "
            f"shifts up to roughly radius {_format_float(report.stable_radius)}, "
            "but it is no longer stable by radius "
            f"{_format_float(report.broken_radius)}. The reported claim has a "
            "finite breakdown point under this Q family."
        )
    if report.status == "not_found":
        return (
            "The reported decision remains stable across the entire requested "
            f"search range, through radius {_format_float(report.radius_max)}. "
            "A breakdown point may still exist outside this range."
        )
    return (
        "The reported decision is not stable even at the minimum requested "
        f"radius {_format_float(report.radius_min)}. Tighten the stress family "
        "or inspect the underlying public representation before interpreting "
        "larger radii."
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(float(value))
    if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e6):
        return f"{float(value):.4e}"
    return f"{float(value):.4f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
