"""Named linear feasibility intervals for arbitrary scalar variables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from scipy.optimize import linprog


DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS = (
    "Intervals are feasibility bounds, not point estimates or confidence "
    "intervals.",
    "Results are conditional on the supplied variables, linear constraints, "
    "target expressions, and active scenario definitions.",
    "A wide, infeasible, or unbounded interval is a property of the encoded "
    "constraint system; it does not by itself validate or invalidate any "
    "external estimate.",
)


@dataclass(frozen=True)
class NamedLinearVariable:
    """One named scalar variable in a linear feasibility problem."""

    name: str
    lower: float | None = None
    upper: float | None = None
    label: str | None = None
    unit: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name cannot be empty")
        lower = None if self.lower is None else _finite_float(self.lower, "lower")
        upper = None if self.upper is None else _finite_float(self.upper, "upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("variable lower bound cannot exceed upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower": self.lower,
            "upper": self.upper,
            "label": self.label,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True)
class NamedLinearExpression:
    """Linear expression ``constant + sum_i coefficients[i] * x_i``."""

    coefficients: Mapping[str, float]
    constant: float = 0.0
    label: str | None = None

    def __post_init__(self) -> None:
        coefficients: dict[str, float] = {}
        for raw_name, raw_value in self.coefficients.items():
            name = str(raw_name)
            value = _finite_float(raw_value, f"coefficient[{name!r}]")
            if value != 0.0:
                coefficients[name] = value
        if not coefficients:
            raise ValueError("linear expression must contain at least one coefficient")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "constant",
            _finite_float(self.constant, "constant"),
        )

    def evaluate(self, assignment: Mapping[str, float]) -> float:
        return self.constant + sum(
            coefficient * float(assignment[name])
            for name, coefficient in self.coefficients.items()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficients": dict(self.coefficients),
            "constant": self.constant,
            "label": self.label,
        }


@dataclass(frozen=True)
class NamedLinearConstraint:
    """Linear interval constraint ``lower <= expression <= upper``."""

    name: str
    expression: NamedLinearExpression | Mapping[str, float] | str
    lower: float | None = None
    upper: float | None = None
    kind: str = "linear"
    provenance: str | None = None
    description: str | None = None
    verified: bool | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name cannot be empty")
        expression = coerce_named_linear_expression(self.expression)
        lower = None if self.lower is None else _finite_float(self.lower, "lower")
        upper = None if self.upper is None else _finite_float(self.upper, "upper")
        if lower is None and upper is None:
            raise ValueError("constraint must supply lower, upper, or both")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("constraint lower bound cannot exceed upper bound")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "metadata",
            None if self.metadata is None else dict(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression.as_dict(),
            "lower": self.lower,
            "upper": self.upper,
            "kind": self.kind,
            "provenance": self.provenance,
            "description": self.description,
            "verified": self.verified,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class NamedLinearTarget:
    """Named linear target expression optimized over a feasibility set."""

    name: str
    expression: NamedLinearExpression | Mapping[str, float] | str
    label: str | None = None
    unit: str | None = None
    scale: float = 1.0
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target name cannot be empty")
        object.__setattr__(
            self,
            "expression",
            coerce_named_linear_expression(self.expression),
        )
        object.__setattr__(self, "scale", _positive_float(self.scale, "scale"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression.as_dict(),
            "label": self.label,
            "unit": self.unit,
            "scale": self.scale,
            "description": self.description,
        }


@dataclass(frozen=True)
class NamedLinearScenario:
    """Named active-constraint scenario for interval evaluation."""

    name: str
    constraints: Sequence[str]
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name cannot be empty")
        object.__setattr__(
            self,
            "constraints",
            tuple(str(name) for name in self.constraints),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "constraints": list(self.constraints),
            "description": self.description,
        }


@dataclass(frozen=True)
class NamedLinearFeasibilityProblem:
    """Linear feasibility problem with named variables, targets, and scenarios."""

    variables: Sequence[NamedLinearVariable | str | Mapping[str, Any]]
    constraints: Sequence[NamedLinearConstraint | Mapping[str, Any]]
    targets: Sequence[NamedLinearTarget | Mapping[str, Any]]
    scenarios: Sequence[NamedLinearScenario | Mapping[str, Any]]
    title: str = "Named Linear Feasibility Report"
    description: str | None = None
    limitations: Sequence[str] = DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS

    def __post_init__(self) -> None:
        variables = tuple(coerce_named_linear_variable(row) for row in self.variables)
        constraints = tuple(
            coerce_named_linear_constraint(row) for row in self.constraints
        )
        targets = tuple(coerce_named_linear_target(row) for row in self.targets)
        scenarios = tuple(coerce_named_linear_scenario(row) for row in self.scenarios)
        if not variables:
            raise ValueError("problem must contain at least one variable")
        if not targets:
            raise ValueError("problem must contain at least one target")
        if not scenarios:
            raise ValueError("problem must contain at least one scenario")

        variable_names = [row.name for row in variables]
        duplicate_variables = _duplicates(variable_names)
        if duplicate_variables:
            raise ValueError(f"duplicate variable names: {duplicate_variables!r}")
        known_variables = set(variable_names)
        constraint_names = [row.name for row in constraints]
        duplicate_constraints = _duplicates(constraint_names)
        if duplicate_constraints:
            raise ValueError(f"duplicate constraint names: {duplicate_constraints!r}")
        known_constraints = set(constraint_names)
        target_names = [row.name for row in targets]
        duplicate_targets = _duplicates(target_names)
        if duplicate_targets:
            raise ValueError(f"duplicate target names: {duplicate_targets!r}")

        for constraint in constraints:
            _validate_expression_variables(
                constraint.expression,
                known_variables,
                owner=f"constraint {constraint.name!r}",
            )
        for target in targets:
            _validate_expression_variables(
                target.expression,
                known_variables,
                owner=f"target {target.name!r}",
            )
        for scenario in scenarios:
            missing = sorted(set(scenario.constraints) - known_constraints)
            if missing:
                raise ValueError(
                    f"scenario {scenario.name!r} references unknown constraints: "
                    f"{missing!r}"
                )

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.variables)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "variables": [row.as_dict() for row in self.variables],
            "constraints": [row.as_dict() for row in self.constraints],
            "targets": [row.as_dict() for row in self.targets],
            "scenarios": [row.as_dict() for row in self.scenarios],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class NamedLinearConstraintDiagnostic:
    """One endpoint-side diagnostic for an active linear constraint."""

    scenario: str
    target: str
    endpoint: str
    constraint: str
    side: str
    kind: str
    provenance: str | None
    expression_value: float
    bound: float
    slack: float
    binding: bool
    solver_marginal: float | None
    target_marginal: float | None
    dual_magnitude: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target": self.target,
            "endpoint": self.endpoint,
            "constraint": self.constraint,
            "side": self.side,
            "kind": self.kind,
            "provenance": self.provenance,
            "expression_value": self.expression_value,
            "bound": self.bound,
            "slack": self.slack,
            "binding": self.binding,
            "solver_marginal": self.solver_marginal,
            "target_marginal": self.target_marginal,
            "dual_magnitude": self.dual_magnitude,
        }


@dataclass(frozen=True)
class NamedLinearEndpoint:
    """One endpoint solve for a target under a scenario."""

    scenario: str
    target: str
    sense: str
    status: str
    value: float | None = None
    scaled_value: float | None = None
    assignment: Mapping[str, float] | None = None
    binding_constraints: tuple[str, ...] = ()
    binding_constraint_sides: tuple[str, ...] = ()
    constraint_diagnostics: tuple[NamedLinearConstraintDiagnostic, ...] = ()
    solver_status: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_constraints", tuple(self.binding_constraints))
        object.__setattr__(
            self,
            "binding_constraint_sides",
            tuple(self.binding_constraint_sides),
        )
        object.__setattr__(
            self,
            "constraint_diagnostics",
            tuple(self.constraint_diagnostics),
        )
        if self.assignment is not None:
            object.__setattr__(self, "assignment", dict(self.assignment))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target": self.target,
            "sense": self.sense,
            "status": self.status,
            "value": self.value,
            "scaled_value": self.scaled_value,
            "assignment": self.assignment,
            "binding_constraints": self.binding_constraints,
            "binding_constraint_sides": self.binding_constraint_sides,
            "constraint_diagnostics": [
                row.as_dict() for row in self.constraint_diagnostics
            ],
            "solver_status": self.solver_status,
            "message": self.message,
        }


@dataclass(frozen=True)
class NamedLinearInterval:
    """Feasible target interval under one named scenario."""

    scenario: str
    target: str
    lower: float | None
    upper: float | None
    scaled_lower: float | None
    scaled_upper: float | None
    status: str
    lower_endpoint: NamedLinearEndpoint
    upper_endpoint: NamedLinearEndpoint

    @property
    def width(self) -> float | None:
        if self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower

    @property
    def scaled_width(self) -> float | None:
        if self.scaled_lower is None or self.scaled_upper is None:
            return None
        return self.scaled_upper - self.scaled_lower

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "target": self.target,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "scaled_lower": self.scaled_lower,
            "scaled_upper": self.scaled_upper,
            "scaled_width": self.scaled_width,
            "status": self.status,
            "lower_endpoint": self.lower_endpoint.as_dict(),
            "upper_endpoint": self.upper_endpoint.as_dict(),
        }


@dataclass(frozen=True)
class NamedLinearConstraintAttribution:
    """Effect of relaxing one active constraint group for one interval."""

    target: str
    scenario: str
    group: str
    constraints: tuple[str, ...]
    constraint_count: int
    kind: str | None
    provenance: str | None
    verified_count: int
    full_lower: float | None
    full_upper: float | None
    full_width: float | None
    relaxed_lower: float | None
    relaxed_upper: float | None
    relaxed_width: float | None
    relaxed_status: str
    lower_tightening: float | None
    upper_tightening: float | None
    width_increase: float | None
    width_increase_percent: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "scenario": self.scenario,
            "group": self.group,
            "constraints": self.constraints,
            "constraint_count": self.constraint_count,
            "kind": self.kind,
            "provenance": self.provenance,
            "verified_count": self.verified_count,
            "full_lower": self.full_lower,
            "full_upper": self.full_upper,
            "full_width": self.full_width,
            "relaxed_lower": self.relaxed_lower,
            "relaxed_upper": self.relaxed_upper,
            "relaxed_width": self.relaxed_width,
            "relaxed_status": self.relaxed_status,
            "lower_tightening": self.lower_tightening,
            "upper_tightening": self.upper_tightening,
            "width_increase": self.width_increase,
            "width_increase_percent": self.width_increase_percent,
        }


@dataclass(frozen=True)
class NamedLinearConstraintAttributionReport:
    """Leave-one-group-out interval attribution for a named linear report."""

    source_report: "NamedLinearFeasibilityReport"
    target: str
    scenario: str
    group_by: str
    baseline_interval: NamedLinearInterval
    rows: tuple[NamedLinearConstraintAttribution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))

    @property
    def title(self) -> str:
        return "Named Linear Constraint Attribution"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "target": self.target,
            "scenario": self.scenario,
            "group_by": self.group_by,
            "baseline_interval": self.baseline_interval.as_dict(),
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    "title": self.title,
                    "target": self.target,
                    "scenario": self.scenario,
                    "group_by": self.group_by,
                    "baseline_lower": self.baseline_interval.lower,
                    "baseline_upper": self.baseline_interval.upper,
                    "baseline_width": self.baseline_interval.width,
                    "row_count": len(self.rows),
                },
            ),
            "constraint_attribution": tuple(row.as_dict() for row in self.rows),
        }

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Target: `{_escape_markdown(self.target)}`",
            f"- Scenario: `{_escape_markdown(self.scenario)}`",
            f"- Grouping: `{_escape_markdown(self.group_by)}`",
            "",
            "Each row removes one active constraint group, re-solves the target "
            "interval, and reports how much wider the interval becomes. Larger "
            "width increases indicate constraints that do more work in the "
            "encoded feasibility problem.",
            "",
            "## Ranked Constraint Values",
            "",
        ]
        lines.extend(_attribution_table(self.rows))
        return "\n".join(lines)


@dataclass(frozen=True)
class NamedLinearFeasibilityReport:
    """Interval report for a named linear feasibility problem."""

    problem: NamedLinearFeasibilityProblem
    intervals: tuple[NamedLinearInterval, ...]
    backend: str = "scipy-linprog"

    @property
    def title(self) -> str:
        return self.problem.title

    def interval(self, *, target: str, scenario: str) -> NamedLinearInterval:
        for row in self.intervals:
            if row.target == target and row.scenario == scenario:
                return row
        raise KeyError(f"unknown interval target={target!r}, scenario={scenario!r}")

    def width_reduction(
        self,
        *,
        target: str,
        baseline_scenario: str,
        comparison_scenario: str,
    ) -> dict[str, Any]:
        baseline = self.interval(target=target, scenario=baseline_scenario)
        comparison = self.interval(target=target, scenario=comparison_scenario)
        reduction = None
        reduction_percent = None
        if baseline.width is not None and comparison.width is not None:
            reduction = baseline.width - comparison.width
            reduction_percent = (
                None
                if baseline.width <= 0.0
                else 100.0 * reduction / baseline.width
            )
        return {
            "target": target,
            "baseline_scenario": baseline_scenario,
            "comparison_scenario": comparison_scenario,
            "baseline_width": baseline.width,
            "comparison_width": comparison.width,
            "width_reduction": reduction,
            "width_reduction_percent": reduction_percent,
            "baseline_scaled_width": baseline.scaled_width,
            "comparison_scaled_width": comparison.scaled_width,
        }

    def attribute_constraints(
        self,
        *,
        target: str,
        scenario: str,
        group_by: str = "constraint",
        groups: Mapping[str, Sequence[str]] | None = None,
        top: int | None = None,
    ) -> NamedLinearConstraintAttributionReport:
        """Rank active constraints by leave-one-group-out interval widening."""

        return attribute_named_linear_constraints(
            self,
            target=target,
            scenario=scenario,
            group_by=group_by,
            groups=groups,
            top=top,
        )

    def audit_claim(
        self,
        claim: "NamedLinearClaim",
    ) -> "NamedLinearClaimAudit":
        """Audit a named-linear claim against this interval report."""

        return audit_named_linear_claim(self, claim)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "backend": self.backend,
            "problem": self.problem.as_dict(),
            "intervals": [row.as_dict() for row in self.intervals],
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        constraint_lookup = {row.name: row for row in self.problem.constraints}
        target_lookup = {row.name: row for row in self.problem.targets}
        return {
            "summary": (
                {
                    "title": self.title,
                    "backend": self.backend,
                    "description": self.problem.description,
                    "variable_count": len(self.problem.variables),
                    "constraint_count": len(self.problem.constraints),
                    "target_count": len(self.problem.targets),
                    "scenario_count": len(self.problem.scenarios),
                    "interval_count": len(self.intervals),
                },
            ),
            "variables": tuple(row.as_dict() for row in self.problem.variables),
            "constraints": tuple(row.as_dict() for row in self.problem.constraints),
            "targets": tuple(row.as_dict() for row in self.problem.targets),
            "scenarios": tuple(row.as_dict() for row in self.problem.scenarios),
            "active_constraints": tuple(
                {
                    "scenario": scenario.name,
                    "constraint": name,
                    "kind": constraint_lookup[name].kind,
                    "provenance": constraint_lookup[name].provenance,
                    "verified": constraint_lookup[name].verified,
                }
                for scenario in self.problem.scenarios
                for name in scenario.constraints
            ),
            "intervals": tuple(
                {
                    "scenario": row.scenario,
                    "target": row.target,
                    "target_label": target_lookup[row.target].label,
                    "unit": target_lookup[row.target].unit,
                    "lower": row.lower,
                    "upper": row.upper,
                    "width": row.width,
                    "scaled_lower": row.scaled_lower,
                    "scaled_upper": row.scaled_upper,
                    "scaled_width": row.scaled_width,
                    "status": row.status,
                }
                for row in self.intervals
            ),
            "endpoints": tuple(
                endpoint.as_dict()
                for row in self.intervals
                for endpoint in (row.lower_endpoint, row.upper_endpoint)
            ),
            "endpoint_constraint_diagnostics": tuple(
                diagnostic.as_dict()
                for row in self.intervals
                for endpoint in (row.lower_endpoint, row.upper_endpoint)
                for diagnostic in endpoint.constraint_diagnostics
            ),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.problem.limitations
            ),
        }

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        lines = [f"# {_escape_markdown(self.title)}", ""]
        if self.problem.description:
            lines.extend([self.problem.description, ""])
        lines.extend(
            [
                "## Interpretation",
                "",
                "Each interval is the minimum and maximum value of a linear "
                "target expression over the active linear constraints in that "
                "scenario. This is a feasibility bound, not a point estimate.",
                "",
                "## Feasible Intervals",
                "",
            ]
        )
        lines.extend(_interval_table(self))
        lines.extend(["", "## Scenarios", ""])
        for scenario in self.problem.scenarios:
            lines.append(f"### {_escape_markdown(scenario.name)}")
            if scenario.description:
                lines.extend(["", scenario.description])
            counts = _constraint_kind_counts(scenario, self.problem.constraints)
            lines.extend(
                [
                    "",
                    f"- Active constraints: {len(scenario.constraints)}",
                    "- Constraint kinds: "
                    + ", ".join(
                        f"{kind}={count}" for kind, count in sorted(counts.items())
                    ),
                    "",
                ]
            )
        lines.extend(["## Binding Endpoint Constraints", ""])
        lines.extend(_binding_table(self))
        lines.extend(["", "## Dual / Binding Constraint Diagnostics", ""])
        lines.extend(_dual_diagnostic_table(self))
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.problem.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class NamedLinearClaim:
    """Bound claim about a target interval under one named scenario."""

    target: str
    scenario: str
    lower_at_least: float | None = None
    upper_at_most: float | None = None
    label: str | None = None
    description: str | None = None
    attribution_top: int = 5
    diagnostic_top: int = 8

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("claim target cannot be empty")
        if not self.scenario:
            raise ValueError("claim scenario cannot be empty")
        lower = (
            None
            if self.lower_at_least is None
            else _finite_float(self.lower_at_least, "lower_at_least")
        )
        upper = (
            None
            if self.upper_at_most is None
            else _finite_float(self.upper_at_most, "upper_at_most")
        )
        if lower is None and upper is None:
            raise ValueError("claim must specify lower_at_least or upper_at_most")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("claim lower_at_least cannot exceed upper_at_most")
        object.__setattr__(self, "lower_at_least", lower)
        object.__setattr__(self, "upper_at_most", upper)
        object.__setattr__(
            self,
            "attribution_top",
            _positive_int(self.attribution_top, "attribution_top"),
        )
        object.__setattr__(
            self,
            "diagnostic_top",
            _positive_int(self.diagnostic_top, "diagnostic_top"),
        )

    def audit(self, report: NamedLinearFeasibilityReport) -> "NamedLinearClaimAudit":
        """Audit this claim against a named-linear feasibility report."""

        return audit_named_linear_claim(report, self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "scenario": self.scenario,
            "lower_at_least": self.lower_at_least,
            "upper_at_most": self.upper_at_most,
            "label": self.label,
            "description": self.description,
            "attribution_top": self.attribution_top,
            "diagnostic_top": self.diagnostic_top,
        }

    @property
    def statement(self) -> str:
        parts = []
        if self.lower_at_least is not None:
            parts.append(f"{self.target} >= {self.lower_at_least:g}")
        if self.upper_at_most is not None:
            parts.append(f"{self.target} <= {self.upper_at_most:g}")
        return " and ".join(parts)


@dataclass(frozen=True)
class NamedLinearClaimAudit:
    """Review artifact for a named-linear interval claim."""

    claim: NamedLinearClaim
    interval: NamedLinearInterval
    verdict: str
    support_margin: float | None
    condition_rows: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    attribution: NamedLinearConstraintAttributionReport | None = None
    endpoint_diagnostics: tuple[NamedLinearConstraintDiagnostic, ...] = ()
    source_title: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_rows", tuple(self.condition_rows))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(
            self,
            "endpoint_diagnostics",
            tuple(self.endpoint_diagnostics),
        )

    @property
    def title(self) -> str:
        return "Named Linear Claim Audit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source_title": self.source_title,
            "claim": self.claim.as_dict(),
            "statement": self.claim.statement,
            "interval": self.interval.as_dict(),
            "verdict": self.verdict,
            "support_margin": self.support_margin,
            "condition_rows": list(self.condition_rows),
            "reasons": list(self.reasons),
            "attribution": None
            if self.attribution is None
            else self.attribution.as_dict(),
            "endpoint_diagnostics": [
                row.as_dict() for row in self.endpoint_diagnostics
            ],
        }

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "claim_summary": (
                {
                    "title": self.title,
                    "source_title": self.source_title,
                    "target": self.claim.target,
                    "scenario": self.claim.scenario,
                    "statement": self.claim.statement,
                    "verdict": self.verdict,
                    "lower": self.interval.lower,
                    "upper": self.interval.upper,
                    "width": self.interval.width,
                    "status": self.interval.status,
                    "support_margin": self.support_margin,
                },
            ),
            "claim_conditions": self.condition_rows,
            "claim_reasons": tuple({"reason": reason} for reason in self.reasons),
            "claim_attribution": ()
            if self.attribution is None
            else tuple(row.as_dict() for row in self.attribution.rows),
            "claim_endpoint_diagnostics": tuple(
                row.as_dict() for row in self.endpoint_diagnostics
            ),
        }

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Verdict: **{self.verdict}**",
            f"- Claim: `{_escape_markdown(self.claim.statement)}`",
            f"- Target: `{_escape_markdown(self.claim.target)}`",
            f"- Scenario: `{_escape_markdown(self.claim.scenario)}`",
            "- Feasible interval: "
            f"[{_format_optional(self.interval.lower, '')}, "
            f"{_format_optional(self.interval.upper, '')}]",
        ]
        if self.support_margin is not None:
            lines.append(
                f"- Margin to failure: {_format_optional(self.support_margin, '')}"
            )
        if self.claim.description:
            lines.extend(["", self.claim.description])
        lines.extend(["", "## Condition Checks", ""])
        lines.extend(_claim_condition_table(self.condition_rows))
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in self.reasons)
        if self.attribution is not None:
            lines.extend(["", "## Constraint Attribution", ""])
            lines.extend(_attribution_table(self.attribution.rows))
        lines.extend(["", "## Endpoint Binding / Dual Diagnostics", ""])
        lines.extend(_diagnostic_rows_table(self.endpoint_diagnostics))
        lines.extend(
            [
                "",
                "## Limitations",
                "",
                "- This is a feasibility claim audit, not a point estimate or "
                "confidence interval.",
                "- The verdict is conditional on the supplied variables, linear "
                "constraints, target expression, and active scenario.",
            ]
        )
        return "\n".join(lines)


def named_linear_variable(
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    label: str | None = None,
    unit: str | None = None,
    description: str | None = None,
) -> NamedLinearVariable:
    return NamedLinearVariable(
        name=name,
        lower=lower,
        upper=upper,
        label=label,
        unit=unit,
        description=description,
    )


def named_linear_expression(
    coefficients: NamedLinearExpression | Mapping[str, float] | str,
    *,
    constant: float = 0.0,
    label: str | None = None,
) -> NamedLinearExpression:
    return coerce_named_linear_expression(
        coefficients,
        constant=constant,
        label=label,
    )


def named_linear_constraint(
    name: str,
    coefficients: NamedLinearExpression | Mapping[str, float] | str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    constant: float = 0.0,
    kind: str = "linear",
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NamedLinearConstraint:
    return NamedLinearConstraint(
        name=name,
        expression=named_linear_expression(coefficients, constant=constant),
        lower=lower,
        upper=upper,
        kind=kind,
        provenance=provenance,
        description=description,
        verified=verified,
        metadata=metadata,
    )


def named_linear_target(
    name: str,
    expression: NamedLinearExpression | Mapping[str, float] | str,
    *,
    label: str | None = None,
    unit: str | None = None,
    scale: float = 1.0,
    description: str | None = None,
) -> NamedLinearTarget:
    return NamedLinearTarget(
        name=name,
        expression=expression,
        label=label,
        unit=unit,
        scale=scale,
        description=description,
    )


def named_linear_scenario(
    name: str,
    constraints: Sequence[str],
    *,
    description: str | None = None,
) -> NamedLinearScenario:
    return NamedLinearScenario(
        name=name,
        constraints=constraints,
        description=description,
    )


def named_linear_feasibility_problem(
    *,
    variables: Sequence[NamedLinearVariable | str | Mapping[str, Any]],
    constraints: Sequence[NamedLinearConstraint | Mapping[str, Any]],
    targets: Sequence[NamedLinearTarget | Mapping[str, Any]],
    scenarios: Sequence[NamedLinearScenario | Mapping[str, Any]],
    title: str = "Named Linear Feasibility Report",
    description: str | None = None,
    limitations: Sequence[str] = DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS,
) -> NamedLinearFeasibilityProblem:
    return NamedLinearFeasibilityProblem(
        variables=variables,
        constraints=constraints,
        targets=targets,
        scenarios=scenarios,
        title=title,
        description=description,
        limitations=limitations,
    )


def named_linear_claim(
    *,
    target: str,
    scenario: str,
    lower_at_least: float | None = None,
    upper_at_most: float | None = None,
    label: str | None = None,
    description: str | None = None,
    attribution_top: int = 5,
    diagnostic_top: int = 8,
) -> NamedLinearClaim:
    return NamedLinearClaim(
        target=target,
        scenario=scenario,
        lower_at_least=lower_at_least,
        upper_at_most=upper_at_most,
        label=label,
        description=description,
        attribution_top=attribution_top,
        diagnostic_top=diagnostic_top,
    )


def solve_named_linear_feasibility(
    problem: NamedLinearFeasibilityProblem | Mapping[str, Any],
) -> NamedLinearFeasibilityReport:
    """Evaluate all target intervals under all named linear scenarios."""

    if not isinstance(problem, NamedLinearFeasibilityProblem):
        problem = NamedLinearFeasibilityProblem(**dict(problem))
    intervals: list[NamedLinearInterval] = []
    constraints_by_name = {row.name: row for row in problem.constraints}
    for scenario in problem.scenarios:
        active_constraints = tuple(
            constraints_by_name[name] for name in scenario.constraints
        )
        for target in problem.targets:
            lower_endpoint = _solve_endpoint(
                problem,
                scenario=scenario,
                constraints=active_constraints,
                target=target,
                sense="min",
            )
            upper_endpoint = _solve_endpoint(
                problem,
                scenario=scenario,
                constraints=active_constraints,
                target=target,
                sense="max",
            )
            intervals.append(
                NamedLinearInterval(
                    scenario=scenario.name,
                    target=target.name,
                    lower=lower_endpoint.value,
                    upper=upper_endpoint.value,
                    scaled_lower=lower_endpoint.scaled_value,
                    scaled_upper=upper_endpoint.scaled_value,
                    status=_interval_status(lower_endpoint, upper_endpoint),
                    lower_endpoint=lower_endpoint,
                    upper_endpoint=upper_endpoint,
                )
            )
    return NamedLinearFeasibilityReport(problem=problem, intervals=tuple(intervals))


def audit_named_linear_claim(
    report: NamedLinearFeasibilityReport,
    claim: NamedLinearClaim | Mapping[str, Any],
) -> NamedLinearClaimAudit:
    """Audit whether a named-linear interval report supports a bound claim."""

    if not isinstance(claim, NamedLinearClaim):
        claim = NamedLinearClaim(**dict(claim))
    interval = report.interval(target=claim.target, scenario=claim.scenario)
    condition_rows = tuple(_claim_condition_rows(claim, interval))
    verdict = _claim_verdict(interval, condition_rows)
    support_margin = _claim_support_margin(verdict, condition_rows)
    attribution = None
    if interval.status == "bounded":
        attribution = report.attribute_constraints(
            target=claim.target,
            scenario=claim.scenario,
            top=claim.attribution_top,
        )
    endpoint_diagnostics = _claim_endpoint_diagnostics(
        claim,
        interval,
        top=claim.diagnostic_top,
    )
    return NamedLinearClaimAudit(
        claim=claim,
        interval=interval,
        verdict=verdict,
        support_margin=support_margin,
        condition_rows=condition_rows,
        reasons=tuple(_claim_reasons(verdict, interval, condition_rows)),
        attribution=attribution,
        endpoint_diagnostics=endpoint_diagnostics,
        source_title=report.title,
    )


def attribute_named_linear_constraints(
    report: NamedLinearFeasibilityReport,
    *,
    target: str,
    scenario: str,
    group_by: str = "constraint",
    groups: Mapping[str, Sequence[str]] | None = None,
    top: int | None = None,
) -> NamedLinearConstraintAttributionReport:
    """Rank active constraints by how much they narrow one target interval.

    The attribution is local to the supplied target, scenario, and grouping. It
    removes each active constraint group, re-solves the interval, and measures
    how much wider the feasible interval becomes.
    """

    if top is not None and top <= 0:
        raise ValueError("top must be positive when supplied")
    problem = report.problem
    scenario_row = _scenario_by_name(problem, scenario)
    target_row = _target_by_name(problem, target)
    baseline = report.interval(target=target, scenario=scenario)
    constraint_lookup = {row.name: row for row in problem.constraints}
    group_map = _constraint_group_map(
        scenario_row,
        constraint_lookup,
        group_by=group_by,
        groups=groups,
    )
    rows: list[NamedLinearConstraintAttribution] = []
    active_names = tuple(scenario_row.constraints)
    for group, removed_names in group_map.items():
        relaxed_names = tuple(name for name in active_names if name not in removed_names)
        relaxed_scenario = NamedLinearScenario(
            name=f"{scenario} without {group}",
            constraints=relaxed_names,
        )
        relaxed_constraints = tuple(constraint_lookup[name] for name in relaxed_names)
        relaxed = _solve_interval(
            problem,
            scenario=relaxed_scenario,
            constraints=relaxed_constraints,
            target=target_row,
        )
        lower_tightening = None
        upper_tightening = None
        width_increase = None
        width_increase_percent = None
        if baseline.lower is not None and relaxed.lower is not None:
            lower_tightening = baseline.lower - relaxed.lower
        if baseline.upper is not None and relaxed.upper is not None:
            upper_tightening = relaxed.upper - baseline.upper
        if baseline.width is not None and relaxed.width is not None:
            width_increase = relaxed.width - baseline.width
            width_increase_percent = (
                None
                if baseline.width <= 0.0
                else 100.0 * width_increase / baseline.width
            )
        removed_constraints = tuple(constraint_lookup[name] for name in removed_names)
        rows.append(
            NamedLinearConstraintAttribution(
                target=target,
                scenario=scenario,
                group=group,
                constraints=tuple(removed_names),
                constraint_count=len(removed_names),
                kind=_common_value(row.kind for row in removed_constraints),
                provenance=_common_value(
                    row.provenance for row in removed_constraints
                ),
                verified_count=sum(1 for row in removed_constraints if row.verified),
                full_lower=baseline.lower,
                full_upper=baseline.upper,
                full_width=baseline.width,
                relaxed_lower=relaxed.lower,
                relaxed_upper=relaxed.upper,
                relaxed_width=relaxed.width,
                relaxed_status=relaxed.status,
                lower_tightening=lower_tightening,
                upper_tightening=upper_tightening,
                width_increase=width_increase,
                width_increase_percent=width_increase_percent,
            )
        )
    rows.sort(key=lambda row: (-_attribution_rank_value(row), row.group))
    if top is not None:
        rows = rows[:top]
    return NamedLinearConstraintAttributionReport(
        source_report=report,
        target=target,
        scenario=scenario,
        group_by=group_by,
        baseline_interval=baseline,
        rows=tuple(rows),
    )


def coerce_named_linear_variable(
    value: NamedLinearVariable | str | Mapping[str, Any],
) -> NamedLinearVariable:
    if isinstance(value, NamedLinearVariable):
        return value
    if isinstance(value, str):
        return NamedLinearVariable(name=value)
    if isinstance(value, Mapping):
        return NamedLinearVariable(**dict(value))
    raise TypeError("variables must be NamedLinearVariable, string, or mapping")


def coerce_named_linear_expression(
    value: NamedLinearExpression | Mapping[str, float] | str,
    *,
    constant: float = 0.0,
    label: str | None = None,
) -> NamedLinearExpression:
    if isinstance(value, NamedLinearExpression):
        return value
    if isinstance(value, str):
        return NamedLinearExpression({value: 1.0}, constant=constant, label=label)
    if isinstance(value, Mapping):
        if "coefficients" in value:
            payload = dict(value)
            return NamedLinearExpression(
                payload["coefficients"],
                constant=float(payload.get("constant", constant)),
                label=payload.get("label", label),
            )
        return NamedLinearExpression(value, constant=constant, label=label)
    raise TypeError("expression must be NamedLinearExpression, mapping, or string")


def coerce_named_linear_constraint(
    value: NamedLinearConstraint | Mapping[str, Any],
) -> NamedLinearConstraint:
    if isinstance(value, NamedLinearConstraint):
        return value
    if isinstance(value, Mapping):
        return NamedLinearConstraint(**dict(value))
    raise TypeError("constraints must be NamedLinearConstraint or mapping")


def coerce_named_linear_target(
    value: NamedLinearTarget | Mapping[str, Any],
) -> NamedLinearTarget:
    if isinstance(value, NamedLinearTarget):
        return value
    if isinstance(value, Mapping):
        return NamedLinearTarget(**dict(value))
    raise TypeError("targets must be NamedLinearTarget or mapping")


def coerce_named_linear_scenario(
    value: NamedLinearScenario | Mapping[str, Any],
) -> NamedLinearScenario:
    if isinstance(value, NamedLinearScenario):
        return value
    if isinstance(value, Mapping):
        return NamedLinearScenario(**dict(value))
    raise TypeError("scenarios must be NamedLinearScenario or mapping")


@dataclass(frozen=True)
class _InequalityRow:
    constraint: NamedLinearConstraint
    side: str
    coefficients: tuple[float, ...]
    rhs: float
    bound: float


def _solve_endpoint(
    problem: NamedLinearFeasibilityProblem,
    *,
    scenario: NamedLinearScenario,
    constraints: Sequence[NamedLinearConstraint],
    target: NamedLinearTarget,
    sense: str,
) -> NamedLinearEndpoint:
    variables = tuple(problem.variables)
    index = {variable.name: i for i, variable in enumerate(variables)}
    objective = [0.0] * len(variables)
    for name, coefficient in target.expression.coefficients.items():
        objective[index[name]] = coefficient if sense == "min" else -coefficient

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    inequality_rows: list[_InequalityRow] = []
    for constraint in constraints:
        row = _coefficient_row(constraint.expression, index)
        constant = constraint.expression.constant
        if constraint.upper is not None:
            a_ub.append(row)
            b_ub.append(constraint.upper - constant)
            inequality_rows.append(
                _InequalityRow(
                    constraint=constraint,
                    side="upper",
                    coefficients=tuple(row),
                    rhs=constraint.upper - constant,
                    bound=constraint.upper,
                )
            )
        if constraint.lower is not None:
            lower_row = [-value for value in row]
            a_ub.append(lower_row)
            b_ub.append(constant - constraint.lower)
            inequality_rows.append(
                _InequalityRow(
                    constraint=constraint,
                    side="lower",
                    coefficients=tuple(lower_row),
                    rhs=constant - constraint.lower,
                    bound=constraint.lower,
                )
            )

    result = linprog(
        objective,
        A_ub=a_ub if a_ub else None,
        b_ub=b_ub if b_ub else None,
        bounds=[(variable.lower, variable.upper) for variable in variables],
        method="highs",
    )
    if not result.success:
        return NamedLinearEndpoint(
            scenario=scenario.name,
            target=target.name,
            sense=sense,
            status=_endpoint_status(result.status),
            solver_status=int(result.status),
            message=str(result.message),
        )

    assignment = {
        variable.name: float(result.x[index[variable.name]]) for variable in variables
    }
    value = target.expression.evaluate(assignment)
    diagnostics = _constraint_diagnostics(
        scenario=scenario,
        target=target,
        sense=sense,
        assignment=assignment,
        rows=inequality_rows,
        marginals=_inequality_marginals(result),
    )
    return NamedLinearEndpoint(
        scenario=scenario.name,
        target=target.name,
        sense=sense,
        status="optimal",
        value=value,
        scaled_value=value / target.scale,
        assignment=assignment,
        binding_constraints=_binding_constraints(diagnostics),
        binding_constraint_sides=_binding_constraint_sides(diagnostics),
        constraint_diagnostics=diagnostics,
        solver_status=int(result.status),
        message=str(result.message),
    )


def _solve_interval(
    problem: NamedLinearFeasibilityProblem,
    *,
    scenario: NamedLinearScenario,
    constraints: Sequence[NamedLinearConstraint],
    target: NamedLinearTarget,
) -> NamedLinearInterval:
    lower_endpoint = _solve_endpoint(
        problem,
        scenario=scenario,
        constraints=constraints,
        target=target,
        sense="min",
    )
    upper_endpoint = _solve_endpoint(
        problem,
        scenario=scenario,
        constraints=constraints,
        target=target,
        sense="max",
    )
    return NamedLinearInterval(
        scenario=scenario.name,
        target=target.name,
        lower=lower_endpoint.value,
        upper=upper_endpoint.value,
        scaled_lower=lower_endpoint.scaled_value,
        scaled_upper=upper_endpoint.scaled_value,
        status=_interval_status(lower_endpoint, upper_endpoint),
        lower_endpoint=lower_endpoint,
        upper_endpoint=upper_endpoint,
    )


def _coefficient_row(
    expression: NamedLinearExpression,
    index: Mapping[str, int],
) -> list[float]:
    row = [0.0] * len(index)
    for name, coefficient in expression.coefficients.items():
        row[index[name]] = coefficient
    return row


def _constraint_diagnostics(
    *,
    scenario: NamedLinearScenario,
    target: NamedLinearTarget,
    sense: str,
    assignment: Mapping[str, float],
    rows: Sequence[_InequalityRow],
    marginals: Sequence[float | None],
    tol: float = 1e-7,
) -> tuple[NamedLinearConstraintDiagnostic, ...]:
    diagnostics: list[NamedLinearConstraintDiagnostic] = []
    for index, row in enumerate(rows):
        expression_value = row.constraint.expression.evaluate(assignment)
        if row.side == "upper":
            slack = row.bound - expression_value
        else:
            slack = expression_value - row.bound
        solver_marginal = marginals[index] if index < len(marginals) else None
        target_marginal = None
        dual_magnitude = None
        if solver_marginal is not None:
            target_marginal = solver_marginal if sense == "min" else -solver_marginal
            dual_magnitude = abs(target_marginal)
        diagnostics.append(
            NamedLinearConstraintDiagnostic(
                scenario=scenario.name,
                target=target.name,
                endpoint=sense,
                constraint=row.constraint.name,
                side=row.side,
                kind=row.constraint.kind,
                provenance=row.constraint.provenance,
                expression_value=expression_value,
                bound=row.bound,
                slack=max(0.0, slack) if abs(slack) <= tol else slack,
                binding=abs(slack) <= tol,
                solver_marginal=solver_marginal,
                target_marginal=target_marginal,
                dual_magnitude=dual_magnitude,
            )
        )
    return tuple(diagnostics)


def _inequality_marginals(result: Any) -> tuple[float | None, ...]:
    ineqlin = getattr(result, "ineqlin", None)
    marginals = getattr(ineqlin, "marginals", None)
    if marginals is None:
        return ()
    return tuple(float(value) for value in marginals)


def _binding_constraints(
    diagnostics: Sequence[NamedLinearConstraintDiagnostic],
) -> tuple[str, ...]:
    names: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic.binding and diagnostic.constraint not in names:
            names.append(diagnostic.constraint)
    return tuple(names)


def _binding_constraint_sides(
    diagnostics: Sequence[NamedLinearConstraintDiagnostic],
) -> tuple[str, ...]:
    return tuple(
        f"{diagnostic.constraint}:{diagnostic.side}"
        for diagnostic in diagnostics
        if diagnostic.binding
    )


def _endpoint_status(status: int) -> str:
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    return "failed"


def _interval_status(
    lower: NamedLinearEndpoint,
    upper: NamedLinearEndpoint,
) -> str:
    if lower.status == "optimal" and upper.status == "optimal":
        return "bounded"
    if lower.status == "infeasible" or upper.status == "infeasible":
        return "infeasible"
    if lower.status == "unbounded" or upper.status == "unbounded":
        return "unbounded"
    return "failed"


def _validate_expression_variables(
    expression: NamedLinearExpression,
    known_variables: set[str],
    *,
    owner: str,
) -> None:
    missing = sorted(set(expression.coefficients) - known_variables)
    if missing:
        raise ValueError(f"{owner} references unknown variables: {missing!r}")


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _scenario_by_name(
    problem: NamedLinearFeasibilityProblem,
    name: str,
) -> NamedLinearScenario:
    for scenario in problem.scenarios:
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown scenario: {name!r}")


def _target_by_name(
    problem: NamedLinearFeasibilityProblem,
    name: str,
) -> NamedLinearTarget:
    for target in problem.targets:
        if target.name == name:
            return target
    raise KeyError(f"unknown target: {name!r}")


def _constraint_group_map(
    scenario: NamedLinearScenario,
    constraint_lookup: Mapping[str, NamedLinearConstraint],
    *,
    group_by: str,
    groups: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    active = set(scenario.constraints)
    if groups is not None:
        mapped = {
            str(group): tuple(str(name) for name in names)
            for group, names in groups.items()
        }
        for group, names in mapped.items():
            missing = sorted(set(names) - active)
            if missing:
                raise ValueError(
                    f"group {group!r} references constraints not active in "
                    f"scenario {scenario.name!r}: {missing!r}"
                )
            if not names:
                raise ValueError(f"group {group!r} must contain constraints")
        return mapped
    if group_by == "constraint":
        return {name: (name,) for name in scenario.constraints}
    grouped: dict[str, list[str]] = {}
    if group_by == "kind":
        for name in scenario.constraints:
            group = constraint_lookup[name].kind
            grouped.setdefault(group, []).append(name)
        return {group: tuple(names) for group, names in grouped.items()}
    if group_by == "provenance":
        for name in scenario.constraints:
            group = constraint_lookup[name].provenance or "unprovenanced"
            grouped.setdefault(group, []).append(name)
        return {group: tuple(names) for group, names in grouped.items()}
    raise ValueError(
        "group_by must be 'constraint', 'kind', or 'provenance' when groups "
        "is not supplied"
    )


def _common_value(values: Iterable[str | None]) -> str | None:
    unique = {value for value in values if value is not None}
    if len(unique) == 1:
        return next(iter(unique))
    if len(unique) > 1:
        return "mixed"
    return None


def _attribution_rank_value(row: NamedLinearConstraintAttribution) -> float:
    if row.width_increase is not None:
        return row.width_increase
    if row.relaxed_status == "unbounded":
        return float("inf")
    return float("-inf")


def _claim_condition_rows(
    claim: NamedLinearClaim,
    interval: NamedLinearInterval,
) -> list[dict[str, Any]]:
    rows = []
    if claim.lower_at_least is not None:
        threshold = claim.lower_at_least
        status = "inconclusive"
        margin = None
        if interval.status == "bounded":
            if interval.lower is not None and interval.lower >= threshold:
                status = "pass"
                margin = interval.lower - threshold
            elif interval.upper is not None and interval.upper < threshold:
                status = "fail"
                margin = interval.upper - threshold
            elif interval.lower is not None:
                margin = interval.lower - threshold
        rows.append(
            {
                "condition": f"{claim.target} >= {threshold:g}",
                "type": "lower_at_least",
                "threshold": threshold,
                "status": status,
                "margin": margin,
                "certifying_endpoint": "lower",
                "endpoint_value": interval.lower,
                "opposite_endpoint_value": interval.upper,
            }
        )
    if claim.upper_at_most is not None:
        threshold = claim.upper_at_most
        status = "inconclusive"
        margin = None
        if interval.status == "bounded":
            if interval.upper is not None and interval.upper <= threshold:
                status = "pass"
                margin = threshold - interval.upper
            elif interval.lower is not None and interval.lower > threshold:
                status = "fail"
                margin = threshold - interval.lower
            elif interval.upper is not None:
                margin = threshold - interval.upper
        rows.append(
            {
                "condition": f"{claim.target} <= {threshold:g}",
                "type": "upper_at_most",
                "threshold": threshold,
                "status": status,
                "margin": margin,
                "certifying_endpoint": "upper",
                "endpoint_value": interval.upper,
                "opposite_endpoint_value": interval.lower,
            }
        )
    return rows


def _claim_verdict(
    interval: NamedLinearInterval,
    condition_rows: Sequence[Mapping[str, Any]],
) -> str:
    if interval.status != "bounded":
        return "inconclusive"
    statuses = {str(row["status"]) for row in condition_rows}
    if "fail" in statuses:
        return "fail"
    if statuses == {"pass"}:
        return "pass"
    return "inconclusive"


def _claim_support_margin(
    verdict: str,
    condition_rows: Sequence[Mapping[str, Any]],
) -> float | None:
    if verdict != "pass":
        return None
    margins = [row["margin"] for row in condition_rows if row["margin"] is not None]
    if not margins:
        return None
    return float(min(margins))


def _claim_reasons(
    verdict: str,
    interval: NamedLinearInterval,
    condition_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    if interval.status != "bounded":
        return [
            "The active scenario did not produce a bounded feasible interval, "
            "so the claim cannot be certified."
        ]
    if verdict == "pass":
        return ["The full feasible interval satisfies every asserted claim bound."]
    if verdict == "fail":
        return [
            f"Condition `{row['condition']}` is contradicted by the full feasible "
            "interval."
            for row in condition_rows
            if row["status"] == "fail"
        ]
    return [
        f"Condition `{row['condition']}` is not certified because the feasible "
        "interval crosses the asserted threshold."
        for row in condition_rows
        if row["status"] == "inconclusive"
    ]


def _claim_endpoint_diagnostics(
    claim: NamedLinearClaim,
    interval: NamedLinearInterval,
    *,
    top: int,
) -> tuple[NamedLinearConstraintDiagnostic, ...]:
    endpoints = []
    if claim.lower_at_least is not None:
        endpoints.append(interval.lower_endpoint)
    if claim.upper_at_most is not None:
        endpoints.append(interval.upper_endpoint)
    diagnostics = [
        diagnostic
        for endpoint in endpoints
        for diagnostic in endpoint.constraint_diagnostics
        if diagnostic.binding
        or (diagnostic.dual_magnitude is not None and diagnostic.dual_magnitude > 1e-9)
    ]
    diagnostics.sort(
        key=lambda row: (
            -(row.dual_magnitude or 0.0),
            not row.binding,
            row.endpoint,
            row.constraint,
            row.side,
        )
    )
    return tuple(diagnostics[:top])


def _finite_float(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_float(value: float, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _interval_table(report: NamedLinearFeasibilityReport) -> list[str]:
    target_lookup = {target.name: target for target in report.problem.targets}
    lines = [
        "| Scenario | Target | Lower | Upper | Width | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.intervals:
        target = target_lookup[row.target]
        unit = f" {target.unit}" if target.unit else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row.scenario),
                    _escape_markdown(target.label or row.target),
                    _format_optional(row.scaled_lower, unit),
                    _format_optional(row.scaled_upper, unit),
                    _format_optional(row.scaled_width, unit),
                    row.status,
                ]
            )
            + " |"
        )
    return lines


def _binding_table(report: NamedLinearFeasibilityReport) -> list[str]:
    lines = [
        "| Scenario | Target | Endpoint | Binding constraint sides |",
        "| --- | --- | --- | --- |",
    ]
    for interval in report.intervals:
        for endpoint in (interval.lower_endpoint, interval.upper_endpoint):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(endpoint.scenario),
                        _escape_markdown(endpoint.target),
                        endpoint.sense,
                        _escape_markdown(
                            ", ".join(endpoint.binding_constraint_sides) or "none"
                        ),
                    ]
                )
                + " |"
            )
    return lines


def _dual_diagnostic_table(
    report: NamedLinearFeasibilityReport,
    *,
    top: int = 24,
) -> list[str]:
    diagnostics = [
        diagnostic
        for interval in report.intervals
        for endpoint in (interval.lower_endpoint, interval.upper_endpoint)
        for diagnostic in endpoint.constraint_diagnostics
        if diagnostic.binding
        or (diagnostic.dual_magnitude is not None and diagnostic.dual_magnitude > 1e-9)
    ]
    diagnostics.sort(
        key=lambda row: (
            -(row.dual_magnitude or 0.0),
            not row.binding,
            row.scenario,
            row.target,
            row.endpoint,
            row.constraint,
            row.side,
        )
    )
    diagnostics = diagnostics[:top]
    lines = [
        "| Scenario | Target | Endpoint | Constraint | Side | Binding | Slack | Target marginal | Dual magnitude |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    if not diagnostics:
        lines.append("| n/a | n/a | n/a | none | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in diagnostics:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row.scenario),
                    _escape_markdown(row.target),
                    row.endpoint,
                    _escape_markdown(row.constraint),
                    row.side,
                    "yes" if row.binding else "no",
                    _format_optional(row.slack, ""),
                    _format_optional(row.target_marginal, ""),
                    _format_optional(row.dual_magnitude, ""),
                ]
            )
            + " |"
        )
    return lines


def _claim_condition_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Condition | Status | Endpoint | Endpoint value | Margin |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(str(row["condition"])),
                    str(row["status"]),
                    str(row["certifying_endpoint"]),
                    _format_optional(row["endpoint_value"], ""),
                    _format_optional(row["margin"], ""),
                ]
            )
            + " |"
        )
    return lines


def _diagnostic_rows_table(
    rows: Sequence[NamedLinearConstraintDiagnostic],
) -> list[str]:
    lines = [
        "| Endpoint | Constraint | Side | Binding | Slack | Target marginal | Dual magnitude |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    if not rows:
        lines.append("| n/a | none | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.endpoint,
                    _escape_markdown(row.constraint),
                    row.side,
                    "yes" if row.binding else "no",
                    _format_optional(row.slack, ""),
                    _format_optional(row.target_marginal, ""),
                    _format_optional(row.dual_magnitude, ""),
                ]
            )
            + " |"
        )
    return lines


def _attribution_table(
    rows: Sequence[NamedLinearConstraintAttribution],
) -> list[str]:
    lines = [
        "| Group | Kind | Constraints | Relaxed width | Width increase | Lower tightening | Upper tightening |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row.group),
                    _escape_markdown(row.kind or "mixed"),
                    str(row.constraint_count),
                    _format_optional(row.relaxed_width, ""),
                    _format_optional(row.width_increase, ""),
                    _format_optional(row.lower_tightening, ""),
                    _format_optional(row.upper_tightening, ""),
                ]
            )
            + " |"
        )
    return lines


def _constraint_kind_counts(
    scenario: NamedLinearScenario,
    constraints: Sequence[NamedLinearConstraint],
) -> dict[str, int]:
    lookup = {constraint.name: constraint for constraint in constraints}
    counts: dict[str, int] = {}
    for name in scenario.constraints:
        kind = lookup[name].kind
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _format_optional(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}{unit}"


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


__all__ = [
    "DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS",
    "NamedLinearConstraint",
    "NamedLinearConstraintAttribution",
    "NamedLinearConstraintAttributionReport",
    "NamedLinearConstraintDiagnostic",
    "NamedLinearEndpoint",
    "NamedLinearExpression",
    "NamedLinearFeasibilityProblem",
    "NamedLinearFeasibilityReport",
    "NamedLinearInterval",
    "NamedLinearClaim",
    "NamedLinearClaimAudit",
    "NamedLinearScenario",
    "NamedLinearTarget",
    "NamedLinearVariable",
    "audit_named_linear_claim",
    "coerce_named_linear_constraint",
    "coerce_named_linear_expression",
    "coerce_named_linear_scenario",
    "coerce_named_linear_target",
    "coerce_named_linear_variable",
    "attribute_named_linear_constraints",
    "named_linear_claim",
    "named_linear_constraint",
    "named_linear_expression",
    "named_linear_feasibility_problem",
    "named_linear_scenario",
    "named_linear_target",
    "named_linear_variable",
    "solve_named_linear_feasibility",
]
