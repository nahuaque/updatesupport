"""Named linear feasibility intervals for arbitrary scalar variables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    solver_status: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_constraints", tuple(self.binding_constraints))
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
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.problem.limitations)
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
    for constraint in constraints:
        row = _coefficient_row(constraint.expression, index)
        constant = constraint.expression.constant
        if constraint.upper is not None:
            a_ub.append(row)
            b_ub.append(constraint.upper - constant)
        if constraint.lower is not None:
            a_ub.append([-value for value in row])
            b_ub.append(constant - constraint.lower)

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
    return NamedLinearEndpoint(
        scenario=scenario.name,
        target=target.name,
        sense=sense,
        status="optimal",
        value=value,
        scaled_value=value / target.scale,
        assignment=assignment,
        binding_constraints=_binding_constraints(assignment, constraints),
        solver_status=int(result.status),
        message=str(result.message),
    )


def _coefficient_row(
    expression: NamedLinearExpression,
    index: Mapping[str, int],
) -> list[float]:
    row = [0.0] * len(index)
    for name, coefficient in expression.coefficients.items():
        row[index[name]] = coefficient
    return row


def _binding_constraints(
    assignment: Mapping[str, float],
    constraints: Sequence[NamedLinearConstraint],
    *,
    tol: float = 1e-7,
) -> tuple[str, ...]:
    names = []
    for constraint in constraints:
        value = constraint.expression.evaluate(assignment)
        lower_binding = (
            constraint.lower is not None and abs(value - constraint.lower) <= tol
        )
        upper_binding = (
            constraint.upper is not None and abs(value - constraint.upper) <= tol
        )
        if lower_binding or upper_binding:
            names.append(constraint.name)
    return tuple(names)


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


def _finite_float(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
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
        "| Scenario | Target | Endpoint | Binding constraints |",
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
                            ", ".join(endpoint.binding_constraints) or "none"
                        ),
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
    "NamedLinearEndpoint",
    "NamedLinearExpression",
    "NamedLinearFeasibilityProblem",
    "NamedLinearFeasibilityReport",
    "NamedLinearInterval",
    "NamedLinearScenario",
    "NamedLinearTarget",
    "NamedLinearVariable",
    "coerce_named_linear_constraint",
    "coerce_named_linear_expression",
    "coerce_named_linear_scenario",
    "coerce_named_linear_target",
    "coerce_named_linear_variable",
    "named_linear_constraint",
    "named_linear_expression",
    "named_linear_feasibility_problem",
    "named_linear_scenario",
    "named_linear_target",
    "named_linear_variable",
    "solve_named_linear_feasibility",
]
