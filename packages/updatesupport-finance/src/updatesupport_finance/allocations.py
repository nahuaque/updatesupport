"""Claim-aware feasible allocations for disclosure review artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import updatesupport as us
from updatesupport.artifacts import ReportArtifactMixin


@dataclass(frozen=True)
class DisclosureAllocation:
    role: str
    target_value: float
    assignment: Mapping[str, float]
    checks: tuple[us.NamedLinearAssignmentCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment", MappingProxyType(dict(self.assignment)))
        object.__setattr__(self, "checks", tuple(self.checks))

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "target_value": self.target_value,
            "assignment": dict(self.assignment),
            "checks": [row.as_dict() for row in self.checks],
        }


@dataclass(frozen=True)
class DisclosureAllocationReport(ReportArtifactMixin):
    allocations: tuple[DisclosureAllocation, ...]
    attempts: tuple[dict[str, Any], ...]
    threshold_margin: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "allocations": [row.as_dict() for row in self.allocations],
            "attempts": [dict(row) for row in self.attempts],
            "threshold_margin": self.threshold_margin,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "disclosure_allocations": tuple(
                {
                    "role": row.role,
                    "target_value": row.target_value,
                    "variable": name,
                    "value": value,
                }
                for row in self.allocations
                for name, value in row.assignment.items()
            ),
            "disclosure_allocation_checks": tuple(
                {"role": row.role, **check.as_dict()}
                for row in self.allocations
                for check in row.checks
            ),
            "disclosure_allocation_attempts": self.attempts,
        }

    def to_markdown(self) -> str:
        lines = [
            "## Feasible Allocations",
            "",
            "Allocations satisfy the modeled constraints within the recorded numerical tolerance.",
            "Supporting allocations satisfy the claim; opposing allocations cross a claim boundary.",
            "",
            "| Role | Target value | Variable | Value |",
            "| --- | ---: | --- | ---: |",
        ]
        for row in self.to_tables()["disclosure_allocations"]:
            name = str(row["variable"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['role']} | {row['target_value']:.10g} | {name} | {row['value']:.10g} |"
            )
        lines.extend(
            [
                "",
                "| Role | Constraint sides checked | Variable-bound sides checked | Maximum violation |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for allocation in self.allocations:
            constraint_count = sum(
                row.kind == "constraint" for row in allocation.checks
            )
            bound_count = sum(row.kind == "variable_bound" for row in allocation.checks)
            violation = max((row.violation for row in allocation.checks), default=0.0)
            lines.append(
                f"| {allocation.role} | {constraint_count} | {bound_count} | {violation:.10g} |"
            )
        lines.extend(
            [
                "",
                f"- Opposing-allocation separation margin: {self.threshold_margin:.10g}.",
            ]
        )
        for row in self.attempts:
            lines.append(f"- {row['role']}: {row['status']}.")
        lines.extend(
            [
                "",
                "An unavailable allocation is not itself a claim verdict; inspect the claim audit and solve status.",
            ]
        )
        return "\n".join(lines)


def disclosure_allocations(
    report: us.NamedLinearFeasibilityReport,
    *,
    target: str,
    tier: str,
    claim: us.NamedLinearClaim | None = None,
) -> DisclosureAllocationReport:
    """Produce endpoint allocations, or feasible examples on each claim side.

    Supporting allocations are solved with the complete claim imposed, so a
    two-sided claim can have an interior witness even when both extrema fail.
    Opposing solves use a positive separation because LPs cannot encode strict
    inequalities. No synthetic allocation is returned for a failed solve.
    """
    if claim is not None and (claim.target != target or claim.scenario != tier):
        raise ValueError("claim must match the allocation target and tier")
    interval = report.interval(target=target, scenario=tier)
    allocations = []
    attempts = []
    margin = 10 * report.problem.feasibility_tolerance

    def collect(role: str, endpoint: us.NamedLinearEndpoint) -> None:
        status = endpoint.status
        if (
            status == "optimal"
            and endpoint.assignment is not None
            and endpoint.value is not None
        ):
            value = endpoint.value
            if claim is not None:
                supports = (
                    claim.lower_at_least is None or value >= claim.lower_at_least
                ) and (claim.upper_at_most is None or value <= claim.upper_at_most)
                if (role == "supporting") != supports:
                    status = "numerical_boundary"
            if status == "optimal":
                checks = us.check_named_linear_assignment(
                    report.problem, endpoint.assignment, scenario=tier
                )
                if all(row.passed for row in checks):
                    allocations.append(
                        DisclosureAllocation(role, value, endpoint.assignment, checks)
                    )
                else:
                    status = "numerical_error"
        attempts.append({"role": role, "status": status})

    if interval.status not in {"bounded", "unbounded"}:
        return DisclosureAllocationReport(
            (), ({"role": "source_problem", "status": interval.status},), margin
        )
    if claim is None:
        collect("lower_endpoint", interval.lower_endpoint)
        collect("upper_endpoint", interval.upper_endpoint)
    else:
        threshold_scale = max(
            abs(claim.lower_at_least or 0), abs(claim.upper_at_most or 0)
        )
        margin = max(
            margin,
            2 * (claim.absolute_tolerance + claim.relative_tolerance * threshold_scale),
        )
        target_row = next(row for row in report.problem.targets if row.name == target)
        scenario = next(row for row in report.problem.scenarios if row.name == tier)
        name = "__allocation_claim"
        while name in {row.name for row in report.problem.constraints}:
            name += "_"
        requests = [
            (
                "supporting",
                claim.lower_at_least,
                claim.upper_at_most,
                "min" if claim.lower_at_least is not None else "max",
            )
        ]
        if claim.lower_at_least is not None:
            requests.append(
                ("opposing_lower", None, claim.lower_at_least - margin, "max")
            )
        if claim.upper_at_most is not None:
            requests.append(
                ("opposing_upper", claim.upper_at_most + margin, None, "min")
            )
        for role, lower, upper, sense in requests:
            # Prefer explanatory extrema over barely crossing a threshold.
            # The constrained solve is needed for interior or unbounded cases.
            endpoints = (interval.lower_endpoint, interval.upper_endpoint)
            if role == "supporting" and claim.upper_at_most is None:
                endpoints = tuple(reversed(endpoints))
            selected = next(
                (
                    endpoint
                    for endpoint in endpoints
                    if endpoint.status == "optimal"
                    and endpoint.value is not None
                    and (lower is None or endpoint.value >= lower)
                    and (upper is None or endpoint.value <= upper)
                ),
                None,
            )
            if selected is not None:
                collect(role, selected)
                continue
            if role == "supporting" and lower is not None and upper is not None:
                feasible_lower = (
                    max(lower, interval.lower) if interval.lower is not None else lower
                )
                feasible_upper = (
                    min(upper, interval.upper) if interval.upper is not None else upper
                )
                if feasible_lower <= feasible_upper:
                    lower = upper = feasible_lower / 2 + feasible_upper / 2
            boundary = us.named_linear_constraint(
                name, target_row.expression, lower=lower, upper=upper
            )
            problem = replace(
                report.problem,
                constraints=(*report.problem.constraints, boundary),
                targets=(target_row,),
                scenarios=(
                    replace(scenario, constraints=(*scenario.constraints, name)),
                ),
            )
            candidate = us.solve_named_linear_feasibility(problem).interval(
                target=target, scenario=tier
            )
            endpoint = (
                candidate.lower_endpoint if sense == "min" else candidate.upper_endpoint
            )
            collect(role, endpoint)
    return DisclosureAllocationReport(tuple(allocations), tuple(attempts), margin)
