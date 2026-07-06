"""Disclosure triangulation helpers built on core named-linear feasibility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import updatesupport as us


DisclosureVariable = us.NamedLinearVariable
DisclosureExpression = us.NamedLinearExpression
DisclosureConstraint = us.NamedLinearConstraint
DisclosureConstraintAttribution = us.NamedLinearConstraintAttribution
DisclosureConstraintAttributionReport = us.NamedLinearConstraintAttributionReport
DisclosureConstraintDiagnostic = us.NamedLinearConstraintDiagnostic
DisclosureClaim = us.NamedLinearClaim
DisclosureClaimAudit = us.NamedLinearClaimAudit
DisclosureTarget = us.NamedLinearTarget
DisclosureTier = us.NamedLinearScenario
DisclosureTriangulationSpec = us.NamedLinearFeasibilityProblem
DisclosureTriangulationReport = us.NamedLinearFeasibilityReport


def disclosure_variable(
    name: str,
    *,
    lower: float | None = 0.0,
    upper: float | None = None,
    label: str | None = None,
    unit: str | None = None,
    description: str | None = None,
) -> DisclosureVariable:
    """Create a nonnegative scalar disclosure variable by default."""

    return us.named_linear_variable(
        name,
        lower=lower,
        upper=upper,
        label=label,
        unit=unit,
        description=description,
    )


def disclosure_constraint(
    name: str,
    coefficients: us.NamedLinearExpression | Mapping[str, float] | str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    constant: float = 0.0,
    category: str = "disclosure",
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DisclosureConstraint:
    """Create a provenance-aware linear disclosure constraint."""

    return us.named_linear_constraint(
        name,
        coefficients,
        lower=lower,
        upper=upper,
        constant=constant,
        kind=category,
        provenance=provenance,
        description=description,
        verified=verified,
        metadata=metadata,
    )


def exact_disclosure_constraint(
    name: str,
    coefficients: us.NamedLinearExpression | Mapping[str, float] | str,
    value: float,
    *,
    constant: float = 0.0,
    category: str = "disclosure",
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DisclosureConstraint:
    """Create an equality disclosure constraint."""

    return disclosure_constraint(
        name,
        coefficients,
        lower=value,
        upper=value,
        constant=constant,
        category=category,
        provenance=provenance,
        description=description,
        verified=verified,
        metadata=metadata,
    )


def interval_disclosure_constraint(
    name: str,
    coefficients: us.NamedLinearExpression | Mapping[str, float] | str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    constant: float = 0.0,
    category: str = "disclosure_interval",
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DisclosureConstraint:
    """Create a one-sided or two-sided interval disclosure constraint."""

    return disclosure_constraint(
        name,
        coefficients,
        lower=lower,
        upper=upper,
        constant=constant,
        category=category,
        provenance=provenance,
        description=description,
        verified=verified,
        metadata=metadata,
    )


def containment_constraint(
    name: str,
    *,
    child: str,
    parent: str,
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DisclosureConstraint:
    """Constrain a component to be no larger than a containing total."""

    return disclosure_constraint(
        name,
        {child: 1.0, parent: -1.0},
        upper=0.0,
        category="containment",
        provenance=provenance,
        description=description,
        verified=verified,
        metadata=metadata,
    )


def rounded_growth_constraints(
    name: str,
    *,
    current: str,
    previous: str,
    growth_percent: float,
    rounding: float = 0.5,
    category: str = "rounded_growth_disclosure",
    provenance: str | None = None,
    description: str | None = None,
    verified: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[DisclosureConstraint, DisclosureConstraint]:
    """Encode a rounded growth-rate disclosure as two linear constraints.

    The helper assumes the previous-period variable is nonnegative. A rounded
    growth statement such as ``39%`` with ``rounding=0.5`` is encoded as the
    inclusive relaxation ``current / previous - 1 in [38.5%, 39.5%]``.
    """

    lower_growth = float(growth_percent) - float(rounding)
    upper_growth = float(growth_percent) + float(rounding)
    lower_ratio = 1.0 + lower_growth / 100.0
    upper_ratio = 1.0 + upper_growth / 100.0
    common_metadata = {
        "growth_percent": float(growth_percent),
        "rounding": float(rounding),
        "lower_growth_percent": lower_growth,
        "upper_growth_percent": upper_growth,
        "assumes_nonnegative_previous": True,
    }
    if metadata:
        common_metadata.update(dict(metadata))
    return (
        disclosure_constraint(
            f"{name}_lower",
            {current: 1.0, previous: -lower_ratio},
            lower=0.0,
            category=category,
            provenance=provenance,
            description=description,
            verified=verified,
            metadata=common_metadata,
        ),
        disclosure_constraint(
            f"{name}_upper",
            {current: 1.0, previous: -upper_ratio},
            upper=0.0,
            category=category,
            provenance=provenance,
            description=description,
            verified=verified,
            metadata=common_metadata,
        ),
    )


def disclosure_target(
    name: str,
    expression: us.NamedLinearExpression | Mapping[str, float] | str,
    *,
    label: str | None = None,
    unit: str | None = None,
    scale: float = 1.0,
    description: str | None = None,
) -> DisclosureTarget:
    """Create a triangulation target expression."""

    return us.named_linear_target(
        name,
        expression,
        label=label,
        unit=unit,
        scale=scale,
        description=description,
    )


def disclosure_tier(
    name: str,
    constraints: Sequence[str],
    *,
    description: str | None = None,
) -> DisclosureTier:
    """Create a named tier of active disclosure assumptions."""

    return us.named_linear_scenario(
        name,
        constraints,
        description=description,
    )


def disclosure_triangulation_spec(
    *,
    variables: Sequence[DisclosureVariable | str | Mapping[str, Any]],
    constraints: Sequence[DisclosureConstraint | Mapping[str, Any]],
    targets: Sequence[DisclosureTarget | Mapping[str, Any]],
    tiers: Sequence[DisclosureTier | Mapping[str, Any]],
    title: str = "Disclosure Triangulation Report",
    description: str | None = None,
    limitations: Sequence[str] = us.DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS,
) -> DisclosureTriangulationSpec:
    """Build a generic disclosure-triangulation feasibility spec."""

    return us.named_linear_feasibility_problem(
        variables=variables,
        constraints=constraints,
        targets=targets,
        scenarios=tiers,
        title=title,
        description=description,
        limitations=limitations,
    )


def triangulate_disclosure(
    spec: DisclosureTriangulationSpec | Mapping[str, Any],
) -> DisclosureTriangulationReport:
    """Solve a disclosure triangulation spec with the core named-linear solver."""

    return us.solve_named_linear_feasibility(spec)


def attribute_disclosure_constraints(
    report: DisclosureTriangulationReport,
    *,
    target: str,
    tier: str,
    group_by: str = "constraint",
    groups: Mapping[str, Sequence[str]] | None = None,
    top: int | None = None,
) -> DisclosureConstraintAttributionReport:
    """Rank disclosure constraints by leave-one-group-out interval widening."""

    return us.attribute_named_linear_constraints(
        report,
        target=target,
        scenario=tier,
        group_by=group_by,
        groups=groups,
        top=top,
    )


def disclosure_claim(
    *,
    target: str,
    tier: str,
    lower_at_least: float | None = None,
    upper_at_most: float | None = None,
    label: str | None = None,
    description: str | None = None,
    attribution_top: int = 5,
    diagnostic_top: int = 8,
) -> DisclosureClaim:
    """Create a claim about a disclosure-triangulation target interval."""

    return us.named_linear_claim(
        target=target,
        scenario=tier,
        lower_at_least=lower_at_least,
        upper_at_most=upper_at_most,
        label=label,
        description=description,
        attribution_top=attribution_top,
        diagnostic_top=diagnostic_top,
    )


def audit_disclosure_claim(
    report: DisclosureTriangulationReport,
    claim: DisclosureClaim | Mapping[str, Any],
) -> DisclosureClaimAudit:
    """Audit whether a disclosure report supports a bound claim."""

    return us.audit_named_linear_claim(report, claim)


__all__ = [
    "DisclosureClaim",
    "DisclosureClaimAudit",
    "DisclosureConstraint",
    "DisclosureConstraintAttribution",
    "DisclosureConstraintAttributionReport",
    "DisclosureConstraintDiagnostic",
    "DisclosureExpression",
    "DisclosureTarget",
    "DisclosureTier",
    "DisclosureTriangulationReport",
    "DisclosureTriangulationSpec",
    "DisclosureVariable",
    "audit_disclosure_claim",
    "attribute_disclosure_constraints",
    "containment_constraint",
    "disclosure_claim",
    "disclosure_constraint",
    "disclosure_target",
    "disclosure_tier",
    "disclosure_triangulation_spec",
    "disclosure_variable",
    "exact_disclosure_constraint",
    "interval_disclosure_constraint",
    "rounded_growth_constraints",
    "triangulate_disclosure",
]
