"""Disclosure triangulation helpers built on core named-linear feasibility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS = (
    "The audit pack reports feasibility bounds over the encoded disclosure "
    "constraint system; it is not a point estimate or confidence interval.",
    "The verdict depends on the supplied variables, linear constraints, active "
    "tier, and claim threshold.",
    "A narrow interval means the modeled disclosures constrain the target under "
    "the stated assumptions; it does not validate unmodeled business facts.",
    "A wide interval means the modeled disclosures leave room for multiple "
    "allocations; it does not imply management's reported disclosures are "
    "incorrect.",
)


@dataclass(frozen=True)
class DisclosureAuditPack:
    """Analyst-facing disclosure audit artifact.

    The pack wraps a solved disclosure-triangulation report around one headline
    target and, optionally, one claim about that target. It preserves source
    links, assumptions, attribution, and binding diagnostics in one reviewable
    Markdown/JSON/DataFrame object.
    """

    report: DisclosureTriangulationReport
    target: str
    tier: str
    title: str | None = None
    claim_audit: DisclosureClaimAudit | None = None
    attribution: DisclosureConstraintAttributionReport | None = None
    sources: Sequence[Mapping[str, Any]] = ()
    assumptions: Sequence[str] = ()
    reviewer_notes: Sequence[str] = ()
    diagnostic_top: int = 8
    limitations: Sequence[str] = DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS

    def __post_init__(self) -> None:
        # Validate target/tier eagerly so an audit pack cannot point at a
        # missing interval.
        self.report.interval(target=self.target, scenario=self.tier)
        object.__setattr__(self, "sources", tuple(dict(row) for row in self.sources))
        object.__setattr__(
            self,
            "assumptions",
            tuple(str(row) for row in self.assumptions),
        )
        object.__setattr__(
            self,
            "reviewer_notes",
            tuple(str(row) for row in self.reviewer_notes),
        )
        object.__setattr__(
            self,
            "diagnostic_top",
            _positive_int(self.diagnostic_top, "diagnostic_top"),
        )
        object.__setattr__(
            self,
            "limitations",
            tuple(str(row) for row in self.limitations),
        )

    @property
    def audit_title(self) -> str:
        return self.title or f"{self.report.title} Audit Pack"

    @property
    def interval(self) -> us.NamedLinearInterval:
        return self.report.interval(target=self.target, scenario=self.tier)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.audit_title,
            "target": self.target,
            "tier": self.tier,
            "interval": self.interval.as_dict(),
            "claim_audit": None
            if self.claim_audit is None
            else self.claim_audit.as_dict(),
            "attribution": None
            if self.attribution is None
            else self.attribution.as_dict(),
            "sources": list(self.sources),
            "assumptions": list(self.assumptions),
            "reviewer_notes": list(self.reviewer_notes),
            "active_constraints": list(_active_constraint_rows(self.report, self.tier)),
            "binding_diagnostics": [
                row.as_dict() for row in self._top_binding_diagnostics()
            ],
            "limitations": list(self.limitations),
            "triangulation_report": self.report.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return us.report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        interval = self.interval
        tables: dict[str, tuple[dict[str, Any], ...]] = {
            "disclosure_audit_summary": (
                {
                    "title": self.audit_title,
                    "source_report": self.report.title,
                    "target": self.target,
                    "target_label": _target_label(self.report, self.target),
                    "tier": self.tier,
                    "verdict": None
                    if self.claim_audit is None
                    else self.claim_audit.verdict,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "width": interval.width,
                    "scaled_lower": interval.scaled_lower,
                    "scaled_upper": interval.scaled_upper,
                    "scaled_width": interval.scaled_width,
                    "unit": _target_unit(self.report, self.target),
                    "status": interval.status,
                },
            ),
            "disclosure_audit_sources": tuple(
                {"position": index + 1, **dict(row)}
                for index, row in enumerate(self.sources)
            ),
            "disclosure_audit_active_constraints": tuple(
                _active_constraint_rows(self.report, self.tier)
            ),
            "disclosure_audit_tier_intervals": tuple(
                _target_tier_rows(self.report, self.target)
            ),
            "disclosure_audit_binding_diagnostics": tuple(
                row.as_dict() for row in self._top_binding_diagnostics()
            ),
            "disclosure_audit_assumptions": tuple(
                {"assumption": row} for row in self.assumptions
            ),
            "disclosure_audit_reviewer_notes": tuple(
                {"note": row} for row in self.reviewer_notes
            ),
            "disclosure_audit_limitations": tuple(
                {"limitation": row} for row in self.limitations
            ),
        }
        if self.claim_audit is not None:
            tables["disclosure_audit_claim_conditions"] = tuple(
                dict(row) for row in self.claim_audit.condition_rows
            )
            tables["disclosure_audit_claim_reasons"] = tuple(
                {"reason": row} for row in self.claim_audit.reasons
            )
        if self.attribution is not None:
            tables["disclosure_audit_constraint_attribution"] = tuple(
                row.as_dict() for row in self.attribution.rows
            )
        return tables

    def to_dataframes(self) -> dict[str, Any]:
        return us.tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        interval = self.interval
        unit = _target_unit(self.report, self.target)
        lines = [f"# {_escape_markdown(self.audit_title)}", ""]
        if self.report.problem.description:
            lines.extend([self.report.problem.description, ""])
        lines.extend(
            [
                "## Review Question",
                "",
                f"- Target: `{_escape_markdown(self.target)}`",
                f"- Tier: `{_escape_markdown(self.tier)}`",
            ]
        )
        if self.claim_audit is not None:
            lines.append(
                f"- Claim: `{_escape_markdown(self.claim_audit.claim.statement)}`"
            )
            lines.append(f"- Verdict: **{_escape_markdown(self.claim_audit.verdict)}**")
        lines.extend(["", "## Headline Interval", ""])
        lines.extend(_headline_interval_table(interval, unit))
        lines.extend(["", "## Tier Comparison", ""])
        lines.extend(_tier_interval_table(self.report, self.target))
        if self.sources:
            lines.extend(["", "## Source Disclosures", ""])
            lines.extend(_source_table(self.sources))
        lines.extend(["", "## Modeled Constraints", ""])
        lines.extend(_active_constraint_table(self.report, self.tier))
        if self.claim_audit is not None:
            lines.extend(["", "## Claim Audit", ""])
            lines.extend(_claim_summary_table(self.claim_audit))
            if self.claim_audit.reasons:
                lines.extend(["", "Reasons:"])
                lines.extend(f"- {reason}" for reason in self.claim_audit.reasons)
        if self.attribution is not None:
            lines.extend(["", "## Binding / Value Drivers", ""])
            lines.extend(_attribution_table(self.attribution))
        diagnostics = self._top_binding_diagnostics()
        if diagnostics:
            lines.extend(["", "## Endpoint Diagnostics", ""])
            lines.extend(_diagnostic_table(diagnostics))
        if self.assumptions:
            lines.extend(["", "## Assumptions", ""])
            lines.extend(f"- {row}" for row in self.assumptions)
        if self.reviewer_notes:
            lines.extend(["", "## Reviewer Notes", ""])
            lines.extend(f"- {row}" for row in self.reviewer_notes)
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {row}" for row in self.limitations)
        return "\n".join(lines)

    def _top_binding_diagnostics(self) -> tuple[DisclosureConstraintDiagnostic, ...]:
        rows = [
            row
            for endpoint in (self.interval.lower_endpoint, self.interval.upper_endpoint)
            for row in endpoint.constraint_diagnostics
            if row.binding
            or (row.dual_magnitude is not None and row.dual_magnitude > 0)
        ]
        rows.sort(
            key=lambda row: (
                0 if row.binding else 1,
                -(row.dual_magnitude or 0.0),
                row.endpoint,
                row.constraint,
                row.side,
            )
        )
        return tuple(rows[: self.diagnostic_top])


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


def disclosure_audit_pack(
    report: DisclosureTriangulationReport,
    *,
    claim: DisclosureClaim | Mapping[str, Any] | None = None,
    target: str | None = None,
    tier: str | None = None,
    title: str | None = None,
    sources: Sequence[Mapping[str, Any]] = (),
    assumptions: Sequence[str] = (),
    reviewer_notes: Sequence[str] = (),
    group_by: str = "constraint",
    attribution_top: int | None = 8,
    diagnostic_top: int = 8,
    limitations: Sequence[str] = DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS,
) -> DisclosureAuditPack:
    """Build an analyst-facing disclosure audit pack.

    The helper takes an already-solved disclosure triangulation report and
    packages one headline target, optional claim audit, source metadata,
    attribution, and binding diagnostics into a review artifact.
    """

    if claim is not None and not isinstance(claim, us.NamedLinearClaim):
        claim = us.NamedLinearClaim(**dict(claim))
    claim_audit = None
    if claim is not None:
        target = claim.target
        tier = claim.scenario
        claim_audit = audit_disclosure_claim(report, claim)
    if target is None or tier is None:
        raise ValueError("disclosure_audit_pack requires target and tier or a claim")
    attribution = None
    interval = report.interval(target=target, scenario=tier)
    if interval.status == "bounded":
        attribution = attribute_disclosure_constraints(
            report,
            target=target,
            tier=tier,
            group_by=group_by,
            top=attribution_top,
        )
    return DisclosureAuditPack(
        report=report,
        target=target,
        tier=tier,
        title=title,
        claim_audit=claim_audit,
        attribution=attribution,
        sources=sources,
        assumptions=assumptions,
        reviewer_notes=reviewer_notes,
        diagnostic_top=diagnostic_top,
        limitations=limitations,
    )


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _target_lookup(
    report: DisclosureTriangulationReport,
) -> dict[str, DisclosureTarget]:
    return {row.name: row for row in report.problem.targets}


def _target_label(report: DisclosureTriangulationReport, target: str) -> str | None:
    return _target_lookup(report)[target].label


def _target_unit(report: DisclosureTriangulationReport, target: str) -> str | None:
    return _target_lookup(report)[target].unit


def _target_tier_rows(
    report: DisclosureTriangulationReport,
    target: str,
) -> list[dict[str, Any]]:
    rows = []
    for interval in report.intervals:
        if interval.target != target:
            continue
        rows.append(
            {
                "tier": interval.scenario,
                "lower": interval.scaled_lower,
                "upper": interval.scaled_upper,
                "width": interval.scaled_width,
                "status": interval.status,
            }
        )
    return rows


def _active_constraint_rows(
    report: DisclosureTriangulationReport,
    tier: str,
) -> list[dict[str, Any]]:
    constraints = {row.name: row for row in report.problem.constraints}
    scenarios = {row.name: row for row in report.problem.scenarios}
    scenario = scenarios[tier]
    rows = []
    for name in scenario.constraints:
        constraint = constraints[name]
        rows.append(
            {
                "constraint": constraint.name,
                "kind": constraint.kind,
                "lower": constraint.lower,
                "upper": constraint.upper,
                "verified": constraint.verified,
                "provenance": constraint.provenance,
                "description": constraint.description,
            }
        )
    return rows


def _headline_interval_table(
    interval: us.NamedLinearInterval,
    unit: str | None,
) -> list[str]:
    return [
        "| Target | Tier | Lower | Upper | Width | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        "| "
        + " | ".join(
            [
                _escape_markdown(interval.target),
                _escape_markdown(interval.scenario),
                _format_optional(interval.scaled_lower, unit=unit),
                _format_optional(interval.scaled_upper, unit=unit),
                _format_optional(interval.scaled_width, unit=unit),
                _escape_markdown(interval.status),
            ]
        )
        + " |",
    ]


def _tier_interval_table(
    report: DisclosureTriangulationReport,
    target: str,
) -> list[str]:
    unit = _target_unit(report, target)
    lines = [
        "| Tier | Lower | Upper | Width | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in _target_tier_rows(report, target):
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row["tier"]),
                    _format_optional(row["lower"], unit=unit),
                    _format_optional(row["upper"], unit=unit),
                    _format_optional(row["width"], unit=unit),
                    _escape_markdown(row["status"]),
                ]
            )
            + " |"
        )
    return lines


def _source_table(sources: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Source | Value | URL | Notes |",
        "| --- | ---: | --- | --- |",
    ]
    for row in sources:
        label = _escape_markdown(str(row.get("label", row.get("source", ""))))
        value = _escape_markdown(str(row.get("value", "")))
        url = str(row.get("url", "") or "")
        description = _escape_markdown(str(row.get("description", "")))
        link = f"[link]({url})" if url else ""
        lines.append(f"| {label} | {value} | {link} | {description} |")
    return lines


def _active_constraint_table(
    report: DisclosureTriangulationReport,
    tier: str,
) -> list[str]:
    lines = [
        "| Constraint | Kind | Bound | Verified | Provenance |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in _active_constraint_rows(report, tier):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape_markdown(row['constraint'])}`",
                    _escape_markdown(str(row["kind"])),
                    _constraint_bound(row["lower"], row["upper"]),
                    _verified_label(row["verified"]),
                    _escape_markdown(str(row.get("provenance") or "")),
                ]
            )
            + " |"
        )
    return lines


def _claim_summary_table(claim_audit: DisclosureClaimAudit) -> list[str]:
    lines = [
        "| Condition | Status | Endpoint | Endpoint value | Margin |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in claim_audit.condition_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(str(row.get("condition", ""))),
                    _escape_markdown(str(row.get("status", ""))),
                    _escape_markdown(
                        str(row.get("endpoint") or row.get("certifying_endpoint") or "")
                    ),
                    _format_optional(row.get("endpoint_value")),
                    _format_optional(row.get("margin")),
                ]
            )
            + " |"
        )
    return lines


def _attribution_table(
    attribution: DisclosureConstraintAttributionReport,
) -> list[str]:
    lines = [
        "| Group | Kind | Relaxed width | Width increase | Lower tightening | Upper tightening |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in attribution.rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape_markdown(row.group)}`",
                    _escape_markdown(str(row.kind or "")),
                    _format_optional(row.relaxed_width),
                    _format_optional(row.width_increase),
                    _format_optional(row.lower_tightening),
                    _format_optional(row.upper_tightening),
                ]
            )
            + " |"
        )
    return lines


def _diagnostic_table(
    diagnostics: Sequence[DisclosureConstraintDiagnostic],
) -> list[str]:
    lines = [
        "| Endpoint | Constraint | Side | Binding | Slack | Target marginal | Dual magnitude |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in diagnostics:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown(row.endpoint),
                    f"`{_escape_markdown(row.constraint)}`",
                    _escape_markdown(row.side),
                    "yes" if row.binding else "no",
                    _format_optional(row.slack),
                    _format_optional(row.target_marginal),
                    _format_optional(row.dual_magnitude),
                ]
            )
            + " |"
        )
    return lines


def _constraint_bound(lower: float | None, upper: float | None) -> str:
    if lower is not None and upper is not None and lower == upper:
        return _format_optional(lower)
    if lower is not None and upper is not None:
        return f"[{_format_optional(lower)}, {_format_optional(upper)}]"
    if lower is not None:
        return f">= {_format_optional(lower)}"
    if upper is not None:
        return f"<= {_format_optional(upper)}"
    return "n/a"


def _verified_label(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _format_optional(value: Any, unit: str | None = None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape_markdown(str(value))
    if abs(numeric) >= 1_000:
        rendered = f"{numeric:,.0f}"
    elif abs(numeric) >= 10:
        rendered = f"{numeric:.4g}"
    else:
        rendered = f"{numeric:.4f}".rstrip("0").rstrip(".")
    if unit:
        return f"{rendered} {_escape_markdown(unit)}"
    return rendered


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS",
    "DisclosureAuditPack",
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
    "disclosure_audit_pack",
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
