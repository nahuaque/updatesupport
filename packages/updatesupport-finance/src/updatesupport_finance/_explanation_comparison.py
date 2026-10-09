"""Compare economic explanations across explicit accounting mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import updatesupport as us
from updatesupport.artifacts import ReportArtifactMixin

from .disclosure import find_disclosure_conflict
from .evidence import DisclosureFact, validate_disclosure_evidence
from .scenarios import constraint_evidence_role


_LIMITATIONS = (
    "Compatibility is conditional on the supplied facts, accounting mappings and explanation catalog; it is not probability or causal attribution.",
    "The catalog need not cover every economic explanation. A sole surviving catalog entry does not establish the actual company explanation.",
    "Missing movements remain as modeled. No unknown amount, source coverage or customer identity is inferred from a caption label.",
    "Evidence outcomes are hypothetical policies. Minimal bundles are inclusion-minimal within the supplied finite catalog, not optimal by cost or likelihood.",
)


def _name(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _names(values, label):
    if isinstance(values, str):
        raise ValueError(label + " must be a sequence of names, not a string")
    values = tuple(values)
    if not all(map(_name, values)) or len(set(values)) != len(values):
        raise ValueError(label + " must be unique nonempty strings")
    return values


def _policies(name, rows):
    rows = tuple(rows)
    if any(not isinstance(c, us.NamedLinearConstraint) for c in rows):
        raise ValueError("alternatives require named linear constraints")
    if len({c.name for c in rows}) != len(rows):
        raise ValueError("constraint names must be unique within an alternative")
    for c in rows:
        if (
            c.kind in {"reported_fact", "derived_fact"}
            or constraint_evidence_role(c) == "reported_fact"
            or (c.metadata or {}).get("fact_variable") is not None
        ):
            raise ValueError(
                "source measurements belong in the base problem, not hypothetical alternatives"
            )
    return tuple(
        replace(
            c,
            kind="analyst_policy",
            metadata={
                **(c.metadata or {}),
                "evidence_role": "analyst_policy",
                "policy_name": name,
            },
        )
        for c in rows
    )


@dataclass(frozen=True)
class DisclosureAlternative:
    """A named mapping or economic explanation, selected as analyst policy.

    Constraints only add restrictions; source facts and shared identities cannot
    be replaced or removed. Empty constraints explicitly represent an open
    mapping or unconstrained explanation. Use separate instances in the
    ``mappings`` and ``explanations`` arguments.
    """

    name: str
    description: str
    constraints: Sequence[us.NamedLinearConstraint] = ()

    def __post_init__(self):
        if not _name(self.name) or not _name(self.description):
            raise ValueError("alternatives need a name and description")
        object.__setattr__(self, "constraints", _policies(self.name, self.constraints))

    def as_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "constraints": [c.as_dict() for c in self.constraints],
        }


@dataclass(frozen=True)
class DisclosureExplanationCase:
    mapping: str | None
    explanation: str | None
    tier: str
    status: str
    intervals: Mapping[str, us.NamedLinearInterval]
    witness: Mapping[str, float] | None
    conflict: us.NamedLinearConflictReport | None

    def __post_init__(self):
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        if self.witness is not None:
            object.__setattr__(self, "witness", MappingProxyType(dict(self.witness)))

    def as_dict(self):
        return {
            "mapping": self.mapping,
            "explanation": self.explanation,
            "tier": self.tier,
            "status": self.status,
            "intervals": {
                n: {"lower": i.lower, "upper": i.upper, "status": i.status}
                for n, i in self.intervals.items()
            },
            "witness": None if self.witness is None else dict(self.witness),
            "conflict": None if self.conflict is None else self.conflict.as_dict(),
        }


@dataclass(frozen=True)
class DisclosureExplanationComparison(ReportArtifactMixin):
    source_problem: us.NamedLinearFeasibilityProblem
    base_tier: str
    mappings: tuple[DisclosureAlternative, ...]
    explanations: tuple[DisclosureAlternative, ...]
    report: us.NamedLinearFeasibilityReport
    source: DisclosureExplanationCase
    mapping_checks: tuple[DisclosureExplanationCase, ...]
    cases: tuple[DisclosureExplanationCase, ...]
    evidence: tuple[DisclosureFact, ...] = ()

    def case(self, mapping: str, explanation: str) -> DisclosureExplanationCase:
        for c in self.cases:
            if (c.mapping, c.explanation) == (mapping, explanation):
                return c
        raise ValueError("unknown mapping/explanation pair")

    def summary(self) -> tuple[dict[str, Any], ...]:
        feasible = {c.mapping for c in self.mapping_checks if c.status == "feasible"}
        unknown_mapping = any(c.status == "undetermined" for c in self.mapping_checks)
        rows = []
        for explanation in self.explanations:
            cells = [
                c
                for c in self.cases
                if c.explanation == explanation.name and c.mapping in feasible
            ]
            possible = tuple(c.mapping for c in cells if c.status == "feasible")
            unknown = tuple(c.mapping for c in cells if c.status == "undetermined")
            if not feasible or unknown_mapping or unknown:
                status = "undetermined"
            elif not possible:
                status = "excluded_in_all_feasible_mappings"
            elif set(possible) == feasible:
                status = "possible_in_all_feasible_mappings"
            else:
                status = "mapping_dependent"
            rows.append(
                {
                    "explanation": explanation.name,
                    "status": status,
                    "possible_mappings": list(possible),
                    "undetermined_mappings": list(unknown),
                }
            )
        return tuple(rows)

    def snapshot_context(self):
        return {
            "explanation_comparison": {
                "base_tier": self.base_tier,
                "source_tier": self.source.tier,
                "mappings": [m.as_dict() for m in self.mappings],
                "explanations": [e.as_dict() for e in self.explanations],
                "case_tiers": [
                    {"mapping": c.mapping, "explanation": c.explanation, "tier": c.tier}
                    for c in self.cases
                ],
                "limitations": list(_LIMITATIONS),
            }
        }

    def as_dict(self):
        return {
            **self.snapshot_context(),
            "source": self.source.as_dict(),
            "mapping_checks": [c.as_dict() for c in self.mapping_checks],
            "cases": [c.as_dict() for c in self.cases],
            "summary": list(self.summary()),
            "evidence": [f.as_dict() for f in self.evidence],
            "report": self.report.as_dict(),
        }

    def to_tables(self):
        return {
            "explanation_summary": self.summary(),
            "accounting_mapping_status": tuple(
                {"mapping": c.mapping, "status": c.status} for c in self.mapping_checks
            ),
            "explanation_matrix": tuple(
                {
                    "mapping": c.mapping,
                    "explanation": c.explanation,
                    "tier": c.tier,
                    "status": c.status,
                    "target": n,
                    "lower": i.lower,
                    "upper": i.upper,
                    "interval_status": i.status,
                }
                for c in self.cases
                for n, i in c.intervals.items()
            ),
            "explanation_cases": tuple(
                {
                    "mapping": c.mapping,
                    "explanation": c.explanation,
                    "tier": c.tier,
                    "status": c.status,
                }
                for c in self.cases
            ),
        }

    def to_markdown(self):
        lines = [
            "# Financial explanation comparison",
            "",
            "## Accounting mappings",
            "",
            "| Mapping | Source compatibility |",
            "|---|---|",
        ]
        lines += [f"| {_escape(c.mapping)} | {c.status} |" for c in self.mapping_checks]
        lines += [
            "",
            "## Explanation compatibility",
            "",
            "| Mapping | Explanation | Status |",
            "|---|---|---|",
        ]
        lines += [
            f"| {_escape(c.mapping)} | {_escape(c.explanation)} | {c.status} |"
            for c in self.cases
        ]
        lines += ["", "## Explanation summary", ""]
        lines += [
            f"- {_escape(row['explanation'])}: {row['status']}"
            for row in self.summary()
        ]
        lines += ["", "## Limitations", "", *["- " + x for x in _LIMITATIONS]]
        return "\n".join(lines)


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _alternatives(rows, label):
    rows = tuple(rows)
    if not rows or any(not isinstance(c, DisclosureAlternative) for c in rows):
        raise ValueError(label + " require at least one DisclosureAlternative")
    if len({c.name for c in rows}) != len(rows):
        raise ValueError(label + " names must be unique")
    return rows


def _unique(name, known):
    original, suffix = name, 0
    while name in known:
        suffix += 1
        name = f"{original}:{suffix}"
    known.add(name)
    return name


def _append(rows, choices, kind):
    known = {r.name for r in rows}
    active = []
    for index, choice in enumerate(choices):
        names = []
        for c in choice.constraints:
            name = _unique(f"{kind}:{index}:{c.name}", known)
            rows.append(
                replace(
                    c,
                    name=name,
                    metadata={
                        **(c.metadata or {}),
                        "comparison_role": kind,
                        "alternative_name": choice.name,
                        "original_constraint": c.name,
                    },
                )
            )
            names.append(name)
        active.append(tuple(names))
    return active


def compare_disclosure_explanations(
    problem: us.NamedLinearFeasibilityProblem,
    *,
    base_tier: str,
    mappings: Sequence[DisclosureAlternative],
    explanations: Sequence[DisclosureAlternative],
    evidence: Sequence[DisclosureFact] = (),
    targets: Sequence[str] | None = None,
    include_conflicts: bool = True,
) -> DisclosureExplanationComparison:
    """Cross every mapping with every explanation while preserving base history.

    Base constraints must contain only relationships common to all mappings.
    Represent disputed relationships in ``mappings``. Every alternative is an
    additive analyst policy. No fact or shared primitive is substituted.
    ``targets=()`` solves compatibility only; a fixed auxiliary probe supplies
    a checked witness even when every financial target is unbounded.
    """
    mappings, explanations = (
        _alternatives(mappings, "mapping"),
        _alternatives(explanations, "explanation"),
    )
    base = next((s for s in problem.scenarios if s.name == base_tier), None)
    if base is None:
        raise ValueError("unknown base tier")
    if type(include_conflicts) is not bool:
        raise ValueError("include_conflicts must be a boolean")
    names = (
        tuple(t.name for t in problem.targets)
        if targets is None
        else _names(targets, "targets")
    )
    if len(set(names)) != len(names) or not set(names) <= {
        t.name for t in problem.targets
    }:
        raise ValueError("targets must name unique existing targets")
    rows = [c for c in problem.constraints if c.name in base.constraints]
    mapping_rows = _append(rows, mappings, "mapping")
    explanation_rows = _append(rows, explanations, "explanation")
    probe = _unique(
        "explanation_feasibility_probe",
        set(problem.variable_names) | {t.name for t in problem.targets},
    )
    scenarios = [us.NamedLinearScenario("source", base.constraints)]
    for i, m in enumerate(mappings):
        scenarios.append(
            us.NamedLinearScenario(
                f"mapping:{i}", (*base.constraints, *mapping_rows[i]), m.description
            )
        )
        for j, e in enumerate(explanations):
            scenarios.append(
                us.NamedLinearScenario(
                    f"case:{i}:{j}",
                    (*base.constraints, *mapping_rows[i], *explanation_rows[j]),
                    m.description + "; " + e.description,
                )
            )
    compiled = replace(
        problem,
        constraints=rows,
        scenarios=scenarios,
        variables=(
            *problem.variables,
            us.NamedLinearVariable(
                probe,
                lower=0,
                upper=0,
                description="Auxiliary feasibility probe; not a financial measurement",
            ),
        ),
        targets=(
            *[t for t in problem.targets if t.name in names],
            us.NamedLinearTarget(probe, probe),
        ),
    )
    evidence = tuple(evidence)
    if evidence:
        diagnostics = validate_disclosure_evidence(evidence, problem=compiled)
        if diagnostics:
            raise ValueError(
                "invalid disclosure evidence: " + "; ".join(d.code for d in diagnostics)
            )
    report = us.solve_named_linear_feasibility(compiled)

    def case(tier, mapping=None, explanation=None):
        intervals = [
            report.interval(target=t.name, scenario=tier) for t in compiled.targets
        ]
        check = report.interval(target=probe, scenario=tier)
        if all(i.status == "infeasible" for i in intervals):
            status, witness = "infeasible", None
        elif (
            all(i.status in {"bounded", "unbounded"} for i in intervals)
            and check.lower_endpoint.status == "optimal"
        ):
            witness = check.lower_endpoint.assignment
            status = (
                "feasible"
                if witness is not None
                and all(
                    row.passed
                    for row in us.check_named_linear_assignment(
                        compiled, witness, scenario=tier
                    )
                )
                else "undetermined"
            )
        else:
            status, witness = "undetermined", None
        return DisclosureExplanationCase(
            mapping,
            explanation,
            tier,
            status,
            {i.target: i for i in intervals if i.target != probe},
            witness,
            find_disclosure_conflict(compiled, tier=tier)
            if status == "infeasible" and include_conflicts
            else None,
        )

    return DisclosureExplanationComparison(
        problem,
        base_tier,
        mappings,
        explanations,
        report,
        case("source"),
        tuple(case(f"mapping:{i}", m.name) for i, m in enumerate(mappings)),
        tuple(
            case(f"case:{i}:{j}", m.name, e.name)
            for i, m in enumerate(mappings)
            for j, e in enumerate(explanations)
        ),
        evidence,
    )
