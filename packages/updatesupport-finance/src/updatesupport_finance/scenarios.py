"""Declared evidence roles, constraint policies and complete named stress solves."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

import updatesupport as us

from .disclosure import find_disclosure_conflict


class EvidenceRole(str, Enum):
    REPORTED_FACT = "reported_fact"
    ACCOUNTING_RELATIONSHIP = "accounting_relationship"
    MANAGEMENT_EXPECTATION = "management_expectation"
    ANALYST_POLICY = "analyst_policy"


def constraint_evidence_role(constraint):
    role = (constraint.metadata or {}).get("evidence_role")
    if role is not None:
        return EvidenceRole(role).value
    return {
        "reported_fact": "reported_fact",
        "derived_fact": "accounting_relationship",
        "reconciliation": "accounting_relationship",
        "stock_flow": "accounting_relationship",
    }.get(constraint.kind, "unclassified")


@dataclass(frozen=True)
class ConstraintPolicy:
    name: str
    role: EvidenceRole | str
    constraints: Sequence[us.NamedLinearConstraint]

    def __post_init__(self):
        if not self.name or not self.constraints:
            raise ValueError("policies need a name and constraints")
        role = EvidenceRole(self.role).value
        if len({c.name for c in self.constraints}) != len(self.constraints):
            raise ValueError("policy constraint names must be unique")
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "constraints",
            tuple(
                replace(
                    c,
                    metadata={
                        **(c.metadata or {}),
                        "evidence_role": role,
                        "policy_name": self.name,
                    },
                )
                for c in self.constraints
            ),
        )

    def as_dict(self):
        return {
            "name": self.name,
            "role": self.role,
            "constraints": [c.as_dict() for c in self.constraints],
        }


def disclosure_support_basis(problem, tier):
    scenario = next((s for s in problem.scenarios if s.name == tier), None)
    if scenario is None:
        raise ValueError("unknown tier")
    active = [c for c in problem.constraints if c.name in scenario.constraints]
    roles = sorted({constraint_evidence_role(c) for c in active})
    conditional = set(roles) - {"reported_fact", "accounting_relationship"}
    return {
        "basis": "conditional" if conditional else "reported_evidence",
        "roles": roles,
        "policies": sorted(
            {
                c.metadata["policy_name"]
                for c in active
                if (c.metadata or {}).get("policy_name")
            }
        ),
        "interpretation": "Support is conditional on declared measurement and accounting relationships.",
    }


@dataclass(frozen=True)
class DisclosureStressCase:
    name: str
    base_tier: str
    replacements: Mapping[str, us.NamedLinearConstraint | None]
    description: str
    historical_counterfactual: bool = False
    model_counterfactual: bool = False

    def __post_init__(self):
        if (
            not self.name
            or not self.base_tier
            or not self.description
            or not self.replacements
        ):
            raise ValueError(
                "stress cases need a name, base tier, replacements and description"
            )
        if (
            type(self.historical_counterfactual) is not bool
            or type(self.model_counterfactual) is not bool
        ):
            raise ValueError("counterfactual switches must be explicit booleans")
        object.__setattr__(
            self, "replacements", MappingProxyType(dict(self.replacements))
        )


def compile_disclosure_scenarios(problem, cases):
    """Replace named assumptions and re-solve every dependent identity.

    No endpoint is patched after solving. Changes to history or accounting
    equations require explicit counterfactual declarations. Unknown evidence
    roles must first be classified with ConstraintPolicy.
    """
    constraints, scenarios = list(problem.constraints), list(problem.scenarios)
    lookup = {c.name: c for c in constraints}
    tiers = {s.name: s for s in scenarios}
    for case in cases:
        if case.name in tiers or case.base_tier not in tiers:
            raise ValueError("stress names must be unique and base tier must exist")
        base = tiers[case.base_tier]
        if not set(case.replacements) <= set(base.constraints):
            raise ValueError("stress replacements must name active base constraints")
        active = []
        for name in base.constraints:
            if name not in case.replacements:
                active.append(name)
                continue
            original = lookup[name]
            role = constraint_evidence_role(original)
            if (
                role == "reported_fact"
                or original.kind in {"reported_fact", "derived_fact"}
            ) and not case.historical_counterfactual:
                raise ValueError(
                    "changing reported history requires historical_counterfactual=True"
                )
            if (
                role == "accounting_relationship"
                and original.kind != "derived_fact"
                and not case.model_counterfactual
            ):
                raise ValueError(
                    "changing accounting relationships requires model_counterfactual=True"
                )
            if role == "unclassified":
                raise ValueError(
                    "classify the original constraint's evidence role before stressing it"
                )
            replacement = case.replacements[name]
            if replacement is None:
                continue
            metadata = dict(replacement.metadata or {})
            # An altered observation is no longer a source fact. Retain lineage
            # without falsely linking the counterfactual bounds to reported data.
            for key in ("fact_variable", "fact_ids", "exact", "rounding_radius"):
                metadata.pop(key, None)
            metadata.update(
                evidence_role="analyst_policy",
                policy_name=case.name,
                replaces=name,
                original_role=role,
                original_fact_ids=list((original.metadata or {}).get("fact_ids", ())),
            )
            cloned = replace(
                replacement, name=f"stress:{case.name}:{name}", metadata=metadata
            )
            constraints.append(cloned)
            active.append(cloned.name)
        scenario = us.NamedLinearScenario(
            case.name, active, description=case.description
        )
        scenarios.append(scenario)
        tiers[case.name] = scenario
        lookup.update({c.name: c for c in constraints})
    return replace(problem, constraints=constraints, scenarios=scenarios)


@dataclass(frozen=True)
class DisclosureScenarioAnalysis:
    report: us.NamedLinearFeasibilityReport
    conflicts: Mapping
    support: Mapping

    def as_dict(self):
        return {
            "report": self.report.as_dict(),
            "conflicts": {k: v.as_dict() for k, v in self.conflicts.items()},
            "support": dict(self.support),
        }


def run_disclosure_scenarios(problem, cases=()):
    compiled = compile_disclosure_scenarios(problem, cases)
    report = us.solve_named_linear_feasibility(compiled)
    conflicts = {
        s.name: find_disclosure_conflict(compiled, tier=s.name)
        for s in compiled.scenarios
        if any(
            report.interval(target=t.name, scenario=s.name).status == "infeasible"
            for t in compiled.targets
        )
    }
    return DisclosureScenarioAnalysis(
        report,
        conflicts,
        {
            s.name: disclosure_support_basis(compiled, s.name)
            for s in compiled.scenarios
        },
    )


def break_even_analysis(
    problem, *, target, tier, conditions, release_constraints=(), name="break_even"
):
    """Bound an existing decision variable under explicit decision conditions.

    For example, release an issuance assumption and require ending cash to equal
    opening cash to bound required issuance. A >= cash floor instead gives a
    minimum with an often unbounded upper endpoint. No risk limit is invented.
    """
    base = next((s for s in problem.scenarios if s.name == tier), None)
    if (
        base is None
        or target not in {t.name for t in problem.targets}
        or not conditions
    ):
        raise ValueError(
            "break-even requires a known target/tier and explicit conditions"
        )
    release = tuple(release_constraints)
    if not set(release) <= set(base.constraints):
        raise ValueError("release constraints must be active in the base tier")
    lookup = {c.name: c for c in problem.constraints}
    if any(
        (
            constraint_evidence_role(lookup[n])
            not in {"analyst_policy", "management_expectation"}
            or lookup[n].kind in {"reported_fact", "derived_fact"}
        )
        for n in release
    ):
        raise ValueError(
            "break-even can only release declared policies or expectations"
        )
    policy = ConstraintPolicy(name, EvidenceRole.ANALYST_POLICY, conditions)
    compiled = replace(
        problem,
        constraints=(*problem.constraints, *policy.constraints),
        scenarios=(
            *problem.scenarios,
            us.NamedLinearScenario(
                name,
                (
                    *[n for n in base.constraints if n not in release],
                    *[c.name for c in policy.constraints],
                ),
            ),
        ),
    )
    report = us.solve_named_linear_feasibility(compiled)
    return DisclosureScenarioAnalysis(
        report,
        {name: find_disclosure_conflict(compiled, tier=name)}
        if report.interval(target=target, scenario=name).status == "infeasible"
        else {},
        {name: disclosure_support_basis(compiled, name)},
    )
