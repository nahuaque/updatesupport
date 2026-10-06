"""Declared measurement semantics and source-preserving evidence compilation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from math import isclose, isfinite, log10
from types import MappingProxyType
from typing import Any

from .evidence import DisclosureFact, _timestamp, validate_disclosure_evidence
from .relationships import reconciliation_constraint, stock_flow_constraint


@dataclass(frozen=True)
class MeasurementDefinition:
    """Caller-declared meaning; definitions are never inferred from field names.

    ``kind`` distinguishes instant stocks, period flows, future schedules and
    lifetime commitments. ``scope`` can name cash restrictions, customer basis,
    consolidation perimeter or included debt components. ``sign`` describes
    the source convention; cash conversion remains an explicit multiplier.
    """

    name: str
    kind: str
    unit: str
    currency: str | None = None
    scope: Mapping[str, str] = field(default_factory=dict)
    sign: str = "as_reported"

    def __post_init__(self):
        if not self.name or not self.unit:
            raise ValueError("measurement name and unit are required")
        if self.kind not in {"stock", "flow", "schedule", "lifetime"}:
            raise ValueError("unknown measurement kind")
        if self.sign not in {"as_reported", "cash_inflow_positive", "asset_increase"}:
            raise ValueError("unknown sign convention")
        if any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.scope.items()
        ):
            raise ValueError("scope must map strings to strings")
        object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))

    def as_dict(self):
        return {**vars(self), "scope": dict(self.scope)}


@dataclass(frozen=True)
class MeasurementObservation:
    fact: DisclosureFact
    definition: MeasurementDefinition
    complete: bool | None = None
    components: Sequence[str] = ()
    cash_multiplier: float = 1.0

    def __post_init__(self):
        if self.complete is not None and type(self.complete) is not bool:
            raise ValueError("complete must be a boolean or unknown")
        if isinstance(self.cash_multiplier, bool) or self.cash_multiplier not in {
            -1,
            1,
        }:
            raise ValueError("cash_multiplier must explicitly be -1 or 1")
        components = tuple(self.components)
        if len(set(components)) != len(components) or any(not c for c in components):
            raise ValueError("components must be unique fact IDs")
        object.__setattr__(self, "components", components)

    def as_dict(self):
        return {
            "fact": self.fact.as_dict(),
            "definition": self.definition.as_dict(),
            "complete": self.complete,
            "components": list(self.components),
            "cash_multiplier": self.cash_multiplier,
        }


@dataclass(frozen=True)
class DisclosureRequirement:
    name: str
    definition: MeasurementDefinition
    entity: str
    period_end: str | None = None
    period_start: str | None = None
    max_age_days: int | None = None
    required_components: Sequence[str] = ()
    require_complete: bool = False

    def __post_init__(self):
        if not self.name or not self.entity:
            raise ValueError("requirement name and entity are required")
        for value in (self.period_start, self.period_end):
            if value is not None:
                date.fromisoformat(value)
        if (
            self.period_start
            and self.period_end
            and self.period_start > self.period_end
        ):
            raise ValueError("requirement period is reversed")
        if self.max_age_days is not None and (
            type(self.max_age_days) is not int or self.max_age_days < 0
        ):
            raise ValueError("max_age_days must be a nonnegative integer")
        if type(self.require_complete) is not bool:
            raise ValueError("require_complete must be a boolean")
        components = tuple(self.required_components)
        if len(set(components)) != len(components) or any(not c for c in components):
            raise ValueError("required components must be unique fact IDs")
        object.__setattr__(self, "required_components", components)

    def as_dict(self):
        return {
            **vars(self),
            "definition": self.definition.as_dict(),
            "required_components": list(self.required_components),
        }


@dataclass(frozen=True)
class NormalizedDisclosureFact:
    source: DisclosureFact
    fact: DisclosureFact
    source_units_per_solver_unit: float
    cash_multiplier: float

    def as_dict(self):
        return {
            "source": self.source.as_dict(),
            "normalized": self.fact.as_dict(),
            "source_units_per_solver_unit": self.source_units_per_solver_unit,
            "cash_multiplier": self.cash_multiplier,
        }


def normalize_disclosure_fact(
    fact: DisclosureFact,
    *,
    unit: str,
    source_units_per_solver_unit: float = 1,
    cash_multiplier: float = 1,
) -> NormalizedDisclosureFact:
    """Convert numerical units/sign while retaining the original source fact.

    Factors must be powers of ten so XBRL precision remains exactly expressible.
    Currency never changes. This is numerical normalization, not an FX or
    accounting conversion. The solver tolerance is not loosened.
    """
    factor = float(source_units_per_solver_unit)
    if not unit or not isfinite(factor) or factor <= 0:
        raise ValueError("a positive finite unit factor and unit are required")
    exponent = round(log10(factor))
    if not isclose(log10(factor), exponent, abs_tol=1e-12, rel_tol=0):
        raise ValueError("unit factors must be powers of ten")
    if isinstance(cash_multiplier, bool) or cash_multiplier not in {-1, 1}:
        raise ValueError("cash_multiplier must be -1 or 1")
    normalized = replace(
        fact,
        value=fact.base_value * cash_multiplier / factor,
        scale=1,
        unit=unit,
        decimals=None if fact.decimals is None else fact.decimals + exponent,
    )
    return NormalizedDisclosureFact(fact, normalized, factor, cash_multiplier)


def validate_normalization_records(records, facts):
    """Validate portable source-to-solver maps, including precision and signs."""
    lookup = {f.fact_id: f for f in facts}
    seen = set()
    for row in records:
        source = DisclosureFact(**row["source"])
        normalized = DisclosureFact(**row["normalized"])
        expected = normalize_disclosure_fact(
            source,
            unit=normalized.unit,
            source_units_per_solver_unit=row["source_units_per_solver_unit"],
            cash_multiplier=row["cash_multiplier"],
        ).fact
        if (
            source.fact_id in seen
            or expected != normalized
            or lookup.get(source.fact_id) != normalized
        ):
            raise ValueError("invalid or duplicate source normalization record")
        seen.add(source.fact_id)


@dataclass(frozen=True)
class CompiledDisclosureEvidence:
    facts: tuple[DisclosureFact, ...]
    normalizations: tuple[NormalizedDisclosureFact, ...]
    ledger: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    as_of: str

    def require_valid(self):
        bad = [r for r in self.ledger if r["status"] != "satisfied"]
        if bad or self.diagnostics:
            raise ValueError(
                "incomplete disclosure evidence: "
                + "; ".join(
                    [
                        f"{r['requirement']}: {r['status']} ({', '.join(r['issues'])})"
                        for r in bad
                    ]
                    + [r["code"] for r in self.diagnostics]
                )
            )
        return self

    def fact_for(self, requirement):
        if self.diagnostics:
            raise ValueError(
                "compiled evidence has unresolved lineage/context diagnostics"
            )
        row = next((r for r in self.ledger if r["requirement"] == requirement), None)
        if row is None or row["status"] != "satisfied":
            raise ValueError("requirement is absent or not satisfied")
        return next(f for f in self.facts if f.fact_id == row["fact_id"])

    def snapshot_context(self):
        return {
            "normalizations": [r.as_dict() for r in self.normalizations],
            "measurement_ledger": [dict(r) for r in self.ledger],
            "measurement_diagnostics": [dict(r) for r in self.diagnostics],
        }


def compile_disclosure_evidence(
    observations: Sequence[MeasurementObservation],
    *,
    requirements: Sequence[DisclosureRequirement],
    as_of: str,
    unit_scales: Mapping[str, tuple[str, float]] | None = None,
) -> CompiledDisclosureEvidence:
    """Select available versions and quarantine incompatible/partial inputs.

    Completeness is assessed against supplied component requirements; unknown
    provider omissions cannot be detected without that contract. Missing inputs
    never become zero. Dependencies are retained for existing lineage checks.
    """
    observations, requirements = tuple(observations), tuple(requirements)
    cutoff = _timestamp(as_of)
    if len({r.name for r in requirements}) != len(requirements):
        raise ValueError("requirement names must be unique")
    by_id = {}
    for obs in observations:
        previous = by_id.get(obs.fact.fact_id)
        if previous is not None and previous != obs:
            raise ValueError("conflicting observation fact IDs")
        by_id[obs.fact.fact_id] = obs

    def dependency_issues(obs, path=()):
        fid = obs.fact.fact_id
        if fid in path:
            return ["cyclic_dependency:" + fid]
        result = []
        if obs.complete is False:
            result.append("incomplete_dependency:" + fid)
        if obs.fact.available_at > cutoff:
            result.append("unavailable_dependency:" + fid)
        if (
            obs.fact.unit != obs.definition.unit
            or obs.fact.currency != obs.definition.currency
            or (obs.definition.kind == "stock" and obs.fact.period_start is not None)
            or (obs.definition.kind == "flow" and obs.fact.period_start is None)
        ):
            result.append("incompatible_dependency:" + fid)
        for parent in set((*obs.fact.derivation_inputs, *obs.components)):
            if parent not in by_id:
                result.append("missing_dependency:" + parent)
            else:
                result.extend(dependency_issues(by_id[parent], (*path, fid)))
        return result

    selected, ledger = {}, []
    today = date.fromisoformat(cutoff[:10])
    for req in requirements:
        candidates = [
            o
            for o in observations
            if o.definition.name == req.definition.name and o.fact.entity == req.entity
        ]
        issues, status, chosen = [], "missing", None
        available = [o for o in candidates if o.fact.available_at <= cutoff]
        if candidates and not available:
            status = "unavailable"
        if available:
            compatible = [
                o
                for o in available
                if o.definition == req.definition
                and o.fact.unit == req.definition.unit
                and o.fact.currency == req.definition.currency
                and (req.definition.kind != "stock" or o.fact.period_start is None)
                and (req.definition.kind != "flow" or o.fact.period_start is not None)
                and (
                    req.period_start is None or o.fact.period_start == req.period_start
                )
                and (req.period_end is None or o.fact.period_end == req.period_end)
            ]
            status = "incompatible"
            if compatible:
                key = max((o.fact.period_end, o.fact.available_at) for o in compatible)
                latest = [
                    o
                    for o in compatible
                    if (o.fact.period_end, o.fact.available_at) == key
                ]
                # Equal metadata does not establish equivalence of competing versions.
                unique = {o.fact.fact_id: o for o in latest}
                status = "ambiguous" if len(unique) > 1 else "satisfied"
                if len(unique) == 1:
                    chosen = next(iter(unique.values()))
                    if (
                        req.definition.kind in {"stock", "flow"}
                        and chosen.fact.period_end > cutoff[:10]
                    ):
                        status, issues = "incompatible", ["future_accounting_period"]
                    elif (
                        req.max_age_days is not None
                        and (today - date.fromisoformat(chosen.fact.period_end)).days
                        > req.max_age_days
                    ):
                        status, issues = "stale", ["period_age_exceeds_policy"]
                    missing = [
                        c
                        for c in req.required_components
                        if c not in chosen.components
                        or c not in by_id
                        or by_id[c].fact.available_at > cutoff
                    ]
                    if (
                        chosen.complete is False
                        or (req.require_complete and chosen.complete is not True)
                        or missing
                    ):
                        status = "incomplete"
                        issues.extend(["component:" + c for c in missing])
                        if chosen.complete is not True:
                            issues.append("completeness_unconfirmed")
                    dependencies = dependency_issues(chosen)
                    if dependencies:
                        status = "incomplete"
                        issues.extend(dependencies)
                    if status == "satisfied":
                        selected[chosen.fact.fact_id] = chosen
        ledger.append(
            {
                "requirement": req.name,
                "status": status,
                "fact_id": None if chosen is None else chosen.fact.fact_id,
                "issues": issues,
                "definition": req.definition.as_dict(),
                "required_components": list(req.required_components),
            }
        )
    # Preserve lineage, including explicitly required components, without hiding
    # invalid dependencies by dropping them from validation.
    pending = list(selected.values())
    while pending:
        obs = pending.pop()
        for parent in (*obs.fact.derivation_inputs, *obs.components):
            if parent in by_id and parent not in selected:
                selected[parent] = by_id[parent]
                pending.append(by_id[parent])
    normalizations = []
    for obs in selected.values():
        unit, factor = (unit_scales or {}).get(obs.fact.unit, (obs.fact.unit, 1))
        normalizations.append(
            normalize_disclosure_fact(
                obs.fact,
                unit=unit,
                source_units_per_solver_unit=factor,
                cash_multiplier=obs.cash_multiplier,
            )
        )
    facts = tuple(r.fact for r in normalizations)
    diagnostics = tuple(
        r.as_dict() for r in validate_disclosure_evidence(facts, as_of=cutoff)
    )
    return CompiledDisclosureEvidence(
        facts, tuple(normalizations), tuple(ledger), diagnostics, cutoff
    )


@dataclass(frozen=True)
class CashMeasure:
    name: str
    components: Mapping[str, float]
    scope: str

    def __post_init__(self):
        if not self.name or not self.scope or not self.components:
            raise ValueError("cash measures require name, components and scope")
        terms = {k: float(v) for k, v in self.components.items()}
        if any(not k or not isfinite(v) for k, v in terms.items()):
            raise ValueError("invalid cash measure components")
        object.__setattr__(self, "components", MappingProxyType(terms))


def cash_measure_constraints(
    measures: Sequence[CashMeasure], *, primitives: Sequence[str]
):
    """Expand shared primitives; reject cycles and undeclared cash movements."""
    measures, primitives = tuple(measures), tuple(primitives)
    lookup = {m.name: m for m in measures}
    if (
        len(lookup) != len(measures)
        or len(set(primitives)) != len(primitives)
        or set(lookup) & set(primitives)
    ):
        raise ValueError("cash measure and primitive names must be unique")
    resolved = {}

    def expand(name, path=()):
        if name in primitives:
            return {name: 1.0}
        if name in path:
            raise ValueError("cyclic cash measures")
        if name not in lookup:
            raise ValueError("undeclared cash primitive")
        if name not in resolved:
            terms = {}
            for child, weight in lookup[name].components.items():
                for primitive, coefficient in expand(child, (*path, name)).items():
                    terms[primitive] = terms.get(primitive, 0) + weight * coefficient
            resolved[name] = {k: v for k, v in terms.items() if v}
        return resolved[name]

    result = []
    for measure in measures:
        terms = expand(measure.name)
        if not terms:
            raise ValueError("cash measure expands to zero")
        result.append(
            reconciliation_constraint(
                measure.name + ":definition",
                total=measure.name,
                components=terms,
                metadata={
                    "evidence_role": "accounting_relationship",
                    "cash_scope": measure.scope,
                    "primitive_components": terms,
                },
            )
        )
    return tuple(result)


def cash_bridge_constraint(
    name,
    *,
    opening,
    closing,
    movements,
    opening_scope,
    closing_scope,
    scope_adjustment=None,
    provenance=None,
):
    """Reconcile scopes explicitly; adjustment is closing-scope minus opening-scope cash."""
    if not opening_scope or not closing_scope:
        raise ValueError("cash scopes must be supplied")
    if opening_scope != closing_scope and scope_adjustment is None:
        raise ValueError("different cash scopes require an explicit scope adjustment")
    result = stock_flow_constraint(
        name,
        opening=opening,
        closing=closing,
        movements=movements,
        residual=scope_adjustment,
        provenance=provenance,
    )
    return replace(
        result,
        metadata={
            "evidence_role": "accounting_relationship",
            "opening_scope": opening_scope,
            "closing_scope": closing_scope,
            "scope_adjustment": scope_adjustment,
        },
    )
