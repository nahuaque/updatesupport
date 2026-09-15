"""Vendor-neutral financial facts and explicit evidence-to-constraint links."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from math import isfinite
from types import MappingProxyType
from typing import Any

import updatesupport as us


def _timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("availability timestamps must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class DisclosureFact:
    """One version of one fact, with precision expressed in base units.

    ``value * scale`` is the base-unit amount. XBRL-style ``decimals=-6``
    denotes rounding to millions of base units. ``None`` means unknown
    precision, not exactness. ``available_at`` is the evidence availability
    timestamp; it must not be inferred from the accounting period end.
    """

    fact_id: str
    entity: str
    concept: str
    value: float
    unit: str
    period_end: str
    available_at: str
    source_id: str
    source_url: str
    period_start: str | None = None
    currency: str | None = None
    scale: float = 1.0
    decimals: int | None = None
    dimensions: Mapping[str, str] = field(default_factory=dict)
    reported: bool = True
    derivation: str | None = None
    derivation_inputs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for name in ("fact_id", "entity", "concept", "unit", "source_id", "source_url"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("value", "scale"):
            value = float(getattr(self, name))
            if not isfinite(value) or (name == "scale" and value <= 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not isfinite(self.base_value):
            raise ValueError("scaled value must be finite")
        end = date.fromisoformat(self.period_end)
        if (
            self.period_start is not None
            and date.fromisoformat(self.period_start) > end
        ):
            raise ValueError("period_start must not follow period_end")
        object.__setattr__(self, "available_at", _timestamp(self.available_at))
        if self.decimals is not None:
            if isinstance(self.decimals, bool) or not isinstance(self.decimals, int):
                raise ValueError("decimals must be an integer or None")
            if not -300 <= self.decimals <= 300:
                raise ValueError("decimals must be between -300 and 300")
        if not isinstance(self.reported, bool):
            raise ValueError("reported must be a boolean")
        inputs = tuple(self.derivation_inputs)
        if self.reported and (self.derivation or inputs):
            raise ValueError("reported facts cannot carry a derivation")
        if not self.reported and (not self.derivation or not inputs):
            raise ValueError("derived facts require a derivation and input fact IDs")
        if self.fact_id in inputs:
            raise ValueError("a fact cannot derive from itself")
        if any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.dimensions.items()
        ):
            raise ValueError("dimensions must map string axes to string members")
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "derivation_inputs", inputs)

    @property
    def base_value(self) -> float:
        return self.value * self.scale

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "dimensions": dict(self.dimensions),
            "derivation_inputs": list(self.derivation_inputs),
        }


@dataclass(frozen=True)
class DisclosureEvidenceDiagnostic:
    code: str
    fact_ids: tuple[str, ...]
    message: str
    constraint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "fact_ids": list(self.fact_ids)}


_CONTEXT_FIELDS = ("entity", "unit", "currency", "period", "dimensions")


def _context(fact: DisclosureFact, name: str) -> Any:
    if name == "period":
        return fact.period_start, fact.period_end
    if name == "dimensions":
        return tuple(sorted(fact.dimensions.items()))
    return getattr(fact, name)


def validate_disclosure_evidence(
    facts: Sequence[DisclosureFact],
    *,
    problem: us.NamedLinearFeasibilityProblem | None = None,
    as_of: str | None = None,
) -> tuple[DisclosureEvidenceDiagnostic, ...]:
    """Check versions, availability, derivation lineage, and linked contexts.

    Different periods or dimensions may be intentional in a bridge or rollup;
    each link must name these differences in ``varying_fields``. This checks
    structural compatibility, not the truth of an accounting relationship.
    """
    cutoff = None if as_of is None else _timestamp(as_of)
    rows: list[DisclosureEvidenceDiagnostic] = []

    def add(
        code: str, ids: Sequence[str], message: str, constraint: str | None = None
    ) -> None:
        rows.append(DisclosureEvidenceDiagnostic(code, tuple(ids), message, constraint))

    lookup: dict[str, DisclosureFact] = {}
    for fact in facts:
        if fact.fact_id in lookup:
            add("duplicate_id", [fact.fact_id], "Fact IDs must be unique.")
        lookup[fact.fact_id] = fact
        if cutoff is not None and fact.available_at > cutoff:
            add(
                "future_evidence",
                [fact.fact_id],
                "Fact was unavailable at the requested as-of time.",
            )
    # Historical derivation inputs can retain old versions. Check the versions
    # used together in each scenario, not mutually exclusive assumption tiers.
    scopes = (
        [(None, set(lookup))]
        if problem is None
        else [
            (
                scenario.name,
                {
                    fact_id
                    for constraint in problem.constraints
                    if constraint.name in scenario.constraints
                    for fact_id in (constraint.metadata or {}).get("fact_ids", ())
                },
            )
            for scenario in problem.scenarios
        ]
    )
    for scenario_name, active_ids in scopes:
        contexts: dict[tuple[Any, ...], DisclosureFact] = {}
        for fact in facts:
            if fact.fact_id not in active_ids:
                continue
            key = (fact.concept, *(_context(fact, name) for name in _CONTEXT_FIELDS))
            previous = contexts.get(key)
            if previous is not None:
                code = (
                    "duplicate_context"
                    if previous.source_id == fact.source_id
                    else "mixed_versions"
                )
                add(
                    code,
                    [previous.fact_id, fact.fact_id],
                    f"Select one version per modeled fact context (scenario={scenario_name!r}).",
                )
            contexts[key] = fact
    for fact in facts:
        for parent_id in fact.derivation_inputs:
            parent = lookup.get(parent_id)
            if parent is None:
                add(
                    "missing_derivation_input",
                    [fact.fact_id, parent_id],
                    "Derivation input is absent.",
                )
            elif parent.available_at > fact.available_at:
                add(
                    "future_derivation_input",
                    [fact.fact_id, parent_id],
                    "Derived fact predates an input.",
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(fact_id: str) -> None:
        if fact_id in visiting:
            add("cyclic_derivation", [fact_id], "Derivation inputs contain a cycle.")
            return
        if fact_id in visited or fact_id not in lookup:
            return
        visiting.add(fact_id)
        for parent_id in lookup[fact_id].derivation_inputs:
            visit(parent_id)
        visiting.remove(fact_id)
        visited.add(fact_id)

    for fact_id in lookup:
        visit(fact_id)
    if problem is not None:
        for constraint in problem.constraints:
            metadata = constraint.metadata or {}
            ids = tuple(metadata.get("fact_ids", ()))
            varying = set(metadata.get("varying_fields", ()))
            if varying - set(_CONTEXT_FIELDS):
                add(
                    "unknown_context_field",
                    ids,
                    "Unknown varying_fields entry.",
                    constraint.name,
                )
            missing = [fact_id for fact_id in ids if fact_id not in lookup]
            if missing:
                add(
                    "missing_fact",
                    missing,
                    "Constraint references absent facts.",
                    constraint.name,
                )
                continue
            variable_name = metadata.get("fact_variable")
            if variable_name is not None and len(ids) == 1:
                fact = lookup[ids[0]]
                variable = next(
                    (v for v in problem.variables if v.name == variable_name), None
                )
                if variable is None or variable.unit != fact.unit:
                    add(
                        "incompatible_variable_unit",
                        ids,
                        "A fact variable must declare the fact's base unit.",
                        constraint.name,
                    )
                if not metadata.get("exact") and fact.decimals is None:
                    add(
                        "unknown_precision",
                        ids,
                        "Fact precision is unspecified.",
                        constraint.name,
                    )
                else:
                    radius = (
                        0.0 if metadata.get("exact") else 0.5 * 10.0 ** (-fact.decimals)
                    )
                    if (
                        constraint.expression.coefficients != {variable_name: 1.0}
                        or constraint.expression.constant != 0.0
                        or constraint.lower != fact.base_value - radius
                        or constraint.upper != fact.base_value + radius
                    ):
                        add(
                            "stale_fact_constraint",
                            ids,
                            "Compiled bounds no longer match the linked fact.",
                            constraint.name,
                        )
            for name in _CONTEXT_FIELDS:
                if (
                    name not in varying
                    and len({_context(lookup[i], name) for i in ids}) > 1
                ):
                    add(
                        "incompatible_" + name,
                        ids,
                        f"Linked facts differ in {name}; declare the relationship explicitly.",
                        constraint.name,
                    )
    return tuple(rows)


def link_disclosure_evidence(
    constraint: us.NamedLinearConstraint,
    facts: Sequence[DisclosureFact],
    *,
    varying_fields: Sequence[str] = (),
) -> us.NamedLinearConstraint:
    """Attach stable source IDs and declare intentional context differences."""
    if not facts:
        raise ValueError("an evidence link requires at least one fact")
    if set(varying_fields) - set(_CONTEXT_FIELDS):
        raise ValueError("unknown varying_fields entry")
    linked = replace(
        constraint,
        metadata={
            **(constraint.metadata or {}),
            "fact_ids": [row.fact_id for row in facts],
            "varying_fields": list(varying_fields),
        },
    )
    return linked


def disclosure_fact_constraint(
    name: str,
    variable: str,
    fact: DisclosureFact,
    *,
    exact: bool = False,
) -> us.NamedLinearConstraint:
    """Compile a fact to a base-unit interval; exactness requires opt-in.

    Rounding bounds are inclusive relaxations. The variable must use the fact's
    base unit; the helper does not perform currency or accounting conversions.
    """
    if not exact and fact.decimals is None:
        raise ValueError(
            "unknown precision: supply decimals or explicitly choose exact=True"
        )
    radius = 0.0 if exact else 0.5 * 10.0 ** (-fact.decimals)
    constraint = us.named_linear_constraint(
        name,
        variable,
        lower=fact.base_value - radius,
        upper=fact.base_value + radius,
        kind="reported_fact" if fact.reported else "derived_fact",
        provenance=fact.source_url,
        metadata={"exact": exact, "rounding_radius": radius, "fact_variable": variable},
    )
    return link_disclosure_evidence(constraint, [fact])
