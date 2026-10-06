"""Declared disclosure hierarchies and margins compiled to linear problems."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import updatesupport as us

from .evidence import (
    DisclosureFact,
    disclosure_fact_constraint,
    validate_disclosure_evidence,
)
from .relationships import reconciliation_constraint
from .shares import (
    AllocationShareMargin,
    ShareThreshold,
    percentage_constraints,
    share_threshold_target,
)


@dataclass(frozen=True)
class AllocationMember:
    name: str
    role: str = "leaf"
    children: Sequence[str] = ()
    qualifier: str | None = None

    def __post_init__(self):
        if not self.name or self.role not in {"leaf", "subtotal"}:
            raise ValueError("allocation members need a name and leaf/subtotal role")
        children = tuple(self.children)
        if (self.role == "subtotal") != bool(children) or len(set(children)) != len(
            children
        ):
            raise ValueError("only subtotals have nonempty, unique children")
        object.__setattr__(self, "children", children)

    def as_dict(self):
        return {**vars(self), "children": list(self.children)}


@dataclass(frozen=True)
class AllocationMargin:
    name: str
    fact: DisclosureFact
    rows: Sequence[str] | None = None
    columns: Sequence[str] | None = None
    exact: bool = False
    measure: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("margin name is required")
        if not isinstance(self.exact, bool):
            raise ValueError("exact must be an explicit boolean")
        for axis in ("rows", "columns"):
            value = getattr(self, axis)
            if isinstance(value, str):
                raise TypeError("margin selectors must be sequences of member names")
            if value is not None:
                object.__setattr__(self, axis, tuple(value))


def _axis(members):
    members = tuple(members)
    by_name = {m.name: m for m in members}
    if not members or len(by_name) != len(members):
        raise ValueError("axis members must be nonempty and unique")
    resolved = {}

    def descend(name, path=()):
        if name not in by_name:
            raise ValueError(f"unknown hierarchy member {name!r}")
        if name in path:
            raise ValueError("cyclic allocation hierarchy")
        if name in resolved:
            return resolved[name]
        member = by_name[name]
        leaves = (
            [name]
            if member.role == "leaf"
            else [x for c in member.children for x in descend(c, (*path, name))]
        )
        if len(set(leaves)) != len(leaves):
            raise ValueError(f"overlapping children in subtotal {name!r}")
        resolved[name] = tuple(leaves)
        return resolved[name]

    for name in by_name:
        descend(name)
    leaves = tuple(m.name for m in members if m.role == "leaf")
    return members, leaves, resolved


def _select(names, leaves, hierarchy):
    if names is None:
        return leaves
    if isinstance(names, str):
        raise TypeError("selectors must be sequences")
    result = []
    for name in names:
        if name not in hierarchy:
            raise ValueError(f"unknown allocation member {name!r}")
        result.extend(hierarchy[name])
    if not result or len(set(result)) != len(result):
        raise ValueError(
            "selection must be nonempty and cannot overlap subtotal/children"
        )
    return tuple(result)


@dataclass(frozen=True)
class AllocationTable:
    row_members: tuple[AllocationMember, ...]
    column_members: tuple[AllocationMember, ...]
    cell_variables: Mapping[tuple[str, str], str]
    variables: tuple
    constraints: tuple
    facts: tuple[DisclosureFact, ...]
    assumptions: tuple[str, ...]
    unit: str
    as_of: str
    title: str
    amount_margins: tuple[AllocationMargin, ...] = ()

    def target(self, name, *, rows=None, columns=None):
        _, row_leaves, row_hierarchy = _axis(self.row_members)
        _, col_leaves, col_hierarchy = _axis(self.column_members)
        selected_rows = _select(rows, row_leaves, row_hierarchy)
        selected_cols = _select(columns, col_leaves, col_hierarchy)
        return us.NamedLinearTarget(
            name,
            {
                self.cell_variables[r, c]: 1.0
                for r in selected_rows
                for c in selected_cols
            },
            unit=self.unit,
        )

    def share_target(
        self, name, *, denominator, threshold, rows=None, columns=None, label=None
    ):
        by_name = {m.name: (i, m) for i, m in enumerate(self.amount_margins)}
        if denominator not in by_name:
            raise ValueError("share denominator must name an amount margin")
        index, margin = by_name[denominator]
        _, row_leaves, row_hierarchy = _axis(self.row_members)
        _, col_leaves, col_hierarchy = _axis(self.column_members)
        if not set(_select(rows, row_leaves, row_hierarchy)) <= set(
            _select(margin.rows, row_leaves, row_hierarchy)
        ) or not set(_select(columns, col_leaves, col_hierarchy)) <= set(
            _select(margin.columns, col_leaves, col_hierarchy)
        ):
            raise ValueError(
                "share numerator must be contained in its denominator selection"
            )
        if margin.fact.decimals is None:
            raise ValueError(
                "denominator needs known precision to establish positivity"
            )
        numerator = self.target(name, rows=rows, columns=columns).expression
        return share_threshold_target(
            name,
            numerator=numerator,
            denominator=f"margin_{index}",
            threshold=threshold,
            denominator_lower_bound=margin.fact.base_value
            - 0.5 * 10 ** (-margin.fact.decimals),
            denominator_fact=margin.fact,
            unit=self.unit,
            label=label,
        )

    def problem(self, targets, *, scenarios=None):
        targets = tuple(targets)
        positivity = tuple(
            t.positivity for t in targets if isinstance(t, ShareThreshold)
        )
        constraints = self.constraints + positivity
        targets = tuple(
            t.target if isinstance(t, ShareThreshold) else t for t in targets
        )
        if scenarios is None:
            scenarios = (
                us.NamedLinearScenario("reported", tuple(c.name for c in constraints)),
            )
        else:
            scenarios = tuple(
                us.NamedLinearScenario(
                    s.name,
                    tuple(
                        dict.fromkeys((*s.constraints, *(c.name for c in positivity)))
                    ),
                    description=s.description,
                )
                for s in scenarios
            )
        return us.NamedLinearFeasibilityProblem(
            self.variables,
            constraints,
            targets,
            scenarios,
            title=self.title,
            description="; ".join(self.assumptions),
        )

    def as_dict(self):
        return {
            "row_members": [m.as_dict() for m in self.row_members],
            "column_members": [m.as_dict() for m in self.column_members],
            "cells": [
                {"row": r, "column": c, "variable": v}
                for (r, c), v in self.cell_variables.items()
            ],
            "variables": [v.as_dict() for v in self.variables],
            "constraints": [c.as_dict() for c in self.constraints],
            "facts": [f.as_dict() for f in self.facts],
            "assumptions": list(self.assumptions),
            "unit": self.unit,
            "as_of": self.as_of,
            "title": self.title,
        }


def allocation_table(
    *,
    rows: Sequence[AllocationMember],
    columns: Sequence[AllocationMember],
    margins: Sequence[AllocationMargin | AllocationShareMargin],
    measure: str,
    unit: str,
    as_of: str,
    exhaustive: bool,
    structural_zeros: Sequence[tuple[str, str]] = (),
    assumptions: Sequence[str] = (),
    title: str = "Disclosure Allocation Table",
) -> AllocationTable:
    """Build nonnegative leaf cells, reported margins, and declared exclusions.

    ``exhaustive=True`` is the caller's assertion that both leaf axes partition
    this measure. Add explicit 'other' leaves when needed. Subtotals describe
    leaf unions and are never added as extra cells. Qualifiers remain literal
    labels; e.g. 'China including HK/TW' is not inferred to mean mainland China.
    Unknown reporting precision requires an explicit exact=True per margin.
    A margin with a different source concept needs an explicit measure override
    declaring equivalence; the original fact concept remains in evidence.
    """
    if exhaustive is not True:
        raise ValueError(
            "declare exhaustive axes or add explicit other/residual leaves"
        )
    row_members, row_leaves, row_hierarchy = _axis(rows)
    column_members, col_leaves, col_hierarchy = _axis(columns)
    margins = tuple(margins)
    if len({m.name for m in margins}) != len(margins):
        raise ValueError("margin names must be unique")
    facts = tuple(m.fact for m in margins)
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("each margin must reference a distinct fact ID")
    amount_margins = tuple(m for m in margins if isinstance(m, AllocationMargin))
    share_margins = tuple(m for m in margins if isinstance(m, AllocationShareMargin))
    if not amount_margins or len(amount_margins) + len(share_margins) != len(margins):
        raise ValueError("tables require amount margins and declared margin types")
    contexts = {
        (
            m.fact.entity,
            m.fact.period_start,
            m.fact.period_end,
            m.fact.unit,
            m.fact.currency,
        )
        for m in amount_margins
    }
    if len(contexts) > 1 or any(
        m.fact.unit != unit or (m.measure or m.fact.concept) != measure
        for m in amount_margins
    ):
        raise ValueError(
            "margin facts must share entity, period, measure, unit, and currency"
        )
    evidence_diagnostics = validate_disclosure_evidence(facts, as_of=as_of)
    if evidence_diagnostics:
        raise ValueError(
            "invalid margin evidence: "
            + "; ".join(d.code for d in evidence_diagnostics)
        )
    cells = {
        (r, c): f"cell_{i}_{j}"
        for i, r in enumerate(row_leaves)
        for j, c in enumerate(col_leaves)
    }
    row_qualifiers = {m.name: m.qualifier for m in row_members if m.qualifier}
    col_qualifiers = {m.name: m.qualifier for m in column_members if m.qualifier}
    variables = [
        us.NamedLinearVariable(
            v,
            lower=0,
            unit=unit,
            label=f"{r} / {c}",
            description=f"Row qualifier: {row_qualifiers.get(r)}; column qualifier: {col_qualifiers.get(c)}",
        )
        for (r, c), v in cells.items()
    ]
    constraints = []
    for index, margin in enumerate(amount_margins):
        name = f"margin_{index}"
        variables.append(
            us.NamedLinearVariable(name, lower=0, unit=unit, label=margin.name)
        )
        constraints.append(
            disclosure_fact_constraint(
                f"{margin.name}:fact", name, margin.fact, exact=margin.exact
            )
        )
        selected_rows = _select(margin.rows, row_leaves, row_hierarchy)
        selected_cols = _select(margin.columns, col_leaves, col_hierarchy)
        constraints.append(
            reconciliation_constraint(
                f"{margin.name}:allocation",
                total=name,
                components=[cells[r, c] for r in selected_rows for c in selected_cols],
                provenance=margin.fact.source_url,
                metadata={
                    "evidence_role": "accounting_relationship",
                    "fact_ids": [margin.fact.fact_id],
                    "varying_fields": [],
                    "rows": list(selected_rows),
                    "columns": list(selected_cols),
                    "row_qualifiers": row_qualifiers,
                    "column_qualifiers": col_qualifiers,
                    "declared_measure": measure,
                    "source_concept": margin.fact.concept,
                },
            )
        )
    by_name = {m.name: (i, m) for i, m in enumerate(amount_margins)}
    for margin in share_margins:
        if margin.denominator not in by_name:
            raise ValueError("share denominator must name an amount margin")
        index, denominator = by_name[margin.denominator]
        if (margin.fact.entity, margin.fact.period_start, margin.fact.period_end) != (
            denominator.fact.entity,
            denominator.fact.period_start,
            denominator.fact.period_end,
        ):
            raise ValueError("share and denominator must share entity and period")
        selected_rows = _select(margin.rows, row_leaves, row_hierarchy)
        selected_cols = _select(margin.columns, col_leaves, col_hierarchy)
        if not set(selected_rows) <= set(
            _select(denominator.rows, row_leaves, row_hierarchy)
        ) or not set(selected_cols) <= set(
            _select(denominator.columns, col_leaves, col_hierarchy)
        ):
            raise ValueError(
                "share numerator must be contained in its denominator selection"
            )
        constraints.extend(
            percentage_constraints(
                margin.name,
                numerator={
                    cells[r, c]: 1 for r in selected_rows for c in selected_cols
                },
                denominator=f"margin_{index}",
                fact=margin.fact,
                bounds=margin.bounds,
                exact=margin.exact,
                policy=margin.policy,
                role=margin.role,
                denominator_nonnegative=True,
            )
        )
    zeros = tuple(tuple(x) for x in structural_zeros)
    if len(set(zeros)) != len(zeros):
        raise ValueError("structural zeros must be unique")
    for index, cell in enumerate(zeros):
        if cell not in cells:
            raise ValueError("structural zeros must name leaf cells")
        constraints.append(
            us.NamedLinearConstraint(
                f"exclusion_{index}",
                cells[cell],
                lower=0,
                upper=0,
                kind="structural_exclusion",
                metadata={
                    "declared_assumption": True,
                    "cell": list(cell),
                    "evidence_role": "analyst_policy",
                },
            )
        )
    declared = (
        "Both leaf axes exhaustively partition the declared measure.",
        "Structural zeros are caller-declared exclusions, not inferred facts.",
        *assumptions,
    )
    return AllocationTable(
        row_members,
        column_members,
        MappingProxyType(cells),
        tuple(variables),
        tuple(constraints),
        facts,
        tuple(declared),
        unit,
        as_of,
        title,
        amount_margins,
    )
