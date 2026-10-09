"""Reviewed customer-advance mappings over the shared disclosure solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import updatesupport as us

from .briefs import DisclosureScope
from .scenarios import DisclosureScenarioAnalysis, break_even_analysis


_KINDS = {
    "net_cash",
    "cash_receipt",
    "cash_refund",
    "recognition",
    "unpaid_opening",
    "unpaid_closing",
    "unpaid_change",
    "noncash",
}
_SIGNED = {"net_cash", "unpaid_change", "noncash"}
_REDUCTIONS = {"cash_refund", "recognition", "unpaid_opening"}


@dataclass(frozen=True)
class AdvanceMovement:
    """One reviewed primitive, with its amount constrained separately.

    Gross receipts, refunds, recognition and unpaid endpoint balances are
    nonnegative magnitudes. Net cash, unpaid changes and noncash adjustments
    are signed. ``effect=-1`` maps a source-positive noncash reduction; it
    does not change that primitive's signed domain. Use a separate declared
    constraint if the source establishes a nonnegative magnitude.
    """

    name: str
    kind: str
    description: str
    effect: int = 1
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.description or self.kind not in _KINDS:
            raise ValueError("movement needs a name, description and known kind")
        if (
            isinstance(self.effect, bool)
            or self.effect not in {-1, 1}
            or (self.kind != "noncash" and self.effect != 1)
        ):
            raise ValueError("only noncash movements accept effect=-1")

    @property
    def coefficient(self) -> float:
        return -1.0 if self.kind in _REDUCTIONS else float(self.effect)

    def as_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class CustomerAdvanceBridge:
    """Compile a declared liability roll-forward without inferring receipts.

    ``complete=True`` asserts that the reviewed mapping covers every movement
    class. It does not assert that every amount is known: a signed, unrestricted
    noncash primitive is appropriate for unresolved financing, acquisition,
    currency and reclassification effects. A mapping that may omit movements
    must use ``complete=False``; its signed residual remains *unclassified*.

    All inputs must already share the scope's numerical unit, currency, entity,
    cohort and duration. Opening/closing liabilities and unpaid balances are
    endpoint stocks for that duration. The caller reviews accounting meanings
    and compiles source amounts/precision with the existing evidence APIs.
    """

    name: str
    scope: DisclosureScope
    opening: str
    closing: str
    movements: Sequence[AdvanceMovement]
    accounting_regime: str
    recognition_basis: str
    complete: bool
    completeness_basis: str
    cash_flow_adjustment: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.name,
                self.opening,
                self.closing,
                self.accounting_regime,
                self.recognition_basis,
                self.completeness_basis,
            )
        ):
            raise ValueError("bridge needs names and explicit accounting/mapping bases")
        if type(self.complete) is not bool:
            raise ValueError("complete must be an explicit boolean")
        if self.cash_flow_adjustment == "":
            raise ValueError("cash-flow adjustment name cannot be empty")
        if not isinstance(self.scope, DisclosureScope) or not self.scope.currency:
            raise ValueError("bridge needs a DisclosureScope with a currency")
        if not self.scope.period_start or self.scope.window_kind in {
            "instant",
            "stock",
            "lifetime",
            "schedule",
        }:
            raise ValueError("bridge needs a duration scope, not a stock or commitment")
        movements = tuple(self.movements)
        if not all(isinstance(m, AdvanceMovement) for m in movements):
            raise ValueError("movements must be AdvanceMovement objects")
        object.__setattr__(self, "movements", movements)
        kinds = {m.kind for m in movements}
        if not kinds & {"net_cash", "cash_receipt", "cash_refund"}:
            raise ValueError("declare net cash or gross cash movements explicitly")
        if "net_cash" in kinds and kinds & {"cash_receipt", "cash_refund"}:
            raise ValueError("net cash cannot be mixed with gross receipts/refunds")
        if "unpaid_change" in kinds and kinds & {"unpaid_opening", "unpaid_closing"}:
            raise ValueError("unpaid change cannot be mixed with unpaid endpoints")
        if ("unpaid_opening" in kinds) != ("unpaid_closing" in kinds):
            raise ValueError(
                "declare both unpaid endpoints, including an explicit zero"
            )
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise ValueError(
                "bridge primitive and generated variable names must be unique"
            )

    def _name(self, suffix: str) -> str:
        return f"{self.name}:{suffix}"

    @property
    def net_cash(self) -> str:
        return self._name("net_cash")

    @property
    def noncash(self) -> str:
        return self._name("noncash")

    @property
    def recognition(self) -> str:
        return self._name("recognition")

    @property
    def unpaid_change(self) -> str:
        return self._name("unpaid_change")

    @property
    def unclassified(self) -> str | None:
        return None if self.complete else self._name("unclassified")

    @property
    def cash_flow_gap(self) -> str | None:
        return self._name("cash_flow_gap") if self.cash_flow_adjustment else None

    @property
    def _groups(self) -> Mapping[str, set[str]]:
        return {
            self.net_cash: {"net_cash", "cash_receipt", "cash_refund"},
            self.noncash: {"noncash"},
            self.recognition: {"recognition"},
            self.unpaid_change: {"unpaid_opening", "unpaid_closing", "unpaid_change"},
        }

    @property
    def variables(self) -> tuple[us.NamedLinearVariable, ...]:
        rows = [
            us.NamedLinearVariable(n, lower=0, unit=self.scope.unit)
            for n in (self.opening, self.closing)
        ]
        rows.extend(
            us.NamedLinearVariable(
                m.name,
                lower=None if m.kind in _SIGNED else 0,
                unit=self.scope.unit,
                description=m.description,
            )
            for m in self.movements
        )
        rows.extend(
            us.NamedLinearVariable(n, unit=self.scope.unit) for n in self._groups
        )
        rows.extend(
            us.NamedLinearVariable(n, unit=self.scope.unit)
            for n in (self.unclassified, self.cash_flow_adjustment, self.cash_flow_gap)
            if n is not None
        )
        return tuple(rows)

    @property
    def constraints(self) -> tuple[us.NamedLinearConstraint, ...]:
        metadata = {
            "evidence_role": "accounting_relationship",
            "customer_advance_bridge": self.as_dict(),
        }

        def identity(suffix, coefficients, description):
            return us.NamedLinearConstraint(
                self._name(suffix),
                coefficients,
                lower=0,
                upper=0,
                kind="reconciliation",
                description=description,
                provenance=self.completeness_basis,
                metadata=metadata,
            )

        rows = []
        for aggregate, kinds in self._groups.items():
            rows.append(
                identity(
                    f"sum:{aggregate.rsplit(':', 1)[-1]}",
                    {
                        aggregate: 1,
                        **{
                            m.name: -(1 if m.kind == "recognition" else m.coefficient)
                            for m in self.movements
                            if m.kind in kinds
                        },
                    },
                    "Sum of supplied movement primitives; omitted classes remain "
                    "in the unclassified residual when the mapping is incomplete.",
                )
            )
        coefficients = {
            self.closing: 1,
            self.opening: -1,
            self.net_cash: -1,
            self.unpaid_change: -1,
            self.noncash: -1,
            self.recognition: 1,
        }
        if self.unclassified:
            coefficients[self.unclassified] = -1
        rows.append(
            identity(
                "roll_forward",
                coefficients,
                "Closing = opening + net cash + unpaid change + net noncash "
                "- recognition + any unclassified residual.",
            )
        )
        if self.cash_flow_adjustment:
            rows.append(
                identity(
                    "cash_flow_diagnostic",
                    {
                        self.closing: 1,
                        self.opening: -1,
                        self.cash_flow_adjustment: -1,
                        self.cash_flow_gap: -1,
                    },
                    "Liability change minus the indirect cash-flow adjustment; "
                    "diagnostic only, with no cause or receipts attribution.",
                )
            )
        return tuple(rows)

    @property
    def targets(self) -> tuple[us.NamedLinearTarget, ...]:
        labels = {
            self.net_cash: (
                "Net advance cash (receipts less refunds)"
                if self.complete
                else "Supplied net advance cash; omitted cash may be in the residual"
            ),
            self.noncash: "Net noncash liability additions, excluding unpaid bills",
            self.recognition: f"Recognition: {self.recognition_basis}",
            self.unpaid_change: "Change in unpaid advance billings",
        }
        if self.unclassified:
            labels[self.unclassified] = "Unclassified movements; may include cash"
        if self.cash_flow_gap:
            labels[self.cash_flow_gap] = "Liability / indirect cash-flow diagnostic gap"
        return tuple(
            us.NamedLinearTarget(n, n, label=label, unit=self.scope.unit)
            for n, label in labels.items()
        )

    def problem(
        self,
        *,
        constraints: Sequence[us.NamedLinearConstraint] = (),
        variables: Sequence[us.NamedLinearVariable] = (),
        targets: Sequence[us.NamedLinearTarget] = (),
        tiers: Sequence[us.NamedLinearScenario] | None = None,
        title: str = "Customer advance reconciliation",
    ) -> us.NamedLinearFeasibilityProblem:
        """Combine source constraints/policies, retaining identities in every tier.

        Primitive bounds belong in explicit evidence or policy constraints;
        generated variables cannot be replaced. Additional variables must use
        the same numerical unit. Normalize evidence before compiling bounds.
        """
        if any(v.unit != self.scope.unit for v in variables):
            raise ValueError(
                "additional variables must use the bridge's numerical unit"
            )
        structural = self.constraints
        all_constraints = (*structural, *constraints)
        if tiers is None:
            tiers = (
                us.NamedLinearScenario("reported", [c.name for c in all_constraints]),
            )
        else:
            tiers = tuple(
                us.NamedLinearScenario(
                    s.name,
                    tuple(
                        dict.fromkeys((*s.constraints, *(c.name for c in structural)))
                    ),
                    description=s.description,
                )
                for s in tiers
            )
        return us.NamedLinearFeasibilityProblem(
            (*self.variables, *variables),
            all_constraints,
            (*self.targets, *targets),
            tiers,
            title=title,
            description=self.completeness_basis,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope.as_dict(),
            "opening": self.opening,
            "closing": self.closing,
            "movements": [m.as_dict() for m in self.movements],
            "accounting_regime": self.accounting_regime,
            "recognition_basis": self.recognition_basis,
            "complete": self.complete,
            "completeness_basis": self.completeness_basis,
            "cash_flow_adjustment": self.cash_flow_adjustment,
            "cash_basis": "net receipts less refunds, including collections of old unpaid bills",
        }

    def snapshot_context(self) -> dict[str, Any]:
        return {
            "scope": self.scope.as_dict(),
            "customer_advance_bridge": self.as_dict(),
        }


def minimum_advance_explanation(
    bridge: CustomerAdvanceBridge,
    problem: us.NamedLinearFeasibilityProblem,
    *,
    cash_ceiling: float | us.NamedLinearExpression | Mapping[str, float] | str,
    ceiling_description: str,
    tier: str = "reported",
    release_constraints: Sequence[str] = (),
    name: str = "advance_explanation",
) -> DisclosureScenarioAnalysis:
    """Bound net noncash additions required by a hypothetical net cash ceiling.

    Supply an amount or linear expression, e.g. ``{"cfo": .8}``, in the
    bridge's numerical unit. This solves a necessary alternative accounting
    explanation, not the probability of that explanation or recurring CFO.
    Negative minima are retained. Unbounded/infeasible results and endpoint
    witnesses use the existing scenario report. An incomplete mapping cannot
    attribute its unknown residual to noncash and is rejected here.
    """
    if not bridge.complete:
        raise ValueError("noncash attribution requires a declared complete mapping")
    if not ceiling_description:
        raise ValueError("describe the hypothetical cash ceiling explicitly")
    base = next((s for s in problem.scenarios if s.name == tier), None)
    expected = {c.name: c for c in bridge.constraints}
    actual = {c.name: c for c in problem.constraints}
    if (
        base is None
        or not expected.keys() <= set(base.constraints)
        or any(actual.get(n) != c for n, c in expected.items())
        or any(v not in problem.variables for v in bridge.variables)
        or any(t not in problem.targets for t in bridge.targets)
    ):
        raise ValueError(
            "problem/tier must retain the bridge variables, targets and identities"
        )
    if isinstance(cash_ceiling, (int, float)):
        if not isfinite(cash_ceiling):
            raise ValueError("cash ceiling must be finite")
        coefficients, constant = {bridge.net_cash: 1}, -float(cash_ceiling)
    else:
        ceiling = us.NamedLinearTarget("ceiling", cash_ceiling).expression
        units = {v.name: v.unit for v in problem.variables}
        if any(units.get(n) != bridge.scope.unit for n in ceiling.coefficients):
            raise ValueError(
                "ceiling variables must exist and use the bridge's numerical unit"
            )
        coefficients = {n: -c for n, c in ceiling.coefficients.items()}
        coefficients[bridge.net_cash] = coefficients.get(bridge.net_cash, 0) + 1
        constant = -ceiling.constant
    if not any(coefficients.values()):
        raise ValueError("cash ceiling must impose a nontrivial condition")
    condition = us.NamedLinearConstraint(
        f"{name}:cash_ceiling",
        us.NamedLinearExpression(coefficients, constant=constant),
        upper=0,
        description=ceiling_description,
        metadata={"scope": bridge.scope.as_dict(), "hypothetical": True},
    )
    return break_even_analysis(
        problem,
        target=bridge.noncash,
        tier=tier,
        conditions=(condition,),
        release_constraints=release_constraints,
        name=name,
    )
