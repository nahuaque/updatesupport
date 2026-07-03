"""Target functional contracts for update-support problems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Hashable


class UnsupportedTargetError(TypeError):
    """Raised when a target functional is outside the supported contract."""


@dataclass(frozen=True)
class TargetContract:
    """Human- and machine-readable contract for a target functional."""

    kind: str
    name: str
    formula: str
    description: str
    fixed_after_compilation: bool
    supports_adequacy: bool
    supports_interval: bool
    supports_fiber_decomposition: bool
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "fixed_after_compilation": self.fixed_after_compilation,
            "supports_adequacy": self.supports_adequacy,
            "supports_interval": self.supports_interval,
            "supports_fiber_decomposition": self.supports_fiber_decomposition,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class UnsupportedTarget:
    """Explicit marker for target functionals not yet supported by core solvers."""

    name: str
    kind: str
    formula: str
    description: str = ""
    reason: str = (
        "This target functional is not supported by the current fixed linear "
        "target backend."
    )
    limitations: tuple[str, ...] = ()

    @property
    def contract(self) -> TargetContract:
        return TargetContract(
            kind=self.kind,
            name=self.name,
            formula=self.formula,
            description=self.description or self.reason,
            fixed_after_compilation=False,
            supports_adequacy=False,
            supports_interval=False,
            supports_fiber_decomposition=False,
            limitations=(self.reason, *self.limitations),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "reason": self.reason,
            "contract": self.contract.as_dict(),
        }


@dataclass(frozen=True)
class LinearTarget:
    """Fixed linear plug-in target ``psi(q) = sum_d h(d) q(d)``."""

    values: Mapping[Hashable, float]
    name: str = "linear_target"
    description: str = "fixed linear plug-in target"
    source: str | None = None

    def __post_init__(self) -> None:
        values = {state: float(value) for state, value in self.values.items()}
        nonfinite = [state for state, value in values.items() if not isfinite(value)]
        if nonfinite:
            raise ValueError(f"linear target has non-finite values: {nonfinite!r}")
        object.__setattr__(self, "values", values)

    @property
    def contract(self) -> TargetContract:
        limitations = (
            "Hidden-cell target values are fixed after compilation.",
            "Nonlinear, ratio, or representation-dependent targets must be "
            "explicitly reformulated before using this contract.",
        )
        return TargetContract(
            kind="linear",
            name=self.name,
            formula="psi(q) = sum_d h(d) q(d)",
            description=self.description,
            fixed_after_compilation=True,
            supports_adequacy=True,
            supports_interval=True,
            supports_fiber_decomposition=True,
            limitations=limitations,
        )

    def value(self, state: Hashable) -> float:
        return self.values[state]

    def dot(self, states: Sequence[Hashable], vector: Sequence[float]) -> float:
        return sum(self.values[state] * vector[i] for i, state in enumerate(states))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "state_count": len(self.values),
            "contract": self.contract.as_dict(),
        }


def coerce_linear_target(
    states: Sequence[Hashable],
    value: Any,
    *,
    name: str = "linear_target",
    description: str = "fixed linear plug-in target",
    source: str | None = None,
) -> LinearTarget:
    """Coerce mappings, sequences, callables, or LinearTarget objects."""

    if isinstance(value, LinearTarget):
        _validate_target_states(states, value.values)
        return value

    raise_if_unsupported_target(value, context="FiniteProblem.estimand")

    if callable(value):
        return LinearTarget(
            {state: float(value(state)) for state in states},
            name=name,
            description=description,
            source=source,
        )

    if isinstance(value, Mapping):
        missing = [state for state in states if state not in value]
        if missing:
            raise ValueError(f"estimand is missing states: {missing!r}")
        return LinearTarget(
            {state: float(value[state]) for state in states},
            name=name,
            description=description,
            source=source,
        )

    try:
        value_len = len(value)
    except TypeError as exc:
        raise UnsupportedTargetError(
            "FiniteProblem.estimand must be a LinearTarget, mapping, sequence, "
            "or callable fixed linear target. Nonlinear or "
            "representation-dependent targets require an explicit future "
            "target-functional backend."
        ) from exc

    if value_len != len(states):
        raise ValueError("estimand sequence must have one value per state")
    return LinearTarget(
        {state: float(value[i]) for i, state in enumerate(states)},
        name=name,
        description=description,
        source=source,
    )


def _validate_target_states(
    states: Sequence[Hashable],
    values: Mapping[Hashable, float],
) -> None:
    missing = [state for state in states if state not in values]
    if missing:
        raise ValueError(f"linear target is missing states: {missing!r}")


def raise_if_unsupported_target(value: Any, *, context: str) -> None:
    """Raise when ``value`` declares a non-linear or unsupported target contract."""

    contract = declared_target_contract(value)
    if contract is None:
        return
    if isinstance(value, LinearTarget):
        return
    if contract.kind == "linear" and contract.supports_interval:
        raise UnsupportedTargetError(
            f"{context} declares a linear target contract but is not a "
            "`LinearTarget`. The current backend accepts `LinearTarget`, "
            "mapping, sequence, or callable fixed linear targets."
        )
    limitations = "; ".join(contract.limitations)
    detail = f" {limitations}" if limitations else ""
    raise UnsupportedTargetError(
        f"{context} received unsupported target {contract.name!r} "
        f"(kind={contract.kind!r}, formula={contract.formula!r}). "
        "The current core supports only fixed linear plug-in targets of the "
        "form `psi(q) = sum_d h(d) q(d)`. Nonlinear, ratio, distributional, "
        "or representation-dependent targets must be explicitly reformulated "
        f"or handled by a future target-functional backend.{detail}"
    )


def declared_target_contract(value: Any) -> TargetContract | None:
    """Return a declared target contract from marker/future target objects."""

    if isinstance(value, TargetContract):
        return value
    for attribute in ("contract", "target_contract"):
        contract = getattr(value, attribute, None)
        if isinstance(contract, TargetContract):
            return contract
    return None
