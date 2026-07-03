"""Target functional contracts for update-support problems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
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

    def point_value(self, state: Hashable) -> float:
        return self.value(state)

    def support_key(self, state: Hashable) -> float:
        return self.value(state)

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


@dataclass(frozen=True)
class RatioTarget:
    """Fixed ratio target ``psi(q) = <n,q> / <w,q>``.

    The denominator must be strictly positive on every retained state. General
    constrained-Q ratio support is intentionally narrow; saturated public-fiber
    environments solve ratio intervals exactly, standard CVXPY can solve local
    fixed-public-law ratio intervals through DQCP, and linear backends can use a
    ratio target when the denominator is constant over the state space.
    """

    numerator: Mapping[Hashable, float]
    denominator: Mapping[Hashable, float] | float
    name: str = "ratio_target"
    description: str = "fixed linear-fractional ratio target"
    source: str | None = None

    def __post_init__(self) -> None:
        numerator = _finite_float_mapping(self.numerator, name="ratio numerator")
        denominator: Mapping[Hashable, float] | float
        if isinstance(self.denominator, Mapping):
            denominator = _finite_float_mapping(
                self.denominator,
                name="ratio denominator",
            )
            nonpositive = [
                state for state, value in denominator.items() if value <= 0.0
            ]
            if nonpositive:
                raise ValueError(f"ratio denominator must be positive: {nonpositive!r}")
        elif isinstance(self.denominator, Real):
            denominator = float(self.denominator)
            if not isfinite(denominator):
                raise ValueError("ratio denominator must be finite")
            if denominator <= 0.0:
                raise ValueError("ratio denominator must be positive")
        else:
            raise TypeError("ratio denominator must be a mapping or positive scalar")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    @property
    def contract(self) -> TargetContract:
        limitations = (
            "Denominator values are fixed after compilation and must be strictly positive.",
            "Exact ratio interval solving currently supports finite/enumerated "
            "Q, public-fiber-saturated Q, and CVXPY local/fixed-public-law "
            "DQCP solves. Other constrained-Q ratio optimization is future work.",
            "Ratio targets do not have an additive public-fiber contribution "
            "decomposition.",
        )
        return TargetContract(
            kind="ratio",
            name=self.name,
            formula="psi(q) = sum_d n(d) q(d) / sum_d w(d) q(d)",
            description=self.description,
            fixed_after_compilation=True,
            supports_adequacy=True,
            supports_interval=True,
            supports_fiber_decomposition=False,
            limitations=limitations,
        )

    @property
    def values(self) -> dict[Hashable, float]:
        return {
            state: self.numerator_value(state) / self.denominator_value(state)
            for state in self.numerator
        }

    def numerator_value(self, state: Hashable) -> float:
        return self.numerator[state]

    def denominator_value(self, state: Hashable) -> float:
        if isinstance(self.denominator, Mapping):
            return self.denominator[state]
        return float(self.denominator)

    def point_value(self, state: Hashable) -> float:
        return self.numerator_value(state) / self.denominator_value(state)

    def value(self, state: Hashable) -> float:
        return self.point_value(state)

    def support_key(self, state: Hashable) -> float | tuple[float, float]:
        if isinstance(self.denominator, Mapping):
            return (self.numerator_value(state), self.denominator_value(state))
        return self.point_value(state)

    def numerator_dot(
        self,
        states: Sequence[Hashable],
        vector: Sequence[float],
    ) -> float:
        return sum(
            self.numerator_value(state) * vector[i] for i, state in enumerate(states)
        )

    def denominator_dot(
        self,
        states: Sequence[Hashable],
        vector: Sequence[float],
    ) -> float:
        return sum(
            self.denominator_value(state) * vector[i] for i, state in enumerate(states)
        )

    def dot(self, states: Sequence[Hashable], vector: Sequence[float]) -> float:
        if self.has_constant_denominator(states):
            return sum(
                self.point_value(state) * vector[i] for i, state in enumerate(states)
            )
        denominator = self.denominator_dot(states, vector)
        if denominator <= 0.0:
            raise ValueError("ratio denominator must be positive")
        return self.numerator_dot(states, vector) / denominator

    def has_constant_denominator(self, states: Sequence[Hashable]) -> bool:
        if not states:
            return True
        first = self.denominator_value(states[0])
        return all(
            abs(self.denominator_value(state) - first) <= 1e-12 for state in states[1:]
        )

    def numerator_vector(self, states: Sequence[Hashable]) -> tuple[float, ...]:
        return tuple(self.numerator_value(state) for state in states)

    def denominator_vector(self, states: Sequence[Hashable]) -> tuple[float, ...]:
        return tuple(self.denominator_value(state) for state in states)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "ratio",
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "state_count": len(self.numerator),
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


def coerce_target(
    states: Sequence[Hashable],
    value: Any,
    *,
    name: str = "linear_target",
    description: str = "fixed linear plug-in target",
    source: str | None = None,
) -> LinearTarget | RatioTarget:
    """Coerce a supported target functional."""

    if isinstance(value, RatioTarget):
        _validate_ratio_target_states(states, value)
        return value
    return coerce_linear_target(
        states,
        value,
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


def _validate_ratio_target_states(
    states: Sequence[Hashable],
    target: RatioTarget,
) -> None:
    missing_numerator = [state for state in states if state not in target.numerator]
    if missing_numerator:
        raise ValueError(f"ratio numerator is missing states: {missing_numerator!r}")
    if isinstance(target.denominator, Mapping):
        missing_denominator = [
            state for state in states if state not in target.denominator
        ]
        if missing_denominator:
            raise ValueError(
                f"ratio denominator is missing states: {missing_denominator!r}"
            )


def _finite_float_mapping(
    value: Mapping[Hashable, float],
    *,
    name: str,
) -> dict[Hashable, float]:
    coerced = {state: float(item) for state, item in value.items()}
    nonfinite = [state for state, item in coerced.items() if not isfinite(item)]
    if nonfinite:
        raise ValueError(f"{name} has non-finite values: {nonfinite!r}")
    return coerced


def raise_if_unsupported_target(value: Any, *, context: str) -> None:
    """Raise when ``value`` declares a non-linear or unsupported target contract."""

    contract = declared_target_contract(value)
    if contract is None:
        return
    if isinstance(value, LinearTarget | RatioTarget):
        return
    if contract.kind == "linear" and contract.supports_interval:
        raise UnsupportedTargetError(
            f"{context} declares a linear target contract but is not a "
            "`LinearTarget`. The current backend accepts `LinearTarget`, "
            "mapping, sequence, or callable fixed linear targets."
        )
    if contract.kind == "ratio" and contract.supports_interval:
        raise UnsupportedTargetError(
            f"{context} declares a ratio target contract but is not a "
            "`RatioTarget`. The current ratio backend accepts `RatioTarget` "
            "objects with fixed numerator and positive denominator maps."
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
