"""Target functional contracts for update-support problems."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Hashable


class UnsupportedTargetError(TypeError):
    """Raised when a target functional is outside the supported contract."""


@dataclass(frozen=True)
class TargetCapabilities:
    """Capability flags for target-level operations."""

    supports_adequacy: bool
    supports_interval: bool
    supports_fiber_decomposition: bool

    @classmethod
    def none(cls) -> "TargetCapabilities":
        return cls(
            supports_adequacy=False,
            supports_interval=False,
            supports_fiber_decomposition=False,
        )

    @classmethod
    def linear(cls) -> "TargetCapabilities":
        return cls(
            supports_adequacy=True,
            supports_interval=True,
            supports_fiber_decomposition=True,
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "supports_adequacy": self.supports_adequacy,
            "supports_interval": self.supports_interval,
            "supports_fiber_decomposition": self.supports_fiber_decomposition,
        }


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

    @property
    def capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(
            supports_adequacy=self.supports_adequacy,
            supports_interval=self.supports_interval,
            supports_fiber_decomposition=self.supports_fiber_decomposition,
        )

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
            "capabilities": self.capabilities.as_dict(),
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
class ProcedureTargetContext:
    """Context supplied when compiling a representation-dependent target.

    A procedure target is compiled by tabular workflows before a finite
    ``FiniteProblem`` is constructed. The compiler should return a column name
    or row metric for the supplied public representation.
    """

    data: Any
    public: tuple[str, ...]
    hidden: tuple[str, ...]
    weight: str | None = None
    min_cell_weight: float = 1.0
    q: Any = "saturated"
    q_radius: float | None = None

    @property
    def row_count(self) -> int | None:
        try:
            return len(self.data)
        except TypeError:
            return None

    def as_dict(self) -> dict[str, Any]:
        q_value = self.q if isinstance(self.q, str) else repr(self.q)
        return {
            "data_type": type(self.data).__name__,
            "row_count": self.row_count,
            "public": self.public,
            "hidden": self.hidden,
            "weight": self.weight,
            "min_cell_weight": self.min_cell_weight,
            "q": q_value,
            "q_radius": self.q_radius,
        }


@dataclass(frozen=True)
class ProcedureTarget:
    """Representation-dependent target compiled by tabular workflows.

    ``compiler`` receives a :class:`ProcedureTargetContext` and must return the
    actual column name or row metric to use for that representation. The compiled
    target is then treated as a fixed hidden-cell target by the finite solver.
    """

    name: str
    compiler: Callable[[ProcedureTargetContext], Any]
    description: str = ""
    formula: str = (
        "public representation -> compiled target values -> reported aggregate"
    )
    limitations: tuple[str, ...] = ()

    @property
    def contract(self) -> TargetContract:
        limitations = (
            "Procedure targets are not finite-problem estimands until compiled.",
            "Procedure-aware workflows re-run the compiler for each public "
            "representation or sensitivity scenario.",
            "Procedure comparisons should not be interpreted as transporting one "
            "unchanged target functional across representations.",
            *self.limitations,
        )
        return TargetContract(
            kind="procedure",
            name=self.name,
            formula=self.formula,
            description=self.description
            or "representation-dependent reporting procedure",
            fixed_after_compilation=False,
            supports_adequacy=False,
            supports_interval=False,
            supports_fiber_decomposition=False,
            limitations=limitations,
        )

    @property
    def compiler_name(self) -> str:
        return getattr(self.compiler, "__name__", type(self.compiler).__name__)

    def compile(self, context: ProcedureTargetContext) -> Any:
        """Return the column name or row metric for ``context``."""

        return self.compiler(context)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "procedure",
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "compiler": self.compiler_name,
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
class MomentTransformTarget:
    """Fixed transform of one or more linear moments.

    The mathematical form is ``psi(q) = g(mu(q))`` where each moment is
    ``mu_j(q) = sum_d m_j(d) q(d)``. The current finite solvers support this
    target only when ``g`` is declared affine through ``affine_coefficients`` and
    ``intercept``. Non-affine moment transforms remain explicit target-contract
    objects with false capability flags.
    """

    moments: Mapping[str, Mapping[Hashable, float]]
    transform: Callable[[Mapping[str, float]], float]
    name: str = "moment_transform"
    description: str = "fixed transform of linear moments"
    affine_coefficients: Mapping[str, float] | None = None
    intercept: float = 0.0
    capabilities: TargetCapabilities | None = None
    formula: str | None = None
    source: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not callable(self.transform):
            raise TypeError("moment transform must be callable")
        moments = {}
        for name, values in self.moments.items():
            moment_name = str(name)
            if moment_name in moments:
                raise ValueError("moment names must be unique after string coercion")
            moments[moment_name] = _finite_float_mapping(
                values,
                name=f"moment {name!r}",
            )
        object.__setattr__(self, "moments", moments)

        intercept = float(self.intercept)
        if not isfinite(intercept):
            raise ValueError("moment transform intercept must be finite")
        object.__setattr__(self, "intercept", intercept)

        if self.affine_coefficients is not None:
            coefficients = _finite_float_mapping(
                self.affine_coefficients,
                name="moment affine coefficients",
            )
            unknown = sorted(set(coefficients) - set(moments))
            if unknown:
                raise ValueError(
                    f"affine coefficients reference unknown moments: {unknown!r}"
                )
            object.__setattr__(self, "affine_coefficients", coefficients)

        if self.capabilities is not None and not isinstance(
            self.capabilities, TargetCapabilities
        ):
            raise TypeError("capabilities must be a TargetCapabilities object")
        if (
            self.affine_coefficients is None
            and self.capabilities is not None
            and (
                self.capabilities.supports_adequacy
                or self.capabilities.supports_interval
                or self.capabilities.supports_fiber_decomposition
            )
        ):
            raise ValueError(
                "non-affine MomentTransformTarget capability flags cannot be true"
            )

    @property
    def is_affine(self) -> bool:
        return self.affine_coefficients is not None

    @property
    def resolved_capabilities(self) -> TargetCapabilities:
        if self.capabilities is not None:
            return self.capabilities
        if self.is_affine:
            return TargetCapabilities.linear()
        return TargetCapabilities.none()

    @property
    def supports_linear_backend(self) -> bool:
        capabilities = self.resolved_capabilities
        return (
            self.is_affine
            and capabilities.supports_adequacy
            and capabilities.supports_interval
        )

    @property
    def contract(self) -> TargetContract:
        capabilities = self.resolved_capabilities
        limitations = list(self.limitations)
        if self.is_affine:
            limitations.append(
                "Affine moment transforms are solved through their equivalent "
                "fixed linear hidden-cell target values."
            )
        else:
            limitations.append(
                "Non-affine moment transforms are not solved by the current "
                "finite transport backends."
            )
            limitations.append(
                "Use an affine reformulation, RatioTarget, ProcedureTarget, or a "
                "dedicated nonlinear target backend."
            )
        return TargetContract(
            kind="moment_transform",
            name=self.name,
            formula=self.formula or self._default_formula(),
            description=self.description,
            fixed_after_compilation=True,
            supports_adequacy=capabilities.supports_adequacy,
            supports_interval=capabilities.supports_interval,
            supports_fiber_decomposition=capabilities.supports_fiber_decomposition,
            limitations=tuple(limitations),
        )

    @property
    def values(self) -> dict[Hashable, float]:
        self._require_supported_affine()
        states = self._moment_states()
        return {state: self.point_value(state) for state in states}

    def moment_value(self, moment: str, state: Hashable) -> float:
        return self.moments[moment][state]

    def moment_vector(
        self,
        moment: str,
        states: Sequence[Hashable],
    ) -> tuple[float, ...]:
        return tuple(self.moment_value(moment, state) for state in states)

    def moment_dot(
        self,
        moment: str,
        states: Sequence[Hashable],
        vector: Sequence[float],
    ) -> float:
        return sum(
            self.moment_value(moment, state) * vector[i]
            for i, state in enumerate(states)
        )

    def moment_values(
        self,
        states: Sequence[Hashable],
        vector: Sequence[float],
    ) -> dict[str, float]:
        return {
            moment: self.moment_dot(moment, states, vector) for moment in self.moments
        }

    def dot(self, states: Sequence[Hashable], vector: Sequence[float]) -> float:
        self._require_supported_affine()
        coefficients = self._affine_coefficients()
        total = sum(vector)
        return self.intercept * total + sum(
            coefficients.get(moment, 0.0) * self.moment_dot(moment, states, vector)
            for moment in self.moments
        )

    def point_value(self, state: Hashable) -> float:
        if self.is_affine:
            coefficients = self._affine_coefficients()
            value = self.intercept + sum(
                coefficients.get(moment, 0.0) * self.moment_value(moment, state)
                for moment in self.moments
            )
        else:
            value = self.transform(
                {moment: self.moment_value(moment, state) for moment in self.moments}
            )
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"moment transform {self.name!r} evaluated non-finite")
        return value

    def value(self, state: Hashable) -> float:
        return self.point_value(state)

    def support_key(self, state: Hashable) -> float | tuple[float, ...]:
        if self.is_affine:
            return self.point_value(state)
        return tuple(self.moment_value(moment, state) for moment in self.moments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "moment_transform",
            "name": self.name,
            "description": self.description,
            "formula": self.contract.formula,
            "moment_names": tuple(self.moments),
            "moment_count": len(self.moments),
            "affine": self.is_affine,
            "affine_coefficients": None
            if self.affine_coefficients is None
            else dict(self.affine_coefficients),
            "intercept": self.intercept,
            "source": self.source,
            "contract": self.contract.as_dict(),
        }

    def _default_formula(self) -> str:
        if self.is_affine:
            terms = [
                f"{coefficient:g} E_q[{moment}]"
                for moment, coefficient in self._affine_coefficients().items()
                if abs(coefficient) > 0.0
            ]
            if self.intercept:
                terms.insert(0, f"{self.intercept:g}")
            return "psi(q) = " + (" + ".join(terms) if terms else "0")
        names = ", ".join(f"E_q[{moment}]" for moment in self.moments)
        return f"psi(q) = g({names})"

    def _affine_coefficients(self) -> Mapping[str, float]:
        if self.affine_coefficients is None:
            return {}
        return self.affine_coefficients

    def _moment_states(self) -> tuple[Hashable, ...]:
        state_order: list[Hashable] = []
        seen: set[Hashable] = set()
        for values in self.moments.values():
            for state in values:
                if state not in seen:
                    seen.add(state)
                    state_order.append(state)
        return tuple(state_order)

    def _require_supported_affine(self) -> None:
        if self.supports_linear_backend:
            return
        raise UnsupportedTargetError(
            "MomentTransformTarget is supported by the current finite solvers "
            "only when it declares affine_coefficients and capability flags for "
            "adequacy and interval solving."
        )


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
            "callable fixed linear target, RatioTarget, or supported "
            "MomentTransformTarget. Unsupported nonlinear or "
            "representation-dependent targets require an explicit "
            "target-functional workflow."
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
) -> LinearTarget | MomentTransformTarget | RatioTarget:
    """Coerce a supported target functional."""

    if isinstance(value, ProcedureTarget):
        raise UnsupportedTargetError(
            "ProcedureTarget must be compiled by `from_dataframe(...)` or another "
            "procedure-aware workflow before constructing `FiniteProblem`. "
            "Procedure compilers must return a column name or RowMetric that can "
            "be converted to a fixed target functional."
        )
    if isinstance(value, MomentTransformTarget):
        _validate_moment_target_states(states, value)
        if not value.supports_linear_backend:
            raise UnsupportedTargetError(
                "MomentTransformTarget is supported by `FiniteProblem` only for "
                "affine transforms of linear moments. Provide "
                "`affine_coefficients` or use a dedicated nonlinear target "
                "backend. Capability flags: "
                f"{value.resolved_capabilities.as_dict()!r}."
            )
        return value
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


def _validate_moment_target_states(
    states: Sequence[Hashable],
    target: MomentTransformTarget,
) -> None:
    for moment, values in target.moments.items():
        missing = [state for state in states if state not in values]
        if missing:
            raise ValueError(f"moment {moment!r} is missing states: {missing!r}")


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
    if contract.kind == "procedure":
        raise UnsupportedTargetError(
            f"{context} received procedure target {contract.name!r}. "
            "ProcedureTarget objects must be compiled by `from_dataframe(...)` "
            "or another procedure-aware workflow before a finite transport "
            "problem is constructed."
        )
    if contract.kind == "moment_transform":
        raise UnsupportedTargetError(
            f"{context} received moment-transform target {contract.name!r}. "
            "MomentTransformTarget is a finite target functional; pass affine "
            "moment transforms to `FiniteProblem`, not as row-level dataframe "
            "targets. Non-affine moment transforms require a dedicated nonlinear "
            "target backend."
        )
    limitations = "; ".join(contract.limitations)
    detail = f" {limitations}" if limitations else ""
    raise UnsupportedTargetError(
        f"{context} received unsupported target {contract.name!r} "
        f"(kind={contract.kind!r}, formula={contract.formula!r}). "
        "The current core supports only fixed linear plug-in targets directly, "
        "plus supported RatioTarget objects and affine MomentTransformTarget "
        "objects. "
        "Unsupported nonlinear, distributional, or representation-dependent "
        "targets must be explicitly reformulated or handled by a dedicated "
        f"target-functional backend.{detail}"
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
