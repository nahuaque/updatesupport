"""Admissible environment classes for finite update-support problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence

from .artifacts import ReportArtifactMixin
from .partition import Partition
from .results import (
    AdequacyResult,
    ConstraintDual,
    TransportResult,
    UncertainLinearConfidenceCoreResult,
    Witness,
)
from .targets import (
    MomentTransformTarget,
    UncertainLinearTarget,
    UnsupportedTargetError,
)


class LPError(RuntimeError):
    """Raised when a linear program cannot be solved successfully."""


class CvxpyError(RuntimeError):
    """Raised when a CVXPY optimization problem cannot be solved successfully."""


class Environment(Protocol):
    name: str

    def check_support(self, problem, support: Partition) -> AdequacyResult: ...

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult: ...

    def global_transport(self, problem) -> TransportResult: ...


def _dict_from_public_vector(problem, vector: Sequence[float]) -> dict[Hashable, float]:
    return {
        public_value: vector[i] for i, public_value in enumerate(problem.public_values)
    }


def _is_nonlinear_ratio_problem(problem) -> bool:
    return problem.has_ratio_target and not problem.has_linear_target


def _same_key(left: Hashable, right: Hashable, *, tol: float) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(
            abs(float(left_item) - float(right_item)) <= tol
            for left_item, right_item in zip(left, right, strict=True)
        )
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return left == right


CvxpyConstraintBuilder = Callable[
    [Any, Any, tuple[Hashable, ...], Mapping[Hashable, int]], Sequence[Any]
]
CvxpyParameterFactory = Callable[..., Any]
CvxpyParameterizedConstraintBuilder = Callable[
    [
        Any,
        Any,
        tuple[Hashable, ...],
        Mapping[Hashable, int],
        CvxpyParameterFactory,
    ],
    Sequence[Any],
]


@dataclass(frozen=True)
class CvxpyConstraintMetadata:
    """CVXPY constraint plus metadata used for dual diagnostics."""

    constraint: Any
    name: str
    kind: str = "custom"
    sense: str | None = None
    variable: str | None = None
    state: Hashable | None = None
    public_value: Hashable | None = None
    states: tuple[Hashable, ...] = ()
    public_values: tuple[Hashable, ...] = ()


def cvxpy_constraint(
    constraint: Any,
    *,
    name: str,
    kind: str = "custom",
    sense: str | None = None,
    variable: str | None = None,
    state: Hashable | None = None,
    public_value: Hashable | None = None,
    states: Sequence[Hashable] = (),
    public_values: Sequence[Hashable] = (),
) -> CvxpyConstraintMetadata:
    """Attach diagnostic metadata to a custom CVXPY constraint."""

    return CvxpyConstraintMetadata(
        constraint=constraint,
        name=name,
        kind=kind,
        sense=sense,
        variable=variable,
        state=state,
        public_value=public_value,
        states=tuple(states),
        public_values=tuple(public_values),
    )


@dataclass(frozen=True)
class _CvxpySolveResult:
    vector: tuple[float, ...]
    value: float
    duals: tuple[ConstraintDual, ...] = ()


@dataclass(frozen=True)
class SupportFunctionResult:
    """Value and optimizer for one support-function evaluation."""

    direction: tuple[float, ...]
    value: float
    vector: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "value": self.value,
            "vector": self.vector,
        }


@dataclass(frozen=True)
class SupportFunctionIntervalResult:
    """Lower/upper interval from support-function evaluations."""

    direction: tuple[float, ...]
    lower: float
    upper: float
    diameter: float
    lower_vector: tuple[float, ...]
    upper_vector: tuple[float, ...]
    lower_support_value: float
    upper_support_value: float
    lower_support_result: SupportFunctionResult
    upper_support_result: SupportFunctionResult
    lower_duals: tuple[ConstraintDual, ...] = ()
    upper_duals: tuple[ConstraintDual, ...] = ()

    @property
    def duals(self) -> tuple[ConstraintDual, ...]:
        return self.lower_duals + self.upper_duals

    def dual_summary(
        self, *, top: int | None = 10, min_magnitude: float = 0.0
    ) -> tuple[ConstraintDual, ...]:
        rows = [row for row in self.duals if row.magnitude >= min_magnitude]
        rows.sort(key=lambda row: row.magnitude, reverse=True)
        return tuple(rows if top is None else rows[:top])

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "lower": self.lower,
            "upper": self.upper,
            "diameter": self.diameter,
            "lower_vector": self.lower_vector,
            "upper_vector": self.upper_vector,
            "lower_support_value": self.lower_support_value,
            "upper_support_value": self.upper_support_value,
            "lower_support_result": self.lower_support_result.as_dict(),
            "upper_support_result": self.upper_support_result.as_dict(),
            "lower_duals": [row.as_dict() for row in self.lower_duals],
            "upper_duals": [row.as_dict() for row in self.upper_duals],
            "duals": [row.as_dict() for row in self.duals],
        }


@dataclass(frozen=True)
class SupportFunctionTargetInterval:
    """Support-function interval for one named linear target direction."""

    name: str
    direction: tuple[float, ...]
    lower: float
    upper: float
    diameter: float
    lower_distribution: Mapping[Hashable, float]
    upper_distribution: Mapping[Hashable, float]
    lower_support_value: float
    upper_support_value: float
    lower_duals: tuple[ConstraintDual, ...] = ()
    upper_duals: tuple[ConstraintDual, ...] = ()

    @property
    def duals(self) -> tuple[ConstraintDual, ...]:
        return self.lower_duals + self.upper_duals

    def dual_summary(
        self, *, top: int | None = 10, min_magnitude: float = 0.0
    ) -> tuple[ConstraintDual, ...]:
        rows = [row for row in self.duals if row.magnitude >= min_magnitude]
        rows.sort(key=lambda row: row.magnitude, reverse=True)
        return tuple(rows if top is None else rows[:top])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "lower": self.lower,
            "upper": self.upper,
            "diameter": self.diameter,
            "lower_distribution": dict(self.lower_distribution),
            "upper_distribution": dict(self.upper_distribution),
            "lower_support_value": self.lower_support_value,
            "upper_support_value": self.upper_support_value,
            "lower_duals": [row.as_dict() for row in self.lower_duals],
            "upper_duals": [row.as_dict() for row in self.upper_duals],
            "duals": [row.as_dict() for row in self.duals],
        }


@dataclass(frozen=True)
class SupportFunctionReport(ReportArtifactMixin):
    """Multi-target support-function interval and dual diagnostic report."""

    title: str
    targets: tuple[SupportFunctionTargetInterval, ...]
    states: tuple[Hashable, ...]
    public_values: tuple[Hashable, ...]
    public_law: Mapping[Hashable, float] | None = None
    backend: str = "support-function-cvxpy"

    @property
    def target_count(self) -> int:
        return len(self.targets)

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def public_cell_count(self) -> int:
        return len(self.public_values)

    @property
    def max_diameter(self) -> float:
        if not self.targets:
            return 0.0
        return max(target.diameter for target in self.targets)

    def dual_summary(
        self, *, top: int | None = 10, min_magnitude: float = 0.0
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for target in self.targets:
            for endpoint, duals in (
                ("lower", target.lower_duals),
                ("upper", target.upper_duals),
            ):
                for dual in duals:
                    if dual.magnitude < min_magnitude:
                        continue
                    payload = dual.as_dict()
                    payload["target"] = target.name
                    payload["endpoint"] = endpoint
                    rows.append(payload)
        rows.sort(key=lambda row: float(row["magnitude"]), reverse=True)
        return tuple(rows if top is None else rows[:top])

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "backend": self.backend,
            "target_count": self.target_count,
            "state_count": self.state_count,
            "public_cell_count": self.public_cell_count,
            "public_law": dict(self.public_law)
            if self.public_law is not None
            else None,
            "max_diameter": self.max_diameter,
            "targets": [target.as_dict() for target in self.targets],
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        target_rows = tuple(
            {
                "target": target.name,
                "lower": target.lower,
                "upper": target.upper,
                "diameter": target.diameter,
                "lower_support_value": target.lower_support_value,
                "upper_support_value": target.upper_support_value,
                "lower_dual_count": len(target.lower_duals),
                "upper_dual_count": len(target.upper_duals),
            }
            for target in self.targets
        )
        dual_rows = self.dual_summary(top=None)
        return {
            "summary": (
                {
                    "title": self.title,
                    "backend": self.backend,
                    "target_count": self.target_count,
                    "state_count": self.state_count,
                    "public_cell_count": self.public_cell_count,
                    "max_diameter": self.max_diameter,
                },
            ),
            "targets": target_rows,
            "dual_diagnostics": dual_rows,
        }

    def to_markdown(
        self, *, top_duals: int = 10, min_dual_magnitude: float = 0.0
    ) -> str:
        lines = [
            f"# {_markdown_escape(self.title)}",
            "",
            (
                f"Backend: `{self.backend}`. Evaluated {self.target_count} "
                f"linear target direction(s) over {self.state_count} hidden state(s)."
            ),
            "",
            "| Target | Lower | Upper | Ambiguity width |",
            "| --- | ---: | ---: | ---: |",
        ]
        for target in self.targets:
            lines.append(
                "| "
                f"{_markdown_escape(target.name)} | "
                f"{_format_float(target.lower)} | "
                f"{_format_float(target.upper)} | "
                f"{_format_float(target.diameter)} |"
            )
        rows = self.dual_summary(top=top_duals, min_magnitude=min_dual_magnitude)
        if rows:
            lines.extend(
                [
                    "",
                    "## Dual Diagnostics",
                    "",
                    "| Target | Endpoint | Constraint | Kind | Magnitude | Residual |",
                    "| --- | --- | --- | --- | ---: | ---: |",
                ]
            )
            for row in rows:
                lines.append(
                    "| "
                    f"{_markdown_escape(str(row['target']))} | "
                    f"{_markdown_escape(str(row['endpoint']))} | "
                    f"{_markdown_escape(str(row['name']))} | "
                    f"{_markdown_escape(str(row['kind']))} | "
                    f"{_format_float(float(row['magnitude']))} | "
                    f"{_format_optional_float(row.get('residual'))} |"
                )
        return "\n".join(lines)


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return _format_float(float(value))


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


@dataclass(frozen=True)
class ConvexAdmissibleSet:
    """A CVXPY-defined convex admissible set over hidden distributions."""

    variable: Any
    constraints: Sequence[Any]
    records: Sequence[CvxpyConstraintMetadata] = ()
    name: str = "convex admissible set"

    def support_function(self) -> Any:
        """Return CVXPY's support-function transform for this set."""

        try:
            from cvxpy.transforms.suppfunc import SuppFunc
        except ImportError as exc:  # pragma: no cover - CVXPY API availability
            raise CvxpyError(
                "CVXPY support-function transforms are unavailable. "
                "Install a CVXPY version that provides cvxpy.transforms.suppfunc."
            ) from exc
        return SuppFunc(self.variable, list(self.constraints))

    def support_expression(self, direction: Sequence[float]) -> Any:
        """Return the CVXPY expression ``sigma_Q(direction)``."""

        return self.support_function()(tuple(float(value) for value in direction))

    def support_value(
        self,
        direction: Sequence[float],
        *,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> SupportFunctionResult:
        """Evaluate ``sup_{q in Q} <direction, q>`` with CVXPY."""

        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - exercised without optional dep
            raise CvxpyError(
                "CVXPY support is optional; install it with "
                "`uv sync --extra cvxpy` or `pip install updatesupport[cvxpy]`."
            ) from exc

        direction_vector = tuple(float(value) for value in direction)
        expression = self.support_expression(direction_vector)
        problem = cp.Problem(cp.Minimize(-expression), [])
        options = dict(solver_options or {})
        try:
            if solver is None:
                problem.solve(**options)
            else:
                problem.solve(solver=solver, **options)
        except Exception as exc:  # pragma: no cover - solver exceptions vary
            raise CvxpyError(str(exc)) from exc
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise CvxpyError(f"CVXPY support-function status: {problem.status}")
        if self.variable.value is None:
            raise CvxpyError("CVXPY support function did not return an optimizer")
        if expression.value is None:
            raise CvxpyError("CVXPY support function did not return a value")
        return SupportFunctionResult(
            direction=direction_vector,
            value=float(expression.value),
            vector=tuple(float(value) for value in self.variable.value),
        )

    def support_interval(
        self,
        direction: Sequence[float],
        *,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
        tol: float = 1e-9,
    ) -> SupportFunctionIntervalResult:
        """Evaluate ``[-sigma_Q(-h), sigma_Q(h)]`` for ``direction`` h."""

        direction_vector = tuple(float(value) for value in direction)
        lower_direction = tuple(-value for value in direction_vector)
        lower_support = self.support_value(
            lower_direction,
            solver=solver,
            solver_options=solver_options,
        )
        upper_support = self.support_value(
            direction_vector,
            solver=solver,
            solver_options=solver_options,
        )
        lower = -lower_support.value
        upper = upper_support.value
        if lower > upper and abs(lower - upper) <= tol:
            lower = upper = 0.5 * (lower + upper)
        return SupportFunctionIntervalResult(
            direction=direction_vector,
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            lower_vector=lower_support.vector,
            upper_vector=upper_support.vector,
            lower_support_value=lower_support.value,
            upper_support_value=upper_support.value,
            lower_support_result=lower_support,
            upper_support_result=upper_support,
        )


@dataclass(frozen=True)
class _CvxpyPairSolveResult:
    q1: tuple[float, ...]
    q2: tuple[float, ...]
    value: float
    duals: tuple[ConstraintDual, ...] = ()


@dataclass(frozen=True)
class LinearConstraint:
    """A linear constraint over state probabilities."""

    coefficients: Mapping[Hashable, float] | Sequence[float]
    sense: str
    rhs: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError("constraint sense must be one of '<=', '>=', '=='")


def linear_constraint(
    coefficients: Mapping[Hashable, float] | Sequence[float],
    sense: str,
    rhs: float,
    *,
    name: str | None = None,
) -> LinearConstraint:
    return LinearConstraint(
        coefficients=coefficients, sense=sense, rhs=float(rhs), name=name
    )


def leq(
    coefficients: Mapping[Hashable, float] | Sequence[float],
    rhs: float,
    *,
    name: str | None = None,
) -> LinearConstraint:
    return linear_constraint(coefficients, "<=", rhs, name=name)


def geq(
    coefficients: Mapping[Hashable, float] | Sequence[float],
    rhs: float,
    *,
    name: str | None = None,
) -> LinearConstraint:
    return linear_constraint(coefficients, ">=", rhs, name=name)


def eq(
    coefficients: Mapping[Hashable, float] | Sequence[float],
    rhs: float,
    *,
    name: str | None = None,
) -> LinearConstraint:
    return linear_constraint(coefficients, "==", rhs, name=name)


@dataclass(frozen=True)
class PublicFiberSaturated:
    """All conditional reweightings inside public fibers are admissible.

    ``public_marginals=None`` means the full simplex over public values. A mapping
    fixes a single public law. A sequence of mappings is interpreted as vertices
    of a finite public-marginal polytope for global closed-form maximization.
    """

    public_marginals: (
        Mapping[Hashable, float] | Sequence[Mapping[Hashable, float]] | None
    ) = None
    name: str = "public-fiber-saturated"

    @classmethod
    def fixed(cls, public_law: Mapping[Hashable, float]) -> "PublicFiberSaturated":
        return cls(public_marginals=public_law)

    @classmethod
    def vertices(
        cls, public_laws: Sequence[Mapping[Hashable, float]]
    ) -> "PublicFiberSaturated":
        return cls(public_marginals=tuple(public_laws))

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        problem._validate_support_over_public(support)
        if _is_nonlinear_ratio_problem(problem):
            if support != problem.public_partition():
                problem._require_linear_target(
                    "PublicFiberSaturated.check_support for non-public supports"
                )
            transport = self.global_transport(problem)
            if transport.diameter <= problem.tol:
                return AdequacyResult(adequate=True, support=support)
            if transport.q_lower is None or transport.q_upper is None:
                raise RuntimeError("ratio support check did not produce witnesses")
            witness = Witness(
                q1=transport.q_lower,
                q2=transport.q_upper,
                psi_q1=transport.lower,
                psi_q2=transport.upper,
                gap=transport.diameter,
                public_law=transport.public_law
                if transport.public_law is not None
                else problem.public_law(transport.q_lower),
                support_law=problem.support_law(transport.q_lower, support),
            )
            return AdequacyResult(
                adequate=False,
                support=support,
                gap=transport.diameter,
                witness=witness,
                reason=(
                    "public fiber contains hidden cells that can change the "
                    "ratio target under the selected public law"
                ),
            )
        ranges = self._support_block_ranges(problem, support)
        public_law, worst_public, worst_pair, worst_gap = self._worst_support_gap(
            problem,
            ranges,
        )

        if worst_gap <= problem.tol or worst_public is None or worst_pair is None:
            return AdequacyResult(adequate=True, support=support)

        q1 = {state: 0.0 for state in problem.states}
        q2 = {state: 0.0 for state in problem.states}
        for public_value, mass in public_law.items():
            if public_value == worst_public:
                q1[worst_pair[0]] += mass
                q2[worst_pair[1]] += mass
            else:
                state = problem.public_fibers[public_value][0]
                q1[state] += mass
                q2[state] += mass
        witness = Witness(
            q1=q1,
            q2=q2,
            psi_q1=problem.psi(q1),
            psi_q2=problem.psi(q2),
            gap=worst_gap,
            public_law=problem.public_law(q1),
            support_law=problem.support_law(q1, support),
        )
        return AdequacyResult(
            adequate=False,
            support=support,
            gap=worst_gap,
            witness=witness,
            reason="support merges states with different estimand values inside a public fiber",
        )

    def least_support(self, problem) -> Partition:
        """Return the saturated least support for the configured public laws."""

        relevant_public_values = self._relevant_public_values(problem)
        blocks: list[list[Hashable]] = []
        keys: list[tuple[Hashable, float | None]] = []
        for state in problem.states:
            public_value = problem.public_map[state]
            h_value = (
                problem._target_support_key(state)
                if (public_value in relevant_public_values)
                else None
            )
            for i, (existing_public, existing_h) in enumerate(keys):
                if existing_public != public_value:
                    continue
                if h_value is None and existing_h is None:
                    blocks[i].append(state)
                    break
                if (
                    h_value is not None
                    and existing_h is not None
                    and _same_key(
                        existing_h,
                        h_value,
                        tol=problem.tol,
                    )
                ):
                    blocks[i].append(state)
                    break
            else:
                keys.append((public_value, h_value))
                blocks.append([state])
        return Partition.from_blocks(blocks, universe=problem.states)

    def _support_block_ranges(
        self, problem, support: Partition
    ) -> dict[Hashable, tuple[float, tuple[Hashable, Hashable] | None]]:
        ranges: dict[Hashable, tuple[float, tuple[Hashable, Hashable] | None]] = {
            public_value: (0.0, None) for public_value in problem.public_values
        }
        for block in support.blocks:
            public_value = problem.public_map[block[0]]
            min_state = min(block, key=lambda state: problem.estimand_map[state])
            max_state = max(block, key=lambda state: problem.estimand_map[state])
            gap = problem.estimand_map[max_state] - problem.estimand_map[min_state]
            current_gap, _ = ranges[public_value]
            if gap > current_gap:
                ranges[public_value] = (gap, (min_state, max_state))
        return ranges

    def _worst_support_gap(
        self,
        problem,
        ranges: Mapping[Hashable, tuple[float, tuple[Hashable, Hashable] | None]],
    ) -> tuple[
        dict[Hashable, float],
        Hashable | None,
        tuple[Hashable, Hashable] | None,
        float,
    ]:
        best_public_law: dict[Hashable, float] | None = None
        best_public_value: Hashable | None = None
        best_pair: tuple[Hashable, Hashable] | None = None
        best_gap = 0.0

        for public_law in self._candidate_public_laws(problem):
            for public_value, mass in public_law.items():
                gap, pair = ranges[public_value]
                weighted_gap = mass * gap
                if pair is not None and weighted_gap > best_gap:
                    best_public_law = public_law
                    best_public_value = public_value
                    best_pair = pair
                    best_gap = weighted_gap

        if best_public_law is None:
            best_public_law = {value: 0.0 for value in problem.public_values}
        return best_public_law, best_public_value, best_pair, best_gap

    def _candidate_public_laws(self, problem) -> tuple[dict[Hashable, float], ...]:
        public_marginals = self.public_marginals
        if public_marginals is None:
            laws = []
            for public_value in problem.public_values:
                law = {value: 0.0 for value in problem.public_values}
                law[public_value] = 1.0
                laws.append(law)
            return tuple(laws)
        if isinstance(public_marginals, Mapping):
            return (problem._coerce_public_law(public_marginals),)
        if not public_marginals:
            raise ValueError("public_marginals vertices must be non-empty")
        return tuple(
            problem._coerce_public_law(public_law) for public_law in public_marginals
        )

    def _relevant_public_values(self, problem) -> set[Hashable]:
        relevant: set[Hashable] = set()
        for public_law in self._candidate_public_laws(problem):
            for public_value, mass in public_law.items():
                if mass > problem.tol:
                    relevant.add(public_value)
        return relevant

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            return self._ratio_local_transport(problem, public_law)

        p = problem._coerce_public_law(public_law)
        ranges = problem.fiber_ranges()
        lower = 0.0
        upper = 0.0
        q_lower = {state: 0.0 for state in problem.states}
        q_upper = {state: 0.0 for state in problem.states}

        for public_value, mass in p.items():
            fiber = problem.public_fibers[public_value]
            min_state = min(fiber, key=lambda state: problem.estimand_map[state])
            max_state = max(fiber, key=lambda state: problem.estimand_map[state])
            lower += mass * problem.estimand_map[min_state]
            upper += mass * problem.estimand_map[max_state]
            q_lower[min_state] += mass
            q_upper[max_state] += mass

        diameter = sum(
            p[public_value] * ranges[public_value]
            for public_value in problem.public_values
        )
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=diameter,
            public_law=p,
            q_lower=q_lower,
            q_upper=q_upper,
        )

    def global_transport(self, problem) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            return self._ratio_global_transport(problem)

        ranges = problem.fiber_ranges()
        public_marginals = self.public_marginals
        if public_marginals is None:
            worst_public = max(problem.public_values, key=lambda value: ranges[value])
            public_law = {value: 0.0 for value in problem.public_values}
            public_law[worst_public] = 1.0
            return self.local_transport(problem, public_law)

        if isinstance(public_marginals, Mapping):
            return self.local_transport(problem, public_marginals)

        candidates = [
            self.local_transport(problem, public_law) for public_law in public_marginals
        ]
        if not candidates:
            raise ValueError("public_marginals vertices must be non-empty")
        return max(candidates, key=lambda result: result.diameter)

    def _ratio_local_transport(
        self,
        problem,
        public_law: Mapping[Hashable, float],
    ) -> TransportResult:
        p = problem._coerce_public_law(public_law)
        q_lower, lower = self._solve_ratio_single(problem, p, maximize=False)
        q_upper, upper = self._solve_ratio_single(problem, p, maximize=True)
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            public_law=p,
            q_lower=problem._distribution_from_vector(q_lower),
            q_upper=problem._distribution_from_vector(q_upper),
        )

    def _ratio_global_transport(self, problem) -> TransportResult:
        public_marginals = self.public_marginals
        if public_marginals is None:
            candidates = []
            for public_value in problem.public_values:
                public_law = {value: 0.0 for value in problem.public_values}
                public_law[public_value] = 1.0
                candidates.append(self._ratio_local_transport(problem, public_law))
            return max(candidates, key=lambda result: result.diameter)

        if isinstance(public_marginals, Mapping):
            return self._ratio_local_transport(problem, public_marginals)

        candidates = [
            self._ratio_local_transport(problem, public_law)
            for public_law in public_marginals
        ]
        if not candidates:
            raise ValueError("public_marginals vertices must be non-empty")
        return max(candidates, key=lambda result: result.diameter)

    def _solve_ratio_single(
        self,
        problem,
        public_law: Mapping[Hashable, float],
        *,
        maximize: bool,
    ) -> tuple[tuple[float, ...], float]:
        import numpy as np
        from scipy.optimize import linprog

        target = problem.target_functional
        numerator = np.array(target.numerator_vector(problem.states), dtype=float)
        denominator = np.array(target.denominator_vector(problem.states), dtype=float)
        n = len(problem.states)
        objective = np.concatenate([numerator, np.array([0.0])])
        c = -objective if maximize else objective

        a_eq = [np.concatenate([denominator, np.array([0.0])])]
        b_eq = [1.0]
        for public_value in problem.public_values:
            indicator = np.array(
                [
                    1.0 if problem.public_map[state] == public_value else 0.0
                    for state in problem.states
                ],
                dtype=float,
            )
            a_eq.append(
                np.concatenate(
                    [indicator, np.array([-float(public_law[public_value])])]
                )
            )
            b_eq.append(0.0)

        result = linprog(
            c,
            A_eq=np.array(a_eq, dtype=float),
            b_eq=np.array(b_eq, dtype=float),
            bounds=[(0.0, None) for _ in range(n + 1)],
            method="highs",
        )
        if not result.success:
            raise LPError(result.message)

        y = np.array(result.x[:n], dtype=float)
        tau = float(result.x[n])
        if tau <= problem.tol:
            raise LPError("ratio transform returned a non-positive normalization")
        q = y / tau
        q = np.where(np.abs(q) <= problem.tol, 0.0, q)
        if np.any(q < -1e-7):
            raise LPError("ratio transform returned a negative probability")
        q = np.maximum(q, 0.0)
        total = float(np.sum(q))
        if abs(total - 1.0) > 1e-7:
            q = q / total
        return tuple(float(value) for value in q), float(np.dot(numerator, y))


@dataclass(frozen=True)
class FiniteEnvironments:
    """A finite enumerated environment class Q."""

    distributions: Sequence[Mapping[Hashable, float] | Sequence[float]]
    name: str = "finite"

    def _vectors(self, problem) -> tuple[tuple[float, ...], ...]:
        if not self.distributions:
            raise ValueError("FiniteEnvironments requires at least one distribution")
        return tuple(
            problem._coerce_distribution(distribution)
            for distribution in self.distributions
        )

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        problem._validate_support_over_public(support)
        vectors = self._vectors(problem)
        best_gap = 0.0
        best_pair: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        for q1 in vectors:
            for q2 in vectors:
                if problem._same_vector(
                    problem._pushforward_vector(q1, support),
                    problem._pushforward_vector(q2, support),
                ):
                    gap = abs(problem._dot_estimand(q1) - problem._dot_estimand(q2))
                    if gap > best_gap:
                        best_gap = gap
                        best_pair = (q1, q2)

        if best_gap <= problem.tol:
            return AdequacyResult(adequate=True, support=support)
        if best_pair is None:
            raise RuntimeError("inadequate support did not produce a witness pair")
        witness = problem._witness_from_vectors(best_pair[0], best_pair[1], support)
        return AdequacyResult(
            adequate=False,
            support=support,
            gap=best_gap,
            witness=witness,
            reason="finite environments share the support law but disagree on the estimand",
        )

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        target = tuple(
            problem._coerce_public_law(public_law)[value]
            for value in problem.public_values
        )
        candidates = [
            q
            for q in self._vectors(problem)
            if problem._same_vector(
                problem._public_vector_from_distribution_vector(q), target
            )
        ]
        if not candidates:
            raise ValueError("no enumerated environment has the requested public law")
        return problem._interval_from_vectors(candidates, public_law=public_law)

    def global_transport(self, problem) -> TransportResult:
        vectors = self._vectors(problem)
        best: TransportResult | None = None
        for q in vectors:
            public_vector = problem._public_vector_from_distribution_vector(q)
            same_public = [
                candidate
                for candidate in vectors
                if problem._same_vector(
                    problem._public_vector_from_distribution_vector(candidate),
                    public_vector,
                )
            ]
            result = problem._interval_from_vectors(
                same_public,
                public_law=_dict_from_public_vector(problem, public_vector),
            )
            if best is None or result.diameter > best.diameter:
                best = result
        if best is None:
            raise RuntimeError("finite environments did not produce a transport result")
        return best


@dataclass(frozen=True)
class LineSegment:
    """Continuous environments q(t) = center + t * direction with bounded t."""

    center: Mapping[Hashable, float] | Sequence[float]
    direction: Mapping[Hashable, float] | Sequence[float]
    radius: float
    name: str = "line-segment"

    def _data(self, problem) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        center = problem._coerce_distribution(self.center)
        direction = problem._coerce_vector(self.direction)
        if abs(sum(direction)) > problem.tol:
            raise ValueError("line-segment direction must sum to zero")
        for sign in (-1.0, 1.0):
            endpoint = tuple(
                center[i] + sign * self.radius * direction[i]
                for i in range(len(center))
            )
            if any(value < -problem.tol for value in endpoint):
                raise ValueError("line-segment endpoints must be non-negative")
        return center, direction

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        problem._require_linear_target("LineSegment.check_support")
        problem._validate_support_over_public(support)
        center, direction = self._data(problem)
        support_direction = problem._pushforward_vector(direction, support)
        estimand_direction = problem._dot_estimand(direction)
        if (
            not problem._is_zero_vector(support_direction)
            or abs(estimand_direction) <= problem.tol
        ):
            return AdequacyResult(adequate=True, support=support)

        q_low, q_high = self._endpoints_for_estimand(problem, center, direction)
        witness = problem._witness_from_vectors(q_low, q_high, support)
        return AdequacyResult(
            adequate=False,
            support=support,
            gap=witness.gap,
            witness=witness,
            reason="line-segment direction is invisible to the support but moves the estimand",
        )

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        problem._require_linear_target("LineSegment.local_transport")
        center, direction = self._data(problem)
        p_target = tuple(
            problem._coerce_public_law(public_law)[value]
            for value in problem.public_values
        )
        p_center = problem._public_vector_from_distribution_vector(center)
        p_direction = problem._public_vector_from_distribution_vector(direction)

        if problem._is_zero_vector(p_direction):
            if not problem._same_vector(p_center, p_target):
                raise ValueError("requested public law is not on this line segment")
            q_low, q_high = self._endpoints_for_estimand(problem, center, direction)
            return problem._interval_from_vectors(
                (q_low, q_high), public_law=public_law
            )

        t_value: float | None = None
        for center_value, direction_value, target_value in zip(
            p_center, p_direction, p_target, strict=True
        ):
            if abs(direction_value) <= problem.tol:
                if not isclose(center_value, target_value, abs_tol=problem.tol):
                    raise ValueError("requested public law is not on this line segment")
                continue
            candidate_t = (target_value - center_value) / direction_value
            if t_value is None:
                t_value = candidate_t
            elif not isclose(t_value, candidate_t, abs_tol=problem.tol):
                raise ValueError("requested public law is not on this line segment")

        if (
            t_value is None
            or t_value < -self.radius - problem.tol
            or t_value > self.radius + problem.tol
        ):
            raise ValueError("requested public law is not on this line segment")
        q = tuple(center[i] + t_value * direction[i] for i in range(len(center)))
        psi = problem._dot_estimand(q)
        return TransportResult(
            lower=psi,
            upper=psi,
            diameter=0.0,
            public_law=problem._coerce_public_law(public_law),
            q_lower=problem._distribution_from_vector(q),
            q_upper=problem._distribution_from_vector(q),
        )

    def global_transport(self, problem) -> TransportResult:
        problem._require_linear_target("LineSegment.global_transport")
        center, direction = self._data(problem)
        p_direction = problem._public_vector_from_distribution_vector(direction)
        if not problem._is_zero_vector(p_direction):
            public_law = _dict_from_public_vector(
                problem, problem._public_vector_from_distribution_vector(center)
            )
            return self.local_transport(problem, public_law)

        q_low, q_high = self._endpoints_for_estimand(problem, center, direction)
        public_law = _dict_from_public_vector(
            problem, problem._public_vector_from_distribution_vector(center)
        )
        return problem._interval_from_vectors((q_low, q_high), public_law=public_law)

    def _endpoints_for_estimand(self, problem, center, direction):
        estimand_direction = problem._dot_estimand(direction)
        if estimand_direction >= 0:
            q_low_sign, q_high_sign = -1.0, 1.0
        else:
            q_low_sign, q_high_sign = 1.0, -1.0
        q_low = tuple(
            center[i] + q_low_sign * self.radius * direction[i]
            for i in range(len(center))
        )
        q_high = tuple(
            center[i] + q_high_sign * self.radius * direction[i]
            for i in range(len(center))
        )
        return q_low, q_high


@dataclass(frozen=True)
class PolytopeEnvironments:
    """A finite-linear polytope ``Q`` solved with ``scipy.optimize.linprog``.

    The probability simplex is implicit: every environment has nonnegative
    coordinates and total mass one. Additional constraints are linear
    equalities or inequalities over the state probabilities.
    """

    constraints: Sequence[
        LinearConstraint | tuple[Mapping[Hashable, float] | Sequence[float], str, float]
    ] = ()
    bounds: (
        Mapping[Hashable, tuple[float | None, float | None]]
        | Sequence[tuple[float | None, float | None]]
        | None
    ) = None
    method: str = "highs"
    name: str = "polytope"

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        problem._require_linear_target("PolytopeEnvironments.check_support")
        problem._validate_support_over_public(support)
        q1, q2, gap = self._max_support_disagreement(problem, support)
        if gap <= problem.tol:
            return AdequacyResult(adequate=True, support=support)

        witness = problem._witness_from_vectors(q1, q2, support)
        return AdequacyResult(
            adequate=False,
            support=support,
            gap=witness.gap,
            witness=witness,
            reason="polytope contains environments with equal support law and different estimand values",
        )

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        problem._require_linear_target("PolytopeEnvironments.local_transport")
        p = problem._coerce_public_law(public_law)
        public_equalities = self._public_equalities(problem, p)
        h = self._estimand_vector(problem)

        q_lower, lower = self._solve_single(
            problem, h, maximize=False, equalities=public_equalities
        )
        q_upper, upper = self._solve_single(
            problem, h, maximize=True, equalities=public_equalities
        )
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            public_law=p,
            q_lower=problem._distribution_from_vector(q_lower),
            q_upper=problem._distribution_from_vector(q_upper),
        )

    def global_transport(self, problem) -> TransportResult:
        problem._require_linear_target("PolytopeEnvironments.global_transport")
        q1, q2, gap = self._max_support_disagreement(
            problem, problem.public_partition()
        )
        psi_q1 = problem._dot_estimand(q1)
        psi_q2 = problem._dot_estimand(q2)
        if psi_q1 <= psi_q2:
            q_lower, q_upper = q1, q2
            lower, upper = psi_q1, psi_q2
        else:
            q_lower, q_upper = q2, q1
            lower, upper = psi_q2, psi_q1
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=gap,
            public_law=problem.public_law(q_lower),
            q_lower=problem._distribution_from_vector(q_lower),
            q_upper=problem._distribution_from_vector(q_upper),
        )

    def _normalized_constraints(self) -> tuple[LinearConstraint, ...]:
        normalized = []
        for constraint in self.constraints:
            if isinstance(constraint, LinearConstraint):
                normalized.append(constraint)
                continue
            if len(constraint) != 3:
                raise ValueError("constraint tuples must be (coefficients, sense, rhs)")
            coefficients, sense, rhs = constraint
            normalized.append(LinearConstraint(coefficients, sense, float(rhs)))
        return tuple(normalized)

    def _bounds(self, problem) -> list[tuple[float | None, float | None]]:
        if self.bounds is None:
            return [(0.0, None) for _ in problem.states]
        if isinstance(self.bounds, Mapping):
            return [self.bounds.get(state, (0.0, None)) for state in problem.states]
        if len(self.bounds) != len(problem.states):
            raise ValueError("bounds sequence must have one entry per state")
        return list(self.bounds)

    def _single_constraint_matrices(self, problem):
        import numpy as np

        a_ub = []
        b_ub = []
        a_eq = [np.ones(len(problem.states), dtype=float)]
        b_eq = [1.0]

        for constraint in self._normalized_constraints():
            vector = np.array(
                problem._coerce_vector(constraint.coefficients), dtype=float
            )
            if constraint.sense == "<=":
                a_ub.append(vector)
                b_ub.append(float(constraint.rhs))
            elif constraint.sense == ">=":
                a_ub.append(-vector)
                b_ub.append(float(-constraint.rhs))
            elif constraint.sense == "==":
                a_eq.append(vector)
                b_eq.append(float(constraint.rhs))
            else:
                raise ValueError(f"unsupported constraint sense: {constraint.sense!r}")

        return (
            np.array(a_ub, dtype=float) if a_ub else None,
            np.array(b_ub, dtype=float) if b_ub else None,
            np.array(a_eq, dtype=float),
            np.array(b_eq, dtype=float),
        )

    def _solve_single(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        equalities: Sequence[tuple[Sequence[float], float]] = (),
    ) -> tuple[tuple[float, ...], float]:
        import numpy as np
        from scipy.optimize import linprog

        a_ub, b_ub, a_eq, b_eq = self._single_constraint_matrices(problem)
        objective_vector = np.array(objective, dtype=float)
        c = -objective_vector if maximize else objective_vector

        if equalities:
            extra_a_eq = np.array([vector for vector, _ in equalities], dtype=float)
            extra_b_eq = np.array([rhs for _, rhs in equalities], dtype=float)
            a_eq = np.vstack([a_eq, extra_a_eq])
            b_eq = np.concatenate([b_eq, extra_b_eq])

        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=self._bounds(problem),
            method=self.method,
        )
        if not result.success:
            raise LPError(result.message)

        x = tuple(float(value) for value in result.x)
        return x, float(np.dot(objective_vector, result.x))

    def _solve_pair(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        import numpy as np
        from scipy.optimize import linprog

        n = len(problem.states)
        a_ub, b_ub, a_eq, b_eq = self._single_constraint_matrices(problem)
        objective_vector = np.array(objective, dtype=float)
        c = -np.concatenate([objective_vector, -objective_vector])

        pair_a_ub = None
        pair_b_ub = None
        if a_ub is not None:
            zeros = np.zeros_like(a_ub)
            pair_a_ub = np.vstack(
                [
                    np.hstack([a_ub, zeros]),
                    np.hstack([zeros, a_ub]),
                ]
            )
            pair_b_ub = np.concatenate([b_ub, b_ub])

        zeros_eq = np.zeros_like(a_eq)
        pair_a_eq = np.vstack(
            [
                np.hstack([a_eq, zeros_eq]),
                np.hstack([zeros_eq, a_eq]),
            ]
        )
        pair_b_eq = np.concatenate([b_eq, b_eq])

        support_equalities = []
        for block in support.blocks:
            indicator = np.zeros(n, dtype=float)
            for state in block:
                indicator[problem.states.index(state)] = 1.0
            support_equalities.append(np.concatenate([indicator, -indicator]))
        if support_equalities:
            pair_a_eq = np.vstack(
                [pair_a_eq, np.array(support_equalities, dtype=float)]
            )
            pair_b_eq = np.concatenate(
                [pair_b_eq, np.zeros(len(support_equalities), dtype=float)]
            )

        bounds = self._bounds(problem) + self._bounds(problem)
        result = linprog(
            c,
            A_ub=pair_a_ub,
            b_ub=pair_b_ub,
            A_eq=pair_a_eq,
            b_eq=pair_b_eq,
            bounds=bounds,
            method=self.method,
        )
        if not result.success:
            raise LPError(result.message)

        q1 = tuple(float(value) for value in result.x[:n])
        q2 = tuple(float(value) for value in result.x[n:])
        value = float(np.dot(objective_vector, result.x[:n] - result.x[n:]))
        return q1, q2, value

    def _max_support_disagreement(
        self, problem, support: Partition
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        h = self._estimand_vector(problem)
        q_pos_1, q_pos_2, pos_value = self._solve_pair(problem, h, support=support)
        q_neg_1, q_neg_2, neg_value = self._solve_pair(
            problem, [-value for value in h], support=support
        )

        pos_gap = abs(pos_value)
        neg_gap = abs(neg_value)
        if pos_gap >= neg_gap:
            return q_pos_1, q_pos_2, pos_gap
        return q_neg_1, q_neg_2, neg_gap

    def _estimand_vector(self, problem) -> tuple[float, ...]:
        return tuple(problem.estimand_map[state] for state in problem.states)

    def _public_equalities(
        self,
        problem,
        public_law: Mapping[Hashable, float],
    ) -> tuple[tuple[tuple[float, ...], float], ...]:
        equalities = []
        for public_value in problem.public_values:
            indicator = tuple(
                1.0 if problem.public_map[state] == public_value else 0.0
                for state in problem.states
            )
            equalities.append((indicator, float(public_law[public_value])))
        return tuple(equalities)


@dataclass(frozen=True)
class CvxpyEnvironments:
    """A convex finite environment class ``Q`` solved with CVXPY.

    The probability simplex is implicit. Linear constraints use the same helper
    objects accepted by ``PolytopeEnvironments``. Extra convex restrictions can
    be supplied as builders receiving ``(cp, q, states, state_index)`` and
    returning CVXPY constraints for the probability vector ``q``.
    """

    constraints: Sequence[
        LinearConstraint | tuple[Mapping[Hashable, float] | Sequence[float], str, float]
    ] = ()
    constraint_builders: Sequence[CvxpyConstraintBuilder] = ()
    fixed_public_law: Mapping[Hashable, float] | None = None
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None
    name: str = "cvxpy"

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        problem._validate_support_over_public(support)
        if _is_nonlinear_ratio_problem(problem):
            if support != problem.public_partition() or self.fixed_public_law is None:
                problem._require_linear_target("CvxpyEnvironments.check_support")
            transport = self.global_transport(problem)
            if transport.diameter <= problem.tol:
                return AdequacyResult(adequate=True, support=support)
            if transport.q_lower is None or transport.q_upper is None:
                raise RuntimeError("ratio support check did not produce witnesses")
            witness = Witness(
                q1=transport.q_lower,
                q2=transport.q_upper,
                psi_q1=transport.lower,
                psi_q2=transport.upper,
                gap=transport.diameter,
                public_law=transport.public_law
                if transport.public_law is not None
                else problem.public_law(transport.q_lower),
                support_law=problem.support_law(transport.q_lower, support),
            )
            return AdequacyResult(
                adequate=False,
                support=support,
                gap=transport.diameter,
                witness=witness,
                reason=(
                    "convex environment contains same-public-law distributions "
                    "with different ratio target values"
                ),
            )
        problem._require_linear_target("CvxpyEnvironments.check_support")
        q1, q2, gap = self._max_support_disagreement(problem, support)
        if gap <= problem.tol:
            return AdequacyResult(adequate=True, support=support)

        witness = problem._witness_from_vectors(q1, q2, support)
        return AdequacyResult(
            adequate=False,
            support=support,
            gap=witness.gap,
            witness=witness,
            reason="convex environment contains equal-support laws with different estimand values",
        )

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            return self._ratio_local_transport(problem, public_law)
        problem._require_linear_target("CvxpyEnvironments.local_transport")
        p = problem._coerce_public_law(public_law)
        h = self._estimand_vector(problem)

        lower_result = self._solve_single_result(
            problem, h, maximize=False, public_law=p
        )
        upper_result = self._solve_single_result(
            problem, h, maximize=True, public_law=p
        )
        return TransportResult(
            lower=lower_result.value,
            upper=upper_result.value,
            diameter=max(0.0, upper_result.value - lower_result.value),
            public_law=p,
            q_lower=problem._distribution_from_vector(lower_result.vector),
            q_upper=problem._distribution_from_vector(upper_result.vector),
            duals=lower_result.duals + upper_result.duals,
        )

    def global_transport(self, problem) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            if self.fixed_public_law is None:
                problem._require_linear_target("CvxpyEnvironments.global_transport")
            return self.local_transport(problem, self.fixed_public_law)
        problem._require_linear_target("CvxpyEnvironments.global_transport")
        if self.fixed_public_law is not None:
            return self.local_transport(problem, self.fixed_public_law)

        result = self._max_support_disagreement_result(
            problem,
            problem.public_partition(),
        )
        psi_q1 = problem._dot_estimand(result.q1)
        psi_q2 = problem._dot_estimand(result.q2)
        if psi_q1 <= psi_q2:
            q_lower, q_upper = result.q1, result.q2
            lower, upper = psi_q1, psi_q2
        else:
            q_lower, q_upper = result.q2, result.q1
            lower, upper = psi_q2, psi_q1
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, abs(result.value)),
            public_law=problem.public_law(q_lower),
            q_lower=problem._distribution_from_vector(q_lower),
            q_upper=problem._distribution_from_vector(q_upper),
            duals=result.duals,
        )

    def convex_admissible_set(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> ConvexAdmissibleSet:
        """Build the CVXPY admissible distribution set for this problem."""

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        records = self._labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
        )
        return ConvexAdmissibleSet(
            variable=q,
            constraints=tuple(record.constraint for record in records),
            records=tuple(records),
            name=self.name,
        )

    def moment_transform_endpoint(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
        maximize: bool,
    ) -> TransportResult:
        """Solve one convex-compatible MomentTransformTarget endpoint."""

        target = problem.target_functional
        if not isinstance(target, MomentTransformTarget):
            raise TypeError("problem target is not a MomentTransformTarget")
        if maximize and not target.supports_exact_upper_endpoint:
            raise UnsupportedTargetError(
                "MomentTransformTarget maximization is exact only for concave "
                "CVXPY-compatible transforms."
            )
        if not maximize and not target.supports_exact_lower_endpoint:
            raise UnsupportedTargetError(
                "MomentTransformTarget minimization is exact only for convex "
                "CVXPY-compatible transforms."
            )

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        state_index = {state: i for i, state in enumerate(problem.states)}
        moment_exprs = {}
        for moment in target.moments:
            vector = target.moment_vector(moment, problem.states)
            moment_exprs[moment] = cp.sum(cp.multiply(vector, q))
        objective_expr = target.cvxpy_expression(cp, moment_exprs)
        records = self._labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(objective_expr) if maximize else cp.Minimize(objective_expr),
            [record.constraint for record in records],
        )
        if not cvx_problem.is_dcp():
            direction = "maximization" if maximize else "minimization"
            raise CvxpyError(
                f"MomentTransformTarget {direction} is not DCP. Convex targets "
                "support minimization and concave targets support maximization."
            )
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        moment_values = {
            moment: sum(
                target.moment_value(moment, state) * vector[state_index[state]]
                for state in problem.states
            )
            for moment in target.moments
        }
        value = target.transform_value(moment_values)
        q_distribution = problem._distribution_from_vector(vector)
        solve = "upper" if maximize else "lower"
        return TransportResult(
            lower=value,
            upper=value,
            diameter=0.0,
            public_law=problem._coerce_public_law(public_law),
            q_lower=q_distribution,
            q_upper=q_distribution,
            duals=self._constraint_duals(records, solve=solve),
        )

    def uncertain_linear_confidence_core(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
    ) -> UncertainLinearConfidenceCoreResult:
        """Solve the SOCP common confidence core for an uncertain linear target."""

        target = problem.target_functional
        if not isinstance(target, UncertainLinearTarget):
            raise TypeError("problem target is not an UncertainLinearTarget")
        p = problem._coerce_public_law(public_law)
        lower_result = self._solve_uncertain_linear_confidence_bound(
            problem,
            public_law=p,
            bound="lower",
        )
        upper_result = self._solve_uncertain_linear_confidence_bound(
            problem,
            public_law=p,
            bound="upper",
        )
        lower = lower_result.value
        upper = upper_result.value
        return UncertainLinearConfidenceCoreResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            empty_gap=max(0.0, lower - upper),
            public_law=p,
            q_lower=problem._distribution_from_vector(lower_result.vector),
            q_upper=problem._distribution_from_vector(upper_result.vector),
            duals=lower_result.duals + upper_result.duals,
        )

    def _ratio_local_transport(
        self,
        problem,
        public_law: Mapping[Hashable, float],
    ) -> TransportResult:
        p = problem._coerce_public_law(public_law)
        lower_result = self._solve_ratio_single_result(
            problem,
            maximize=False,
            public_law=p,
        )
        upper_result = self._solve_ratio_single_result(
            problem,
            maximize=True,
            public_law=p,
        )
        return TransportResult(
            lower=lower_result.value,
            upper=upper_result.value,
            diameter=max(0.0, upper_result.value - lower_result.value),
            public_law=p,
            q_lower=problem._distribution_from_vector(lower_result.vector),
            q_upper=problem._distribution_from_vector(upper_result.vector),
            duals=lower_result.duals + upper_result.duals,
        )

    def _normalized_constraints(self) -> tuple[LinearConstraint, ...]:
        normalized = []
        for constraint in self.constraints:
            if isinstance(constraint, LinearConstraint):
                normalized.append(constraint)
                continue
            if len(constraint) != 3:
                raise ValueError("constraint tuples must be (coefficients, sense, rhs)")
            coefficients, sense, rhs = constraint
            normalized.append(LinearConstraint(coefficients, sense, float(rhs)))
        return tuple(normalized)

    def _solve_single(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> tuple[tuple[float, ...], float]:
        result = self._solve_single_result(
            problem,
            objective,
            maximize=maximize,
            public_law=public_law,
        )
        return result.vector, result.value

    def _solve_uncertain_linear_confidence_bound(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
        bound: str,
    ) -> _CvxpySolveResult:
        import numpy as np

        target = problem.target_functional
        if not isinstance(target, UncertainLinearTarget):
            raise TypeError("problem target is not an UncertainLinearTarget")
        if bound not in {"lower", "upper"}:
            raise ValueError("bound must be 'lower' or 'upper'")

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        means = np.array(
            [target.value(state) for state in problem.states],
            dtype=float,
        )
        standard_errors = np.array(
            [target.standard_error(state) for state in problem.states],
            dtype=float,
        )
        mean_expr = cp.sum(cp.multiply(means, q))
        standard_error_expr = cp.norm(cp.multiply(standard_errors, q), 2)
        confidence_expr = mean_expr - (
            target.confidence_multiplier * standard_error_expr
        )
        if bound == "upper":
            confidence_expr = mean_expr + (
                target.confidence_multiplier * standard_error_expr
            )
        records = self._labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(confidence_expr)
            if bound == "lower"
            else cp.Minimize(confidence_expr),
            [record.constraint for record in records],
        )
        if not cvx_problem.is_dcp():
            raise CvxpyError(
                "UncertainLinearTarget confidence-core endpoint is not DCP. "
                "The lower endpoint should be a concave maximization and the "
                "upper endpoint should be a convex minimization."
            )
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        mean = float(np.dot(means, vector))
        standard_error = target.standard_error_for_distribution(
            problem.states,
            vector,
        )
        sign = -1.0 if bound == "lower" else 1.0
        return _CvxpySolveResult(
            vector=vector,
            value=mean + sign * target.confidence_multiplier * standard_error,
            duals=self._constraint_duals(
                records,
                solve=f"confidence_core_{bound}",
            ),
        )

    def _solve_single_result(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> _CvxpySolveResult:
        import numpy as np

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        objective_vector = np.array(objective, dtype=float)
        objective_expr = cp.sum(cp.multiply(objective_vector, q))
        records = self._labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(objective_expr) if maximize else cp.Minimize(objective_expr),
            [record.constraint for record in records],
        )
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        solve = "upper" if maximize else "lower"
        return _CvxpySolveResult(
            vector=vector,
            value=float(np.dot(objective_vector, vector)),
            duals=self._constraint_duals(records, solve=solve),
        )

    def _solve_ratio_single_result(
        self,
        problem,
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> _CvxpySolveResult:
        cp = self._cvxpy()
        q = cp.Variable(len(problem.states), nonneg=True)
        target = problem.target_functional
        numerator = tuple(
            float(value) for value in target.numerator_vector(problem.states)
        )
        denominator = tuple(
            float(value) for value in target.denominator_vector(problem.states)
        )
        numerator_expr = cp.sum(cp.multiply(numerator, q))
        denominator_expr = cp.sum(cp.multiply(denominator, q))
        objective_expr = numerator_expr / denominator_expr
        records = self._ratio_labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(objective_expr) if maximize else cp.Minimize(objective_expr),
            [record.constraint for record in records],
        )
        if not cvx_problem.is_dqcp():
            raise CvxpyError(
                "RatioTarget objective is not DQCP under the current CVXPY "
                "constraints. Use PublicFiberSaturated, FiniteEnvironments, or "
                "a constant-denominator LinearTarget reformulation."
            )
        values = [problem.estimand_map[state] for state in problem.states]
        span = max(values) - min(values)
        margin = max(1.0, span) * 0.1
        self._solve_problem(
            cvx_problem,
            qcp=True,
            low=min(values) - margin,
            high=max(values) + margin,
        )
        vector = self._clean_vector(problem, q.value)
        solve = "upper" if maximize else "lower"
        return _CvxpySolveResult(
            vector=vector,
            value=problem._dot_estimand(vector),
            duals=self._constraint_duals(records, solve=solve),
        )

    def _ratio_labeled_constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> list[CvxpyConstraintMetadata]:
        records = [
            cvxpy_constraint(
                cp.sum(q) == 1,
                name="probability normalization",
                kind="normalization",
                sense="==",
                variable=variable,
            ),
        ]
        records.extend(
            self._linear_constraint_records(cp, problem, q, variable=variable)
        )

        fixed_public_law = self._fixed_public_law(problem)
        if fixed_public_law is not None:
            if public_law is not None and not self._same_public_law(
                problem, fixed_public_law, public_law
            ):
                raise ValueError("requested public law conflicts with fixed_public_law")
            records.extend(
                self._public_law_constraint_records(
                    cp,
                    problem,
                    q,
                    fixed_public_law,
                    variable=variable,
                )
            )
        elif public_law is not None:
            records.extend(
                self._public_law_constraint_records(
                    cp,
                    problem,
                    q,
                    public_law,
                    variable=variable,
                )
            )

        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        for i, builder in enumerate(self.constraint_builders):
            built = builder(cp, q, states, state_index)
            records.extend(
                self._coerce_constraint_records(
                    built,
                    default_name=f"custom constraint builder {i}",
                    default_kind="custom",
                    variable=variable,
                )
            )
        return records

    def _solve_pair(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        result = self._solve_pair_result(problem, objective, support=support)
        return result.q1, result.q2, result.value

    def _solve_pair_result(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> _CvxpyPairSolveResult:
        import numpy as np

        cp = self._cvxpy()
        n = len(problem.states)
        q1 = cp.Variable(n)
        q2 = cp.Variable(n)
        objective_vector = np.array(objective, dtype=float)
        records = [
            *self._labeled_constraints_for_variable(
                cp,
                problem,
                q1,
                variable="q1",
            ),
            *self._labeled_constraints_for_variable(
                cp,
                problem,
                q2,
                variable="q2",
            ),
        ]
        state_index = {state: i for i, state in enumerate(problem.states)}
        for i, block in enumerate(support.blocks):
            indices = [state_index[state] for state in block]
            records.append(
                cvxpy_constraint(
                    cp.sum(q1[indices]) == cp.sum(q2[indices]),
                    name=f"support-law equality block {i}",
                    kind="support_equality",
                    sense="==",
                    states=block,
                )
            )

        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective_vector, q1 - q2))),
            [record.constraint for record in records],
        )
        self._solve_problem(cvx_problem)
        q1_vector = self._clean_vector(problem, q1.value)
        q2_vector = self._clean_vector(problem, q2.value)
        value = float(
            np.dot(objective_vector, np.array(q1_vector) - np.array(q2_vector))
        )
        return _CvxpyPairSolveResult(
            q1=q1_vector,
            q2=q2_vector,
            value=value,
            duals=self._constraint_duals(records, solve="gap"),
        )

    def _max_support_disagreement(
        self, problem, support: Partition
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        result = self._max_support_disagreement_result(problem, support)
        return result.q1, result.q2, max(0.0, abs(result.value))

    def _max_support_disagreement_result(
        self, problem, support: Partition
    ) -> _CvxpyPairSolveResult:
        result = self._solve_pair_result(
            problem,
            self._estimand_vector(problem),
            support=support,
        )
        if result.value >= 0.0:
            return result
        return _CvxpyPairSolveResult(
            q1=result.q2,
            q2=result.q1,
            value=abs(result.value),
            duals=result.duals,
        )

    def _constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> list[Any]:
        return [
            record.constraint
            for record in self._labeled_constraints_for_variable(
                cp,
                problem,
                q,
                variable="q",
                public_law=public_law,
            )
        ]

    def _labeled_constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> list[CvxpyConstraintMetadata]:
        records = [
            cvxpy_constraint(
                q >= 0,
                name="state probability lower bound",
                kind="lower_bound",
                sense=">=",
                variable=variable,
                states=problem.states,
            ),
            cvxpy_constraint(
                cp.sum(q) == 1,
                name="probability normalization",
                kind="normalization",
                sense="==",
                variable=variable,
            ),
        ]
        records.extend(
            self._linear_constraint_records(cp, problem, q, variable=variable)
        )

        fixed_public_law = self._fixed_public_law(problem)
        if fixed_public_law is not None:
            if public_law is not None and not self._same_public_law(
                problem, fixed_public_law, public_law
            ):
                raise ValueError("requested public law conflicts with fixed_public_law")
            records.extend(
                self._public_law_constraint_records(
                    cp,
                    problem,
                    q,
                    fixed_public_law,
                    variable=variable,
                )
            )
        elif public_law is not None:
            records.extend(
                self._public_law_constraint_records(
                    cp,
                    problem,
                    q,
                    public_law,
                    variable=variable,
                )
            )

        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        for i, builder in enumerate(self.constraint_builders):
            built = builder(cp, q, states, state_index)
            records.extend(
                self._coerce_constraint_records(
                    built,
                    default_name=f"custom constraint builder {i}",
                    default_kind="custom",
                    variable=variable,
                )
            )
        return records

    def _linear_constraints(self, cp, problem, q) -> list[Any]:
        return [
            record.constraint
            for record in self._linear_constraint_records(cp, problem, q, variable="q")
        ]

    def _linear_constraint_records(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        constraints = []
        for i, constraint in enumerate(self._normalized_constraints()):
            vector = problem._coerce_vector(constraint.coefficients)
            expr = cp.sum(cp.multiply(tuple(float(value) for value in vector), q))
            name = constraint.name or f"linear constraint {i}"
            if constraint.sense == "<=":
                cvx_constraint = expr <= float(constraint.rhs)
            elif constraint.sense == ">=":
                cvx_constraint = expr >= float(constraint.rhs)
            elif constraint.sense == "==":
                cvx_constraint = expr == float(constraint.rhs)
            else:
                raise ValueError(f"unsupported constraint sense: {constraint.sense!r}")
            constraints.append(
                cvxpy_constraint(
                    cvx_constraint,
                    name=name,
                    kind="linear",
                    sense=constraint.sense,
                    variable=variable,
                )
            )
        return constraints

    def _public_law_constraints(
        self,
        cp,
        problem,
        q,
        public_law: Mapping[Hashable, float],
    ) -> list[Any]:
        return [
            record.constraint
            for record in self._public_law_constraint_records(
                cp,
                problem,
                q,
                public_law,
                variable="q",
            )
        ]

    def _public_law_constraint_records(
        self,
        cp,
        problem,
        q,
        public_law: Mapping[Hashable, float],
        *,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        state_index = {state: i for i, state in enumerate(problem.states)}
        constraints = []
        for public_value in problem.public_values:
            indices = [
                state_index[state] for state in problem.public_fibers[public_value]
            ]
            constraints.append(
                cvxpy_constraint(
                    cp.sum(q[indices]) == float(public_law[public_value]),
                    name=f"public-law equality {public_value!r}",
                    kind="public_law",
                    sense="==",
                    variable=variable,
                    public_value=public_value,
                )
            )
        return constraints

    def _coerce_constraint_records(
        self,
        constraints: Sequence[Any],
        *,
        default_name: str,
        default_kind: str,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        records = []
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, CvxpyConstraintMetadata):
                if constraint.variable is None:
                    constraint = CvxpyConstraintMetadata(
                        constraint=constraint.constraint,
                        name=constraint.name,
                        kind=constraint.kind,
                        sense=constraint.sense,
                        variable=variable,
                        state=constraint.state,
                        public_value=constraint.public_value,
                        states=constraint.states,
                        public_values=constraint.public_values,
                    )
                records.append(constraint)
                continue
            records.append(
                cvxpy_constraint(
                    constraint,
                    name=f"{default_name} constraint {i}",
                    kind=default_kind,
                    variable=variable,
                )
            )
        return records

    def _constraint_duals(
        self,
        records: Sequence[CvxpyConstraintMetadata],
        *,
        solve: str,
    ) -> tuple[ConstraintDual, ...]:
        import numpy as np

        rows = []
        for record in records:
            dual_value = record.constraint.dual_value
            if dual_value is None:
                continue
            residual = self._constraint_residual(record.constraint)
            array = np.asarray(dual_value, dtype=float)
            if array.ndim == 0:
                signed = float(array)
                rows.append(
                    ConstraintDual(
                        solve=solve,
                        name=record.name,
                        kind=record.kind,
                        magnitude=abs(signed),
                        signed_value=signed,
                        sense=record.sense,
                        variable=record.variable,
                        state=record.state,
                        public_value=record.public_value,
                        residual=residual,
                    )
                )
                continue

            flat_states = record.states
            flat_public_values = record.public_values
            for index in np.ndindex(array.shape):
                signed = float(array[index])
                if abs(signed) <= 1e-10:
                    continue
                flat_index = int(np.ravel_multi_index(index, array.shape))
                state = (
                    flat_states[flat_index]
                    if flat_index < len(flat_states)
                    else record.state
                )
                public_value = (
                    flat_public_values[flat_index]
                    if flat_index < len(flat_public_values)
                    else record.public_value
                )
                rows.append(
                    ConstraintDual(
                        solve=solve,
                        name=record.name,
                        kind=record.kind,
                        magnitude=abs(signed),
                        signed_value=signed,
                        sense=record.sense,
                        variable=record.variable,
                        state=state,
                        public_value=public_value,
                        index=tuple(int(item) for item in index),
                        residual=residual,
                    )
                )
        return tuple(rows)

    def _constraint_residual(self, constraint) -> float | None:
        try:
            residual = constraint.violation()
        except Exception:  # pragma: no cover - CVXPY residual support varies
            return None
        try:
            import numpy as np

            return float(np.max(np.abs(np.asarray(residual, dtype=float))))
        except Exception:  # pragma: no cover - defensive for unusual residuals
            return None

    def _fixed_public_law(self, problem) -> dict[Hashable, float] | None:
        if self.fixed_public_law is None:
            return None
        return problem._coerce_public_law(self.fixed_public_law)

    def _same_public_law(
        self,
        problem,
        left: Mapping[Hashable, float],
        right: Mapping[Hashable, float],
    ) -> bool:
        return problem._same_vector(
            tuple(left[value] for value in problem.public_values),
            tuple(right[value] for value in problem.public_values),
        )

    def _solve_problem(
        self,
        problem,
        *,
        qcp: bool = False,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        options = dict(self.solver_options or {})
        if qcp:
            options.setdefault("qcp", True)
            options.setdefault("eps", 1e-5)
            if low is not None:
                options.setdefault("low", low)
            if high is not None:
                options.setdefault("high", high)
        solver = self._normalized_solver_name()
        try:
            if solver is None:
                problem.solve(**options)
            else:
                problem.solve(solver=solver, **options)
        except Exception as exc:  # pragma: no cover - solver exceptions vary by backend
            raise CvxpyError(str(exc)) from exc

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise CvxpyError(f"CVXPY problem status: {problem.status}")

    def _normalized_solver_name(self) -> str | None:
        if self.solver is None:
            return None
        requested = str(self.solver)
        installed = {
            str(solver).upper(): str(solver)
            for solver in self._cvxpy().installed_solvers()
        }
        matched = installed.get(requested.upper())
        if matched is not None:
            return matched

        hint = ""
        if requested.upper() == "SCIP":
            hint = (
                " Install SCIP support with `pip install updatesupport[scip]` "
                "or `uv add updatesupport[scip]`."
            )
        raise CvxpyError(f"CVXPY solver {requested!r} is not installed.{hint}")

    def _clean_vector(self, problem, value) -> tuple[float, ...]:
        if value is None:
            raise CvxpyError("CVXPY did not return a solution vector")
        vector = tuple(float(item) for item in value)
        if any(item < -1e-7 for item in vector):
            raise CvxpyError("CVXPY returned a negative probability")
        return tuple(
            0.0 if item < 0.0 or abs(item) <= problem.tol else item for item in vector
        )

    def _estimand_vector(self, problem) -> tuple[float, ...]:
        return tuple(problem.estimand_map[state] for state in problem.states)

    @staticmethod
    def _cvxpy():
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - exercised without optional dep
            raise CvxpyError(
                "CVXPY support is optional; install it with "
                "`uv sync --extra cvxpy` or `pip install updatesupport[cvxpy]`."
            ) from exc
        return cp


@dataclass(frozen=True)
class SupportFunctionBackend(CvxpyEnvironments):
    """CVXPY backend that evaluates linear intervals via support functions."""

    name: str = "support-function-cvxpy"

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            return super().local_transport(problem, public_law)
        problem._require_linear_target("SupportFunctionBackend.local_transport")
        p = problem._coerce_public_law(public_law)
        h = self._estimand_vector(problem)
        lower_result = self._support_function_result(
            problem,
            tuple(-value for value in h),
            public_law=p,
            solve="lower",
        )
        upper_result = self._support_function_result(
            problem,
            h,
            public_law=p,
            solve="upper",
        )
        lower = -lower_result.value
        upper = upper_result.value
        if lower > upper and abs(lower - upper) <= problem.tol:
            lower = upper = 0.5 * (lower + upper)
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            public_law=p,
            q_lower=problem._distribution_from_vector(lower_result.vector),
            q_upper=problem._distribution_from_vector(upper_result.vector),
            duals=lower_result.duals + upper_result.duals,
        )

    def global_transport(self, problem) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            return super().global_transport(problem)
        problem._require_linear_target("SupportFunctionBackend.global_transport")
        if self.fixed_public_law is not None:
            return self.local_transport(problem, self.fixed_public_law)
        return super().global_transport(problem)

    def support_value(
        self,
        problem,
        direction: Mapping[Hashable, float] | Sequence[float],
        *,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> SupportFunctionResult:
        """Evaluate the support function of this backend's admissible set."""

        direction_vector = problem._coerce_vector(direction)
        result = self._support_function_result(
            problem,
            direction_vector,
            public_law=public_law,
            solve="support",
        )
        return SupportFunctionResult(
            direction=direction_vector,
            value=result.value,
            vector=result.vector,
        )

    def support_interval(
        self,
        problem,
        direction: Sequence[float] | None = None,
        *,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> SupportFunctionIntervalResult:
        """Evaluate a support-function interval for a direction or target."""

        if direction is None:
            problem._require_linear_target("SupportFunctionBackend.support_interval")
            direction = self._estimand_vector(problem)
        return self._support_interval_result(
            problem,
            direction,
            public_law=public_law,
        )

    def multi_target_intervals(
        self,
        problem,
        targets: Mapping[str, Mapping[Hashable, float] | Sequence[float]],
        *,
        public_law: Mapping[Hashable, float] | None = None,
        title: str = "Support-Function Multi-Target Report",
    ) -> SupportFunctionReport:
        """Evaluate support-function intervals for several linear directions."""

        if not targets:
            raise ValueError("targets must contain at least one named direction")
        report_public_law = (
            problem._coerce_public_law(public_law)
            if public_law is not None
            else self._fixed_public_law(problem)
        )
        intervals = []
        for raw_name, direction in targets.items():
            name = str(raw_name)
            if not name:
                raise ValueError("target names must be non-empty")
            direction_vector = problem._coerce_vector(direction)
            interval = self._target_interval(
                problem,
                name=name,
                direction=direction_vector,
                public_law=report_public_law,
            )
            intervals.append(interval)
        return SupportFunctionReport(
            title=title,
            targets=tuple(intervals),
            states=tuple(problem.states),
            public_values=tuple(problem.public_values),
            public_law=report_public_law,
            backend=self.name,
        )

    def _support_interval_result(
        self,
        problem,
        direction: Mapping[Hashable, float] | Sequence[float],
        *,
        public_law: Mapping[Hashable, float] | None,
    ) -> SupportFunctionIntervalResult:
        direction_vector = problem._coerce_vector(direction)
        lower_result = self._support_function_result(
            problem,
            tuple(-value for value in direction_vector),
            public_law=public_law,
            solve="lower",
        )
        upper_result = self._support_function_result(
            problem,
            direction_vector,
            public_law=public_law,
            solve="upper",
        )
        lower = -lower_result.value
        upper = upper_result.value
        if lower > upper and abs(lower - upper) <= problem.tol:
            lower = upper = 0.5 * (lower + upper)
        return SupportFunctionIntervalResult(
            direction=direction_vector,
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            lower_vector=lower_result.vector,
            upper_vector=upper_result.vector,
            lower_support_value=lower_result.value,
            upper_support_value=upper_result.value,
            lower_support_result=SupportFunctionResult(
                direction=tuple(-value for value in direction_vector),
                value=lower_result.value,
                vector=lower_result.vector,
            ),
            upper_support_result=SupportFunctionResult(
                direction=direction_vector,
                value=upper_result.value,
                vector=upper_result.vector,
            ),
            lower_duals=lower_result.duals,
            upper_duals=upper_result.duals,
        )

    def _target_interval(
        self,
        problem,
        *,
        name: str,
        direction: Mapping[Hashable, float] | Sequence[float],
        public_law: Mapping[Hashable, float] | None,
    ) -> SupportFunctionTargetInterval:
        interval = self._support_interval_result(
            problem,
            direction,
            public_law=public_law,
        )
        return SupportFunctionTargetInterval(
            name=name,
            direction=interval.direction,
            lower=interval.lower,
            upper=interval.upper,
            diameter=interval.diameter,
            lower_distribution=problem._distribution_from_vector(interval.lower_vector),
            upper_distribution=problem._distribution_from_vector(interval.upper_vector),
            lower_support_value=interval.lower_support_value,
            upper_support_value=interval.upper_support_value,
            lower_duals=interval.lower_duals,
            upper_duals=interval.upper_duals,
        )

    def _support_function_result(
        self,
        problem,
        direction: Sequence[float],
        *,
        public_law: Mapping[Hashable, float] | None,
        solve: str,
    ) -> _CvxpySolveResult:
        direction_vector = tuple(float(value) for value in direction)
        if len(direction_vector) != len(problem.states):
            raise ValueError("support direction must have one value per state")
        admissible_set = self.convex_admissible_set(problem, public_law=public_law)
        result = admissible_set.support_value(
            direction_vector,
            solver=self._normalized_solver_name(),
            solver_options=self.solver_options,
        )
        vector = self._clean_vector(problem, result.vector)
        duals = self._constraint_duals(admissible_set.records, solve=solve)
        return _CvxpySolveResult(
            vector=vector,
            value=result.value,
            duals=duals,
        )


def support_function_report(
    problem,
    targets: Mapping[str, Mapping[Hashable, float] | Sequence[float]],
    *,
    public_law: Mapping[Hashable, float] | None = None,
    title: str = "Support-Function Multi-Target Report",
) -> SupportFunctionReport:
    """Evaluate several linear target directions with a support-function backend."""

    env = problem.environments
    if not isinstance(env, SupportFunctionBackend):
        raise TypeError(
            "support_function_report requires a SupportFunctionBackend. "
            "Use a CVXPY Q preset with backend='support_function' or set "
            "problem.environments to SupportFunctionBackend."
        )
    return env.multi_target_intervals(
        problem,
        targets,
        public_law=public_law,
        title=title,
    )


@dataclass(frozen=True)
class BatchedCvxpyEnvironments(CvxpyEnvironments):
    """CVXPY environment with batched local interval solves.

    ``batched_local_transport`` solves independent fixed-public-law lower/upper
    endpoint problems with variables shaped ``(scenario, state)``. Existing
    one-dimensional custom CVXPY constraint builders are applied to each
    scenario slice, so current TV, chi-square, KL, and Wasserstein presets can
    be reused before specialized tensor builders are needed.
    """

    scenario_constraint_builders: Sequence[Sequence[CvxpyConstraintBuilder]] | None = (
        None
    )
    scenario_names: Sequence[str] = ()
    name: str = "batched-cvxpy"

    def batched_local_transport(
        self,
        problem,
        public_laws: Sequence[Mapping[Hashable, float]],
    ) -> tuple[TransportResult, ...]:
        if _is_nonlinear_ratio_problem(problem):
            raise UnsupportedTargetError(
                "BatchedCvxpyEnvironments does not yet support non-linear "
                "RatioTarget solves."
            )
        problem._require_linear_target(
            "BatchedCvxpyEnvironments.batched_local_transport"
        )
        laws = tuple(
            problem._coerce_public_law(public_law) for public_law in public_laws
        )
        if not laws:
            return ()
        fixed_public_law = self._fixed_public_law(problem)
        if fixed_public_law is not None:
            for public_law in laws:
                if not self._same_public_law(problem, fixed_public_law, public_law):
                    raise ValueError(
                        "requested public law conflicts with fixed_public_law"
                    )

        import numpy as np

        cp = self._cvxpy()
        scenario_count = len(laws)
        state_count = len(problem.states)
        q_lower = cp.Variable((scenario_count, state_count))
        q_upper = cp.Variable((scenario_count, state_count))
        records = [
            *self._batched_labeled_constraints_for_variable(
                cp,
                problem,
                q_lower,
                variable="q_lower",
                public_laws=laws,
            ),
            *self._batched_labeled_constraints_for_variable(
                cp,
                problem,
                q_upper,
                variable="q_upper",
                public_laws=laws,
            ),
        ]
        objective = np.array(self._estimand_vector(problem), dtype=float)
        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum((q_upper - q_lower) @ objective)),
            [record.constraint for record in records],
        )
        self._solve_problem(cvx_problem)

        lower_values = np.asarray(q_lower.value, dtype=float)
        upper_values = np.asarray(q_upper.value, dtype=float)
        duals = self._constraint_duals(records, solve="batched-local")
        results = []
        for scenario_index, public_law in enumerate(laws):
            lower_vector = self._clean_vector(problem, lower_values[scenario_index])
            upper_vector = self._clean_vector(problem, upper_values[scenario_index])
            lower = float(np.dot(objective, np.array(lower_vector, dtype=float)))
            upper = float(np.dot(objective, np.array(upper_vector, dtype=float)))
            if lower > upper and abs(lower - upper) <= problem.tol:
                lower = upper = 0.5 * (lower + upper)
            results.append(
                TransportResult(
                    lower=lower,
                    upper=upper,
                    diameter=max(0.0, upper - lower),
                    public_law=public_law,
                    q_lower=problem._distribution_from_vector(lower_vector),
                    q_upper=problem._distribution_from_vector(upper_vector),
                    duals=self._scenario_duals(duals, scenario_index),
                )
            )
        return tuple(results)

    def _batched_labeled_constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
        public_laws: Sequence[Mapping[Hashable, float]],
    ) -> list[CvxpyConstraintMetadata]:
        scenario_count = len(public_laws)
        records = [
            cvxpy_constraint(
                q >= 0,
                name="batched state probability lower bound",
                kind="lower_bound",
                sense=">=",
                variable=variable,
                states=tuple(
                    state for _ in range(scenario_count) for state in problem.states
                ),
            ),
            cvxpy_constraint(
                cp.sum(q, axis=1) == 1,
                name="batched probability normalization",
                kind="normalization",
                sense="==",
                variable=variable,
            ),
        ]
        records.extend(
            self._batched_linear_constraint_records(
                cp,
                problem,
                q,
                variable=variable,
            )
        )
        records.extend(
            self._batched_public_law_constraint_records(
                cp,
                problem,
                q,
                public_laws,
                variable=variable,
            )
        )
        records.extend(
            self._batched_custom_constraint_records(
                cp,
                problem,
                q,
                variable=variable,
                scenario_count=scenario_count,
            )
        )
        return records

    def _batched_linear_constraint_records(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        import numpy as np

        records = []
        for i, constraint in enumerate(self._normalized_constraints()):
            vector = np.array(
                problem._coerce_vector(constraint.coefficients), dtype=float
            )
            expr = q @ vector
            rhs = float(constraint.rhs)
            if constraint.sense == "<=":
                cvx_constraint = expr <= rhs
            elif constraint.sense == ">=":
                cvx_constraint = expr >= rhs
            elif constraint.sense == "==":
                cvx_constraint = expr == rhs
            else:
                raise ValueError(f"unsupported constraint sense: {constraint.sense!r}")
            records.append(
                cvxpy_constraint(
                    cvx_constraint,
                    name=constraint.name or f"batched linear constraint {i}",
                    kind="linear",
                    sense=constraint.sense,
                    variable=variable,
                )
            )
        return records

    def _batched_public_law_constraint_records(
        self,
        cp,
        problem,
        q,
        public_laws: Sequence[Mapping[Hashable, float]],
        *,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        import numpy as np

        public_indicator = np.array(
            [
                [
                    1.0 if problem.public_map[state] == public_value else 0.0
                    for state in problem.states
                ]
                for public_value in problem.public_values
            ],
            dtype=float,
        )
        public_array = np.array(
            [
                [
                    float(public_law[public_value])
                    for public_value in problem.public_values
                ]
                for public_law in public_laws
            ],
            dtype=float,
        )
        return [
            cvxpy_constraint(
                q @ public_indicator.T == public_array,
                name="batched public-law equality",
                kind="public_law",
                sense="==",
                variable=variable,
                public_values=tuple(
                    public_value
                    for _ in range(len(public_laws))
                    for public_value in problem.public_values
                ),
            )
        ]

    def _batched_custom_constraint_records(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
        scenario_count: int,
    ) -> list[CvxpyConstraintMetadata]:
        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        records = []
        for scenario_index, builders in enumerate(
            self._builders_by_scenario(scenario_count)
        ):
            scenario_variable = f"{variable}[{scenario_index}]"
            for i, builder in enumerate(builders):
                built = builder(cp, q[scenario_index, :], states, state_index)
                records.extend(
                    self._coerce_constraint_records(
                        built,
                        default_name=(
                            f"scenario {scenario_index} custom constraint builder {i}"
                        ),
                        default_kind="custom",
                        variable=scenario_variable,
                    )
                )
        return records

    def _builders_by_scenario(
        self,
        scenario_count: int,
    ) -> tuple[tuple[CvxpyConstraintBuilder, ...], ...]:
        if self.scenario_constraint_builders is None:
            return tuple(tuple(self.constraint_builders) for _ in range(scenario_count))
        builders = tuple(
            tuple(builder_group) for builder_group in self.scenario_constraint_builders
        )
        if len(builders) != scenario_count:
            raise ValueError(
                "scenario_constraint_builders must have one entry per scenario"
            )
        return builders

    def _scenario_duals(
        self,
        duals: Sequence[ConstraintDual],
        scenario_index: int,
    ) -> tuple[ConstraintDual, ...]:
        scenario_suffix = f"[{scenario_index}]"
        selected = []
        for dual in duals:
            if (
                dual.index is not None
                and dual.index
                and dual.index[0] == scenario_index
            ):
                selected.append(dual)
                continue
            if dual.variable is not None and dual.variable.endswith(scenario_suffix):
                selected.append(dual)
        return tuple(selected)


@dataclass(frozen=True)
class _ParameterizedCvxpySingleProblem:
    problem: Any
    q: Any
    objective: Any
    public_law: Any | None
    extra_parameters: dict[str, Any]
    records: tuple[CvxpyConstraintMetadata, ...]


@dataclass(frozen=True)
class _ParameterizedCvxpyPairProblem:
    problem: Any
    q1: Any
    q2: Any
    objective: Any
    public_law: Any | None
    extra_parameters: dict[str, Any]
    records: tuple[CvxpyConstraintMetadata, ...]


@dataclass(frozen=True)
class ParameterizedCvxpyEnvironments(CvxpyEnvironments):
    """CVXPY environment with cached problems and mutable parameters.

    This backend is intended for sensitivity sweeps over a fixed finite state
    space. The objective and public-law equalities are CVXPY parameters, and
    custom parameterized constraints can add scalar/vector parameters such as a
    divergence radius.
    """

    parameterized_constraint_builders: Sequence[
        CvxpyParameterizedConstraintBuilder
    ] = ()
    parameter_values: Mapping[str, Any] = field(default_factory=dict)
    name: str = "parameterized-cvxpy"
    _single_cache: dict[tuple[Any, ...], _ParameterizedCvxpySingleProblem] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _pair_cache: dict[tuple[Any, ...], _ParameterizedCvxpyPairProblem] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_values", dict(self.parameter_values))

    def set_parameter(self, name: str, value: Any) -> "ParameterizedCvxpyEnvironments":
        """Set a parameter value in-place and return ``self`` for chaining."""

        self.parameter_values[name] = value
        return self

    def with_parameters(self, **values: Any) -> "ParameterizedCvxpyEnvironments":
        """Return a copy with updated parameter values and an empty problem cache."""

        parameters = dict(self.parameter_values)
        parameters.update(values)
        return ParameterizedCvxpyEnvironments(
            constraints=self.constraints,
            constraint_builders=self.constraint_builders,
            fixed_public_law=self.fixed_public_law,
            solver=self.solver,
            solver_options=self.solver_options,
            name=self.name,
            parameterized_constraint_builders=self.parameterized_constraint_builders,
            parameter_values=parameters,
        )

    def clear_cache(self) -> None:
        """Discard cached CVXPY problem objects."""

        self._single_cache.clear()
        self._pair_cache.clear()

    def cache_info(self) -> dict[str, int]:
        return {
            "single_problems": len(self._single_cache),
            "pair_problems": len(self._pair_cache),
        }

    def check_support(self, problem, support: Partition) -> AdequacyResult:
        if _is_nonlinear_ratio_problem(problem):
            problem._require_linear_target(
                "ParameterizedCvxpyEnvironments.check_support"
            )
        return super().check_support(problem, support)

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            problem._require_linear_target(
                "ParameterizedCvxpyEnvironments.local_transport"
            )
        return super().local_transport(problem, public_law)

    def global_transport(self, problem) -> TransportResult:
        if _is_nonlinear_ratio_problem(problem):
            problem._require_linear_target(
                "ParameterizedCvxpyEnvironments.global_transport"
            )
        return super().global_transport(problem)

    def moment_transform_endpoint(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
        maximize: bool,
    ) -> TransportResult:
        target = problem.target_functional
        if not isinstance(target, MomentTransformTarget):
            raise TypeError("problem target is not a MomentTransformTarget")
        if maximize and not target.supports_exact_upper_endpoint:
            raise UnsupportedTargetError(
                "MomentTransformTarget maximization is exact only for concave "
                "CVXPY-compatible transforms."
            )
        if not maximize and not target.supports_exact_lower_endpoint:
            raise UnsupportedTargetError(
                "MomentTransformTarget minimization is exact only for convex "
                "CVXPY-compatible transforms."
            )

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        public_law_parameter = cp.Parameter(len(problem.public_values))
        extra_parameters: dict[str, Any] = {}
        records = self._parameterized_labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law_parameter,
            extra_parameters=extra_parameters,
        )
        moment_exprs = {}
        for moment in target.moments:
            vector = target.moment_vector(moment, problem.states)
            moment_exprs[moment] = cp.sum(cp.multiply(vector, q))
        objective_expr = target.cvxpy_expression(cp, moment_exprs)
        cvx_problem = cp.Problem(
            cp.Maximize(objective_expr) if maximize else cp.Minimize(objective_expr),
            [record.constraint for record in records],
        )
        if not cvx_problem.is_dcp():
            direction = "maximization" if maximize else "minimization"
            raise CvxpyError(
                f"MomentTransformTarget {direction} is not DCP. Convex targets "
                "support minimization and concave targets support maximization."
            )
        effective_public_law = self._effective_public_law(problem, public_law)
        public_law_parameter.value = self._public_law_array(
            problem,
            effective_public_law,
        )
        self._set_extra_parameter_values(extra_parameters)
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        state_index = {state: i for i, state in enumerate(problem.states)}
        moment_values = {
            moment: sum(
                target.moment_value(moment, state) * vector[state_index[state]]
                for state in problem.states
            )
            for moment in target.moments
        }
        value = target.transform_value(moment_values)
        q_distribution = problem._distribution_from_vector(vector)
        solve = "upper" if maximize else "lower"
        return TransportResult(
            lower=value,
            upper=value,
            diameter=0.0,
            public_law=effective_public_law,
            q_lower=q_distribution,
            q_upper=q_distribution,
            duals=self._constraint_duals(records, solve=solve),
        )

    def uncertain_linear_confidence_core(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
    ) -> UncertainLinearConfidenceCoreResult:
        effective_public_law = self._effective_public_law(problem, public_law)
        if effective_public_law is None:
            raise ValueError("public_law is required for confidence-core solving")
        return super().uncertain_linear_confidence_core(
            problem,
            public_law=effective_public_law,
        )

    def _solve_uncertain_linear_confidence_bound(
        self,
        problem,
        *,
        public_law: Mapping[Hashable, float],
        bound: str,
    ) -> _CvxpySolveResult:
        import numpy as np

        target = problem.target_functional
        if not isinstance(target, UncertainLinearTarget):
            raise TypeError("problem target is not an UncertainLinearTarget")
        if bound not in {"lower", "upper"}:
            raise ValueError("bound must be 'lower' or 'upper'")

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        public_law_parameter = cp.Parameter(len(problem.public_values))
        extra_parameters: dict[str, Any] = {}
        records = self._parameterized_labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law_parameter,
            extra_parameters=extra_parameters,
        )
        means = np.array(
            [target.value(state) for state in problem.states],
            dtype=float,
        )
        standard_errors = np.array(
            [target.standard_error(state) for state in problem.states],
            dtype=float,
        )
        mean_expr = cp.sum(cp.multiply(means, q))
        standard_error_expr = cp.norm(cp.multiply(standard_errors, q), 2)
        confidence_expr = mean_expr - (
            target.confidence_multiplier * standard_error_expr
        )
        if bound == "upper":
            confidence_expr = mean_expr + (
                target.confidence_multiplier * standard_error_expr
            )
        cvx_problem = cp.Problem(
            cp.Maximize(confidence_expr)
            if bound == "lower"
            else cp.Minimize(confidence_expr),
            [record.constraint for record in records],
        )
        if not cvx_problem.is_dcp():
            raise CvxpyError(
                "UncertainLinearTarget confidence-core endpoint is not DCP. "
                "The lower endpoint should be a concave maximization and the "
                "upper endpoint should be a convex minimization."
            )
        public_law_parameter.value = self._public_law_array(problem, public_law)
        self._set_extra_parameter_values(extra_parameters)
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        mean = float(np.dot(means, vector))
        standard_error = target.standard_error_for_distribution(
            problem.states,
            vector,
        )
        sign = -1.0 if bound == "lower" else 1.0
        return _CvxpySolveResult(
            vector=vector,
            value=mean + sign * target.confidence_multiplier * standard_error,
            duals=self._constraint_duals(
                records,
                solve=f"confidence_core_{bound}",
            ),
        )

    def _solve_single(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> tuple[tuple[float, ...], float]:
        result = self._solve_single_result(
            problem,
            objective,
            maximize=maximize,
            public_law=public_law,
        )
        return result.vector, result.value

    def _solve_single_result(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> _CvxpySolveResult:
        import numpy as np

        effective_public_law = self._effective_public_law(problem, public_law)
        compiled = self._single_problem(
            problem,
            use_public_law=effective_public_law is not None,
        )
        objective_vector = np.array(objective, dtype=float)
        compiled.objective.value = objective_vector if maximize else -objective_vector
        if compiled.public_law is not None:
            compiled.public_law.value = self._public_law_array(
                problem, effective_public_law
            )
        self._set_extra_parameter_values(compiled.extra_parameters)
        self._solve_problem(compiled.problem)
        vector = self._clean_vector(problem, compiled.q.value)
        solve = "upper" if maximize else "lower"
        return _CvxpySolveResult(
            vector=vector,
            value=float(np.dot(objective_vector, vector)),
            duals=self._constraint_duals(compiled.records, solve=solve),
        )

    def _solve_pair(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        result = self._solve_pair_result(problem, objective, support=support)
        return result.q1, result.q2, result.value

    def _solve_pair_result(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> _CvxpyPairSolveResult:
        import numpy as np

        fixed_public_law = self._fixed_public_law(problem)
        compiled = self._pair_problem(
            problem,
            support=support,
            use_public_law=fixed_public_law is not None,
        )
        objective_vector = np.array(objective, dtype=float)
        compiled.objective.value = objective_vector
        if compiled.public_law is not None:
            compiled.public_law.value = self._public_law_array(
                problem, fixed_public_law
            )
        self._set_extra_parameter_values(compiled.extra_parameters)
        self._solve_problem(compiled.problem)
        q1_vector = self._clean_vector(problem, compiled.q1.value)
        q2_vector = self._clean_vector(problem, compiled.q2.value)
        value = float(
            np.dot(objective_vector, np.array(q1_vector) - np.array(q2_vector))
        )
        return _CvxpyPairSolveResult(
            q1=q1_vector,
            q2=q2_vector,
            value=value,
            duals=self._constraint_duals(compiled.records, solve="gap"),
        )

    def _single_problem(
        self,
        problem,
        *,
        use_public_law: bool,
    ) -> _ParameterizedCvxpySingleProblem:
        key = (
            id(problem),
            tuple(problem.states),
            tuple(problem.public_values),
            use_public_law,
        )
        cached = self._single_cache.get(key)
        if cached is not None:
            return cached

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        objective = cp.Parameter(len(problem.states))
        public_law = (
            cp.Parameter(len(problem.public_values)) if use_public_law else None
        )
        extra_parameters: dict[str, Any] = {}
        records = self._parameterized_labeled_constraints_for_variable(
            cp,
            problem,
            q,
            variable="q",
            public_law=public_law,
            extra_parameters=extra_parameters,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective, q))),
            [record.constraint for record in records],
        )
        compiled = _ParameterizedCvxpySingleProblem(
            problem=cvx_problem,
            q=q,
            objective=objective,
            public_law=public_law,
            extra_parameters=extra_parameters,
            records=tuple(records),
        )
        self._single_cache[key] = compiled
        return compiled

    def _pair_problem(
        self,
        problem,
        *,
        support: Partition,
        use_public_law: bool,
    ) -> _ParameterizedCvxpyPairProblem:
        key = (
            id(problem),
            tuple(problem.states),
            tuple(problem.public_values),
            hash(support),
            use_public_law,
        )
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached

        cp = self._cvxpy()
        n = len(problem.states)
        q1 = cp.Variable(n)
        q2 = cp.Variable(n)
        objective = cp.Parameter(n)
        public_law = (
            cp.Parameter(len(problem.public_values)) if use_public_law else None
        )
        extra_parameters: dict[str, Any] = {}
        records = [
            *self._parameterized_labeled_constraints_for_variable(
                cp,
                problem,
                q1,
                variable="q1",
                public_law=public_law,
                extra_parameters=extra_parameters,
            ),
            *self._parameterized_labeled_constraints_for_variable(
                cp,
                problem,
                q2,
                variable="q2",
                public_law=public_law,
                extra_parameters=extra_parameters,
            ),
        ]
        state_index = {state: i for i, state in enumerate(problem.states)}
        for i, block in enumerate(support.blocks):
            indices = [state_index[state] for state in block]
            records.append(
                cvxpy_constraint(
                    cp.sum(q1[indices]) == cp.sum(q2[indices]),
                    name=f"support-law equality block {i}",
                    kind="support_equality",
                    sense="==",
                    states=block,
                )
            )

        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective, q1 - q2))),
            [record.constraint for record in records],
        )
        compiled = _ParameterizedCvxpyPairProblem(
            problem=cvx_problem,
            q1=q1,
            q2=q2,
            objective=objective,
            public_law=public_law,
            extra_parameters=extra_parameters,
            records=tuple(records),
        )
        self._pair_cache[key] = compiled
        return compiled

    def _parameterized_constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        public_law,
        extra_parameters: dict[str, Any],
    ) -> list[Any]:
        return [
            record.constraint
            for record in self._parameterized_labeled_constraints_for_variable(
                cp,
                problem,
                q,
                variable="q",
                public_law=public_law,
                extra_parameters=extra_parameters,
            )
        ]

    def _parameterized_labeled_constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        variable: str,
        public_law,
        extra_parameters: dict[str, Any],
    ) -> list[CvxpyConstraintMetadata]:
        records = [
            cvxpy_constraint(
                q >= 0,
                name="state probability lower bound",
                kind="lower_bound",
                sense=">=",
                variable=variable,
                states=problem.states,
            ),
            cvxpy_constraint(
                cp.sum(q) == 1,
                name="probability normalization",
                kind="normalization",
                sense="==",
                variable=variable,
            ),
        ]
        records.extend(
            self._linear_constraint_records(cp, problem, q, variable=variable)
        )
        if public_law is not None:
            records.extend(
                self._public_law_parameter_constraint_records(
                    cp,
                    problem,
                    q,
                    public_law,
                    variable=variable,
                )
            )

        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        for i, builder in enumerate(self.constraint_builders):
            built = builder(cp, q, states, state_index)
            records.extend(
                self._coerce_constraint_records(
                    built,
                    default_name=f"custom constraint builder {i}",
                    default_kind="custom",
                    variable=variable,
                )
            )

        parameter = self._parameter_factory(cp, extra_parameters)
        for i, builder in enumerate(self.parameterized_constraint_builders):
            built = builder(cp, q, states, state_index, parameter)
            records.extend(
                self._coerce_constraint_records(
                    built,
                    default_name=f"parameterized constraint builder {i}",
                    default_kind="parameterized",
                    variable=variable,
                )
            )
        return records

    def _public_law_parameter_constraints(
        self,
        cp,
        problem,
        q,
        public_law,
    ) -> list[Any]:
        return [
            record.constraint
            for record in self._public_law_parameter_constraint_records(
                cp,
                problem,
                q,
                public_law,
                variable="q",
            )
        ]

    def _public_law_parameter_constraint_records(
        self,
        cp,
        problem,
        q,
        public_law,
        *,
        variable: str,
    ) -> list[CvxpyConstraintMetadata]:
        state_index = {state: i for i, state in enumerate(problem.states)}
        constraints = []
        for i, public_value in enumerate(problem.public_values):
            indices = [
                state_index[state] for state in problem.public_fibers[public_value]
            ]
            constraints.append(
                cvxpy_constraint(
                    cp.sum(q[indices]) == public_law[i],
                    name=f"public-law equality {public_value!r}",
                    kind="public_law",
                    sense="==",
                    variable=variable,
                    public_value=public_value,
                )
            )
        return constraints

    def _parameter_factory(
        self,
        cp,
        parameters: dict[str, Any],
    ) -> CvxpyParameterFactory:
        def parameter(name: str, **kwargs: Any):
            if name not in parameters:
                parameters[name] = cp.Parameter(**kwargs)
            return parameters[name]

        return parameter

    def _set_extra_parameter_values(self, parameters: Mapping[str, Any]) -> None:
        for name, parameter in parameters.items():
            if name not in self.parameter_values:
                raise CvxpyError(f"missing CVXPY parameter value: {name!r}")
            parameter.value = self.parameter_values[name]

    def _effective_public_law(
        self,
        problem,
        public_law: Mapping[Hashable, float] | None,
    ) -> dict[Hashable, float] | None:
        fixed_public_law = self._fixed_public_law(problem)
        if fixed_public_law is not None:
            if public_law is not None and not self._same_public_law(
                problem, fixed_public_law, public_law
            ):
                raise ValueError("requested public law conflicts with fixed_public_law")
            return fixed_public_law
        if public_law is None:
            return None
        return problem._coerce_public_law(public_law)

    def _public_law_array(self, problem, public_law):
        import numpy as np

        if public_law is None:
            raise CvxpyError("internal error: public_law parameter has no value")
        return np.array(
            [public_law[value] for value in problem.public_values],
            dtype=float,
        )
