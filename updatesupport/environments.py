"""Admissible environment classes for finite update-support problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence

from .partition import Partition
from .results import AdequacyResult, TransportResult, Witness


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
        worst_gap = 0.0
        worst_pair: tuple[Hashable, Hashable] | None = None
        for block in support.blocks:
            values = [(state, problem.estimand_map[state]) for state in block]
            for left_state, left_value in values:
                for right_state, right_value in values:
                    gap = abs(left_value - right_value)
                    if gap > worst_gap:
                        worst_gap = gap
                        worst_pair = (left_state, right_state)

        if worst_gap <= problem.tol:
            return AdequacyResult(adequate=True, support=support)
        if worst_pair is None:
            raise RuntimeError("inadequate support did not produce a witness pair")

        q1 = {state: 0.0 for state in problem.states}
        q2 = {state: 0.0 for state in problem.states}
        q1[worst_pair[0]] = 1.0
        q2[worst_pair[1]] = 1.0
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

    def local_transport(
        self, problem, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
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
    """Continuous environments q(t) = center + t * direction, |t| <= radius."""

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
        p = problem._coerce_public_law(public_law)
        h = self._estimand_vector(problem)

        q_lower, lower = self._solve_single(problem, h, maximize=False, public_law=p)
        q_upper, upper = self._solve_single(problem, h, maximize=True, public_law=p)
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=max(0.0, upper - lower),
            public_law=p,
            q_lower=problem._distribution_from_vector(q_lower),
            q_upper=problem._distribution_from_vector(q_upper),
        )

    def global_transport(self, problem) -> TransportResult:
        if self.fixed_public_law is not None:
            return self.local_transport(problem, self.fixed_public_law)

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

    def _solve_single(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> tuple[tuple[float, ...], float]:
        import numpy as np

        cp = self._cvxpy()
        q = cp.Variable(len(problem.states))
        objective_vector = np.array(objective, dtype=float)
        objective_expr = cp.sum(cp.multiply(objective_vector, q))
        cvx_problem = cp.Problem(
            cp.Maximize(objective_expr) if maximize else cp.Minimize(objective_expr),
            self._constraints_for_variable(cp, problem, q, public_law=public_law),
        )
        self._solve_problem(cvx_problem)
        vector = self._clean_vector(problem, q.value)
        return vector, float(np.dot(objective_vector, vector))

    def _solve_pair(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        import numpy as np

        cp = self._cvxpy()
        n = len(problem.states)
        q1 = cp.Variable(n)
        q2 = cp.Variable(n)
        objective_vector = np.array(objective, dtype=float)
        constraints = [
            *self._constraints_for_variable(cp, problem, q1),
            *self._constraints_for_variable(cp, problem, q2),
        ]
        state_index = {state: i for i, state in enumerate(problem.states)}
        for block in support.blocks:
            indices = [state_index[state] for state in block]
            constraints.append(cp.sum(q1[indices]) == cp.sum(q2[indices]))

        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective_vector, q1 - q2))),
            constraints,
        )
        self._solve_problem(cvx_problem)
        q1_vector = self._clean_vector(problem, q1.value)
        q2_vector = self._clean_vector(problem, q2.value)
        value = float(
            np.dot(objective_vector, np.array(q1_vector) - np.array(q2_vector))
        )
        return q1_vector, q2_vector, value

    def _max_support_disagreement(
        self, problem, support: Partition
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        q1, q2, value = self._solve_pair(
            problem, self._estimand_vector(problem), support=support
        )
        return q1, q2, max(0.0, abs(value))

    def _constraints_for_variable(
        self,
        cp,
        problem,
        q,
        *,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> list[Any]:
        constraints = [q >= 0, cp.sum(q) == 1]
        constraints.extend(self._linear_constraints(cp, problem, q))

        fixed_public_law = self._fixed_public_law(problem)
        if fixed_public_law is not None:
            if public_law is not None and not self._same_public_law(
                problem, fixed_public_law, public_law
            ):
                raise ValueError("requested public law conflicts with fixed_public_law")
            constraints.extend(
                self._public_law_constraints(cp, problem, q, fixed_public_law)
            )
        elif public_law is not None:
            constraints.extend(self._public_law_constraints(cp, problem, q, public_law))

        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        for builder in self.constraint_builders:
            constraints.extend(builder(cp, q, states, state_index))
        return constraints

    def _linear_constraints(self, cp, problem, q) -> list[Any]:
        constraints = []
        for constraint in self._normalized_constraints():
            vector = problem._coerce_vector(constraint.coefficients)
            expr = cp.sum(cp.multiply(tuple(float(value) for value in vector), q))
            if constraint.sense == "<=":
                constraints.append(expr <= float(constraint.rhs))
            elif constraint.sense == ">=":
                constraints.append(expr >= float(constraint.rhs))
            elif constraint.sense == "==":
                constraints.append(expr == float(constraint.rhs))
            else:
                raise ValueError(f"unsupported constraint sense: {constraint.sense!r}")
        return constraints

    def _public_law_constraints(
        self,
        cp,
        problem,
        q,
        public_law: Mapping[Hashable, float],
    ) -> list[Any]:
        state_index = {state: i for i, state in enumerate(problem.states)}
        constraints = []
        for public_value in problem.public_values:
            indices = [
                state_index[state] for state in problem.public_fibers[public_value]
            ]
            constraints.append(cp.sum(q[indices]) == float(public_law[public_value]))
        return constraints

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

    def _solve_problem(self, problem) -> None:
        options = dict(self.solver_options or {})
        try:
            if self.solver is None:
                problem.solve(**options)
            else:
                problem.solve(solver=self.solver, **options)
        except Exception as exc:  # pragma: no cover - solver exceptions vary by backend
            raise CvxpyError(str(exc)) from exc

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise CvxpyError(f"CVXPY problem status: {problem.status}")

    def _clean_vector(self, problem, value) -> tuple[float, ...]:
        if value is None:
            raise CvxpyError("CVXPY did not return a solution vector")
        vector = tuple(float(item) for item in value)
        if any(item < -1e-7 for item in vector):
            raise CvxpyError("CVXPY returned a negative probability")
        return tuple(
            0.0 if item < 0.0 or abs(item) <= problem.tol else item
            for item in vector
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
class _ParameterizedCvxpySingleProblem:
    problem: Any
    q: Any
    objective: Any
    public_law: Any | None
    extra_parameters: dict[str, Any]


@dataclass(frozen=True)
class _ParameterizedCvxpyPairProblem:
    problem: Any
    q1: Any
    q2: Any
    objective: Any
    public_law: Any | None
    extra_parameters: dict[str, Any]


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
    _single_cache: dict[
        tuple[Any, ...], _ParameterizedCvxpySingleProblem
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _pair_cache: dict[
        tuple[Any, ...], _ParameterizedCvxpyPairProblem
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

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

    def _solve_single(
        self,
        problem,
        objective: Sequence[float],
        *,
        maximize: bool,
        public_law: Mapping[Hashable, float] | None = None,
    ) -> tuple[tuple[float, ...], float]:
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
        return vector, float(np.dot(objective_vector, vector))

    def _solve_pair(
        self,
        problem,
        objective: Sequence[float],
        *,
        support: Partition,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float]:
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
            compiled.public_law.value = self._public_law_array(problem, fixed_public_law)
        self._set_extra_parameter_values(compiled.extra_parameters)
        self._solve_problem(compiled.problem)
        q1_vector = self._clean_vector(problem, compiled.q1.value)
        q2_vector = self._clean_vector(problem, compiled.q2.value)
        value = float(
            np.dot(objective_vector, np.array(q1_vector) - np.array(q2_vector))
        )
        return q1_vector, q2_vector, value

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
        public_law = cp.Parameter(len(problem.public_values)) if use_public_law else None
        extra_parameters: dict[str, Any] = {}
        constraints = self._parameterized_constraints_for_variable(
            cp,
            problem,
            q,
            public_law=public_law,
            extra_parameters=extra_parameters,
        )
        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective, q))),
            constraints,
        )
        compiled = _ParameterizedCvxpySingleProblem(
            problem=cvx_problem,
            q=q,
            objective=objective,
            public_law=public_law,
            extra_parameters=extra_parameters,
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
        public_law = cp.Parameter(len(problem.public_values)) if use_public_law else None
        extra_parameters: dict[str, Any] = {}
        constraints = [
            *self._parameterized_constraints_for_variable(
                cp,
                problem,
                q1,
                public_law=public_law,
                extra_parameters=extra_parameters,
            ),
            *self._parameterized_constraints_for_variable(
                cp,
                problem,
                q2,
                public_law=public_law,
                extra_parameters=extra_parameters,
            ),
        ]
        state_index = {state: i for i, state in enumerate(problem.states)}
        for block in support.blocks:
            indices = [state_index[state] for state in block]
            constraints.append(cp.sum(q1[indices]) == cp.sum(q2[indices]))

        cvx_problem = cp.Problem(
            cp.Maximize(cp.sum(cp.multiply(objective, q1 - q2))),
            constraints,
        )
        compiled = _ParameterizedCvxpyPairProblem(
            problem=cvx_problem,
            q1=q1,
            q2=q2,
            objective=objective,
            public_law=public_law,
            extra_parameters=extra_parameters,
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
        constraints = [q >= 0, cp.sum(q) == 1]
        constraints.extend(self._linear_constraints(cp, problem, q))
        if public_law is not None:
            constraints.extend(
                self._public_law_parameter_constraints(cp, problem, q, public_law)
            )

        state_index = {state: i for i, state in enumerate(problem.states)}
        states = tuple(problem.states)
        for builder in self.constraint_builders:
            constraints.extend(builder(cp, q, states, state_index))

        parameter = self._parameter_factory(cp, extra_parameters)
        for builder in self.parameterized_constraint_builders:
            constraints.extend(builder(cp, q, states, state_index, parameter))
        return constraints

    def _public_law_parameter_constraints(
        self,
        cp,
        problem,
        q,
        public_law,
    ) -> list[Any]:
        state_index = {state: i for i, state in enumerate(problem.states)}
        constraints = []
        for i, public_value in enumerate(problem.public_values):
            indices = [
                state_index[state] for state in problem.public_fibers[public_value]
            ]
            constraints.append(cp.sum(q[indices]) == public_law[i])
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
