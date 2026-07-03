"""User-facing finite update-support problem object."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Hashable, Mapping, Sequence

from .environments import Environment, PublicFiberSaturated
from .partition import (
    Partition,
    PartitionError,
    common_coarsening,
    partitions_refining_public,
)
from .results import (
    AdequacyResult,
    CardinalGapResult,
    LeastSupportResult,
    TransportResult,
    Witness,
)
from .targets import (
    LinearTarget,
    MomentTransformTarget,
    ProcedureTarget,
    RatioTarget,
    TargetContract,
    UnsupportedTargetError,
    coerce_target,
)


class TooManyPartitions(RuntimeError):
    """Raised when exhaustive support enumeration would be too large."""


StateMap = Mapping[Hashable, Hashable] | Callable[[Hashable], Hashable]
EstimandMap = (
    Mapping[Hashable, float]
    | Sequence[float]
    | Callable[[Hashable], float]
    | LinearTarget
    | MomentTransformTarget
    | ProcedureTarget
    | RatioTarget
)


@dataclass
class FiniteProblem:
    """Finite hidden-state update-support problem.

    Parameters mirror the paper's finite setup:

    - ``states`` is the hidden state space ``D``.
    - ``public`` is the projection ``pi: D -> O``.
    - ``estimand`` is a fixed target functional over hidden distributions.
    - ``environments`` is the admissible class ``Q``.
    """

    states: Sequence[Hashable]
    public: StateMap
    estimand: EstimandMap
    environments: Environment | None = None
    tol: float = 1e-9

    def __post_init__(self) -> None:
        self.states = tuple(self.states)
        if not self.states:
            raise ValueError("states must be non-empty")
        if len(set(self.states)) != len(self.states):
            raise ValueError("states must be unique")

        self.public_map = self._coerce_state_map(self.public, name="public")
        self.target_functional = coerce_target(self.states, self.estimand)
        self.estimand_map = dict(self.target_functional.values)
        self.public_values = tuple(
            dict.fromkeys(self.public_map[state] for state in self.states)
        )
        self.public_fibers = {
            public_value: tuple(
                state for state in self.states if self.public_map[state] == public_value
            )
            for public_value in self.public_values
        }
        if self.environments is None:
            self.environments = PublicFiberSaturated()

    def _coerce_state_map(
        self, value: StateMap, *, name: str
    ) -> dict[Hashable, Hashable]:
        if callable(value):
            return {state: value(state) for state in self.states}
        missing = [state for state in self.states if state not in value]
        if missing:
            raise ValueError(f"{name} is missing states: {missing!r}")
        return {state: value[state] for state in self.states}

    @property
    def target_contract(self) -> TargetContract:
        return self.target_functional.contract

    @property
    def has_linear_target(self) -> bool:
        if isinstance(self.target_functional, LinearTarget | MomentTransformTarget):
            return True
        return isinstance(
            self.target_functional, RatioTarget
        ) and self.target_functional.has_constant_denominator(self.states)

    @property
    def has_ratio_target(self) -> bool:
        return isinstance(self.target_functional, RatioTarget)

    def _require_linear_target(self, context: str) -> None:
        if self.has_linear_target:
            return
        contract = self.target_contract
        raise UnsupportedTargetError(
            f"{context} currently requires a fixed linear target. "
            f"Received {contract.kind!r} target {contract.name!r}. "
            "Use PublicFiberSaturated, FiniteEnvironments, or "
            "CvxpyEnvironments with a fixed public law for RatioTarget; "
            "otherwise compile the ratio to a LinearTarget when the denominator "
            "is known to be invariant."
        )

    def _coerce_vector(
        self, value: Mapping[Hashable, float] | Sequence[float]
    ) -> tuple[float, ...]:
        if isinstance(value, Mapping):
            extra = set(value) - set(self.states)
            if extra:
                raise ValueError(f"vector contains unknown states: {extra!r}")
            return tuple(float(value.get(state, 0.0)) for state in self.states)
        if len(value) != len(self.states):
            raise ValueError("vector sequence must have one value per state")
        return tuple(float(item) for item in value)

    def _coerce_distribution(
        self,
        value: Mapping[Hashable, float] | Sequence[float],
    ) -> tuple[float, ...]:
        vector = self._coerce_vector(value)
        if any(item < -self.tol for item in vector):
            raise ValueError("distributions must be non-negative")
        total = sum(vector)
        if abs(total - 1.0) > self.tol:
            raise ValueError(f"distribution must sum to one, got {total}")
        return tuple(0.0 if abs(item) <= self.tol else item for item in vector)

    def _coerce_public_law(
        self, value: Mapping[Hashable, float]
    ) -> dict[Hashable, float]:
        extra = set(value) - set(self.public_values)
        if extra:
            raise ValueError(f"public law contains unknown public values: {extra!r}")
        public_law = {
            public_value: float(value.get(public_value, 0.0))
            for public_value in self.public_values
        }
        if any(item < -self.tol for item in public_law.values()):
            raise ValueError("public laws must be non-negative")
        total = sum(public_law.values())
        if abs(total - 1.0) > self.tol:
            raise ValueError(f"public law must sum to one, got {total}")
        return {
            public_value: 0.0 if abs(mass) <= self.tol else mass
            for public_value, mass in public_law.items()
        }

    def _distribution_from_vector(
        self, vector: Sequence[float]
    ) -> dict[Hashable, float]:
        return {state: float(vector[i]) for i, state in enumerate(self.states)}

    def _same_vector(self, left: Sequence[float], right: Sequence[float]) -> bool:
        return len(left) == len(right) and all(
            abs(a - b) <= self.tol for a, b in zip(left, right, strict=True)
        )

    def _is_zero_vector(self, vector: Sequence[float]) -> bool:
        return all(abs(item) <= self.tol for item in vector)

    def _dot_estimand(self, vector: Sequence[float]) -> float:
        return self.target_functional.dot(self.states, vector)

    def _target_support_key(self, state: Hashable) -> Hashable:
        if (
            isinstance(self.target_functional, RatioTarget)
            and not self.has_linear_target
        ):
            return (
                self.target_functional.numerator_value(state),
                self.target_functional.denominator_value(state),
            )
        return self.target_functional.point_value(state)

    def _pushforward_vector(
        self, vector: Sequence[float], partition: Partition
    ) -> tuple[float, ...]:
        state_index = {state: i for i, state in enumerate(self.states)}
        return tuple(
            sum(vector[state_index[state]] for state in block)
            for block in partition.blocks
        )

    def _public_vector_from_distribution_vector(
        self, vector: Sequence[float]
    ) -> tuple[float, ...]:
        state_index = {state: i for i, state in enumerate(self.states)}
        return tuple(
            sum(
                vector[state_index[state]] for state in self.public_fibers[public_value]
            )
            for public_value in self.public_values
        )

    def _validate_support_over_public(self, support: Partition) -> None:
        if set(support.states) != set(self.states):
            raise PartitionError("support must partition this problem's states")
        if not support.is_support_over(self.public_map):
            raise PartitionError("support must refine the public projection")

    def _witness_from_vectors(
        self,
        q1: Sequence[float],
        q2: Sequence[float],
        support: Partition | None = None,
    ) -> Witness:
        psi_q1 = self._dot_estimand(q1)
        psi_q2 = self._dot_estimand(q2)
        return Witness(
            q1=self._distribution_from_vector(q1),
            q2=self._distribution_from_vector(q2),
            psi_q1=psi_q1,
            psi_q2=psi_q2,
            gap=abs(psi_q1 - psi_q2),
            public_law={
                public_value: mass
                for public_value, mass in zip(
                    self.public_values,
                    self._public_vector_from_distribution_vector(q1),
                    strict=True,
                )
            },
            support_law=self._pushforward_vector(q1, support)
            if support is not None
            else None,
        )

    def _interval_from_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        public_law: Mapping[Hashable, float] | None,
    ) -> TransportResult:
        if not vectors:
            raise ValueError("at least one vector is required")
        ordered = sorted(vectors, key=self._dot_estimand)
        lower_vector = ordered[0]
        upper_vector = ordered[-1]
        lower = self._dot_estimand(lower_vector)
        upper = self._dot_estimand(upper_vector)
        return TransportResult(
            lower=lower,
            upper=upper,
            diameter=upper - lower,
            public_law=dict(public_law) if public_law is not None else None,
            q_lower=self._distribution_from_vector(lower_vector),
            q_upper=self._distribution_from_vector(upper_vector),
        )

    def public_partition(self) -> Partition:
        return Partition.from_mapping(self.public_map, universe=self.states)

    def discrete_support(self) -> Partition:
        return Partition.discrete(self.states)

    def estimand_partition(self) -> Partition:
        """The saturated least support: quotient by joint values of (public, h)."""

        blocks: list[list[Hashable]] = []
        keys: list[tuple[Hashable, float]] = []
        for state in self.states:
            public_value = self.public_map[state]
            h_value = self._target_support_key(state)
            for i, (existing_public, existing_h) in enumerate(keys):
                if existing_public == public_value and _same_support_key(
                    existing_h, h_value, tol=self.tol
                ):
                    blocks[i].append(state)
                    break
            else:
                keys.append((public_value, h_value))
                blocks.append([state])
        return Partition.from_blocks(blocks, universe=self.states)

    def psi(self, distribution: Mapping[Hashable, float] | Sequence[float]) -> float:
        return self._dot_estimand(self._coerce_distribution(distribution))

    def public_law(
        self, distribution: Mapping[Hashable, float] | Sequence[float]
    ) -> dict[Hashable, float]:
        vector = self._coerce_distribution(distribution)
        return {
            public_value: mass
            for public_value, mass in zip(
                self.public_values,
                self._public_vector_from_distribution_vector(vector),
                strict=True,
            )
        }

    def support_law(
        self,
        distribution: Mapping[Hashable, float] | Sequence[float],
        support: Partition,
    ) -> tuple[float, ...]:
        self._validate_support_over_public(support)
        return self._pushforward_vector(
            self._coerce_distribution(distribution), support
        )

    def fiber_ranges(self) -> dict[Hashable, float]:
        ranges: dict[Hashable, float] = {}
        for public_value, fiber in self.public_fibers.items():
            values = [self.estimand_map[state] for state in fiber]
            ranges[public_value] = max(values) - min(values)
        return ranges

    def check_support(self, support: Partition) -> AdequacyResult:
        return self.environments.check_support(self, support)

    def check_public(self) -> AdequacyResult:
        return self.check_support(self.public_partition())

    def is_public_adequate(self) -> bool:
        return self.check_public().adequate

    def adequate_supports(self, *, max_states: int = 9) -> tuple[Partition, ...]:
        if len(self.states) > max_states:
            raise TooManyPartitions(
                f"support enumeration grows by Bell numbers; got {len(self.states)} states "
                f"with max_states={max_states}"
            )
        supports = partitions_refining_public(self.states, self.public_map)
        return tuple(
            support for support in supports if self.check_support(support).adequate
        )

    def minimal_supports(self, *, max_states: int = 9) -> tuple[Partition, ...]:
        adequate = self.adequate_supports(max_states=max_states)
        minimal = []
        for support in adequate:
            has_coarser_adequate = any(
                support.is_strictly_finer_than(candidate) for candidate in adequate
            )
            if not has_coarser_adequate:
                minimal.append(support)
        return tuple(minimal)

    def least_support(self, *, max_states: int = 9) -> LeastSupportResult:
        if isinstance(self.environments, PublicFiberSaturated):
            support = self.environments.least_support(self)
            return LeastSupportResult(
                exists=True,
                support=support,
                minimal_supports=(support,),
                common_coarsening=support,
                reason=(
                    "public-fiber saturation gives the quotient by joint values "
                    "of the public projection and target support key on public "
                    "fibers with admissible positive mass"
                ),
            )

        adequate = self.adequate_supports(max_states=max_states)
        minimal = self.minimal_supports(max_states=max_states)
        coarsening = common_coarsening(adequate, universe=self.states)
        result = self.check_support(coarsening)
        if result.adequate:
            return LeastSupportResult(
                exists=True,
                support=coarsening,
                minimal_supports=minimal,
                common_coarsening=coarsening,
            )
        return LeastSupportResult(
            exists=False,
            support=None,
            minimal_supports=minimal,
            common_coarsening=coarsening,
            failure_witness=result.witness,
            reason="common coarsening of adequate supports is not adequate",
        )

    def local_transport_modulus(
        self, public_law: Mapping[Hashable, float]
    ) -> TransportResult:
        return self.environments.local_transport(self, public_law)

    def global_transport_modulus(self) -> TransportResult:
        return self.environments.global_transport(self)

    def public_descent_gap(self) -> float:
        return self.global_transport_modulus().diameter

    def partial_identification_interval(
        self,
        public_law: Mapping[Hashable, float],
    ) -> TransportResult:
        return self.local_transport_modulus(public_law)

    def cardinal_gap(self) -> CardinalGapResult:
        least = self.least_support()
        if not least.exists or least.support is None:
            raise ValueError("cardinal gap requires an existing least support")

        blocks_by_public_value: dict[Hashable, int] = {}
        for public_value, fiber in self.public_fibers.items():
            count = 0
            fiber_set = set(fiber)
            for block in least.support.blocks:
                if set(block) <= fiber_set:
                    count += 1
            blocks_by_public_value[public_value] = count

        max_blocks = max(blocks_by_public_value.values())
        return CardinalGapResult(
            max_gap_bits=log2(max_blocks),
            total_gap_bits=sum(
                log2(count) for count in blocks_by_public_value.values()
            ),
            blocks_by_public_value=blocks_by_public_value,
        )

    def report(self) -> "ProblemReport":
        return ProblemReport(self)


@dataclass(frozen=True)
class ProblemReport:
    problem: FiniteProblem

    def to_markdown(self) -> str:
        problem = self.problem
        least = problem.least_support()
        transport = problem.global_transport_modulus()
        lines = [
            "# Update Support Report",
            "",
            f"- States: {len(problem.states)}",
            f"- Public fibers: {len(problem.public_values)}",
            f"- Environment class: {problem.environments.name}",
            f"- Public adequate: {'yes' if problem.is_public_adequate() else 'no'}",
            f"- Least adequate support exists: {'yes' if least.exists else 'no'}",
            f"- Global transport modulus: {transport.diameter:g}",
        ]
        if least.support is not None:
            lines.append(f"- Least support blocks: {len(least.support)}")
        ranges = problem.fiber_ranges()
        if ranges:
            lines.extend(["", "## Fiber Ranges"])
            for public_value, value in sorted(
                ranges.items(), key=lambda item: str(item[0])
            ):
                lines.append(f"- {public_value}: {value:g}")
        return "\n".join(lines)


def _same_support_key(left: Hashable, right: Hashable, *, tol: float) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(
            abs(float(left_item) - float(right_item)) <= tol
            for left_item, right_item in zip(left, right, strict=True)
        )
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return left == right
