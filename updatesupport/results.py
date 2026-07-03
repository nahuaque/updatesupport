"""Structured result objects returned by the public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping

from .partition import Partition


Distribution = Mapping[Hashable, float]


@dataclass(frozen=True)
class Witness:
    """Two admissible environments that the observed support cannot distinguish."""

    q1: Distribution
    q2: Distribution
    psi_q1: float
    psi_q2: float
    gap: float
    public_law: Mapping[Hashable, float]
    support_law: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AdequacyResult:
    adequate: bool
    support: Partition
    gap: float = 0.0
    witness: Witness | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LeastSupportResult:
    exists: bool
    support: Partition | None
    minimal_supports: tuple[Partition, ...]
    common_coarsening: Partition | None = None
    failure_witness: Witness | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ConstraintDual:
    """Dual multiplier diagnostic for one solved optimization constraint."""

    solve: str
    name: str
    kind: str
    magnitude: float
    signed_value: float | None = None
    sense: str | None = None
    variable: str | None = None
    state: Hashable | None = None
    public_value: Hashable | None = None
    index: tuple[int, ...] | None = None
    residual: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "solve": self.solve,
            "name": self.name,
            "kind": self.kind,
            "magnitude": self.magnitude,
            "signed_value": self.signed_value,
            "sense": self.sense,
            "variable": self.variable,
            "state": self.state,
            "public_value": self.public_value,
            "index": self.index,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class TransportResult:
    lower: float
    upper: float
    diameter: float
    public_law: Mapping[Hashable, float] | None = None
    q_lower: Distribution | None = None
    q_upper: Distribution | None = None
    duals: tuple[ConstraintDual, ...] = ()
    bound_type: str = "exact"
    lower_bound_type: str = "exact"
    upper_bound_type: str = "exact"
    notes: tuple[str, ...] = ()

    def dual_summary(
        self, *, top: int | None = 10, min_magnitude: float = 0.0
    ) -> tuple[ConstraintDual, ...]:
        rows = [row for row in self.duals if row.magnitude >= min_magnitude]
        rows.sort(key=lambda row: row.magnitude, reverse=True)
        return tuple(rows if top is None else rows[:top])


@dataclass(frozen=True)
class UncertainLinearConfidenceCoreResult:
    """Common confidence-core interval for an uncertain linear target.

    The lower endpoint is the maximum lower confidence bound over admissible
    hidden compositions. The upper endpoint is the minimum upper confidence
    bound. If ``lower > upper``, the composition-specific confidence bands have
    no common overlap and ``empty_gap`` records the separation.
    """

    lower: float
    upper: float
    diameter: float
    empty_gap: float
    public_law: Mapping[Hashable, float]
    q_lower: Distribution
    q_upper: Distribution
    duals: tuple[ConstraintDual, ...] = ()
    method: str = "socp_confidence_core"

    @property
    def nonempty(self) -> bool:
        return self.empty_gap <= 0.0

    @property
    def empty(self) -> bool:
        return not self.nonempty

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lower": self.lower,
            "upper": self.upper,
            "diameter": self.diameter,
            "empty_gap": self.empty_gap,
            "nonempty": self.nonempty,
            "empty": self.empty,
            "public_law": dict(self.public_law),
            "q_lower": dict(self.q_lower),
            "q_upper": dict(self.q_upper),
            "duals": [row.as_dict() for row in self.duals],
        }


@dataclass(frozen=True)
class CardinalGapResult:
    max_gap_bits: float
    total_gap_bits: float
    blocks_by_public_value: Mapping[Hashable, int]
