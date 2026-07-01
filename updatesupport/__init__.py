"""Finite causal support adequacy and transport ambiguity in Python."""

from .environments import (
    FiniteEnvironments,
    LinearConstraint,
    LineSegment,
    LPError,
    PolytopeEnvironments,
    PublicFiberSaturated,
    eq,
    geq,
    leq,
    linear_constraint,
)
from .partition import Partition, PartitionError
from .problem import FiniteProblem, TooManyPartitions
from .results import (
    AdequacyResult,
    CardinalGapResult,
    LeastSupportResult,
    TransportResult,
    Witness,
)

__all__ = [
    "AdequacyResult",
    "CardinalGapResult",
    "eq",
    "FiniteEnvironments",
    "FiniteProblem",
    "geq",
    "leq",
    "LeastSupportResult",
    "LinearConstraint",
    "LineSegment",
    "linear_constraint",
    "LPError",
    "Partition",
    "PartitionError",
    "PolytopeEnvironments",
    "PublicFiberSaturated",
    "TooManyPartitions",
    "TransportResult",
    "Witness",
]
