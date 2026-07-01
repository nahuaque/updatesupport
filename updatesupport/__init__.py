"""Representation adequacy and transport-stability auditing in Python."""

from .data import GroupedProblem, from_dataframe
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
from .report import (
    PublicDescentReport,
    PublicFiberDiagnostic,
    RefinementCandidate,
    public_descent_report,
    public_fiber_diagnostics,
    recommend_refinements,
)
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
    "from_dataframe",
    "geq",
    "GroupedProblem",
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
    "PublicDescentReport",
    "PublicFiberDiagnostic",
    "public_descent_report",
    "public_fiber_diagnostics",
    "recommend_refinements",
    "RefinementCandidate",
    "TooManyPartitions",
    "TransportResult",
    "Witness",
]
