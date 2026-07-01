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
from .presets import QPreset, q_bounded_shift, q_observed, q_saturated
from .report import (
    PublicDescentReport,
    PublicFiberDiagnostic,
    RefinementCandidate,
    SensitivityReport,
    SensitivityRow,
    public_descent_report,
    public_fiber_diagnostics,
    recommend_refinements,
    sensitivity_report,
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
    "QPreset",
    "q_bounded_shift",
    "q_observed",
    "q_saturated",
    "recommend_refinements",
    "RefinementCandidate",
    "SensitivityReport",
    "SensitivityRow",
    "sensitivity_report",
    "TooManyPartitions",
    "TransportResult",
    "Witness",
]
