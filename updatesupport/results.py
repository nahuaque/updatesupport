"""Structured result objects returned by the public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

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
class TransportResult:
    lower: float
    upper: float
    diameter: float
    public_law: Mapping[Hashable, float] | None = None
    q_lower: Distribution | None = None
    q_upper: Distribution | None = None


@dataclass(frozen=True)
class CardinalGapResult:
    max_gap_bits: float
    total_gap_bits: float
    blocks_by_public_value: Mapping[Hashable, int]
