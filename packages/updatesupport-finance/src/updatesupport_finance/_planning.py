"""Finite subset enumeration and tie-preserving frontier selection."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import combinations
from typing import TypeVar

_T = TypeVar("_T")


def iter_subsets(
    choices: Sequence[_T], *, max_size: int | None = None
) -> Iterator[tuple[_T, ...]]:
    """Enumerate in increasing size, retaining catalog order within each size."""
    maximum = len(choices) if max_size is None else max_size
    for size in range(maximum + 1):
        yield from combinations(choices, size)


def pareto_frontier(
    rows: Sequence[_T], dimensions: Callable[[_T], Sequence[float]]
) -> tuple[_T, ...]:
    """Retain all choices without a strictly dominating choice, including ties."""
    vectors = [tuple(dimensions(row)) for row in rows]
    return tuple(
        row
        for row, vector in zip(rows, vectors)
        if not any(
            all(a <= b + 1e-12 for a, b in zip(other_vector, vector))
            and any(a < b - 1e-12 for a, b in zip(other_vector, vector))
            for other, other_vector in zip(rows, vectors)
            if other is not row
        )
    )
