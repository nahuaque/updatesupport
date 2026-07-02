"""Metric abstractions for plugin-provided target values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class RowMetric:
    """A row-level metric that can be used as a ``from_dataframe`` target.

    Domain extensions can return ``RowMetric`` objects to keep their
    domain-specific target logic outside the core package while still compiling
    to the same finite update-support problem.
    """

    name: str
    func: Callable[[Mapping[str, Any]], float]
    columns: tuple[str, ...] = ()
    description: str = ""

    def evaluate(self, row: Mapping[str, Any]) -> float:
        value = float(self.func(row))
        if not isfinite(value):
            raise ValueError(f"metric {self.name!r} evaluated to a non-finite value")
        return value


def row_metric(
    name: str,
    func: Callable[[Mapping[str, Any]], float],
    *,
    columns: tuple[str, ...] | list[str] = (),
    description: str = "",
) -> RowMetric:
    """Create a reusable row-level metric for ``from_dataframe``."""

    return RowMetric(
        name=name,
        func=func,
        columns=tuple(columns),
        description=description,
    )


def target_name(target: str | RowMetric) -> str:
    """Return the display name for a column or row metric target."""

    if isinstance(target, RowMetric):
        return target.name
    return target


def target_description(target: str | RowMetric) -> str:
    """Return a human-readable description for a target."""

    if isinstance(target, RowMetric):
        return target.description or target.name
    return target


def evaluate_target(
    row: Mapping[str, Any],
    target: str | RowMetric,
    *,
    get_value: Callable[[Mapping[str, Any], str], Any],
) -> float:
    """Evaluate a column target or row metric against one record."""

    if isinstance(target, RowMetric):
        return target.evaluate(row)
    value = float(get_value(row, target))
    if not isfinite(value):
        raise ValueError(f"target {target!r} evaluated to a non-finite value")
    return value
