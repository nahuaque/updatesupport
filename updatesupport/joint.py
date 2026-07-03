"""Model-assisted joint public/hidden distribution utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence

from .data import TabularTarget, from_dataframe


_MODEL_WEIGHT_COLUMN = "__updatesupport_joint_weight__"
_MODEL_TARGET_COLUMN = "__updatesupport_joint_target__"


@dataclass(frozen=True)
class JointCell:
    """One retained hidden cell in a fitted nonparametric joint distribution."""

    hidden_cell: tuple[Hashable, ...]
    public_value: tuple[Hashable, ...]
    probability: float
    total_weight: float
    target_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_cell": self.hidden_cell,
            "public_value": self.public_value,
            "probability": self.probability,
            "total_weight": self.total_weight,
            "target_value": self.target_value,
        }


@dataclass(frozen=True)
class JointDistributionDraw:
    """One model-assisted draw of hidden-cell masses."""

    draw_index: int
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    cells: tuple[JointCell, ...]
    probabilities: tuple[float, ...]
    total_weight: float
    weight_column: str = _MODEL_WEIGHT_COLUMN
    target_column: str = _MODEL_TARGET_COLUMN

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return weighted hidden-cell records consumable by report helpers."""

        records = []
        for cell, probability in zip(self.cells, self.probabilities, strict=True):
            row = {
                column: value
                for column, value in zip(
                    self.hidden_columns,
                    cell.hidden_cell,
                    strict=True,
                )
            }
            row[self.weight_column] = probability * self.total_weight
            row[self.target_column] = cell.target_value
            records.append(row)
        return tuple(records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "probabilities": self.probabilities,
            "total_weight": self.total_weight,
            "weight_column": self.weight_column,
            "target_column": self.target_column,
        }


@dataclass(frozen=True)
class NonparametricJointDistribution:
    """Fitted empirical public/hidden cell law with bootstrap draw support."""

    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target_name: str
    cells: tuple[JointCell, ...]
    total_weight: float
    rows_seen: int
    method: str = "bayesian_bootstrap"
    effective_sample_size: float | None = None
    smoothing: float = 1e-9

    def __post_init__(self) -> None:
        method = self.method.strip().lower().replace("-", "_")
        if method not in {"bayesian_bootstrap", "empirical"}:
            raise ValueError("method must be 'bayesian_bootstrap' or 'empirical'")
        object.__setattr__(self, "method", method)
        if not self.cells:
            raise ValueError("cells must contain at least one joint cell")
        if self.total_weight <= 0:
            raise ValueError("total_weight must be positive")
        if self.rows_seen <= 0:
            raise ValueError("rows_seen must be positive")
        if self.effective_sample_size is not None and self.effective_sample_size <= 0:
            raise ValueError("effective_sample_size must be positive")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def draw(
        self,
        *,
        draw_index: int = 1,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> JointDistributionDraw:
        """Draw one weighted hidden-cell composition."""

        rng = random.Random(seed)  # nosec B311
        return self._draw_with_rng(
            rng,
            draw_index=draw_index,
            weight_column=weight_column,
            target_column=target_column,
        )

    def iter_draws(
        self,
        count: int,
        *,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> tuple[JointDistributionDraw, ...]:
        """Return ``count`` independent model-assisted draws."""

        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)  # nosec B311
        return tuple(
            self._draw_with_rng(
                rng,
                draw_index=index,
                weight_column=weight_column,
                target_column=target_column,
            )
            for index in range(1, count + 1)
        )

    def draw_records(
        self,
        *,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> tuple[dict[str, Any], ...]:
        """Return one draw as weighted cell records."""

        return self.draw(
            seed=seed,
            weight_column=weight_column,
            target_column=target_column,
        ).records()

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target_name": self.target_name,
            "cell_count": self.cell_count,
            "total_weight": self.total_weight,
            "rows_seen": self.rows_seen,
            "method": self.method,
            "effective_sample_size": self.effective_sample_size,
            "smoothing": self.smoothing,
            "cells": [cell.as_dict() for cell in self.cells],
        }

    def _draw_with_rng(
        self,
        rng: random.Random,
        *,
        draw_index: int,
        weight_column: str,
        target_column: str,
    ) -> JointDistributionDraw:
        if self.method == "empirical":
            probabilities = tuple(cell.probability for cell in self.cells)
        else:
            probabilities = self._bayesian_bootstrap_probabilities(rng)
        return JointDistributionDraw(
            draw_index=draw_index,
            public_columns=self.public_columns,
            hidden_columns=self.hidden_columns,
            cells=self.cells,
            probabilities=probabilities,
            total_weight=self.total_weight,
            weight_column=weight_column,
            target_column=target_column,
        )

    def _bayesian_bootstrap_probabilities(
        self,
        rng: random.Random,
    ) -> tuple[float, ...]:
        effective_n = (
            float(self.rows_seen)
            if self.effective_sample_size is None
            else float(self.effective_sample_size)
        )
        gammas = [
            rng.gammavariate(
                max(cell.probability * effective_n, self.smoothing),
                1.0,
            )
            for cell in self.cells
        ]
        total = sum(gammas)
        if total <= 0:
            return tuple(cell.probability for cell in self.cells)
        return tuple(value / total for value in gammas)


def fit_joint_distribution(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    weight: str | None = None,
    method: str = "bayesian_bootstrap",
    min_cell_weight: float = 1.0,
    effective_sample_size: float | None = None,
    smoothing: float = 1e-9,
) -> NonparametricJointDistribution:
    """Fit a nonparametric joint law over retained public/hidden cells."""

    grouped = from_dataframe(
        data,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        min_cell_weight=min_cell_weight,
        q="observed",
    )
    cells = tuple(
        JointCell(
            hidden_cell=state,
            public_value=grouped.problem.public_map[state],
            probability=float(grouped.cell_weights[state]),
            total_weight=float(grouped.cell_weights[state] * grouped.total_weight),
            target_value=float(grouped.problem.estimand_map[state]),
        )
        for state in grouped.problem.states
    )
    rows_seen = (
        1 if grouped.diagnostics is None else max(1, int(grouped.diagnostics.rows_seen))
    )
    return NonparametricJointDistribution(
        public_columns=tuple(public),
        hidden_columns=tuple(hidden),
        target_name=_target_label(target),
        cells=cells,
        total_weight=float(grouped.total_weight),
        rows_seen=rows_seen,
        method=method,
        effective_sample_size=effective_sample_size,
        smoothing=smoothing,
    )


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return str(getattr(target, "name", type(target).__name__))


def joint_draw_records(
    draw: JointDistributionDraw | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return records from a draw or pass through an existing record sequence."""

    if isinstance(draw, JointDistributionDraw):
        return draw.records()
    return tuple(draw)


__all__ = [
    "JointCell",
    "JointDistributionDraw",
    "NonparametricJointDistribution",
    "fit_joint_distribution",
    "joint_draw_records",
]
