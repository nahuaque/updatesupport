"""Financial row metrics for updatesupport."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import updatesupport as us


def expected_loss(
    *,
    pd: str,
    lgd: str,
    name: str = "expected_loss_rate",
) -> us.RowMetric:
    """Return an expected-loss-rate metric ``PD * LGD``.

    Pass an exposure column such as EAD as ``weight``/``exposure`` when compiling
    a portfolio to obtain an exposure-weighted expected loss rate.
    """

    return us.row_metric(
        name,
        lambda row: _number(row, pd) * _number(row, lgd),
        columns=(pd, lgd),
        description="expected loss rate",
    )


def expected_loss_amount(
    *,
    pd: str,
    lgd: str,
    ead: str,
    name: str = "expected_loss_amount",
) -> us.RowMetric:
    """Return a dollar/exposure amount metric ``PD * LGD * EAD``."""

    return us.row_metric(
        name,
        lambda row: _number(row, pd) * _number(row, lgd) * _number(row, ead),
        columns=(pd, lgd, ead),
        description="expected loss amount",
    )


def default_rate(
    *,
    default: str,
    name: str = "default_rate",
) -> us.RowMetric:
    """Return a binary default-rate metric from a default indicator column."""

    return us.row_metric(
        name,
        lambda row: 1.0 if _number(row, default) > 0 else 0.0,
        columns=(default,),
        description="default rate",
    )


def loss_given_default(
    *,
    loss: str,
    exposure: str,
    name: str = "loss_given_default",
) -> us.RowMetric:
    """Return an LGD metric ``loss / exposure``."""

    def evaluate(row: Mapping[str, Any]) -> float:
        exposure_value = _number(row, exposure)
        if exposure_value <= 0:
            raise ValueError("LGD exposure must be positive")
        return _number(row, loss) / exposure_value

    return us.row_metric(
        name,
        evaluate,
        columns=(loss, exposure),
        description="loss given default",
    )


def _number(row: Mapping[str, Any], column: str) -> float:
    try:
        value = float(row[column])
    except KeyError as exc:
        raise ValueError(f"row is missing required finance column {column!r}") from exc
    if not isfinite(value):
        raise ValueError(f"finance column {column!r} must be finite")
    return value
