"""Financial row metrics for updatesupport."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, sqrt
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


def expected_loss_standard_error(
    *,
    pd: str,
    lgd: str,
    pd_standard_error: str,
    lgd_standard_error: str,
    correlation: float = 0.0,
    name: str = "expected_loss_standard_error",
) -> us.RowMetric:
    """Return a delta-method standard error for ``PD * LGD``.

    Use this as ``metric_standard_error=...`` or ``target_standard_error=...``
    when a portfolio supplies row-level uncertainty for PD and LGD estimates.
    """

    rho = float(correlation)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must be between -1 and 1")

    def evaluate(row: Mapping[str, Any]) -> float:
        pd_value = _number(row, pd)
        lgd_value = _number(row, lgd)
        pd_se = _nonnegative_number(row, pd_standard_error)
        lgd_se = _nonnegative_number(row, lgd_standard_error)
        variance = (
            (lgd_value * pd_se) ** 2
            + (pd_value * lgd_se) ** 2
            + 2.0 * rho * pd_value * lgd_value * pd_se * lgd_se
        )
        return sqrt(max(0.0, variance))

    return us.row_metric(
        name,
        evaluate,
        columns=(pd, lgd, pd_standard_error, lgd_standard_error),
        description="expected loss rate standard error",
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


def _nonnegative_number(row: Mapping[str, Any], column: str) -> float:
    value = _number(row, column)
    if value < 0.0:
        raise ValueError(f"finance column {column!r} must be non-negative")
    return value
