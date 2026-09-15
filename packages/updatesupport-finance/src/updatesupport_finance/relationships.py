"""Small, explicit financial relationships compiled to named linear constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import updatesupport as us

from .disclosure import disclosure_constraint, interval_disclosure_constraint


def rounded_amount_constraint(
    name: str,
    expression: us.NamedLinearExpression | Mapping[str, float] | str,
    value: float,
    *,
    increment: float,
    provenance: str | None = None,
) -> us.NamedLinearConstraint:
    """Inclusive relaxation of rounding to the nearest positive increment.

    Value, increment, and expression must use the same units. The caller chooses
    the rounding convention; this does not infer one from displayed digits.
    """
    increment = float(increment)
    if not isfinite(increment) or increment <= 0:
        raise ValueError("increment must be finite and positive")
    return interval_disclosure_constraint(
        name,
        expression,
        lower=value - increment / 2,
        upper=value + increment / 2,
        category="rounded_amount",
        provenance=provenance,
        metadata={
            "value": value,
            "increment": increment,
            "rounding": "nearest_inclusive",
        },
    )


def reconciliation_constraint(
    name: str,
    *,
    total: str,
    components: Sequence[str] | Mapping[str, float],
    residual: str | None = None,
    provenance: str | None = None,
    description: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> us.NamedLinearConstraint:
    """Declare total = signed components + optional residual.

    Components form an exhaustive reconciliation only because the caller
    declares it. For a signed residual, create its variable with ``lower=None``;
    finance variables otherwise default to nonnegative.
    """
    if isinstance(components, str):
        raise ValueError(
            "components must be a sequence of names or a coefficient mapping"
        )
    terms = (
        dict(components)
        if isinstance(components, Mapping)
        else dict.fromkeys(components, 1.0)
    )
    if not terms or len(terms) != len(components):
        raise ValueError("components must be nonempty and unique")
    if (
        total in terms
        or residual == total
        or (residual is not None and residual in terms)
    ):
        raise ValueError("total, components, and residual must have distinct names")
    if residual is not None:
        terms[residual] = 1.0
    return disclosure_constraint(
        name,
        {total: 1.0, **{key: -value for key, value in terms.items()}},
        lower=0.0,
        upper=0.0,
        category="reconciliation",
        provenance=provenance,
        description=description,
        metadata=metadata,
    )


def stock_flow_constraint(
    name: str,
    *,
    opening: str,
    closing: str,
    movements: Mapping[str, float],
    residual: str | None = None,
    provenance: str | None = None,
) -> us.NamedLinearConstraint:
    """Declare closing = opening + signed movements + optional residual."""
    if opening in movements:
        raise ValueError("opening balance cannot also be a movement")
    return reconciliation_constraint(
        name,
        total=closing,
        components={opening: 1.0, **movements},
        residual=residual,
        provenance=provenance,
        description="Closing balance = opening balance + signed movements + residual.",
    )
