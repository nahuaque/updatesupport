"""Rounded percentage evidence and linear share-threshold claims."""

from dataclasses import dataclass
from math import isfinite

import updatesupport as us
from updatesupport.linear_feasibility import (
    coerce_named_linear_expression as _coerce_expression,
)

from .evidence import DisclosureFact


def _difference(numerator, denominator, weight):
    numerator, denominator = map(_coerce_expression, (numerator, denominator))
    coefficients = dict(numerator.coefficients)
    for name, coefficient in denominator.coefficients.items():
        coefficients[name] = coefficients.get(name, 0) - weight * coefficient
    return us.NamedLinearExpression(
        {k: v for k, v in coefficients.items() if v},
        numerator.constant - weight * denominator.constant,
    )


def percentage_constraints(
    name,
    *,
    numerator,
    denominator,
    fact: DisclosureFact,
    bounds=None,
    exact=False,
    policy=None,
    role="reported_fact",
    denominator_nonnegative=False,
    varying_fields=(),
):
    """Encode a share as two inequalities; bounds, when supplied, are ratios.

    Nonnegativity of the denominator is a caller declaration, not inferred from
    its name. Precision comes from the source decimals or an explicit policy.
    Qualifiers such as 'approximately' never choose a policy automatically.
    ``varying_fields`` declares intentional stock/schedule context differences.
    """
    if denominator_nonnegative is not True:
        raise ValueError("percentage constraints require a nonnegative denominator")
    if fact.currency is not None or fact.unit not in {"ratio", "percent", "%"}:
        raise ValueError("percentage facts need ratio/percent units and no currency")
    if type(exact) is not bool:
        raise ValueError("exact must be an explicit boolean")
    if bounds is not None:
        if exact or not policy:
            raise ValueError(
                "explicit percentage bounds require a policy and exact=False"
            )
        low, high = bounds
    else:
        if fact.decimals is None and not exact:
            raise ValueError(
                "percentage precision is unknown; specify bounds or exact=True"
            )
        divisor = 100 if fact.unit in {"percent", "%"} else 1
        radius = 0 if exact else 0.5 * 10 ** (-fact.decimals)
        if not 0 <= fact.base_value / divisor <= 1:
            raise ValueError("displayed share must lie between zero and one")
        low, high = (
            max(0, (fact.base_value - radius) / divisor),
            min(1, (fact.base_value + radius) / divisor),
        )
    if not all(isfinite(v) for v in (low, high)) or not 0 <= low <= high <= 1:
        raise ValueError("share bounds must lie between zero and one")
    if role not in {"reported_fact", "management_expectation", "analyst_policy"}:
        raise ValueError("invalid share evidence role")
    num, den = map(_coerce_expression, (numerator, denominator))
    metadata = {
        "fact_ids": [fact.fact_id],
        "varying_fields": list(varying_fields),
        "evidence_role": role,
        "share_bounds": [low, high],
        "precision_policy": policy or ("exact" if exact else "source decimals"),
        "numerator": num.as_dict(),
        "denominator": den.as_dict(),
        "denominator_nonnegative": True,
    }
    return (
        us.NamedLinearConstraint(
            name + ":lower",
            _difference(num, den, low),
            lower=0,
            kind="percentage_margin",
            provenance=fact.source_url,
            metadata=metadata,
        ),
        us.NamedLinearConstraint(
            name + ":upper",
            _difference(num, den, high),
            upper=0,
            kind="percentage_margin",
            provenance=fact.source_url,
            metadata=metadata,
        ),
    )


@dataclass(frozen=True)
class ShareThreshold:
    """A linear excess target together with its mandatory positive denominator."""

    target: us.NamedLinearTarget
    positivity: us.NamedLinearConstraint


def share_threshold_target(
    name,
    *,
    numerator,
    denominator,
    threshold,
    denominator_lower_bound,
    label=None,
    unit=None,
    denominator_fact=None,
):
    """N/D >= t becomes N-tD >= 0, with a strictly positive denominator.

    Include ``positivity`` in every audited tier. AllocationTable.problem does
    this automatically. Without a supporting amount fact the floor is labeled
    as an analyst policy. This is a threshold test, not a ratio optimizer.
    """
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("share threshold must lie between zero and one")
    if not isfinite(denominator_lower_bound) or denominator_lower_bound <= 0:
        raise ValueError("a strictly positive denominator floor is required")
    den = _coerce_expression(denominator)
    metadata = {"evidence_role": "analyst_policy", "share_domain": True}
    if denominator_fact is not None:
        if (
            denominator_fact.decimals is None
            or denominator_lower_bound
            > denominator_fact.base_value - 0.5 * 10 ** (-denominator_fact.decimals)
            or den.constant
            or len(den.coefficients) != 1
            or next(iter(den.coefficients.values())) != 1
        ):
            raise ValueError(
                "denominator floor must be supported by a direct, positive amount fact"
            )
        metadata.update(
            evidence_role="reported_fact", fact_ids=[denominator_fact.fact_id]
        )
    return ShareThreshold(
        us.NamedLinearTarget(
            name, _difference(numerator, den, threshold), label=label, unit=unit
        ),
        us.NamedLinearConstraint(
            name + ":positive_denominator",
            den,
            lower=denominator_lower_bound,
            kind="share_domain",
            metadata=metadata,
        ),
    )


@dataclass(frozen=True)
class AllocationShareMargin:
    name: str
    fact: DisclosureFact
    denominator: str
    rows: tuple[str, ...] | None = None
    columns: tuple[str, ...] | None = None
    bounds: tuple[float, float] | None = None
    exact: bool = False
    policy: str | None = None
    role: str = "reported_fact"

    def __post_init__(self):
        if not self.name or not self.denominator:
            raise ValueError("share name and amount-margin denominator are required")
        for axis in ("rows", "columns"):
            selector = getattr(self, axis)
            if isinstance(selector, str):
                raise TypeError("share selectors must be sequences")
            if selector is not None:
                object.__setattr__(self, axis, tuple(selector))
