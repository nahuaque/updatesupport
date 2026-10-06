"""Portfolio mandate constructors using the core's shared linear Q constraints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

import updatesupport as us

from .compilation import CompiledPortfolio


@dataclass(frozen=True)
class PortfolioMandate:
    """Caps and reallocation restrictions on the normalized covered book.

    locked_issuers fixes issuer totals; preserve_columns fixes every category
    total in those columns (e.g. country or industry). These restrictions allow
    transfers within groups. They describe hypothetical compositions, not trade
    execution, liquidity, lots, or a whole-fund mandate with unknown positions.
    """

    issuer_caps: Mapping[str, float] = field(default_factory=dict)
    locked_issuers: Sequence[str] = ()
    preserve_columns: Sequence[str] = ()

    def __post_init__(self):
        caps = {k: float(v) for k, v in self.issuer_caps.items()}
        if any(not k or not isfinite(v) or not 0 <= v <= 1 for k, v in caps.items()):
            raise ValueError("issuer caps must be finite shares in [0, 1]")
        object.__setattr__(self, "issuer_caps", MappingProxyType(caps))
        object.__setattr__(self, "locked_issuers", tuple(self.locked_issuers))
        object.__setattr__(self, "preserve_columns", tuple(self.preserve_columns))

    def as_dict(self):
        return {
            "issuer_caps": dict(self.issuer_caps),
            "locked_issuers": list(self.locked_issuers),
            "preserve_columns": list(self.preserve_columns),
            "denominator": "covered_book",
        }


def q_portfolio_mandate(
    portfolio: CompiledPortfolio,
    *,
    hidden: Sequence[str],
    mandate: PortfolioMandate,
    base_q="saturated",
    solver: str | None = None,
) -> us.QPreset:
    """Compile issuer/group indicator moments; forward and inverse reuse this Q.

    The observed book must satisfy caps. Unknown issuers and restrictions absent
    from hidden columns raise errors instead of being averaged or ignored.
    Use respect_q=True for the core/finance minimum-breaking-witness API.
    """
    hidden = tuple(hidden)
    if "issuer_id" not in hidden or not set(mandate.preserve_columns).issubset(hidden):
        raise ValueError("hidden must contain issuer_id and every restricted column")
    if portfolio.coverage.covered_value <= 0:
        raise ValueError("mandates require a nonempty covered book")
    masses = defaultdict(float)
    for row in portfolio.rows:
        if row["weight"] > 0:
            masses[tuple(row[c] for c in hidden)] += (
                row["weight"] / portfolio.coverage.covered_value
            )
    issuer_index = hidden.index("issuer_id")
    issuers = {s[issuer_index] for s in masses}
    if (set(mandate.issuer_caps) | set(mandate.locked_issuers)) - issuers:
        raise ValueError("mandate references issuers absent from the covered book")
    moments, lower, upper = {}, {}, {}

    def indicator(name, column_index, value):
        moments[name] = {s: float(s[column_index] == value) for s in masses}
        return sum(masses[s] * moments[name][s] for s in masses)

    for issuer, cap in mandate.issuer_caps.items():
        name = f"issuer_cap:{issuer}"
        observed = indicator(name, issuer_index, issuer)
        if observed > cap + 1e-12:
            raise ValueError(f"observed issuer {issuer!r} exceeds its cap")
        upper[name] = cap
    for issuer in mandate.locked_issuers:
        name = f"locked_issuer:{issuer}"
        lower[name] = upper[name] = indicator(name, issuer_index, issuer)
    for column in mandate.preserve_columns:
        index = hidden.index(column)
        for value in sorted({s[index] for s in masses}, key=str):
            name = f"preserve:{column}:{value}"
            lower[name] = upper[name] = indicator(name, index, value)
    if not moments:
        return us.QSpec.from_value(base_q).to_preset()
    return us.q_intersection(
        base_q,
        us.q_moment_bounds(moments, lower=lower, upper=upper, solver=solver),
        solver=solver,
    )
