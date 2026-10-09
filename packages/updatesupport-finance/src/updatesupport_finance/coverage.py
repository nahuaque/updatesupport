"""Portfolio denominators and fixed-weight bounds for missing observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Sequence

from updatesupport.artifacts import ReportArtifactMixin

from .evidence import _timestamp


@dataclass(frozen=True)
class PortfolioPosition:
    """One supplied long position; value is in the universe's currency.

    Security-to-issuer resolution is supplied by the caller. Multiple share
    classes can reference the same issuer without losing their separate values.
    """

    position_id: str
    value: float
    issuer_id: str | None = None
    security_type: str = "equity"
    label: str | None = None

    def __post_init__(self):
        if not self.position_id or not self.security_type:
            raise ValueError("position_id and security_type are required")
        value = float(self.value)
        if not isfinite(value) or value < 0:
            raise ValueError("position value must be finite and nonnegative")
        object.__setattr__(self, "value", value)

    def as_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class PortfolioUniverse:
    """Supplied universe and an explicit security-type eligibility policy.

    This first scope model is for long portfolios. It does not infer gross/net
    denominators, exchange rates, or whether a vendor supplied every position.
    """

    positions: Sequence[PortfolioPosition]
    as_of: str
    currency: str
    eligible_types: Sequence[str] = ("equity",)
    name: str = "Portfolio"

    def __post_init__(self):
        positions = tuple(self.positions)
        if len({p.position_id for p in positions}) != len(positions):
            raise ValueError("position IDs must be unique")
        if not positions or fsum(p.value for p in positions) <= 0:
            raise ValueError("universe must have positive total value")
        if not self.currency or not self.eligible_types:
            raise ValueError("currency and eligible_types are required")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "eligible_types", tuple(self.eligible_types))
        object.__setattr__(self, "as_of", _timestamp(self.as_of))

    def as_dict(self):
        return {
            **vars(self),
            "positions": [p.as_dict() for p in self.positions],
            "eligible_types": list(self.eligible_types),
        }


@dataclass(frozen=True)
class PortfolioCoverageEntry:
    position_id: str
    value: float
    status: str
    reason: str
    metric_value: float | None = None
    fact_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in {"covered", "excluded", "unmapped", "missing"}:
            raise ValueError("unknown coverage status")
        if not self.reason:
            raise ValueError("coverage entries require a reason")
        if not isfinite(self.value) or self.value < 0:
            raise ValueError("coverage value must be finite and nonnegative")
        if (self.status == "covered") != (self.metric_value is not None):
            raise ValueError("only covered entries have a metric value")
        if self.metric_value is not None and not isfinite(self.metric_value):
            raise ValueError("metric value must be finite")
        object.__setattr__(self, "fact_ids", tuple(self.fact_ids))

    def as_dict(self):
        return {**vars(self), "fact_ids": list(self.fact_ids)}


@dataclass(frozen=True)
class PortfolioCoverageReport(ReportArtifactMixin):
    universe: PortfolioUniverse
    entries: Sequence[PortfolioCoverageEntry]
    target_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        entries = tuple(self.entries)
        expected = {p.position_id: p for p in self.universe.positions}
        if len(entries) != len(expected) or {e.position_id for e in entries} != set(
            expected
        ):
            raise ValueError("coverage ledger must contain every position exactly once")
        for entry in entries:
            position = expected[entry.position_id]
            if entry.value != position.value:
                raise ValueError("coverage values must match the original universe")
            eligible = position.security_type in self.universe.eligible_types
            if eligible == (entry.status == "excluded"):
                raise ValueError("coverage exclusion must match the eligibility policy")
        if self.target_bounds is not None:
            lo, hi = (float(x) for x in self.target_bounds)
            if not isfinite(lo) or not isfinite(hi) or lo > hi:
                raise ValueError("target_bounds must be finite and ordered")
            if any(
                e.metric_value is not None and not lo <= e.metric_value <= hi
                for e in entries
            ):
                raise ValueError("observed metric lies outside target_bounds")
            object.__setattr__(self, "target_bounds", (lo, hi))
        object.__setattr__(self, "entries", entries)

    @property
    def supplied_value(self):
        return fsum(e.value for e in self.entries)

    @property
    def eligible_value(self):
        return fsum(e.value for e in self.entries if e.status != "excluded")

    @property
    def covered_value(self):
        return fsum(e.value for e in self.entries if e.status == "covered")

    @property
    def unknown_value(self):
        return fsum(
            e.value for e in self.entries if e.status in {"missing", "unmapped"}
        )

    @property
    def covered_mean(self):
        if self.covered_value == 0:
            return None
        return (
            fsum(
                e.value * e.metric_value for e in self.entries if e.status == "covered"
            )
            / self.covered_value
        )

    def lift_interval(self, interval: tuple[float, float] | None):
        """Lift covered-book bounds to eligible scope with unknown weights fixed.

        Missing values range independently over target_bounds. No conclusion
        about eligible scope is returned for an unbounded unknown component.
        """
        total = self.eligible_value
        if total == 0:
            return None
        if self.unknown_value > 0 and self.target_bounds is None:
            return None
        if self.covered_value > 0 and interval is None:
            return None
        lo, hi = interval if self.covered_value > 0 else (0.0, 0.0)
        if not isfinite(lo) or not isfinite(hi) or lo > hi:
            raise ValueError("covered interval must be finite and ordered")
        missing_lo, missing_hi = self.target_bounds or (0.0, 0.0)
        return (
            (self.covered_value * lo + self.unknown_value * missing_lo) / total,
            (self.covered_value * hi + self.unknown_value * missing_hi) / total,
        )

    @property
    def observed_bounds(self):
        value = self.covered_mean
        return self.lift_interval(None if value is None else (value, value))

    def as_dict(self):
        return {
            "universe": self.universe.as_dict(),
            "entries": [e.as_dict() for e in self.entries],
            "target_bounds": self.target_bounds,
            "supplied_value": self.supplied_value,
            "eligible_value": self.eligible_value,
            "covered_value": self.covered_value,
            "unknown_value": self.unknown_value,
            "covered_mean": self.covered_mean,
            "covered_share_supplied": self.covered_value / self.supplied_value,
            "covered_share_eligible": self.covered_value / self.eligible_value
            if self.eligible_value
            else None,
            "observed_bounds": self.observed_bounds,
            "assumption": "Unknown position weights are fixed; missing metric values range over target_bounds.",
        }

    def to_markdown(self):
        payload = self.as_dict()
        return (
            f"# {self.universe.name} Coverage\n\n"
            f"- Covered value: {self.covered_value:,.2f} {self.universe.currency}\n"
            f"- Eligible value: {self.eligible_value:,.2f} {self.universe.currency}\n"
            f"- Supplied value: {self.supplied_value:,.2f} {self.universe.currency}\n"
            f"- Unknown eligible value: {self.unknown_value:,.2f}\n"
            f"- Eligible observed bounds: {self.observed_bounds}\n\n{payload['assumption']}"
        )

    def to_tables(self):
        """Export the full ledger alongside its original universe totals."""
        payload = self.as_dict()
        return {
            "portfolio_coverage_summary": (
                {
                    **{
                        k: v
                        for k, v in payload.items()
                        if k not in {"universe", "entries"}
                    },
                    **{
                        k: v for k, v in payload["universe"].items() if k != "positions"
                    },
                },
            ),
            "portfolio_coverage_entries": tuple(e.as_dict() for e in self.entries),
        }
