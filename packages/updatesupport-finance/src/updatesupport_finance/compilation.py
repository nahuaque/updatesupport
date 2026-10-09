"""Provider-neutral, point-in-time holdings/fundamentals compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

from .coverage import (
    PortfolioCoverageEntry,
    PortfolioCoverageReport,
    PortfolioPosition,
    PortfolioUniverse,
)
from .evidence import DisclosureFact, _timestamp, validate_disclosure_evidence


@dataclass(frozen=True)
class FundamentalObservation:
    fact: DisclosureFact
    period_kind: str
    normalized_period_end: str | None = None

    def __post_init__(self):
        if not self.period_kind:
            raise ValueError(
                "period_kind must be explicitly supplied (e.g. FY, Q, TTM)"
            )
        if self.normalized_period_end is not None:
            date.fromisoformat(self.normalized_period_end)

    def as_dict(self):
        return {
            "fact": self.fact.as_dict(),
            "period_kind": self.period_kind,
            "normalized_period_end": self.normalized_period_end,
        }


@dataclass(frozen=True)
class TaxonomyAssignment:
    issuer_id: str
    labels: Mapping[str, str]
    scheme: str
    version: str | None = None
    available_at: str | None = None
    effective_at: str | None = None
    source_id: str | None = None

    def __post_init__(self):
        if not self.issuer_id or not self.scheme or not self.labels:
            raise ValueError("issuer, scheme, and taxonomy labels are required")
        if any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in self.labels.items()
        ):
            raise ValueError("taxonomy labels must be nonempty strings")
        reserved = {
            "position_id",
            "issuer_id",
            "weight",
            "metric",
            "fact_id",
            "target_status",
        }
        if reserved.intersection(self.labels):
            raise ValueError(
                "taxonomy labels cannot overwrite compiled evidence columns"
            )
        for key in ("available_at", "effective_at"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, _timestamp(value))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    def as_dict(self):
        return {**vars(self), "labels": dict(self.labels)}


@dataclass(frozen=True)
class PortfolioMetricPolicy:
    """Explicit fact context and scalar transform; no vendor concept guessing.

    ``below`` returns 1 when base_value < threshold, else 0. ``value`` returns
    base_value, so monetary values require a single declared currency.
    """

    concept: str
    unit: str
    period_kind: str = "FY"
    transform: str = "below"
    threshold: float = 0.0
    currency: str | None = None
    dimensions: Mapping[str, str] = field(default_factory=dict)
    target_bounds: tuple[float, float] | None = None
    max_age_days: int | None = None
    required_labels: Sequence[str] = ("sector", "industry")
    strict_taxonomy_as_of: bool = False

    def __post_init__(self):
        if not self.concept or not self.unit or not self.period_kind:
            raise ValueError("concept, unit, and period_kind are required")
        if self.transform not in {"below", "value"} or not isfinite(self.threshold):
            raise ValueError("transform must be below/value with a finite threshold")
        if self.transform == "value" and self.currency is None:
            raise ValueError("value metrics require an explicit currency context")
        if self.max_age_days is not None and (
            not isinstance(self.max_age_days, int) or self.max_age_days < 0
        ):
            raise ValueError("max_age_days must be a nonnegative integer")
        bounds = (0.0, 1.0) if self.transform == "below" else self.target_bounds
        if bounds is not None:
            lo, hi = map(float, bounds)
            if not isfinite(lo) or not isfinite(hi) or lo > hi:
                raise ValueError("target_bounds must be finite and ordered")
            bounds = (lo, hi)
        object.__setattr__(self, "target_bounds", bounds)
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))
        object.__setattr__(self, "required_labels", tuple(self.required_labels))

    def as_dict(self):
        return {
            **vars(self),
            "dimensions": dict(self.dimensions),
            "required_labels": list(self.required_labels),
        }


@dataclass(frozen=True)
class CompiledPortfolio:
    universe: PortfolioUniverse
    policy: PortfolioMetricPolicy
    observations: tuple[FundamentalObservation, ...]
    taxonomy: tuple[TaxonomyAssignment, ...]
    rows: tuple[Mapping, ...]
    coverage: PortfolioCoverageReport
    diagnostics: tuple[Mapping, ...]

    def __post_init__(self):
        if (
            self.coverage.universe != self.universe
            or self.coverage.target_bounds != self.policy.target_bounds
        ):
            raise ValueError(
                "compiled coverage must match its universe and metric policy"
            )
        covered = {
            e.position_id: e for e in self.coverage.entries if e.status == "covered"
        }
        rows = tuple(MappingProxyType(dict(r)) for r in self.rows)
        positions = {p.position_id: p for p in self.universe.positions}
        if len(rows) != len(covered) or {r["position_id"] for r in rows} != set(
            covered
        ):
            raise ValueError("compiled rows must match covered ledger positions")
        for row in rows:
            entry, position = covered[row["position_id"]], positions[row["position_id"]]
            if (
                row["metric"] != entry.metric_value
                or row["weight"] != entry.value
                or row["issuer_id"] != position.issuer_id
                or row["fact_id"] not in entry.fact_ids
            ):
                raise ValueError(
                    "compiled row values, issuer, and fact IDs must match the ledger"
                )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "taxonomy", tuple(self.taxonomy))
        object.__setattr__(
            self,
            "diagnostics",
            tuple(MappingProxyType(dict(x)) for x in self.diagnostics),
        )

    def as_dict(self):
        return {
            "universe": self.universe.as_dict(),
            "policy": self.policy.as_dict(),
            "observations": [x.as_dict() for x in self.observations],
            "taxonomy": [x.as_dict() for x in self.taxonomy],
            "rows": [dict(x) for x in self.rows],
            "coverage": self.coverage.as_dict(),
            "diagnostics": [dict(x) for x in self.diagnostics],
        }


def _restore_compiled_portfolio(data: Mapping) -> CompiledPortfolio:
    """Recompile portable inputs through the same point-in-time selection path.

    Snapshot owners compare the result with their stored evidence; neither
    stored rows nor a stored coverage ledger bypass compilation.
    """
    raw_universe = dict(data["universe"])
    raw_universe["positions"] = [
        PortfolioPosition(**p) for p in raw_universe["positions"]
    ]
    return compile_portfolio_evidence(
        PortfolioUniverse(**raw_universe),
        observations=[
            FundamentalObservation(
                DisclosureFact(**x["fact"]),
                x["period_kind"],
                x["normalized_period_end"],
            )
            for x in data["observations"]
        ],
        taxonomy=[TaxonomyAssignment(**t) for t in data["taxonomy"]],
        policy=PortfolioMetricPolicy(**data["policy"]),
    )


def compile_portfolio_evidence(
    universe: PortfolioUniverse,
    *,
    observations: Sequence[FundamentalObservation],
    taxonomy: Sequence[TaxonomyAssignment] = (),
    policy: PortfolioMetricPolicy,
) -> CompiledPortfolio:
    """Select eligible facts available by universe.as_of and account for every position.

    Select the latest fiscal end, then the latest available version. Ties across
    sources/contexts are quarantined rather than silently resolved. Original
    fiscal dates, normalized labels, and derivation inputs remain in evidence.
    Undated taxonomy is usable only under the explicit non-strict policy and
    produces a diagnostic; it is never claimed to be historical taxonomy.
    """
    observations, taxonomy = tuple(observations), tuple(taxonomy)
    all_facts = {x.fact.fact_id: x.fact for x in observations}
    if len(all_facts) != len(observations):
        raise ValueError("observation fact IDs must be unique")
    cutoff = _timestamp(universe.as_of)
    cutoff_date = datetime.fromisoformat(cutoff).date()
    diagnostics, rows, entries = [], [], []
    selection = {}
    for position in universe.positions:
        reason, status, metric, ids = "", "missing", None, ()
        if position.security_type not in universe.eligible_types:
            status, reason = (
                "excluded",
                f"security_type={position.security_type} is outside the eligibility policy",
            )
        elif not position.issuer_id:
            status, reason = "unmapped", "No stable issuer mapping supplied"
        else:
            issuer = position.issuer_id
            if issuer not in selection:
                candidates = [
                    x
                    for x in observations
                    if x.fact.entity == issuer
                    and x.fact.concept == policy.concept
                    and x.fact.unit == policy.unit
                    and (policy.currency is None or x.fact.currency == policy.currency)
                    and dict(x.fact.dimensions) == dict(policy.dimensions)
                    and x.period_kind == policy.period_kind
                    and _timestamp(x.fact.available_at) <= cutoff
                    and date.fromisoformat(x.fact.period_end) <= cutoff_date
                    and (
                        policy.max_age_days is None
                        or (cutoff_date - date.fromisoformat(x.fact.period_end)).days
                        <= policy.max_age_days
                    )
                ]
                if candidates:
                    best = max(
                        (x.fact.period_end, _timestamp(x.fact.available_at))
                        for x in candidates
                    )
                    candidates = [
                        x
                        for x in candidates
                        if (x.fact.period_end, _timestamp(x.fact.available_at)) == best
                    ]
                chosen = candidates[0] if len(candidates) == 1 else None
                failure = (
                    "Ambiguous latest fact context/version"
                    if len(candidates) > 1
                    else "No compatible fact available as of the holdings cutoff"
                )
                if chosen:
                    # Validate the selected fact and its complete derivation closure,
                    # without treating historical versions as simultaneously active.
                    closure, pending = {}, [chosen.fact.fact_id]
                    while pending:
                        fact_id = pending.pop()
                        if fact_id in closure:
                            continue
                        fact = all_facts.get(fact_id)
                        if fact is not None:
                            closure[fact_id] = fact
                            pending.extend(fact.derivation_inputs)
                    problems = validate_disclosure_evidence(
                        tuple(closure.values()), as_of=universe.as_of
                    )
                    if problems:
                        failure = "Invalid selected evidence: " + "; ".join(
                            d.code for d in problems
                        )
                        chosen = None
                classifications = [
                    t
                    for t in taxonomy
                    if t.issuer_id == issuer
                    and (t.available_at is None or _timestamp(t.available_at) <= cutoff)
                    and (t.effective_at is None or _timestamp(t.effective_at) <= cutoff)
                ]
                if policy.strict_taxonomy_as_of:
                    classifications = [
                        t
                        for t in classifications
                        if t.available_at and t.effective_at and t.version
                    ]
                if classifications:
                    # Undated rows are a fallback only; conflicting rows at a date are ambiguous.
                    best_time = max(
                        (t.effective_at or "", t.available_at or "")
                        for t in classifications
                    )
                    classifications = [
                        t
                        for t in classifications
                        if (t.effective_at or "", t.available_at or "") == best_time
                    ]
                classification = (
                    classifications[0] if len(classifications) == 1 else None
                )
                if policy.required_labels and (
                    classification is None
                    or not set(policy.required_labels).issubset(classification.labels)
                ):
                    chosen, failure = (
                        None,
                        "Missing, ambiguous, or historically unavailable required taxonomy",
                    )
                selection[issuer] = chosen, classification, failure
            chosen, classification, failure = selection[issuer]
            if chosen is None:
                reason = failure
            else:
                metric = (
                    float(chosen.fact.base_value < policy.threshold)
                    if policy.transform == "below"
                    else chosen.fact.base_value
                )
                ids = (chosen.fact.fact_id,)
                status, reason = (
                    "covered",
                    "Selected latest compatible fiscal period/version available at cutoff",
                )
                row = {
                    "position_id": position.position_id,
                    "issuer_id": issuer,
                    "weight": position.value,
                    "metric": metric,
                    "fact_id": ids[0],
                    "target_status": "below"
                    if chosen.fact.base_value < policy.threshold
                    else "at_or_above",
                }
                if classification:
                    row.update(classification.labels)
                rows.append(MappingProxyType(row))
                if (
                    chosen.normalized_period_end
                    and chosen.normalized_period_end != chosen.fact.period_end
                ):
                    diagnostics.append(
                        {
                            "code": "fiscal_date_normalization",
                            "position_id": position.position_id,
                            "reported": chosen.fact.period_end,
                            "normalized": chosen.normalized_period_end,
                        }
                    )
                if classification and (
                    not classification.version
                    or not classification.available_at
                    or not classification.effective_at
                ):
                    diagnostics.append(
                        {
                            "code": "taxonomy_as_of_unknown",
                            "position_id": position.position_id,
                            "scheme": classification.scheme,
                            "message": "Current labels do not establish historical classification",
                        }
                    )
        entries.append(
            PortfolioCoverageEntry(
                position.position_id, position.value, status, reason, metric, ids
            )
        )
    coverage = PortfolioCoverageReport(universe, entries, policy.target_bounds)
    return CompiledPortfolio(
        universe,
        policy,
        observations,
        taxonomy,
        tuple(rows),
        coverage,
        tuple(MappingProxyType(x) for x in diagnostics),
    )
