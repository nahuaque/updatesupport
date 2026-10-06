"""Finance headline truth, summary support, and coverage in one artifact."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import updatesupport as us
from updatesupport.artifacts import ReportArtifactMixin

from .compilation import CompiledPortfolio


def _verdict(rule, interval):
    if interval is None:
        return "inconclusive"
    lower, upper = (rule.evaluate(x) for x in interval)
    if lower != upper:
        return "inconclusive"
    return "supported" if lower == rule.pass_label else "contradicted"


@dataclass(frozen=True)
class PortfolioHeadlineReport(ReportArtifactMixin):
    portfolio: CompiledPortfolio
    configuration: Mapping[str, Any]
    audit: Any
    design: Any
    breaking: Any = None

    def __post_init__(self):
        _validate_configuration(self.configuration)
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )

    @property
    def decision(self):
        return us.DecisionRule.from_value(self.configuration["decision"])

    @property
    def observed_bounds(self):
        if self.configuration["scope"] == "eligible":
            return self.portfolio.coverage.observed_bounds
        value = self.portfolio.coverage.covered_mean
        return None if value is None else (value, value)

    @property
    def summary_bounds(self):
        interval = (
            None
            if self.audit is None
            else (self.audit.interval.lower, self.audit.interval.upper)
        )
        if self.configuration["scope"] == "eligible":
            return self.portfolio.coverage.lift_interval(interval)
        return interval

    @property
    def actual_headline(self):
        return _verdict(self.decision, self.observed_bounds)

    @property
    def summary_support(self):
        return _verdict(self.decision, self.summary_bounds)

    @cached_property
    def refinements(self):
        if self.design is None:
            return ()
        direct = set(self.configuration["direct_target_refinements"])
        result = []
        for option in self.design.repair_plan.options:
            columns = tuple(option.columns)
            # Evaluate the refinement in the headline's scope. A covered-book
            # repair cannot erase uncertainty in unknown eligible positions.
            public = tuple(dict.fromkeys((*self.configuration["public"], *columns)))
            claim = _claim(self.configuration, public=public, refinements=())
            refined = us.audit_claim([dict(r) for r in self.portfolio.rows], claim)
            interval = (refined.interval.lower, refined.interval.upper)
            if self.configuration["scope"] == "eligible":
                interval = self.portfolio.coverage.lift_interval(interval)
            details = option.as_dict()
            details["covered_claim_certified"] = details.pop("certifies_claim")
            result.append(
                {
                    **details,
                    "kind": "direct_target"
                    if direct.intersection(columns)
                    else "independent",
                    "headline_bounds": interval,
                    "headline_support": _verdict(self.decision, interval),
                }
            )
        return tuple(result)

    @property
    def transfers(self):
        if self.breaking is None:
            return ()
        book_value = self.portfolio.coverage.covered_value
        return tuple(
            {
                **t.as_dict(),
                "covered_percentage_points": 100 * t.mass,
                "eligible_percentage_points": 100
                * t.mass
                * book_value
                / self.portfolio.coverage.eligible_value,
                "amount": t.mass * book_value,
                "currency": self.portfolio.universe.currency,
            }
            for t in self.breaking.transfers
        )

    def as_dict(self):
        return {
            "headline": self.configuration["headline"],
            "scope": self.configuration["scope"],
            "actual_headline": self.actual_headline,
            "summary_support": self.summary_support,
            "observed_bounds": self.observed_bounds,
            "summary_bounds": self.summary_bounds,
            "coverage": self.portfolio.coverage.as_dict(),
            "diagnostics": [dict(d) for d in self.portfolio.diagnostics],
            "audit": None if self.audit is None else self.audit.as_dict(),
            "design": None if self.design is None else self.design.as_dict(),
            "refinements": list(self.refinements),
            "minimum_breaking_witness": None
            if self.breaking is None
            else self.breaking.as_dict(),
            "transfers": list(self.transfers),
            "limitations": [
                "Summary bounds reweight covered hidden cells with public totals fixed; unknown eligible weights remain fixed.",
                "Headline support is conditional on the supplied universe, metric policy, taxonomy, and Q.",
                "Minimum breaking witnesses, when requested, refer to the covered book. They do not assign missing fundamentals.",
                "Direct-target refinements disclose the target classification itself; independent refinements use other supplied labels.",
            ],
        }

    def to_markdown(self):
        payload = self.as_dict()
        lines = [
            f"# {payload['headline']}",
            "",
            f"- Scope: {payload['scope']}",
            f"- Actual headline: **{self.actual_headline}**",
            f"- Summary support: **{self.summary_support}**",
            f"- Observed bounds: {self.observed_bounds}",
            f"- Summary bounds: {self.summary_bounds}",
            f"- Covered / eligible value: {self.portfolio.coverage.covered_value:,.2f} / {self.portfolio.coverage.eligible_value:,.2f}",
        ]
        if self.breaking is not None:
            lines.extend(
                [
                    f"- Covered-book breaking witness: {self.breaking.status}",
                    f"- Minimum TV distance: {self.breaking.witness_tv_distance}",
                ]
            )
        if payload["refinements"]:
            lines.extend(
                [
                    "",
                    "## Refinements",
                    "",
                    "| columns | kind | headline support | bounds |",
                    "| --- | --- | --- | --- |",
                ]
            )
            lines.extend(
                f"| {', '.join(r['columns'])} | {r['kind']} | {r['headline_support']} | {r['headline_bounds']} |"
                for r in payload["refinements"]
            )
        lines.extend(
            ["", "## Limitations", "", *[f"- {x}" for x in payload["limitations"]]]
        )
        return "\n".join(lines)


def _claim(config, *, public=None, refinements=None):
    return us.claim(
        config["headline"],
        public=config["public"] if public is None else public,
        hidden=config["hidden"],
        target="metric",
        weight="weight",
        q=config["q"],
        candidate_refinements=config["candidate_refinements"]
        if refinements is None
        else refinements,
        decision=config["decision"],
        min_cell_weight=0,
        max_dropped_weight_share=0,
    )


def _validate_configuration(config):
    required = {
        "headline",
        "decision",
        "public",
        "hidden",
        "scope",
        "q",
        "candidate_refinements",
        "direct_target_refinements",
        "include_breaking",
        "respect_q",
        "threshold_margin",
    }
    if set(config) != required:
        raise ValueError("invalid portfolio headline configuration fields")
    if config["scope"] not in {"covered", "eligible"}:
        raise ValueError("scope must be covered or eligible")
    rule = us.DecisionRule.from_value(config["decision"])
    if not isfinite(rule.threshold) or rule.pass_label == rule.fail_label:
        raise ValueError("decision needs a finite threshold and distinct labels")
    if not isfinite(config["threshold_margin"]) or config["threshold_margin"] <= 0:
        raise ValueError("threshold_margin must be finite and positive")
    for name in (
        "public",
        "hidden",
        "candidate_refinements",
        "direct_target_refinements",
    ):
        values = config[name]
        if (
            isinstance(values, str)
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain unique column names")
    if not config["public"] or not {
        "issuer_id",
        *config["public"],
        *config["candidate_refinements"],
    }.issubset(config["hidden"]):
        raise ValueError(
            "hidden must include issuer_id and every public/refinement column"
        )
    if not (
        {"target_status", "metric"} & set(config["candidate_refinements"])
    ).issubset(config["direct_target_refinements"]):
        raise ValueError(
            "metric/target_status must be declared direct-target refinements"
        )
    if not isinstance(config["respect_q"], bool) or not isinstance(
        config["include_breaking"], bool
    ):
        raise ValueError("respect_q and include_breaking must be booleans")
    _claim(config)


def portfolio_headline_report(
    portfolio: CompiledPortfolio,
    *,
    headline: str,
    decision: us.DecisionRule | Mapping,
    public: Sequence[str],
    hidden: Sequence[str],
    scope: str = "covered",
    q: Any = "saturated",
    candidate_refinements: Sequence[str] = (),
    direct_target_refinements: Sequence[str] = ("target_status",),
    include_breaking: bool = True,
    respect_q: bool = True,
    threshold_margin: float = 1e-6,
) -> PortfolioHeadlineReport:
    """Audit a headline without conflating truth, decision invariance, and coverage.

    Eligible scope lifts the covered audit through missing-data bounds with
    unknown position weights held fixed. A breaking witness explicitly refers
    to the covered book. Hidden columns must contain all public/refinement
    columns and issuer_id, so transfers identify issuers rather than averages.
    """
    decision = us.DecisionRule.from_value(decision)
    for values in (public, hidden, candidate_refinements, direct_target_refinements):
        if isinstance(values, str):
            raise TypeError("columns must be sequences of names")
    public, hidden, candidate_refinements = (
        tuple(public),
        tuple(hidden),
        tuple(candidate_refinements),
    )
    direct = tuple(direct_target_refinements)
    config = {
        "headline": headline,
        "decision": decision.as_dict(),
        "public": public,
        "hidden": hidden,
        "scope": scope,
        "q": q,
        "candidate_refinements": candidate_refinements,
        "direct_target_refinements": direct,
        "include_breaking": include_breaking,
        "respect_q": respect_q,
        "threshold_margin": threshold_margin,
    }
    _validate_configuration(config)
    if portfolio.coverage.covered_value == 0:
        return PortfolioHeadlineReport(portfolio, config, None, None)
    rows = [dict(r) for r in portfolio.rows]
    audit = us.audit_claim(rows, _claim(config))
    if not audit.coverage_sufficient:
        raise ValueError(
            "core grouping dropped portfolio value; scope cannot be preserved"
        )
    design = us.design_public_report(audit)
    breaking = (
        audit.breaking_witness(threshold_margin=threshold_margin, respect_q=respect_q)
        if include_breaking
        else None
    )
    return PortfolioHeadlineReport(portfolio, config, audit, design, breaking)
