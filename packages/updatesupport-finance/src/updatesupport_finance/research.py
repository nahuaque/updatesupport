"""Finite report/evidence planning under fixed nonjoint position weights."""

from dataclasses import dataclass, field
from itertools import combinations
from math import fsum, isfinite
from types import MappingProxyType
from typing import Mapping

from updatesupport.artifacts import ReportArtifactMixin

from .joint_portfolio import _pareto


@dataclass(frozen=True)
class EvidenceAcquisition:
    """Conditional width reduction, not an assertion about the eventual value.

    A company package can require several sources/facts. Unknown availability
    is preserved; unavailable actions are never selected. Residual widths are
    caller-supplied attainable bounds, conditional on successful acquisition.
    """

    name: str
    position_ids: tuple[str, ...]
    residual_widths: Mapping[str, float]
    availability: str = "unknown"
    cost: float | None = None
    source_requirements: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or not self.position_ids or not self.residual_widths:
            raise ValueError(
                "an acquisition requires a name, positions, and residual widths"
            )
        if self.availability not in {"available", "unknown", "unavailable"}:
            raise ValueError("invalid source availability")
        if len(set(self.position_ids)) != len(self.position_ids):
            raise ValueError("acquisition positions must be unique")
        if any(not isfinite(float(w)) or w < 0 for w in self.residual_widths.values()):
            raise ValueError("residual widths must be finite and nonnegative")
        if self.cost is not None and (not isfinite(self.cost) or self.cost < 0):
            raise ValueError("cost must be finite and nonnegative")
        object.__setattr__(self, "position_ids", tuple(self.position_ids))
        object.__setattr__(self, "source_requirements", tuple(self.source_requirements))
        object.__setattr__(
            self, "residual_widths", MappingProxyType(dict(self.residual_widths))
        )

    def as_dict(self):
        return {**vars(self), "residual_widths": dict(self.residual_widths)}


@dataclass(frozen=True)
class PortfolioRepairFrontier(ReportArtifactMixin):
    actions: tuple[EvidenceAcquisition, ...]
    plans: tuple[Mapping, ...]
    frontier: tuple[Mapping, ...]
    feasible_frontier: tuple[Mapping, ...]
    ambiguity_limits: Mapping = field(default_factory=dict)

    def as_dict(self):
        return {
            "actions": [a.as_dict() for a in self.actions],
            "plans": list(self.plans),
            "frontier": list(self.frontier),
            "feasible_frontier": list(self.feasible_frontier),
            "ambiguity_limits": dict(self.ambiguity_limits),
            "evaluations": len(self.plans),
            "exact": True,
            "status": "conditional_plans"
            if self.feasible_frontier
            else "no_sufficient_plan",
            "assumptions": [
                "Prospective planning bounds; no acquired-evidence certificate or assumed measurement outcome.",
                "Resolved nonjoint values and weights remain fixed. If they enter movable support, rebuild and rerun.",
                "Widths combine independent per-target missing domains. Source availability is supplied by the caller.",
            ],
        }

    def to_tables(self):
        return {
            "plans": tuple(dict(p) for p in self.plans),
            "actions": tuple(a.as_dict() for a in self.actions),
        }

    def to_markdown(self):
        lines = [
            "# Conditional reporting and evidence plans",
            "",
            "Successful acquisition is assumed only for planning; these are not certificates.",
            "",
            "| Report cells | Evidence packages | Eligible widths | Meets precision |",
            "| ---: | --- | --- | --- |",
        ]
        for p in self.feasible_frontier or self.frontier:
            lines.append(
                f"| {p['public_cells']} | {', '.join(p['actions']) or 'none'} | {p['eligible_widths']} | {p['meets_limits']} |"
            )
        return "\n".join(lines)


def plan_portfolio_repairs(report, actions, *, ambiguity_limits, max_evaluations=4096):
    """Enumerate allowed report/evidence combinations; retain ties and failures.

    Without complete cost estimates the frontier compares cells, package count,
    and each residual width. Cost is an extra dimension only when all eligible
    packages have estimates; no hours or currency cost is fabricated.
    """
    actions = tuple(actions)
    limits = dict(ambiguity_limits)
    if (
        not limits
        or set(limits) - set(report.evidence.metrics)
        or any(not isfinite(v) or v < 0 for v in limits.values())
    ):
        raise ValueError("finite precision limits must name known metrics")
    if len({a.name for a in actions}) != len(actions):
        raise ValueError("acquisition names must be unique")
    ledgers = {
        n: {e.position_id: e for e in c.coverage.entries}
        for n, c in report.evidence.metrics.items()
    }
    for action in actions:
        for name, width in action.residual_widths.items():
            if name not in ledgers:
                raise ValueError("acquisition names an unknown metric")
            bounds = report.evidence.metrics[name].policy.target_bounds
            if bounds is not None and width > bounds[1] - bounds[0]:
                raise ValueError("acquisition cannot widen a declared domain")
            for pid in action.position_ids:
                if pid not in ledgers[name] or ledgers[name][pid].status not in {
                    "missing",
                    "unmapped",
                }:
                    raise ValueError(
                        "acquisition must affect missing nonjoint evidence"
                    )
    allowed = tuple(a for a in actions if a.availability != "unavailable")
    count = len(report.candidates) * 2 ** len(allowed)
    if count > max_evaluations:
        raise ValueError(
            f"exact planning needs {count} evaluations; budget is {max_evaluations}"
        )
    priced = bool(allowed) and all(a.cost is not None for a in allowed)
    reporting_priced = bool(report.candidates) and all(
        c["disclosure_cost"] is not None for c in report.candidates
    )
    plans = []
    for k in range(len(allowed) + 1):
        for subset in combinations(allowed, k):
            narrowed = {}
            for a in subset:
                for name, w in a.residual_widths.items():
                    for pid in a.position_ids:
                        narrowed[name, pid] = min(w, narrowed.get((name, pid), w))
            for candidate in report.candidates:
                widths = {}
                for name, compiled in report.evidence.metrics.items():
                    bounds = compiled.policy.target_bounds
                    default = None if bounds is None else bounds[1] - bounds[0]
                    missing = [
                        (e.value, narrowed.get((name, e.position_id), default))
                        for e in compiled.coverage.entries
                        if e.status in {"missing", "unmapped"} and e.value
                    ]
                    total = compiled.coverage.eligible_value
                    widths[name] = (
                        None
                        if not total or any(w is None for _, w in missing)
                        else (
                            report.evidence.common_value
                            * candidate["metrics"][name]["common_width"]
                            + fsum(v * w for v, w in missing)
                        )
                        / total
                    )
                plans.append(
                    {
                        "public_columns": candidate["public_columns"],
                        "public_cells": candidate["public_cells"],
                        "actions": tuple(a.name for a in subset),
                        "package_count": k,
                        "unknown_availability": tuple(
                            a.name for a in subset if a.availability == "unknown"
                        ),
                        "evidence_cost": sum(a.cost for a in subset)
                        if priced
                        else None,
                        "disclosure_cost": candidate["disclosure_cost"],
                        "eligible_widths": widths,
                        "meets_limits": all(
                            widths[n] is not None and widths[n] <= w + 1e-10
                            for n, w in limits.items()
                        ),
                    }
                )

    def dimensions(p):
        return (
            p["public_cells"],
            p["package_count"],
            *((p["evidence_cost"],) if priced else ()),
            *((p["disclosure_cost"],) if reporting_priced else ()),
            *(float("inf") if w is None else w for w in p["eligible_widths"].values()),
        )

    feasible = [p for p in plans if p["meets_limits"]]

    # Feasible trade-offs emphasize reporting/evidence effort. Residual widths
    # remain visible, but a tighter passing width does not erase an effort tie.
    def effort(p):
        return dimensions(p)[: 2 + int(priced) + int(reporting_priced)]

    return PortfolioRepairFrontier(
        actions,
        tuple(plans),
        _pareto(plans, dimensions),
        _pareto(feasible, effort),
        limits,
    )
