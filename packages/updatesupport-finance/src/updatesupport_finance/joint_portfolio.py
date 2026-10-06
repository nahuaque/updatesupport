"""Shared reporting on a validated common book, with per-measure coverage."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, fsum, isfinite
from types import MappingProxyType
from typing import Mapping

import updatesupport as us
from updatesupport.artifacts import ReportArtifactMixin

from .compilation import CompiledPortfolio, compile_portfolio_evidence

_EVIDENCE_COLUMNS = {"metric", "fact_id", "target_status"}


@dataclass(frozen=True)
class JointPortfolioEvidence:
    """Named measurements on one supplied universe and issuer mapping.

    Only the positive-value intersection is movable. Covered values outside
    that intersection and missing position weights stay fixed per measurement.
    Fiscal dates and descriptor provenance remain in the original compilations.
    """

    metrics: Mapping[str, CompiledPortfolio]

    def __post_init__(self):
        metrics = dict(self.metrics)
        if not metrics or any(not isinstance(k, str) or not k for k in metrics):
            raise ValueError("at least one named metric is required")
        if any(not isinstance(v, CompiledPortfolio) for v in metrics.values()):
            raise TypeError("metrics must contain CompiledPortfolio objects")
        universe = next(iter(metrics.values())).universe
        seen, facts, classifications = {}, {}, {}
        for compiled in metrics.values():
            if compiled.universe.as_dict() != universe.as_dict():
                raise ValueError(
                    "joint metrics must have identical universes, cutoff, currency, and eligibility"
                )
            for observation in compiled.observations:
                fact = observation.fact
                if (
                    fact.fact_id in facts
                    and facts[fact.fact_id] != observation.as_dict()
                ):
                    raise ValueError(f"inconsistent shared fact {fact.fact_id}")
                facts[fact.fact_id] = observation.as_dict()
            for assignment in compiled.taxonomy:
                if (
                    assignment.issuer_id in classifications
                    and classifications[assignment.issuer_id] != assignment.as_dict()
                ):
                    raise ValueError(
                        f"inconsistent descriptor provenance for issuer {assignment.issuer_id}"
                    )
                classifications[assignment.issuer_id] = assignment.as_dict()
            issuer_values = {}
            for row in compiled.rows:
                descriptors = {
                    k: v for k, v in row.items() if k not in _EVIDENCE_COLUMNS
                }
                position = row["position_id"]
                if position in seen and descriptors != seen[position]:
                    raise ValueError(
                        f"inconsistent descriptors for position {position}"
                    )
                seen[position] = descriptors
                issuer = row["issuer_id"]
                if issuer in issuer_values and issuer_values[issuer] != row["metric"]:
                    raise ValueError(
                        "share classes must retain the same issuer measurement"
                    )
                issuer_values[issuer] = row["metric"]
        if set(metrics).intersection(
            _EVIDENCE_COLUMNS | {k for r in seen.values() for k in r}
        ):
            raise ValueError(
                "metric names cannot overwrite position or descriptor columns"
            )
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def universe(self):
        return next(iter(self.metrics.values())).universe

    @property
    def rows(self):
        indexes = {
            name: {r["position_id"]: r for r in c.rows}
            for name, c in self.metrics.items()
        }
        first = next(iter(indexes.values()))
        common = set.intersection(*(set(i) for i in indexes.values()))
        result = []
        for p in self.universe.positions:
            if p.position_id not in common or p.value == 0:
                continue
            row = {
                k: v
                for k, v in first[p.position_id].items()
                if k not in _EVIDENCE_COLUMNS
            }
            row.update(
                {
                    name: index[p.position_id]["metric"]
                    for name, index in indexes.items()
                }
            )
            result.append(row)
        return tuple(result)

    @property
    def common_value(self):
        return fsum(r["weight"] for r in self.rows)

    def metric_scope(self, name):
        compiled = self.metrics[name]
        coverage = compiled.coverage
        common = {r["position_id"] for r in self.rows}
        outside = [
            e
            for e in coverage.entries
            if e.status == "covered" and e.position_id not in common
        ]
        fixed = fsum(e.value * e.metric_value for e in outside)
        bounds = compiled.policy.target_bounds
        floor = (
            None
            if coverage.unknown_value and bounds is None
            else (
                coverage.unknown_value
                * (bounds[1] - bounds[0])
                / coverage.eligible_value
                if coverage.eligible_value and bounds
                else 0.0
            )
        )
        return {
            "coverage": coverage.as_dict(),
            "common_position_ids": sorted(common),
            "common_value": self.common_value,
            "common_share_eligible": self.common_value / coverage.eligible_value
            if coverage.eligible_value
            else None,
            "known_outside_common": [e.as_dict() for e in outside],
            "fixed_known_contribution": fixed,
            "unknown_domain": bounds,
            "missing_evidence_floor": floor,
            "observed_eligible_interval": coverage.observed_bounds,
        }

    def lift_interval(self, name, interval):
        if interval is not None and (
            len(interval) != 2
            or not all(isfinite(v) for v in interval)
            or interval[0] > interval[1]
        ):
            raise ValueError("common interval must be finite and ordered")
        scope = self.metric_scope(name)
        coverage = self.metrics[name].coverage
        if not coverage.eligible_value or (self.common_value and interval is None):
            return None
        if coverage.unknown_value and scope["unknown_domain"] is None:
            return None
        lo, hi = interval or (0.0, 0.0)
        a, b = scope["unknown_domain"] or (0.0, 0.0)
        fixed = scope["fixed_known_contribution"]
        return (
            (self.common_value * lo + fixed + coverage.unknown_value * a)
            / coverage.eligible_value,
            (self.common_value * hi + fixed + coverage.unknown_value * b)
            / coverage.eligible_value,
        )

    def as_dict(self):
        return {
            "metrics": {n: c.as_dict() for n, c in self.metrics.items()},
            "common_rows": list(self.rows),
            "common_value": self.common_value,
            "scopes": {n: self.metric_scope(n) for n in self.metrics},
            "assumption": "Common book movable; known nonjoint values and all nonjoint weights fixed. Missing values range independently over each metric domain.",
        }


def compile_portfolio_metrics(universe, *, observations, taxonomy=(), policies):
    """Compile named metric policies once on the same holdings and evidence pool."""
    observations, taxonomy = tuple(observations), tuple(taxonomy)
    return JointPortfolioEvidence(
        {
            name: compile_portfolio_evidence(
                universe, observations=observations, taxonomy=taxonomy, policy=policy
            )
            for name, policy in policies.items()
        }
    )


def _pareto(rows, dimensions):
    # Strict domination retains equivalent choices and ties.
    return tuple(
        r
        for r in rows
        if not any(
            all(a <= b + 1e-12 for a, b in zip(dimensions(s), dimensions(r)))
            and any(a < b - 1e-12 for a, b in zip(dimensions(s), dimensions(r)))
            for s in rows
            if s is not r
        )
    )


@dataclass(frozen=True)
class JointPortfolioReport(ReportArtifactMixin):
    evidence: JointPortfolioEvidence
    configuration: Mapping
    candidates: tuple[Mapping, ...]
    frontier: tuple[Mapping, ...]
    selected: Mapping | None
    status: str

    @property
    def rows(self):
        rows = self.evidence.rows
        for payload in self.configuration["refinements"]:
            rows = us.ConditionalRefinement.from_dict(payload).transform(rows)
        return rows

    def as_dict(self):
        return {
            "status": self.status,
            "configuration": dict(self.configuration),
            "scopes": {n: self.evidence.metric_scope(n) for n in self.evidence.metrics},
            "candidates": list(self.candidates),
            "frontier": list(self.frontier),
            "selected": self.selected,
            "search": {
                "exact": True,
                "scope": "declared global subsets and individual conditional drills",
                "evaluations": len(self.candidates),
                "cost_universe": "positive-value common book",
            },
            "limitations": [
                "Target-aware finite search; no arbitrary taxonomy optimum or causal interpretation.",
                "Uncertainty outside the common book uses fixed weights and independent metric domains.",
                "Annual selection retains each issuer's fiscal period; it does not imply a common calendar year.",
            ],
        }

    def to_tables(self):
        return {
            "coverage": tuple(
                {"metric": n, **self.evidence.metric_scope(n)}
                for n in self.evidence.metrics
            ),
            "representations": tuple(dict(c) for c in self.candidates),
            "scenarios": tuple(
                {"candidate_index": i, "metric": n, **s}
                for i, c in enumerate(self.candidates)
                for n, m in c["metrics"].items()
                for s in m["scenarios"]
            ),
        }

    def to_markdown(self):
        lines = [
            "# Joint portfolio reporting",
            "",
            f"Status: **{self.status}**. Evaluation scope: **{self.configuration['scope']}**. Report-cell cost counts the common book.",
            f"Precision limits: {self.configuration['ambiguity_limits'] or 'none'}. Decision rules: {self.configuration['decisions'] or 'none'}.",
            "",
            "| Metric | Covered positions | Common positions | Missing-evidence floor |",
            "| --- | ---: | ---: | ---: |",
        ]
        for n in self.evidence.metrics:
            s = self.evidence.metric_scope(n)
            floor = s["missing_evidence_floor"]
            lines.append(
                f"| {n.replace('|', '/')} | {sum(e['status'] == 'covered' for e in s['coverage']['entries'])} | {len(s['common_position_ids'])} | {'unknown' if floor is None else f'{floor:.6g}'} |"
            )
        described = self.selected or (self.candidates[0] if self.candidates else None)
        if described:
            lines += [
                "",
                "Selected representation:" if self.selected else "Base representation:",
                "",
                "| Metric | Detailed common mean | Common report interval | Eligible report interval |",
                "| --- | ---: | --- | --- |",
            ]
            for n, m in described["metrics"].items():
                lines.append(
                    f"| {n.replace('|', '/')} | {m['observed_common']:.6g} | {m['common_interval']} | {m['eligible_interval']} |"
                )
        lines += [
            "",
            "| Added disclosure | Cells | Worst common width | Worst eligible width | Contract met |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for c in self.frontier:
            eligible = [m["eligible_width"] for m in c["metrics"].values()]
            lines.append(
                f"| {', '.join(c['added_columns']) or 'base'} | {c['public_cells']} | {max(m['common_width'] for m in c['metrics'].values()):.6g} | {'unknown' if None in eligible else f'{max(eligible):.6g}'} | {c['contract_met']} |"
            )
        return "\n".join(lines)

    def breaking_witness(
        self, metric, *, candidate_index=0, decision, q_index=0, threshold_margin=1e-9
    ):
        """Smallest common-book reversal under the same Q; no eligible-scope claim.

        Multiplying TV mass by common_value gives a transferred holdings value,
        not a forecast loss or trade recommendation.
        """
        c = self.candidates[candidate_index]
        q = self.configuration["q_presets"][metric][q_index]
        hidden = (
            *self.configuration["hidden"],
            *(x["name"] for x in self.configuration["refinements"]),
        )
        claim = us.claim(
            metric,
            target=metric,
            public=c["public_columns"],
            hidden=hidden,
            weight="weight",
            min_cell_weight=0,
            q=q,
            decision=decision,
        )
        witness = claim.audit(self.rows).breaking_witness(
            respect_q=True, threshold_margin=threshold_margin
        )
        radius = witness.witness_tv_distance
        return {
            "scope": "common",
            "common_value": self.evidence.common_value,
            "transferred_value": None
            if radius is None
            else radius * self.evidence.common_value,
            "witness": witness.as_dict(),
        }


def joint_portfolio_report(
    evidence: JointPortfolioEvidence,
    *,
    public,
    hidden,
    candidate_refinements=(),
    conditional_refinements=(),
    q_presets=("saturated",),
    ambiguity_limits=None,
    decisions=None,
    scope="eligible",
    max_added_columns=None,
    max_evaluations=256,
    disclosure_costs=None,
):
    """Exact finite shared frontier; criteria are optional and never invented.

    Every global subset and single local drill is solved on the entire common
    book for every metric/Q. Conditional combinations are outside this search.
    """
    public, hidden = tuple(public), tuple(hidden)
    columns = tuple(candidate_refinements)
    if scope not in {"common", "eligible"}:
        raise ValueError("scope must be common or eligible")
    if (
        len(set(public)) != len(public)
        or len(set(hidden)) != len(hidden)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("column lists must be unique")
    if (
        not public
        or not set(public).issubset(hidden)
        or not set(columns).issubset(hidden)
        or set(columns).intersection(public)
    ):
        raise ValueError(
            "public/candidate columns must be distinct declared hidden descriptors"
        )
    if set(hidden).intersection(
        set(evidence.metrics) | {"weight", "position_id", "fact_id"}
    ):
        raise ValueError(
            "hidden support cannot include target, weight, or position evidence columns"
        )
    refinements = tuple(
        x
        if isinstance(x, us.ConditionalRefinement)
        else us.ConditionalRefinement.from_dict(x)
        for x in conditional_refinements
    )
    names = tuple(x.name for x in refinements)
    if len(set(names)) != len(names) or set(names).intersection(hidden):
        raise ValueError("conditional outputs must be unique new descriptors")
    if any(
        not set(public).issubset(x.predicate)
        or not set(x.predicate).issubset(hidden)
        or x.column not in hidden
        or x.column in x.predicate
        for x in refinements
    ):
        raise ValueError(
            "conditional predicates must include the base public bucket and split another hidden descriptor"
        )
    limits = dict(ambiguity_limits or {})
    costs = dict(disclosure_costs or {})
    if set(costs) - (set(hidden) | set(names)) or any(
        not isfinite(v) or v < 0 for v in costs.values()
    ):
        raise ValueError(
            "disclosure costs must be finite, nonnegative, and name declared columns"
        )
    rules = {
        n: us.DecisionRule.from_value(d).as_dict() for n, d in (decisions or {}).items()
    }
    if (set(limits) | set(rules)) - set(evidence.metrics):
        raise ValueError("criteria refer to an unknown metric")
    if any(not isfinite(float(v)) or v < 0 for v in limits.values()):
        raise ValueError("precision limits must be finite and nonnegative")
    if any(not isfinite(d["threshold"]) for d in rules.values()):
        raise ValueError("decision thresholds must be finite")
    qs = {}
    if isinstance(q_presets, Mapping) and set(q_presets) != set(evidence.metrics):
        raise ValueError("Q mapping must specify every metric")
    for name in evidence.metrics:
        grid = q_presets[name] if isinstance(q_presets, Mapping) else q_presets
        if not grid:
            raise ValueError("each metric requires at least one Q")
        qs[name] = [us.QSpec.from_value(q).as_dict() for q in grid]
    maximum = len(columns) if max_added_columns is None else max_added_columns
    if not isinstance(maximum, int) or not 0 <= maximum <= len(columns):
        raise ValueError("max_added_columns must be within the candidate count")
    count = sum(comb(len(columns), k) for k in range(maximum + 1)) + len(refinements)
    if count > max_evaluations:
        raise ValueError(
            f"exact search needs {count} evaluations; budget is {max_evaluations}"
        )
    config = {
        "public": public,
        "hidden": hidden,
        "candidate_refinements": columns,
        "refinements": [x.as_dict() for x in refinements],
        "q_presets": qs,
        "ambiguity_limits": limits,
        "decisions": rules,
        "scope": scope,
        "max_added_columns": maximum,
        "max_evaluations": max_evaluations,
        "disclosure_costs": costs,
    }
    rows = evidence.rows
    for refinement in refinements:
        rows = refinement.transform(rows)
    if not rows:
        return JointPortfolioReport(
            evidence, config, (), (), None, "inconclusive_no_common_book"
        )
    choices = [s for k in range(maximum + 1) for s in combinations(columns, k)] + [
        (*tuple(k for k in x.predicate if k not in public), x.name) for x in refinements
    ]
    candidates = []
    for added in choices:
        schema = (*public, *added)
        metrics, cells = {}, None
        for name in evidence.metrics:
            evaluated = us.public_representation_frontier(
                rows,
                base_public=schema,
                hidden=(*hidden, *names),
                target=name,
                weight="weight",
                min_cell_weight=0,
                q_presets=[us.QSpec.from_value(q).to_preset() for q in qs[name]],
                candidate_refinements=(),
            ).candidates[0]
            cells = evaluated.public_cells
            lower = min(s.lower for s in evaluated.scenarios)
            upper = max(s.upper for s in evaluated.scenarios)
            interval = (lower, upper)
            eligible = evidence.lift_interval(name, interval)
            tested = interval if scope == "common" else eligible
            checks = []
            if name in limits:
                checks.append(
                    tested is not None and tested[1] - tested[0] <= limits[name] + 1e-10
                )
            if name in rules:
                rule = us.DecisionRule.from_value(rules[name])
                checks.append(
                    tested is not None
                    and rule.evaluate(tested[0])
                    == rule.evaluate(tested[1])
                    == rule.pass_label
                )
            metrics[name] = {
                "observed_common": evaluated.observed_value,
                "common_interval": interval,
                "common_width": upper - lower,
                "eligible_interval": eligible,
                "eligible_width": None
                if eligible is None
                else eligible[1] - eligible[0],
                "contract_met": all(checks) if checks else None,
                "scenarios": [s.as_dict() for s in evaluated.scenarios],
            }
        constrained = [
            m["contract_met"] for m in metrics.values() if m["contract_met"] is not None
        ]
        candidates.append(
            {
                "added_columns": added,
                "public_columns": schema,
                "public_cells": cells,
                "added_disclosure_count": len(added),
                "disclosure_cost": sum(costs[n] for n in added)
                if costs and all(n in costs for n in added)
                else None,
                "kind": "conditional" if set(added).intersection(names) else "global",
                "metrics": metrics,
                "contract_met": all(constrained) if constrained else None,
            }
        )

    priced = bool(costs) and all(c["disclosure_cost"] is not None for c in candidates)

    def dimensions(c):
        return (
            c["public_cells"],
            c["added_disclosure_count"],
            *((c["disclosure_cost"],) if priced else ()),
            *(m["common_width"] for m in c["metrics"].values()),
        )

    frontier = _pareto(candidates, dimensions)
    feasible = [c for c in candidates if c["contract_met"] is True]
    selected = (
        min(
            feasible,
            key=lambda c: (
                c["public_cells"],
                c["added_disclosure_count"],
                max(m["common_width"] for m in c["metrics"].values()),
            ),
        )
        if feasible
        else None
    )
    status = (
        "raw_frontier"
        if not (limits or rules)
        else "contract_met"
        if selected
        else "no_allowed_representation_meets_contract"
    )
    return JointPortfolioReport(
        evidence, config, tuple(candidates), frontier, selected, status
    )
