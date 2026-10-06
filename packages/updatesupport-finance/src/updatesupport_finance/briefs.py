"""Compact decision briefs and explicit guards on cross-period comparisons."""

from dataclasses import dataclass, field
from datetime import date
from math import isfinite

from .disclosure import DisclosureAuditPack
from .scenarios import disclosure_support_basis


@dataclass(frozen=True)
class DisclosureScope:
    entity: str
    measure: str
    unit: str
    period_end: str
    window_kind: str
    cohort: str
    period_start: str | None = None
    currency: str | None = None
    cohort_kind: str = "aggregate"
    customer_basis: str | None = None
    geography_basis: str | None = None

    def __post_init__(self):
        if not all(
            (self.entity, self.measure, self.unit, self.window_kind, self.cohort)
        ):
            raise ValueError(
                "scope needs entity, measure, unit, window kind and cohort"
            )
        end = date.fromisoformat(self.period_end)
        if self.period_start and date.fromisoformat(self.period_start) > end:
            raise ValueError("scope period is reversed")
        if self.cohort_kind not in {"aggregate", "identified", "anonymous"}:
            raise ValueError("unknown cohort kind")

    def as_dict(self):
        return vars(self).copy()


def compare_disclosure_scopes(before, after):
    """Assess declared comparability; never match anonymous customer identities.

    Same fiscal window kinds may span different periods. Custom windows must
    match dates. Anonymous cohorts can only be compared as the same saved cohort
    within one period. Passing this guard does not establish causal attribution.
    """
    if before is None or after is None:
        return {
            "status": "unassessed",
            "comparable": None,
            "reasons": ["measurement scope was not supplied"],
            "causal": False,
        }
    before = (
        before if isinstance(before, DisclosureScope) else DisclosureScope(**before)
    )
    after = after if isinstance(after, DisclosureScope) else DisclosureScope(**after)
    reasons = [
        f"different {key}"
        for key in (
            "entity",
            "measure",
            "unit",
            "currency",
            "window_kind",
            "cohort_kind",
            "customer_basis",
            "geography_basis",
        )
        if getattr(before, key) != getattr(after, key)
    ]
    dates_differ = (before.period_start, before.period_end) != (
        after.period_start,
        after.period_end,
    )
    if before.cohort != after.cohort:
        reasons.append("different declared cohorts")
    if "anonymous" in {before.cohort_kind, after.cohort_kind} and dates_differ:
        reasons.append("anonymous identities cannot be linked across periods")
    if before.window_kind in {"custom", "schedule", "lifetime"} and dates_differ:
        reasons.append("different custom or commitment windows")
    return {
        "status": "not_comparable" if reasons else "comparable",
        "comparable": not reasons,
        "reasons": reasons,
        "causal": False,
    }


@dataclass(frozen=True)
class AnalystBaseline:
    label: str
    value: float
    unit: str

    def __post_init__(self):
        if not self.label or not self.unit or not isfinite(self.value):
            raise ValueError("baseline needs a label, finite value and display unit")

    def as_dict(self):
        return vars(self).copy()


@dataclass(frozen=True)
class AnalystFinding:
    question: str
    pack: DisclosureAuditPack
    scope: DisclosureScope
    baseline: AnalystBaseline | None = None
    decision_impacts: tuple[AnalystBaseline, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    unanswered_scope: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.question:
            raise ValueError("an analyst question is required")
        target = next(
            t for t in self.pack.report.problem.targets if t.name == self.pack.target
        )
        if self.baseline is not None and self.baseline.unit != (
            target.unit or self.scope.unit
        ):
            raise ValueError("baseline must use the target's display unit")
        for key in ("decision_impacts", "missing_evidence", "unanswered_scope"):
            object.__setattr__(self, key, tuple(getattr(self, key)))

    def as_dict(self):
        problem = self.pack.report.problem
        target = next(t for t in problem.targets if t.name == self.pack.target)
        claim = self.pack.claim_audit
        return {
            "question": self.question,
            "scope": self.scope.as_dict(),
            "target": target.name,
            "target_label": target.label or target.name,
            "unit": target.unit or self.scope.unit,
            "baseline": None if self.baseline is None else self.baseline.as_dict(),
            "interval": {
                "status": self.pack.interval.status,
                "lower": self.pack.interval.scaled_lower,
                "upper": self.pack.interval.scaled_upper,
            },
            "claim": None
            if claim is None
            else {
                "label": claim.claim.label or claim.claim.statement,
                "verdict": claim.verdict,
            },
            "tier": self.pack.tier,
            "support": disclosure_support_basis(problem, self.pack.tier),
            "tiers": [
                {
                    "tier": s.name,
                    "lower": self.pack.report.interval(
                        target=target.name, scenario=s.name
                    ).scaled_lower,
                    "upper": self.pack.report.interval(
                        target=target.name, scenario=s.name
                    ).scaled_upper,
                    "status": self.pack.report.interval(
                        target=target.name, scenario=s.name
                    ).status,
                    "support": disclosure_support_basis(problem, s.name),
                }
                for s in problem.scenarios
            ],
            "decision_impacts": [v.as_dict() for v in self.decision_impacts],
            "missing_evidence": list(self.missing_evidence),
            "unanswered_scope": list(self.unanswered_scope),
            "assumptions": list(self.pack.assumptions),
            "sources": [
                {"fact_id": f.fact_id, "url": f.source_url, "source_id": f.source_id}
                for f in self.pack.evidence
            ],
        }


def _plain(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value):
    return "unbounded" if value is None else f"{value:,.6g}"


def _bound(value, status):
    return "—" if status in {"infeasible", "numerical_error"} else _number(value)


@dataclass(frozen=True)
class AnalystDecisionBrief:
    title: str
    findings: tuple[AnalystFinding, ...]
    comparison_notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.title or not self.findings:
            raise ValueError("decision briefs require a title and findings")
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "comparison_notes", tuple(self.comparison_notes))

    def as_dict(self):
        return {
            "title": self.title,
            "findings": [f.as_dict() for f in self.findings],
            "comparison_notes": list(self.comparison_notes),
        }

    def to_json(self, **kwargs):
        import json

        return json.dumps(self.as_dict(), allow_nan=False, **kwargs)

    def to_tables(self):
        return {
            "analyst_decisions": tuple(
                {
                    "question": f.question,
                    "scope": f.scope.as_dict(),
                    **f.as_dict()["interval"],
                    "unit": f.as_dict()["unit"],
                    "support": f.as_dict()["support"]["basis"],
                    "verdict": None
                    if f.pack.claim_audit is None
                    else f.pack.claim_audit.verdict,
                }
                for f in self.findings
            )
        }

    def to_markdown(self):
        lines = [f"# {_plain(self.title)}", ""]
        for finding in self.findings:
            row = finding.as_dict()
            scope = row["scope"]
            lines.extend(
                [
                    f"## {_plain(row['question'])}",
                    "",
                    f"{_plain(scope['entity'])} · {_plain(scope['measure'])} · {scope['period_start'] or 'instant'} to {scope['period_end']}",
                    "",
                    f"Cohort: {_plain(scope['cohort'])} ({scope['cohort_kind']}); window: {_plain(scope['window_kind'])}.",
                    "",
                    f"**{_plain(row['target_label'])}: {_bound(row['interval']['lower'], row['interval']['status'])} to {_bound(row['interval']['upper'], row['interval']['status'])} {_plain(row['unit'])}** ({row['interval']['status']}).",
                    "",
                    f"Support: **{row['support']['basis']}**, tier `{_plain(row['tier'])}`; roles: {', '.join(row['support']['roles'])}.",
                    "",
                ]
            )
            if row["claim"]:
                lines.extend(
                    [
                        f"{_plain(row['claim']['label'])}: **{row['claim']['verdict']}**.",
                        "",
                    ]
                )
            if row["baseline"]:
                b = row["baseline"]
                lines.extend(
                    [
                        f"Arithmetic baseline — {_plain(b['label'])}: {_number(b['value'])} {_plain(b['unit'])}.",
                        "",
                    ]
                )
            lines.extend(
                [
                    "| Evidence / policy tier | Status | Lower | Upper | Support |",
                    "| --- | --- | ---: | ---: | --- |",
                ]
            )
            lines.extend(
                f"| {_plain(t['tier'])} | {t['status']} | {_bound(t['lower'], t['status'])} | {_bound(t['upper'], t['status'])} | {t['support']['basis']} |"
                for t in row["tiers"]
            )
            for heading, values in (
                (
                    "Decision / stress impact",
                    [
                        f"{v['label']}: {_number(v['value'])} {v['unit']}"
                        for v in row["decision_impacts"]
                    ],
                ),
                ("Missing evidence", row["missing_evidence"]),
                ("Unanswered scope", row["unanswered_scope"]),
                ("Assumptions", row["assumptions"]),
            ):
                if values:
                    lines.extend(
                        ["", f"### {heading}", "", *[f"- {_plain(v)}" for v in values]]
                    )
            sources = list(dict.fromkeys(s["url"] for s in row["sources"]))
            if sources:
                lines.extend(
                    [
                        "",
                        "Sources: "
                        + ", ".join(
                            f"[source {i + 1}]({url})" for i, url in enumerate(sources)
                        ),
                        "",
                    ]
                )
        if self.comparison_notes:
            lines.extend(
                [
                    "",
                    "## Comparison guards",
                    "",
                    *[f"- {_plain(n)}" for n in self.comparison_notes],
                ]
            )
        lines.extend(
            [
                "",
                "Bounds depend on supplied evidence, declared measurement relationships and selected policies. Comparisons are descriptive; they do not establish causality.",
            ]
        )
        return "\n".join(lines)
