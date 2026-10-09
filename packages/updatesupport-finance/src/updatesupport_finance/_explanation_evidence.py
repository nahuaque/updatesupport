"""Hypothetical evidence outcomes and bounded separating-bundle search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import updatesupport as us
from updatesupport.artifacts import ReportArtifactMixin

from ._planning import iter_subsets
from ._explanation_comparison import (
    DisclosureExplanationComparison,
    _LIMITATIONS,
    _append,
    _escape,
    _name,
    _names,
    _policies,
    compare_disclosure_explanations,
)


@dataclass(frozen=True)
class DisclosureEvidenceRequest:
    """One hypothetical measurement/review outcome, with explicit dependencies.

    ``mapping_names`` selects mappings retained by a hypothetical scope review;
    excluded mappings are not relabeled infeasible. ``requires`` names other
    request outcomes that must accompany this one in a valid bundle.
    """

    name: str
    description: str
    constraints: Sequence[us.NamedLinearConstraint] = ()
    mapping_names: Sequence[str] | None = None
    requires: Sequence[str] = ()

    def __post_init__(self):
        if not _name(self.name) or not _name(self.description):
            raise ValueError("requests need a name and description")
        rows = _policies(self.name, self.constraints)
        mappings = (
            None
            if self.mapping_names is None
            else _names(self.mapping_names, "mapping names")
        )
        dependencies = _names(self.requires, "dependency names")
        if mappings == () or (not rows and mappings is None):
            raise ValueError(
                "a request needs constraints or a nonempty mapping selection"
            )
        if self.name in dependencies:
            raise ValueError("a request cannot require itself")
        object.__setattr__(self, "constraints", rows)
        object.__setattr__(self, "mapping_names", mappings)
        object.__setattr__(self, "requires", dependencies)

    def as_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "constraints": [c.as_dict() for c in self.constraints],
            "mapping_names": None
            if self.mapping_names is None
            else list(self.mapping_names),
            "requires": list(self.requires),
            "evidence_role": "hypothetical_outcome",
        }


@dataclass(frozen=True)
class DisclosureEvidenceBundle:
    requests: tuple[str, ...]
    status: str
    sufficient: bool
    retained_mappings: tuple[str, ...]
    comparison: DisclosureExplanationComparison | None

    def as_dict(self):
        return {
            "requests": list(self.requests),
            "status": self.status,
            "sufficient": self.sufficient,
            "retained_mappings": list(self.retained_mappings),
            "mapping_checks": []
            if self.comparison is None
            else [
                {"mapping": c.mapping, "status": c.status}
                for c in self.comparison.mapping_checks
            ],
            "cases": []
            if self.comparison is None
            else [
                {"mapping": c.mapping, "explanation": c.explanation, "status": c.status}
                for c in self.comparison.cases
            ],
            "comparison": None
            if self.comparison is None
            else self.comparison.as_dict(),
        }


@dataclass(frozen=True)
class DisclosureEvidenceOutcome:
    """Compatibility after selected hypothetical evidence, without a goal.

    ``compatible`` means at least one declared explanation survives in every
    source-compatible retained mapping. It never identifies an actual cause.
    Inspect the comparison for individual explanations and target bounds.
    """

    requests: tuple[str, ...]
    status: str
    retained_mappings: tuple[str, ...]
    comparison: DisclosureExplanationComparison | None
    request_definitions: tuple[DisclosureEvidenceRequest, ...] = ()

    def as_dict(self):
        return {
            "requests": list(self.requests),
            "request_definitions": [r.as_dict() for r in self.request_definitions],
            "status": self.status,
            "retained_mappings": list(self.retained_mappings),
            "basis": "Selected hypothetical evidence outcomes; not observed company evidence",
            "comparison": None
            if self.comparison is None
            else self.comparison.as_dict(),
        }

    def to_json(self, **kwargs):
        return us.report_to_json(self, **kwargs)

    def to_markdown(self):
        lines = [
            "# Hypothetical evidence outcome",
            "",
            "Selected: " + ", ".join(map(_escape, self.requests or ("None",))),
            "",
            "Status: " + self.status,
            "",
        ]
        if self.comparison is not None:
            lines.append(self.comparison.to_markdown())
        return "\n".join(lines)


def _request_catalog(comparison, requests):
    requests = tuple(requests)
    if any(not isinstance(r, DisclosureEvidenceRequest) for r in requests) or len(
        {r.name for r in requests}
    ) != len(requests):
        raise ValueError("requests must have unique names")
    known = {r.name: r for r in requests}
    mapping_names = {m.name for m in comparison.mappings}
    for r in requests:
        if not set(r.requires) <= known.keys() or (
            r.mapping_names is not None and not set(r.mapping_names) <= mapping_names
        ):
            raise ValueError("unknown request dependency or mapping name")
    visiting, visited = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("request dependencies must be acyclic")
        if name not in visited:
            visiting.add(name)
            for parent in known[name].requires:
                visit(parent)
            visiting.remove(name)
            visited.add(name)

    for name in known:
        visit(name)
    return requests


def _evaluate_outcome(comparison, selected, *, targets, include_conflicts):
    keys = tuple(r.name for r in selected)
    allowed = {m.name for m in comparison.mappings}
    for r in selected:
        if r.mapping_names is not None:
            allowed &= set(r.mapping_names)
    retained = tuple(m for m in comparison.mappings if m.name in allowed)
    analysis = None
    if any(not set(r.requires) <= set(keys) for r in selected):
        status = "missing_dependencies"
    elif not retained:
        status = "no_retained_mappings"
    else:
        rows = list(comparison.source_problem.constraints)
        active = _append(rows, selected, "request")
        tier = next(
            s
            for s in comparison.source_problem.scenarios
            if s.name == comparison.base_tier
        )
        problem = replace(
            comparison.source_problem,
            constraints=rows,
            scenarios=(
                replace(
                    tier,
                    constraints=(
                        *tier.constraints,
                        *(n for group in active for n in group),
                    ),
                ),
            ),
        )
        analysis = compare_disclosure_explanations(
            problem,
            base_tier=tier.name,
            mappings=retained,
            explanations=comparison.explanations,
            evidence=comparison.evidence,
            targets=targets,
            include_conflicts=include_conflicts,
        )
        feasible = {
            c.mapping for c in analysis.mapping_checks if c.status == "feasible"
        }
        cells = [c for c in analysis.cases if c.mapping in feasible]
        if any(c.status == "undetermined" for c in (*analysis.mapping_checks, *cells)):
            status = "undetermined"
        elif not feasible:
            status = "inconsistent_with_all_mappings"
        elif any(
            not any(c.mapping == m and c.status == "feasible" for c in cells)
            for m in feasible
        ):
            status = "uncovered_mapping"
        else:
            status = "compatible"
    return DisclosureEvidenceOutcome(
        keys, status, tuple(m.name for m in retained), analysis, tuple(selected)
    )


def evaluate_disclosure_evidence(
    comparison: DisclosureExplanationComparison,
    requests: Sequence[DisclosureEvidenceRequest],
    *,
    selected: Sequence[str] | None = None,
    targets: Sequence[str] | None = (),
    include_conflicts: bool = False,
) -> DisclosureEvidenceOutcome:
    """Show what selected hypothetical answers leave standing, without a goal.

    Supply a validated catalog and select outcome names; ``selected=None`` uses
    all entries and ``selected=()`` evaluates no added evidence. Dependency and
    mapping guards match the evidence planner. ``targets=None`` bounds all
    financial targets; the default solves compatibility only. Mutually
    inconsistent outcomes remain inconsistent, not proof of an explanation.
    """
    requests = _request_catalog(comparison, requests)
    if type(include_conflicts) is not bool:
        raise ValueError("include_conflicts must be a boolean")
    if targets is not None:
        targets = _names(targets, "targets")
        if not set(targets) <= {t.name for t in comparison.source_problem.targets}:
            raise ValueError("targets must name existing financial targets")
    names = (
        tuple(r.name for r in requests)
        if selected is None
        else _names(selected, "selected")
    )
    if not set(names) <= {r.name for r in requests}:
        raise ValueError("selected must name existing evidence outcomes")
    return _evaluate_outcome(
        comparison,
        tuple(r for r in requests if r.name in names),
        targets=targets,
        include_conflicts=include_conflicts,
    )


@dataclass(frozen=True)
class DisclosureEvidencePlan(ReportArtifactMixin):
    comparison: DisclosureExplanationComparison
    requests: tuple[DisclosureEvidenceRequest, ...]
    eliminate: tuple[str, ...]
    bundles: tuple[DisclosureEvidenceBundle, ...]
    minimal_bundles: tuple[DisclosureEvidenceBundle, ...]

    def as_dict(self):
        return {
            "comparison": self.comparison.as_dict(),
            "eliminate": list(self.eliminate),
            "requests": [r.as_dict() for r in self.requests],
            "bundles": [b.as_dict() for b in self.bundles],
            "minimal_bundles": [list(b.requests) for b in self.minimal_bundles],
            "basis": "Hypothetical outcomes, inclusion-minimal within this finite catalog",
            "limitations": list(_LIMITATIONS),
        }

    def to_tables(self):
        minimal = {b.requests for b in self.minimal_bundles}
        return {
            "evidence_bundles": tuple(
                {
                    "requests": ", ".join(b.requests),
                    "status": b.status,
                    "sufficient": b.sufficient,
                    "minimal": b.requests in minimal,
                }
                for b in self.bundles
            )
        }

    def to_markdown(self):
        lines = [
            "# Evidence outcomes separating financial explanations",
            "",
            "Eliminate from the declared catalog: "
            + ", ".join(map(_escape, self.eliminate)),
            "",
            "| Hypothetical outcomes | Status | Inclusion-minimal sufficient |",
            "|---|---|---|",
        ]
        minimal = {b.requests for b in self.minimal_bundles}
        lines += [
            f"| {_escape(', '.join(b.requests) or 'No additional evidence')} | {b.status} | {b.requests in minimal} |"
            for b in self.bundles
        ]
        lines += ["", "## Limitations", "", *["- " + x for x in _LIMITATIONS]]
        return "\n".join(lines)


def plan_disclosure_evidence(
    comparison: DisclosureExplanationComparison,
    requests: Sequence[DisclosureEvidenceRequest],
    *,
    eliminate: Sequence[str],
    max_bundles: int = 64,
) -> DisclosureEvidencePlan:
    """Find inclusion-minimal outcome bundles eliminating selected explanations.

    Each bundle must leave a compatible explanation in every source-compatible
    retained mapping. Contradictory evidence, uncovered mappings, solver failures
    and an empty retained mapping set never count as separation. Search includes
    the empty bundle and is bounded explicitly because catalog search is exponential.
    """
    requests, eliminate = (
        _request_catalog(comparison, requests),
        _names(eliminate, "eliminate"),
    )
    if (
        not eliminate
        or len(set(eliminate)) != len(eliminate)
        or not set(eliminate) <= {e.name for e in comparison.explanations}
    ):
        raise ValueError("eliminate must name unique existing explanations")
    if (
        isinstance(max_bundles, bool)
        or not isinstance(max_bundles, int)
        or max_bundles < 1
        or 2 ** len(requests) > max_bundles
    ):
        raise ValueError(
            "request catalog exceeds max_bundles; bound the catalog or explicitly raise the limit"
        )
    bundles = []
    for selected in iter_subsets(requests):
        outcome = _evaluate_outcome(
            comparison, selected, targets=(), include_conflicts=False
        )
        status, analysis = outcome.status, outcome.comparison
        if status == "compatible":
            feasible = {
                c.mapping for c in analysis.mapping_checks if c.status == "feasible"
            }
            cells = [c for c in analysis.cases if c.mapping in feasible]
            if any(
                c.status == "feasible" and c.explanation in eliminate for c in cells
            ):
                status = "not_separated"
            else:
                status = "separated"
        bundles.append(
            DisclosureEvidenceBundle(
                outcome.requests,
                status,
                status == "separated",
                outcome.retained_mappings,
                analysis,
            )
        )
    minimal = tuple(
        b
        for b in bundles
        if b.sufficient
        and not any(q.sufficient and set(q.requests) < set(b.requests) for q in bundles)
    )
    return DisclosureEvidencePlan(
        comparison, requests, eliminate, tuple(bundles), minimal
    )
