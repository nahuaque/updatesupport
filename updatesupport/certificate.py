"""Representation-stability certification over frontier search results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .frontier import (
    PublicRepresentationCandidate,
    PublicRepresentationFrontier,
    public_representation_frontier,
)


@dataclass(frozen=True)
class RepresentationStabilityCertificate:
    """Review-ready decision artifact for a public reporting representation."""

    frontier: PublicRepresentationFrontier
    selected_candidate: PublicRepresentationCandidate | None
    status: str
    exact_required: bool = True
    title: str = "Representation Stability Certificate"
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "inconclusive"}:
            raise ValueError("status must be 'pass', 'fail', or 'inconclusive'")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def passed(self) -> bool:
        """Whether the selected representation is certified under the spec."""

        return self.status == "pass"

    @property
    def inconclusive(self) -> bool:
        """Whether the run found evidence but cannot certify it as conclusive."""

        return self.status == "inconclusive"

    @property
    def failed(self) -> bool:
        """Whether no evaluated representation satisfied the certificate."""

        return self.status == "fail"

    @property
    def search_exact(self) -> bool:
        """Whether the frontier searched exact endpoint values throughout."""

        return (
            self.frontier.search_trace is not None and self.frontier.search_trace.exact
        )

    @property
    def certified_candidate(self) -> PublicRepresentationCandidate | None:
        """Return the selected candidate only when the certificate passed."""

        if not self.passed:
            return None
        return self.selected_candidate

    @property
    def best_evaluated_candidate(self) -> PublicRepresentationCandidate | None:
        """Most stable evaluated representation, regardless of certification."""

        if not self.frontier.candidates:
            return None
        return min(
            self.frontier.candidates,
            key=lambda row: (
                row.max_ambiguity,
                row.mean_ambiguity,
                row.public_cells,
                row.added_column_count,
                row.added_columns,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        selected = None
        if self.selected_candidate is not None:
            selected = self.selected_candidate.as_dict()
        certified = None
        if self.certified_candidate is not None:
            certified = self.certified_candidate.as_dict()
        best = None
        if self.best_evaluated_candidate is not None:
            best = self.best_evaluated_candidate.as_dict()
        return {
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "inconclusive": self.inconclusive,
            "failed": self.failed,
            "exact_required": self.exact_required,
            "search_exact": self.search_exact,
            "ambiguity_limit": self.frontier.ambiguity_limit,
            "bucket_budget": self.frontier.bucket_budget,
            "base_public": self.frontier.base_public,
            "hidden_columns": self.frontier.hidden_columns,
            "candidate_refinements": self.frontier.candidate_refinements,
            "certified_candidate": certified,
            "selected_candidate": selected,
            "best_evaluated_candidate": best,
            "reasons": self.reasons,
            "limitations": self.limitations,
            "screening": None
            if self.frontier.screening is None
            else self.frontier.screening.as_dict(),
            "frontier": self.frontier.as_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Certification status: **{self.status.upper()}**",
            f"- Exact search required: {'yes' if self.exact_required else 'no'}",
            f"- Search guarantee: {_search_guarantee(self.frontier)}",
            f"- Ambiguity limit: {_format_optional_float(self.frontier.ambiguity_limit)}",
        ]
        if self.frontier.bucket_budget is not None:
            lines.append(f"- Public-cell budget: {self.frontier.bucket_budget}")

        if self.passed and self.selected_candidate is not None:
            lines.extend(
                [
                    "- Certified representation: "
                    f"`{_column_label(self.selected_candidate.public_columns)}`",
                    "- Added public columns: "
                    f"`{_column_label(self.selected_candidate.added_columns)}`",
                    "- Worst certified ambiguity: "
                    f"{self.selected_candidate.max_ambiguity:.4f}",
                    f"- Public cells: {self.selected_candidate.public_cells}",
                ]
            )
        elif self.inconclusive and self.selected_candidate is not None:
            lines.extend(
                [
                    "- Provisional representation: "
                    f"`{_column_label(self.selected_candidate.public_columns)}`",
                    "- Provisional worst ambiguity: "
                    f"{self.selected_candidate.max_ambiguity:.4f}",
                ]
            )
        else:
            lines.append(
                "- Certified representation: none of the evaluated candidates "
                "satisfied the certificate requirements."
            )

        if self.reasons:
            lines.extend(["", "## Decision", ""])
            lines.extend(f"- {reason}" for reason in self.reasons)

        lines.extend(["", "## Certification Basis", ""])
        lines.extend(
            [
                f"- Base public columns: `{_column_label(self.frontier.base_public)}`",
                f"- Hidden columns: `{_column_label(self.frontier.hidden_columns)}`",
                "- Candidate refinements: "
                f"`{_column_label(self.frontier.candidate_refinements)}`",
                f"- Evaluated representations: {len(self.frontier.candidates)}",
                f"- Pareto frontier representations: {len(self.frontier.frontier)}",
                f"- Hidden-set scenarios: {len(self.frontier.hidden_sets)}",
                "- Minimum hidden-cell weights: "
                f"{', '.join(f'{value:g}' for value in self.frontier.min_cell_weights)}",
            ]
        )
        if self.frontier.search_trace is not None:
            trace = self.frontier.search_trace
            lines.extend(
                [
                    f"- Search mode: {trace.search}",
                    "- Search evaluations: "
                    f"{trace.evaluated_candidates}/{trace.candidate_space_size}",
                    f"- Stress-test scenarios per representation: {trace.scenario_count}",
                    f"- Search stopping reason: {trace.stopping_reason}",
                ]
            )
        if self.frontier.screening is not None:
            lines.extend(_screening_summary_markdown(self.frontier.screening))

        selected = self.selected_candidate or self.best_evaluated_candidate
        if selected is not None:
            heading = (
                "Certified Scenario Evidence"
                if self.passed
                else "Best Available Scenario Evidence"
            )
            lines.extend(["", f"## {heading}", ""])
            lines.extend(_scenario_table(selected))

        if self.limitations:
            lines.extend(["", "## Limitations", ""])
            lines.extend(f"- {limitation}" for limitation in self.limitations)

        return "\n".join(lines)

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import report_dataframes

        return report_dataframes(self)


def certify_public_representation(
    data: Any,
    *,
    ambiguity_limit: float,
    exact_required: bool = True,
    require_exact: bool | None = None,
    screening_backend: str | None = None,
    screening_exact_fallback: bool = True,
    title: str = "Representation Stability Certificate",
    frontier_title: str = "Public Representation Frontier Evidence",
    **frontier_kwargs: Any,
) -> RepresentationStabilityCertificate:
    """Certify a public representation against a frontier stress-test search.

    The certificate passes when the frontier finds an evaluated representation
    whose worst-case ambiguity is no larger than ``ambiguity_limit`` and whose
    public-cell count is within ``bucket_budget`` when a budget is supplied. By
    default, heuristic frontier searches are marked inconclusive rather than
    passed; set ``exact_required=False`` to allow heuristic certificates.
    """

    if ambiguity_limit is None:
        raise ValueError("ambiguity_limit is required for certification")
    if ambiguity_limit < 0:
        raise ValueError("ambiguity_limit must be non-negative")
    if require_exact is not None:
        exact_required = require_exact
    frontier = public_representation_frontier(
        data,
        ambiguity_limit=ambiguity_limit,
        screening_backend=screening_backend,
        screening_exact_fallback=screening_exact_fallback,
        title=frontier_title,
        **frontier_kwargs,
    )
    selected = _select_candidate_for_certificate(frontier)
    status, reasons = _certificate_status(
        frontier,
        selected,
        exact_required=exact_required,
    )
    return RepresentationStabilityCertificate(
        frontier=frontier,
        selected_candidate=selected,
        status=status,
        exact_required=exact_required,
        title=title,
        reasons=reasons,
        limitations=_certificate_limitations(frontier),
    )


def _select_candidate_for_certificate(
    frontier: PublicRepresentationFrontier,
) -> PublicRepresentationCandidate | None:
    if frontier.ambiguity_limit is None:
        raise ValueError("frontier must have an ambiguity_limit to certify")
    stable = [
        row
        for row in frontier.candidates
        if row.max_ambiguity <= frontier.ambiguity_limit
        and (
            frontier.bucket_budget is None or row.public_cells <= frontier.bucket_budget
        )
    ]
    if not stable:
        return None
    return min(
        stable,
        key=lambda row: (
            row.public_cells,
            row.added_column_count,
            row.max_ambiguity,
            row.added_columns,
        ),
    )


def _certificate_status(
    frontier: PublicRepresentationFrontier,
    selected: PublicRepresentationCandidate | None,
    *,
    exact_required: bool,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    certifiable_search = _certifiable_search_coverage(frontier)
    if exact_required and not certifiable_search:
        reasons.append(
            "The frontier search was heuristic or did not otherwise certify "
            "the full requested candidate space under the requested exactness "
            "standard."
        )
        if selected is not None:
            reasons.append(
                "A stable evaluated representation was found, but it is "
                "reported as provisional because exact search was required."
            )
        else:
            reasons.append(
                "No stable evaluated representation was found, but an unevaluated "
                "candidate may still satisfy the certificate."
            )
        return "inconclusive", tuple(reasons)
    if selected is None:
        limit = frontier.ambiguity_limit
        budget = frontier.bucket_budget
        reasons.append(
            "No evaluated representation satisfied the ambiguity limit"
            + (f" ({limit:.4f})" if limit is not None else "")
            + (" and public-cell budget" if budget is not None else "")
            + "."
        )
        best = frontier.best_under_bucket_budget() or _best_candidate(frontier)
        if best is not None:
            reasons.append(
                "The best evaluated alternative had "
                f"{best.public_cells} public cells and max ambiguity "
                f"{best.max_ambiguity:.4f}."
            )
        return "fail", tuple(reasons)
    reasons.append(
        "The selected representation satisfied every evaluated stress-test "
        "scenario under the declared ambiguity limit."
    )
    if frontier.bucket_budget is not None:
        reasons.append(
            "The selected representation also satisfied the declared "
            "public-cell budget."
        )
    if frontier.search_trace is not None and frontier.search_trace.exact:
        reasons.append(
            "The search was exhaustive over the declared candidate space, so the "
            "certificate covers every requested candidate within the search bounds."
        )
    elif _conservative_screening_covers_exhaustive_search(frontier):
        reasons.append(
            "The candidate search was exhaustive over the declared candidate "
            "space. Conservative screened endpoint bounds were sufficient to "
            "certify the selected representation under the ambiguity limit."
        )
    else:
        reasons.append(
            "The search was heuristic; certification is limited to evaluated "
            "candidates because exact search was not required."
        )
    return "pass", tuple(reasons)


def _certificate_limitations(
    frontier: PublicRepresentationFrontier,
) -> tuple[str, ...]:
    limitations = [
        "This is not a confidence interval and does not include sampling, model, "
        "or survey-design uncertainty.",
        "The certificate is conditional on the retained support, hidden columns, "
        "minimum-cell filtering, target compilation, and declared Q stress tests.",
        "The certificate does not cover unseen hidden cells or future support "
        "drift outside the retained state space.",
    ]
    if frontier.search_trace is not None and not frontier.search_trace.exact:
        if frontier.search_trace.search in {"mip", "mip_oracle", "mip_minimum"}:
            limitations.append(
                "The MIP-backed search did not provide an exact guarantee for "
                "the full certificate constraints; unevaluated representations "
                "may still matter for reporting-only budgets, proxy objectives, "
                "or unsupported objectives."
            )
        else:
            limitations.append(
                "The frontier search was heuristic; unevaluated candidate "
                "representations may dominate the selected representation."
            )
    if frontier.screened_refinements:
        limitations.append(
            "Some requested refinement columns were screened out before search; "
            "see screened-refinement outputs for details."
        )
    return tuple(limitations)


def _best_candidate(
    frontier: PublicRepresentationFrontier,
) -> PublicRepresentationCandidate | None:
    if not frontier.candidates:
        return None
    return min(
        frontier.candidates,
        key=lambda row: (
            row.max_ambiguity,
            row.mean_ambiguity,
            row.public_cells,
            row.added_column_count,
            row.added_columns,
        ),
    )


def _search_guarantee(frontier: PublicRepresentationFrontier) -> str:
    if frontier.search_trace is None:
        return "unknown"
    if frontier.search_trace.exact:
        if frontier.search_trace.search in {"mip", "mip_oracle", "mip_minimum"}:
            return frontier.search_trace.optimization_guarantee or (
                "MIP-exact over the declared discrete search objective"
            )
        return "exhaustive over the declared candidate space"
    if _conservative_screening_covers_exhaustive_search(frontier):
        return (
            "exhaustive over candidate representations with conservative "
            "screened endpoint certificates"
        )
    if frontier.search_trace.search in {"mip", "mip_oracle", "mip_minimum"}:
        return frontier.search_trace.optimization_guarantee or (
            "MIP-backed search without a full certificate guarantee"
        )
    return "heuristic over evaluated candidates only"


def _certifiable_search_coverage(frontier: PublicRepresentationFrontier) -> bool:
    return (
        frontier.search_trace is not None
        and (
            frontier.search_trace.exact
            or _conservative_screening_covers_exhaustive_search(frontier)
        )
    )


def _conservative_screening_covers_exhaustive_search(
    frontier: PublicRepresentationFrontier,
) -> bool:
    trace = frontier.search_trace
    screening = frontier.screening
    if trace is None or screening is None:
        return False
    return (
        trace.search == "exhaustive"
        and trace.stopping_reason == "completed"
        and trace.evaluated_candidates >= trace.candidate_space_size
        and screening.certified_count == screening.endpoint_count
    )


def _column_label(columns: tuple[str, ...]) -> str:
    if not columns:
        return "none"
    return " x ".join(columns)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "not supplied"
    return f"{value:.4f}"


def _scenario_table(candidate: PublicRepresentationCandidate) -> list[str]:
    lines = [
        "| scenario | Q | public cells | hidden cells | observed | lower | upper | ambiguity | public adequate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate.scenarios:
        lines.append(
            "| "
            f"{row.scenario} | {row.q_name} | {row.public_cells} | "
            f"{row.hidden_cells} | {row.observed_value:.4f} | "
            f"{row.lower:.4f} | {row.upper:.4f} | {row.ambiguity:.4f} | "
            f"{'yes' if row.public_adequate else 'no'} |"
        )
    return lines


def _screening_summary_markdown(screening: Any) -> list[str]:
    return [
        f"- Screening backend: {screening.backend}",
        "- Screening endpoints certified: "
        f"{screening.certified_count}/{screening.endpoint_count}",
        f"- Conservative endpoints used: {screening.conservative_endpoint_count}",
        f"- Exact fallback enabled: {'yes' if screening.exact_fallback else 'no'}",
        f"- Exact fallback solves run: {screening.exact_solve_count}",
        f"- Exact endpoint solves avoided: {screening.exact_solve_avoided_count}",
    ]


__all__ = [
    "RepresentationStabilityCertificate",
    "certify_public_representation",
]
