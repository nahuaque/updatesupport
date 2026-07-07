"""Experimental optional integration with the ``residopt`` compiler.

This module is intentionally narrow. It exposes a first endpoint compiler for
L2-budget hidden-composition stress tests by mapping the fixed-public-law
subspace to a residopt ellipsoidal support atom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from math import isclose
from typing import Any, Hashable

import numpy as np

from .artifacts import ReportArtifactMixin
from .data import GroupedProblem, TabularTarget, from_dataframe
from .presets import QPreset
from .problem import FiniteProblem
from .targets import LinearTarget, UncertainLinearTarget


_RESIDOPT_L2_NOTES = (
    "Public-law equality is preserved by projecting shifts into the "
    "nullspace of the public-incidence matrix.",
    "Hidden-cell nonnegativity is not enforced in this experimental adapter; "
    "the interval is conservative for the exact simplex-constrained endpoint.",
    "The residopt certificate is exact for the compiled ellipsoidal support "
    "atom, not for every constraint in the original updatesupport Q set.",
)


@dataclass(frozen=True)
class ResidOptAvailability:
    """Availability status for the optional ``residopt`` backend."""

    available: bool
    version: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ResidOptEndpointCertificate:
    """Certificate for one endpoint compiled through ``residopt``."""

    endpoint: str
    label: str
    template: str | None
    output: str | None
    strategy: str | None
    solver_status: str | None
    value: float
    exact_for_compiled_support: bool
    exact_for_updatesupport_q: bool
    conservative_for_updatesupport_q: bool
    reason: str
    residopt_diagnostics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "label": self.label,
            "template": self.template,
            "output": self.output,
            "strategy": self.strategy,
            "solver_status": self.solver_status,
            "value": self.value,
            "exact_for_compiled_support": self.exact_for_compiled_support,
            "exact_for_updatesupport_q": self.exact_for_updatesupport_q,
            "conservative_for_updatesupport_q": self.conservative_for_updatesupport_q,
            "reason": self.reason,
            "residopt_diagnostics": list(self.residopt_diagnostics),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ResidOptEndpointReport(ReportArtifactMixin):
    """Compiled endpoint interval returned by the experimental residopt backend."""

    title: str
    observed_value: float
    lower: float
    upper: float
    lower_support_value: float
    upper_support_value: float
    radius: float
    q_name: str
    target_name: str
    target_description: str
    state_count: int
    public_cell_count: int
    nullspace_dimension: int
    public_columns: tuple[str, ...] = ()
    hidden_columns: tuple[str, ...] = ()
    lower_certificate: ResidOptEndpointCertificate | None = None
    upper_certificate: ResidOptEndpointCertificate | None = None
    compiled_templates_built: int = 0
    support_solves: int = 0
    notes: tuple[str, ...] = ()
    backend: str = "residopt"

    @property
    def ambiguity(self) -> float:
        return self.upper - self.lower

    @property
    def exact_for_updatesupport_q(self) -> bool:
        certificates = [
            certificate
            for certificate in (self.lower_certificate, self.upper_certificate)
            if certificate is not None
        ]
        return bool(certificates) and all(
            certificate.exact_for_updatesupport_q for certificate in certificates
        )

    @property
    def conservative_for_updatesupport_q(self) -> bool:
        certificates = [
            certificate
            for certificate in (self.lower_certificate, self.upper_certificate)
            if certificate is not None
        ]
        return bool(certificates) and all(
            certificate.conservative_for_updatesupport_q for certificate in certificates
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "backend": self.backend,
            "q_name": self.q_name,
            "radius": self.radius,
            "target_name": self.target_name,
            "target_description": self.target_description,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "lower_support_value": self.lower_support_value,
            "upper_support_value": self.upper_support_value,
            "compiled_templates_built": self.compiled_templates_built,
            "support_solves": self.support_solves,
            "state_count": self.state_count,
            "public_cell_count": self.public_cell_count,
            "nullspace_dimension": self.nullspace_dimension,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "exact_for_updatesupport_q": self.exact_for_updatesupport_q,
            "conservative_for_updatesupport_q": self.conservative_for_updatesupport_q,
            "lower_certificate": None
            if self.lower_certificate is None
            else self.lower_certificate.as_dict(),
            "upper_certificate": None
            if self.upper_certificate is None
            else self.upper_certificate.as_dict(),
            "notes": self.notes,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        certificates = tuple(
            certificate.as_dict()
            for certificate in (self.lower_certificate, self.upper_certificate)
            if certificate is not None
        )
        return {
            "summary": (
                {
                    "title": self.title,
                    "backend": self.backend,
                    "q_name": self.q_name,
                    "radius": self.radius,
                    "target_name": self.target_name,
                    "observed_value": self.observed_value,
                    "lower": self.lower,
                    "upper": self.upper,
                    "ambiguity": self.ambiguity,
                    "compiled_templates_built": self.compiled_templates_built,
                    "support_solves": self.support_solves,
                    "state_count": self.state_count,
                    "public_cell_count": self.public_cell_count,
                    "nullspace_dimension": self.nullspace_dimension,
                    "exact_for_updatesupport_q": self.exact_for_updatesupport_q,
                    "conservative_for_updatesupport_q": (
                        self.conservative_for_updatesupport_q
                    ),
                },
            ),
            "certificates": certificates,
            "notes": tuple({"note": note} for note in self.notes),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {_markdown_escape(self.title)}",
            "",
            (
                f"Backend: `{self.backend}`. Q preset: `{self.q_name}` "
                f"with L2 radius `{_format_float(self.radius)}`."
            ),
            (
                f"Compiled templates built for this call: "
                f"`{self.compiled_templates_built}`. Support solves: "
                f"`{self.support_solves}`."
            ),
            "",
            "| Observed | Lower | Upper | Ambiguity width |",
            "| ---: | ---: | ---: | ---: |",
            (
                f"| {_format_float(self.observed_value)} | "
                f"{_format_float(self.lower)} | "
                f"{_format_float(self.upper)} | "
                f"{_format_float(self.ambiguity)} |"
            ),
            "",
            "## Certificate Summary",
            "",
            "| Endpoint | Label | Template | Status | Exact for original Q | Conservative |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for certificate in (self.lower_certificate, self.upper_certificate):
            if certificate is None:
                continue
            lines.append(
                "| "
                f"{_markdown_escape(certificate.endpoint)} | "
                f"{_markdown_escape(certificate.label)} | "
                f"{_markdown_escape(str(certificate.template))} | "
                f"{_markdown_escape(str(certificate.solver_status))} | "
                f"{certificate.exact_for_updatesupport_q} | "
                f"{certificate.conservative_for_updatesupport_q} |"
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {_markdown_escape(note)}" for note in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class ResidOptRefinementScreenCandidate:
    """One candidate public representation screened through ``residopt``."""

    columns: tuple[str, ...]
    added_columns: tuple[str, ...]
    public_cells: int
    hidden_cells: int
    observed_value: float | None
    conservative_lower: float | None
    conservative_upper: float | None
    conservative_ambiguity: float | None
    exact_lower: float | None = None
    exact_upper: float | None = None
    exact_ambiguity: float | None = None
    conservative_reduction: float | None = None
    exact_reduction: float | None = None
    final_reduction: float | None = None
    final_reduction_percent: float | None = None
    certified_by_screen: bool = False
    exact_solve_run: bool = False
    exact_solve_avoided: bool = False
    compiler_cache_hit: bool = False
    compiled_templates_built: int = 0
    support_solves: int = 0
    status: str = "screen_only"
    reason: str = ""
    error: str | None = None

    @property
    def final_lower(self) -> float | None:
        return (
            self.exact_lower
            if self.exact_lower is not None
            else self.conservative_lower
        )

    @property
    def final_upper(self) -> float | None:
        return (
            self.exact_upper
            if self.exact_upper is not None
            else self.conservative_upper
        )

    @property
    def final_ambiguity(self) -> float | None:
        return (
            self.exact_ambiguity
            if self.exact_ambiguity is not None
            else self.conservative_ambiguity
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "added_columns": self.added_columns,
            "public_cells": self.public_cells,
            "hidden_cells": self.hidden_cells,
            "observed_value": self.observed_value,
            "conservative_lower": self.conservative_lower,
            "conservative_upper": self.conservative_upper,
            "conservative_ambiguity": self.conservative_ambiguity,
            "exact_lower": self.exact_lower,
            "exact_upper": self.exact_upper,
            "exact_ambiguity": self.exact_ambiguity,
            "final_lower": self.final_lower,
            "final_upper": self.final_upper,
            "final_ambiguity": self.final_ambiguity,
            "conservative_reduction": self.conservative_reduction,
            "exact_reduction": self.exact_reduction,
            "final_reduction": self.final_reduction,
            "final_reduction_percent": self.final_reduction_percent,
            "certified_by_screen": self.certified_by_screen,
            "exact_solve_run": self.exact_solve_run,
            "exact_solve_avoided": self.exact_solve_avoided,
            "compiler_cache_hit": self.compiler_cache_hit,
            "compiled_templates_built": self.compiled_templates_built,
            "support_solves": self.support_solves,
            "status": self.status,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class ResidOptRefinementScreenReport(ReportArtifactMixin):
    """Cached residopt screen over candidate public refinements."""

    title: str
    base_public: tuple[str, ...]
    hidden: tuple[str, ...]
    q_name: str
    radius: float
    ambiguity_limit: float | None
    exact_fallback: bool
    candidates: tuple[ResidOptRefinementScreenCandidate, ...]
    compiler_cache_size: int
    grouped_cache_size: int
    notes: tuple[str, ...] = ()
    backend: str = "residopt"

    @property
    def screened_count(self) -> int:
        return len(self.candidates)

    @property
    def certified_count(self) -> int:
        return sum(1 for row in self.candidates if row.certified_by_screen)

    @property
    def exact_solve_count(self) -> int:
        return sum(1 for row in self.candidates if row.exact_solve_run)

    @property
    def exact_solve_avoided_count(self) -> int:
        return sum(1 for row in self.candidates if row.exact_solve_avoided)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.candidates if row.error is not None)

    @property
    def support_solve_count(self) -> int:
        return sum(row.support_solves for row in self.candidates)

    @property
    def compiled_template_count(self) -> int:
        return sum(row.compiled_templates_built for row in self.candidates)

    @property
    def best_by_conservative_ambiguity(
        self,
    ) -> ResidOptRefinementScreenCandidate | None:
        rows = [
            row for row in self.candidates if row.conservative_ambiguity is not None
        ]
        if not rows:
            return None
        return min(
            rows,
            key=lambda row: (
                float("inf")
                if row.conservative_ambiguity is None
                else row.conservative_ambiguity
            ),
        )

    @property
    def best_by_final_ambiguity(self) -> ResidOptRefinementScreenCandidate | None:
        rows = [row for row in self.candidates if row.final_ambiguity is not None]
        if not rows:
            return None
        return min(
            rows,
            key=lambda row: (
                float("inf") if row.final_ambiguity is None else row.final_ambiguity
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        best_conservative = self.best_by_conservative_ambiguity
        best_final = self.best_by_final_ambiguity
        return {
            "title": self.title,
            "backend": self.backend,
            "base_public": self.base_public,
            "hidden": self.hidden,
            "q_name": self.q_name,
            "radius": self.radius,
            "ambiguity_limit": self.ambiguity_limit,
            "exact_fallback": self.exact_fallback,
            "screened_count": self.screened_count,
            "certified_count": self.certified_count,
            "exact_solve_count": self.exact_solve_count,
            "exact_solve_avoided_count": self.exact_solve_avoided_count,
            "error_count": self.error_count,
            "support_solve_count": self.support_solve_count,
            "compiled_template_count": self.compiled_template_count,
            "compiler_cache_size": self.compiler_cache_size,
            "grouped_cache_size": self.grouped_cache_size,
            "best_by_conservative_ambiguity": None
            if best_conservative is None
            else best_conservative.as_dict(),
            "best_by_final_ambiguity": None
            if best_final is None
            else best_final.as_dict(),
            "candidates": [row.as_dict() for row in self.candidates],
            "notes": self.notes,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    "title": self.title,
                    "backend": self.backend,
                    "q_name": self.q_name,
                    "radius": self.radius,
                    "ambiguity_limit": self.ambiguity_limit,
                    "exact_fallback": self.exact_fallback,
                    "screened_count": self.screened_count,
                    "certified_count": self.certified_count,
                    "exact_solve_count": self.exact_solve_count,
                    "exact_solve_avoided_count": self.exact_solve_avoided_count,
                    "error_count": self.error_count,
                    "support_solve_count": self.support_solve_count,
                    "compiled_template_count": self.compiled_template_count,
                    "compiler_cache_size": self.compiler_cache_size,
                    "grouped_cache_size": self.grouped_cache_size,
                },
            ),
            "candidates": tuple(row.as_dict() for row in self.candidates),
            "notes": tuple({"note": note} for note in self.notes),
        }

    def to_markdown(self) -> str:
        limit = (
            "none"
            if self.ambiguity_limit is None
            else _format_float(self.ambiguity_limit)
        )
        lines = [
            f"# {_markdown_escape(self.title)}",
            "",
            (
                f"Backend: `{self.backend}`. Q preset: `{self.q_name}` "
                f"with L2 radius `{_format_float(self.radius)}`. "
                f"Ambiguity limit: `{limit}`."
            ),
            (
                f"Screened `{self.screened_count}` representations; "
                f"certified `{self.certified_count}` by conservative screen; "
                f"ran `{self.exact_solve_count}` exact fallback solves; "
                f"avoided `{self.exact_solve_avoided_count}` exact solves."
            ),
            (
                f"Compiler cache size: `{self.compiler_cache_size}`. "
                f"Compiled templates built: `{self.compiled_template_count}`. "
                f"Support solves: `{self.support_solve_count}`."
            ),
            "",
            "| Public columns | Added columns | Public cells | Conservative ambiguity | Exact ambiguity | Final reduction | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in self.candidates:
            lines.append(
                "| "
                f"{_markdown_escape(', '.join(row.columns))} | "
                f"{_markdown_escape(', '.join(row.added_columns) or '(base)')} | "
                f"{row.public_cells} | "
                f"{_format_optional_float(row.conservative_ambiguity)} | "
                f"{_format_optional_float(row.exact_ambiguity)} | "
                f"{_format_optional_float(row.final_reduction)} | "
                f"{_markdown_escape(row.status)} |"
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {_markdown_escape(note)}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class ResidOptRefinementScreenContext:
    """Reusable cached screening context for refinement searches.

    The context materializes the input rows once, caches compiled
    :class:`GroupedProblem` objects by public representation, and caches one
    :class:`ResidOptL2EndpointCompiler` per compiled public representation.
    """

    data: Any
    public: Sequence[str]
    hidden: Sequence[str]
    target: TabularTarget
    weight: str | None = None
    min_cell_weight: float = 1.0
    q: Any = "saturated"
    q_radius: float | None = None
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None
    _data: Any = field(init=False, repr=False)
    public_columns: tuple[str, ...] = field(init=False)
    hidden_columns: tuple[str, ...] = field(init=False)
    grouped_cache: dict[tuple[str, ...], GroupedProblem] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    compiler_cache: dict[tuple[str, ...], ResidOptL2EndpointCompiler] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.public_columns = tuple(self.public)
        self.hidden_columns = tuple(self.hidden)
        self._data = _repeatable_data(self.data)
        if self.min_cell_weight < 0:
            raise ValueError("min_cell_weight must be non-negative")
        missing_public = [
            column
            for column in self.public_columns
            if column not in self.hidden_columns
        ]
        if missing_public:
            raise ValueError(
                f"public columns must also be hidden columns: {missing_public!r}"
            )

    def screen(
        self,
        *,
        candidate_refinements: Sequence[str] | None = None,
        max_added_columns: int = 1,
        include_base: bool = True,
        ambiguity_limit: float | None = None,
        exact_fallback: bool = True,
        top: int | None = None,
        title: str = "ResidOpt Refinement Screening Report",
    ) -> ResidOptRefinementScreenReport:
        """Screen candidate public refinements with cached residopt compilers."""

        if max_added_columns < 1:
            raise ValueError("max_added_columns must be at least 1")
        if top is not None and top < 0:
            raise ValueError("top must be non-negative")
        if ambiguity_limit is not None and ambiguity_limit < 0:
            raise ValueError("ambiguity_limit must be non-negative")

        candidate_refinements_tuple = (
            tuple(candidate_refinements)
            if candidate_refinements is not None
            else tuple(
                column
                for column in self.hidden_columns
                if column not in self.public_columns
            )
        )
        public_representations = _candidate_public_representations(
            public=self.public_columns,
            hidden=self.hidden_columns,
            candidate_refinements=candidate_refinements_tuple,
            max_added_columns=max_added_columns,
            include_base=include_base,
        )
        rows = tuple(
            self._screen_public_representation(
                columns,
                added_columns=tuple(
                    column for column in columns if column not in self.public_columns
                ),
                ambiguity_limit=ambiguity_limit,
                exact_fallback=exact_fallback,
            )
            for columns in public_representations
        )
        rows = _attach_reductions(rows)
        rows = _sort_refinement_screen_rows(rows, include_base=include_base, top=top)

        grouped = self._grouped(self.public_columns)
        radius = _require_l2_grouped(grouped).radius
        return ResidOptRefinementScreenReport(
            title=title,
            base_public=self.public_columns,
            hidden=self.hidden_columns,
            q_name="l2_budget",
            radius=float(radius),
            ambiguity_limit=ambiguity_limit,
            exact_fallback=exact_fallback,
            candidates=rows,
            compiler_cache_size=len(self.compiler_cache),
            grouped_cache_size=len(self.grouped_cache),
            notes=_RESIDOPT_L2_NOTES,
        )

    def evaluate_public_representation(
        self,
        added_columns: Sequence[str] = (),
        *,
        ambiguity_limit: float | None = None,
        exact_fallback: bool = True,
    ) -> ResidOptRefinementScreenCandidate:
        """Screen one public representation, reusing this context's caches."""

        ordered_added = tuple(dict.fromkeys(added_columns))
        unknown = [
            column
            for column in ordered_added
            if column not in self.hidden_columns or column in self.public_columns
        ]
        if unknown:
            raise ValueError(
                "added_columns must be hidden columns not already public; "
                f"invalid columns: {unknown!r}"
            )
        return self._screen_public_representation(
            self.public_columns + ordered_added,
            added_columns=ordered_added,
            ambiguity_limit=ambiguity_limit,
            exact_fallback=exact_fallback,
        )

    def _screen_public_representation(
        self,
        columns: tuple[str, ...],
        *,
        added_columns: tuple[str, ...],
        ambiguity_limit: float | None,
        exact_fallback: bool,
    ) -> ResidOptRefinementScreenCandidate:
        grouped = self._grouped(columns)
        public_cells = len(grouped.problem.public_values)
        hidden_cells = len(grouped.problem.states)
        cache_hit = columns in self.compiler_cache
        try:
            compiler = self._compiler(columns)
            screen = compiler.interval(
                title="ResidOpt Refinement Candidate Screen",
            )
            certified = (
                ambiguity_limit is not None
                and screen.ambiguity <= ambiguity_limit + grouped.problem.tol
            )
            if certified:
                return ResidOptRefinementScreenCandidate(
                    columns=columns,
                    added_columns=added_columns,
                    public_cells=public_cells,
                    hidden_cells=hidden_cells,
                    observed_value=screen.observed_value,
                    conservative_lower=screen.lower,
                    conservative_upper=screen.upper,
                    conservative_ambiguity=screen.ambiguity,
                    certified_by_screen=True,
                    exact_solve_avoided=True,
                    compiler_cache_hit=cache_hit,
                    compiled_templates_built=screen.compiled_templates_built,
                    support_solves=screen.support_solves,
                    status="screen_certified",
                    reason=(
                        "The conservative residopt interval is within the "
                        "requested ambiguity limit, so no exact fallback solve "
                        "is needed."
                    ),
                )
            if exact_fallback:
                exact = grouped.problem.global_transport_modulus()
                exact_pass = (
                    ambiguity_limit is not None
                    and exact.diameter <= ambiguity_limit + grouped.problem.tol
                )
                return ResidOptRefinementScreenCandidate(
                    columns=columns,
                    added_columns=added_columns,
                    public_cells=public_cells,
                    hidden_cells=hidden_cells,
                    observed_value=screen.observed_value,
                    conservative_lower=screen.lower,
                    conservative_upper=screen.upper,
                    conservative_ambiguity=screen.ambiguity,
                    exact_lower=exact.lower,
                    exact_upper=exact.upper,
                    exact_ambiguity=exact.diameter,
                    exact_solve_run=True,
                    compiler_cache_hit=cache_hit,
                    compiled_templates_built=screen.compiled_templates_built,
                    support_solves=screen.support_solves,
                    status=("fallback_exact_pass" if exact_pass else "fallback_exact"),
                    reason=(
                        "The conservative residopt interval was inconclusive, "
                        "so the exact updatesupport endpoint was solved."
                    ),
                )
            return ResidOptRefinementScreenCandidate(
                columns=columns,
                added_columns=added_columns,
                public_cells=public_cells,
                hidden_cells=hidden_cells,
                observed_value=screen.observed_value,
                conservative_lower=screen.lower,
                conservative_upper=screen.upper,
                conservative_ambiguity=screen.ambiguity,
                compiler_cache_hit=cache_hit,
                compiled_templates_built=screen.compiled_templates_built,
                support_solves=screen.support_solves,
                status="screen_only",
                reason=(
                    "The conservative residopt interval was computed without "
                    "running exact fallback."
                ),
            )
        except Exception as exc:
            if exact_fallback:
                try:
                    exact = grouped.problem.global_transport_modulus()
                except Exception as exact_exc:
                    return ResidOptRefinementScreenCandidate(
                        columns=columns,
                        added_columns=added_columns,
                        public_cells=public_cells,
                        hidden_cells=hidden_cells,
                        observed_value=None,
                        conservative_lower=None,
                        conservative_upper=None,
                        conservative_ambiguity=None,
                        compiler_cache_hit=cache_hit,
                        exact_solve_run=True,
                        status="error",
                        reason="Both the residopt screen and exact fallback failed.",
                        error=f"{type(exc).__name__}: {exc}; exact fallback: {type(exact_exc).__name__}: {exact_exc}",
                    )
                return ResidOptRefinementScreenCandidate(
                    columns=columns,
                    added_columns=added_columns,
                    public_cells=public_cells,
                    hidden_cells=hidden_cells,
                    observed_value=None,
                    conservative_lower=None,
                    conservative_upper=None,
                    conservative_ambiguity=None,
                    exact_lower=exact.lower,
                    exact_upper=exact.upper,
                    exact_ambiguity=exact.diameter,
                    exact_solve_run=True,
                    compiler_cache_hit=cache_hit,
                    status="screen_error_fallback_exact",
                    reason=(
                        "Residopt screening failed; the exact updatesupport "
                        "endpoint was solved."
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            return ResidOptRefinementScreenCandidate(
                columns=columns,
                added_columns=added_columns,
                public_cells=public_cells,
                hidden_cells=hidden_cells,
                observed_value=None,
                conservative_lower=None,
                conservative_upper=None,
                conservative_ambiguity=None,
                compiler_cache_hit=cache_hit,
                status="screen_error",
                reason="Residopt screening failed and exact fallback was disabled.",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _grouped(self, columns: tuple[str, ...]) -> GroupedProblem:
        if columns not in self.grouped_cache:
            self.grouped_cache[columns] = from_dataframe(
                self._data,
                public=columns,
                hidden=self.hidden_columns,
                target=self.target,
                weight=self.weight,
                min_cell_weight=self.min_cell_weight,
                q=self.q,
                q_radius=self.q_radius,
            )
            _require_l2_grouped(self.grouped_cache[columns])
        return self.grouped_cache[columns]

    def _compiler(self, columns: tuple[str, ...]) -> ResidOptL2EndpointCompiler:
        if columns not in self.compiler_cache:
            self.compiler_cache[columns] = ResidOptL2EndpointCompiler.from_grouped(
                self._grouped(columns),
                solver=self.solver,
                solver_options=self.solver_options,
            )
        return self.compiler_cache[columns]


def residopt_refinement_screen(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    candidate_refinements: Sequence[str] | None = None,
    weight: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    max_added_columns: int = 1,
    include_base: bool = True,
    ambiguity_limit: float | None = None,
    exact_fallback: bool = True,
    top: int | None = None,
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
    title: str = "ResidOpt Refinement Screening Report",
) -> ResidOptRefinementScreenReport:
    """Screen candidate public refinements with cached residopt compilers.

    This experimental helper is useful when many candidate public
    representations are evaluated under the same L2 hidden-composition stress
    test. The conservative screen can certify candidates whose ambiguity is
    already below ``ambiguity_limit`` and skip their exact CVXPY endpoint solve.
    """

    context = ResidOptRefinementScreenContext(
        data=data,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        min_cell_weight=min_cell_weight,
        q=q,
        q_radius=q_radius,
        solver=solver,
        solver_options=solver_options,
    )
    return context.screen(
        candidate_refinements=candidate_refinements,
        max_added_columns=max_added_columns,
        include_base=include_base,
        ambiguity_limit=ambiguity_limit,
        exact_fallback=exact_fallback,
        top=top,
        title=title,
    )


def residopt_available() -> ResidOptAvailability:
    """Return whether the optional ``residopt`` backend can be imported."""

    try:
        import_module("residopt")
    except ImportError:
        return ResidOptAvailability(
            available=False,
            reason=(
                "residopt is not importable. Install it with "
                "`pip install 'updatesupport[residopt]'` or "
                "`uv add 'updatesupport[residopt]'`."
            ),
        )
    try:
        installed_version = version("residopt")
    except PackageNotFoundError:
        installed_version = None
    return ResidOptAvailability(available=True, version=installed_version)


@dataclass
class ResidOptL2EndpointCompiler:
    """Reusable residopt endpoint compiler for fixed-public-law L2 stress tests.

    Create one compiler for a compiled :class:`GroupedProblem`, then call
    :meth:`interval` repeatedly for different linear directions. The compiler
    caches the public-law nullspace and a parameterized residopt support
    template, so repeated calls avoid rebuilding the SOCP.
    """

    problem: FiniteProblem
    observed_distribution: np.ndarray
    public_law: dict[Hashable, float]
    radius: float
    public_columns: tuple[str, ...] = ()
    hidden_columns: tuple[str, ...] = ()
    solver: str | None = None
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    nullspace: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))
    _support_template: "_ResidOptSupportTemplate | None" = field(
        default=None,
        init=False,
        repr=False,
    )
    compiled_template_count: int = field(default=0, init=False)
    support_solve_count: int = field(default=0, init=False)

    @classmethod
    def from_grouped(
        cls,
        grouped: GroupedProblem,
        *,
        observed_distribution: Mapping[Hashable, float] | Sequence[float] | None = None,
        public_law: Mapping[Hashable, float] | None = None,
        radius: float | None = None,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> "ResidOptL2EndpointCompiler":
        """Build a reusable compiler from a tabular ``GroupedProblem``."""

        base = _prepare_base_inputs(
            grouped,
            observed_distribution=observed_distribution,
            public_law=public_law,
            radius=radius,
            solver=solver,
            solver_options=solver_options,
        )
        return cls._from_base(base)

    @classmethod
    def from_problem(
        cls,
        problem: FiniteProblem,
        *,
        observed_distribution: Mapping[Hashable, float] | Sequence[float],
        public_law: Mapping[Hashable, float],
        radius: float,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> "ResidOptL2EndpointCompiler":
        """Build a reusable compiler from an explicit finite problem."""

        base = _prepare_base_inputs(
            problem,
            observed_distribution=observed_distribution,
            public_law=public_law,
            radius=radius,
            solver=solver,
            solver_options=solver_options,
        )
        return cls._from_base(base)

    @classmethod
    def _from_base(
        cls,
        base: "_PreparedResidOptBase",
    ) -> "ResidOptL2EndpointCompiler":
        incidence = _public_incidence_matrix(base.problem)
        nullspace = _orthonormal_nullspace(incidence, tol=base.problem.tol)
        return cls(
            problem=base.problem,
            observed_distribution=base.observed_distribution,
            public_law=base.public_law,
            radius=base.radius,
            public_columns=base.public_columns,
            hidden_columns=base.hidden_columns,
            solver=base.solver,
            solver_options=base.solver_options,
            nullspace=nullspace,
        )

    @property
    def state_count(self) -> int:
        return len(self.problem.states)

    @property
    def public_cell_count(self) -> int:
        return len(self.problem.public_values)

    @property
    def nullspace_dimension(self) -> int:
        return int(self.nullspace.shape[1])

    def interval(
        self,
        direction: Mapping[Hashable, float] | Sequence[float] | None = None,
        *,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
        title: str = "ResidOpt L2 Endpoint Compilation",
    ) -> ResidOptEndpointReport:
        """Evaluate an interval, reusing the cached support template when possible."""

        direction_vector = _coerce_direction(self.problem, direction)
        observed_value = float(np.dot(direction_vector, self.observed_distribution))
        projected_direction = _project_to_nullspace(self.nullspace, direction_vector)
        projected_negative_direction = _project_to_nullspace(
            self.nullspace,
            -direction_vector,
        )

        compiled_before = self.compiled_template_count
        solves_before = self.support_solve_count
        lower_compiled = self._solve_support(
            projected_negative_direction,
            endpoint="lower",
            solver=solver,
            solver_options=solver_options,
        )
        upper_compiled = self._solve_support(
            projected_direction,
            endpoint="upper",
            solver=solver,
            solver_options=solver_options,
        )
        target_contract = self.problem.target_contract
        return ResidOptEndpointReport(
            title=title,
            observed_value=observed_value,
            lower=observed_value - lower_compiled.value,
            upper=observed_value + upper_compiled.value,
            lower_support_value=lower_compiled.value,
            upper_support_value=upper_compiled.value,
            radius=self.radius,
            q_name="l2_budget",
            target_name=target_contract.name,
            target_description=target_contract.description,
            state_count=self.state_count,
            public_cell_count=self.public_cell_count,
            nullspace_dimension=self.nullspace_dimension,
            public_columns=self.public_columns,
            hidden_columns=self.hidden_columns,
            lower_certificate=lower_compiled.certificate,
            upper_certificate=upper_compiled.certificate,
            compiled_templates_built=(self.compiled_template_count - compiled_before),
            support_solves=self.support_solve_count - solves_before,
            notes=_RESIDOPT_L2_NOTES,
        )

    def _solve_support(
        self,
        projected_direction: np.ndarray,
        *,
        endpoint: str,
        solver: str | None,
        solver_options: Mapping[str, Any] | None,
    ) -> "_CompiledSupport":
        if (
            self.radius == 0.0
            or self.nullspace_dimension == 0
            or np.linalg.norm(projected_direction, ord=2) == 0.0
        ):
            return _CompiledSupport(
                value=0.0,
                certificate=_deterministic_certificate(
                    endpoint=endpoint,
                    value=0.0,
                    reason=(
                        "No support solve is needed because the projected "
                        "direction has zero admissible L2 variation."
                    ),
                ),
            )

        template = self._get_support_template()
        effective_solver = self.solver if solver is None else solver
        effective_solver_options = (
            self.solver_options if solver_options is None else dict(solver_options)
        )
        result = template.solve(
            projected_direction,
            endpoint=endpoint,
            solver=effective_solver,
            solver_options=effective_solver_options,
        )
        self.support_solve_count += 1
        return result

    def _get_support_template(self) -> "_ResidOptSupportTemplate":
        if self._support_template is None:
            self._support_template = _compile_residopt_support_template(
                dimension=self.nullspace_dimension,
                radius=self.radius,
            )
            self.compiled_template_count += 1
        return self._support_template


def residopt_l2_support_interval(
    grouped_or_problem: GroupedProblem | FiniteProblem,
    *,
    direction: Mapping[Hashable, float] | Sequence[float] | None = None,
    observed_distribution: Mapping[Hashable, float] | Sequence[float] | None = None,
    public_law: Mapping[Hashable, float] | None = None,
    radius: float | None = None,
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
    title: str = "ResidOpt L2 Endpoint Compilation",
) -> ResidOptEndpointReport:
    """Compile a conservative L2 endpoint interval through ``residopt``.

    The compiled problem fixes the public law by projecting hidden-distribution
    shifts into the public-incidence nullspace and constrains the shift by an
    L2 radius. The current slice drops hidden-cell nonnegativity constraints, so
    the returned interval is a conservative upper envelope for the exact
    simplex-constrained ``updatesupport`` endpoint.
    """

    compiler = ResidOptL2EndpointCompiler._from_base(
        _prepare_base_inputs(
            grouped_or_problem,
            observed_distribution=observed_distribution,
            public_law=public_law,
            radius=radius,
            solver=solver,
            solver_options=solver_options,
        )
    )
    return compiler.interval(direction=direction, title=title)


@dataclass(frozen=True)
class _PreparedResidOptBase:
    problem: FiniteProblem
    observed_distribution: np.ndarray
    public_law: dict[Hashable, float]
    radius: float
    solver: str | None
    solver_options: Mapping[str, Any]
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedResidOptInputs:
    problem: FiniteProblem
    direction: np.ndarray
    observed_distribution: np.ndarray
    public_law: dict[Hashable, float]
    radius: float
    solver: str | None
    solver_options: Mapping[str, Any]
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]


@dataclass(frozen=True)
class _CompiledSupport:
    value: float
    certificate: ResidOptEndpointCertificate


@dataclass
class _ResidOptSupportTemplate:
    dimension: int
    radius: float
    compiled: Any
    direction_parameter: Any
    solve_count: int = 0

    def solve(
        self,
        projected_direction: np.ndarray,
        *,
        endpoint: str,
        solver: str | None,
        solver_options: Mapping[str, Any],
    ) -> _CompiledSupport:
        self.direction_parameter.value = np.asarray(projected_direction, dtype=float)
        solve_kwargs = dict(solver_options)
        if solver is not None:
            solve_kwargs["solver"] = solver
        value = float(self.compiled.solve(**solve_kwargs))
        status = self.compiled.problem.status
        if status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"residopt endpoint solve failed with status {status!r}")
        self.solve_count += 1

        residopt_certificate = (
            self.compiled.certificates[0] if self.compiled.certificates else None
        )
        metadata = (
            {}
            if residopt_certificate is None
            else dict(residopt_certificate.metadata or {})
        )
        metadata["public_nullspace_dimension"] = self.dimension
        metadata["projected_direction_norm"] = float(
            np.linalg.norm(projected_direction, ord=2)
        )
        metadata["parameterized_template"] = True
        metadata["template_solve_count"] = self.solve_count
        reason = (
            "residopt certifies the parameterized ellipsoidal support atom "
            "exactly. updatesupport marks the endpoint conservative because "
            "this adapter relaxes away hidden-cell nonnegativity from the "
            "original Q set."
        )
        return _CompiledSupport(
            value=value,
            certificate=ResidOptEndpointCertificate(
                endpoint=endpoint,
                label=_enum_value(getattr(residopt_certificate, "label", "exact")),
                template=getattr(residopt_certificate, "template", None),
                output=getattr(residopt_certificate, "output", None),
                strategy=_enum_value(getattr(residopt_certificate, "strategy", None)),
                solver_status=status,
                value=value,
                exact_for_compiled_support=True,
                exact_for_updatesupport_q=False,
                conservative_for_updatesupport_q=True,
                reason=reason,
                residopt_diagnostics=tuple(
                    getattr(residopt_certificate, "diagnostics", ()) or ()
                ),
                metadata=metadata,
            ),
        )


def _prepare_inputs(
    grouped_or_problem: GroupedProblem | FiniteProblem,
    *,
    direction: Mapping[Hashable, float] | Sequence[float] | None,
    observed_distribution: Mapping[Hashable, float] | Sequence[float] | None,
    public_law: Mapping[Hashable, float] | None,
    radius: float | None,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
) -> _PreparedResidOptInputs:
    base = _prepare_base_inputs(
        grouped_or_problem,
        observed_distribution=observed_distribution,
        public_law=public_law,
        radius=radius,
        solver=solver,
        solver_options=solver_options,
    )
    return _PreparedResidOptInputs(
        problem=base.problem,
        direction=_coerce_direction(base.problem, direction),
        observed_distribution=base.observed_distribution,
        public_law=base.public_law,
        radius=base.radius,
        solver=base.solver,
        solver_options=base.solver_options,
        public_columns=base.public_columns,
        hidden_columns=base.hidden_columns,
    )


def _candidate_public_representations(
    *,
    public: tuple[str, ...],
    hidden: tuple[str, ...],
    candidate_refinements: tuple[str, ...],
    max_added_columns: int,
    include_base: bool,
) -> tuple[tuple[str, ...], ...]:
    valid_refinements = tuple(
        dict.fromkeys(
            column
            for column in candidate_refinements
            if column not in public and column in hidden
        )
    )
    rows: list[tuple[str, ...]] = []
    if include_base:
        rows.append(public)
    for count in range(1, max_added_columns + 1):
        for added_columns in combinations(valid_refinements, count):
            rows.append(public + tuple(added_columns))
    return tuple(dict.fromkeys(rows))


def _attach_reductions(
    rows: tuple[ResidOptRefinementScreenCandidate, ...],
) -> tuple[ResidOptRefinementScreenCandidate, ...]:
    if not rows:
        return rows
    base = next((row for row in rows if not row.added_columns), None)
    if base is None:
        return rows
    base_conservative = base.conservative_ambiguity
    base_exact = base.exact_ambiguity
    base_final = base.final_ambiguity

    reduced_rows: list[ResidOptRefinementScreenCandidate] = []
    for row in rows:
        conservative_reduction = (
            None
            if base_conservative is None or row.conservative_ambiguity is None
            else base_conservative - row.conservative_ambiguity
        )
        exact_reduction = (
            None
            if base_exact is None or row.exact_ambiguity is None
            else base_exact - row.exact_ambiguity
        )
        final_reduction = (
            None
            if base_final is None or row.final_ambiguity is None
            else base_final - row.final_ambiguity
        )
        final_reduction_percent = (
            None
            if final_reduction is None or base_final is None or base_final <= 0
            else 100.0 * final_reduction / base_final
        )
        reduced_rows.append(
            replace(
                row,
                conservative_reduction=conservative_reduction,
                exact_reduction=exact_reduction,
                final_reduction=final_reduction,
                final_reduction_percent=final_reduction_percent,
            )
        )
    return tuple(reduced_rows)


def _sort_refinement_screen_rows(
    rows: tuple[ResidOptRefinementScreenCandidate, ...],
    *,
    include_base: bool,
    top: int | None,
) -> tuple[ResidOptRefinementScreenCandidate, ...]:
    base_rows = tuple(row for row in rows if not row.added_columns)
    refinement_rows = [row for row in rows if row.added_columns]
    refinement_rows.sort(
        key=lambda row: (
            float("inf") if row.final_ambiguity is None else row.final_ambiguity,
            row.added_columns,
        )
    )
    if top is not None:
        refinement_rows = refinement_rows[:top]
    if include_base:
        return base_rows + tuple(refinement_rows)
    return tuple(refinement_rows)


def _require_l2_grouped(grouped: GroupedProblem) -> QPreset:
    if grouped.q is None or grouped.q.name != "l2_budget":
        raise ValueError(
            "ResidOpt refinement screening currently supports only q_l2_budget(...)."
        )
    if grouped.q.radius is None:
        raise ValueError("q_l2_budget radius is required for residopt screening")
    return grouped.q


def _repeatable_data(data: Any) -> Any:
    if hasattr(data, "to_dict"):
        return data
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return data
    return tuple(data)


def _prepare_base_inputs(
    grouped_or_problem: GroupedProblem | FiniteProblem,
    *,
    observed_distribution: Mapping[Hashable, float] | Sequence[float] | None,
    public_law: Mapping[Hashable, float] | None,
    radius: float | None,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
) -> _PreparedResidOptBase:
    if isinstance(grouped_or_problem, GroupedProblem):
        grouped = grouped_or_problem
        problem = grouped.problem
        if observed_distribution is None:
            observed_distribution = grouped.cell_weights
        if public_law is None:
            public_law = grouped.public_law
        if radius is None and grouped.q is not None and grouped.q.name == "l2_budget":
            radius = grouped.q.radius
        if solver is None and grouped.q is not None:
            solver = grouped.q.solver
        if solver_options is None and grouped.q is not None:
            solver_options = grouped.q.solver_options
        public_columns = grouped.public_columns
        hidden_columns = grouped.hidden_columns
    else:
        problem = grouped_or_problem
        public_columns = ()
        hidden_columns = ()

    if observed_distribution is None:
        raise ValueError(
            "observed_distribution is required when passing a raw FiniteProblem"
        )
    if public_law is None:
        raise ValueError("public_law is required when passing a raw FiniteProblem")
    if radius is None:
        raise ValueError(
            "radius is required unless grouped_or_problem was compiled with "
            "q_l2_budget(...)"
        )
    if radius < 0:
        raise ValueError("radius must be non-negative")

    observed_vector = problem._coerce_distribution(observed_distribution)
    public_law_dict = problem._coerce_public_law(public_law)
    observed_public_law = problem.public_law(observed_vector)
    mismatches = {
        key: (observed_public_law[key], public_law_dict[key])
        for key in problem.public_values
        if not isclose(
            observed_public_law[key],
            public_law_dict[key],
            rel_tol=0.0,
            abs_tol=problem.tol,
        )
    }
    if mismatches:
        raise ValueError(
            "observed_distribution public law must match public_law for "
            f"delta-space compilation; mismatches: {mismatches!r}"
        )

    return _PreparedResidOptBase(
        problem=problem,
        observed_distribution=np.asarray(observed_vector, dtype=float),
        public_law=public_law_dict,
        radius=float(radius),
        solver=solver,
        solver_options=dict(solver_options or {}),
        public_columns=public_columns,
        hidden_columns=hidden_columns,
    )


def _coerce_direction(
    problem: FiniteProblem,
    direction: Mapping[Hashable, float] | Sequence[float] | None,
) -> np.ndarray:
    if direction is None:
        if not isinstance(
            problem.target_functional, LinearTarget | UncertainLinearTarget
        ):
            raise TypeError(
                "residopt_l2_support_interval currently requires an explicit "
                "direction for non-linear target contracts."
            )
        direction_vector = tuple(
            problem.estimand_map[state] for state in problem.states
        )
    else:
        direction_vector = problem._coerce_vector(direction)
    return np.asarray(direction_vector, dtype=float)


def _compile_residopt_support_template(
    *,
    dimension: int,
    radius: float,
) -> _ResidOptSupportTemplate:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    residopt = _load_residopt()
    cp = import_module("cvxpy")
    decision = cp.Variable(dimension + 1, name="cached_l2_support_decision")
    direction_parameter = cp.Parameter(dimension, name="projected_direction")
    selector = np.concatenate(
        [np.zeros((dimension, 1)), np.eye(dimension)],
        axis=1,
    )
    epigraph_selector = np.concatenate([[1.0], np.zeros(dimension)])
    atom = residopt.EllipsoidSupportAtom(
        atom_id="cached_l2_public_nullspace_support",
        C=np.eye(dimension),
        rho=float(radius),
        S=selector,
        s0=np.zeros(dimension),
        g=epigraph_selector,
        h=0.0,
    )
    compiler = residopt.ResidualCompiler()
    compiled = compiler.compile_problem(
        objective=decision[0],
        x=decision,
        atoms=(atom,),
        base_constraints=(decision[1:] == direction_parameter,),
        minimize=True,
    )
    if compiled.oracle_atoms:
        raise RuntimeError("residopt selected oracle mode for the endpoint atom")
    return _ResidOptSupportTemplate(
        dimension=dimension,
        radius=float(radius),
        compiled=compiled,
        direction_parameter=direction_parameter,
    )


def _compile_residopt_ellipsoid_support(
    projected_direction: np.ndarray,
    *,
    radius: float,
    endpoint: str,
    solver: str | None,
    solver_options: Mapping[str, Any],
) -> _CompiledSupport:
    dimension = int(projected_direction.shape[0])
    if dimension == 0 or np.linalg.norm(projected_direction, ord=2) == 0.0:
        return _CompiledSupport(
            value=0.0,
            certificate=_deterministic_certificate(
                endpoint=endpoint,
                value=0.0,
                reason="The target direction is orthogonal to all public-law-preserving shifts.",
            ),
        )
    return _compile_residopt_support_template(
        dimension=dimension,
        radius=radius,
    ).solve(
        projected_direction,
        endpoint=endpoint,
        solver=solver,
        solver_options=solver_options,
    )


def _load_residopt() -> Any:
    try:
        return import_module("residopt")
    except ImportError as exc:
        raise ImportError(
            "residopt is required for residopt_l2_support_interval(). "
            "Install it with `pip install 'updatesupport[residopt]'` or "
            "`uv add 'updatesupport[residopt]'`."
        ) from exc


def _deterministic_certificate(
    *,
    endpoint: str,
    value: float,
    reason: str,
) -> ResidOptEndpointCertificate:
    return ResidOptEndpointCertificate(
        endpoint=endpoint,
        label="exact",
        template="deterministic_public_nullspace",
        output="closed_form",
        strategy="closed_form",
        solver_status="not_solved",
        value=float(value),
        exact_for_compiled_support=True,
        exact_for_updatesupport_q=True,
        conservative_for_updatesupport_q=True,
        reason=reason,
    )


def _public_incidence_matrix(problem: FiniteProblem) -> np.ndarray:
    state_index = {state: i for i, state in enumerate(problem.states)}
    incidence = np.zeros((len(problem.public_values), len(problem.states)))
    for row_index, public_value in enumerate(problem.public_values):
        for state in problem.public_fibers[public_value]:
            incidence[row_index, state_index[state]] = 1.0
    return incidence


def _orthonormal_nullspace(matrix: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        rank = 0
    else:
        threshold = tol * max(matrix.shape) * max(float(singular_values[0]), 1.0)
        rank = int(np.sum(singular_values > threshold))
    return vh[rank:].T.copy()


def _project_to_nullspace(nullspace: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.dot(nullspace.T, direction)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)


__all__ = [
    "ResidOptAvailability",
    "ResidOptEndpointCertificate",
    "ResidOptEndpointReport",
    "ResidOptL2EndpointCompiler",
    "ResidOptRefinementScreenCandidate",
    "ResidOptRefinementScreenContext",
    "ResidOptRefinementScreenReport",
    "residopt_available",
    "residopt_l2_support_interval",
    "residopt_refinement_screen",
]
