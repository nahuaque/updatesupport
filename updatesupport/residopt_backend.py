"""Experimental optional integration with the ``residopt`` compiler.

This module is intentionally narrow. It exposes a first endpoint compiler for
L2-budget hidden-composition stress tests by mapping the fixed-public-law
subspace to a residopt ellipsoidal support atom.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from math import isclose
from typing import Any, Hashable

import numpy as np

from .data import GroupedProblem
from .problem import FiniteProblem
from .targets import LinearTarget, UncertainLinearTarget


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
class ResidOptEndpointReport:
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

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())

    def to_json(self, **kwargs: Any) -> str:
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(_json_ready(self.as_dict()), **options)

    def to_markdown(self) -> str:
        lines = [
            f"# {_markdown_escape(self.title)}",
            "",
            (
                f"Backend: `{self.backend}`. Q preset: `{self.q_name}` "
                f"with L2 radius `{_format_float(self.radius)}`."
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


def residopt_available() -> ResidOptAvailability:
    """Return whether the optional ``residopt`` backend can be imported."""

    try:
        import_module("residopt")
    except ImportError:
        return ResidOptAvailability(
            available=False,
            reason=(
                "residopt is not importable. Install it or add the sibling "
                "repo to PYTHONPATH, for example `PYTHONPATH=../residopt/src`."
            ),
        )
    try:
        installed_version = version("residopt")
    except PackageNotFoundError:
        installed_version = None
    return ResidOptAvailability(available=True, version=installed_version)


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

    prepared = _prepare_inputs(
        grouped_or_problem,
        direction=direction,
        observed_distribution=observed_distribution,
        public_law=public_law,
        radius=radius,
        solver=solver,
        solver_options=solver_options,
    )
    problem = prepared.problem
    target_contract = problem.target_contract

    incidence = _public_incidence_matrix(problem)
    nullspace = _orthonormal_nullspace(incidence, tol=problem.tol)
    observed_value = float(prepared.direction @ prepared.observed_distribution)
    projected_direction = _project_to_nullspace(nullspace, prepared.direction)
    projected_negative_direction = _project_to_nullspace(nullspace, -prepared.direction)

    if prepared.radius == 0.0 or nullspace.shape[1] == 0:
        lower_support = 0.0
        upper_support = 0.0
        lower_certificate = _deterministic_certificate(
            endpoint="lower",
            value=0.0,
            reason="No admissible L2 hidden-composition shift is available.",
        )
        upper_certificate = _deterministic_certificate(
            endpoint="upper",
            value=0.0,
            reason="No admissible L2 hidden-composition shift is available.",
        )
    else:
        lower_compiled = _compile_residopt_ellipsoid_support(
            projected_negative_direction,
            radius=prepared.radius,
            endpoint="lower",
            solver=prepared.solver,
            solver_options=prepared.solver_options,
        )
        upper_compiled = _compile_residopt_ellipsoid_support(
            projected_direction,
            radius=prepared.radius,
            endpoint="upper",
            solver=prepared.solver,
            solver_options=prepared.solver_options,
        )
        lower_support = lower_compiled.value
        upper_support = upper_compiled.value
        lower_certificate = lower_compiled.certificate
        upper_certificate = upper_compiled.certificate

    notes = (
        "Public-law equality is preserved by projecting shifts into the "
        "nullspace of the public-incidence matrix.",
        "Hidden-cell nonnegativity is not enforced in this experimental adapter; "
        "the interval is conservative for the exact simplex-constrained endpoint.",
        "The residopt certificate is exact for the compiled ellipsoidal support "
        "atom, not for every constraint in the original updatesupport Q set.",
    )
    return ResidOptEndpointReport(
        title=title,
        observed_value=observed_value,
        lower=observed_value - lower_support,
        upper=observed_value + upper_support,
        lower_support_value=lower_support,
        upper_support_value=upper_support,
        radius=prepared.radius,
        q_name="l2_budget",
        target_name=target_contract.name,
        target_description=target_contract.description,
        state_count=len(problem.states),
        public_cell_count=len(problem.public_values),
        nullspace_dimension=int(nullspace.shape[1]),
        public_columns=prepared.public_columns,
        hidden_columns=prepared.hidden_columns,
        lower_certificate=lower_certificate,
        upper_certificate=upper_certificate,
        notes=notes,
    )


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

    if direction is None:
        if not isinstance(problem.target_functional, LinearTarget | UncertainLinearTarget):
            raise TypeError(
                "residopt_l2_support_interval currently requires an explicit "
                "direction for non-linear target contracts."
            )
        direction_vector = tuple(
            problem.estimand_map[state] for state in problem.states
        )
    else:
        direction_vector = problem._coerce_vector(direction)

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

    return _PreparedResidOptInputs(
        problem=problem,
        direction=np.asarray(direction_vector, dtype=float),
        observed_distribution=np.asarray(observed_vector, dtype=float),
        public_law=public_law_dict,
        radius=float(radius),
        solver=solver,
        solver_options=dict(solver_options or {}),
        public_columns=public_columns,
        hidden_columns=hidden_columns,
    )


def _compile_residopt_ellipsoid_support(
    projected_direction: np.ndarray,
    *,
    radius: float,
    endpoint: str,
    solver: str | None,
    solver_options: Mapping[str, Any],
) -> _CompiledSupport:
    residopt = _load_residopt()
    cp = import_module("cvxpy")
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

    gamma = cp.Variable(1, name=f"{endpoint}_support")
    atom = residopt.EllipsoidSupportAtom(
        atom_id=f"{endpoint}_l2_public_nullspace_support",
        C=np.eye(dimension),
        rho=float(radius),
        S=np.zeros((dimension, 1)),
        s0=projected_direction,
        g=np.array([1.0]),
        h=0.0,
    )
    compiler = residopt.ResidualCompiler()
    compiled = compiler.compile_problem(
        objective=gamma[0],
        x=gamma,
        atoms=(atom,),
        base_constraints=(),
        minimize=True,
    )
    if compiled.oracle_atoms:
        raise RuntimeError("residopt selected oracle mode for the endpoint atom")

    solve_kwargs = dict(solver_options)
    if solver is not None:
        solve_kwargs["solver"] = solver
    value = float(compiled.solve(**solve_kwargs))
    status = compiled.problem.status
    if status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"residopt endpoint solve failed with status {status!r}")

    residopt_certificate = compiled.certificates[0] if compiled.certificates else None
    metadata = (
        {}
        if residopt_certificate is None
        else dict(residopt_certificate.metadata or {})
    )
    metadata["public_nullspace_dimension"] = dimension
    metadata["projected_direction_norm"] = float(
        np.linalg.norm(projected_direction, ord=2)
    )
    reason = (
        "residopt certifies the ellipsoidal support atom exactly. updatesupport "
        "marks the endpoint conservative because this adapter relaxes away "
        "hidden-cell nonnegativity from the original Q set."
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


def _load_residopt() -> Any:
    try:
        return import_module("residopt")
    except ImportError as exc:
        raise ImportError(
            "residopt is required for residopt_l2_support_interval(). "
            "Install it or run with `PYTHONPATH=../residopt/src` while using "
            "the sibling checkout."
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


__all__ = [
    "ResidOptAvailability",
    "ResidOptEndpointCertificate",
    "ResidOptEndpointReport",
    "residopt_available",
    "residopt_l2_support_interval",
]
