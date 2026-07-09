"""Direct minimum-distance witnesses that break threshold claims."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Hashable, Mapping, Sequence

from scipy.optimize import linprog

from .artifacts import ReportArtifactMixin
from .data import GroupedProblem, from_dataframe
from .environments import CvxpyError, LPError
from .presets import _mahalanobis_transform

if TYPE_CHECKING:
    from .claim import ClaimAudit, ClaimSpec, DecisionRule


@dataclass(frozen=True)
class ClaimBreakingCellShift:
    """Observed-to-witness change for one retained hidden cell."""

    public_value: Hashable
    state: Hashable
    target_value: float
    observed_mass: float
    witness_mass: float
    mass_change: float
    target_contribution_change: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_value": self.public_value,
            "state": self.state,
            "target_value": self.target_value,
            "observed_mass": self.observed_mass,
            "witness_mass": self.witness_mass,
            "mass_change": self.mass_change,
            "target_contribution_change": self.target_contribution_change,
        }


@dataclass(frozen=True)
class ClaimBreakingTransfer:
    """Mass transfer between two hidden cells in one public fiber."""

    public_value: Hashable
    source_state: Hashable
    destination_state: Hashable
    mass: float
    source_target: float
    destination_target: float
    target_change: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_value": self.public_value,
            "source_state": self.source_state,
            "destination_state": self.destination_state,
            "mass": self.mass,
            "source_target": self.source_target,
            "destination_target": self.destination_target,
            "target_change": self.target_change,
        }


@dataclass(frozen=True)
class MinimumClaimBreakingWitnessReport(ReportArtifactMixin):
    """Closest fixed-public hidden composition that falsifies a claim rule."""

    title: str
    claim_name: str
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target: str
    decision: DecisionRule
    status: str
    distance_metric: str
    distance: float | None
    witness_tv_distance: float | None
    threshold_margin: float
    breaking_boundary: float
    observed_value: float
    observed_decision: str
    witness_value: float | None
    witness_decision: str | None
    public_law_error: float | None
    solver: str
    solver_status: str
    exact: bool
    cells: tuple[ClaimBreakingCellShift, ...] = ()
    transfers: tuple[ClaimBreakingTransfer, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"found", "infeasible", "already_broken"}:
            raise ValueError(
                "status must be 'found', 'infeasible', or 'already_broken'"
            )
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "transfers", tuple(self.transfers))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def found(self) -> bool:
        return self.status == "found"

    @property
    def total_transferred_mass(self) -> float:
        return sum(row.mass for row in self.transfers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "claim_name": self.claim_name,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target": self.target,
            "decision": self.decision.as_dict(),
            "status": self.status,
            "found": self.found,
            "distance_metric": self.distance_metric,
            "distance": self.distance,
            "witness_tv_distance": self.witness_tv_distance,
            "total_transferred_mass": self.total_transferred_mass,
            "threshold_margin": self.threshold_margin,
            "breaking_boundary": self.breaking_boundary,
            "observed_value": self.observed_value,
            "observed_decision": self.observed_decision,
            "witness_value": self.witness_value,
            "witness_decision": self.witness_decision,
            "public_law_error": self.public_law_error,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "exact": self.exact,
            "cells": [row.as_dict() for row in self.cells],
            "transfers": [row.as_dict() for row in self.transfers],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return summary, cell-shift, transfer, and limitation tables."""

        summary = {
            key: value
            for key, value in self.as_dict().items()
            if key not in {"cells", "transfers", "limitations"}
        }
        return {
            "summary": (summary,),
            "cell_shifts": tuple(row.as_dict() for row in self.cells),
            "transfers": tuple(row.as_dict() for row in self.transfers),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }

    def to_markdown(self) -> str:
        """Render an analyst-facing interpretation of the closest witness."""

        lines = [
            f"# {self.title}",
            "",
            "## Result",
            "",
            f"- Claim: {self.claim_name}",
            f"- Decision rule: `{self.decision.name}`",
            f"- Observed value: {_format_float(self.observed_value)} "
            f"(`{self.observed_decision}`)",
            f"- Status: `{self.status}`",
            f"- Distance geometry: `{self.distance_metric}`",
        ]
        if self.found:
            lines.extend(
                [
                    f"- Minimum distance: {_format_float(self.distance)}",
                    "- Witness value: "
                    f"{_format_float(self.witness_value)} "
                    f"(`{self.witness_decision}`)",
                    "- Witness total-variation distance: "
                    f"{_format_float(self.witness_tv_distance)}",
                    "- Public-law preservation error: "
                    f"{_format_float(self.public_law_error)}",
                ]
            )
        elif self.status == "already_broken":
            lines.append(
                "- No recomposition is required: the observed value already fails "
                "the declared pass condition."
            )
        else:
            lines.append(
                "- No retained-support recomposition can cross the requested "
                "margin-separated decision boundary while preserving the public law."
            )

        lines.extend(["", "## Interpretation", ""])
        if self.found:
            geometry = (
                "total probability mass that must be reassigned"
                if self.distance_metric == "tv"
                else f"size of the recomposition in {self.distance_metric} geometry"
            )
            lines.extend(
                [
                    "This is the closest decision-breaking hidden composition on "
                    "the retained empirical support, with every public-bucket mass "
                    "held fixed.",
                    "",
                    f"The optimized distance measures the {geometry}. The witness "
                    f"moves the target from {_format_float(self.observed_value)} to "
                    f"{_format_float(self.witness_value)}, across the "
                    f"margin-separated boundary {_format_float(self.breaking_boundary)}.",
                    "",
                    "This is a deterministic sensitivity witness, not a confidence "
                    "interval or a probability that the claim is false.",
                ]
            )
        elif self.status == "infeasible":
            lines.append(
                "The claim cannot be broken using only the retained hidden cells and "
                "fixed public marginals. This is support-relative, not an absolute "
                "robustness guarantee."
            )
        else:
            lines.append(
                "The declared pass condition is false at the observed hidden-cell law, "
                "so its minimum breaking distance is zero."
            )

        if self.transfers:
            lines.extend(
                [
                    "",
                    "## Within-Fiber Transfers",
                    "",
                    "| public bucket | source | destination | mass | target change |",
                    "|:---|:---|:---|---:|---:|",
                ]
            )
            for row in self.transfers:
                lines.append(
                    "| "
                    f"{_format_key(self.public_columns, row.public_value)} | "
                    f"{_format_key(self.hidden_columns, row.source_state)} | "
                    f"{_format_key(self.hidden_columns, row.destination_state)} | "
                    f"{_format_float(row.mass)} | "
                    f"{_format_float(row.target_change)} |"
                )

        lines.extend(["", "## Solver Certificate", ""])
        lines.extend(
            [
                f"- Solver: `{self.solver}`",
                f"- Solver status: `{self.solver_status}`",
                f"- Convex optimum: {'yes' if self.exact else 'no'}",
                f"- Threshold separation margin: {_format_float(self.threshold_margin)}",
            ]
        )
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


def minimum_claim_breaking_witness(
    data: Any,
    claim: ClaimSpec | Mapping[str, Any],
    *,
    distance: str = "tv",
    covariance: Mapping[tuple[Hashable, Hashable], float]
    | Sequence[Sequence[float]]
    | None = None,
    threshold_margin: float = 1e-8,
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> MinimumClaimBreakingWitnessReport:
    """Find the closest fixed-public recomposition that fails a claim rule.

    The first-cut inverse problem uses the retained hidden-cell support and the
    observed public law. It does not impose the claim's forward Q preset beyond
    those public-law constraints.
    """

    from .claim import ClaimSpec

    if not isinstance(claim, ClaimSpec):
        claim = ClaimSpec.from_dict(claim)
    grouped = _grouped_for_claim(data, claim)
    return _minimum_claim_breaking_witness_grouped(
        grouped,
        claim=claim,
        distance=distance,
        covariance=covariance,
        threshold_margin=threshold_margin,
        solver=solver,
        solver_options=solver_options,
        title=title,
    )


def _minimum_claim_breaking_witness_from_audit(
    audit: ClaimAudit,
    *,
    distance: str = "tv",
    covariance: Mapping[tuple[Hashable, Hashable], float]
    | Sequence[Sequence[float]]
    | None = None,
    threshold_margin: float = 1e-8,
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> MinimumClaimBreakingWitnessReport:
    """Reuse an audit's compiled primary problem for an inverse witness solve."""

    return _minimum_claim_breaking_witness_grouped(
        audit.primary.grouped,
        claim=audit.claim,
        distance=distance,
        covariance=covariance,
        threshold_margin=threshold_margin,
        solver=solver,
        solver_options=solver_options,
        title=title,
    )


def _minimum_claim_breaking_witness_grouped(
    grouped: GroupedProblem,
    *,
    claim: ClaimSpec,
    distance: str,
    covariance: Mapping[tuple[Hashable, Hashable], float]
    | Sequence[Sequence[float]]
    | None,
    threshold_margin: float,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
    title: str | None,
) -> MinimumClaimBreakingWitnessReport:
    decision = claim.decision
    if decision is None:
        raise ValueError("minimum claim-breaking witnesses require claim.decision")
    if not grouped.problem.has_linear_target:
        grouped.problem._require_linear_target(
            "minimum claim-breaking witness analysis"
        )
    metric = str(distance).strip().lower().replace("-", "_")
    if metric not in {"tv", "l2", "mahalanobis"}:
        raise ValueError("distance must be 'tv', 'l2', or 'mahalanobis'")
    margin = float(threshold_margin)
    if not isfinite(margin) or margin <= 0:
        raise ValueError("threshold_margin must be finite and positive")
    if metric == "mahalanobis" and covariance is None:
        raise ValueError("covariance is required for Mahalanobis distance")
    if metric != "mahalanobis" and covariance is not None:
        raise ValueError("covariance is only used with distance='mahalanobis'")

    states = grouped.problem.states
    q0 = tuple(float(grouped.cell_weights[state]) for state in states)
    h = tuple(float(grouped.problem.estimand_map[state]) for state in states)
    observed_value = sum(mass * value for mass, value in zip(q0, h, strict=True))
    observed_decision = decision.evaluate(observed_value)
    boundary, direction = _breaking_boundary(decision, margin)
    report_title = title or f"{claim.estimate_name} Minimum Breaking Witness"

    if observed_decision == decision.fail_label:
        return _build_report(
            grouped,
            claim=claim,
            status="already_broken",
            metric=metric,
            distance_value=0.0,
            margin=margin,
            boundary=boundary,
            observed_value=observed_value,
            witness=q0,
            solver="none",
            solver_status="observed claim already fails",
            title=report_title,
        )

    if metric == "tv":
        witness, distance_value, solver_name, solver_status = _solve_tv(
            grouped,
            q0=q0,
            h=h,
            boundary=boundary,
            direction=direction,
        )
    else:
        witness, distance_value, solver_name, solver_status = _solve_conic(
            grouped,
            q0=q0,
            h=h,
            boundary=boundary,
            direction=direction,
            metric=metric,
            covariance=covariance,
            solver=solver,
            solver_options=solver_options,
        )

    status = "found" if witness is not None else "infeasible"
    return _build_report(
        grouped,
        claim=claim,
        status=status,
        metric=metric,
        distance_value=distance_value,
        margin=margin,
        boundary=boundary,
        observed_value=observed_value,
        witness=witness,
        solver=solver_name,
        solver_status=solver_status,
        title=report_title,
    )


def _grouped_for_claim(data: Any, claim: ClaimSpec) -> GroupedProblem:
    if isinstance(data, GroupedProblem):
        if tuple(data.public_columns) != tuple(claim.public):
            raise ValueError("GroupedProblem public columns do not match the claim")
        if tuple(data.hidden_columns) != tuple(claim.hidden):
            raise ValueError("GroupedProblem hidden columns do not match the claim")
        return data
    return from_dataframe(
        data,
        public=claim.public,
        hidden=claim.hidden,
        target=claim.target,
        weight=claim.weight,
        min_cell_weight=claim.min_cell_weight,
        q="saturated",
    )


def _breaking_boundary(decision: DecisionRule, margin: float) -> tuple[float, str]:
    if decision.operator in {">=", ">"}:
        return decision.threshold - margin, "upper"
    return decision.threshold + margin, "lower"


def _solve_tv(
    grouped: GroupedProblem,
    *,
    q0: Sequence[float],
    h: Sequence[float],
    boundary: float,
    direction: str,
) -> tuple[tuple[float, ...] | None, float | None, str, str]:
    import numpy as np

    states = grouped.problem.states
    n = len(states)
    objective = np.concatenate([np.zeros(n), 0.5 * np.ones(n)])
    a_eq = []
    b_eq = []
    for public_value in grouped.problem.public_values:
        row = np.zeros(2 * n)
        for index, state in enumerate(states):
            if grouped.problem.public_map[state] == public_value:
                row[index] = 1.0
        a_eq.append(row)
        b_eq.append(grouped.public_law[public_value])

    a_ub = []
    b_ub = []
    target_row = np.zeros(2 * n)
    target_row[:n] = h if direction == "upper" else -np.asarray(h)
    a_ub.append(target_row)
    b_ub.append(boundary if direction == "upper" else -boundary)
    for index in range(n):
        positive = np.zeros(2 * n)
        positive[index] = 1.0
        positive[n + index] = -1.0
        a_ub.append(positive)
        b_ub.append(q0[index])

        negative = np.zeros(2 * n)
        negative[index] = -1.0
        negative[n + index] = -1.0
        a_ub.append(negative)
        b_ub.append(-q0[index])

    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=[(0.0, None)] * (2 * n),
        method="highs",
    )
    if result.status == 2:
        return None, None, "scipy.optimize.linprog[highs]", result.message
    if not result.success or result.x is None or result.fun is None:
        raise LPError(f"minimum TV witness solve failed: {result.message}")
    witness = _clean_witness(result.x[:n], tol=grouped.problem.tol)
    return (
        witness,
        float(result.fun),
        "scipy.optimize.linprog[highs]",
        result.message,
    )


def _solve_conic(
    grouped: GroupedProblem,
    *,
    q0: Sequence[float],
    h: Sequence[float],
    boundary: float,
    direction: str,
    metric: str,
    covariance: Mapping[tuple[Hashable, Hashable], float]
    | Sequence[Sequence[float]]
    | None,
    solver: str | None,
    solver_options: Mapping[str, Any] | None,
) -> tuple[tuple[float, ...] | None, float | None, str, str]:
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CvxpyError(
            "L2 and Mahalanobis witnesses require CVXPY. Install it with "
            "`pip install updatesupport[cvxpy]` or `uv add updatesupport[cvxpy]`."
        ) from exc

    states = grouped.problem.states
    q = cp.Variable(len(states), nonneg=True)
    q0_array = np.asarray(q0, dtype=float)
    h_array = np.asarray(h, dtype=float)
    constraints = []
    for public_value in grouped.problem.public_values:
        indices = [
            index
            for index, state in enumerate(states)
            if grouped.problem.public_map[state] == public_value
        ]
        constraints.append(cp.sum(q[indices]) == grouped.public_law[public_value])
    target_expression = h_array @ q
    if direction == "upper":
        constraints.append(target_expression <= boundary)
    else:
        constraints.append(target_expression >= boundary)

    delta = q - q0_array
    if metric == "l2":
        distance_expression = cp.norm(delta, 2)
    else:
        transform = _mahalanobis_transform(covariance, states)
        distance_expression = cp.norm(transform @ delta, 2)
    problem = cp.Problem(cp.Minimize(distance_expression), constraints)
    options = dict(solver_options or {})
    try:
        if solver is None:
            problem.solve(**options)
        else:
            installed = {str(item).upper() for item in cp.installed_solvers()}
            if str(solver).upper() not in installed:
                raise CvxpyError(f"CVXPY solver {solver!r} is not installed")
            problem.solve(solver=solver, **options)
    except CvxpyError:
        raise
    except Exception as exc:  # pragma: no cover - solver errors vary
        raise CvxpyError(str(exc)) from exc

    solver_name = (
        str(problem.solver_stats.solver_name)
        if problem.solver_stats is not None
        else (solver or "cvxpy-default")
    )
    if problem.status in {"infeasible", "infeasible_inaccurate"}:
        return None, None, solver_name, problem.status
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise CvxpyError(f"CVXPY minimum witness status: {problem.status}")
    if q.value is None or problem.value is None:
        raise CvxpyError("CVXPY minimum witness did not return an optimizer")
    witness = _clean_witness(q.value, tol=grouped.problem.tol)
    return witness, float(problem.value), solver_name, problem.status


def _build_report(
    grouped: GroupedProblem,
    *,
    claim: ClaimSpec,
    status: str,
    metric: str,
    distance_value: float | None,
    margin: float,
    boundary: float,
    observed_value: float,
    witness: Sequence[float] | None,
    solver: str,
    solver_status: str,
    title: str,
) -> MinimumClaimBreakingWitnessReport:
    decision = claim.decision
    if decision is None:  # pragma: no cover - validated at the public boundary
        raise ValueError("minimum claim-breaking witnesses require claim.decision")
    states = grouped.problem.states
    q0 = tuple(float(grouped.cell_weights[state]) for state in states)
    target_values = tuple(
        float(grouped.problem.estimand_map[state]) for state in states
    )
    if witness is None:
        cells: tuple[ClaimBreakingCellShift, ...] = ()
        transfers: tuple[ClaimBreakingTransfer, ...] = ()
        witness_value = None
        witness_decision = None
        tv_distance = None
        public_law_error = None
    else:
        witness_tuple = tuple(float(value) for value in witness)
        witness_value = sum(
            mass * target
            for mass, target in zip(witness_tuple, target_values, strict=True)
        )
        witness_decision = decision.evaluate(witness_value)
        cells = _cell_shifts(grouped, q0=q0, witness=witness_tuple)
        transfers = _transfers(grouped, q0=q0, witness=witness_tuple)
        tv_distance = 0.5 * sum(
            abs(left - right) for left, right in zip(q0, witness_tuple, strict=True)
        )
        public_law_error = _public_law_error(grouped, witness_tuple)
        if status == "found" and witness_decision != decision.fail_label:
            error = (
                "minimum witness did not cross the decision threshold after "
                "numerical solving; increase threshold_margin"
            )
            if metric == "tv":
                raise LPError(error)
            raise CvxpyError(error)
        if public_law_error > 1e-7:
            error = (
                "minimum witness does not preserve the public law within "
                f"tolerance (error={public_law_error:g})"
            )
            if metric == "tv":
                raise LPError(error)
            raise CvxpyError(error)

    return MinimumClaimBreakingWitnessReport(
        title=title,
        claim_name=claim.estimate_name,
        public_columns=tuple(grouped.public_columns),
        hidden_columns=tuple(grouped.hidden_columns),
        target=str(grouped.target_column),
        decision=decision,
        status=status,
        distance_metric=metric,
        distance=distance_value,
        witness_tv_distance=tv_distance,
        threshold_margin=margin,
        breaking_boundary=boundary,
        observed_value=observed_value,
        observed_decision=decision.evaluate(observed_value),
        witness_value=witness_value,
        witness_decision=witness_decision,
        public_law_error=public_law_error,
        solver=solver,
        solver_status=solver_status,
        exact=True,
        cells=cells,
        transfers=transfers,
        limitations=(
            "The result holds the observed public law fixed and only redistributes "
            "mass over retained hidden cells.",
            "Hidden means observed by the analyst but not included in the public "
            "reporting representation; it does not mean statistically unobserved.",
            "The minimum is relative to this refinement, empirical target values, "
            "distance geometry, and threshold margin.",
            "No forward Q preset is imposed beyond fixed public marginals in this "
            "first-cut inverse solve.",
            "This is deterministic composition sensitivity, not statistical "
            "uncertainty or a confidence interval.",
        ),
    )


def _cell_shifts(
    grouped: GroupedProblem,
    *,
    q0: Sequence[float],
    witness: Sequence[float],
) -> tuple[ClaimBreakingCellShift, ...]:
    rows = []
    for state, observed_mass, witness_mass in zip(
        grouped.problem.states, q0, witness, strict=True
    ):
        target_value = float(grouped.problem.estimand_map[state])
        mass_change = witness_mass - observed_mass
        rows.append(
            ClaimBreakingCellShift(
                public_value=grouped.problem.public_map[state],
                state=state,
                target_value=target_value,
                observed_mass=observed_mass,
                witness_mass=witness_mass,
                mass_change=mass_change,
                target_contribution_change=mass_change * target_value,
            )
        )
    return tuple(sorted(rows, key=lambda row: abs(row.mass_change), reverse=True))


def _transfers(
    grouped: GroupedProblem,
    *,
    q0: Sequence[float],
    witness: Sequence[float],
) -> tuple[ClaimBreakingTransfer, ...]:
    index = {state: position for position, state in enumerate(grouped.problem.states)}
    tol = grouped.problem.tol
    rows = []
    for public_value, fiber in grouped.problem.public_fibers.items():
        sources = [
            [state, q0[index[state]] - witness[index[state]]]
            for state in fiber
            if q0[index[state]] - witness[index[state]] > tol
        ]
        destinations = [
            [state, witness[index[state]] - q0[index[state]]]
            for state in fiber
            if witness[index[state]] - q0[index[state]] > tol
        ]
        sources.sort(key=lambda item: item[1], reverse=True)
        destinations.sort(key=lambda item: item[1], reverse=True)
        source_index = 0
        destination_index = 0
        while source_index < len(sources) and destination_index < len(destinations):
            source_state, source_mass = sources[source_index]
            destination_state, destination_mass = destinations[destination_index]
            moved = min(float(source_mass), float(destination_mass))
            source_target = float(grouped.problem.estimand_map[source_state])
            destination_target = float(grouped.problem.estimand_map[destination_state])
            rows.append(
                ClaimBreakingTransfer(
                    public_value=public_value,
                    source_state=source_state,
                    destination_state=destination_state,
                    mass=moved,
                    source_target=source_target,
                    destination_target=destination_target,
                    target_change=moved * (destination_target - source_target),
                )
            )
            sources[source_index][1] = float(source_mass) - moved
            destinations[destination_index][1] = float(destination_mass) - moved
            if sources[source_index][1] <= tol:
                source_index += 1
            if destinations[destination_index][1] <= tol:
                destination_index += 1
    return tuple(sorted(rows, key=lambda row: row.mass, reverse=True))


def _public_law_error(grouped: GroupedProblem, witness: Sequence[float]) -> float:
    index = {state: position for position, state in enumerate(grouped.problem.states)}
    return max(
        abs(
            sum(witness[index[state]] for state in fiber)
            - grouped.public_law[public_value]
        )
        for public_value, fiber in grouped.problem.public_fibers.items()
    )


def _clean_witness(values: Sequence[float], *, tol: float) -> tuple[float, ...]:
    cleaned = []
    for value in values:
        number = float(value)
        if number < -1e-7:
            raise ValueError("solver returned a materially negative probability")
        cleaned.append(0.0 if number < 0 or abs(number) <= tol else number)
    return tuple(cleaned)


def _format_key(columns: Sequence[str], value: Hashable) -> str:
    values = value if isinstance(value, tuple) else (value,)
    return ", ".join(
        f"{column}={item}" for column, item in zip(columns, values, strict=False)
    )


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"
