"""Named admissible-environment presets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence

from .environments import (
    BatchedCvxpyEnvironments,
    CvxpyEnvironments,
    Environment,
    FiniteEnvironments,
    ParameterizedCvxpyEnvironments,
    PolytopeEnvironments,
    PublicFiberSaturated,
    cvxpy_constraint,
    eq,
)


@dataclass(frozen=True)
class QPreset:
    """Reusable preset for constructing an admissible environment ``Q``."""

    name: str
    radius: float | None = None
    cost: Any | None = None
    backend: str | None = None
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QEnvironment:
    environment: Environment
    preset: QPreset | None
    name: str
    description: str


def q_saturated() -> QPreset:
    """Allow arbitrary hidden reweighting inside each observed public fiber."""

    return QPreset("saturated")


def q_observed() -> QPreset:
    """Use only the observed hidden distribution."""

    return QPreset("observed")


def q_bounded_shift(radius: float = 0.5) -> QPreset:
    """Limit each hidden-cell mass to a relative band around its observed mass."""

    return QPreset("bounded_shift", radius=float(radius))


def q_tv_budget(
    radius: float,
    *,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain total variation distance from the observed hidden distribution."""

    return QPreset(
        "tv_budget",
        radius=float(radius),
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
    )


def q_chi_square_budget(
    radius: float,
    *,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain Pearson chi-square divergence from the observed distribution."""

    return QPreset(
        "chi_square_budget",
        radius=float(radius),
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
    )


def q_kl_budget(
    radius: float,
    *,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain KL divergence from the observed hidden distribution."""

    return QPreset(
        "kl_budget",
        radius=float(radius),
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
    )


def q_wasserstein(
    cost: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    radius: float,
    *,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain Wasserstein distance using an explicit hidden-cell cost matrix."""

    return QPreset(
        "wasserstein",
        radius=float(radius),
        cost=cost,
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
    )


def resolve_q_environment(
    q: Any,
    *,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    cell_weights: Mapping[Hashable, float],
    q_radius: float | None = None,
) -> QEnvironment:
    """Build an environment from a preset or pass through a custom environment."""

    preset = normalize_q_preset(q, q_radius=q_radius)
    if preset is None:
        name = getattr(q, "name", q.__class__.__name__)
        return QEnvironment(
            environment=q,
            preset=None,
            name=str(name),
            description="custom admissible environment",
        )

    if preset.name == "saturated":
        return QEnvironment(
            environment=PublicFiberSaturated.fixed(public_law),
            preset=preset,
            name=q_name(preset),
            description=(
                "arbitrary reweighting among retained hidden cells inside each "
                "observed public cell"
            ),
        )

    if preset.name == "observed":
        return QEnvironment(
            environment=FiniteEnvironments([cell_weights], name="observed"),
            preset=preset,
            name=q_name(preset),
            description="only the observed hidden distribution is admissible",
        )

    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        constraints = _public_law_constraints(public_law, public_map)
        bounds = {
            state: (
                max(0.0, mass * (1.0 - radius)),
                min(1.0, mass * (1.0 + radius)),
            )
            for state, mass in cell_weights.items()
        }
        return QEnvironment(
            environment=PolytopeEnvironments(
                constraints=constraints,
                bounds=bounds,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=(
                "fixed observed public law with each hidden-cell mass constrained "
                f"to +/- {100 * radius:.1f}% of its observed mass"
            ),
        )

    if preset.name == "tv_budget":
        radius = _tv_radius(preset)
        if radius == 0.0:
            return _point_environment(
                cell_weights,
                preset,
                description=(
                    "fixed observed public law with total variation distance from "
                    "the observed hidden distribution <= 0"
                ),
            )
        backend = _backend_name(preset, default="cvxpy")
        if backend == "parameterized_cvxpy":
            return QEnvironment(
                environment=ParameterizedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    parameterized_constraint_builders=(
                        _tv_parameterized_constraint_builder(cell_weights),
                    ),
                    parameter_values={"radius": radius},
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with total variation distance from "
                    f"the observed hidden distribution <= {radius:g}"
                ),
            )
        if backend == "batched_cvxpy":
            return QEnvironment(
                environment=BatchedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    constraint_builders=(_tv_constraint_builder(cell_weights, radius),),
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with total variation distance from "
                    f"the observed hidden distribution <= {radius:g}"
                ),
            )
        _require_backend(preset, "cvxpy")
        return QEnvironment(
            environment=CvxpyEnvironments(
                fixed_public_law=public_law,
                constraint_builders=(_tv_constraint_builder(cell_weights, radius),),
                solver=preset.solver,
                solver_options=preset.solver_options,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=(
                "fixed observed public law with total variation distance from "
                f"the observed hidden distribution <= {radius:g}"
            ),
        )

    if preset.name == "chi_square_budget":
        radius = _chi_square_radius(preset)
        if radius == 0.0:
            return _point_environment(
                cell_weights,
                preset,
                description=(
                    "fixed observed public law with Pearson chi-square divergence "
                    "from the observed hidden distribution <= 0"
                ),
            )
        backend = _backend_name(preset, default="cvxpy")
        if backend == "parameterized_cvxpy":
            return QEnvironment(
                environment=ParameterizedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    parameterized_constraint_builders=(
                        _chi_square_parameterized_constraint_builder(cell_weights),
                    ),
                    parameter_values={"radius": radius},
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with Pearson chi-square divergence "
                    f"from the observed hidden distribution <= {radius:g}"
                ),
            )
        if backend == "batched_cvxpy":
            return QEnvironment(
                environment=BatchedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    constraint_builders=(
                        _chi_square_constraint_builder(cell_weights, radius),
                    ),
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with Pearson chi-square divergence "
                    f"from the observed hidden distribution <= {radius:g}"
                ),
            )
        _require_backend(preset, "cvxpy")
        return QEnvironment(
            environment=CvxpyEnvironments(
                fixed_public_law=public_law,
                constraint_builders=(
                    _chi_square_constraint_builder(cell_weights, radius),
                ),
                solver=preset.solver,
                solver_options=preset.solver_options,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=(
                "fixed observed public law with Pearson chi-square divergence "
                f"from the observed hidden distribution <= {radius:g}"
            ),
        )

    if preset.name == "kl_budget":
        radius = _kl_radius(preset)
        if radius == 0.0:
            return _point_environment(
                cell_weights,
                preset,
                description=(
                    "fixed observed public law with KL divergence from the "
                    "observed hidden distribution <= 0"
                ),
            )
        backend = _backend_name(preset, default="cvxpy")
        if backend == "parameterized_cvxpy":
            return QEnvironment(
                environment=ParameterizedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    parameterized_constraint_builders=(
                        _kl_parameterized_constraint_builder(cell_weights),
                    ),
                    parameter_values={"radius": radius},
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with KL divergence from the observed "
                    f"hidden distribution <= {radius:g}"
                ),
            )
        if backend == "batched_cvxpy":
            return QEnvironment(
                environment=BatchedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    constraint_builders=(_kl_constraint_builder(cell_weights, radius),),
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with KL divergence from the observed "
                    f"hidden distribution <= {radius:g}"
                ),
            )
        _require_backend(preset, "cvxpy")
        return QEnvironment(
            environment=CvxpyEnvironments(
                fixed_public_law=public_law,
                constraint_builders=(_kl_constraint_builder(cell_weights, radius),),
                solver=preset.solver,
                solver_options=preset.solver_options,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=(
                "fixed observed public law with KL divergence from the observed "
                f"hidden distribution <= {radius:g}"
            ),
        )

    if preset.name == "wasserstein":
        radius = _wasserstein_radius(preset)
        if preset.cost is None:
            raise ValueError("q_wasserstein requires an explicit cost matrix")
        backend = _backend_name(preset, default="cvxpy")
        if backend == "parameterized_cvxpy":
            return QEnvironment(
                environment=ParameterizedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    parameterized_constraint_builders=(
                        _wasserstein_parameterized_constraint_builder(
                            cell_weights,
                            preset.cost,
                        ),
                    ),
                    parameter_values={"radius": radius},
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with Wasserstein cost from the "
                    f"observed hidden distribution <= {radius:g}"
                ),
            )
        if backend == "batched_cvxpy":
            return QEnvironment(
                environment=BatchedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    constraint_builders=(
                        _wasserstein_constraint_builder(
                            cell_weights,
                            preset.cost,
                            radius,
                        ),
                    ),
                    solver=preset.solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=(
                    "fixed observed public law with Wasserstein cost from the "
                    f"observed hidden distribution <= {radius:g}"
                ),
            )
        _require_backend(preset, "cvxpy")
        return QEnvironment(
            environment=CvxpyEnvironments(
                fixed_public_law=public_law,
                constraint_builders=(
                    _wasserstein_constraint_builder(cell_weights, preset.cost, radius),
                ),
                solver=preset.solver,
                solver_options=preset.solver_options,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=(
                "fixed observed public law with Wasserstein cost from the "
                f"observed hidden distribution <= {radius:g}"
            ),
        )

    raise ValueError(f"unsupported Q preset: {preset.name!r}")


def _point_environment(
    cell_weights: Mapping[Hashable, float],
    preset: QPreset,
    *,
    description: str,
) -> QEnvironment:
    return QEnvironment(
        environment=FiniteEnvironments([cell_weights], name=q_name(preset)),
        preset=preset,
        name=q_name(preset),
        description=description,
    )


def normalize_q_preset(q: Any, *, q_radius: float | None = None) -> QPreset | None:
    if isinstance(q, QPreset):
        preset = _canonical_preset(q)
    elif isinstance(q, str):
        preset = _canonical_preset(QPreset(q))
    else:
        if q_radius is not None:
            raise ValueError("q_radius can only be used with a named Q preset")
        return None

    if q_radius is not None:
        if preset.name not in {
            "bounded_shift",
            "chi_square_budget",
            "kl_budget",
            "tv_budget",
            "wasserstein",
        }:
            raise ValueError(
                "q_radius is only valid for bounded, chi-square, KL, TV, "
                "or Wasserstein Q presets"
            )
        if preset.radius is not None and float(preset.radius) != float(q_radius):
            raise ValueError("q_radius conflicts with the QPreset radius")
        preset = replace(preset, radius=float(q_radius))

    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "tv_budget":
        radius = _tv_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "chi_square_budget":
        radius = _chi_square_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "kl_budget":
        radius = _kl_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "wasserstein":
        radius = _wasserstein_radius(preset)
        preset = replace(preset, radius=radius)

    return preset


def q_name(q: Any, *, q_radius: float | None = None) -> str:
    preset = normalize_q_preset(q, q_radius=q_radius)
    if preset is None:
        return str(getattr(q, "name", q.__class__.__name__))
    if preset.name == "bounded_shift":
        return f"bounded_shift(radius={_bounded_radius(preset):g})"
    if preset.name == "tv_budget":
        return f"tv_budget(radius={_tv_radius(preset):g})"
    if preset.name == "chi_square_budget":
        return f"chi_square_budget(radius={_chi_square_radius(preset):g})"
    if preset.name == "kl_budget":
        return f"kl_budget(radius={_kl_radius(preset):g})"
    if preset.name == "wasserstein":
        return f"wasserstein(radius={_wasserstein_radius(preset):g})"
    return preset.name


def q_description(q: Any, *, q_radius: float | None = None) -> str:
    preset = normalize_q_preset(q, q_radius=q_radius)
    if preset is None:
        return "custom admissible environment"
    if preset.name == "saturated":
        return (
            "arbitrary reweighting among retained hidden cells inside each observed "
            "public cell"
        )
    if preset.name == "observed":
        return "only the observed hidden distribution is admissible"
    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        return (
            "fixed observed public law with each hidden-cell mass constrained "
            f"to +/- {100 * radius:.1f}% of its observed mass"
        )
    if preset.name == "tv_budget":
        radius = _tv_radius(preset)
        return (
            "fixed observed public law with total variation distance from "
            f"the observed hidden distribution <= {radius:g}"
        )
    if preset.name == "chi_square_budget":
        radius = _chi_square_radius(preset)
        return (
            "fixed observed public law with Pearson chi-square divergence from "
            f"the observed hidden distribution <= {radius:g}"
        )
    if preset.name == "kl_budget":
        radius = _kl_radius(preset)
        return (
            "fixed observed public law with KL divergence from the observed "
            f"hidden distribution <= {radius:g}"
        )
    if preset.name == "wasserstein":
        radius = _wasserstein_radius(preset)
        return (
            "fixed observed public law with Wasserstein cost from the observed "
            f"hidden distribution <= {radius:g}"
        )
    raise ValueError(f"unsupported Q preset: {preset.name!r}")


def _canonical_preset(preset: QPreset) -> QPreset:
    aliases = {
        "public_fiber_saturated": "saturated",
        "public-fiber-saturated": "saturated",
        "saturated": "saturated",
        "observed": "observed",
        "point": "observed",
        "observed_only": "observed",
        "bounded": "bounded_shift",
        "bounded-shift": "bounded_shift",
        "bounded_shift": "bounded_shift",
        "total-variation": "tv_budget",
        "total_variation": "tv_budget",
        "tv": "tv_budget",
        "tv_budget": "tv_budget",
        "chi-square": "chi_square_budget",
        "chi_square": "chi_square_budget",
        "chi-square-budget": "chi_square_budget",
        "chi_square_budget": "chi_square_budget",
        "chi2": "chi_square_budget",
        "chisquare": "chi_square_budget",
        "kl": "kl_budget",
        "kl-budget": "kl_budget",
        "kl_budget": "kl_budget",
        "kullback-leibler": "kl_budget",
        "relative-entropy": "kl_budget",
        "relative_entropy": "kl_budget",
        "w1": "wasserstein",
        "wasserstein": "wasserstein",
    }
    key = preset.name.strip().lower()
    try:
        name = aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Q preset: {preset.name!r}") from exc
    return replace(preset, name=name)


def _bounded_radius(preset: QPreset) -> float:
    radius = 0.5 if preset.radius is None else float(preset.radius)
    if radius < 0:
        raise ValueError("bounded_shift radius must be non-negative")
    return radius


def _tv_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("tv_budget radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("tv_budget radius must be non-negative")
    return radius


def _chi_square_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("chi_square_budget radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("chi_square_budget radius must be non-negative")
    return radius


def _kl_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("kl_budget radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("kl_budget radius must be non-negative")
    return radius


def _wasserstein_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("wasserstein radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("wasserstein radius must be non-negative")
    return radius


def _require_backend(preset: QPreset, expected: str) -> None:
    backend = _backend_name(preset, default=expected)
    if backend != expected:
        raise ValueError(f"{preset.name} currently supports only backend={expected!r}")


def _backend_name(preset: QPreset, *, default: str) -> str:
    return (preset.backend or default).strip().lower()


def _tv_constraint_builder(cell_weights: Mapping[Hashable, float], radius: float):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        return (
            cvxpy_constraint(
                cp.norm1(q - observed) <= 2.0 * radius,
                name="total-variation budget",
                kind="tv_budget",
                sense="<=",
            ),
        )

    return build


def _tv_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    *,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        radius = parameter(parameter_name, nonneg=True)
        return (
            cvxpy_constraint(
                cp.norm1(q - observed) <= 2.0 * radius,
                name="total-variation budget",
                kind="tv_budget",
                sense="<=",
            ),
        )

    return build


def _chi_square_constraint_builder(
    cell_weights: Mapping[Hashable, float], radius: float
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        scale = 1.0 / np.sqrt(observed)
        return (
            cvxpy_constraint(
                cp.sum_squares(cp.multiply(scale, q - observed)) <= radius,
                name="chi-square budget",
                kind="chi_square_budget",
                sense="<=",
            ),
        )

    return build


def _chi_square_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    *,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        scale = 1.0 / np.sqrt(observed)
        radius = parameter(parameter_name, nonneg=True)
        return (
            cvxpy_constraint(
                cp.sum_squares(cp.multiply(scale, q - observed)) <= radius,
                name="chi-square budget",
                kind="chi_square_budget",
                sense="<=",
            ),
        )

    return build


def _kl_constraint_builder(cell_weights: Mapping[Hashable, float], radius: float):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        return (
            cvxpy_constraint(
                cp.sum(cp.rel_entr(q, observed)) <= radius,
                name="KL budget",
                kind="kl_budget",
                sense="<=",
            ),
        )

    return build


def _kl_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    *,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        radius = parameter(parameter_name, nonneg=True)
        return (
            cvxpy_constraint(
                cp.sum(cp.rel_entr(q, observed)) <= radius,
                name="KL budget",
                kind="kl_budget",
                sense="<=",
            ),
        )

    return build


def _wasserstein_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    cost: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    radius: float,
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        cost_matrix = _coerce_cost_matrix(cost, states)
        gamma = cp.Variable((len(states), len(states)), nonneg=True)
        return (
            cvxpy_constraint(
                cp.sum(gamma, axis=1) == observed,
                name="Wasserstein source marginal",
                kind="wasserstein_source_marginal",
                sense="==",
                states=states,
            ),
            cvxpy_constraint(
                cp.sum(gamma, axis=0) == q,
                name="Wasserstein target marginal",
                kind="wasserstein_target_marginal",
                sense="==",
                states=states,
            ),
            cvxpy_constraint(
                cp.sum(cp.multiply(cost_matrix, gamma)) <= radius,
                name="Wasserstein budget",
                kind="wasserstein_budget",
                sense="<=",
            ),
        )

    return build


def _wasserstein_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    cost: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    *,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        cost_matrix = _coerce_cost_matrix(cost, states)
        gamma = cp.Variable((len(states), len(states)), nonneg=True)
        radius = parameter(parameter_name, nonneg=True)
        return (
            cvxpy_constraint(
                cp.sum(gamma, axis=1) == observed,
                name="Wasserstein source marginal",
                kind="wasserstein_source_marginal",
                sense="==",
                states=states,
            ),
            cvxpy_constraint(
                cp.sum(gamma, axis=0) == q,
                name="Wasserstein target marginal",
                kind="wasserstein_target_marginal",
                sense="==",
                states=states,
            ),
            cvxpy_constraint(
                cp.sum(cp.multiply(cost_matrix, gamma)) <= radius,
                name="Wasserstein budget",
                kind="wasserstein_budget",
                sense="<=",
            ),
        )

    return build


def _coerce_cost_matrix(
    cost: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    states: Sequence[Hashable],
):
    import numpy as np

    n = len(states)
    if isinstance(cost, Mapping):
        rows = []
        missing = []
        for left_state in states:
            row = []
            for right_state in states:
                key = (left_state, right_state)
                if key in cost:
                    row.append(_nonnegative_cost(cost[key], key=key))
                elif left_state == right_state:
                    row.append(0.0)
                else:
                    missing.append(key)
                    row.append(0.0)
            rows.append(row)
        if missing:
            preview = ", ".join(repr(key) for key in missing[:3])
            if len(missing) > 3:
                preview += ", ..."
            raise ValueError(f"wasserstein cost matrix is missing pairs: {preview}")
        return np.array(rows, dtype=float)

    if len(cost) != n:
        raise ValueError("wasserstein cost matrix must have one row per hidden state")
    rows = []
    for row_index, row in enumerate(cost):
        if len(row) != n:
            raise ValueError(
                "wasserstein cost matrix must have one column per hidden state"
            )
        rows.append(
            [
                _nonnegative_cost(value, key=(states[row_index], states[column_index]))
                for column_index, value in enumerate(row)
            ]
        )
    return np.array(rows, dtype=float)


def _nonnegative_cost(value: float, *, key: tuple[Hashable, Hashable]) -> float:
    cost = float(value)
    if not isfinite(cost) or cost < 0:
        raise ValueError(
            f"wasserstein cost for {key!r} must be finite and non-negative"
        )
    return cost


def _public_law_constraints(
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
):
    constraints = []
    for public_value, mass in public_law.items():
        coefficients = {
            state: 1.0
            for state, state_public_value in public_map.items()
            if state_public_value == public_value
        }
        constraints.append(
            eq(coefficients, float(mass), name=f"public_law[{public_value!r}]")
        )
    return tuple(constraints)
