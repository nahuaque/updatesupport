"""Named admissible-environment presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence

from .environments import (
    BatchedCvxpyEnvironments,
    ConvexAdmissibleSet,
    CvxpyEnvironments,
    CvxpyConstraintBuilder,
    Environment,
    FiniteEnvironments,
    ParameterizedCvxpyEnvironments,
    CvxpyParameterizedConstraintBuilder,
    PolytopeEnvironments,
    PublicFiberSaturated,
    SupportFunctionBackend,
    SupportFunctionIntervalResult,
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
    settings: Mapping[str, Any] | None = None

    def __and__(self, other: Any) -> "QPreset":
        """Return the intersection of two Q presets."""

        return q_intersection(self, other)


@dataclass(frozen=True)
class QEnvironment:
    environment: Environment
    preset: QPreset | None
    name: str
    description: str


@dataclass(frozen=True)
class _CovariateBalanceArrays:
    names: tuple[Hashable, ...]
    matrix: Any
    baseline: Any
    inverse_scale: Any


@dataclass(frozen=True)
class CvxpyAdmissibleSetSpec:
    """Reusable CVXPY constraints for one admissible Q preset."""

    preset: QPreset
    fixed_public_law: Mapping[Hashable, float]
    constraint_builders: Sequence[CvxpyConstraintBuilder] = ()
    parameterized_constraint_builders: Sequence[
        CvxpyParameterizedConstraintBuilder
    ] = ()
    parameter_values: Mapping[str, Any] = field(default_factory=dict)
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixed_public_law",
            {
                public_value: float(mass)
                for public_value, mass in self.fixed_public_law.items()
            },
        )
        object.__setattr__(self, "constraint_builders", tuple(self.constraint_builders))
        object.__setattr__(
            self,
            "parameterized_constraint_builders",
            tuple(self.parameterized_constraint_builders),
        )
        object.__setattr__(self, "parameter_values", dict(self.parameter_values))
        object.__setattr__(self, "name", self.name or q_name(self.preset))
        object.__setattr__(
            self,
            "description",
            self.description or q_description(self.preset),
        )

    def __and__(self, other: "CvxpyAdmissibleSetSpec") -> "CvxpyAdmissibleSetSpec":
        """Return the intersection of two CVXPY admissible-set specs."""

        return self.intersect(other)

    def intersect(
        self,
        *others: "CvxpyAdmissibleSetSpec",
        name: str | None = None,
        description: str | None = None,
        solver: str | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> "CvxpyAdmissibleSetSpec":
        """Conjoin this admissible set's constraints with other specs."""

        specs = (self,) + tuple(others)
        _validate_same_fixed_public_law(specs)
        merged_solver = solver if solver is not None else _merged_solver(specs)
        merged_solver_options = (
            dict(solver_options)
            if solver_options is not None
            else _merged_solver_options(specs)
        )
        preset = q_intersection(
            *(spec.preset for spec in specs),
            solver=merged_solver,
            solver_options=merged_solver_options,
        )
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=self.fixed_public_law,
            constraint_builders=tuple(
                builder for spec in specs for builder in spec.constraint_builders
            ),
            parameterized_constraint_builders=(),
            parameter_values={},
            solver=merged_solver,
            solver_options=merged_solver_options,
            name=name or q_name(preset),
            description=description or q_description(preset),
        )

    def environment(self, backend: str = "cvxpy") -> Environment:
        """Materialize this constraint spec as a CVXPY-compatible environment."""

        backend_key = backend.strip().lower().replace("-", "_")
        if backend_key == "cvxpy":
            return CvxpyEnvironments(
                fixed_public_law=self.fixed_public_law,
                constraint_builders=self.constraint_builders,
                solver=self.solver,
                solver_options=self.solver_options,
                name=self.name or q_name(self.preset),
            )
        if backend_key == "support_function":
            return SupportFunctionBackend(
                fixed_public_law=self.fixed_public_law,
                constraint_builders=self.constraint_builders,
                solver=self.solver,
                solver_options=self.solver_options,
                name=self.name or q_name(self.preset),
            )
        if backend_key == "batched_cvxpy":
            return BatchedCvxpyEnvironments(
                fixed_public_law=self.fixed_public_law,
                constraint_builders=self.constraint_builders,
                solver=self.solver,
                solver_options=self.solver_options,
                name=self.name or q_name(self.preset),
            )
        if backend_key == "parameterized_cvxpy":
            if not self.parameterized_constraint_builders:
                raise ValueError(
                    f"{self.name} does not expose parameterized CVXPY constraints"
                )
            return ParameterizedCvxpyEnvironments(
                fixed_public_law=self.fixed_public_law,
                parameterized_constraint_builders=self.parameterized_constraint_builders,
                parameter_values=self.parameter_values,
                solver=self.solver,
                solver_options=self.solver_options,
                name=self.name or q_name(self.preset),
            )
        raise ValueError(f"unsupported CVXPY admissible-set backend: {backend!r}")

    def convex_admissible_set(self, problem) -> ConvexAdmissibleSet:
        """Build a concrete CVXPY admissible set for ``problem``."""

        env = self.environment("support_function")
        return env.convex_admissible_set(
            problem,
            public_law=self.fixed_public_law,
        )

    def support_interval(
        self,
        problem,
        direction: Sequence[float] | None = None,
    ) -> SupportFunctionIntervalResult:
        """Evaluate ``[-sigma_Q(-h), sigma_Q(h)]`` for this preset spec."""

        env = self.environment("support_function")
        if not isinstance(env, SupportFunctionBackend):
            raise TypeError(
                "support interval evaluation requires SupportFunctionBackend"
            )
        return env.support_interval(
            problem,
            direction,
            public_law=self.fixed_public_law,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "preset": self.preset.name,
            "radius": self.preset.radius,
            "backend": self.preset.backend,
            "fixed_public_law": dict(self.fixed_public_law),
            "constraint_builder_count": len(self.constraint_builders),
            "parameterized_constraint_builder_count": len(
                self.parameterized_constraint_builders
            ),
            "parameter_values": dict(self.parameter_values),
            "solver": self.solver,
            "solver_options": None
            if self.solver_options is None
            else dict(self.solver_options),
        }


def q_saturated() -> QPreset:
    """Allow arbitrary hidden reweighting inside each observed public fiber."""

    return QPreset("saturated")


def q_observed() -> QPreset:
    """Use only the observed hidden distribution."""

    return QPreset("observed")


def q_intersection(
    *components: Any,
    backend: str | None = None,
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Intersect several admissible Q presets.

    The first implementation slice supports convex CVXPY-compatible component
    presets plus ``saturated`` and ``observed``. Mixed-integer components are
    deliberately rejected by the CVXPY admissible-set compiler.
    """

    if not components:
        raise ValueError("q_intersection requires at least one component preset")
    normalized = _flatten_intersection_components(
        tuple(_normalize_intersection_component(component) for component in components)
    )
    component_backends = {
        component.backend for component in normalized if component.backend
    }
    component_solvers = {
        component.solver for component in normalized if component.solver
    }
    final_backend = backend
    if final_backend is None and len(component_backends) == 1:
        final_backend = next(iter(component_backends))
    final_solver = solver
    if final_solver is None and len(component_solvers) == 1:
        final_solver = next(iter(component_solvers))
    return QPreset(
        "intersection",
        backend=final_backend,
        solver=final_solver,
        solver_options=None if solver_options is None else dict(solver_options),
        settings={"components": normalized},
    )


def q_bounded_shift(radius: float = 0.5) -> QPreset:
    """Limit each hidden-cell mass to a relative band around its observed mass."""

    return QPreset("bounded_shift", radius=float(radius))


def q_fiber_support_floor(
    min_active: int,
    *,
    min_share: float,
    max_active: int | None = None,
    backend: str = "cvxpy",
    solver: str | None = "SCIP",
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Require each public fiber to keep several active hidden cells.

    The preset is mixed-integer: inside every retained public fiber, at least
    ``min_active`` hidden cells must carry at least ``min_share`` of that
    public fiber's mass. ``max_active`` optionally caps the number of active
    hidden cells in each public fiber. Use a MIP-capable CVXPY solver such as
    SCIP.
    """

    settings = {
        "min_active": int(min_active),
        "min_share": float(min_share),
    }
    if max_active is not None:
        settings["max_active"] = int(max_active)
    return QPreset(
        "fiber_support_floor",
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
        settings=settings,
    )


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


def q_l2_budget(
    radius: float,
    *,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain L2 distance from the observed hidden distribution."""

    return QPreset(
        "l2_budget",
        radius=float(radius),
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
    )


def q_covariate_balance(
    radius: float,
    moments: Mapping[Hashable, Mapping[Hashable, float] | Sequence[float]]
    | Sequence[Sequence[float]],
    *,
    baseline: Mapping[Hashable, float] | Sequence[float] | None = None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None = None,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain standardized hidden covariate-moment drift.

    ``moments`` maps moment names to values on retained hidden cells, or supplies
    a row-by-cell matrix in hidden-state order. By default the baseline and
    scale are computed from the observed hidden distribution.
    """

    settings: dict[str, Any] = {}
    if baseline is not None:
        settings["baseline"] = baseline
    if scale is not None:
        settings["scale"] = scale
    return QPreset(
        "covariate_balance",
        radius=float(radius),
        cost=moments,
        backend=backend,
        solver=solver,
        solver_options=None if solver_options is None else dict(solver_options),
        settings=settings or None,
    )


def q_mahalanobis_budget(
    radius: float,
    *,
    covariance: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> QPreset:
    """Constrain Mahalanobis distance from the observed hidden distribution."""

    return QPreset(
        "mahalanobis_budget",
        radius=float(radius),
        cost=covariance,
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


def cvxpy_admissible_set_spec(
    q: Any,
    *,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    cell_weights: Mapping[Hashable, float],
    q_radius: float | None = None,
) -> CvxpyAdmissibleSetSpec:
    """Expose CVXPY admissible-set constraints for a compatible Q preset."""

    preset = normalize_q_preset(q, q_radius=q_radius)
    if preset is None:
        raise TypeError("cvxpy_admissible_set_spec requires a named Q preset")
    return _cvxpy_admissible_set_spec_from_preset(
        preset,
        public_law=public_law,
        public_map=public_map,
        cell_weights=cell_weights,
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

    if preset.name == "intersection":
        backend = _backend_name(preset, default="cvxpy")
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    if preset.name == "fiber_support_floor":
        min_active, min_share, max_active = _fiber_support_floor_settings(preset)
        _validate_fiber_support_floor(
            min_active=min_active,
            max_active=max_active,
            public_law=public_law,
            public_map=public_map,
            states=tuple(cell_weights),
        )
        backend = _backend_name(preset, default="cvxpy")
        builder = _fiber_support_floor_constraint_builder(
            public_law=public_law,
            public_map=public_map,
            min_active=min_active,
            min_share=min_share,
            max_active=max_active,
        )
        solver = preset.solver or "SCIP"
        if backend == "batched_cvxpy":
            return QEnvironment(
                environment=BatchedCvxpyEnvironments(
                    fixed_public_law=public_law,
                    constraint_builders=(builder,),
                    solver=solver,
                    solver_options=preset.solver_options,
                    name=q_name(preset),
                ),
                preset=preset,
                name=q_name(preset),
                description=q_description(preset),
            )
        _require_backend(preset, "cvxpy")
        return QEnvironment(
            environment=CvxpyEnvironments(
                fixed_public_law=public_law,
                constraint_builders=(builder,),
                solver=solver,
                solver_options=preset.solver_options,
                name=q_name(preset),
            ),
            preset=preset,
            name=q_name(preset),
            description=q_description(preset),
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
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

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
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

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
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    if preset.name == "l2_budget":
        radius = _l2_radius(preset)
        if radius == 0.0:
            return _point_environment(
                cell_weights,
                preset,
                description=(
                    "fixed observed public law with L2 distance from the observed "
                    "hidden distribution <= 0"
                ),
            )
        backend = _backend_name(preset, default="cvxpy")
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    if preset.name == "covariate_balance":
        _covariate_balance_radius(preset)
        if preset.cost is None:
            raise ValueError("q_covariate_balance requires moment values")
        backend = _backend_name(preset, default="cvxpy")
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    if preset.name == "mahalanobis_budget":
        radius = _mahalanobis_radius(preset)
        if preset.cost is None:
            raise ValueError("q_mahalanobis_budget requires a covariance matrix")
        _mahalanobis_transform(preset.cost, tuple(cell_weights))
        if radius == 0.0:
            return _point_environment(
                cell_weights,
                preset,
                description=(
                    "fixed observed public law with Mahalanobis distance from "
                    "the observed hidden distribution <= 0"
                ),
            )
        backend = _backend_name(preset, default="cvxpy")
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    if preset.name == "wasserstein":
        if preset.cost is None:
            raise ValueError("q_wasserstein requires an explicit cost matrix")
        backend = _backend_name(preset, default="cvxpy")
        spec = _cvxpy_admissible_set_spec_from_preset(
            preset,
            public_law=public_law,
            public_map=public_map,
            cell_weights=cell_weights,
        )
        return _q_environment_from_cvxpy_spec(spec, backend=backend)

    raise ValueError(f"unsupported Q preset: {preset.name!r}")


def _cvxpy_admissible_set_spec_from_preset(
    preset: QPreset,
    *,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    cell_weights: Mapping[Hashable, float],
) -> CvxpyAdmissibleSetSpec:
    _validate_cvxpy_spec_inputs(
        public_law=public_law,
        public_map=public_map,
        cell_weights=cell_weights,
    )
    if preset.name == "intersection":
        components = _intersection_components(preset)
        component_specs = []
        observed_builders = []
        saturated_count = 0
        for component in components:
            if component.name == "saturated":
                saturated_count += 1
                continue
            if component.name == "observed":
                observed_builders.append(_observed_constraint_builder(cell_weights))
                continue
            component_specs.append(
                _cvxpy_admissible_set_spec_from_preset(
                    component,
                    public_law=public_law,
                    public_map=public_map,
                    cell_weights=cell_weights,
                )
            )
        if not component_specs and not observed_builders and saturated_count == 0:
            raise ValueError("q_intersection requires at least one component preset")
        all_specs = tuple(component_specs)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=tuple(observed_builders)
            + tuple(
                builder for spec in all_specs for builder in spec.constraint_builders
            ),
            parameterized_constraint_builders=(),
            parameter_values={},
            solver=preset.solver
            if preset.solver is not None
            else _merged_solver(all_specs),
            solver_options=preset.solver_options
            if preset.solver_options is not None
            else _merged_solver_options(all_specs),
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(
                _bounded_shift_constraint_builder(cell_weights, radius),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "tv_budget":
        radius = _tv_radius(preset)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(_tv_constraint_builder(cell_weights, radius),),
            parameterized_constraint_builders=(
                _tv_parameterized_constraint_builder(cell_weights),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "chi_square_budget":
        radius = _chi_square_radius(preset)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(_chi_square_constraint_builder(cell_weights, radius),),
            parameterized_constraint_builders=(
                _chi_square_parameterized_constraint_builder(cell_weights),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "kl_budget":
        radius = _kl_radius(preset)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(_kl_constraint_builder(cell_weights, radius),),
            parameterized_constraint_builders=(
                _kl_parameterized_constraint_builder(cell_weights),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "l2_budget":
        radius = _l2_radius(preset)
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(_l2_constraint_builder(cell_weights, radius),),
            parameterized_constraint_builders=(
                _l2_parameterized_constraint_builder(cell_weights),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "covariate_balance":
        radius = _covariate_balance_radius(preset)
        if preset.cost is None:
            raise ValueError("q_covariate_balance requires moment values")
        baseline = None if preset.settings is None else preset.settings.get("baseline")
        scale = None if preset.settings is None else preset.settings.get("scale")
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(
                _covariate_balance_constraint_builder(
                    cell_weights,
                    preset.cost,
                    radius,
                    baseline=baseline,
                    scale=scale,
                ),
            ),
            parameterized_constraint_builders=(
                _covariate_balance_parameterized_constraint_builder(
                    cell_weights,
                    preset.cost,
                    baseline=baseline,
                    scale=scale,
                ),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "mahalanobis_budget":
        radius = _mahalanobis_radius(preset)
        if preset.cost is None:
            raise ValueError("q_mahalanobis_budget requires a covariance matrix")
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(
                _mahalanobis_constraint_builder(cell_weights, preset.cost, radius),
            ),
            parameterized_constraint_builders=(
                _mahalanobis_parameterized_constraint_builder(
                    cell_weights,
                    preset.cost,
                ),
            ),
            parameter_values={"radius": radius},
            solver=preset.solver,
            solver_options=preset.solver_options,
            name=q_name(preset),
            description=q_description(preset),
        )

    if preset.name == "wasserstein":
        radius = _wasserstein_radius(preset)
        if preset.cost is None:
            raise ValueError("q_wasserstein requires an explicit cost matrix")
        return CvxpyAdmissibleSetSpec(
            preset=preset,
            fixed_public_law=public_law,
            constraint_builders=(
                _wasserstein_constraint_builder(cell_weights, preset.cost, radius),
            ),
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
            description=q_description(preset),
        )

    if preset.name == "fiber_support_floor":
        raise ValueError(
            "fiber_support_floor is mixed-integer and does not expose a convex "
            "CVXPY admissible-set spec"
        )
    raise ValueError(
        f"{preset.name!r} does not expose CVXPY admissible-set constraints"
    )


def _validate_cvxpy_spec_inputs(
    *,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    cell_weights: Mapping[Hashable, float],
) -> None:
    missing_states = [state for state in cell_weights if state not in public_map]
    if missing_states:
        raise ValueError(f"public_map is missing states: {missing_states!r}")
    missing_public = [
        public_map[state]
        for state in cell_weights
        if public_map[state] not in public_law
    ]
    if missing_public:
        raise ValueError(f"public_law is missing public values: {missing_public!r}")


def _validate_same_fixed_public_law(
    specs: Sequence[CvxpyAdmissibleSetSpec],
) -> None:
    if not specs:
        return
    first = specs[0].fixed_public_law
    for spec in specs[1:]:
        if set(first) != set(spec.fixed_public_law):
            raise ValueError("intersected specs must use the same fixed public law")
        for public_value, mass in first.items():
            if abs(float(mass) - float(spec.fixed_public_law[public_value])) > 1e-9:
                raise ValueError("intersected specs must use the same fixed public law")


def _merged_solver(specs: Sequence[CvxpyAdmissibleSetSpec]) -> str | None:
    solvers = {spec.solver for spec in specs if spec.solver is not None}
    if len(solvers) > 1:
        raise ValueError(
            "intersected specs have conflicting solvers; pass solver=... explicitly"
        )
    return next(iter(solvers)) if solvers else None


def _merged_solver_options(
    specs: Sequence[CvxpyAdmissibleSetSpec],
) -> Mapping[str, Any] | None:
    options = [
        dict(spec.solver_options) for spec in specs if spec.solver_options is not None
    ]
    if not options:
        return None
    first = options[0]
    if any(option != first for option in options[1:]):
        raise ValueError(
            "intersected specs have conflicting solver_options; pass "
            "solver_options=... explicitly"
        )
    return first


def _q_environment_from_cvxpy_spec(
    spec: CvxpyAdmissibleSetSpec,
    *,
    backend: str,
) -> QEnvironment:
    return QEnvironment(
        environment=spec.environment(backend),
        preset=spec.preset,
        name=spec.name or q_name(spec.preset),
        description=spec.description or q_description(spec.preset),
    )


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
            "covariate_balance",
            "kl_budget",
            "l2_budget",
            "mahalanobis_budget",
            "tv_budget",
            "wasserstein",
        }:
            raise ValueError(
                "q_radius is only valid for bounded, chi-square, KL, L2, "
                "covariate-balance, Mahalanobis, TV, or Wasserstein Q presets"
            )
        if preset.radius is not None and float(preset.radius) != float(q_radius):
            raise ValueError("q_radius conflicts with the QPreset radius")
        preset = replace(preset, radius=float(q_radius))

    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "fiber_support_floor":
        _fiber_support_floor_settings(preset)
    elif preset.name == "intersection":
        if q_radius is not None:
            raise ValueError("q_radius cannot be used with q_intersection")
        preset = replace(
            preset,
            settings={"components": _intersection_components(preset)},
        )
    elif preset.name == "tv_budget":
        radius = _tv_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "chi_square_budget":
        radius = _chi_square_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "kl_budget":
        radius = _kl_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "l2_budget":
        radius = _l2_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "covariate_balance":
        radius = _covariate_balance_radius(preset)
        preset = replace(preset, radius=radius)
    elif preset.name == "mahalanobis_budget":
        radius = _mahalanobis_radius(preset)
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
    if preset.name == "fiber_support_floor":
        min_active, min_share, max_active = _fiber_support_floor_settings(preset)
        name = f"fiber_support_floor(min_active={min_active}, min_share={min_share:g}"
        if max_active is not None:
            name += f", max_active={max_active}"
        return name + ")"
    if preset.name == "intersection":
        components = ", ".join(
            q_name(component) for component in _intersection_components(preset)
        )
        return f"intersection({components})"
    if preset.name == "tv_budget":
        return f"tv_budget(radius={_tv_radius(preset):g})"
    if preset.name == "chi_square_budget":
        return f"chi_square_budget(radius={_chi_square_radius(preset):g})"
    if preset.name == "kl_budget":
        return f"kl_budget(radius={_kl_radius(preset):g})"
    if preset.name == "l2_budget":
        return f"l2_budget(radius={_l2_radius(preset):g})"
    if preset.name == "covariate_balance":
        return f"covariate_balance(radius={_covariate_balance_radius(preset):g})"
    if preset.name == "mahalanobis_budget":
        return f"mahalanobis_budget(radius={_mahalanobis_radius(preset):g})"
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
    if preset.name == "fiber_support_floor":
        min_active, min_share, max_active = _fiber_support_floor_settings(preset)
        description = (
            "fixed observed public law with at least "
            f"{min_active} active hidden cells per public fiber, each carrying "
            f"at least {100 * min_share:g}% of that public fiber's mass"
        )
        if max_active is not None:
            description += f", and at most {max_active} active hidden cells"
        return description
    if preset.name == "intersection":
        components = "; ".join(
            q_description(component) for component in _intersection_components(preset)
        )
        return f"intersection of admissible Q presets: {components}"
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
    if preset.name == "l2_budget":
        radius = _l2_radius(preset)
        return (
            "fixed observed public law with L2 distance from the observed "
            f"hidden distribution <= {radius:g}"
        )
    if preset.name == "covariate_balance":
        radius = _covariate_balance_radius(preset)
        return (
            "fixed observed public law with standardized hidden covariate-moment "
            f"shift <= {radius:g}"
        )
    if preset.name == "mahalanobis_budget":
        radius = _mahalanobis_radius(preset)
        return (
            "fixed observed public law with covariance-standardized Mahalanobis "
            f"distance from the observed hidden distribution <= {radius:g}"
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
        "and": "intersection",
        "intersect": "intersection",
        "intersection": "intersection",
        "meet": "intersection",
        "q_intersection": "intersection",
        "bounded": "bounded_shift",
        "bounded-shift": "bounded_shift",
        "bounded_shift": "bounded_shift",
        "fiber-support-floor": "fiber_support_floor",
        "fiber_support": "fiber_support_floor",
        "fiber_support_floor": "fiber_support_floor",
        "support-floor": "fiber_support_floor",
        "support_floor": "fiber_support_floor",
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
        "l2": "l2_budget",
        "l2-budget": "l2_budget",
        "l2_budget": "l2_budget",
        "euclidean": "l2_budget",
        "euclidean_budget": "l2_budget",
        "balance": "covariate_balance",
        "balance-budget": "covariate_balance",
        "balance_budget": "covariate_balance",
        "covariate-balance": "covariate_balance",
        "covariate_balance": "covariate_balance",
        "covariate-balance-budget": "covariate_balance",
        "covariate_balance_budget": "covariate_balance",
        "moment-balance": "covariate_balance",
        "moment_balance": "covariate_balance",
        "mahalanobis": "mahalanobis_budget",
        "mahalanobis-budget": "mahalanobis_budget",
        "mahalanobis_budget": "mahalanobis_budget",
        "ellipsoid": "mahalanobis_budget",
        "ellipsoidal": "mahalanobis_budget",
        "w1": "wasserstein",
        "wasserstein": "wasserstein",
    }
    key = preset.name.strip().lower()
    try:
        name = aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Q preset: {preset.name!r}") from exc
    return replace(preset, name=name)


def _intersection_components(preset: QPreset) -> tuple[QPreset, ...]:
    settings = dict(preset.settings or {})
    raw_components = settings.get("components")
    if raw_components is None:
        raise ValueError("q_intersection requires settings with 'components'")
    if isinstance(raw_components, str) or not isinstance(raw_components, Sequence):
        raise TypeError("q_intersection components must be a sequence of Q presets")
    components = tuple(
        _normalize_intersection_component(component) for component in raw_components
    )
    flattened = _flatten_intersection_components(components)
    if not flattened:
        raise ValueError("q_intersection requires at least one component preset")
    return flattened


def _flatten_intersection_components(
    components: Sequence[QPreset],
) -> tuple[QPreset, ...]:
    flattened: list[QPreset] = []
    for component in components:
        if component.name == "intersection":
            flattened.extend(_intersection_components(component))
        else:
            flattened.append(component)
    return tuple(flattened)


def _normalize_intersection_component(component: Any) -> QPreset:
    if isinstance(component, Mapping):
        component = _q_preset_from_mapping(component)
    preset = normalize_q_preset(component)
    if preset is None:
        raise TypeError("q_intersection components must be named built-in Q presets")
    return preset


def _q_preset_from_mapping(value: Mapping[str, Any]) -> QPreset:
    if "name" not in value:
        raise ValueError("Q preset mapping must contain 'name'")
    return QPreset(
        name=str(value["name"]),
        radius=None if value.get("radius") is None else float(value["radius"]),
        cost=value.get("cost"),
        backend=None if value.get("backend") is None else str(value["backend"]),
        solver=None if value.get("solver") is None else str(value["solver"]),
        solver_options=value.get("solver_options"),
        settings=value.get("settings"),
    )


def _bounded_radius(preset: QPreset) -> float:
    radius = 0.5 if preset.radius is None else float(preset.radius)
    if radius < 0:
        raise ValueError("bounded_shift radius must be non-negative")
    return radius


def _fiber_support_floor_settings(
    preset: QPreset,
) -> tuple[int, float, int | None]:
    settings = dict(preset.settings or {})
    try:
        min_active = int(settings["min_active"])
        min_share = float(settings["min_share"])
    except KeyError as exc:
        raise ValueError(
            "fiber_support_floor requires settings with 'min_active' and 'min_share'"
        ) from exc

    max_active_value = settings.get("max_active")
    max_active = None if max_active_value is None else int(max_active_value)
    if min_active <= 0:
        raise ValueError("fiber_support_floor min_active must be positive")
    if max_active is not None and max_active < min_active:
        raise ValueError("fiber_support_floor max_active must be >= min_active")
    if min_share <= 0.0:
        raise ValueError("fiber_support_floor min_share must be positive")
    if min_active * min_share > 1.0 + 1e-12:
        raise ValueError("fiber_support_floor min_active * min_share must be <= 1")
    return min_active, min_share, max_active


def _validate_fiber_support_floor(
    *,
    min_active: int,
    max_active: int | None,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    states: tuple[Hashable, ...],
) -> None:
    counts = {public_value: 0 for public_value in public_law}
    for state in states:
        counts[public_map[state]] = counts.get(public_map[state], 0) + 1

    sparse = [
        public_value
        for public_value, mass in public_law.items()
        if float(mass) > 0.0 and counts.get(public_value, 0) < min_active
    ]
    if sparse:
        raise ValueError(
            "fiber_support_floor requires at least min_active retained hidden "
            f"cells in each positive-mass public fiber; failing fibers: {sparse!r}"
        )
    if max_active is not None:
        empty_capacity = [
            public_value
            for public_value, mass in public_law.items()
            if float(mass) > 0.0 and counts.get(public_value, 0) == 0
        ]
        if empty_capacity:
            raise ValueError(
                "fiber_support_floor found positive-mass public fibers without "
                f"retained hidden cells: {empty_capacity!r}"
            )


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


def _l2_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("l2_budget radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("l2_budget radius must be non-negative")
    return radius


def _covariate_balance_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("covariate_balance radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("covariate_balance radius must be non-negative")
    return radius


def _mahalanobis_radius(preset: QPreset) -> float:
    if preset.radius is None:
        raise ValueError("mahalanobis_budget radius is required")
    radius = float(preset.radius)
    if radius < 0:
        raise ValueError("mahalanobis_budget radius must be non-negative")
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


def _fiber_support_floor_constraint_builder(
    *,
    public_law: Mapping[Hashable, float],
    public_map: Mapping[Hashable, Hashable],
    min_active: int,
    min_share: float,
    max_active: int | None,
):
    public_masses = {
        public_value: float(mass) for public_value, mass in public_law.items()
    }

    def build(cp, q, states, state_index):
        z = cp.Variable(len(states), boolean=True)
        records = []
        for state in states:
            public_value = public_map[state]
            public_mass = public_masses[public_value]
            index = state_index[state]
            records.append(
                cvxpy_constraint(
                    q[index] <= public_mass * z[index],
                    name="fiber support inactive-cell upper bound",
                    kind="support_activation",
                    sense="<=",
                    state=state,
                    public_value=public_value,
                )
            )
            records.append(
                cvxpy_constraint(
                    q[index] >= min_share * public_mass * z[index],
                    name="fiber support active-cell mass floor",
                    kind="support_activation",
                    sense=">=",
                    state=state,
                    public_value=public_value,
                )
            )

        fibers: dict[Hashable, list[int]] = {}
        for state in states:
            fibers.setdefault(public_map[state], []).append(state_index[state])

        for public_value, indices in fibers.items():
            if public_masses[public_value] <= 0.0:
                continue
            records.append(
                cvxpy_constraint(
                    cp.sum(z[indices]) >= min_active,
                    name=f"fiber support active-cell floor {public_value!r}",
                    kind="support_cardinality",
                    sense=">=",
                    public_value=public_value,
                )
            )
            if max_active is not None:
                records.append(
                    cvxpy_constraint(
                        cp.sum(z[indices]) <= max_active,
                        name=f"fiber support active-cell cap {public_value!r}",
                        kind="support_cardinality",
                        sense="<=",
                        public_value=public_value,
                    )
                )
        return tuple(records)

    return build


def _observed_constraint_builder(cell_weights: Mapping[Hashable, float]):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        return (
            cvxpy_constraint(
                q == observed,
                name="observed hidden distribution",
                kind="observed",
                sense="==",
                states=states,
            ),
        )

    return build


def _bounded_shift_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    radius: float,
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, state_index):
        records = []
        for state in states:
            observed = observed_by_state[state]
            lower = max(0.0, observed * (1.0 - radius))
            upper = min(1.0, observed * (1.0 + radius))
            index = state_index[state]
            records.append(
                cvxpy_constraint(
                    q[index] >= lower,
                    name=f"bounded-shift lower {state!r}",
                    kind="bounded_shift",
                    sense=">=",
                    state=state,
                )
            )
            records.append(
                cvxpy_constraint(
                    q[index] <= upper,
                    name=f"bounded-shift upper {state!r}",
                    kind="bounded_shift",
                    sense="<=",
                    state=state,
                )
            )
        return tuple(records)

    return build


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


def _l2_constraint_builder(cell_weights: Mapping[Hashable, float], radius: float):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        return (
            cvxpy_constraint(
                cp.norm(q - observed, 2) <= radius,
                name="L2 budget",
                kind="l2_budget",
                sense="<=",
            ),
        )

    return build


def _l2_parameterized_constraint_builder(
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
                cp.norm(q - observed, 2) <= radius,
                name="L2 budget",
                kind="l2_budget",
                sense="<=",
            ),
        )

    return build


def _covariate_balance_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    moments: Mapping[Hashable, Mapping[Hashable, float] | Sequence[float]]
    | Sequence[Sequence[float]],
    radius: float,
    *,
    baseline: Mapping[Hashable, float] | Sequence[float] | None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None,
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        arrays = _covariate_balance_arrays(
            moments,
            states,
            observed_by_state,
            baseline=baseline,
            scale=scale,
        )
        shift = arrays.matrix @ q - arrays.baseline
        standardized_shift = cp.multiply(arrays.inverse_scale, shift)
        return (
            cvxpy_constraint(
                cp.norm(standardized_shift, 2) <= radius,
                name="covariate-balance budget",
                kind="covariate_balance",
                sense="<=",
                states=states,
            ),
        )

    return build


def _covariate_balance_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    moments: Mapping[Hashable, Mapping[Hashable, float] | Sequence[float]]
    | Sequence[Sequence[float]],
    *,
    baseline: Mapping[Hashable, float] | Sequence[float] | None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        arrays = _covariate_balance_arrays(
            moments,
            states,
            observed_by_state,
            baseline=baseline,
            scale=scale,
        )
        radius = parameter(parameter_name, nonneg=True)
        shift = arrays.matrix @ q - arrays.baseline
        standardized_shift = cp.multiply(arrays.inverse_scale, shift)
        return (
            cvxpy_constraint(
                cp.norm(standardized_shift, 2) <= radius,
                name="covariate-balance budget",
                kind="covariate_balance",
                sense="<=",
                states=states,
            ),
        )

    return build


def _mahalanobis_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    covariance: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    radius: float,
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        transform = _mahalanobis_transform(covariance, states)
        return (
            cvxpy_constraint(
                cp.norm(transform @ (q - observed), 2) <= radius,
                name="Mahalanobis budget",
                kind="mahalanobis_budget",
                sense="<=",
            ),
        )

    return build


def _mahalanobis_parameterized_constraint_builder(
    cell_weights: Mapping[Hashable, float],
    covariance: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    *,
    parameter_name: str = "radius",
):
    observed_by_state = {state: float(mass) for state, mass in cell_weights.items()}

    def build(cp, q, states, _state_index, parameter):
        import numpy as np

        observed = np.array([observed_by_state[state] for state in states], dtype=float)
        transform = _mahalanobis_transform(covariance, states)
        radius = parameter(parameter_name, nonneg=True)
        return (
            cvxpy_constraint(
                cp.norm(transform @ (q - observed), 2) <= radius,
                name="Mahalanobis budget",
                kind="mahalanobis_budget",
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


def _covariate_balance_arrays(
    moments: Mapping[Hashable, Mapping[Hashable, float] | Sequence[float]]
    | Sequence[Sequence[float]],
    states: Sequence[Hashable],
    observed_by_state: Mapping[Hashable, float],
    *,
    baseline: Mapping[Hashable, float] | Sequence[float] | None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None,
) -> _CovariateBalanceArrays:
    import numpy as np

    names, rows = _coerce_covariate_moment_rows(moments, states)
    matrix = np.array(rows, dtype=float)
    observed = np.array([observed_by_state[state] for state in states], dtype=float)
    baseline_vector = _coerce_covariate_baseline(
        baseline,
        names=names,
        matrix=matrix,
        observed=observed,
    )
    scale_vector = _coerce_covariate_scale(
        scale,
        names=names,
        matrix=matrix,
        observed=observed,
    )
    return _CovariateBalanceArrays(
        names=names,
        matrix=matrix,
        baseline=baseline_vector,
        inverse_scale=1.0 / scale_vector,
    )


def _coerce_covariate_moment_rows(
    moments: Mapping[Hashable, Mapping[Hashable, float] | Sequence[float]]
    | Sequence[Sequence[float]],
    states: Sequence[Hashable],
) -> tuple[tuple[Hashable, ...], list[list[float]]]:
    if isinstance(moments, Mapping):
        if not moments:
            raise ValueError("q_covariate_balance requires at least one moment")
        names = tuple(moments)
        rows = [
            _coerce_covariate_moment_values(
                moments[name],
                states,
                moment_name=name,
            )
            for name in names
        ]
        return names, rows

    if isinstance(moments, str | bytes):
        raise TypeError("q_covariate_balance moments must be a mapping or matrix")

    rows = []
    for index, row in enumerate(moments):
        rows.append(
            _coerce_covariate_moment_values(
                row,
                states,
                moment_name=f"moment_{index + 1}",
            )
        )
    if not rows:
        raise ValueError("q_covariate_balance requires at least one moment")
    names = tuple(f"moment_{index + 1}" for index in range(len(rows)))
    return names, rows


def _coerce_covariate_moment_values(
    values: Mapping[Hashable, float] | Sequence[float],
    states: Sequence[Hashable],
    *,
    moment_name: Hashable,
) -> list[float]:
    if isinstance(values, Mapping):
        missing = [state for state in states if state not in values]
        if missing:
            preview = ", ".join(repr(state) for state in missing[:3])
            if len(missing) > 3:
                preview += ", ..."
            raise ValueError(
                "q_covariate_balance moment "
                f"{moment_name!r} is missing hidden states: {preview}"
            )
        return [
            _finite_covariate_value(values[state], label=f"moment {moment_name!r}")
            for state in states
        ]

    if isinstance(values, str | bytes):
        raise TypeError(
            f"q_covariate_balance moment {moment_name!r} must be numeric values"
        )
    if len(values) != len(states):
        raise ValueError(
            "q_covariate_balance sequence moments must have one value per hidden "
            f"state; moment {moment_name!r} has {len(values)} values for "
            f"{len(states)} states"
        )
    return [
        _finite_covariate_value(value, label=f"moment {moment_name!r}")
        for value in values
    ]


def _coerce_covariate_baseline(
    baseline: Mapping[Hashable, float] | Sequence[float] | None,
    *,
    names: Sequence[Hashable],
    matrix: Any,
    observed: Any,
):
    import numpy as np

    if baseline is None:
        return matrix @ observed
    if isinstance(baseline, Mapping):
        missing = [name for name in names if name not in baseline]
        if missing:
            raise ValueError(
                f"q_covariate_balance baseline is missing moments: {missing[:3]!r}"
            )
        return np.array(
            [
                _finite_covariate_value(
                    baseline[name],
                    label=f"baseline {name!r}",
                )
                for name in names
            ],
            dtype=float,
        )
    if isinstance(baseline, str | bytes):
        raise TypeError("q_covariate_balance baseline must be numeric values")
    if len(baseline) != len(names):
        raise ValueError("q_covariate_balance baseline must have one value per moment")
    return np.array(
        [_finite_covariate_value(value, label="baseline") for value in baseline],
        dtype=float,
    )


def _coerce_covariate_scale(
    scale: Mapping[Hashable, float] | Sequence[float] | float | None,
    *,
    names: Sequence[Hashable],
    matrix: Any,
    observed: Any,
):
    import numpy as np

    if scale is None:
        total = float(np.sum(observed))
        if total <= 0.0:
            raise ValueError("q_covariate_balance observed mass must be positive")
        mean = (matrix @ observed) / total
        second_moment = (matrix * matrix) @ observed / total
        variance = np.maximum(second_moment - mean * mean, 0.0)
        scale_vector = np.sqrt(variance)
        scale_vector[scale_vector <= 1e-12] = 1.0
        return scale_vector

    if isinstance(scale, int | float):
        scalar = _positive_covariate_scale(scale, label="scale")
        return np.full(len(names), scalar, dtype=float)

    if isinstance(scale, Mapping):
        missing = [name for name in names if name not in scale]
        if missing:
            raise ValueError(
                f"q_covariate_balance scale is missing moments: {missing[:3]!r}"
            )
        return np.array(
            [
                _positive_covariate_scale(scale[name], label=f"scale {name!r}")
                for name in names
            ],
            dtype=float,
        )

    if isinstance(scale, str | bytes):
        raise TypeError("q_covariate_balance scale must be numeric values")
    if len(scale) != len(names):
        raise ValueError("q_covariate_balance scale must have one value per moment")
    return np.array(
        [_positive_covariate_scale(value, label="scale") for value in scale],
        dtype=float,
    )


def _finite_covariate_value(value: float, *, label: str) -> float:
    coerced = float(value)
    if not isfinite(coerced):
        raise ValueError(f"q_covariate_balance {label} must be finite")
    return coerced


def _positive_covariate_scale(value: float, *, label: str) -> float:
    scale = _finite_covariate_value(value, label=label)
    if scale <= 0.0:
        raise ValueError(f"q_covariate_balance {label} must be positive")
    return scale


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


def _mahalanobis_transform(
    covariance: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    states: Sequence[Hashable],
):
    import numpy as np

    covariance_matrix = _coerce_covariance_matrix(covariance, states)
    try:
        cholesky = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "mahalanobis covariance matrix must be positive definite"
        ) from exc
    identity = np.eye(len(states), dtype=float)
    return np.linalg.solve(cholesky, identity)


def _coerce_covariance_matrix(
    covariance: Mapping[tuple[Hashable, Hashable], float] | Sequence[Sequence[float]],
    states: Sequence[Hashable],
):
    import numpy as np

    n = len(states)
    if isinstance(covariance, Mapping):
        rows = []
        missing = []
        for left_state in states:
            row = []
            for right_state in states:
                key = (left_state, right_state)
                if key in covariance:
                    row.append(_finite_covariance(covariance[key], key=key))
                else:
                    missing.append(key)
                    row.append(0.0)
            rows.append(row)
        if missing:
            preview = ", ".join(repr(key) for key in missing[:3])
            if len(missing) > 3:
                preview += ", ..."
            raise ValueError(
                f"mahalanobis covariance matrix is missing pairs: {preview}"
            )
        matrix = np.array(rows, dtype=float)
    else:
        if len(covariance) != n:
            raise ValueError(
                "mahalanobis covariance matrix must have one row per hidden state"
            )
        rows = []
        for row_index, row in enumerate(covariance):
            if len(row) != n:
                raise ValueError(
                    "mahalanobis covariance matrix must have one column per hidden "
                    "state"
                )
            rows.append(
                [
                    _finite_covariance(
                        value,
                        key=(states[row_index], states[column_index]),
                    )
                    for column_index, value in enumerate(row)
                ]
            )
        matrix = np.array(rows, dtype=float)

    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError("mahalanobis covariance matrix must be symmetric")
    return matrix


def _nonnegative_cost(value: float, *, key: tuple[Hashable, Hashable]) -> float:
    cost = float(value)
    if not isfinite(cost) or cost < 0:
        raise ValueError(
            f"wasserstein cost for {key!r} must be finite and non-negative"
        )
    return cost


def _finite_covariance(value: float, *, key: tuple[Hashable, Hashable]) -> float:
    covariance = float(value)
    if not isfinite(covariance):
        raise ValueError(f"mahalanobis covariance for {key!r} must be finite")
    return covariance


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
