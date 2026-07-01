"""Named admissible-environment presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Mapping

from .environments import (
    Environment,
    FiniteEnvironments,
    PolytopeEnvironments,
    PublicFiberSaturated,
    eq,
)


@dataclass(frozen=True)
class QPreset:
    """Reusable preset for constructing an admissible environment ``Q``."""

    name: str
    radius: float | None = None


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

    raise ValueError(f"unsupported Q preset: {preset.name!r}")


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
        if preset.name != "bounded_shift":
            raise ValueError("q_radius is only valid for q='bounded_shift'")
        if preset.radius is not None and float(preset.radius) != float(q_radius):
            raise ValueError("q_radius conflicts with the QPreset radius")
        preset = QPreset(preset.name, float(q_radius))

    if preset.name == "bounded_shift":
        radius = _bounded_radius(preset)
        preset = QPreset(preset.name, radius)

    return preset


def q_name(q: Any, *, q_radius: float | None = None) -> str:
    preset = normalize_q_preset(q, q_radius=q_radius)
    if preset is None:
        return str(getattr(q, "name", q.__class__.__name__))
    if preset.name == "bounded_shift":
        return f"bounded_shift(radius={_bounded_radius(preset):g})"
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
    }
    key = preset.name.strip().lower()
    try:
        name = aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported Q preset: {preset.name!r}") from exc
    return QPreset(name=name, radius=preset.radius)


def _bounded_radius(preset: QPreset) -> float:
    radius = 0.5 if preset.radius is None else float(preset.radius)
    if radius < 0:
        raise ValueError("bounded_shift radius must be non-negative")
    return radius


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
