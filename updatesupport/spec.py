"""Serializable audit specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .certificate import (
    RepresentationStabilityCertificate,
    certify_public_representation,
)
from .frontier import PublicRepresentationFrontier, public_representation_frontier
from .presets import QPreset, normalize_q_preset
from .report import PublicDescentReport, SensitivityReport
from .report import public_descent_report, sensitivity_report


AuditReport = (
    PublicDescentReport
    | SensitivityReport
    | PublicRepresentationFrontier
    | RepresentationStabilityCertificate
)


@dataclass(frozen=True)
class QSpec:
    """Serializable description of a built-in Q preset."""

    name: str
    radius: float | None = None
    backend: str | None = None
    cost: Any | None = None
    solver: str | None = None
    solver_options: Mapping[str, Any] | None = None
    settings: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.solver_options is not None and not isinstance(
            self.solver_options, Mapping
        ):
            raise TypeError("solver_options must be a mapping or None")
        if self.settings is not None and not isinstance(self.settings, Mapping):
            raise TypeError("settings must be a mapping or None")
        preset = normalize_q_preset(
            QPreset(
                name=self.name,
                radius=self.radius,
                cost=self.cost,
                backend=self.backend,
                solver=self.solver,
                solver_options=self.solver_options,
                settings=self.settings,
            )
        )
        if preset is None:
            raise TypeError("QSpec only supports built-in Q presets")
        object.__setattr__(self, "name", preset.name)
        object.__setattr__(self, "radius", preset.radius)
        object.__setattr__(self, "cost", preset.cost)
        object.__setattr__(self, "backend", preset.backend)
        object.__setattr__(self, "solver", preset.solver)
        object.__setattr__(
            self,
            "solver_options",
            None if preset.solver_options is None else dict(preset.solver_options),
        )
        object.__setattr__(
            self,
            "settings",
            None if preset.settings is None else dict(preset.settings),
        )

    @classmethod
    def from_value(cls, value: Any) -> "QSpec":
        """Normalize a string, mapping, QPreset, or QSpec into a QSpec."""

        if isinstance(value, QSpec):
            return value
        if isinstance(value, QPreset):
            return cls(
                name=value.name,
                radius=value.radius,
                backend=value.backend,
                cost=value.cost,
                solver=value.solver,
                solver_options=value.solver_options,
                settings=value.settings,
            )
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            unknown_keys = set(value) - {
                "name",
                "radius",
                "backend",
                "cost",
                "solver",
                "solver_options",
                "settings",
            }
            if unknown_keys:
                raise ValueError(
                    "QSpec contains unsupported keys: "
                    f"{sorted(str(key) for key in unknown_keys)!r}"
                )
            if "name" not in value:
                raise ValueError("QSpec mapping must contain 'name'")
            backend = value.get("backend")
            solver = value.get("solver")
            solver_options = value.get("solver_options")
            settings = value.get("settings")
            return cls(
                name=str(value["name"]),
                radius=None if value.get("radius") is None else float(value["radius"]),
                backend=None if backend is None else str(backend),
                cost=value.get("cost"),
                solver=None if solver is None else str(solver),
                solver_options=solver_options,
                settings=settings,
            )
        raise TypeError(
            "Q presets in AuditSpec must be strings, mappings, QPreset, or QSpec"
        )

    def to_preset(self) -> QPreset:
        """Return the runtime QPreset represented by this spec."""

        return QPreset(
            name=self.name,
            radius=self.radius,
            cost=self.cost,
            backend=self.backend,
            solver=self.solver,
            solver_options=self.solver_options,
            settings=self.settings,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.radius is not None:
            payload["radius"] = self.radius
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.solver is not None:
            payload["solver"] = self.solver
        if self.solver_options is not None:
            payload["solver_options"] = dict(self.solver_options)
        if self.settings is not None:
            payload["settings"] = _q_settings_payload(self.name, self.settings)
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Alias for as_dict(), matching AuditSpec."""

        return self.as_dict()


def _q_settings_payload(
    name: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(settings)
    if name == "intersection" and "components" in payload:
        payload["components"] = [
            QSpec.from_value(component).as_dict() for component in payload["components"]
        ]
    return payload


@dataclass(frozen=True)
class AuditSpec:
    """Serializable configuration for a standard update-support audit."""

    public: Sequence[str]
    hidden: Sequence[str]
    target: str
    kind: str = "public_descent"
    weight: str | None = None
    candidate_refinements: Sequence[str] = ()
    q: Any = "saturated"
    q_presets: Sequence[Any] | None = None
    min_cell_weight: float = 1.0
    min_cell_weights: Sequence[float] | None = None
    hidden_sets: Sequence[Sequence[str]] | None = None
    top: int = 10
    title: str | None = None
    target_description: str | None = None
    observed_label: str | None = None
    row_count_label: str = "Rows"
    raise_errors: bool = False
    ambiguity_limit: float | None = None
    bucket_budget: int | None = None
    max_added_columns: int | None = None
    search: str = "exhaustive"
    beam_width: int = 12
    max_evaluations: int | None = None
    must_include: Sequence[str] = ()
    must_exclude: Sequence[str] = ()
    enforce_bucket_budget: bool = False
    include_base: bool = True
    exact_required: bool = True
    screening_backend: str | None = None
    screening_exact_fallback: bool = True

    def __post_init__(self) -> None:
        kind = _normalize_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "public", _string_tuple(self.public, "public"))
        object.__setattr__(self, "hidden", _string_tuple(self.hidden, "hidden"))
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("target must be a non-empty column name")
        if self.weight is not None and not isinstance(self.weight, str):
            raise TypeError("weight must be a column name or None")
        object.__setattr__(
            self,
            "candidate_refinements",
            _string_tuple(self.candidate_refinements, "candidate_refinements"),
        )
        object.__setattr__(self, "q", QSpec.from_value(self.q))
        if self.q_presets is not None:
            object.__setattr__(
                self,
                "q_presets",
                tuple(QSpec.from_value(value) for value in self.q_presets),
            )
        object.__setattr__(
            self,
            "min_cell_weight",
            _nonnegative_float(
                self.min_cell_weight,
                "min_cell_weight",
            ),
        )
        if self.min_cell_weights is not None:
            object.__setattr__(
                self,
                "min_cell_weights",
                _float_tuple(self.min_cell_weights, "min_cell_weights"),
            )
        if self.hidden_sets is not None:
            object.__setattr__(
                self,
                "hidden_sets",
                tuple(
                    _string_tuple(hidden_set, "hidden_sets")
                    for hidden_set in self.hidden_sets
                ),
            )
        if self.top < 0:
            raise ValueError("top must be non-negative")
        if self.ambiguity_limit is not None:
            object.__setattr__(
                self,
                "ambiguity_limit",
                _nonnegative_float(self.ambiguity_limit, "ambiguity_limit"),
            )
        if self.bucket_budget is not None and self.bucket_budget < 0:
            raise ValueError("bucket_budget must be non-negative")
        if self.max_added_columns is not None and self.max_added_columns < 0:
            raise ValueError("max_added_columns must be non-negative")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_evaluations is not None and self.max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        object.__setattr__(
            self,
            "must_include",
            _string_tuple(self.must_include, "must_include"),
        )
        object.__setattr__(
            self,
            "must_exclude",
            _string_tuple(self.must_exclude, "must_exclude"),
        )
        if self.screening_backend is not None:
            screening_backend = str(self.screening_backend).lower()
            if screening_backend not in {"residopt", "residopt_l2"}:
                raise ValueError(
                    "screening_backend must be None, 'residopt', or 'residopt_l2'"
                )
            object.__setattr__(self, "screening_backend", screening_backend)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditSpec":
        """Build an AuditSpec from a JSON-compatible mapping."""

        return cls(**dict(payload))

    @classmethod
    def from_json(cls, text: str) -> "AuditSpec":
        """Build an AuditSpec from a JSON string."""

        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise TypeError("AuditSpec JSON must decode to an object")
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "public": list(self.public),
            "hidden": list(self.hidden),
            "target": self.target,
            "weight": self.weight,
            "candidate_refinements": list(self.candidate_refinements),
            "q": self.q.as_dict(),
            "q_presets": None
            if self.q_presets is None
            else [preset.as_dict() for preset in self.q_presets],
            "min_cell_weight": self.min_cell_weight,
            "min_cell_weights": None
            if self.min_cell_weights is None
            else list(self.min_cell_weights),
            "hidden_sets": None
            if self.hidden_sets is None
            else [list(hidden_set) for hidden_set in self.hidden_sets],
            "top": self.top,
            "title": self.title,
            "target_description": self.target_description,
            "observed_label": self.observed_label,
            "row_count_label": self.row_count_label,
            "raise_errors": self.raise_errors,
            "ambiguity_limit": self.ambiguity_limit,
            "bucket_budget": self.bucket_budget,
            "max_added_columns": self.max_added_columns,
            "search": self.search,
            "beam_width": self.beam_width,
            "max_evaluations": self.max_evaluations,
            "must_include": list(self.must_include),
            "must_exclude": list(self.must_exclude),
            "enforce_bucket_budget": self.enforce_bucket_budget,
            "include_base": self.include_base,
            "exact_required": self.exact_required,
            "screening_backend": self.screening_backend,
            "screening_exact_fallback": self.screening_exact_fallback,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for as_dict()."""

        return self.as_dict()

    def to_json(self, **kwargs: Any) -> str:
        """Serialize this spec to JSON."""

        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.as_dict(), **options)

    def run(self, data: Any) -> "AuditRun":
        """Execute this spec against tabular data."""

        return run_audit(self, data)


@dataclass(frozen=True)
class AuditRun:
    """Executed audit, preserving both the spec and generated report."""

    spec: AuditSpec
    report: AuditReport

    def to_markdown(self) -> str:
        return self.report.to_markdown()

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "report_type": self.spec.kind,
            "report": self.report.as_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for as_dict()."""

        return self.as_dict()

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the executed spec and structured report output to JSON."""

        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return named list-of-dict tables for the executed audit."""

        from .exports import report_tables

        return report_tables(self)

    def to_dataframes(self) -> dict[str, Any]:
        """Return named pandas DataFrames for the executed audit."""

        from .exports import report_dataframes

        return report_dataframes(self)


def run_audit(spec: AuditSpec | Mapping[str, Any], data: Any) -> AuditRun:
    """Execute a serializable AuditSpec against tabular data."""

    if not isinstance(spec, AuditSpec):
        spec = AuditSpec.from_dict(spec)

    if spec.kind == "public_descent":
        report = public_descent_report(
            data,
            public=spec.public,
            hidden=spec.hidden,
            target=spec.target,
            weight=spec.weight,
            candidate_refinements=spec.candidate_refinements,
            top=spec.top,
            min_cell_weight=spec.min_cell_weight,
            title=spec.title or "Public Descent Report",
            target_description=spec.target_description or "target value",
            observed_label=spec.observed_label or "Observed value",
            row_count_label=spec.row_count_label,
            q=spec.q.to_preset(),
        )
    elif spec.kind == "sensitivity":
        report = sensitivity_report(
            data,
            public=spec.public,
            hidden=spec.hidden,
            target=spec.target,
            weight=spec.weight,
            min_cell_weights=spec.min_cell_weights or (spec.min_cell_weight,),
            hidden_sets=spec.hidden_sets,
            q_presets=None
            if spec.q_presets is None
            else tuple(preset.to_preset() for preset in spec.q_presets),
            title=spec.title or "Public Descent Sensitivity Report",
            raise_errors=spec.raise_errors,
        )
    elif spec.kind == "frontier":
        q_presets = (
            (spec.q.to_preset(),)
            if spec.q_presets is None
            else tuple(preset.to_preset() for preset in spec.q_presets)
        )
        report = public_representation_frontier(
            data,
            base_public=spec.public,
            hidden=spec.hidden,
            target=spec.target,
            weight=spec.weight,
            candidate_refinements=spec.candidate_refinements,
            min_cell_weight=spec.min_cell_weight,
            min_cell_weights=spec.min_cell_weights,
            hidden_sets=spec.hidden_sets,
            q_presets=q_presets,
            ambiguity_limit=spec.ambiguity_limit,
            bucket_budget=spec.bucket_budget,
            max_added_columns=spec.max_added_columns,
            search=spec.search,
            beam_width=spec.beam_width,
            max_evaluations=spec.max_evaluations,
            must_include=spec.must_include,
            must_exclude=spec.must_exclude,
            enforce_bucket_budget=spec.enforce_bucket_budget,
            include_base=spec.include_base,
            screening_backend=spec.screening_backend,
            screening_exact_fallback=spec.screening_exact_fallback,
            title=spec.title or "Public Representation Frontier",
        )
    elif spec.kind == "certificate":
        q_presets = (
            (spec.q.to_preset(),)
            if spec.q_presets is None
            else tuple(preset.to_preset() for preset in spec.q_presets)
        )
        if spec.ambiguity_limit is None:
            raise ValueError("certificate audits require ambiguity_limit")
        report = certify_public_representation(
            data,
            base_public=spec.public,
            hidden=spec.hidden,
            target=spec.target,
            weight=spec.weight,
            candidate_refinements=spec.candidate_refinements,
            min_cell_weight=spec.min_cell_weight,
            min_cell_weights=spec.min_cell_weights,
            hidden_sets=spec.hidden_sets,
            q_presets=q_presets,
            ambiguity_limit=spec.ambiguity_limit,
            bucket_budget=spec.bucket_budget,
            max_added_columns=spec.max_added_columns,
            search=spec.search,
            beam_width=spec.beam_width,
            max_evaluations=spec.max_evaluations,
            must_include=spec.must_include,
            must_exclude=spec.must_exclude,
            enforce_bucket_budget=spec.enforce_bucket_budget,
            include_base=spec.include_base,
            exact_required=spec.exact_required,
            screening_backend=spec.screening_backend,
            screening_exact_fallback=spec.screening_exact_fallback,
            title=spec.title or "Representation Stability Certificate",
        )
    else:
        raise ValueError(f"unsupported audit kind: {spec.kind!r}")

    return AuditRun(spec=spec, report=report)


def _normalize_kind(kind: str) -> str:
    aliases = {
        "public": "public_descent",
        "public_descent": "public_descent",
        "report": "public_descent",
        "sensitivity": "sensitivity",
        "robustness": "sensitivity",
        "frontier": "frontier",
        "public_representation_frontier": "frontier",
        "certificate": "certificate",
        "certification": "certificate",
        "certify": "certificate",
        "representation_certificate": "certificate",
        "representation_stability_certificate": "certificate",
    }
    key = kind.strip().lower().replace("-", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported audit kind: {kind!r}") from exc


def _string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of column names, not a string")
    result = tuple(values)
    if not result and name in {"public", "hidden"}:
        raise ValueError(f"{name} must contain at least one column")
    if not all(isinstance(value, str) and value for value in result):
        raise ValueError(f"{name} must contain non-empty column names")
    return result


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_nonnegative_float(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
