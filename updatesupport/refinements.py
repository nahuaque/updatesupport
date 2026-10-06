"""Portable local bucket splits and finite, target-aware drill suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data import _hashable_category, _iter_records


def _category(value):
    if isinstance(value, (tuple, list)):
        return tuple(_category(x) for x in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("refinement predicates require portable finite categories")


@dataclass(frozen=True)
class ConditionalRefinement:
    """Split one declared public bucket by a descriptor.

    The generated column uses tagged categories, so an outside-bucket label
    cannot collide with a descriptor value. Other public buckets remain intact.
    """

    name: str
    column: str
    predicate: Mapping[str, Any]

    def __post_init__(self):
        if not self.name or not self.column or not self.predicate:
            raise ValueError(
                "name, column, and a nonempty public predicate are required"
            )
        if self.name == self.column or self.name in self.predicate:
            raise ValueError("refinement output must be a new column")
        if any(not isinstance(k, str) or not k for k in self.predicate):
            raise ValueError("predicate columns must be nonempty strings")
        object.__setattr__(
            self,
            "predicate",
            MappingProxyType({k: _category(v) for k, v in self.predicate.items()}),
        )

    def as_dict(self):
        return {
            "name": self.name,
            "column": self.column,
            "predicate": dict(self.predicate),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)

    def transform(self, data):
        result = []
        for raw in _iter_records(data):
            row = dict(raw)
            if self.name in row:
                raise ValueError(f"refinement output already exists: {self.name}")
            # Read every required descriptor, including outside the bucket:
            # missing columns are a schema error rather than an implicit label.
            value = _category(row[self.column])
            matches = all(_category(row[k]) == v for k, v in self.predicate.items())
            row[self.name] = ("split", value) if matches else ("unsplit",)
            result.append(row)
        return tuple(result)


@dataclass(frozen=True)
class ConditionalRefinementSuggestions:
    candidates: tuple[ConditionalRefinement, ...]
    possible_count: int
    max_candidates: int

    @property
    def exact(self):
        return self.possible_count == len(self.candidates)

    def as_dict(self):
        return {
            "candidates": [c.as_dict() for c in self.candidates],
            "possible_count": self.possible_count,
            "max_candidates": self.max_candidates,
            "exact": self.exact,
            "scope": "single descriptor splits of mixed-target base public buckets",
            "selection": "target-aware; no causal interpretation",
        }


def suggest_conditional_refinements(
    data,
    *,
    public: Sequence[str],
    targets: Sequence[str],
    candidate_columns: Sequence[str],
    weight: str | None = None,
    excluded_columns: Sequence[str] = ("issuer_id", "position_id"),
    max_candidates: int = 128,
) -> ConditionalRefinementSuggestions:
    """Enumerate finite local drills; evaluation must use the complete Q.

    Identity and direct target columns are excluded. This discovers separators
    in the supplied measurements, not a general or globally minimal taxonomy.
    """
    if not public or not targets or max_candidates < 0:
        raise ValueError("public/targets must be nonempty and budget nonnegative")
    if len(set(public)) != len(public) or len(set(targets)) != len(targets):
        raise ValueError("public and target columns must be unique")
    excluded = set(excluded_columns) | set(targets) | set(public)
    if weight:
        excluded.add(weight)
    columns = tuple(dict.fromkeys(c for c in candidate_columns if c not in excluded))
    buckets = {}
    existing = set()
    for row in _iter_records(data):
        existing.update(row)
        mass = 1.0 if weight is None else float(row[weight])
        if not isfinite(mass) or mass < 0:
            raise ValueError("weights must be finite and nonnegative")
        if mass == 0:
            continue
        for t in targets:
            if not isfinite(float(row[t])):
                raise ValueError("target values must be finite")
        key = tuple(_category(row[c]) for c in public)
        buckets.setdefault(key, []).append(row)
    candidates, count = [], 0
    for bucket, rows in sorted(buckets.items(), key=lambda x: repr(x[0])):
        if not any(len({float(r[t]) for r in rows}) > 1 for t in targets):
            continue
        for column in columns:
            categories = {_hashable_category(r[column]) for r in rows}
            if len(categories) < 2:
                continue
            # Require a descriptor to separate at least one pair with different
            # measurements. Its full benefit is determined by subsequent solves.
            if not any(
                _hashable_category(a[column]) != _hashable_category(b[column])
                and any(float(a[t]) != float(b[t]) for t in targets)
                for i, a in enumerate(rows)
                for b in rows[i + 1 :]
            ):
                continue
            name = f"conditional_drill_{count}"
            while name in existing:
                name += "_"
            count += 1
            if len(candidates) < max_candidates:
                candidates.append(
                    ConditionalRefinement(name, column, dict(zip(public, bucket)))
                )
    return ConditionalRefinementSuggestions(tuple(candidates), count, max_candidates)
