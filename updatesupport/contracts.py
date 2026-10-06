"""Uncalibrated frozen reporting contracts and support diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose, isfinite
from typing import Mapping

from .claim import ClaimSpec
from .data import _iter_records, from_dataframe
from .policy import FrozenSupportCell, _as_cell, _support_comparison


def _encode(value):
    if isinstance(value, Mapping):
        return {"mapping": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(x) for x in value]}
    if isinstance(value, list):
        return [_encode(x) for x in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise TypeError("report contracts require portable finite values")


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            pairs = [(_decode(k), _decode(v)) for k, v in value["mapping"]]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("duplicate portable mapping keys")
            return result
        if set(value) == {"tuple"}:
            return tuple(_decode(x) for x in value["tuple"])
        raise ValueError("invalid portable encoding")
    if isinstance(value, list):
        return [_decode(x) for x in value]
    return value


@dataclass(frozen=True)
class FrozenReportContract:
    """One fixed claim/schema/Q contract without historical calibration.

    Support contraction is permitted when every reference public fiber remains.
    New hidden support and lost public fibers make the audit inconclusive.
    Q is analyst-declared, not estimated from these reference observations.
    """

    claim: ClaimSpec
    reference_support: tuple[FrozenSupportCell, ...]

    def __post_init__(self):
        if not isinstance(self.claim.target, str):
            raise TypeError("portable contracts require named column targets")
        _encode(self.claim.as_dict())
        support = tuple(self.reference_support)
        states = [c.state for c in support]
        indexes = [self.claim.hidden.index(c) for c in self.claim.public]
        if not support or len(set(states)) != len(states):
            raise ValueError("reference support must be nonempty and unique")
        for cell in support:
            if (
                len(cell.state) != len(self.claim.hidden)
                or len(cell.public_value) != len(indexes)
                or not isfinite(cell.mass)
                or cell.mass <= 0
                or tuple(cell.state[i] for i in indexes) != cell.public_value
            ):
                raise ValueError(
                    "invalid reference support dimensions, mass, or public map"
                )
            _encode(cell.as_dict())
        if not isclose(sum(c.mass for c in support), 1.0, abs_tol=1e-9):
            raise ValueError("reference masses must sum to one")
        object.__setattr__(self, "reference_support", support)

    @classmethod
    def freeze(cls, claim: ClaimSpec, data):
        audit = claim.audit(data)
        grouped = audit.primary.grouped
        return cls(
            claim,
            tuple(
                FrozenSupportCell(
                    _as_cell(s),
                    _as_cell(grouped.problem.public_map[s]),
                    float(grouped.cell_weights[s]),
                )
                for s in sorted(grouped.problem.states, key=str)
            ),
        )

    def audit(self, data):
        rows = tuple(_iter_records(data))
        if not rows:
            return {
                "status": "inconclusive",
                "reason": "no_snapshot",
                "support": None,
                "audit": None,
            }
        # Diagnose support before constructing a Q with state-dependent matrices.
        current = from_dataframe(
            rows,
            public=self.claim.public,
            hidden=self.claim.hidden,
            target=self.claim.target,
            weight=self.claim.weight,
            min_cell_weight=self.claim.min_cell_weight,
            q="saturated",
        )
        support = _support_comparison(self.reference_support, current)
        if not support["support_compatible"]:
            return {
                "status": "inconclusive",
                "reason": "unsupported_support",
                "support": support,
                "audit": None,
            }
        audit = self.claim.audit(rows)
        status = (
            "evaluated_without_criteria"
            if self.claim.ambiguity_limit is None and self.claim.decision is None
            else "inconclusive"
            if audit.inconclusive
            else "pass"
            if audit.passed
            else "review"
        )
        return {
            "status": status,
            "reason": "support_contraction"
            if support["missing_reference_hidden_cells"]
            else "compatible_support",
            "support": support,
            "audit": audit.as_dict(),
        }

    def as_dict(self):
        return {
            "format": "updatesupport.frozen_report_contract",
            "schema": 1,
            "claim": _encode(self.claim.as_dict()),
            "reference_support": _encode([c.as_dict() for c in self.reference_support]),
        }

    @classmethod
    def from_dict(cls, payload):
        if (
            set(payload) != {"format", "schema", "claim", "reference_support"}
            or payload["schema"] != 1
            or payload["format"] != "updatesupport.frozen_report_contract"
        ):
            raise ValueError("invalid frozen report contract schema")
        return cls(
            ClaimSpec.from_dict(_decode(payload["claim"])),
            tuple(
                FrozenSupportCell(**c) for c in _decode(payload["reference_support"])
            ),
        )

    def to_json(self, **kwargs):
        return json.dumps(self.as_dict(), allow_nan=False, **kwargs)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))
