"""Explicit partitions and selected schedules for dated financial flows."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from math import isclose, isfinite

import updatesupport as us

from .measurements import MeasurementDefinition
from .relationships import reconciliation_constraint


@dataclass(frozen=True)
class DisclosureWindow:
    name: str
    start: str
    end: str

    def __post_init__(self):
        if not self.name or date.fromisoformat(self.start) > date.fromisoformat(
            self.end
        ):
            raise ValueError("a named, ordered date window is required")

    def as_dict(self):
        return vars(self).copy()

    def duration(self, basis):
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if basis == "days":
            return (end - start).days + 1
        if basis == "months":
            if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
                raise ValueError("month schedules require whole calendar months")
            return (end.year - start.year) * 12 + end.month - start.month + 1
        raise ValueError("schedule basis must be days or months")


@dataclass(frozen=True)
class TimeWindowAllocation:
    name: str
    total: str
    source_window: DisclosureWindow
    windows: tuple[DisclosureWindow, ...]
    variables: tuple
    constraints: tuple
    cohort: str
    cohort_as_of: str
    role: str

    def schedule_constraints(self, weights, *, policy):
        """Select a timing policy explicitly; unconstrained partitions remain free."""
        if not policy or set(weights) != {w.name for w in self.windows}:
            raise ValueError("a named policy and weights for every window are required")
        if any(not isfinite(v) or v < 0 for v in weights.values()) or not isclose(
            sum(weights.values()), 1, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("nonnegative schedule weights must sum to one")
        return tuple(
            us.NamedLinearConstraint(
                f"{self.name}:schedule:{w.name}",
                {w.name: 1, self.total: -weights[w.name]},
                lower=0,
                upper=0,
                kind="timing_policy",
                description=policy,
                metadata={
                    "evidence_role": "analyst_policy",
                    "policy_name": policy,
                    "cohort": self.cohort,
                    "cohort_as_of": self.cohort_as_of,
                    "window": w.as_dict(),
                },
            )
            for w in self.windows
        )

    def even_schedule_constraints(self, *, basis, policy):
        """Explicitly choose an even monthly or daily conversion assumption."""
        durations = {w.name: w.duration(basis) for w in self.windows}
        total = sum(durations.values())
        return self.schedule_constraints(
            {k: v / total for k, v in durations.items()}, policy=policy
        )

    def subtotal_constraint(self, name, *, total, windows):
        selected = tuple(windows)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= {w.name for w in self.windows}
        ):
            raise ValueError("subtotal needs unique, declared windows")
        return reconciliation_constraint(
            name,
            total=total,
            components=selected,
            metadata={
                "evidence_role": "accounting_relationship",
                "windows": list(selected),
                "cohort": self.cohort,
            },
        )

    def as_dict(self):
        return {
            "name": self.name,
            "total": self.total,
            "source_window": self.source_window.as_dict(),
            "windows": [w.as_dict() for w in self.windows],
            "cohort": self.cohort,
            "cohort_as_of": self.cohort_as_of,
            "role": self.role,
            "variables": [v.as_dict() for v in self.variables],
            "constraints": [c.as_dict() for c in self.constraints],
        }


def time_window_allocation(
    name,
    *,
    total,
    source_window,
    windows,
    unit,
    cohort,
    cohort_as_of,
    measurement: MeasurementDefinition,
    role="management_expectation",
):
    """Partition a flow/schedule, without assigning any within-window timing.

    Stocks and lifetime commitments need a separately declared payment or
    recognition schedule. This helper does not turn them into annual cash flows.
    Bucket names are variable names, allowing cumulative subtotals without
    counting an overlapping period twice.
    """
    if measurement.kind not in {"flow", "schedule"} or measurement.unit != unit:
        raise ValueError(
            "window allocations require a flow/schedule in the declared unit"
        )
    if (
        not name
        or not cohort
        or role not in {"reported_fact", "management_expectation", "analyst_policy"}
    ):
        raise ValueError("name, cohort and evidence role are required")
    date.fromisoformat(cohort_as_of)
    windows = tuple(sorted(windows, key=lambda w: w.start))
    if (
        not windows
        or len({w.name for w in windows}) != len(windows)
        or total in {w.name for w in windows}
    ):
        raise ValueError("partition bucket names must be unique and differ from total")
    next_start = date.fromisoformat(source_window.start)
    for window in windows:
        if date.fromisoformat(window.start) != next_start:
            raise ValueError("windows must form a disjoint, contiguous partition")
        next_start = date.fromisoformat(window.end) + timedelta(days=1)
    if next_start != date.fromisoformat(source_window.end) + timedelta(days=1):
        raise ValueError("windows must cover the full source window")
    variables = tuple(
        us.NamedLinearVariable(w.name, lower=0, unit=unit, label=w.name)
        for w in windows
    )
    constraint = reconciliation_constraint(
        name + ":partition",
        total=total,
        components=[w.name for w in windows],
        metadata={
            "evidence_role": "accounting_relationship",
            "schedule_role": role,
            "measurement": measurement.as_dict(),
            "cohort": cohort,
            "cohort_as_of": cohort_as_of,
            "source_window": source_window.as_dict(),
            "windows": [w.as_dict() for w in windows],
        },
    )
    return TimeWindowAllocation(
        name,
        total,
        source_window,
        windows,
        variables,
        (constraint,),
        cohort,
        cohort_as_of,
        role,
    )
