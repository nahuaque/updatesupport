"""Folktables ACS causal-effect reporting example.

This example shows the integration pattern for causal workflows:

1. Estimate a treatment-effect target on ACS-style rows.
2. Feed the estimated effect into ``updatesupport``.
3. Audit whether coarse public reporting categories are stable to hidden
   composition changes.

The default estimator is a transparent stratified difference in weighted outcome
means. In a production workflow, replace that first stage with DoWhy, EconML,
CausalML, DoubleML, or another causal estimator that produces a row-level or
cell-level ``tau_hat`` target.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us
from examples.folktables_acs import TARGET_COLUMN, add_acs_derived_columns


TREATMENT_COLUMN = "__treated__"
EFFECT_COLUMN = "__tau_hat__"
WEIGHT_COLUMN = "weight"


@dataclass(frozen=True)
class EffectBuildResult:
    """Rows and diagnostics produced by the first-stage effect estimator."""

    rows: tuple[dict[str, Any], ...]
    source_rows: int
    retained_strata: int
    dropped_strata: int
    total_effect_weight: float


def build_stratified_effect_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    treatment_column: str = TREATMENT_COLUMN,
    outcome_column: str = TARGET_COLUMN,
    weight_column: str | None = WEIGHT_COLUMN,
    min_arm_weight: float = 1.0,
    effect_column: str = EFFECT_COLUMN,
) -> EffectBuildResult:
    """Estimate one treatment-effect target per retained hidden stratum.

    The estimator is the treated-minus-control difference in weighted outcome
    means inside each hidden stratum. Hidden strata without enough treated and
    control weight are dropped.
    """

    public_tuple = tuple(public_columns)
    hidden_tuple = tuple(hidden_columns)
    missing_public = [column for column in public_tuple if column not in hidden_tuple]
    if missing_public:
        raise ValueError(
            f"public columns must also be hidden columns: {missing_public!r}"
        )
    if min_arm_weight < 0:
        raise ValueError("min_arm_weight must be non-negative")

    stats: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(
        lambda: {0: [0.0, 0.0], 1: [0.0, 0.0]}
    )
    source_rows = 0
    for row in rows:
        source_rows += 1
        key = tuple(_clean_category(row[column]) for column in hidden_tuple)
        treatment = 1 if _as_treatment(row[treatment_column]) else 0
        outcome = _as_float(row[outcome_column], name=outcome_column)
        weight = _row_weight(row, weight_column)
        stats[key][treatment][0] += weight
        stats[key][treatment][1] += weight * outcome

    effect_rows: list[dict[str, Any]] = []
    dropped = 0
    total_effect_weight = 0.0
    for key in sorted(stats, key=str):
        control_weight, control_sum = stats[key][0]
        treated_weight, treated_sum = stats[key][1]
        if control_weight < min_arm_weight or treated_weight < min_arm_weight:
            dropped += 1
            continue
        control_mean = control_sum / control_weight
        treated_mean = treated_sum / treated_weight
        effect_weight = control_weight + treated_weight
        total_effect_weight += effect_weight
        effect_row = {column: key[index] for index, column in enumerate(hidden_tuple)}
        effect_row[effect_column] = treated_mean - control_mean
        effect_row[WEIGHT_COLUMN] = effect_weight
        effect_rows.append(effect_row)

    return EffectBuildResult(
        rows=tuple(effect_rows),
        source_rows=source_rows,
        retained_strata=len(effect_rows),
        dropped_strata=dropped,
        total_effect_weight=total_effect_weight,
    )


def render_causal_report(
    *,
    effect_result: EffectBuildResult,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    candidate_columns: Sequence[str],
    treatment_label: str,
    outcome_label: str,
    estimator_label: str = "stratified difference in weighted outcome means",
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> str:
    """Render a causal-estimation handoff plus an update-support audit."""

    grouped = us.from_dataframe(
        effect_result.rows,
        public=public_columns,
        hidden=hidden_columns,
        target=EFFECT_COLUMN,
        weight=WEIGHT_COLUMN,
        min_cell_weight=min_cell_weight,
        q=q,
        q_radius=q_radius,
    )
    report = us.public_descent_report(
        grouped,
        source_data=effect_result.rows,
        candidate_refinements=candidate_columns,
        top=top,
        min_cell_weight=min_cell_weight,
        title="Representation Stability Audit",
        target_description="estimated treatment effect",
        observed_label="Observed weighted effect estimate",
        row_count=effect_result.retained_strata,
        row_count_label="Retained effect strata",
        weight_column=WEIGHT_COLUMN,
    )

    lines = [
        "# Folktables ACS Causal-Effect Reporting Demo",
        "",
        "## Causal Estimation Step",
        "",
        f"- Treatment: {treatment_label}",
        f"- Outcome: {outcome_label}",
        f"- Effect estimator: {estimator_label}",
        f"- Source rows: {effect_result.source_rows}",
        f"- Retained hidden strata with both treatment arms: {effect_result.retained_strata}",
        f"- Dropped hidden strata without enough arm support: {effect_result.dropped_strata}",
        f"- Observed weighted effect estimate: {report.observed_value:.4f}",
        "",
        "This first stage creates an estimated effect target, `__tau_hat__`, for "
        "each retained hidden stratum. A causal library can replace this stage "
        "as long as it produces a row-level, unit-level, subgroup-level, or "
        "cell-level effect target.",
        "",
        "## Update-Support Question",
        "",
        "Holding the public reporting distribution fixed, how much could the "
        "reported aggregate effect move if the hidden mix inside those public "
        "categories changed?",
        "",
        report.to_markdown(),
    ]
    return "\n".join(lines)


def load_folktables_causal_source_rows(
    *,
    task: str,
    states: Sequence[str],
    year: int,
    horizon: str,
    download: bool,
    sample_size: int | None,
    random_state: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Load Folktables and derive rows for the causal-effect example."""

    try:
        import pandas as pd
        from folktables import ACSDataSource, ACSEmployment, ACSIncome
    except ImportError as exc:
        raise SystemExit(
            "Install example dependencies with: uv sync --extra examples"
        ) from exc

    task_class = {"income": ACSIncome, "employment": ACSEmployment}[task]
    source = ACSDataSource(survey_year=str(year), horizon=horizon, survey="person")
    acs_data = source.get_data(states=list(states), download=download)
    features, label, _group = task_class.df_to_pandas(acs_data)

    frame = features.copy()
    label_values = label.squeeze() if hasattr(label, "squeeze") else label
    frame[TARGET_COLUMN] = pd.Series(label_values).astype(float).to_numpy()
    if sample_size is not None and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=random_state)

    frame = add_acs_derived_columns(frame)
    frame[TREATMENT_COLUMN] = frame["EDU_BAND"].isin(("ba", "grad")).astype(int)

    public_columns, candidate_columns = default_causal_columns(frame.columns)
    hidden_columns = tuple(dict.fromkeys(public_columns + candidate_columns))
    return (
        frame_to_causal_source_rows(
            frame,
            columns=hidden_columns,
            treatment_column=TREATMENT_COLUMN,
            outcome_column=TARGET_COLUMN,
            weight_column="PWGTP" if "PWGTP" in frame.columns else None,
        ),
        public_columns,
        hidden_columns,
        candidate_columns,
    )


def frame_to_causal_source_rows(
    frame,
    *,
    columns: Sequence[str],
    treatment_column: str,
    outcome_column: str,
    weight_column: str | None = None,
) -> list[dict[str, Any]]:
    needed = list(dict.fromkeys(tuple(columns) + (treatment_column, outcome_column)))
    if weight_column is not None:
        needed.append(weight_column)
    records = frame[needed].to_dict("records")
    rows = []
    for record in records:
        row = {column: _clean_category(record[column]) for column in columns}
        row[treatment_column] = int(record[treatment_column])
        row[outcome_column] = float(record[outcome_column])
        if weight_column is not None:
            row[WEIGHT_COLUMN] = float(record[weight_column])
        rows.append(row)
    return rows


def default_causal_columns(
    available_columns: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = set(available_columns)
    public = tuple(column for column in ("AGE_BAND", "SEX") if column in available)
    candidates = (
        "OCC_MAJOR",
        "WKHP_BAND",
        "RAC1P",
        "MAR",
        "COW",
        "RELP",
    )
    return public, tuple(column for column in candidates if column in available)


def synthetic_causal_source_rows() -> tuple[
    list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    public = ("AGE_BAND", "SEX")
    candidates = ("OCC_MAJOR", "WKHP_BAND", "RAC1P")
    hidden = public + candidates
    rows = [
        _arm_row("25_34", "1", "tech", "36_45", "1", 1, 0.82, 80),
        _arm_row("25_34", "1", "tech", "36_45", "1", 0, 0.38, 80),
        _arm_row("25_34", "1", "service", "21_35", "1", 1, 0.56, 120),
        _arm_row("25_34", "1", "service", "21_35", "1", 0, 0.50, 120),
        _arm_row("25_34", "2", "tech", "36_45", "2", 1, 0.72, 70),
        _arm_row("25_34", "2", "tech", "36_45", "2", 0, 0.36, 70),
        _arm_row("45_54", "1", "service", "36_45", "1", 1, 0.60, 100),
        _arm_row("45_54", "1", "service", "36_45", "1", 0, 0.52, 100),
        _arm_row("45_54", "2", "admin", "36_45", "2", 1, 0.62, 90),
        _arm_row("45_54", "2", "admin", "36_45", "2", 0, 0.54, 90),
    ]
    return rows, public, hidden, candidates


def _arm_row(
    age: str,
    sex: str,
    occupation: str,
    hours: str,
    race: str,
    treated: int,
    outcome: float,
    weight: float,
) -> dict[str, Any]:
    return {
        "AGE_BAND": age,
        "SEX": sex,
        "OCC_MAJOR": occupation,
        "WKHP_BAND": hours,
        "RAC1P": race,
        TREATMENT_COLUMN: treated,
        TARGET_COLUMN: outcome,
        WEIGHT_COLUMN: weight,
    }


def _clean_category(value: Any) -> str:
    text = str(value)
    return "NA" if text in {"nan", "None", "<NA>"} else text


def _as_treatment(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "treated", "yes", "y"}:
            return True
        if normalized in {"0", "false", "control", "no", "n"}:
            return False
    return bool(int(value))


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _row_weight(row: Mapping[str, Any], weight_column: str | None) -> float:
    if weight_column is None or weight_column not in row:
        return 1.0
    weight = _as_float(row[weight_column], name=weight_column)
    if weight < 0:
        raise ValueError("weights must be non-negative")
    return weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["income", "employment"], default="income")
    parser.add_argument("--states", nargs="+", default=["CA"])
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--horizon", default="1-Year")
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-arm-weight", type=float, default=10.0)
    parser.add_argument("--min-cell-weight", type=float, default=25.0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--q",
        choices=["saturated", "observed", "bounded_shift"],
        default="bounded_shift",
        help="Admissible-environment preset for the update-support report.",
    )
    parser.add_argument(
        "--q-radius",
        type=float,
        default=0.5,
        help="Relative hidden-cell mass radius for --q bounded_shift.",
    )
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a tiny built-in demo without Folktables or network access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_causal_source_rows()
        )
        min_arm_weight = 1.0
        min_cell_weight = 1.0
    else:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            load_folktables_causal_source_rows(
                task=args.task,
                states=args.states,
                year=args.year,
                horizon=args.horizon,
                download=args.download,
                sample_size=args.sample,
                random_state=args.random_state,
            )
        )
        min_arm_weight = args.min_arm_weight
        min_cell_weight = args.min_cell_weight

    q: Any = args.q
    q_radius = None
    if args.q == "bounded_shift":
        q = us.q_bounded_shift(args.q_radius)

    effect_result = build_stratified_effect_rows(
        source_rows,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        min_arm_weight=min_arm_weight,
    )
    if not effect_result.rows:
        raise SystemExit(
            "No hidden strata have both treatment arms after min-arm filtering."
        )

    print(
        render_causal_report(
            effect_result=effect_result,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            candidate_columns=candidate_columns,
            treatment_label="BA or graduate degree versus less than BA",
            outcome_label=f"ACS{args.task.title()} target label",
            min_cell_weight=min_cell_weight,
            q=q,
            q_radius=q_radius,
            top=args.top,
        )
    )


if __name__ == "__main__":
    main()
