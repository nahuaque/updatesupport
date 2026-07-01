"""Folktables ACS causal-effect reporting example.

This example shows the integration pattern for causal workflows:

1. Estimate a treatment-effect target on ACS-style rows.
2. Feed the estimated effect into ``updatesupport``.
3. Audit whether coarse public reporting categories are stable to hidden
   composition changes.

The first stage fits an EconML estimator and computes
``tau_hat = estimator.effect(X)``. The second stage is the ``updatesupport``
audit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    feature_columns: tuple[str, ...]
    design_columns: tuple[str, ...]
    estimator_name: str


EstimatorFactory = Callable[[int], Any]


def estimate_effects_with_econml(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    treatment_column: str = TREATMENT_COLUMN,
    outcome_column: str = TARGET_COLUMN,
    weight_column: str | None = WEIGHT_COLUMN,
    effect_column: str = EFFECT_COLUMN,
    random_state: int = 0,
    estimator_factory: EstimatorFactory | None = None,
) -> EffectBuildResult:
    """Fit an EconML estimator and attach ``tau_hat = estimator.effect(X)``.

    ``feature_columns`` are encoded as categorical effect modifiers. The
    returned rows preserve the original columns and add ``effect_column``.
    """

    import numpy as np

    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("rows must contain at least one record")
    feature_tuple = tuple(feature_columns)
    if not feature_tuple:
        raise ValueError("feature_columns must contain at least one column")

    x_matrix, design_columns = _categorical_design_matrix(records, feature_tuple)
    y = np.array(
        [_as_float(row[outcome_column], name=outcome_column) for row in records],
        dtype=float,
    )
    treatment = np.array(
        [1 if _as_treatment(row[treatment_column]) else 0 for row in records],
        dtype=int,
    )
    sample_weight = np.array(
        [_row_weight(row, weight_column) for row in records],
        dtype=float,
    )

    estimator = (
        _default_econml_estimator(random_state)
        if estimator_factory is None
        else estimator_factory(random_state)
    )
    estimator.fit(
        y,
        treatment,
        X=x_matrix,
        sample_weight=sample_weight,
        inference=None,
    )
    tau_hat = np.asarray(estimator.effect(x_matrix), dtype=float).reshape(-1)
    if len(tau_hat) != len(records):
        raise ValueError("estimator.effect(X) must return one effect per row")

    effect_rows = []
    for row, effect in zip(records, tau_hat, strict=True):
        effect_row = dict(row)
        effect_row[effect_column] = float(effect)
        if weight_column is None or weight_column not in effect_row:
            effect_row[WEIGHT_COLUMN] = 1.0
        elif weight_column != WEIGHT_COLUMN:
            effect_row[WEIGHT_COLUMN] = _row_weight(row, weight_column)
        effect_rows.append(effect_row)

    return EffectBuildResult(
        rows=tuple(effect_rows),
        source_rows=len(records),
        feature_columns=feature_tuple,
        design_columns=design_columns,
        estimator_name=estimator.__class__.__name__,
    )


def render_causal_report(
    *,
    effect_result: EffectBuildResult,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    candidate_columns: Sequence[str],
    treatment_label: str,
    outcome_label: str,
    estimator_label: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> str:
    """Render a causal-estimation handoff plus an update-support audit."""

    label = estimator_label or f"EconML {effect_result.estimator_name}"
    report = us.audit_effects(
        effect_result.rows,
        public=public_columns,
        hidden=hidden_columns,
        effect=EFFECT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=candidate_columns,
        min_cell_weight=min_cell_weight,
        title="Representation Stability Audit",
        effect_description="estimated treatment effect",
        observed_label="Observed weighted effect estimate",
        row_count=effect_result.source_rows,
        row_count_label="Rows with EconML effect predictions",
        q=q,
        q_radius=q_radius,
        top=top,
    )

    lines = [
        "# Folktables ACS Causal-Effect Reporting Demo",
        "",
        "## Causal Estimation Step",
        "",
        f"- Treatment: {treatment_label}",
        f"- Outcome: {outcome_label}",
        f"- Effect estimator: {label}",
        "- Effect target: `__tau_hat__ = estimator.effect(X)`",
        f"- Source rows: {effect_result.source_rows}",
        f"- Effect modifier columns: {', '.join(effect_result.feature_columns)}",
        f"- Encoded design columns: {len(effect_result.design_columns)}",
        f"- Observed weighted effect estimate: {report.observed_value:.4f}",
        "",
        "The first stage fits an EconML CATE estimator and writes `__tau_hat__` "
        "for each row. The update-support stage then aggregates those effect "
        "predictions by hidden cell and audits the public reporting categories.",
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


def _default_econml_estimator(random_state: int):
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        raise SystemExit(
            "Install causal example dependencies with: uv sync --extra causal"
        ) from exc

    return CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        model_t=RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=5,
            random_state=random_state,
        ),
        discrete_treatment=True,
        n_estimators=80,
        min_samples_leaf=5,
        cv=2,
        random_state=random_state,
    )


def _categorical_design_matrix(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
):
    import numpy as np

    categories = {
        column: tuple(sorted({_clean_category(row[column]) for row in rows}, key=str))
        for column in columns
    }
    design_columns = tuple(
        f"{column}={value}" for column in columns for value in categories[column]
    )
    matrix = np.zeros((len(rows), len(design_columns)), dtype=float)
    column_index = {name: index for index, name in enumerate(design_columns)}
    for row_index, row in enumerate(rows):
        for column in columns:
            value = _clean_category(row[column])
            matrix[row_index, column_index[f"{column}={value}"]] = 1.0
    return matrix, design_columns


def synthetic_causal_source_rows() -> tuple[
    list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    public = ("AGE_BAND", "SEX")
    candidates = ("OCC_MAJOR", "WKHP_BAND", "RAC1P")
    hidden = public + candidates
    cells = [
        ("25_34", "1", "tech", "36_45", "1", 0.36, 0.34, 1.1),
        ("25_34", "1", "service", "21_35", "1", 0.42, 0.06, 1.4),
        ("25_34", "2", "tech", "36_45", "2", 0.34, 0.26, 1.0),
        ("25_34", "2", "service", "21_35", "2", 0.46, 0.04, 1.2),
        ("45_54", "1", "service", "36_45", "1", 0.52, 0.08, 1.3),
        ("45_54", "2", "admin", "36_45", "2", 0.50, 0.07, 1.1),
    ]
    rows = []
    for cell in cells:
        age, sex, occupation, hours, race, baseline, tau, weight = cell
        for repeat in range(8):
            noise = 0.01 * ((repeat % 3) - 1)
            for treated in (0, 1):
                rows.append(
                    _arm_row(
                        age,
                        sex,
                        occupation,
                        hours,
                        race,
                        treated,
                        baseline + tau * treated + noise,
                        weight,
                    )
                )
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
        help="Run a built-in EconML demo without Folktables or network access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_causal_source_rows()
        )
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
        min_cell_weight = args.min_cell_weight

    q: Any = args.q
    q_radius = None
    if args.q == "bounded_shift":
        q = us.q_bounded_shift(args.q_radius)

    effect_result = estimate_effects_with_econml(
        source_rows,
        feature_columns=hidden_columns,
        random_state=args.random_state,
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
