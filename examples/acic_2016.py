"""ACIC 2016 causal benchmark representation-stability example.

The official ACIC 2016 challenge assets are distributed as an R package. For
real data, export one generated simulation to CSV from R, then run this script
against that CSV. The built-in ``--synthetic`` path provides a no-download smoke
demo with the same handoff shape:

1. Build or estimate a row-level treatment-effect target.
2. Restrict to the treated rows by default, matching the SATT focus of ACIC 2016.
3. Feed the effect target into ``updatesupport`` and audit whether coarse public
   categories are stable to hidden-composition changes.
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


TREATMENT_COLUMN = "z"
OUTCOME_COLUMN = "y"
EFFECT_COLUMN = "__tau_hat__"
WEIGHT_COLUMN = "weight"


@dataclass(frozen=True)
class AcicEffectResult:
    """Rows and diagnostics produced by the ACIC effect-building stage."""

    rows: tuple[dict[str, Any], ...]
    source_rows: int
    feature_columns: tuple[str, ...]
    design_columns: tuple[str, ...]
    estimator_name: str
    effect_source: str


EstimatorFactory = Callable[[int], Any]


def load_acic_2016_source_rows(
    path: str | Path,
    *,
    treatment_column: str = TREATMENT_COLUMN,
    outcome_column: str = OUTCOME_COLUMN,
    weight_column: str | None = None,
    covariate_columns: Sequence[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 0,
    public_count: int = 3,
    candidate_count: int = 5,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Load an exported ACIC 2016 simulation CSV and derive reporting cells."""

    import pandas as pd

    frame = pd.read_csv(path)
    if sample_size is not None and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=random_state)
    return prepare_acic_2016_source_rows(
        frame.to_dict("records"),
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        weight_column=weight_column,
        covariate_columns=covariate_columns,
        public_count=public_count,
        candidate_count=candidate_count,
    )


def prepare_acic_2016_source_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    treatment_column: str = TREATMENT_COLUMN,
    outcome_column: str = OUTCOME_COLUMN,
    weight_column: str | None = None,
    covariate_columns: Sequence[str] | None = None,
    public_count: int = 3,
    candidate_count: int = 5,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive categorical public/hidden cells from ACIC-style covariates."""

    import pandas as pd

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("rows must contain at least one record")
    if treatment_column not in frame.columns:
        raise ValueError(f"missing treatment column: {treatment_column!r}")
    if outcome_column not in frame.columns:
        raise ValueError(f"missing outcome column: {outcome_column!r}")

    covariates = tuple(
        covariate_columns
        or _infer_acic_covariates(
            frame.columns,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            weight_column=weight_column,
        )
    )
    derived, public_columns, candidate_columns = add_acic_derived_columns(
        frame,
        covariate_columns=covariates,
        public_count=public_count,
        candidate_count=candidate_count,
    )
    hidden_columns = tuple(dict.fromkeys(public_columns + candidate_columns))
    extra_columns = _available_oracle_columns(derived.columns)
    source_rows = frame_to_acic_source_rows(
        derived,
        columns=hidden_columns,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        weight_column=weight_column,
        extra_columns=extra_columns,
    )
    return source_rows, public_columns, hidden_columns, candidate_columns


def add_acic_derived_columns(
    frame,
    *,
    covariate_columns: Sequence[str],
    public_count: int = 3,
    candidate_count: int = 5,
):
    """Convert selected ACIC covariates into stable categorical bins."""

    out = frame.copy()
    covariates = tuple(covariate_columns)
    if not covariates:
        raise ValueError("at least one covariate column is required")
    selected = covariates[: public_count + candidate_count]
    derived_columns = []
    for column in selected:
        if column not in out.columns:
            raise ValueError(f"missing covariate column: {column!r}")
        derived_name = _derived_column_name(column)
        out[derived_name] = _bin_covariate(out[column])
        derived_columns.append(derived_name)

    public_columns = tuple(derived_columns[:public_count])
    candidate_columns = tuple(derived_columns[public_count:])
    if not public_columns:
        raise ValueError("public_count must select at least one public column")
    return out, public_columns, candidate_columns


def frame_to_acic_source_rows(
    frame,
    *,
    columns: Sequence[str],
    treatment_column: str,
    outcome_column: str,
    weight_column: str | None = None,
    extra_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    needed = list(dict.fromkeys(tuple(columns) + (treatment_column, outcome_column)))
    if weight_column is not None:
        needed.append(weight_column)
    needed.extend(column for column in extra_columns if column not in needed)

    records = frame[needed].to_dict("records")
    rows = []
    for record in records:
        row = {column: _clean_category(record[column]) for column in columns}
        row[treatment_column] = int(_as_treatment(record[treatment_column]))
        row[outcome_column] = _as_float(record[outcome_column], name=outcome_column)
        if weight_column is not None:
            row[WEIGHT_COLUMN] = _as_float(record[weight_column], name=weight_column)
        for column in extra_columns:
            row[column] = _as_float(record[column], name=column)
        rows.append(row)
    return rows


def estimate_acic_effects_with_econml(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    treatment_column: str = TREATMENT_COLUMN,
    outcome_column: str = OUTCOME_COLUMN,
    weight_column: str | None = WEIGHT_COLUMN,
    effect_column: str = EFFECT_COLUMN,
    random_state: int = 0,
    estimator_factory: EstimatorFactory | None = None,
) -> AcicEffectResult:
    """Fit an EconML CATE estimator and attach ``tau_hat = estimator.effect(X)``."""

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

    return AcicEffectResult(
        rows=tuple(effect_rows),
        source_rows=len(records),
        feature_columns=feature_tuple,
        design_columns=design_columns,
        estimator_name=estimator.__class__.__name__,
        effect_source="econml",
    )


def attach_oracle_effects(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    y0_column: str | None = None,
    y1_column: str | None = None,
    weight_column: str | None = WEIGHT_COLUMN,
    effect_column: str = EFFECT_COLUMN,
) -> AcicEffectResult:
    """Attach an oracle effect from ACIC potential-outcome columns when present."""

    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("rows must contain at least one record")
    y0, y1 = _resolve_oracle_columns(records, y0_column=y0_column, y1_column=y1_column)

    effect_rows = []
    for row in records:
        effect_row = dict(row)
        effect_row[effect_column] = _as_float(row[y1], name=y1) - _as_float(
            row[y0],
            name=y0,
        )
        if weight_column is None or weight_column not in effect_row:
            effect_row[WEIGHT_COLUMN] = 1.0
        elif weight_column != WEIGHT_COLUMN:
            effect_row[WEIGHT_COLUMN] = _row_weight(row, weight_column)
        effect_rows.append(effect_row)

    return AcicEffectResult(
        rows=tuple(effect_rows),
        source_rows=len(records),
        feature_columns=tuple(feature_columns),
        design_columns=(),
        estimator_name=f"oracle {y1}-{y0}",
        effect_source="oracle",
    )


def render_acic_report(
    *,
    effect_result: AcicEffectResult,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    candidate_columns: Sequence[str],
    treatment_column: str = TREATMENT_COLUMN,
    inferential_group: str = "treated",
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    top: int = 8,
) -> str:
    """Render the ACIC causal-estimation handoff plus update-support audit."""

    audit_rows = _inferential_rows(
        effect_result.rows,
        treatment_column=treatment_column,
        inferential_group=inferential_group,
    )
    report = us.audit_effects(
        audit_rows,
        public=public_columns,
        hidden=hidden_columns,
        effect=EFFECT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=candidate_columns,
        min_cell_weight=min_cell_weight,
        title="ACIC 2016 Representation Stability Audit",
        effect_description="row-level treatment effect",
        observed_label="Observed weighted effect target",
        row_count=len(audit_rows),
        row_count_label="Rows in update-support audit",
        q=q,
        q_radius=q_radius,
        top=top,
    )

    lines = [
        "# ACIC 2016 Causal Benchmark Demo",
        "",
        "## Causal Estimation Step",
        "",
        "- Benchmark: 2016 Atlantic Causal Inference Conference competition",
        f"- Effect source: {effect_result.effect_source}",
        f"- Effect estimator: {effect_result.estimator_name}",
        "- Effect target: `__tau_hat__`",
        f"- Source rows: {effect_result.source_rows}",
        f"- Inferential group: {inferential_group}",
        f"- Rows audited by updatesupport: {len(audit_rows)}",
        f"- Effect modifier columns: {', '.join(effect_result.feature_columns)}",
        f"- Encoded design columns: {len(effect_result.design_columns)}",
        f"- Observed weighted effect target: {report.observed_value:.4f}",
        "",
        "ACIC 2016 focuses on causal effect estimation under simulated treatment "
        "assignment and outcomes over real covariates. This example treats the "
        "causal estimator's row-level effect as the target to report, then asks "
        "whether a coarse public representation is stable to hidden composition.",
        "",
        "## Update-Support Question",
        "",
        "Holding the public reporting distribution fixed, how much could the "
        "reported effect move if the hidden mix inside those public cells changed?",
        "",
        report.to_markdown(),
    ]
    return "\n".join(lines)


def synthetic_acic_2016_source_rows() -> tuple[
    list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    """Return a small ACIC-shaped dataset for no-download smoke tests."""

    raw_rows = []
    cells = [
        (22.0, "A", 0.15, 0, 0.2, "north", 1.0, 0, 0.20, 0.40, 0.72),
        (24.0, "A", 0.35, 1, 0.3, "north", 0.0, 0, 0.22, 0.30, 0.68),
        (34.0, "B", 0.40, 1, 0.8, "south", 1.0, 1, 0.35, 0.48, 0.58),
        (44.0, "B", 0.65, 0, 0.6, "south", 0.0, 1, 0.38, 0.44, 0.55),
        (55.0, "C", 0.80, 1, 0.4, "west", 1.0, 1, 0.45, 0.54, 0.50),
        (58.0, "C", 0.90, 0, 0.9, "west", 0.0, 1, 0.48, 0.52, 0.48),
    ]
    for cell_index, cell in enumerate(cells):
        x1, x2, x3, x4, x5, x6, x7, x8, y0, y1, treated_share = cell
        for repeat in range(12):
            treated = 1 if repeat / 12 < treated_share else 0
            noise = 0.005 * ((repeat % 3) - 1)
            raw_rows.append(
                {
                    "x1": x1 + repeat * 0.1,
                    "x2": x2,
                    "x3": x3 + 0.01 * repeat,
                    "x4": x4,
                    "x5": x5 + 0.02 * (repeat % 4),
                    "x6": x6,
                    "x7": x7,
                    "x8": x8,
                    TREATMENT_COLUMN: treated,
                    OUTCOME_COLUMN: y0 + (y1 - y0) * treated + noise,
                    "y0": y0 + noise,
                    "y1": y1 + noise,
                    WEIGHT_COLUMN: 1.0 + 0.1 * cell_index,
                }
            )
    return prepare_acic_2016_source_rows(
        raw_rows,
        weight_column=WEIGHT_COLUMN,
        covariate_columns=("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"),
    )


def _infer_acic_covariates(
    columns: Iterable[str],
    *,
    treatment_column: str,
    outcome_column: str,
    weight_column: str | None,
) -> tuple[str, ...]:
    ignored = {
        treatment_column,
        outcome_column,
        weight_column,
        "id",
        "idx",
        "index",
        "y0",
        "y1",
        "Y0",
        "Y1",
        "mu0",
        "mu1",
        "mu_0",
        "mu_1",
        "tau",
        "effect",
        EFFECT_COLUMN,
    }
    return tuple(column for column in columns if column not in ignored)


def _available_oracle_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = set(columns)
    out = []
    for left, right in _oracle_column_pairs():
        if left in names and right in names:
            out.extend([left, right])
    return tuple(dict.fromkeys(out))


def _resolve_oracle_columns(
    rows: Sequence[Mapping[str, Any]],
    *,
    y0_column: str | None,
    y1_column: str | None,
) -> tuple[str, str]:
    if y0_column is not None and y1_column is not None:
        return y0_column, y1_column
    names = set(rows[0])
    for left, right in _oracle_column_pairs():
        if left in names and right in names:
            return left, right
    raise ValueError("oracle effects require y0/y1 or mu0/mu1 columns")


def _oracle_column_pairs() -> tuple[tuple[str, str], ...]:
    return (
        ("y0", "y1"),
        ("Y0", "Y1"),
        ("mu0", "mu1"),
        ("mu_0", "mu_1"),
    )


def _derived_column_name(column: str) -> str:
    clean = "".join(
        character if character.isalnum() else "_" for character in str(column)
    )
    return f"{clean}_BAND"


def _bin_covariate(series):
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        unique = numeric.nunique(dropna=True)
        if unique > 6:
            labels = {0: "low", 1: "mid", 2: "high"}
            codes = pd.qcut(
                numeric,
                q=min(3, unique),
                labels=False,
                duplicates="drop",
            )
            return codes.map(labels).fillna("NA").astype(str)
    text = series.map(_clean_category)
    counts = text.value_counts(dropna=False)
    common = set(counts.head(12).index)
    return text.map(lambda value: value if value in common else "other").fillna("NA")


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


def _inferential_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment_column: str,
    inferential_group: str,
) -> tuple[dict[str, Any], ...]:
    if inferential_group == "all":
        return tuple(dict(row) for row in rows)
    if inferential_group == "treated":
        selected = tuple(
            dict(row) for row in rows if _as_treatment(row[treatment_column])
        )
        if not selected:
            raise ValueError("inferential_group='treated' selected no rows")
        return selected
    raise ValueError("inferential_group must be 'treated' or 'all'")


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
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--effect-source",
        choices=["econml", "oracle"],
        default="oracle",
        help="Use potential outcomes if available, or fit EconML from observed y/z.",
    )
    parser.add_argument("--treatment-column", default=TREATMENT_COLUMN)
    parser.add_argument("--outcome-column", default=OUTCOME_COLUMN)
    parser.add_argument("--weight-column", default=None)
    parser.add_argument("--y0-column")
    parser.add_argument("--y1-column")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--public-count", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument(
        "--inferential-group",
        choices=["treated", "all"],
        default="treated",
        help="Use treated rows for SATT-style ACIC reporting, or all rows.",
    )
    parser.add_argument("--min-cell-weight", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--q",
        choices=["saturated", "observed", "bounded_shift"],
        default="bounded_shift",
    )
    parser.add_argument("--q-radius", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            synthetic_acic_2016_source_rows()
        )
    elif args.input_csv is not None:
        source_rows, public_columns, hidden_columns, candidate_columns = (
            load_acic_2016_source_rows(
                args.input_csv,
                treatment_column=args.treatment_column,
                outcome_column=args.outcome_column,
                weight_column=args.weight_column,
                sample_size=args.sample,
                random_state=args.random_state,
                public_count=args.public_count,
                candidate_count=args.candidate_count,
            )
        )
    else:
        raise SystemExit("pass --synthetic or --input-csv PATH")

    if args.effect_source == "oracle":
        effect_result = attach_oracle_effects(
            source_rows,
            feature_columns=hidden_columns,
            y0_column=args.y0_column,
            y1_column=args.y1_column,
            weight_column=WEIGHT_COLUMN,
        )
    else:
        effect_result = estimate_acic_effects_with_econml(
            source_rows,
            feature_columns=hidden_columns,
            treatment_column=args.treatment_column,
            outcome_column=args.outcome_column,
            weight_column=WEIGHT_COLUMN,
            random_state=args.random_state,
        )

    q: Any = args.q
    q_radius = None
    if args.q == "bounded_shift":
        q = us.q_bounded_shift(args.q_radius)

    print(
        render_acic_report(
            effect_result=effect_result,
            public_columns=public_columns,
            hidden_columns=hidden_columns,
            candidate_columns=candidate_columns,
            treatment_column=args.treatment_column,
            inferential_group=args.inferential_group,
            min_cell_weight=args.min_cell_weight,
            q=q,
            q_radius=q_radius,
            top=args.top,
        )
    )


if __name__ == "__main__":
    main()
