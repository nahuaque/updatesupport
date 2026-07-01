"""Worked Folktables ACSIncome / ACSEmployment example.

The example treats public categories as the representation an analyst observes,
and hidden categories as finer distinctions that may be reweighted inside each
public cell. The target is the observed task label rate in each hidden cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us


TARGET_COLUMN = "__target__"


GroupedProblem = us.GroupedProblem


def build_problem_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    target_column: str = TARGET_COLUMN,
    weight_column: str | None = None,
    min_cell_weight: float = 1.0,
) -> GroupedProblem:
    """Compatibility wrapper around :func:`updatesupport.from_dataframe`."""

    return us.from_dataframe(
        rows,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        target_column=target_column,
        weight_column=weight_column,
        min_cell_weight=min_cell_weight,
    )


def fiber_diagnostics(
    grouped: GroupedProblem, *, top: int = 10
) -> list[dict[str, Any]]:
    """Compatibility wrapper around :func:`updatesupport.public_fiber_diagnostics`."""

    return [row.as_dict() for row in us.public_fiber_diagnostics(grouped, top=top)]


def refinement_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    public_columns: Sequence[str],
    hidden_columns: Sequence[str],
    candidate_columns: Sequence[str],
    target_column: str = TARGET_COLUMN,
    weight_column: str | None = None,
    min_cell_weight: float = 1.0,
    top: int = 8,
) -> list[dict[str, Any]]:
    """Compatibility wrapper around :func:`updatesupport.recommend_refinements`."""

    return [
        row.as_dict()
        for row in us.recommend_refinements(
            rows,
            public=public_columns,
            hidden=hidden_columns,
            target=target_column,
            weight=weight_column,
            candidate_refinements=candidate_columns,
            min_cell_weight=min_cell_weight,
            top=top,
        )
    ]


def load_folktables_rows(
    *,
    task: str,
    states: Sequence[str],
    year: int,
    horizon: str,
    download: bool,
    sample_size: int | None,
    random_state: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Load Folktables, derive bins, and return rows plus selected columns."""

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
    public_columns, candidate_columns = default_columns(task, frame.columns)
    hidden_columns = tuple(dict.fromkeys(public_columns + candidate_columns))
    rows = frame_to_rows(frame, columns=hidden_columns, target_column=TARGET_COLUMN)
    return rows, public_columns, hidden_columns, candidate_columns


def add_acs_derived_columns(frame):
    """Add coarse bins used by the worked example."""

    import pandas as pd

    out = frame.copy()
    if "AGEP" in out:
        out["AGE_BAND"] = pd.cut(
            pd.to_numeric(out["AGEP"], errors="coerce"),
            bins=[0, 25, 35, 45, 55, 65, 200],
            labels=["under_25", "25_34", "35_44", "45_54", "55_64", "65_plus"],
            include_lowest=True,
        ).astype(str)
    if "SCHL" in out:
        school = pd.to_numeric(out["SCHL"], errors="coerce")
        out["EDU_BAND"] = school.map(_education_band).astype(str)
    if "WKHP" in out:
        out["WKHP_BAND"] = pd.cut(
            pd.to_numeric(out["WKHP"], errors="coerce"),
            bins=[-1, 0, 20, 35, 45, 60, 200],
            labels=["none", "1_20", "21_35", "36_45", "46_60", "60_plus"],
        ).astype(str)
    if "OCCP" in out:
        occ = pd.to_numeric(out["OCCP"], errors="coerce").fillna(-1)
        out["OCC_MAJOR"] = (occ // 1000).astype(int).astype(str)
    return out.fillna("NA")


def _education_band(value: float) -> str:
    if value != value:
        return "NA"
    if value < 16:
        return "lt_hs"
    if value < 20:
        return "hs_or_some_college"
    if value == 21:
        return "ba"
    if value >= 22:
        return "grad"
    return "associate_or_other"


def default_columns(
    task: str, available_columns: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    available = set(available_columns)
    public = tuple(
        column for column in ("AGE_BAND", "EDU_BAND", "SEX") if column in available
    )
    if task == "income":
        candidates = (
            "OCC_MAJOR",
            "COW",
            "WKHP_BAND",
            "RAC1P",
            "MAR",
            "POBP",
            "RELP",
        )
    else:
        candidates = (
            "RAC1P",
            "DIS",
            "CIT",
            "MIG",
            "MAR",
            "RELP",
            "ESP",
            "NATIVITY",
        )
    return public, tuple(column for column in candidates if column in available)


def frame_to_rows(
    frame, *, columns: Sequence[str], target_column: str
) -> list[dict[str, Any]]:
    needed = list(dict.fromkeys(tuple(columns) + (target_column,)))
    records = frame[needed].to_dict("records")
    rows = []
    for record in records:
        row = {column: _clean_category(record[column]) for column in columns}
        row[target_column] = float(record[target_column])
        rows.append(row)
    return rows


def _clean_category(value: Any) -> str:
    text = str(value)
    return "NA" if text in {"nan", "None", "<NA>"} else text


def synthetic_rows() -> tuple[
    list[dict[str, Any]], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    rows = [
        {
            "AGE_BAND": "25_34",
            "EDU_BAND": "ba",
            "SEX": "1",
            "OCC_MAJOR": "1",
            "RAC1P": "1",
            TARGET_COLUMN: 1.0,
            "weight": 80,
        },
        {
            "AGE_BAND": "25_34",
            "EDU_BAND": "ba",
            "SEX": "1",
            "OCC_MAJOR": "2",
            "RAC1P": "1",
            TARGET_COLUMN: 0.2,
            "weight": 70,
        },
        {
            "AGE_BAND": "25_34",
            "EDU_BAND": "ba",
            "SEX": "2",
            "OCC_MAJOR": "1",
            "RAC1P": "2",
            TARGET_COLUMN: 0.7,
            "weight": 60,
        },
        {
            "AGE_BAND": "25_34",
            "EDU_BAND": "ba",
            "SEX": "2",
            "OCC_MAJOR": "2",
            "RAC1P": "2",
            TARGET_COLUMN: 0.1,
            "weight": 40,
        },
        {
            "AGE_BAND": "45_54",
            "EDU_BAND": "hs_or_some_college",
            "SEX": "1",
            "OCC_MAJOR": "3",
            "RAC1P": "1",
            TARGET_COLUMN: 0.5,
            "weight": 100,
        },
        {
            "AGE_BAND": "45_54",
            "EDU_BAND": "hs_or_some_college",
            "SEX": "1",
            "OCC_MAJOR": "4",
            "RAC1P": "1",
            TARGET_COLUMN: 0.3,
            "weight": 50,
        },
    ]
    public = ("AGE_BAND", "EDU_BAND", "SEX")
    candidates = ("OCC_MAJOR", "RAC1P")
    hidden = public + candidates
    return rows, public, hidden, candidates


def format_key(columns: Sequence[str], key: tuple[Any, ...]) -> str:
    return ", ".join(
        f"{column}={value}" for column, value in zip(columns, key, strict=True)
    )


def task_target_description(task: str) -> str:
    if task == "income":
        return "Pr(income exceeds the ACSIncome threshold)"
    if task == "employment":
        return "Pr(employed under the ACSEmployment task definition)"
    return "Pr(target label is 1)"


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_report(
    *,
    task: str,
    grouped: GroupedProblem,
    rows: Sequence[Mapping[str, Any]],
    candidate_columns: Sequence[str],
    top: int,
    min_cell_weight: float,
) -> str:
    report = us.public_descent_report(
        grouped,
        source_data=rows,
        candidate_refinements=candidate_columns,
        top=top,
        min_cell_weight=min_cell_weight,
        title=f"Folktables ACS{task.title()} Update-Support Demo",
        target_description=task_target_description(task),
        observed_label="Observed target rate",
        row_count=len(rows),
        row_count_label="Rows after sampling",
        weight_column="weight" if rows and "weight" in rows[0] else None,
    )
    return report.to_markdown()


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
        rows, public_columns, hidden_columns, candidate_columns = synthetic_rows()
        min_cell_weight = 1.0
    else:
        rows, public_columns, hidden_columns, candidate_columns = load_folktables_rows(
            task=args.task,
            states=args.states,
            year=args.year,
            horizon=args.horizon,
            download=args.download,
            sample_size=args.sample,
            random_state=args.random_state,
        )
        min_cell_weight = args.min_cell_weight

    grouped = build_problem_from_rows(
        rows,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        target_column=TARGET_COLUMN,
        weight_column="weight" if rows and "weight" in rows[0] else None,
        min_cell_weight=min_cell_weight,
    )
    print(
        render_report(
            task=args.task,
            grouped=grouped,
            rows=rows,
            candidate_columns=candidate_columns,
            top=args.top,
            min_cell_weight=min_cell_weight,
        )
    )


if __name__ == "__main__":
    main()
