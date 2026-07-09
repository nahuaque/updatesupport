"""Conformal prediction reporting-stability example.

The example audits whether aggregate conformal-prediction quantities are stable
to hidden subgroup recomposition. It is synthetic and dependency-light: a tiny
linear model is fit with NumPy, split-conformal intervals are computed from a
calibration split, and ``updatesupport`` audits the resulting interval-width,
miscoverage, and threshold-crossing targets.

Run from the repository root with:

    uv run python examples/conformal_reporting_stability.py

Optionally write the Markdown report:

    uv run python examples/conformal_reporting_stability.py \
        --output data/conformal_reporting_stability.md

Optionally write an interval plot, if matplotlib is installed:

    uv run --extra examples python examples/conformal_reporting_stability.py \
        --plot-output data/conformal_reporting_stability.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us


PUBLIC_COLUMNS = ("customer_segment", "product_family", "platform")
HIDDEN_REFINEMENTS = (
    "acquisition_channel",
    "tenure_band",
    "geo_market",
    "support_load",
)
HIDDEN_COLUMNS = PUBLIC_COLUMNS + HIDDEN_REFINEMENTS
CANDIDATE_REFINEMENTS = HIDDEN_REFINEMENTS
WEIGHT_COLUMN = "accounts"
OBSERVED_COLUMN = "observed_risk"
RISK_THRESHOLD = 0.18
ALPHA = 0.10

_SEGMENTS = ("enterprise", "midmarket", "smb")
_PRODUCTS = ("core", "analytics")
_PLATFORMS = ("web", "mobile")
_CHANNELS = ("sales_led", "partner", "self_serve")
_TENURES = ("new", "established")
_GEOS = ("north_america", "europe", "latam")
_SUPPORT = ("low", "high")


def synthetic_conformal_audit_rows() -> list[dict[str, Any]]:
    """Return retained reporting cells for the conformal stability audit."""

    specs = [
        (
            "enterprise",
            "core",
            "web",
            "sales_led",
            "established",
            "north_america",
            "low",
            820,
        ),
        ("enterprise", "core", "web", "partner", "new", "europe", "high", 460),
        (
            "enterprise",
            "analytics",
            "mobile",
            "sales_led",
            "established",
            "north_america",
            "low",
            540,
        ),
        (
            "enterprise",
            "analytics",
            "mobile",
            "self_serve",
            "new",
            "latam",
            "high",
            180,
        ),
        ("midmarket", "core", "web", "partner", "established", "europe", "low", 760),
        ("midmarket", "core", "web", "self_serve", "new", "latam", "high", 320),
        ("midmarket", "core", "mobile", "partner", "new", "north_america", "high", 410),
        (
            "midmarket",
            "core",
            "mobile",
            "sales_led",
            "established",
            "europe",
            "low",
            520,
        ),
        ("midmarket", "analytics", "web", "self_serve", "new", "latam", "high", 300),
        (
            "midmarket",
            "analytics",
            "web",
            "partner",
            "established",
            "north_america",
            "low",
            500,
        ),
        ("smb", "core", "web", "self_serve", "new", "latam", "high", 620),
        ("smb", "core", "web", "partner", "established", "europe", "low", 480),
        ("smb", "core", "mobile", "self_serve", "new", "north_america", "high", 560),
        ("smb", "analytics", "mobile", "self_serve", "new", "latam", "high", 430),
        ("smb", "analytics", "mobile", "partner", "established", "europe", "low", 390),
        (
            "smb",
            "analytics",
            "web",
            "sales_led",
            "established",
            "north_america",
            "low",
            350,
        ),
    ]
    columns = HIDDEN_COLUMNS + (WEIGHT_COLUMN,)
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        row = dict(zip(columns, spec, strict=True))
        row[OBSERVED_COLUMN] = _observed_risk(row, split="audit", replicate=index)
        rows.append(row)
    return rows


def fit_split_conformal_model(
    *,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, float]:
    """Fit the tiny upstream model and return coefficients plus q-hat."""

    train_rows = _synthetic_model_rows(split="train", replicates=2)
    calibration_rows = _synthetic_model_rows(split="calibration", replicates=1)
    coefficients = _fit_linear_model(train_rows)
    scores = []
    for row in calibration_rows:
        prediction = _predict_risk(row, coefficients)
        residual = abs(float(row[OBSERVED_COLUMN]) - prediction)
        scores.append(residual / _local_uncertainty_scale(row))
    return coefficients, _conformal_quantile(scores, alpha=alpha)


def build_conformal_result(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    alpha: float = ALPHA,
) -> us.ConformalAdapterResult:
    """Attach split-conformal predictions and intervals to retained cells."""

    rows = synthetic_conformal_audit_rows() if rows is None else rows
    coefficients, q_hat = fit_split_conformal_model(alpha=alpha)

    predictions: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    observed: list[float] = []
    for row in rows:
        prediction = _predict_risk(row, coefficients)
        half_width = q_hat * _local_uncertainty_scale(row)
        predictions.append(prediction)
        lower.append(max(0.0, prediction - half_width))
        upper.append(min(1.0, prediction + half_width))
        observed.append(float(row[OBSERVED_COLUMN]))

    result = us.adapt_conformal_regression(
        rows,
        prediction=predictions,
        lower=lower,
        upper=upper,
        y_true=observed,
        threshold=RISK_THRESHOLD,
        source="split_conformal_regression",
    )
    return result


def build_stability_report(
    result: us.ConformalAdapterResult | None = None,
    *,
    include_attribution: bool = False,
) -> us.ConformalReportingStabilityReport:
    """Build the conformal reporting-stability report."""

    result = build_conformal_result() if result is None else result
    return result.reporting_stability(
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        weight=WEIGHT_COLUMN,
        candidate_refinements=CANDIDATE_REFINEMENTS,
        ambiguity_limits={
            "y_pred": 0.035,
            "y_lower": 0.040,
            "y_upper": 0.040,
            "interval_width": 0.030,
            "covered": 0.20,
            "miscovered": 0.20,
            "crosses_threshold": 0.25,
        },
        search="beam",
        beam_width=8,
        max_added_columns=2,
        max_evaluations=64,
        exact_required=False,
        include_attribution=include_attribution,
        title="Conformal Prediction Reporting Stability Example",
    )


def render_report(
    *,
    include_attribution: bool = False,
) -> str:
    """Render the conformal reporting-stability example as Markdown."""

    result = build_conformal_result()
    report = build_stability_report(
        result,
        include_attribution=include_attribution,
    )
    rows = result.rows
    observed_width = _weighted_mean(rows, "interval_width", WEIGHT_COLUMN)
    observed_miscoverage = _weighted_mean(rows, "miscovered", WEIGHT_COLUMN)
    observed_crossing = _weighted_mean(rows, "crosses_threshold", WEIGHT_COLUMN)
    target_rows = {row["target"]: row for row in report.to_tables()["targets"]}
    width_row = target_rows["interval_width"]
    crossing_row = target_rows["crosses_threshold"]
    miscoverage_row = target_rows["miscovered"]

    lines = [
        "# Conformal Prediction Reporting Stability Example",
        "",
        "This synthetic demo sits downstream of a conformal prediction workflow. "
        "A small linear model is fit on synthetic training data; a held-out "
        "calibration split produces normalized split-conformal intervals; "
        "`updatesupport` then audits whether aggregate conformal reporting "
        "claims survive hidden subgroup recomposition.",
        "",
        "The public report groups retained accounts by customer segment, product "
        "family, and platform. The retained but not publicly reported cells also "
        "track acquisition channel, tenure, geography, and support load.",
        "",
        "## What The Conformal Layer Supplies",
        "",
        f"- Nominal conformal miscoverage level: {ALPHA:.0%}",
        f"- High-risk threshold used for interval crossing: {RISK_THRESHOLD:.1%}",
        f"- Observed weighted mean interval width: {_percent(observed_width)}",
        f"- Observed weighted miscoverage rate: {_percent(observed_miscoverage)}",
        f"- Observed weighted threshold-crossing rate: {_percent(observed_crossing)}",
        "",
        "Those are ordinary conformal-derived row targets. The next question is "
        "whether the aggregate versions of those targets are stable if the "
        "public customer mix is fixed but retained hidden composition changes "
        "inside the public buckets.",
        "",
        "## Hidden-Composition Readout",
        "",
        "- Mean interval width can move from "
        f"{_percent(width_row['lower'])} to {_percent(width_row['upper'])} "
        "under the declared stress test.",
        "- Miscoverage can move from "
        f"{_percent(miscoverage_row['lower'])} to "
        f"{_percent(miscoverage_row['upper'])}.",
        "- Threshold-crossing burden can move from "
        f"{_percent(crossing_row['lower'])} to "
        f"{_percent(crossing_row['upper'])}.",
        "",
        "Plain English: conformal prediction gives row-level uncertainty "
        "intervals, but a coarse public report can still obscure where the "
        "uncertainty burden sits. If hidden acquisition, tenure, geography, or "
        "support-load mix shifts inside public buckets, the reported aggregate "
        "manual-review or risk-threshold burden can change.",
        "",
        "This is not a conformal prediction interval and not a confidence "
        "interval for model error. It is a representation-stability audit of "
        "conformal-derived reporting targets.",
        "",
        "## Multi-Target Stability Audit",
        "",
    ]
    lines.extend(_without_title(report.to_markdown()))
    return "\n".join(lines)


def plot_target_intervals(
    report: us.ConformalReportingStabilityReport,
    output: Path,
) -> Path:
    """Write a compact target-interval plot if matplotlib is installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting this example requires matplotlib. Install "
            "`updatesupport[examples]` or matplotlib directly."
        ) from exc

    target_rows = list(report.to_tables()["targets"])
    labels = [str(row["label"]) for row in target_rows]
    lower = np.array([float(row["lower"]) for row in target_rows])
    upper = np.array([float(row["upper"]) for row in target_rows])
    observed = np.array([float(row["observed_value"]) for row in target_rows])
    y = np.arange(len(target_rows))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hlines(y, lower, upper, color="#0f766e", linewidth=4, alpha=0.75)
    ax.scatter(observed, y, color="#111827", zorder=3, label="observed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate target value")
    ax.set_title("Conformal targets under hidden-composition stress")
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _synthetic_model_rows(*, split: str, replicates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in _SEGMENTS:
        for product in _PRODUCTS:
            for platform in _PLATFORMS:
                for channel in _CHANNELS:
                    for tenure in _TENURES:
                        for geo in _GEOS:
                            for support in _SUPPORT:
                                base = {
                                    "customer_segment": segment,
                                    "product_family": product,
                                    "platform": platform,
                                    "acquisition_channel": channel,
                                    "tenure_band": tenure,
                                    "geo_market": geo,
                                    "support_load": support,
                                }
                                for replicate in range(replicates):
                                    row = dict(base)
                                    row[OBSERVED_COLUMN] = _observed_risk(
                                        row,
                                        split=split,
                                        replicate=replicate,
                                    )
                                    rows.append(row)
    return rows


def _fit_linear_model(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    x = np.array([_feature_vector(row) for row in rows], dtype=float)
    y = np.array([float(row[OBSERVED_COLUMN]) for row in rows], dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def _predict_risk(row: Mapping[str, Any], coefficients: np.ndarray) -> float:
    value = float(np.dot(np.array(_feature_vector(row), dtype=float), coefficients))
    return min(max(value, 0.0), 1.0)


def _feature_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0,
        _code(row["customer_segment"], _SEGMENTS),
        1.0 if row["product_family"] == "analytics" else 0.0,
        1.0 if row["platform"] == "mobile" else 0.0,
        1.0 if row["acquisition_channel"] == "partner" else 0.0,
        1.0 if row["acquisition_channel"] == "self_serve" else 0.0,
        1.0 if row["tenure_band"] == "new" else 0.0,
        1.0 if row["geo_market"] == "europe" else 0.0,
        1.0 if row["geo_market"] == "latam" else 0.0,
        1.0 if row["support_load"] == "high" else 0.0,
    )


def _true_risk(row: Mapping[str, Any]) -> float:
    value = 0.075
    value += {"enterprise": -0.030, "midmarket": 0.015, "smb": 0.050}[
        row["customer_segment"]
    ]
    value += {"core": -0.010, "analytics": 0.012}[row["product_family"]]
    value += {"web": -0.004, "mobile": 0.014}[row["platform"]]
    value += {
        "sales_led": -0.016,
        "partner": 0.006,
        "self_serve": 0.030,
    }[row["acquisition_channel"]]
    value += {"new": 0.030, "established": -0.010}[row["tenure_band"]]
    value += {"north_america": -0.006, "europe": 0.005, "latam": 0.024}[
        row["geo_market"]
    ]
    value += {"low": -0.008, "high": 0.030}[row["support_load"]]
    if row["acquisition_channel"] == "self_serve" and row["tenure_band"] == "new":
        value += 0.014
    if row["platform"] == "mobile" and row["support_load"] == "high":
        value += 0.012
    if row["customer_segment"] == "enterprise" and row["product_family"] == "analytics":
        value -= 0.006
    return min(max(value, 0.005), 0.45)


def _local_uncertainty_scale(row: Mapping[str, Any]) -> float:
    value = 0.025
    if row["customer_segment"] == "smb":
        value += 0.008
    if row["acquisition_channel"] == "self_serve":
        value += 0.010
    if row["tenure_band"] == "new":
        value += 0.012
    if row["geo_market"] == "latam":
        value += 0.010
    if row["support_load"] == "high":
        value += 0.015
    if row["platform"] == "mobile":
        value += 0.006
    return value


def _observed_risk(row: Mapping[str, Any], *, split: str, replicate: int) -> float:
    noise = _deterministic_noise(row, split=split, replicate=replicate)
    return min(max(_true_risk(row) + noise, 0.0), 1.0)


def _deterministic_noise(
    row: Mapping[str, Any],
    *,
    split: str,
    replicate: int,
) -> float:
    key = "|".join(str(row[column]) for column in HIDDEN_COLUMNS)
    key = f"{split}|{replicate}|{key}"
    raw = sum((index + 1) * ord(char) for index, char in enumerate(key)) % 1009
    centered = (raw / 1008.0) - 0.5
    return centered * _local_uncertainty_scale(row) * 1.55


def _conformal_quantile(scores: Sequence[float], *, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not scores:
        raise ValueError("scores must be non-empty")
    sorted_scores = sorted(float(score) for score in scores)
    rank = math.ceil((len(sorted_scores) + 1) * (1.0 - alpha))
    index = min(max(rank - 1, 0), len(sorted_scores) - 1)
    return sorted_scores[index]


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    weight: str,
) -> float:
    numerator = sum(float(row[target]) * float(row[weight]) for row in rows)
    denominator = sum(float(row[weight]) for row in rows)
    return numerator / denominator


def _without_title(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0]:
            lines = lines[1:]
    return lines


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _code(value: Any, choices: Sequence[Any]) -> float:
    return choices.index(value) / max(len(choices) - 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional PNG output path for a target-interval plot.",
    )
    parser.add_argument(
        "--include-attribution",
        action="store_true",
        help="Include Shapley-style refinement attribution in the audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = render_report(include_attribution=args.include_attribution)
    if args.output is None:
        print(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.plot_output is not None:
        path = plot_target_intervals(build_stability_report(), args.plot_output)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
