"""Generate README sensitivity-analysis plots from built-in examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import updatesupport as us

from examples.acic_2016 import (
    EFFECT_COLUMN,
    WEIGHT_COLUMN,
    attach_oracle_effects,
    synthetic_acic_2016_source_rows,
)


DEFAULT_OUTPUT = Path("docs/assets/sensitivity-analysis-overview.png")


def generate_sensitivity_plot(output: Path = DEFAULT_OUTPUT) -> Path:
    """Generate the README sensitivity-analysis overview figure."""

    matplotlib, plt, sns, pd = _plotting_imports()
    rows, public_columns, hidden_columns, candidate_columns = _acic_oracle_rows()
    sensitivity = us.sensitivity_report(
        rows,
        public=public_columns,
        hidden=hidden_columns,
        target=EFFECT_COLUMN,
        weight=WEIGHT_COLUMN,
        min_cell_weights=[1, 5, 10, 15],
        q_presets=[
            "saturated",
            us.q_bounded_shift(0.5),
            us.q_bounded_shift(0.25),
            "observed",
        ],
    )
    refinements = us.recommend_refinements_sensitivity(
        rows,
        public=public_columns,
        hidden=hidden_columns,
        target=EFFECT_COLUMN,
        weight=WEIGHT_COLUMN,
        candidate_refinements=candidate_columns,
        min_cell_weights=[1, 5, 10, 15],
        q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
        top=None,
    )

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        rc={
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#f7f8fb",
            "grid.color": "#e5e7eb",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#374151",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        },
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.8, 14.2),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#f7f8fb")
    fig.subplots_adjust(left=0.18, right=0.92, top=0.86, bottom=0.11, hspace=0.52)

    _draw_ambiguity_heatmap(sensitivity, axes[0], sns, pd)
    _draw_refinement_ranking(refinements, axes[1], sns, pd)

    fig.suptitle(
        "Update-support sensitivity analysis",
        x=0.08,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.08,
        0.915,
        "ACIC 2016 synthetic oracle example. Widths are hidden-composition stress tests, not confidence intervals.",
        ha="left",
        fontsize=12,
        color="#4b5563",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    matplotlib.rcdefaults()
    return output


def _draw_ambiguity_heatmap(sensitivity, ax: Any, sns: Any, pd: Any) -> None:
    records = [
        {
            "Q preset": _pretty_q(row.q_name),
            "min_cell_weight": int(row.min_cell_weight),
            "ambiguity": row.ambiguity,
            "hidden_cells": row.hidden_cells,
        }
        for row in sensitivity.rows
        if row.status == "ok" and row.ambiguity is not None
    ]
    frame = pd.DataFrame(records)
    q_order = [
        "Saturated\n(any mix)",
        "Bounded shift\nr=0.50",
        "Bounded shift\nr=0.25",
        "Observed only\n(no shift)",
    ]
    pivot = (
        frame.pivot(index="Q preset", columns="min_cell_weight", values="ambiguity")
        .reindex(q_order)
        .sort_index(axis=1)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.color_palette("crest", as_cmap=True),
        annot=True,
        fmt=".3f",
        linewidths=1.4,
        linecolor="#ffffff",
        cbar_kws={"label": "ambiguity width", "shrink": 0.82},
        annot_kws={"fontsize": 10.5, "fontweight": "bold"},
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=10.5)
    colorbar.set_label("ambiguity width", fontsize=11)
    ax.set_title(
        "Ambiguity under stress scenarios",
        loc="left",
        pad=16,
        fontsize=15,
    )
    ax.set_xlabel("minimum hidden-cell weight", labelpad=10, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10.5)
    ax.text(
        0,
        -0.20,
        "Darker cells indicate more hidden-composition ambiguity.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#6b7280",
    )


def _draw_refinement_ranking(refinements, ax: Any, sns: Any, pd: Any) -> None:
    records = [
        {
            "candidate": _pretty_candidate(row.column),
            "mean_reduction": row.mean_reduction,
            "min_reduction": row.min_reduction,
            "max_reduction": row.max_reduction,
            "mean_pct": row.mean_reduction_percent,
            "top_rank_count": row.top_rank_count,
            "scenarios": row.evaluated_scenarios,
        }
        for row in refinements.candidates
    ]
    frame = pd.DataFrame(records).sort_values("mean_reduction", ascending=True)
    palette = sns.color_palette("flare", n_colors=len(frame))
    sns.barplot(
        data=frame,
        x="mean_reduction",
        y="candidate",
        hue="candidate",
        order=frame["candidate"],
        palette=palette,
        legend=False,
        ax=ax,
    )
    y_positions = range(len(frame))
    lower = frame["mean_reduction"] - frame["min_reduction"]
    upper = frame["max_reduction"] - frame["mean_reduction"]
    ax.errorbar(
        frame["mean_reduction"],
        y_positions,
        xerr=[lower, upper],
        fmt="none",
        ecolor="#111827",
        elinewidth=1.6,
        capsize=4,
        alpha=0.85,
    )
    max_x = max(frame["max_reduction"].max(), 0.001)
    ax.set_xlim(0, max_x * 1.28)
    for index, row in enumerate(frame.itertuples(index=False)):
        ax.text(
            row.mean_reduction + max_x * 0.035,
            index,
            f"{row.mean_pct:.0f}% avg\n{row.top_rank_count}/{row.scenarios} top",
            va="center",
            ha="left",
            fontsize=9.8,
            color="#374151",
        )
    ax.set_title("Refinements that reduce ambiguity", loc="left", pad=16, fontsize=15)
    ax.set_xlabel("mean ambiguity reduction across scenarios", labelpad=10, fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10.5)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.text(
        0,
        -0.22,
        "Bars show mean reduction. Whiskers show worst-to-best scenario reduction.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#6b7280",
    )


def _acic_oracle_rows():
    rows, public_columns, hidden_columns, candidate_columns = (
        synthetic_acic_2016_source_rows()
    )
    effect_result = attach_oracle_effects(rows, feature_columns=hidden_columns)
    return effect_result.rows, public_columns, hidden_columns, candidate_columns


def _plotting_imports():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit(
            "Install plot dependencies with: uv sync --extra examples"
        ) from exc
    return matplotlib, plt, sns, pd


def _pretty_q(name: str) -> str:
    if name == "saturated":
        return "Saturated\n(any mix)"
    if name == "observed":
        return "Observed only\n(no shift)"
    if name == "bounded_shift(radius=0.5)":
        return "Bounded shift\nr=0.50"
    if name == "bounded_shift(radius=0.25)":
        return "Bounded shift\nr=0.25"
    return name.replace("_", " ")


def _pretty_candidate(name: str) -> str:
    return name.removesuffix("_BAND").replace("_", " ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = generate_sensitivity_plot(args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
