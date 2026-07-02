"""Synthetic financial model-risk report example.

Run from the repository root with:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/model_risk_portfolio.py

Optionally write the Markdown report:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/model_risk_portfolio.py \
        --output data/finance_model_risk_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import updatesupport as us
import updatesupport_finance as usf


PUBLIC_COLUMNS = ("product", "region", "fico_band", "ltv_band")
HIDDEN_COLUMNS = (
    "product",
    "region",
    "fico_band",
    "ltv_band",
    "broker_channel",
    "employment_type",
    "vintage",
    "hardship_history",
    "documentation_type",
    "local_housing_market",
    "cashflow_pattern",
)
CANDIDATE_REFINEMENTS = (
    "broker_channel",
    "employment_type",
    "vintage",
    "hardship_history",
    "documentation_type",
    "local_housing_market",
    "cashflow_pattern",
)


def synthetic_portfolio_rows() -> list[dict[str, Any]]:
    """Return a small synthetic retail-credit portfolio.

    The public reporting segmentation is intentionally coarse. Several hidden
    subgroups live inside the same public buckets and have different expected
    loss rates, which makes the representation-stability question visible.
    """

    return [
        {
            "account_count": 140,
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "broker_channel": "broker",
            "employment_type": "salaried",
            "vintage": "2024",
            "hardship_history": "none",
            "documentation_type": "full_doc",
            "local_housing_market": "stable",
            "cashflow_pattern": "stable",
            "pd": 0.014,
            "lgd": 0.32,
            "ead": 15_400_000,
        },
        {
            "account_count": 90,
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "broker_channel": "direct",
            "employment_type": "self_employed",
            "vintage": "2023",
            "hardship_history": "prior",
            "documentation_type": "alt_doc",
            "local_housing_market": "cooling",
            "cashflow_pattern": "seasonal",
            "pd": 0.038,
            "lgd": 0.46,
            "ead": 8_100_000,
        },
        {
            "account_count": 110,
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "medium",
            "broker_channel": "broker",
            "employment_type": "salaried",
            "vintage": "2022",
            "hardship_history": "none",
            "documentation_type": "full_doc",
            "local_housing_market": "cooling",
            "cashflow_pattern": "stable",
            "pd": 0.023,
            "lgd": 0.39,
            "ead": 12_650_000,
        },
        {
            "account_count": 70,
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "medium",
            "broker_channel": "correspondent",
            "employment_type": "contractor",
            "vintage": "2021",
            "hardship_history": "prior",
            "documentation_type": "full_doc",
            "local_housing_market": "declining",
            "cashflow_pattern": "volatile",
            "pd": 0.049,
            "lgd": 0.52,
            "ead": 7_700_000,
        },
        {
            "account_count": 85,
            "product": "mortgage",
            "region": "south",
            "fico_band": "near_prime",
            "ltv_band": "high",
            "broker_channel": "broker",
            "employment_type": "salaried",
            "vintage": "2024",
            "hardship_history": "none",
            "documentation_type": "full_doc",
            "local_housing_market": "stable",
            "cashflow_pattern": "stable",
            "pd": 0.058,
            "lgd": 0.56,
            "ead": 7_225_000,
        },
        {
            "account_count": 65,
            "product": "mortgage",
            "region": "south",
            "fico_band": "near_prime",
            "ltv_band": "high",
            "broker_channel": "correspondent",
            "employment_type": "self_employed",
            "vintage": "2022",
            "hardship_history": "current",
            "documentation_type": "alt_doc",
            "local_housing_market": "declining",
            "cashflow_pattern": "volatile",
            "pd": 0.118,
            "lgd": 0.67,
            "ead": 5_525_000,
        },
        {
            "account_count": 160,
            "product": "auto",
            "region": "west",
            "fico_band": "prime",
            "ltv_band": "medium",
            "broker_channel": "dealer",
            "employment_type": "salaried",
            "vintage": "2024",
            "hardship_history": "none",
            "documentation_type": "full_doc",
            "local_housing_market": "stable",
            "cashflow_pattern": "stable",
            "pd": 0.031,
            "lgd": 0.50,
            "ead": 3_840_000,
        },
        {
            "account_count": 100,
            "product": "auto",
            "region": "west",
            "fico_band": "prime",
            "ltv_band": "medium",
            "broker_channel": "online",
            "employment_type": "contractor",
            "vintage": "2023",
            "hardship_history": "prior",
            "documentation_type": "stated_income",
            "local_housing_market": "stable",
            "cashflow_pattern": "seasonal",
            "pd": 0.061,
            "lgd": 0.58,
            "ead": 2_400_000,
        },
        {
            "account_count": 190,
            "product": "card",
            "region": "east",
            "fico_band": "near_prime",
            "ltv_band": "na",
            "broker_channel": "direct",
            "employment_type": "salaried",
            "vintage": "2024",
            "hardship_history": "none",
            "documentation_type": "full_doc",
            "local_housing_market": "stable",
            "cashflow_pattern": "stable",
            "pd": 0.074,
            "lgd": 0.82,
            "ead": 1_900_000,
        },
        {
            "account_count": 120,
            "product": "card",
            "region": "east",
            "fico_band": "near_prime",
            "ltv_band": "na",
            "broker_channel": "affiliate",
            "employment_type": "contractor",
            "vintage": "2022",
            "hardship_history": "current",
            "documentation_type": "thin_file",
            "local_housing_market": "cooling",
            "cashflow_pattern": "volatile",
            "pd": 0.132,
            "lgd": 0.90,
            "ead": 1_200_000,
        },
    ]


def build_report(
    *,
    q_radius: float = 0.35,
    ambiguity_limit: float = 0.006,
) -> usf.ModelRiskReport:
    return usf.model_risk_report(
        synthetic_portfolio_rows(),
        public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        metric=usf.expected_loss(pd="pd", lgd="lgd"),
        exposure="ead",
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q=usf.q_portfolio_mix_shift(radius=q_radius),
        model_id="EL_SYNTHETIC_RETAIL_001",
        portfolio_name="Synthetic retail credit portfolio",
        as_of_date="2026-06-30",
        intended_use="Expected-loss segmentation model review",
        ambiguity_limit=ambiguity_limit,
        public_adequacy_required=False,
        top=5,
    )


def build_frontier(
    *,
    q_radius: float = 0.35,
    ambiguity_limit: float = 0.006,
) -> us.PublicRepresentationFrontier:
    """Choose the smallest stable public segmentation for the portfolio metric."""

    return us.public_representation_frontier(
        synthetic_portfolio_rows(),
        base_public=PUBLIC_COLUMNS,
        hidden=HIDDEN_COLUMNS,
        target=usf.expected_loss(pd="pd", lgd="lgd"),
        weight="ead",
        candidate_refinements=CANDIDATE_REFINEMENTS,
        q_presets=(
            "saturated",
            usf.q_portfolio_mix_shift(radius=q_radius),
            "observed",
        ),
        min_cell_weights=(1.0,),
        ambiguity_limit=ambiguity_limit,
        bucket_budget=12,
        search="beam",
        beam_width=8,
        max_added_columns=3,
        max_evaluations=96,
        title="Synthetic Finance Public Representation Frontier",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic finance model-risk report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    parser.add_argument(
        "--q-radius",
        type=float,
        default=0.35,
        help="Portfolio mix-shift radius for the bounded-shift Q preset.",
    )
    parser.add_argument(
        "--ambiguity-limit",
        type=float,
        default=0.006,
        help="Review threshold for hidden-composition ambiguity.",
    )
    args = parser.parse_args()

    report = build_report(
        q_radius=args.q_radius,
        ambiguity_limit=args.ambiguity_limit,
    )
    frontier = build_frontier(
        q_radius=args.q_radius,
        ambiguity_limit=args.ambiguity_limit,
    )
    markdown = report.to_markdown() + "\n\n" + frontier.to_markdown()
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
