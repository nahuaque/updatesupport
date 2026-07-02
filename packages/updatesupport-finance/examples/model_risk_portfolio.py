"""Synthetic financial model-risk report example."""

from __future__ import annotations

import updatesupport_finance as usf


def synthetic_portfolio_rows():
    return [
        {
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "broker_channel": "broker",
            "employment_type": "salaried",
            "vintage": "2024",
            "pd": 0.018,
            "lgd": 0.35,
            "ead": 120_000,
        },
        {
            "product": "mortgage",
            "region": "north",
            "fico_band": "prime",
            "ltv_band": "low",
            "broker_channel": "direct",
            "employment_type": "self_employed",
            "vintage": "2023",
            "pd": 0.041,
            "lgd": 0.48,
            "ead": 80_000,
        },
        {
            "product": "mortgage",
            "region": "south",
            "fico_band": "near_prime",
            "ltv_band": "high",
            "broker_channel": "broker",
            "employment_type": "contractor",
            "vintage": "2022",
            "pd": 0.072,
            "lgd": 0.56,
            "ead": 60_000,
        },
        {
            "product": "auto",
            "region": "south",
            "fico_band": "near_prime",
            "ltv_band": "high",
            "broker_channel": "dealer",
            "employment_type": "salaried",
            "vintage": "2024",
            "pd": 0.055,
            "lgd": 0.62,
            "ead": 22_000,
        },
    ]


def main() -> None:
    report = usf.model_risk_report(
        synthetic_portfolio_rows(),
        public=["product", "region", "fico_band", "ltv_band"],
        hidden=[
            "product",
            "region",
            "fico_band",
            "ltv_band",
            "broker_channel",
            "employment_type",
            "vintage",
        ],
        metric=usf.expected_loss(pd="pd", lgd="lgd"),
        exposure="ead",
        candidate_refinements=["broker_channel", "employment_type", "vintage"],
        q=usf.q_portfolio_mix_shift(radius=0.35),
        model_id="EL_SYNTHETIC_001",
        portfolio_name="Synthetic retail credit portfolio",
        as_of_date="2026-06-30",
        intended_use="Expected-loss segmentation model review",
        ambiguity_limit=0.0025,
        public_adequacy_required=False,
        top=3,
    )
    print(report.to_markdown())


if __name__ == "__main__":
    main()
