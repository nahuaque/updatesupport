"""updatesupport plugin descriptor for finance extensions."""

from __future__ import annotations

import updatesupport as us

from .metrics import (
    default_rate,
    expected_loss,
    expected_loss_amount,
    loss_given_default,
)
from .portfolio import from_portfolio, model_risk_report
from .presets import q_exposure_weighted_tv, q_portfolio_mix_shift


plugin = us.UpdateSupportPlugin(
    name="finance",
    version="0.1.0",
    description="Financial model-risk metrics, Q presets, and report profiles.",
    metrics={
        "default_rate": default_rate,
        "expected_loss": expected_loss,
        "expected_loss_amount": expected_loss_amount,
        "loss_given_default": loss_given_default,
    },
    q_presets={
        "portfolio_mix_shift": q_portfolio_mix_shift,
        "exposure_weighted_tv": q_exposure_weighted_tv,
    },
    report_profiles={
        "model_risk": model_risk_report,
    },
    compilers={
        "portfolio": from_portfolio,
    },
    metadata=us.PluginMetadata(
        package="updatesupport-finance",
        homepage="https://github.com/nahuaque/updatesupport",
        domain="financial-model-risk",
        tags=(
            "credit-risk",
            "expected-loss",
            "model-validation",
            "portfolio-stability",
        ),
        min_updatesupport_version="0.1.1",
    ),
)
