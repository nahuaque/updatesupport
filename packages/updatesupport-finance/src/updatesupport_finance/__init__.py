"""Financial model-risk extensions for updatesupport."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .metrics import (
    default_rate,
    expected_loss,
    expected_loss_amount,
    loss_given_default,
)
from .portfolio import (
    ModelRiskMetadata,
    ModelRiskReport,
    ReviewThresholds,
    from_portfolio,
    model_risk_report,
)
from .presets import q_exposure_weighted_tv, q_portfolio_mix_shift
from .plugin import plugin

try:
    __version__ = version("updatesupport-finance")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "default_rate",
    "expected_loss",
    "expected_loss_amount",
    "from_portfolio",
    "loss_given_default",
    "ModelRiskMetadata",
    "ModelRiskReport",
    "model_risk_report",
    "plugin",
    "q_exposure_weighted_tv",
    "q_portfolio_mix_shift",
    "ReviewThresholds",
]
