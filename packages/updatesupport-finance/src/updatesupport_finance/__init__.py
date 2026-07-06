"""Financial model-risk extensions for updatesupport."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .disclosure import (
    DisclosureConstraint,
    DisclosureConstraintAttribution,
    DisclosureConstraintAttributionReport,
    DisclosureExpression,
    DisclosureTarget,
    DisclosureTier,
    DisclosureTriangulationReport,
    DisclosureTriangulationSpec,
    DisclosureVariable,
    attribute_disclosure_constraints,
    containment_constraint,
    disclosure_constraint,
    disclosure_target,
    disclosure_tier,
    disclosure_triangulation_spec,
    disclosure_variable,
    exact_disclosure_constraint,
    interval_disclosure_constraint,
    rounded_growth_constraints,
    triangulate_disclosure,
)
from .metrics import (
    default_rate,
    expected_loss,
    expected_loss_amount,
    expected_loss_standard_error,
    loss_given_default,
)
from .portfolio import (
    FinanceStabilityCertificate,
    ModelRiskMetadata,
    ModelRiskReport,
    ReviewThresholds,
    certify_portfolio_segmentation,
    from_portfolio,
    model_assisted_portfolio_uncertainty,
    model_risk_report,
)
from .presets import (
    finance_sensitivity_grid,
    portfolio_concentration_moments,
    portfolio_factor_moments,
    q_exposure_weighted_tv,
    q_factor_exposure_shift,
    q_portfolio_mix_shift,
    q_regional_concentration_shift,
)
from .plugin import plugin

try:
    __version__ = version("updatesupport-finance")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "attribute_disclosure_constraints",
    "containment_constraint",
    "default_rate",
    "disclosure_constraint",
    "disclosure_target",
    "disclosure_tier",
    "disclosure_triangulation_spec",
    "disclosure_variable",
    "DisclosureConstraint",
    "DisclosureConstraintAttribution",
    "DisclosureConstraintAttributionReport",
    "DisclosureExpression",
    "DisclosureTarget",
    "DisclosureTier",
    "DisclosureTriangulationReport",
    "DisclosureTriangulationSpec",
    "DisclosureVariable",
    "expected_loss",
    "expected_loss_amount",
    "expected_loss_standard_error",
    "exact_disclosure_constraint",
    "FinanceStabilityCertificate",
    "finance_sensitivity_grid",
    "certify_portfolio_segmentation",
    "from_portfolio",
    "interval_disclosure_constraint",
    "loss_given_default",
    "ModelRiskMetadata",
    "ModelRiskReport",
    "model_assisted_portfolio_uncertainty",
    "model_risk_report",
    "plugin",
    "portfolio_concentration_moments",
    "portfolio_factor_moments",
    "q_exposure_weighted_tv",
    "q_factor_exposure_shift",
    "q_portfolio_mix_shift",
    "q_regional_concentration_shift",
    "rounded_growth_constraints",
    "ReviewThresholds",
    "triangulate_disclosure",
]
