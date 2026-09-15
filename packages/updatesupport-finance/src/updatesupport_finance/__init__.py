"""Financial model-risk extensions for updatesupport."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .allocations import (
    DisclosureAllocation,
    DisclosureAllocationReport,
    disclosure_allocations,
)
from .evidence import (
    DisclosureEvidenceDiagnostic,
    DisclosureFact,
    disclosure_fact_constraint,
    link_disclosure_evidence,
    validate_disclosure_evidence,
)
from .relationships import (
    reconciliation_constraint,
    rounded_amount_constraint,
    stock_flow_constraint,
)
from .snapshots import (
    DisclosureSnapshot,
    capture_disclosure_snapshot,
    compare_disclosure_snapshots,
)

from .disclosure import (
    DisclosureConflictReport,
    find_disclosure_conflict,
    DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS,
    DisclosureAuditPack,
    DisclosureClaim,
    DisclosureClaimAudit,
    DisclosureConstraint,
    DisclosureConstraintAttribution,
    DisclosureConstraintAttributionReport,
    DisclosureConstraintDiagnostic,
    DisclosureExpression,
    DisclosureTarget,
    DisclosureTier,
    DisclosureTriangulationReport,
    DisclosureTriangulationSpec,
    DisclosureVariable,
    audit_disclosure_claim,
    attribute_disclosure_constraints,
    containment_constraint,
    disclosure_audit_pack,
    disclosure_constraint,
    disclosure_claim,
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
    "DisclosureConflictReport",
    "find_disclosure_conflict",
    "DisclosureAllocation",
    "DisclosureAllocationReport",
    "DisclosureEvidenceDiagnostic",
    "DisclosureFact",
    "DisclosureSnapshot",
    "capture_disclosure_snapshot",
    "compare_disclosure_snapshots",
    "disclosure_allocations",
    "disclosure_fact_constraint",
    "link_disclosure_evidence",
    "reconciliation_constraint",
    "rounded_amount_constraint",
    "stock_flow_constraint",
    "validate_disclosure_evidence",
    "__version__",
    "DEFAULT_DISCLOSURE_AUDIT_LIMITATIONS",
    "audit_disclosure_claim",
    "attribute_disclosure_constraints",
    "containment_constraint",
    "default_rate",
    "disclosure_audit_pack",
    "disclosure_claim",
    "disclosure_constraint",
    "disclosure_target",
    "disclosure_tier",
    "disclosure_triangulation_spec",
    "disclosure_variable",
    "DisclosureAuditPack",
    "DisclosureClaim",
    "DisclosureClaimAudit",
    "DisclosureConstraint",
    "DisclosureConstraintAttribution",
    "DisclosureConstraintAttributionReport",
    "DisclosureConstraintDiagnostic",
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
