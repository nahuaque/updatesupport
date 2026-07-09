"""Representation adequacy and transport-stability auditing in Python."""

# ruff: noqa: F401

from importlib.metadata import PackageNotFoundError, version

from .adapters import (
    ConformalAdapterResult,
    EstimatorAdapterResult,
    adapt_conformal_classification,
    adapt_conformal_regression,
    adapt_dataframe_effects,
    adapt_doubleml_effects,
    adapt_dowhy_effects,
    adapt_econml_effects,
)
from .breakdown import (
    BreakdownCurvePoint,
    BreakdownPointReport,
    breakdown_point,
)
from .calibration import (
    HistoricalTVCalibrationReport,
    HistoricalTVTransition,
    RollingTVBacktest,
    calibrate_tv_radius,
)
from .certificate import (
    RepresentationStabilityCertificate,
    certify_public_representation,
)
from .claim import (
    ClaimAudit,
    ClaimNode,
    ClaimNodeAudit,
    ClaimRepairOption,
    ClaimRepairPlan,
    ClaimRefinementRecommendation,
    ClaimScreeningResult,
    ClaimSpec,
    ClaimTree,
    ClaimTreeAudit,
    DecisionResult,
    DecisionRule,
    ModelAssistedDrawResult,
    ModelAssistedStabilitySummary,
    PublicReportDesign,
    audit_claim,
    audit_claim_tree,
    claim,
    claim_tree,
    design_public_report,
    plan_claim_repair,
    threshold_decision,
)
from .comparison import (
    ComparisonItemResult,
    PairwiseComparisonResult,
    RobustComparisonReport,
    robust_comparison_report,
    robust_ranking_report,
)
from .conformal import (
    ConformalReportingStabilityReport,
    ConformalTargetAudit,
    ConformalTargetSpec,
    conformal_reporting_stability,
)
from .data import DataDiagnostic, DataDiagnostics, GroupedProblem, from_dataframe
from .dowhy import (
    DoWhyRepresentationAudit,
    audit_dowhy_effects,
    dowhy_refutation_from_report,
)
from .environments import (
    BatchedCvxpyEnvironments,
    CvxpyEnvironments,
    CvxpyConstraintMetadata,
    CvxpyError,
    ConvexAdmissibleSet,
    FiniteEnvironments,
    LinearConstraint,
    LineSegment,
    LPError,
    ParameterizedCvxpyEnvironments,
    PolytopeEnvironments,
    PublicFiberSaturated,
    SupportFunctionBackend,
    SupportFunctionIntervalResult,
    SupportFunctionReport,
    SupportFunctionResult,
    SupportFunctionTargetInterval,
    cvxpy_constraint,
    eq,
    geq,
    leq,
    linear_constraint,
    support_function_report,
)
from .exports import (
    report_dataframes,
    report_tables,
    report_to_json,
    tables_to_dataframes,
)
from .frontier import (
    FrontierCandidateExplanation,
    FrontierCloseAlternative,
    FrontierScenarioResult,
    FrontierScreenedRefinement,
    FrontierScreeningSummary,
    FrontierSearchTrace,
    FrontierScenarioComparison,
    PublicRepresentationCandidate,
    PublicRepresentationFrontier,
    public_representation_frontier,
)
from .joint import (
    HiddenCompositionUncertaintyReport,
    HiddenCompositionUncertaintyRow,
    JointCell,
    JointDistributionDraw,
    NonparametricJointDistribution,
    UncertaintyMetricSummary,
    fit_joint_distribution,
    hidden_composition_uncertainty,
    joint_draw_records,
)
from .linear_feasibility import (
    DEFAULT_LINEAR_FEASIBILITY_LIMITATIONS,
    NamedLinearClaim,
    NamedLinearClaimAudit,
    NamedLinearConstraint,
    NamedLinearConstraintAttribution,
    NamedLinearConstraintAttributionReport,
    NamedLinearConstraintDiagnostic,
    NamedLinearEndpoint,
    NamedLinearExpression,
    NamedLinearFeasibilityProblem,
    NamedLinearFeasibilityReport,
    NamedLinearInterval,
    NamedLinearScenario,
    NamedLinearTarget,
    NamedLinearVariable,
    audit_named_linear_claim,
    attribute_named_linear_constraints,
    named_linear_claim,
    named_linear_constraint,
    named_linear_expression,
    named_linear_feasibility_problem,
    named_linear_scenario,
    named_linear_target,
    named_linear_variable,
    solve_named_linear_feasibility,
)
from .metrics import RowMetric, row_metric
from .partition import Partition, PartitionError
from .plugins import (
    PluginMetadata,
    PluginRegistry,
    PluginValidationIssue,
    PluginValidationReport,
    UpdateSupportPlugin,
    assert_valid_plugin,
    discover_plugins,
    get_plugin,
    list_plugins,
    plugin_compiler,
    plugin_metric,
    plugin_q_preset,
    plugin_report_profile,
    register_plugin,
    unregister_plugin,
    validate_plugin,
)
from .problem import FiniteProblem, TooManyPartitions
from .presets import (
    CvxpyAdmissibleSetSpec,
    QPreset,
    cvxpy_admissible_set_spec,
    q_bounded_shift,
    q_chi_square_budget,
    q_covariate_balance,
    q_fiber_support_floor,
    q_intersection,
    q_kl_budget,
    q_l2_budget,
    q_mahalanobis_budget,
    q_observed,
    q_saturated,
    q_tv_budget,
    q_wasserstein,
)
from .report import (
    CausalReportingStabilitySuite,
    EstimatorUncertaintyAdjustment,
    InteractionRefinementCandidate,
    InteractionRefinementReport,
    PublicDescentReport,
    PublicFiberDiagnostic,
    RefinementAttribution,
    RefinementAttributionReport,
    RefinementCandidate,
    RefinementCoalitionEvaluation,
    RefinementSensitivityCandidate,
    RefinementSensitivityReport,
    RefinementSensitivityRow,
    RefinementSensitivityScenario,
    SensitivityReport,
    SensitivityRow,
    SensitivitySummary,
    StatisticalUncertainty,
    WitnessCellShift,
    WitnessFiberShift,
    WitnessReport,
    attribute_refinement_ambiguity,
    audit_effects,
    causal_reporting_stability,
    public_descent_report,
    public_fiber_diagnostics,
    recommend_refinements,
    recommend_refinement_interactions,
    recommend_refinements_sensitivity,
    sensitivity_report,
    witness_report,
)
from .residopt_backend import (
    ResidOptAvailability,
    ResidOptEndpointCertificate,
    ResidOptEndpointReport,
    ResidOptL2EndpointCompiler,
    ResidOptRefinementScreenCandidate,
    ResidOptRefinementScreenContext,
    ResidOptRefinementScreenReport,
    residopt_available,
    residopt_l2_support_interval,
    residopt_refinement_screen,
)
from .results import (
    AdequacyResult,
    CardinalGapResult,
    ConstraintDual,
    LeastSupportResult,
    TransportResult,
    UncertainLinearConfidenceCoreResult,
    Witness,
)
from .rollup import (
    CategoricalRollupCandidate,
    CategoricalRollupDesign,
    design_categorical_rollup,
)
from .spec import AuditRun, AuditSpec, QSpec, run_audit
from .targets import (
    LinearTarget,
    MomentTransformTarget,
    ProcedureTarget,
    ProcedureTargetContext,
    RatioTarget,
    TargetCapabilities,
    TargetContract,
    UncertainLinearTarget,
    UnsupportedTarget,
    UnsupportedTargetError,
)

try:
    __version__ = version("updatesupport")
except PackageNotFoundError:
    __version__ = "0.0.0"

_CLAIM_API = [
    "claim",
    "audit_claim",
    "calibrate_tv_radius",
    "design_categorical_rollup",
    "claim_tree",
    "audit_claim_tree",
    "ClaimSpec",
    "ClaimAudit",
    "HistoricalTVCalibrationReport",
    "CategoricalRollupDesign",
    "ClaimRepairPlan",
    "PublicReportDesign",
    "ClaimTree",
    "ClaimTreeAudit",
    "DecisionRule",
    "DecisionResult",
    "threshold_decision",
    "design_public_report",
]

_DATA_AND_EXPORT_API = [
    "from_dataframe",
    "GroupedProblem",
    "report_to_json",
    "report_tables",
    "report_dataframes",
    "tables_to_dataframes",
]

_REPORTING_API = [
    "public_descent_report",
    "witness_report",
    "sensitivity_report",
    "recommend_refinements",
    "recommend_refinement_interactions",
    "recommend_refinements_sensitivity",
    "attribute_refinement_ambiguity",
    "certify_public_representation",
    "public_representation_frontier",
    "breakdown_point",
    "robust_comparison_report",
    "robust_ranking_report",
]

_PRESET_API = [
    "QPreset",
    "QSpec",
    "q_saturated",
    "q_observed",
    "q_bounded_shift",
    "q_tv_budget",
    "q_chi_square_budget",
    "q_kl_budget",
    "q_l2_budget",
    "q_covariate_balance",
    "q_mahalanobis_budget",
    "q_wasserstein",
    "q_fiber_support_floor",
    "q_intersection",
]

_INTEGRATION_API = [
    "fit_joint_distribution",
    "hidden_composition_uncertainty",
    "audit_effects",
    "causal_reporting_stability",
    "adapt_dataframe_effects",
    "adapt_doubleml_effects",
    "adapt_dowhy_effects",
    "adapt_econml_effects",
    "adapt_conformal_regression",
    "adapt_conformal_classification",
    "conformal_reporting_stability",
    "ConformalTargetSpec",
    "ConformalReportingStabilityReport",
]

_SPEC_AND_EXTENSION_API = [
    "AuditSpec",
    "AuditRun",
    "run_audit",
    "discover_plugins",
    "register_plugin",
    "unregister_plugin",
    "validate_plugin",
]

__all__ = [
    "__version__",
    *_CLAIM_API,
    *_DATA_AND_EXPORT_API,
    *_REPORTING_API,
    *_PRESET_API,
    *_INTEGRATION_API,
    *_SPEC_AND_EXTENSION_API,
]
