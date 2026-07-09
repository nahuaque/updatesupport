High-Level API
==============

Common entry points imported from ``updatesupport``:

.. autosummary::

   updatesupport.claim
   updatesupport.audit_claim
   updatesupport.calibrate_tv_radius
   updatesupport.claim_tree
   updatesupport.audit_claim_tree
   updatesupport.from_dataframe
   updatesupport.run_audit

Lower-level evidence and search entry points:

.. autosummary::

   updatesupport.public_descent_report
   updatesupport.audit_effects
   updatesupport.causal_reporting_stability
   updatesupport.conformal_reporting_stability
   updatesupport.sensitivity_report
   updatesupport.recommend_refinements
   updatesupport.recommend_refinement_interactions
   updatesupport.attribute_refinement_ambiguity
   updatesupport.recommend_refinements_sensitivity
   updatesupport.public_representation_frontier
   updatesupport.certify_public_representation

Core result objects:

.. autosummary::

   updatesupport.ClaimSpec
   updatesupport.ClaimAudit
   updatesupport.HistoricalTVCalibrationReport
   updatesupport.ClaimNode
   updatesupport.ClaimNodeAudit
   updatesupport.ClaimTree
   updatesupport.ClaimTreeAudit
   updatesupport.ClaimRefinementRecommendation
   updatesupport.RefinementAttributionReport
   updatesupport.RefinementAttribution
   updatesupport.RefinementCoalitionEvaluation
   updatesupport.ConformalReportingStabilityReport
   updatesupport.GroupedProblem
   updatesupport.PublicDescentReport
   updatesupport.PublicRepresentationFrontier
   updatesupport.RepresentationStabilityCertificate
   updatesupport.AuditSpec
   updatesupport.AuditRun
