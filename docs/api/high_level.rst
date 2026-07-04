High-Level API
==============

Common entry points imported from ``updatesupport``:

.. autosummary::

   updatesupport.claim
   updatesupport.audit_claim
   updatesupport.from_dataframe
   updatesupport.run_audit

Lower-level evidence and search entry points:

.. autosummary::

   updatesupport.public_descent_report
   updatesupport.audit_effects
   updatesupport.causal_reporting_stability
   updatesupport.sensitivity_report
   updatesupport.recommend_refinements
   updatesupport.recommend_refinements_sensitivity
   updatesupport.public_representation_frontier
   updatesupport.certify_public_representation

Core result objects:

.. autosummary::

   updatesupport.ClaimSpec
   updatesupport.ClaimAudit
   updatesupport.ClaimRefinementRecommendation
   updatesupport.GroupedProblem
   updatesupport.PublicDescentReport
   updatesupport.PublicRepresentationFrontier
   updatesupport.RepresentationStabilityCertificate
   updatesupport.AuditSpec
   updatesupport.AuditRun
