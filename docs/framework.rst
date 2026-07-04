Framework Overview
==================

Core Question
-------------

``updatesupport`` asks a reporting-stability question:

   Holding the public report distribution fixed, how far could the reported
   aggregate target move if hidden composition inside public buckets changed?

The framework is intentionally downstream of estimation. A causal estimator,
survey estimator, model, or business metric supplies retained fine-cell target
values. ``updatesupport`` audits whether the public representation is adequate
for reporting that supplied target.

Here, "hidden" means retained but not publicly reported. It does not mean
unobserved or unknowable. Every reported ambiguity interval is conditional on
the retained refinement and admissible shift class ``Q`` declared for the
audit.

Finite Problem
--------------

A compiled audit has:

``D``
   finite retained fine cells.

``pi: D -> O``
   the public projection from retained cells to public report buckets.

``h(d)``
   supplied retained-cell target values.

``Q``
   an admissible class of retained-cell distributions or composition shifts.

For a linear target, the interval solves:

.. math::

   \inf/\sup \sum_{d \in D} h(d) q(d)

subject to the selected admissible constraints and the fixed observed public
law. The interval width is the transport ambiguity.

Target Contracts
----------------

The default tabular target is linear: ``sum_d h(d) q(d)``. Core also includes
target contracts for supported ratio targets, moment-transform targets, and
procedure-aware workflows. Unsupported nonlinear targets should be reformulated
explicitly before solving.

Transport Presets
-----------------

The package includes several admissible hidden-shift presets:

* saturated public fibers,
* bounded per-cell shifts,
* total-variation budgets,
* chi-square and KL budgets,
* L2 and Mahalanobis budgets,
* Wasserstein budgets,
* covariate-balance constraints,
* support-floor and mixed-integer design helpers.

Convex presets use CVXPY when the ``cvxpy`` extra is installed. Simple finite
linear presets can run without CVXPY.

Reports
-------

The primary user-facing artifact is :class:`updatesupport.ClaimAudit`, produced
by declaring a :class:`updatesupport.ClaimSpec` with :func:`updatesupport.claim`
and calling ``claim.audit(rows_or_frame)``. The claim audit wraps the
lower-level evidence into one verdict: reported estimate, statistical
uncertainty when supplied, hidden-composition interval, counterexample witness,
repair/certification evidence, refinement recommendations, and limitations.

:class:`updatesupport.PublicDescentReport` remains the lower-level evidence
object for the primary partial-ID interval. Use it directly when you do not
want a pass/fail/inconclusive claim verdict.

Refinement And Frontier Search
------------------------------

Refinement tools ask which hidden variables would make the public representation
more stable. Frontier tools search for small public representations that satisfy
ambiguity or bucket-budget constraints.

Estimator Handoffs
------------------

Adapter helpers connect estimator outputs to support audits:

* :func:`updatesupport.adapt_econml_effects`
* :func:`updatesupport.adapt_dowhy_effects`
* :func:`updatesupport.adapt_doubleml_effects`
* :func:`updatesupport.adapt_dataframe_effects`

These helpers do not estimate causal effects themselves. They attach supplied
effect values, such as ``tau_hat = estimator.effect(X)``, to rows and then run
the representation-stability audit.
