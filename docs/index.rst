updatesupport
=============

.. raw:: html

   <section class="hero">
     <div class="hero-kicker">Claim audits for aggregation bias</div>
     <h1>Know whether a reported aggregate survives hidden subgroup recomposition.</h1>
     <p>
       <code>updatesupport</code> audits whether coarse public categories are stable
       enough for a reported estimate, decision, model metric, risk measure, or
       causal effect.
     </p>
     <div class="hero-actions">
       <a class="button primary" href="quickstart.html">Quickstart</a>
       <a class="button secondary" href="api-surface.html">API surface</a>
       <a class="button secondary" href="https://github.com/nahuaque/updatesupport">GitHub</a>
     </div>
   </section>

   <section class="value-grid">
     <a class="value-card" href="reporting-claims.html">
       <span>Claim audits</span>
       <strong>Declare the claim, then audit it.</strong>
       <p>Pass, fail, or explain why the current public representation is inconclusive.</p>
     </a>
     <a class="value-card" href="api-surface.html">
       <span>Simple surface</span>
       <strong>One front door, deeper tools behind it.</strong>
       <p>Use <code>us.claim(...).audit(rows)</code>; drop lower when you need evidence internals.</p>
     </a>
     <a class="value-card" href="public-representation-frontier.html">
       <span>Refinement intelligence</span>
       <strong>Find the smallest useful public representation.</strong>
       <p>Rank refinements, inspect interactions, and search reporting-design frontiers.</p>
     </a>
     <a class="value-card" href="transport-presets.html">
       <span>Stress tests</span>
       <strong>Choose the recomposition model explicitly.</strong>
       <p>Saturated fibers, TV/KL/chi-square budgets, Wasserstein, balance drift, and custom CVXPY sets.</p>
     </a>
   </section>

Quickstart
----------

.. code-block:: bash

   pip install updatesupport

.. code-block:: python

   import updatesupport as us

   claim = us.claim(
       "reported lift remains positive",
       public=["segment"],
       hidden=["segment", "channel", "tenure_band", "device"],
       target="treatment_minus_control_lift",
       weight="users",
       candidate_refinements=["channel", "tenure_band", "device"],
       decision=us.threshold_decision(">=", 0.0),
   )

   audit = claim.audit(rows_or_frame)
   print(audit.to_markdown())

What The Audit Separates
------------------------

.. raw:: html

   <section class="explain-strip">
     <div>
       <strong>Reported estimate</strong>
       <p>The target value supplied by your model, estimator, dashboard, or metric pipeline.</p>
     </div>
     <div>
       <strong>Statistical uncertainty</strong>
       <p>Optional standard errors or intervals from the upstream statistical workflow.</p>
     </div>
     <div>
       <strong>Hidden-composition ambiguity</strong>
       <p>The partial-ID interval induced by retained but not publicly reported subgroups.</p>
     </div>
     <div>
       <strong>Refinement recommendations</strong>
       <p>Which hidden variables most improve the public representation for this claim.</p>
     </div>
   </section>

Documentation Map
-----------------

Start with :doc:`quickstart`, then read :doc:`api-surface` for the consolidated
Python surface. Use :doc:`framework` and :doc:`mathematical-statistical-soundness`
for the modeling assumptions, and :doc:`transport-presets` when choosing a
stress-test family.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   quickstart
   framework
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Core Guides
   :hidden:

   representation-adequacy
   api-surface
   positioning-and-lineage
   mathematical-statistical-soundness
   theory-and-backends
   transport-presets
   historical-tv-calibration
   public-representation-frontier
   categorical-rollup-design
   shared-representation-design
   minimum-claim-breaking-witness
   representation-stability-certificates
   reporting-claims
   audit-specs
   data-diagnostics
   structured-exports
   model-assisted-joint-analysis
   breakdown-point-analysis
   robust-comparison-ranking
   interaction-aware-refinements

.. toctree::
   :maxdepth: 1
   :caption: Integrations And Case Studies
   :hidden:

   causal-library-integration
   conformal-mapie-integration
   residopt-integration
   benchmark-gallery
   revops-funnel-analysis
   extensions
   folktables-acs-income-interpretation

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   releasing
