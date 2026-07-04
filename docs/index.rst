updatesupport Documentation
===========================

``updatesupport`` audits whether a reported claim survives hidden subgroup
recomposition. It quantifies hidden-composition ambiguity: how far an aggregate
rate, effect, or risk metric can move when the public distribution is fixed but
the retained, not-publicly-reported mix inside public buckets changes.

This documentation has three layers:

* a short quickstart for the common tabular workflow,
* framework guides explaining the model, targets, transport presets, and
  reporting artifacts,
* API reference pages generated from the core Python modules.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   framework
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Core Guides

   representation-adequacy
   api-surface
   positioning-and-lineage
   mathematical-statistical-soundness
   theory-and-backends
   transport-presets
   public-representation-frontier
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

   causal-library-integration
   benchmark-gallery
   extensions
   folktables-acs-income-interpretation

.. toctree::
   :maxdepth: 1
   :caption: Project

   releasing
