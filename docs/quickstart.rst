Quickstart
==========

Install the core package:

.. code-block:: bash

   pip install updatesupport

or, in a ``uv`` project:

.. code-block:: bash

   uv add updatesupport

The most common workflow starts with tabular rows or a dataframe. Choose:

* public columns: what the report shows,
* hidden columns: the more detailed retained state space,
* target: the supplied metric, rate, or effect,
* optional weights,
* candidate refinements that could be added to the public representation,
* an optional ambiguity limit or decision rule for the claim.

.. code-block:: python

   import updatesupport as us

   audit = us.claim(
       "Income-threshold rate is stable enough to report",
       public=["AGE_BAND", "EDU_BAND", "SEX"],
       hidden=[
           "AGE_BAND",
           "EDU_BAND",
           "SEX",
           "OCC_MAJOR",
           "WKHP_BAND",
           "RAC1P",
       ],
       target="income_over_threshold",
       weight="sample_weight",
       candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
       ambiguity_limit=0.015,
       min_cell_weight=25,
       q_presets=["saturated"],
   ).audit(rows_or_frame)

   print(audit.to_markdown())

The claim audit separates:

* the pass/fail/inconclusive claim verdict,
* the observed aggregate target value,
* the hidden-composition interval under the selected stress test,
* transport ambiguity, the interval width,
* public adequacy,
* public fibers driving the ambiguity,
* a counterexample witness when the public representation is unstable,
* claim-centered refinement recommendations.

The hidden-composition interval is not a confidence interval. It is a
partial-identification or sensitivity interval conditional on the retained
support, supplied target values, public distribution, and selected admissible
hidden-mix class.

Lower-level evidence reports are still available when you need them directly:
``public_descent_report(...)`` for the primary interval, ``sensitivity_report(...)``
for stress grids, and ``public_representation_frontier(...)`` for public-bucket
design search. They are implementation depth behind the claim workflow rather
than separate starting points for most users.

Optional Extras
---------------

Install extras for heavier workflows:

.. code-block:: bash

   pip install "updatesupport[cvxpy]"      # convex transport presets
   pip install "updatesupport[causal]"     # EconML examples
   pip install "updatesupport[dowhy]"      # DoWhy handoff helpers
   pip install "updatesupport[finance]"    # finance plugin package

Colab Notebooks
---------------

The core repository includes a tutorial notebook showing how ``updatesupport``
sits downstream of DoWhy:

* `DoWhy downstream reporting audit <https://colab.research.google.com/github/nahuaque/updatesupport/blob/main/examples/notebooks/dowhy_downstream_reporting_colab.ipynb>`_
