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
* candidate refinements that could be added to the public representation.

.. code-block:: python

   import updatesupport as us

   report = us.public_descent_report(
       rows_or_frame,
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
       min_cell_weight=25,
       q="saturated",
       title="Income-Threshold Representation Audit",
   )

   print(report.to_markdown())

The report separates:

* the observed aggregate target value,
* the hidden-composition interval under the selected stress test,
* transport ambiguity, the interval width,
* public adequacy,
* public fibers driving the ambiguity,
* refinement recommendations.

The hidden-composition interval is not a confidence interval. It is a
partial-identification or sensitivity interval conditional on the retained
support, supplied target values, public distribution, and selected admissible
hidden-mix class.

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
sits downstream of EconML:

* `EconML downstream reporting audit <https://colab.research.google.com/github/nahuaque/updatesupport/blob/main/examples/notebooks/econml_downstream_reporting_colab.ipynb>`_

