# Positioning and Lineage

`updatesupport` is best understood as an ergonomic audit tool for a specific
partial-identification question:

> Given a public reporting granularity, could a headline aggregate move if the
> retained finer composition inside those public buckets changed within a
> declared admissible class?

The project is not presented as new mathematics. Its contribution is the
framing and tooling around an old and defensible object: fixed public marginals,
retained finer cells, supplied cell-level target values, and lower and upper
aggregate values over a declared stress set.

In more familiar statistical language, the package is about aggregation bias:
the risk that a coarse public table hides enough within-bucket variation to
change a reported aggregate. That places it near ecological-fallacy and
Simpson's-paradox concerns, coarsened protected-attribute audits, subgroup
composition sensitivity, and partial identification for published aggregates.

## The Actual Mathematical Object

A compiled audit has:

- public buckets, meaning the categories visible in the report;
- retained finer cells, historically called "hidden" in the API and docs;
- a fixed target value per retained cell;
- an admissible class `Q` of retained-cell distributions.

The primary interval is:

```text
lower = inf { sum_d h(d) q(d) : q in Q, pi#q = p }
upper = sup { sum_d h(d) q(d) : q in Q, pi#q = p }
```

where `p` is the fixed public law and `pi#q` is the public projection of the
retained-cell distribution. The interval width is the hidden-composition
ambiguity.

Two caveats belong at the top of any serious explanation:

- "Hidden" means not publicly reported at the chosen reporting level. It does
  not mean unobserved, latent, unknowable, or absent from the analyst's retained
  data.
- The bound is relative to the retained refinement and `Q` that the audit
  declares. It is not an absolute bound on composition risk, omitted variables,
  causal confounding, or all possible future populations.

## Lineage

The saturated case is a finite partial-identification calculation. Within each
public fiber, if any retained cell may receive the public bucket mass, the lower
endpoint places mass on the minimum target cell and the upper endpoint places
mass on the maximum target cell. Because public fibers are disjoint and the
public law is fixed, the contributions add:

```text
ambiguity = sum_o p(o) * range_{d: pi(d)=o} h(d)
```

That is not novel. It is close in spirit to ecological-inference and
Frechet-bound reasoning: aggregate or coarse information is fixed, and the
unseen-within-aggregate composition is only partially identified. For example,
Jiang, King, Schmaltz, and Tanner describe ecological inference as learning
about individual behavior from aggregate data and discuss the Duncan-Davis
deterministic-bound lineage in their partial-identification treatment of
ecological regression:
[Ecological Regression with Partial Identification](https://arxiv.org/abs/1804.05803).
Moon's recent aggregate-data paper is another nearby reference point: it
partially identifies linear combinations of conditional means from aggregate
data using optimization over admissible distributions:
[Partial Identification of Individual-Level Parameters Using Aggregate Data in
a Nonparametric Model](https://arxiv.org/abs/2403.07236).

The divergence-budget presets are also standard in distributionally robust
optimization. Total variation, chi-square, KL, Wasserstein, L2, and Mahalanobis
sets are useful because they make the admissible stress set explicit, not
because they are unique to this package.

## What Is Packaged Here

The package makes the following audit routine:

> I am publishing or reviewing a table at this granularity. I have a retained
> finer table or reference data. Is the reported aggregate stable under the
> retained refinements I am deliberately not reporting?

That packaging matters because the raw saturated calculation may be short, but
a defensible review artifact usually needs more than the endpoint values:

- compiler checks from rows to retained cells and public fibers;
- data diagnostics for sparse cells, dropped mass, singleton public fibers, and
  constant-target fibers;
- closed-form saturated intervals, witness distributions, and public-fiber
  decomposition;
- LP and CVXPY backends for bounded, divergence-budget, Wasserstein, balance,
  and concentration stress sets;
- sensitivity grids over `Q`, sparse-cell rules, and retained refinements;
- refinement rankings and public-representation frontier search;
- target guardrails for ratios, moment transforms, procedure-level targets, and
  unsupported nonlinear claims;
- Markdown, JSON, table, and dataframe outputs that can be attached to a review
  pack.

The library value is in the surrounding audit workflow: interpretation,
guardrails, refinement selection, repeatability, and reporting, not merely in
solving a small LP.

## Adjacent Software

`updatesupport` is adjacent to, but distinct from, common sensitivity and
causal-inference tooling.

[SALib](https://salib.readthedocs.io/en/latest/) and the R
[`sensitivity`](https://cran.r-project.org/package=sensitivity) package focus on
global sensitivity analysis of model outputs and input-factor importance. Those
are valuable, but they answer a different question from public-bucket
recomposition under a fixed reported marginal.

[DoubleML sensitivity analysis](https://docs.doubleml.org/stable/guide/sensitivity.html)
focuses on omitted-variable bias for causal parameters. `updatesupport` can sit
downstream of DoubleML, EconML, DoWhy, or another estimator, but it does not
identify effects or bound unobserved confounding. It audits whether the
reporting representation is stable for supplied effect estimates.

Ecological-inference tools usually aim to infer finer individual-level
quantities from aggregate observations. `updatesupport` assumes the analyst has
retained finer cells, or a justified reference/model-assisted refinement, and
asks whether suppressing those cells in the public report leaves the headline
aggregate stable.

## Best-Fit Use Cases

The strongest product wedge is not "general sensitivity analysis." It is
representation adequacy for published or reviewed aggregates.

For organic Python users, the most natural front door is the ACS/Folktables and
fairness-auditing direction. That audience is already used to notebooks,
auditing libraries, coarsened subgroup metrics, and partial-identification
language. The finance plugin is valuable as a domain showcase and proof of the
plugin architecture, but it is not the main organic adoption channel.

Good fits include:

- official statistics and disclosure-avoidance workflows, where an agency has
  richer microdata and chooses a table granularity stable enough to publish;
- fairness or disparity auditing where protected-class labels, proxies, or
  richer subgroup attributes exist in a validation dataset but the public report
  is coarser;
- causal or uplift reporting where treatment-effect estimates are produced
  upstream and the model-review question is whether the public segmentation
  supports the aggregate effect being reported;
- model-risk review, through domain plugins, where a portfolio report uses
  coarse product, region, score, or LTV buckets while validation data retain
  broker, vintage, hardship, documentation, or market refinements.

Poor fits include settings where the relevant finer variables are genuinely
unavailable and there is no defensible reference dataset, model-assisted
refinement, or scenario definition. The library cannot manufacture information
that is not in the retained support or in the declared stress test.

## Scope Boundary

The audit does not discover hidden truth, prove that a public table is correct,
or give an absolute uncertainty interval for all omitted structure.

The claim is conditional: given a retained refinement, target contract,
sparse-cell rule, and admissible class `Q`, the package solves the declared
partial-identification problem and reports which public refinements make the
answer more stable.
