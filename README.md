# updatesupport

[![PyPI Version](https://img.shields.io/pypi/v/updatesupport)](https://pypi.org/project/updatesupport/)

**Check whether a reported aggregate is stable to subgroup recomposition.**

Reports often publish a number by coarse categories. `updatesupport` asks:

> If those public category counts stayed fixed, could the aggregate still move
> because the finer subgroup mix inside them changed?

This is a practical audit for aggregation bias, Simpson's paradox risk,
ecological-fallacy risk, subgroup composition sensitivity, and coarsened
fairness or disparity reports. It quantifies hidden-composition ambiguity: how
much an aggregate rate, effect, or risk metric could move under a declared
recomposition stress test.

**Thirty-second example.** In the Folktables ACSIncome demo, a report grouped by
age band, education band, and sex observes an income-threshold rate of `12.37%`.
Holding those public group proportions fixed, retained subgroup composition can
move the compatible rate to:

```text
11.79% to 13.44%
```

That `1.65` percentage-point width is not sampling error. It is the amount of
aggregation ambiguity left by the public categories. The best one-column
refinement in that example is occupation major group.

Here, "hidden" means **retained but not publicly reported**, not unobserved or
unknowable. The ambiguity interval is always relative to the retained
refinement you choose and the admissible shift class `Q` you declare. It is not
an absolute bound on every possible omitted variable or population shift.

It is useful when a dashboard, model-review pack, policy table, or causal
analysis reports a number by coarse public categories and you need to know:

> Are these categories enough to support this reported estimate?

## Why It Exists

Many reports publish estimates by coarse buckets:

```text
age band x education band x sex
product x region x FICO band x LTV band
segment x channel
```

Those buckets may contain retained subgroups with different target rates or
effects. A confidence interval answers sampling uncertainty. A model metric
answers model fit. `updatesupport` answers a different review question:

> Holding the reported public mix fixed, how far could the aggregate move under
> plausible hidden subgroup shifts?

The output is a review-ready Markdown audit with:

- claim-level pass/fail/inconclusive verdicts
- observed estimate
- hidden-composition stress interval
- transport ambiguity, the interval width
- estimator-uncertainty-aware interval when hidden-cell standard errors are
  supplied
- public adequacy flag
- public cells driving the ambiguity
- refinement recommendations
- sensitivity checks
- public-representation frontier search for small stable reporting designs

This is not a replacement for causal identification, model validation, or
sampling uncertainty. It is an audit layer for the reporting representation.

The project does not claim new mathematics. The core calculation is a
partial-identification / sensitivity-analysis problem with roots in ecological
inference, Frechet-style bounds, and distributionally robust optimization. The
value is the packaged workflow: compile a retained fine table, solve the stress
test, explain the ambiguity, rank refinements, run sensitivity checks, and
emit review-ready artifacts. See
[docs/positioning-and-lineage.md](docs/positioning-and-lineage.md).

## Why the Math Is Sound

`updatesupport` reduces the review question to a finite optimization problem.
Given retained fine cells `D`, public buckets `pi(D)`, supplied retained-cell
target values `h`, and an explicit admissible distribution class `Q`, it
computes:

```text
inf/sup  sum_d h(d) q(d)
subject to q in Q and fixed public law pi#q
```

For saturated reweighting this has a closed-form public-fiber range formula.
For linear `Q` it is solved as a linear program. For convex transport budgets
such as TV, chi-square, KL, and Wasserstein, it is solved as a convex CVXPY
problem with a linear objective.

The interval is statistically meaningful as a partial-identification or
sensitivity interval conditional on the retained support, target values, and
chosen `Q`. It is not a confidence interval. If retained-cell target standard
errors are supplied, reports can add an estimator-uncertainty-aware outer
interval, but causal, survey-design, and broader model uncertainty still belong
to the upstream statistical workflow. With CVXPY-compatible Q sets, reports can
also include an SOCP confidence-core diagnostic showing whether all admissible
composition-specific confidence bands have a common overlap.

The core solver target is fixed after compilation. Most tabular reports compile
to the linear plug-in aggregate `sum_d h(d) q(d)`; `RatioTarget` covers
supported fixed ratio targets; `MomentTransformTarget` covers affine moment
transforms, one-sided convex/concave CVXPY endpoints, and conservative monotone
moment bounds; and `ProcedureTarget` handles representation-dependent reporting
procedures by recompiling the target for each representation before solving.
Other nonlinear targets that depend directly on `q` still need an explicit
reformulation or a dedicated target-functional backend.

See [docs/mathematical-statistical-soundness.md](docs/mathematical-statistical-soundness.md)
for the full assumptions, formulas, backend checks, and limitations.

## Quickstart

```bash
pip install updatesupport
```

or, in a `uv` project:

```bash
uv add updatesupport
```

```python
import updatesupport as us

audit = us.claim(
    "Income-threshold rate is stable enough to report",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
    min_cell_weight=25,
    q_presets=["saturated"],
).audit(rows_or_frame)

print(audit.to_markdown())
```

The claim audit asks:

> If we only report by `AGE_BAND x EDU_BAND x SEX`, how much could the aggregate
> income-threshold rate move if occupation, work hours, and race composition
> changed inside those public cells?

The stress interval is not a confidence interval. It is a
partial-identification / sensitivity interval for hidden composition. The audit
returns one review artifact with the verdict, interval, witness, limitations,
and claim-centered refinement recommendations.

## Front-Door Demo

Start with the Folktables ACSIncome case if you want the shortest path to the
idea. It is close to fairness and disparity-audit workflows because it asks
whether a coarse demographic public report survives retained subgroup
composition changes.

In that demo, the observed target rate is `12.37%`. When the public mix by age
band, education band, and sex is held fixed, hidden subgroup composition can
move the target rate from:

```text
11.79% to 13.44%
```

The transport ambiguity is `1.65` percentage points. In plain English:

> The coarse public categories almost determine the aggregate income-threshold
> rate in this sample, but not quite. Hidden occupational and household
> differences still matter.

See the full analyst interpretation in
[docs/folktables-acs-income-interpretation.md](docs/folktables-acs-income-interpretation.md).

The finance plugin is useful as a domain showcase and model-risk vocabulary
layer, but the ACS/Folktables workflow is the best first demo for most Python
users interested in aggregation bias, ecological fallacy, Simpson's paradox, or
coarsened subgroup reporting.

## Main Workflows

### 1. Validate a Claim

Use `us.claim(...)` when you want one review artifact that certifies the claim,
breaks it with a hidden-composition witness, or proposes a stable repair.
`ClaimSpec` is the underlying dataclass when you want to instantiate or
serialize the spec directly, and `ClaimAudit` is the report object returned by
`.audit(...)`.

```python
claim = us.claim(
    estimate_name="Income-threshold target rate",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
    bucket_budget=40,
    statistical_interval=(0.119, 0.128),
)

verdict = claim.audit(rows_or_frame)
print(verdict.to_markdown())
```

The auditor separates the reported estimate, statistical uncertainty,
hidden-composition ambiguity, public-refinement repair, counterexample witness,
and limitations. `verdict.recommend_refinements()` returns claim-centered
refinement rows: whether a candidate actually repairs the claim, whether it
satisfies the ambiguity limit, and how much ambiguity it removes. See
[docs/reporting-claims.md](docs/reporting-claims.md).

For model-assisted plausibility checks, fit a nonparametric public/hidden joint
distribution and run the claim across sampled hidden compositions:

```python
joint = us.fit_joint_distribution(
    rows_or_frame,
    public=claim.public,
    hidden=claim.hidden,
    target=claim.target,
    weight=claim.weight,
)

verdict = us.audit_claim(
    rows_or_frame,
    claim,
    joint_model=joint,
    joint_draws=500,
    joint_seed=123,
)
```

Use `hidden_composition_uncertainty(...)` when you want the posterior/bootstrap
draw summary as its own report. By default it preserves the observed public
bucket mix and resamples hidden composition inside each public bucket.

See [docs/model-assisted-joint-analysis.md](docs/model-assisted-joint-analysis.md).

### 2. Inspect The Evidence Behind A Claim

Most users should start with `us.claim(...)`. Use `public_descent_report(...)`
when you explicitly want only the lower-level partial-ID evidence without a
claim verdict, decision rule, repair search, or claim-centered recommendation
language.

```python
report = us.public_descent_report(
    rows_or_frame,
    public=["segment"],
    hidden=["segment", "region", "tenure_band"],
    target="outcome_rate",
    candidate_refinements=["region", "tenure_band"],
    q=us.q_bounded_shift(0.5),
)
```

Use `AuditSpec` for serializable lower-level audit configurations that need to
be reviewed or rerun:

```python
spec = us.AuditSpec(
    public=["segment"],
    hidden=["segment", "region", "tenure_band"],
    target="outcome_rate",
    candidate_refinements=["region", "tenure_band"],
    q={"name": "bounded_shift", "radius": 0.5},
)

run = spec.run(rows_or_frame)
print(run.to_markdown())
```

See [docs/audit-specs.md](docs/audit-specs.md).

Reports also expose structured JSON and tables:

```python
run.to_json()
run.to_tables()
run.to_dataframes()  # requires pandas
```

See [docs/structured-exports.md](docs/structured-exports.md).

Reports include pre-solve data diagnostics for sparse cells, dropped mass,
singleton public fibers, constant-target fibers, and skipped refinement
candidates. See [docs/data-diagnostics.md](docs/data-diagnostics.md).

When you need to inspect the actual lower-vs-upper endpoint worlds behind an
ambiguity result, use `report.witness_report()` or `us.witness_report(...)` to
see which hidden cells move while the public distribution stays fixed.

### 3. Add Robustness And Repair Intelligence

Claim audits already embed the primary report, optional certificate search, and
claim-centered refinement recommendations. Use the lower-level robustness tools
when you need to inspect those pieces separately:

- `sensitivity_report(...)` checks Q presets, sparse-cell thresholds, or
  alternate hidden-state definitions.
- `certify_public_representation(...)` finds the smallest evaluated public
  representation satisfying an ambiguity limit.
- `public_representation_frontier(...)` exposes the Pareto frontier behind that
  certificate.
- `breakdown_point(...)` finds the stress radius where a decision or ambiguity
  claim stops passing.

```python
sensitivity = us.sensitivity_report(
    rows_or_frame,
    public=["AGE_BAND", "SEX"],
    hidden=["AGE_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="income_over_threshold",
    min_cell_weights=[1, 10, 25],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
)
```

See [docs/transport-presets.md](docs/transport-presets.md) for guidance on Q
presets and [docs/representation-adequacy.md](docs/representation-adequacy.md)
for interpretation rules.

### 4. Audit Causal or Uplift Reports

Use a causal inference library such as [EconML](https://www.pywhy.org/EconML/),
[DoWhy](https://www.pywhy.org/dowhy/),
[DoubleML](https://docs.doubleml.org/), or
[CausalML](https://causalml.readthedocs.io/en/latest/) to estimate effects
first. Then pass row-level or subgroup-level effects into `updatesupport`.

```python
suite = us.causal_reporting_stability(
    df,
    public=["AGE_BAND", "SEX"],
    hidden=["AGE_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    effect="tau_hat",
    effect_standard_error="tau_se",
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    q=us.q_bounded_shift(0.5),
    statistical_estimate=ate_hat,
    statistical_interval=(ci_low, ci_high),
    statistical_method="causal estimator bootstrap",
)
```

This keeps causal estimate, statistical uncertainty, hidden-composition
ambiguity, estimator-uncertainty-aware adjusted ambiguity, and public refinement
recommendations separate.

Try the tutorial notebook:
[DoWhy downstream reporting audit](https://colab.research.google.com/github/nahuaque/updatesupport/blob/main/examples/notebooks/dowhy_downstream_reporting_colab.ipynb).

For causal/model-review stress tests based on hidden balance drift, use
`q=us.q_covariate_balance(epsilon, moments)` to bound
`||standardized_hidden_moment_shift||_2`.

See [docs/causal-library-integration.md](docs/causal-library-integration.md).

### 5. Financial Model-Risk Plugin

Financial model-risk use cases live in the separate plugin package:

```bash
pip install "updatesupport[finance]"
# or
uv add "updatesupport[finance]"
```

It adds finance-oriented metrics such as expected loss and default rate, plus
generic disclosure-triangulation helpers over the core named-linear feasibility
solver, without cluttering the core package.

See [packages/updatesupport-finance/README.md](packages/updatesupport-finance/README.md).

## Install Extras

```bash
pip install "updatesupport[cvxpy]"      # TV, chi-square, KL, Wasserstein, custom CVXPY Q
pip install "updatesupport[residopt]"   # experimental residopt endpoint screening
pip install "updatesupport[scip]"       # CVXPY plus PySCIPOpt for solver="SCIP" and MIP presets
pip install "updatesupport[examples]"   # Folktables, pandas, plotting examples
pip install "updatesupport[causal]"     # causal-effect examples
pip install "updatesupport[dowhy]"      # CausalRefutation conversion
pip install "updatesupport[finance]"    # financial model-risk plugin package
```

The same extras work with `uv add`:

```bash
uv add "updatesupport[cvxpy]"
uv add "updatesupport[residopt]"
uv add "updatesupport[scip]"
uv add "updatesupport[examples]"
uv add "updatesupport[causal]"
uv add "updatesupport[dowhy]"
uv add "updatesupport[finance]"
```

The causal docs cover [EconML](https://www.pywhy.org/EconML/),
[DoWhy](https://www.pywhy.org/dowhy/),
[DoubleML](https://docs.doubleml.org/), and
[CausalML](https://causalml.readthedocs.io/en/latest/) handoffs.

For common examples and optional integrations:

```bash
pip install "updatesupport[examples,causal,dowhy,cvxpy,finance]"
# or
uv add "updatesupport[examples,causal,dowhy,cvxpy,finance]"
```

`updatesupport[scip]` is intentionally separate because solver wheels and
platform requirements are heavier than the default CVXPY workflow. It enables
SCIP-backed mixed-integer presets such as `q_fiber_support_floor(...)`.

`updatesupport[residopt]` is also separate from the causal extras in this first
release because `residopt` and EconML currently require incompatible
`scikit-learn` ranges. The published `residopt` package currently requires
Python 3.13 or newer.

## Examples

Run the no-download Folktables smoke demo from a source checkout:

```bash
uv run python examples/folktables_acs.py --synthetic
```

Run the no-download AI / ML evaluation stability demo:

```bash
uv run python examples/ml_eval_stability.py
```

Run the no-download product experimentation / A/B testing stability demo:

```bash
uv run python examples/product_experiment_stability.py
```

Run the no-download RevOps funnel stability demo:

```bash
uv run python examples/revops_funnel_stability.py
```

Run the real Folktables ACSIncome example:

```bash
uv run --extra examples python examples/folktables_acs.py \
  --task income \
  --states CA \
  --year 2018 \
  --sample 50000 \
  --min-cell-weight 25
```

Generate the benchmark gallery:

```bash
uv run --extra examples --extra causal python examples/benchmark_gallery.py
```

Run the finance plugin example:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/model_risk_portfolio.py
```

See [docs/benchmark-gallery.md](docs/benchmark-gallery.md) for reproducible
case studies.

## Source Development

```bash
git clone https://github.com/nahuaque/updatesupport.git
cd updatesupport
uv sync --all-packages --group dev --extra cvxpy --extra examples
uv run pytest
```

## Documentation

### Start Here

- [Sphinx documentation entry point](https://nahuaque.github.io/updatesupport/)
- [Quickstart](docs/quickstart.rst)
- [API surface guide](docs/api-surface.md)
- [Framework overview](docs/framework.rst)

### Positioning And Foundations

- [Representation adequacy guide](docs/representation-adequacy.md)
- [Positioning and lineage](docs/positioning-and-lineage.md)
- [Mathematical and statistical soundness](docs/mathematical-statistical-soundness.md)
- [Theory and backend reference](docs/theory-and-backends.md)
- [Transport preset guide](docs/transport-presets.md)
- [Update-Relevant Support: Hume's Missing Descent](https://philpapers.org/go.pl?id=BRUUSH&proxyId=&u=https%3A%2F%2Fphilpapers.org%2Farchive%2FBRUUSH.pdf)

### Claim Audits And Reports

- [Reporting claims](docs/reporting-claims.md)
- [Audit specs](docs/audit-specs.md)
- [Data diagnostics](docs/data-diagnostics.md)
- [Structured exports](docs/structured-exports.md)
- [Representation stability certificates](docs/representation-stability-certificates.md)

### Refinement And Robustness

- [Public representation frontier](docs/public-representation-frontier.md)
- [Model-assisted joint analysis](docs/model-assisted-joint-analysis.md)
- [Breakdown point analysis](docs/breakdown-point-analysis.md)
- [Robust comparison and ranking](docs/robust-comparison-ranking.md)
- [Interaction-aware refinements](docs/interaction-aware-refinements.md)

### Integrations And Case Studies

- [Using `updatesupport` with causal inference libraries](docs/causal-library-integration.md)
- [Benchmark gallery](docs/benchmark-gallery.md)
- [Folktables ACSIncome result interpretation](docs/folktables-acs-income-interpretation.md)

### Project

- [Extension and plugin architecture](docs/extensions.md)
- [Local documentation build notes](docs/README.md)
- [Release guide](docs/releasing.md)
