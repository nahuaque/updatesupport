# updatesupport

[![PyPI Version](https://img.shields.io/pypi/v/updatesupport)](https://pypi.org/project/updatesupport/)

**Audit whether the categories in your report are stable enough for the
estimate you are reporting.**

`updatesupport` quantifies hidden-composition ambiguity: how much an aggregate
rate, effect, or risk metric could move if the public buckets in a report stayed
fixed but the hidden mix inside those buckets changed.

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

Those buckets may hide subgroups with different target rates or effects. A
confidence interval answers sampling uncertainty. A model metric answers model
fit. `updatesupport` answers a different review question:

> Holding the reported public mix fixed, how far could the aggregate move under
> plausible hidden subgroup shifts?

The output is a review-ready Markdown audit with:

- claim-level pass/fail/inconclusive verdicts
- observed estimate
- hidden-composition stress interval
- transport ambiguity, the interval width
- public adequacy flag
- public cells driving the ambiguity
- refinement recommendations
- sensitivity checks
- public-representation frontier search for small stable reporting designs

This is not a replacement for causal identification, model validation, or
sampling uncertainty. It is an audit layer for the reporting representation.

## Why the Math Is Sound

`updatesupport` reduces the review question to a finite optimization problem.
Given hidden cells `D`, public buckets `pi(D)`, supplied hidden-cell target
values `h`, and an explicit admissible distribution class `Q`, it computes:

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
chosen `Q`. It is not a confidence interval and does not estimate causal,
survey-design, or model uncertainty.

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

report = us.public_descent_report(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    min_cell_weight=25,
    q="saturated",
    title="Income-Threshold Representation Audit",
)

print(report.to_markdown())
```

The report asks:

> If we only report by `AGE_BAND x EDU_BAND x SEX`, how much could the aggregate
> income-threshold rate move if occupation, work hours, and race composition
> changed inside those public cells?

The stress interval is not a confidence interval. It is a
partial-identification / sensitivity interval for hidden composition.

## A Concrete Result

In the Folktables ACSIncome demo, the observed target rate is `12.37%`. When the
public mix by age band, education band, and sex is held fixed, hidden subgroup
composition can move the target rate from:

```text
11.79% to 13.44%
```

The transport ambiguity is `1.65` percentage points. In plain English:

> The coarse public categories almost determine the aggregate income-threshold
> rate in this sample, but not quite. Hidden occupational and household
> differences still matter.

See the full analyst interpretation in
[docs/folktables-acs-income-interpretation.md](docs/folktables-acs-income-interpretation.md).

## Main Workflows

### 1. Verify a Reporting Claim

Use `ReportingClaim` when you want one review artifact that certifies the
claim, breaks it with a hidden-composition witness, or proposes a stable repair.

```python
claim = us.ReportingClaim(
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

verdict = us.verify_claim(rows_or_frame, claim)
print(verdict.to_markdown())
```

The verifier separates the reported estimate, statistical uncertainty,
hidden-composition ambiguity, public-refinement repair, counterexample witness,
and limitations. See [docs/reporting-claims.md](docs/reporting-claims.md).

For model-assisted plausibility checks, fit a nonparametric public/hidden joint
distribution and run the claim across sampled joint compositions:

```python
joint = us.fit_joint_distribution(
    rows_or_frame,
    public=claim.public,
    hidden=claim.hidden,
    target=claim.target,
    weight=claim.weight,
)

verdict = us.verify_claim(
    rows_or_frame,
    claim,
    joint_model=joint,
    joint_draws=500,
    joint_seed=123,
)
```

See [docs/model-assisted-joint-analysis.md](docs/model-assisted-joint-analysis.md).

### 2. Audit a Public Report

Use `public_descent_report(...)` when you already know the public buckets,
hidden refinements, and target metric.

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

Use `AuditSpec` when the audit configuration should be serialized, reviewed, or
rerun:

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

### 3. Run Robustness Checks

Use `sensitivity_report(...)` when the conclusion should be checked across Q
presets, sparse-cell thresholds, or alternate hidden-state definitions.

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

### 4. Certify a Stable Public Segmentation

Use `certify_public_representation(...)` when you want a review-ready decision:
the smallest evaluated public bucket design that keeps ambiguity below a
declared threshold, with search assumptions and limitations attached.

```python
certificate = us.certify_public_representation(
    rows_or_frame,
    base_public=["AGE_BAND", "EDU_BAND"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="income_over_threshold",
    candidate_refinements=["SEX", "OCC_MAJOR", "WKHP_BAND"],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
    min_cell_weights=[1, 10, 25],
    ambiguity_limit=0.01,
    bucket_budget=40,
    search="exhaustive",
)

print(certificate.to_markdown())
```

Use `public_representation_frontier(...)` when you want the exploratory Pareto
frontier behind the certificate. The frontier compares public-cell count, added
public columns, and ambiguity across the stress grid. See
[docs/public-representation-frontier.md](docs/public-representation-frontier.md).

### 5. Audit Causal or Uplift Reports

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
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    q=us.q_bounded_shift(0.5),
    statistical_estimate=ate_hat,
    statistical_interval=(ci_low, ci_high),
    statistical_method="causal estimator bootstrap",
)
```

This keeps four things separate: causal estimate, statistical uncertainty,
hidden-composition ambiguity, and public refinement recommendations.

See [docs/causal-library-integration.md](docs/causal-library-integration.md).

### 6. Financial Model-Risk Plugin

Financial model-risk use cases live in the separate plugin package:

```bash
pip install "updatesupport[finance]"
# or
uv add "updatesupport[finance]"
```

It adds finance-oriented metrics such as expected loss and default rate without
cluttering the core package.

See [packages/updatesupport-finance/README.md](packages/updatesupport-finance/README.md).

## Install Extras

```bash
pip install "updatesupport[cvxpy]"      # TV, chi-square, KL, Wasserstein, custom CVXPY Q
pip install "updatesupport[scip]"       # CVXPY plus PySCIPOpt for solver="SCIP" and MIP presets
pip install "updatesupport[examples]"   # Folktables, pandas, plotting examples
pip install "updatesupport[causal]"     # causal-effect examples
pip install "updatesupport[dowhy]"      # CausalRefutation conversion
pip install "updatesupport[finance]"    # financial model-risk plugin package
```

The same extras work with `uv add`:

```bash
uv add "updatesupport[cvxpy]"
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

## Examples

Run the no-download Folktables smoke demo from a source checkout:

```bash
uv run python examples/folktables_acs.py --synthetic
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

- [Representation adequacy guide](docs/representation-adequacy.md)
- [Audit specs](docs/audit-specs.md)
- [Structured exports](docs/structured-exports.md)
- [Data diagnostics](docs/data-diagnostics.md)
- [Mathematical and statistical soundness](docs/mathematical-statistical-soundness.md)
- [Public representation frontier](docs/public-representation-frontier.md)
- [Representation stability certificates](docs/representation-stability-certificates.md)
- [Transport preset guide](docs/transport-presets.md)
- [Using `updatesupport` with causal inference libraries](docs/causal-library-integration.md)
- [Benchmark gallery](docs/benchmark-gallery.md)
- [Theory and backend reference](docs/theory-and-backends.md)
- [Extension and plugin architecture](docs/extensions.md)
- [Release guide](docs/releasing.md)
- [Folktables ACSIncome result interpretation](docs/folktables-acs-income-interpretation.md)
- [Update-Relevant Support: Hume's Missing Descent](https://philpapers.org/go.pl?id=BRUUSH&proxyId=&u=https%3A%2F%2Fphilpapers.org%2Farchive%2FBRUUSH.pdf)
