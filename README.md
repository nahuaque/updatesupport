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

### 1. Audit a Public Report

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

### 2. Run Robustness Checks

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

### 3. Choose a Stable Public Segmentation

Use `public_representation_frontier(...)` when you want the smallest public
bucket design that keeps ambiguity below a review threshold.

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["AGE_BAND", "EDU_BAND"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="income_over_threshold",
    candidate_refinements=["SEX", "OCC_MAJOR", "WKHP_BAND"],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
    min_cell_weights=[1, 10, 25],
    ambiguity_limit=0.01,
    bucket_budget=40,
    search="beam",
)

explanation = frontier.explain_minimal_stable()
if explanation is not None:
    print(explanation.to_markdown())
```

The frontier compares public-cell count, added public columns, and ambiguity
across the stress grid. See
[docs/public-representation-frontier.md](docs/public-representation-frontier.md).

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

### 5. Financial Model-Risk Plugin

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
pip install "updatesupport[examples]"   # Folktables, pandas, plotting examples
pip install "updatesupport[causal]"     # causal-effect examples
pip install "updatesupport[dowhy]"      # CausalRefutation conversion
pip install "updatesupport[finance]"    # financial model-risk plugin package
```

The same extras work with `uv add`:

```bash
uv add "updatesupport[cvxpy]"
uv add "updatesupport[examples]"
uv add "updatesupport[causal]"
uv add "updatesupport[dowhy]"
uv add "updatesupport[finance]"
```

The causal docs cover [EconML](https://www.pywhy.org/EconML/),
[DoWhy](https://www.pywhy.org/dowhy/),
[DoubleML](https://docs.doubleml.org/), and
[CausalML](https://causalml.readthedocs.io/en/latest/) handoffs.

For all examples and optional integrations:

```bash
pip install "updatesupport[examples,causal,dowhy,cvxpy,finance]"
# or
uv add "updatesupport[examples,causal,dowhy,cvxpy,finance]"
```

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
- [Public representation frontier](docs/public-representation-frontier.md)
- [Transport preset guide](docs/transport-presets.md)
- [Using `updatesupport` with causal inference libraries](docs/causal-library-integration.md)
- [Benchmark gallery](docs/benchmark-gallery.md)
- [Theory and backend reference](docs/theory-and-backends.md)
- [Extension and plugin architecture](docs/extensions.md)
- [Release guide](docs/releasing.md)
- [Folktables ACSIncome result interpretation](docs/folktables-acs-income-interpretation.md)
- [Update-Relevant Support: Hume's Missing Descent](https://philpapers.org/go.pl?id=BRUUSH&proxyId=&u=https%3A%2F%2Fphilpapers.org%2Farchive%2FBRUUSH.pdf)
