# updatesupport-finance

Financial model-risk extensions for
[`updatesupport`](https://pypi.org/project/updatesupport/).

`updatesupport-finance` audits whether a public risk segmentation is stable
enough to support a reported portfolio metric.

The core question is:

> If a model report only shows risk by coarse public buckets such as
> `product x region x FICO band x LTV band`, could the reported expected-loss
> estimate materially change if the hidden mix inside those buckets shifted?

This is a segmentation adequacy check for reported risk metrics. It is designed
for model-review and portfolio-monitoring artifacts, not as a replacement for
model validation, calibration, backtesting, or statistical uncertainty analysis.

Install directly:

```bash
pip install updatesupport-finance
uv add updatesupport-finance
```

Or through the core package extra:

```bash
pip install "updatesupport[finance]"
uv add "updatesupport[finance]"
```

The package provides finance-oriented row metrics, Q preset aliases, portfolio
compilation, and a model-risk report profile while keeping financial vocabulary
out of the core `updatesupport` package.

Conic concentration presets require the core CVXPY extra when solved:

```bash
pip install "updatesupport[cvxpy]" updatesupport-finance
uv add "updatesupport[cvxpy]" updatesupport-finance
```

## Why This Is Useful

Financial analysts already monitor model performance, population drift,
calibration, overrides, and scenario sensitivity. Those checks usually ask
whether the model or portfolio changed.

`updatesupport-finance` asks a different question:

> Is the reporting segmentation itself adequate for the metric being reported?

For example, a validation pack may report expected loss by:

- `product`
- `region`
- `fico_band`
- `ltv_band`

But inside those public buckets, hidden composition may vary by:

- broker channel
- employment type
- vintage
- hardship history
- documentation type
- local housing market
- borrower cashflow pattern

If those hidden subgroups have different expected-loss rates, the public
segmentation may not fully support the reported aggregate. The report quantifies
that hidden-composition ambiguity and identifies which hidden variables would
most improve the public segmentation.

## What The Report Separates

The package is intentionally narrow. It separates:

- reported risk estimate: the supplied metric, such as expected loss or default
  rate
- statistical uncertainty: confidence intervals or model uncertainty supplied by
  other workflows
- hidden-composition ambiguity: how far the reported metric can move when hidden
  mix shifts inside fixed public buckets
- refinement recommendations: hidden fields that would make the public
  representation more stable

This is not a confidence interval and not a full model-risk-management system.
It is a reviewable control for one practical question: whether the reporting
representation is stable enough for the risk metric.

## Analyst Workflow

1. Choose public buckets from the model report.
2. Choose hidden refinements that are available internally but not shown in the
   public segmentation.
3. Choose the target risk metric.
4. Choose a plausible hidden-mix shift preset.
5. Set a review threshold for hidden-composition ambiguity.
6. Attach the generated Markdown report to a model-review or monitoring pack.

The review status is deliberately simple:

- `pass`: ambiguity and public adequacy checks are within the chosen thresholds
- `attention required`: the public segmentation may need refinement or explicit
  acceptance of the ambiguity band

## Example

```python
import updatesupport_finance as usf

report = usf.model_risk_report(
    portfolio,
    public=["product", "region", "fico_band", "ltv_band"],
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
        "vintage",
    ],
    metric=usf.expected_loss(pd="pd", lgd="lgd"),
    exposure="ead",
    q=usf.q_portfolio_mix_shift(radius=0.25),
    model_id="EL_RETAIL_2026Q2",
    portfolio_name="Retail credit portfolio",
    as_of_date="2026-06-30",
    intended_use="Expected-loss segmentation model review",
    ambiguity_limit=0.0025,
    public_adequacy_required=False,
)

print(report.to_markdown())
```

The report answers:

- What is the reported portfolio risk estimate?
- What range is still possible under hidden mix shifts?
- Does the ambiguity exceed the review threshold?
- Which public buckets drive the instability?
- Which hidden fields are most valuable as public refinements?
- Which small public segmentation sits on the stability frontier, and why did
  it beat nearby alternatives?

A synthetic portfolio example is available in `examples/model_risk_portfolio.py`
in the source repository:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/model_risk_portfolio.py
```

The example prints both the finance model-risk report and a core
`public_representation_frontier(...)` report for the same expected-loss metric.
The frontier section compares baseline versus selected ambiguity, close
dominated alternatives, and any screened-out refinement fields.

## Portfolio Concentration Stress Presets

Use concentration presets when independent hidden-bucket movement is too
coarse for a model-risk review. These helpers constrain portfolio-level exposure
drift while preserving the observed public segmentation.

Factor exposure drift:

```python
q = usf.q_factor_exposure_shift(
    0.20,
    portfolio,
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
    ],
    factors={
        "macro_beta": "macro_beta",
        "rate_sensitivity": "rate_sensitivity",
        "house_price_beta": "house_price_beta",
    },
    exposure="ead",
)
```

Regional concentration drift:

```python
q = usf.q_regional_concentration_shift(
    0.10,
    portfolio,
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
    ],
    region="region",
    exposure="ead",
)
```

Both helpers compile exposure-weighted hidden-cell moments and route through the
core `q_covariate_balance(...)` preset:

```text
|| standardized_factor_or_concentration_shift ||_2 <= radius
```

In model-review language, this asks:

> If the public risk buckets stay fixed, but hidden portfolio factor exposure or
> regional concentration can drift within this L2 tolerance, how much can the
> reported risk metric move?

This maps naturally to expected loss, default rate, LGD, delinquency, approval
benefit, and capital review where shifts are governed by portfolio exposure
profiles rather than arbitrary independent hidden-cell movement.

To write the Markdown report:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/model_risk_portfolio.py \
  --output data/finance_model_risk_report.md
```
