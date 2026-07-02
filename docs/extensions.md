# Extension and Plugin Architecture

`updatesupport` keeps the core package domain-neutral. Domain-specific use cases
can layer on their own metrics, Q presets, compilers, and report profiles through
small extension packages.

The core package owns:

- finite problem compilation
- public/hidden support logic
- built-in Q presets
- row-metric and plugin protocols
- sensitivity, refinement, and report machinery

Extension packages own domain vocabulary. For example,
`updatesupport-finance` supplies financial model-risk metrics and report
profiles without adding finance-specific concepts to core.

## Installing Plugins

Install a plugin package directly:

```bash
# with pip
pip install updatesupport-finance

# with uv
uv add updatesupport-finance
```

Or install through a core extra:

```bash
# with pip
pip install "updatesupport[finance]"

# with uv
uv add "updatesupport[finance]"
```

## Discovering Plugins

Plugins may register Python entry points under `updatesupport.plugins`.

```python
import updatesupport as us

plugins = us.discover_plugins()
print([plugin.name for plugin in plugins])
```

Direct imports remain supported and are usually clearer in application code:

```python
import updatesupport_finance as usf

metric = usf.expected_loss(pd="pd", lgd="lgd")
```

## Plugin Descriptor

A plugin exposes an `UpdateSupportPlugin` descriptor:

```python
import updatesupport as us

plugin = us.UpdateSupportPlugin(
    name="finance",
    metrics={"expected_loss": expected_loss},
    q_presets={"portfolio_mix_shift": q_portfolio_mix_shift},
    report_profiles={"model_risk": model_risk_report},
    compilers={"portfolio": from_portfolio},
)
```

The descriptor is intentionally lightweight. It gives discovery, namespacing,
and lookup, while each plugin keeps its own user-facing API.

## Row Metrics

Plugins can provide row-level target abstractions with `RowMetric`.

```python
metric = us.row_metric(
    "expected_loss_rate",
    lambda row: row["pd"] * row["lgd"],
    columns=("pd", "lgd"),
    description="expected loss rate",
)

grouped = us.from_dataframe(
    rows,
    public=["product", "region"],
    hidden=["product", "region", "channel"],
    target=metric,
    weight="ead",
)
```

This compiles to the same `GroupedProblem` representation as a target column.
The plugin only controls how the row-level target value is computed.

## First Plugin: updatesupport-finance

`updatesupport-finance` provides:

- `expected_loss(...)`
- `expected_loss_amount(...)`
- `default_rate(...)`
- `loss_given_default(...)`
- `from_portfolio(...)`
- `model_risk_report(...)`
- `q_portfolio_mix_shift(...)`
- `q_exposure_weighted_tv(...)`

Example:

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
    candidate_refinements=["broker_channel", "employment_type", "vintage"],
    q=usf.q_portfolio_mix_shift(radius=0.25),
)

print(report.to_markdown())
```
