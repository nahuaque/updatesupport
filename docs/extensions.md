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
    version="0.1.0",
    description="Financial model-risk extensions for updatesupport.",
    metrics={"expected_loss": expected_loss},
    q_presets={"portfolio_mix_shift": q_portfolio_mix_shift},
    report_profiles={"model_risk": model_risk_report},
    compilers={"portfolio": from_portfolio},
    metadata=us.PluginMetadata(
        package="updatesupport-finance",
        homepage="https://github.com/nahuaque/updatesupport",
        domain="financial-model-risk",
        tags=("credit-risk", "expected-loss", "model-validation"),
        min_updatesupport_version="0.1.1",
    ),
)
```

The descriptor is intentionally lightweight. It gives discovery, namespacing,
and lookup, while each plugin keeps its own user-facing API.

## Plugin SDK Checks

Core validates plugin descriptors before registration. A valid plugin has a
non-empty slug-like name, optional string metadata, and callable values in each
surface map.

```python
report = us.validate_plugin(plugin)
report.raise_for_errors()

print(plugin.as_dict())
```

Use this in plugin tests so packaging errors fail before release:

```python
def test_plugin_contract():
    assert us.validate_plugin(plugin).ok
```

`register_plugin(...)` protects duplicate plugin names by default:

```python
us.register_plugin(plugin)

# Only use replace=True intentionally, for example in an interactive notebook
# after reloading local plugin code.
us.register_plugin(plugin, replace=True)
```

Entry-point discovery uses the same validation and duplicate-name checks. A
package that exposes a plugin entry point should keep the entry point name and
the descriptor name aligned:

```toml
[project.entry-points."updatesupport.plugins"]
finance = "updatesupport_finance:plugin"
```

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
- `finance_sensitivity_grid(...)`
- `from_portfolio(...)`
- `model_risk_report(...)`
- `certify_portfolio_segmentation(...)`
- `q_portfolio_mix_shift(...)`
- `q_exposure_weighted_tv(...)`
- `q_factor_exposure_shift(...)`
- `q_regional_concentration_shift(...)`

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
