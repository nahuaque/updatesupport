# RevOps Funnel Analysis

`updatesupport` can audit whether a RevOps funnel claim survives hidden
pipeline-mix recomposition. The common workflow is a QBR, board, or operating
review where the public report shows a coarse funnel cut such as:

```text
quarter x segment x region
```

but the retained analysis table also contains finer cells such as:

```text
lead source, campaign type, industry, deal-size band, rep ramp
```

Here, "hidden" means retained by the analyst but not included in the public
funnel report. It does not mean unobserved.

## The Analyst Question

The synthetic RevOps example asks:

> The reported MQL-to-SQL conversion rate is above the funnel-health threshold.
> Is that conclusion stable if the public segment mix stays fixed but the
> retained pipeline mix inside those public segments changes?

Run the no-download example:

```bash
uv run python examples/revops_funnel_stability.py
```

The headline result is deliberately uncomfortable:

```text
observed SQL conversion: 19.72%
hidden-composition interval: 14.14% to 25.79%
threshold: 18%
```

The observed metric looks healthy, but the hidden-composition interval crosses
the operating threshold. The claim is therefore not decision-invariant under the
declared stress test.

## What The Report Separates

- **Reported funnel estimate**: the observed MQL-to-SQL conversion rate.
- **Statistical uncertainty**: not estimated by this synthetic example; report
  it separately if your warehouse or forecast model supplies it.
- **Hidden-composition ambiguity**: how much the aggregate can move when the
  public funnel mix is fixed but retained subgroups inside those public buckets
  are recomposed.
- **Refinement recommendation**: which additional public cuts would make the
  claim more stable.

In the synthetic example, one-column screens show that reporting by lead source
or campaign type fully removes the ambiguity, but those refinements create many
public cells. Frontier search finds a more compact repair:

```text
base + deal_size_band, rep_ramp_band
```

That representation keeps the healthy-funnel decision invariant while using
fewer public cells than a full lead-source or campaign-type cut.

## Review Artifacts

The example can write Markdown, JSON, and CSV tables for a review packet:

```bash
uv run python examples/revops_funnel_stability.py \
  --export-dir data/revops_funnel_review
```

The export directory contains:

- `revops_funnel_stability.md`: human-readable review memo.
- `public_report.json`, `claim_audit.json`, `frontier.json`: structured report
  payloads.
- `tables/*.csv`: BI-friendly tables for the summary, worst fibers, witnesses,
  decision audit, refinement recommendations, and frontier candidates.

These artifacts are intended to fit ordinary RevOps review workflows: attach the
Markdown to a model or metric review, load CSVs into a spreadsheet, or ingest the
JSON into a dashboard or evidence archive.

## Bring Your Own CSV

The synthetic examples are only meant to make the wedge visible. For analyst
work, export retained funnel cells from your warehouse or CRM reporting layer and
run the CSV recipe:

```bash
uv run python examples/revops_funnel_from_csv.py \
  --input funnel_cells.csv \
  --mode level \
  --public quarter reported_segment region \
  --hidden quarter reported_segment region lead_source campaign_type industry deal_size_band rep_ramp_band \
  --target sql_conversion_rate \
  --weight mql_count \
  --threshold 0.18 \
  --output-dir data/revops_review
```

The input should be one retained cell per row. For a level claim, the required
columns are:

- the declared public columns,
- the hidden refinement columns,
- a target column such as `sql_conversion_rate`,
- a nonnegative weight column such as `mql_count`.

For a trend claim, use paired current and prior metric columns in the same
retained cell:

```bash
uv run python examples/revops_funnel_from_csv.py \
  --input paired_funnel_cells.csv \
  --mode trend \
  --public reported_segment region \
  --hidden reported_segment region lead_source campaign_type industry deal_size_band rep_ramp_band \
  --current-target FY26Q2_sql_conversion_rate \
  --prior-target FY26Q1_sql_conversion_rate \
  --current-label FY26Q2 \
  --prior-label FY26Q1 \
  --weight comparison_weight \
  --output-dir data/revops_trend_review
```

The trend recipe computes the current-minus-prior cell-level target, audits the
nonnegative-trend claim, and includes a robust current-vs-prior comparison. This
keeps the integration burden low: start from CSV or dataframe exports before
building any Salesforce, HubSpot, or warehouse connector.

## Useful RevOps Claims

The same pattern applies to other funnel and go-to-market claims:

- "SQL conversion remains above the operating threshold."
- "Enterprise win rate is stable quarter over quarter."
- "Pipeline coverage is sufficient for next-quarter target."
- "Expansion conversion improved after the lifecycle campaign."
- "Inbound leads outperform outbound after controlling only for public segment."
- "Forecast-risk concentration is not driven by one hidden channel or rep-ramp
  subgroup."

## Trend Stability

RevOps teams often care more about movement than levels:

> Did conversion actually improve this quarter?

The no-download trend example audits that question:

```bash
uv run python examples/revops_funnel_trend_stability.py
```

It uses paired retained cells for a current and prior quarter. The supplied
target is the hidden-cell trend:

```text
current-quarter SQL conversion - prior-quarter SQL conversion
```

The public report still cuts by segment and region, while retained cells include
lead source, campaign type, industry, deal-size band, and rep ramp.

The synthetic headline is:

```text
observed Q/Q SQL conversion lift: 1.21 percentage points
hidden-composition interval: -1.88 to 4.88 percentage points
```

The observed trend is positive, but the interval crosses zero. Under the
declared stress test, the improvement is not decision-invariant.

This is the RevOps version of a Simpson's-paradox warning: a KPI trend can look
positive at the reported level while retained subgroups inside the same public
segments tell a less stable story. In the synthetic example, the compact repair
is:

```text
base + rep_ramp_band
```

That is a natural operating recommendation: report or monitor the trend
separately for ramping and tenured reps before treating the headline improvement
as certified.

Write trend review artifacts:

```bash
uv run python examples/revops_funnel_trend_stability.py \
  --export-dir data/revops_funnel_trend_review
```

The trend example also includes a robust quarter comparison, so the report can
say both:

- the current quarter is the observed winner;
- the current quarter is not the certified winner under hidden-composition
  recomposition.

The product wedge is not general dashboarding. It is a narrower audit:

> Does this reported RevOps claim remain defensible after plausible retained
> pipeline-mix recomposition, and which extra public cuts would make it
> defensible?

## Caveats

The result is conditional on the retained state space, the target definition,
the public columns, the hidden refinement columns, and the selected Q stress
test. A narrow interval does not prove the funnel model is correct; it says this
specific representation-stability stress test leaves little remaining ambiguity.
