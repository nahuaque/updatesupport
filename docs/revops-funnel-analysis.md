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

## Useful RevOps Claims

The same pattern applies to other funnel and go-to-market claims:

- "SQL conversion remains above the operating threshold."
- "Enterprise win rate is stable quarter over quarter."
- "Pipeline coverage is sufficient for next-quarter target."
- "Expansion conversion improved after the lifecycle campaign."
- "Inbound leads outperform outbound after controlling only for public segment."
- "Forecast-risk concentration is not driven by one hidden channel or rep-ramp
  subgroup."

The product wedge is not general dashboarding. It is a narrower audit:

> Does this reported RevOps claim remain defensible after plausible retained
> pipeline-mix recomposition, and which extra public cuts would make it
> defensible?

## Caveats

The result is conditional on the retained state space, the target definition,
the public columns, the hidden refinement columns, and the selected Q stress
test. A narrow interval does not prove the funnel model is correct; it says this
specific representation-stability stress test leaves little remaining ambiguity.
