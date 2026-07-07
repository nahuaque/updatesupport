# Benchmark Gallery

The benchmark gallery turns the examples into reproducible, saved Markdown
reports. It writes generated artifacts to `data/benchmark_gallery/`, which is
gitignored along with the raw benchmark data.

Run the full local gallery:

```bash
uv run --extra examples --extra causal python examples/benchmark_gallery.py
```

Run only the no-download reports plus the ACIC oracle report:

```bash
uv run --extra examples python examples/benchmark_gallery.py --skip-acic-econml
```

Fetch Folktables ACS data when it is not already cached:

```bash
uv run --extra examples --extra causal python examples/benchmark_gallery.py \
  --folktables-download
```

The generated `data/benchmark_gallery/index.md` links to each saved case study.

## Included Case Studies

- **AI / ML evaluation stability synthetic audit**: a no-download
  model-comparison benchmark example. It reports a positive challenger-minus-
  baseline margin, then shows that the ranking is not invariant when hidden
  task composition shifts inside fixed public task buckets.
- **Product experimentation stability synthetic audit**: a no-download A/B-test
  example. It reports a positive treatment lift, then shows that the launch
  decision is not invariant when acquisition, tenure, geography, plan, and
  device mix can shift inside fixed public experiment buckets.
- **RevOps funnel stability synthetic audit**: a no-download revenue-operations
  example. It reports a healthy MQL-to-SQL conversion rate, then shows that the
  health decision is not invariant when lead source, campaign type, industry,
  deal size, and rep ramp can shift inside fixed public segment buckets. The
  report includes a compact public-representation repair for the claim.
- **RevOps funnel trend stability synthetic audit**: a no-download
  revenue-operations trend example. It reports positive quarter-over-quarter
  MQL-to-SQL lift, then shows that the improvement is not certified when hidden
  pipeline composition can shift inside fixed public segment buckets.
- **Folktables ACSIncome synthetic label-rate audit**: a no-download
  ACS-shaped label-rate example for testing the public/hidden reporting
  workflow. The report includes a public-representation frontier that asks
  which small refinement set stabilizes the label-rate estimate.
- **Folktables ACSIncome real sample label-rate audit**: a sampled real
  Folktables ACS report. By default this uses cached ACS data if available; use
  `--folktables-download` to fetch it. The gallery uses a lower retained-cell
  threshold for this sampled report so the public/hidden structure is visible,
  and includes the same frontier-search section.
- **Folktables ACS synthetic causal-effect audit**: a no-download causal
  handoff that computes row-level effects, then audits whether the coarse
  public representation is stable for reporting the effect.
- **ACIC 2016 oracle SATT-style audit**: a real ACIC CSV report using the
  simulated potential-outcome contrast as the oracle effect target.
- **ACIC 2016 [EconML](https://www.pywhy.org/EconML/) estimated-effect audit**:
  a real ACIC CSV report that fits an [EconML](https://www.pywhy.org/EconML/)
  first stage, computes
  `tau_hat = estimator.effect(X)`, and audits the estimated-effect target.

## ACIC Data

The gallery expects the real ACIC CSV at:

```text
data/acic_2016_p1_s1.csv
```

That file is intentionally not committed. It can be exported from the official
`vdorie/aciccomp` 2016 R package, then reused by both the standalone ACIC
example and this gallery.

If the CSV is absent, the generator still writes the Folktables synthetic
reports and records the ACIC cases as skipped in the index.

## Why ACIC Is Useful Here

ACIC is a strong benchmark for the product wedge because it exposes both:

- **oracle effects** from simulated potential outcomes, which let us ask
  whether the public reporting representation is stable for the true benchmark
  effect target
- **estimated effects** from a causal first stage, which let us ask the same
  question about the analyst's `tau_hat` reporting target

This makes the causal reporting distinction explicit:

- causal estimate: what the first-stage estimator reports
- statistical uncertainty: sampling/model uncertainty around that estimate
- hidden-composition ambiguity: how much the reported aggregate can move under
  admissible hidden reweighting
- public refinement recommendations: which additional public columns reduce
  that hidden-composition ambiguity

The oracle and estimated-effect ACIC reports should be read side by side. If
both have similar ambiguity drivers, the representation problem is not just an
artifact of the first-stage estimator. If they diverge, the estimator and the
reporting representation may be interacting in a way worth diagnosing.

## Reproducibility Notes

The default ACIC estimated-effect report samples 1,000 rows before fitting
[EconML](https://www.pywhy.org/EconML/) so that the gallery remains practical
for local iteration:

```bash
uv run --extra examples --extra causal python examples/benchmark_gallery.py \
  --acic-econml-sample 1000
```

Use the full CSV for a slower benchmark run:

```bash
uv run --extra examples --extra causal python examples/benchmark_gallery.py \
  --acic-econml-sample 0
```

All generated reports are local artifacts. Regenerate them after changing the
compiler, transport presets, report language, or causal handoff code.
