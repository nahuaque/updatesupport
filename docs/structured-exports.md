# Structured Exports

All major report objects can emit structured payloads in addition to Markdown.

```python
report = us.public_descent_report(...)

payload = report.as_dict()
json_text = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()
```

`to_tables()` is dependency-free. It returns a dictionary of named tables, where
each table is a tuple of row dictionaries:

```python
tables["summary"]
tables["worst_fibers"]
tables["refinements"]
```

`to_dataframes()` converts those same tables to pandas DataFrames. Pandas is an
optional dependency; install it directly or use the examples extra:

```bash
pip install "updatesupport[examples]"
# or
uv add "updatesupport[examples]"
```

The same helpers are also available as functions:

```python
us.report_to_json(report)
us.report_tables(report)
us.report_dataframes(report)
```

## AuditRun Exports

When a report is produced by `AuditSpec`, the executed `AuditRun` keeps the spec
and report together:

```python
run = spec.run(rows_or_frame)

run.to_json()
run.to_tables()
run.to_dataframes()
```

The table export includes the serialized spec plus prefixed report tables:

```python
tables["spec"]
tables["report_summary"]
tables["report_refinements"]
```

## Common Table Names

Public-descent reports expose:

- `summary`
- `worst_fibers`
- `refinements`
- `data_diagnostics`
- `dual_diagnostics`

Sensitivity reports expose:

- `summary`
- `scenarios`

Refinement-sensitivity reports expose:

- `summary`
- `refinement_candidates`
- `refinement_scenarios`
- `refinement_rows`

Public-representation frontier reports expose:

- `summary`
- `search_trace`
- `screened_refinements`
- `frontier`
- `dominated`
- `candidates`
- `candidate_scenarios`

Causal reporting suites prefix the component tables, for example:

- `primary_summary`
- `primary_refinements`
- `sensitivity_scenarios`
- `refinement_sensitivity_refinement_candidates`

## JSON Payloads

`to_json()` serializes the report's structured dictionary payload, converting
tuples and other sequence-like values into JSON-compatible arrays. Use this for
review artifacts, model cards, CI outputs, and systems that need stable report
metadata without parsing Markdown.
