# Documentation Build

The Sphinx documentation lives in this `docs/` directory.

Install documentation dependencies:

```bash
uv sync --group docs
```

Build HTML:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

Treat warnings as errors for release checks:

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```
