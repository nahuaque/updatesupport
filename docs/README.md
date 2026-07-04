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

## GitHub Pages Deployment

The repository includes `.github/workflows/docs.yml` to build and deploy the
Sphinx HTML docs to GitHub Pages.

The workflow:

- runs on pushes to `main` when docs, package source, `pyproject.toml`, or
  `uv.lock` change;
- also supports manual runs with `workflow_dispatch`;
- installs the docs dependency group with `uv sync --locked --group docs`;
- builds with `uv run --group docs sphinx-build -W -b html docs docs/_build/html`;
- uploads `docs/_build/html` as a Pages artifact;
- deploys the artifact to the `github-pages` environment.

One-time manual setup in GitHub:

1. Go to the repository's **Settings > Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Confirm GitHub Pages is enabled for the repository.
4. If the `github-pages` environment requires manual approvals, either approve
   deployments when the workflow runs or relax that environment rule under
   **Settings > Environments > github-pages**.

After setup, push to `main` or run the **Docs** workflow manually from the
Actions tab. The deployed URL is shown on the workflow's deploy job summary and
in **Settings > Pages**.
