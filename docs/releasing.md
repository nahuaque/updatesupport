# Releasing

This repository is a monorepo with more than one publishable Python package.
Use package-specific tags so releases are unambiguous.

## Tag Convention

Use:

```text
updatesupport-vX.Y.Z
updatesupport-finance-vX.Y.Z
```

Do not use bare tags such as `v0.1.1` for new releases. Bare version tags are
ambiguous once core and plugin packages can release independently.

Examples:

```text
updatesupport-v0.1.1
updatesupport-finance-v0.1.0
```

## Release Order

If a plugin depends on a new core version, publish core first.

For the first finance plugin release:

1. Publish `updatesupport==0.1.1`.
2. Publish `updatesupport-finance==0.1.0`.

That ordering matters because `updatesupport-finance==0.1.0` depends on
`updatesupport>=0.1.1`, and the core `finance` extra points to
`updatesupport-finance>=0.1.0`.

## Core Package

Check that `pyproject.toml` has the intended core version, then tag and create a
GitHub release:

```bash
git tag -a updatesupport-v0.1.1 -m "updatesupport 0.1.1"
git push origin updatesupport-v0.1.1

gh release create updatesupport-v0.1.1 \
  --verify-tag \
  --title "updatesupport 0.1.1" \
  --generate-notes
```

Publish to TestPyPI first:

```bash
gh workflow run publish.yml -f repository=testpypi
gh run watch
```

Then publish to PyPI:

```bash
gh workflow run publish.yml -f repository=pypi
gh run watch
```

## Finance Plugin

Check that `packages/updatesupport-finance/pyproject.toml` has the intended
plugin version, then tag and create a GitHub release:

```bash
git tag -a updatesupport-finance-v0.1.0 -m "updatesupport-finance 0.1.0"
git push origin updatesupport-finance-v0.1.0

gh release create updatesupport-finance-v0.1.0 \
  --verify-tag \
  --title "updatesupport-finance 0.1.0" \
  --generate-notes
```

Publish to TestPyPI first:

```bash
gh workflow run publish-finance.yml -f repository=testpypi
gh run watch
```

Then publish to PyPI:

```bash
gh workflow run publish-finance.yml -f repository=pypi
gh run watch
```

## Independent Plugin Releases

If only `updatesupport-finance` changes, release only the finance plugin:

```text
updatesupport-finance-v0.1.1
```

No core tag is needed unless core code, metadata, dependency floors, or extras
change.

If only core changes, release only core:

```text
updatesupport-v0.1.2
```

No finance tag is needed unless the finance package also changes or needs a new
core dependency floor.

## Trusted Publishing Environments

The GitHub workflows use PyPI trusted publishing. Configure these environments
in GitHub and PyPI/TestPyPI:

- core: `testpypi`, `pypi`
- finance plugin: `finance-testpypi`, `finance-pypi`

The finance workflow publishes from `dist/finance/`; the core workflow publishes
from the root `dist/`.
