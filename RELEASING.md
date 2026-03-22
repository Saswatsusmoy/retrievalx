# Releasing retrievalx

## Preconditions

- CI is green on `main`
- Security checks are passing (`cargo audit`, `cargo deny`, `pip-audit`)
- Benchmarks reviewed for regressions
- Version bump prepared (`Cargo.toml`, `pyproject.toml` if needed)
- Changelog/release notes drafted

## Release Steps

1. Create and push a semver tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

2. GitHub Actions `release.yml` will:

- build wheels for Linux/macOS/Windows and supported Python versions
- build source distribution
- run distribution validation (`twine check`)
- publish to PyPI via trusted publishing

## Post-release Validation

- Verify package visibility on PyPI
- Install from a clean environment:

```bash
python -m pip install --upgrade retrievalx
python -c "import retrievalx; print('ok')"
```

- Sanity run a quick retrieval example and smoke benchmark

## Rollback and Incident Handling

- If a critical issue is found, publish a patched release immediately.
- Document incident details in release notes and follow-up issues.
