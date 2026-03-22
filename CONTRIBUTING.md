# Contributing to retrievalx

Thank you for contributing to `retrievalx`.

This project is performance-sensitive and correctness-sensitive. Changes are
accepted only when they preserve retrieval quality guarantees and pass all
quality gates.

## Before You Start

1. Read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
2. Open an issue for significant changes (new feature, API change, algorithmic
   change, major refactor).
3. For vulnerabilities, do **not** open a public issue. Follow
   [SECURITY.md](SECURITY.md).

## Local Setup

### Required toolchain

- Rust `1.75+` (stable)
- Python `3.9+`
- `maturin`

### Recommended bootstrap

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

### One-command local checks

```bash
./scripts/check_all.sh
```

This mirrors CI expectations as closely as possible.

## Development Workflow

1. Create a branch from `main`.
2. Keep changes scoped to one logical concern.
3. Add or update tests with every behavior change.
4. Update docs for user-facing or operational changes.
5. Run `./scripts/check_all.sh` before pushing.

## Code Standards

### Rust

- `unsafe` is forbidden.
- Treat clippy warnings as errors.
- Prefer deterministic behavior and explicit bounds checks.
- Keep hot-path allocations and hash lookups minimal.

### Python

- Keep Python as orchestration/API layer; core compute should stay in Rust.
- Keep type hints accurate for public APIs.
- Avoid hidden behavior changes in wrappers around native code.

### API stability

- Public Python API compatibility is required for all non-breaking releases.
- If a breaking change is required, document migration guidance in
  `docs/migration.md` and call it out in release notes.

## Testing Requirements

### Must pass for every PR

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `cargo deny check`
- `cargo audit`
- `ruff check python examples`
- `mypy python/retrievalx`
- `pytest -q python/tests`

### For retrieval/scoring changes

- Add parity tests against exhaustive retrieval where applicable.
- Validate quality metrics changes are intentional and documented.
- Provide benchmark evidence if latency/QPS behavior changes.

## Pull Request Checklist

- [ ] Problem and scope are clearly described.
- [ ] Tests added/updated and passing locally.
- [ ] Documentation updated (README/docs/examples as needed).
- [ ] Security implications evaluated.
- [ ] Backward compatibility evaluated.
- [ ] Benchmarks included for performance-critical changes.

## Commit Guidance

- Keep commits reviewable and focused.
- Use descriptive commit messages with clear intent.
- Do not mix unrelated refactors with functional changes.

## Reporting Performance Regressions

For performance regressions, include:

- corpus size and dataset
- scorer + retrieval strategy + config
- baseline vs current numbers (`build_ms`, `qps`, p95/p99)
- reproduction command

## Need Help?

Use [SUPPORT.md](SUPPORT.md) for support and communication channels.
