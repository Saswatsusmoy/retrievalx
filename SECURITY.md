# Security Policy

`retrievalx` is used in search and RAG pipelines where correctness and data
integrity matter. We treat security reports as high priority.

## Reporting a Vulnerability

Please report vulnerabilities privately through GitHub Security Advisories:

1. Go to `Security` tab of the repository.
2. Click `Report a vulnerability`.
3. Include a clear reproduction and impact assessment.

Do **not** file public issues for undisclosed vulnerabilities.

## Scope

This policy covers:

- Rust crates in `crates/`
- Python bindings and wrappers in `python/`
- Build/release workflows in `.github/workflows/`
- Supply-chain dependencies for Rust and Python packaging

## Supported Versions

| Version | Supported |
|---|---|
| `main` | yes |
| latest release | yes |
| older releases | best effort only |

## Coordinated Disclosure SLA

- Initial response: within 72 hours
- Triage decision: within 7 days
- Remediation plan: within 14 days
- Patch release target:
  - Critical: as soon as practical (typically days)
  - High: within 30 days
  - Medium/Low: next planned release

## Security Hardening Practices

- `unsafe` Rust is forbidden across the workspace.
- CI enforces:
  - `cargo audit`
  - `cargo deny check`
  - Python dependency auditing
- Release artifacts are built in CI and validated before publish.

## What to Include in Reports

- Affected component and version
- Reproduction steps / proof of concept
- Impact (confidentiality, integrity, availability, correctness)
- Suggested mitigation (if known)

## Out of Scope

- Requests for support or general usage questions (use [SUPPORT.md](SUPPORT.md))
- Vulnerabilities in third-party services not controlled by this repository

## Safe Harbor

If you act in good faith, avoid privacy violations/destructive testing, and
report promptly, we will treat your research as authorized.
