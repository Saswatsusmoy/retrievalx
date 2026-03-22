#!/usr/bin/env bash
set -euo pipefail

FAST=0
if [[ "${1:-}" == "--fast" ]]; then
  FAST=1
fi

echo "[check] rustfmt"
cargo fmt --all --check

echo "[check] clippy"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "[check] cargo tests"
cargo test --workspace --all-features

if [[ "$FAST" -eq 0 ]]; then
  echo "[check] cargo deny"
  cargo deny check

  echo "[check] cargo audit"
  cargo audit
fi

echo "[check] python lint"
python -m ruff check python examples

echo "[check] python typing"
python -m mypy python/retrievalx

echo "[check] build wheel"
maturin build --release -m crates/retrievalx-py/Cargo.toml

echo "[check] install wheel"
shopt -s nullglob
wheels=(target/wheels/retrievalx-*.whl)
shopt -u nullglob

if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "No wheels found in target/wheels after build" >&2
  exit 1
fi

wheel_path="${wheels[0]}"
for w in "${wheels[@]}"; do
  if [[ "$w" -nt "$wheel_path" ]]; then
    wheel_path="$w"
  fi
done

python -m pip install --force-reinstall --no-deps "$wheel_path"

echo "[check] pytest"
python -m pytest -q python/tests

echo "[check] complete"
