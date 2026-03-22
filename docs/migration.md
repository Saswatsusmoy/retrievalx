# Migration Guide

## from `rank_bm25`

- Replace in-memory Python scoring with `BM25Index.from_documents`.
- Use `BM25Config` to tune `k1`, `b`, tokenizer pipeline.

## from `bm25s`

- Move scoring and indexing hot path to Rust via wheels.
- Use persistence (`save`, `load`) for repeat startup.

## from Elasticsearch BM25

- No cluster runtime required.
- Keep BM25 behavior while embedding retrieval in your Python process.
- Use built-in fusion for hybrid sparse+dense ranking.
