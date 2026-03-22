# retrievalx Architecture

`retrievalx` is organized as a Rust workspace with clear crate boundaries:

- `retrievalx-core`: indexing, scoring, query execution, fusion.
- `retrievalx-tokenize`: tokenization pipeline, stopwords, stemming.
- `retrievalx-persist`: binary snapshot, WAL, metadata export.
- `retrievalx-eval`: BEIR-style loading, metrics, latency profiling.
- `retrievalx-py`: PyO3 extension module powering the Python package.

Data flow:

1. Python API builds or loads an index through `retrievalx-py`.
2. Core tokenization normalizes input text and query text identically.
3. `retrievalx-core` stores postings, vocabulary stats, and tombstones.
4. Retrieval strategy ranks top-k hits using selected scoring variant.
5. Optional persistence serializes snapshots and WAL for recovery (`load_with_wal` / replay support).
6. Loading supports in-memory and true OS memory-mapped mode (`mode="mmap"` in Python).
7. Sparse vectors can be exported for both query text and documents as `(term_id, weight)` tuples.
