# retrievalx

The complete BM25 engine for Python: all major BM25 variants, production-scale internals, Rust-native performance.

## Highlights

- Rust core + PyO3 bindings, distributed as wheels.
- BM25 variants: Okapi, Plus, L, Adpt, F, T, Atire, Tf-Idf.
- Retrieval strategies: Exhaustive DAAT/TAAT, WAND, Block-Max WAND, MaxScore.
- Incremental updates with tombstones + explicit compaction.
- Persistence with binary snapshots, metadata sidecar, WAL replay, and mmap loading.
- Fusion utilities: RRF, linear combination, score normalization.
- Built-in BEIR benchmark, recall degradation report, and scoring variant comparison runner.

## Quickstart

```python
from retrievalx import BM25Index

index = BM25Index.from_documents([
    "rust and python",
    "information retrieval with bm25",
])

print(index.search("rust retrieval", top_k=5))
```

## Real-world Examples

- `examples/it_ticket_search.py`
- `examples/legal_clause_discovery.py`
- `examples/ecommerce_query_tuning.py`
- `examples/security_log_hunt.py`
- `examples/wal_crash_recovery.py`
- `examples/multilingual_news_monitor.py`
- `examples/production_hybrid_reranking.py`
- `examples/benchmark_retrievalx_vs_rank_bm25.py`

## Build

```bash
./scripts/check_all.sh
```

## Project Policies

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Support policy: [SUPPORT.md](SUPPORT.md)
- Governance model: [GOVERNANCE.md](GOVERNANCE.md)
- Release process: [RELEASING.md](RELEASING.md)

## License

Apache-2.0
