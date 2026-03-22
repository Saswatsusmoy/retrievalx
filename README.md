# retrievalx

[![PyPI](https://img.shields.io/pypi/v/retrievalx)](https://pypi.org/project/retrievalx/)
[![Python](https://img.shields.io/pypi/pyversions/retrievalx)](https://pypi.org/project/retrievalx/)
[![License](https://img.shields.io/pypi/l/retrievalx)](https://github.com/Saswatsusmoy/retrievalx/blob/main/LICENSE)
[![CI](https://github.com/Saswatsusmoy/retrievalx/actions/workflows/ci.yml/badge.svg)](https://github.com/Saswatsusmoy/retrievalx/actions/workflows/ci.yml)

The complete BM25 engine for Python: all major scoring variants, multiple retrieval strategies, Rust-native performance.

**48-92x faster** than rank-bm25 with equal or better retrieval quality on BEIR benchmarks.

## Installation

```bash
pip install retrievalx
```

Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, ARM64), and Windows (x86_64). Python 3.9+.

## Quickstart

```python
from retrievalx import BM25Index

index = BM25Index.from_documents([
    "rust and python",
    "information retrieval with bm25",
    "search engine internals",
])

for hit in index.search("rust retrieval", top_k=2):
    print(f"{hit.doc_id}: {hit.score:.4f}")
```

## Features

### Scoring Variants

BM25 Okapi, Plus, L, Adpt, F (field-weighted), T (term-specific k1), Atire, and Tf-Idf.

```python
from retrievalx import BM25Config, ScoringVariant

config = BM25Config(scoring=ScoringVariant.plus(k1=1.5, b=0.8, delta=1.0))
index = BM25Index.from_documents(docs, config=config)
```

### Retrieval Strategies

Exhaustive DAAT/TAAT, WAND, Block-Max WAND, and MaxScore — choose between exact and approximate top-k retrieval.

```python
from retrievalx import RetrievalStrategy

config = BM25Config(retrieval=RetrievalStrategy.block_max_wand())
```

### Query Types

```python
from retrievalx import BooleanQuery, PhraseQuery, WeightedQuery

# Boolean query
index.search_boolean(BooleanQuery(must=["python"], should=["fast"], must_not=["slow"]))

# Phrase query with proximity window
index.search_phrase(PhraseQuery(terms=["information", "retrieval"], window=2))

# Weighted terms
index.search_weighted(WeightedQuery(weights={"python": 2.0, "search": 1.0}))
```

### Persistence & WAL

```python
# Save and load
index.save("index.bin")
loaded = BM25Index.load("index.bin")              # in-memory
loaded = BM25Index.load("index.bin", mode="mmap")  # memory-mapped

# Write-ahead log for crash recovery
index.enable_wal("index.wal")
index.insert_batch(new_docs)
index.compact_and_flush("index.bin")
```

### Score Fusion

Combine BM25 with dense retrieval or other signals:

```python
from retrievalx import rrf, linear_combination, min_max_normalize

fused = rrf(bm25_results, dense_results, k=60)
fused = linear_combination(bm25_results, dense_results, alpha=0.7)
```

### Evaluation Metrics

```python
from retrievalx import ndcg_at_k, recall_at_k, mrr

ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=10)
```

## Benchmarks

On [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) (5,183 documents, 300 queries):

| Engine | QPS | nDCG@10 | Build (ms) |
|--------|----:|--------:|-----------:|
| rank-bm25 (Okapi) | 134 | 0.5618 | 194 |
| retrievalx (Exhaustive DAAT) | **6,505** | **0.5723** | 121 |
| retrievalx (Block-Max WAND) | **4,919** | **0.5723** | 207 |
| retrievalx (Exhaustive TAAT) | **11,935** | **0.5723** | 170 |

Full results: [docs/benchmarks.md](docs/benchmarks.md) | [docs/rank_bm25_benchmarks.md](docs/rank_bm25_benchmarks.md)

## Architecture

Rust workspace with five crates:

| Crate | Purpose |
|-------|---------|
| `retrievalx-core` | Indexing, scoring, retrieval, query execution, fusion |
| `retrievalx-tokenize` | Unicode tokenization, stemming, stopword filtering |
| `retrievalx-persist` | Binary serialization, mmap, write-ahead log |
| `retrievalx-eval` | IR metrics, BEIR benchmark runner |
| `retrievalx-py` | PyO3 bindings |

Details: [docs/architecture.md](docs/architecture.md) | [docs/algorithms.md](docs/algorithms.md)

## Examples

| Example | Description |
|---------|-------------|
| [quickstart.py](examples/quickstart.py) | Basic indexing and search |
| [it_ticket_search.py](examples/it_ticket_search.py) | IT ticket triage |
| [legal_clause_discovery.py](examples/legal_clause_discovery.py) | Legal document search |
| [ecommerce_query_tuning.py](examples/ecommerce_query_tuning.py) | E-commerce product search |
| [security_log_hunt.py](examples/security_log_hunt.py) | Security log analysis |
| [wal_crash_recovery.py](examples/wal_crash_recovery.py) | WAL crash recovery |
| [production_hybrid_reranking.py](examples/production_hybrid_reranking.py) | Hybrid BM25 + dense reranking |
| [query_expansion_prf.py](examples/query_expansion_prf.py) | Pseudo-relevance feedback |
| [custom_tokenizer.py](examples/custom_tokenizer.py) | Custom tokenizer pipeline |
| [bm25f_structured_docs.py](examples/bm25f_structured_docs.py) | BM25F field-weighted scoring |
| [benchmark_retrievalx_vs_rank_bm25.py](examples/benchmark_retrievalx_vs_rank_bm25.py) | Benchmark vs rank-bm25 |

## Development

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Run all checks (mirrors CI)
./scripts/check_all.sh

# Run benchmarks
pip install -e .[bench]
python examples/benchmark_retrievalx_vs_rank_bm25.py --dataset scifact
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

Apache-2.0
