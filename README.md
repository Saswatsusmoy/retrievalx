# retrievalx

[![PyPI](https://img.shields.io/pypi/v/retrievalx)](https://pypi.org/project/retrievalx/)
[![Downloads](https://img.shields.io/pypi/dm/retrievalx)](https://pypi.org/project/retrievalx/)
[![Python](https://img.shields.io/pypi/pyversions/retrievalx)](https://pypi.org/project/retrievalx/)
[![License](https://img.shields.io/pypi/l/retrievalx)](https://github.com/Saswatsusmoy/retrievalx/blob/main/LICENSE)
[![CI](https://github.com/Saswatsusmoy/retrievalx/actions/workflows/ci.yml/badge.svg)](https://github.com/Saswatsusmoy/retrievalx/actions/workflows/ci.yml)

The complete BM25 engine for Python — all major scoring variants, multiple retrieval strategies, Rust-native performance.

**48-92x faster** than [rank-bm25](https://github.com/dorianbrown/rank_bm25) with equal or better retrieval quality on [BEIR](https://github.com/beir-cellar/beir) benchmarks.

## Installation

```bash
pip install retrievalx
```

Pre-built wheels available for:

| Platform | Architectures |
|----------|--------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64, ARM64 (Apple Silicon) |
| Windows | x86_64 |

Supports **Python 3.9 through 3.15** (including 3.14 and 3.15 pre-releases). Zero dependencies.

## Quickstart

```python
from retrievalx import BM25Index

# Index documents
index = BM25Index.from_documents([
    "rust and python",
    "information retrieval with bm25",
    "search engine internals",
])

# Search
for hit in index.search("rust retrieval", top_k=2):
    print(f"{hit.doc_id}: {hit.score:.4f}")
```

## Features

### 8 Scoring Variants

BM25 Okapi, Plus, L, Adpt, F (field-weighted), T (term-specific k1), Atire, and Tf-Idf.

```python
from retrievalx import BM25Config, ScoringVariant

config = BM25Config(scoring=ScoringVariant.plus(k1=1.5, b=0.8, delta=1.0))
index = BM25Index.from_documents(docs, config=config)
```

### 5 Retrieval Strategies

Exhaustive DAAT/TAAT, WAND, Block-Max WAND, and MaxScore.

```python
from retrievalx import RetrievalStrategy

# Exact retrieval
config = BM25Config(retrieval=RetrievalStrategy.exhaustive_taat())

# Fast approximate top-k
config = BM25Config(retrieval=RetrievalStrategy.block_max_wand())
```

### Advanced Query Types

```python
from retrievalx import BooleanQuery, PhraseQuery, WeightedQuery

# Boolean: must/should/must_not
index.search_boolean(BooleanQuery(must=["python"], should=["fast"], must_not=["slow"]))

# Phrase with proximity window
index.search_phrase(PhraseQuery(terms=["information", "retrieval"], window=2))

# Weighted terms
index.search_weighted(WeightedQuery(weights={"python": 2.0, "search": 1.0}))
```

### Persistence & Crash Recovery

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

### Incremental Updates

```python
# Add documents (with optional IDs)
index.insert_batch(["new document text"])
index.insert_batch([("doc-id", "document with custom ID")])

# Delete and compact
index.delete("doc-id")
index.compact()
```

### Score Fusion

Combine BM25 with dense retrieval or other signals:

```python
from retrievalx import rrf, linear_combination, min_max_normalize

fused = rrf(bm25_results, dense_results, k=60)
fused = linear_combination(bm25_results, dense_results, alpha=0.7)
normalized = min_max_normalize(scores)
```

### Evaluation Metrics

Built-in IR metrics with native acceleration:

```python
from retrievalx import ndcg_at_k, recall_at_k, precision_at_k, mrr, average_precision_at_k

ndcg = ndcg_at_k(ranked_ids, relevant_ids, k=10)
```

### Custom Tokenization

```python
from retrievalx import BM25Config, TokenizerConfig, Tokenizer, Filter, Stemmer

config = BM25Config(
    tokenizer=TokenizerConfig(
        tokenizer=Tokenizer.UNICODE,
        filters=[Filter.LOWERCASE, Filter.stopwords("en"), Filter.length(min_len=2)],
        stemmer=Stemmer.snowball("en"),
    )
)
```

## Benchmarks

On [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) (5,183 documents, 300 queries):

| Engine | QPS | Speedup | nDCG@10 | P50 (ms) |
|--------|----:|--------:|--------:|---------:|
| rank-bm25 (Okapi) | 134 | 1x | 0.5618 | 6.964 |
| **retrievalx** (Exhaustive DAAT) | **6,505** | **48x** | **0.5723** | 0.152 |
| **retrievalx** (Block-Max WAND) | **4,919** | **37x** | **0.5723** | 0.151 |
| **retrievalx** (Exhaustive TAAT) | **11,935** | **89x** | **0.5723** | 0.083 |
| **retrievalx** (MaxScore) | **7,351** | **55x** | **0.5723** | 0.099 |

All retrieval strategies produce identical quality metrics — no accuracy tradeoff.

Full 40-configuration matrix (8 scorers x 5 strategies): [docs/benchmarks.md](docs/benchmarks.md)

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
| [rag_hybrid_pipeline.py](examples/rag_hybrid_pipeline.py) | RAG hybrid retrieval pipeline |
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0
