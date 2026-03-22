# retrievalx vs rank-bm25 Benchmark (Real BEIR Data)

- Generated (UTC): `2026-03-22T06:47:21.486349+00:00`
- Dataset: `scifact` (split `test`)
- Python: `3.12.9`
- Platform: `macOS-26.3.1-arm64-arm-64bit`
- retrievalx: `0.1.0`
- rank-bm25: `0.2.2`
- Corpus size: `5183`
- Query count: `300`
- Top-k: `10`

## Method

- Real corpus/queries/qrels loaded from BEIR.
- Same corpus and query set used for all engines.
- `retrievalx_exhaustive_daat` for parity baseline.
- `retrievalx_block_max_wand` for production-optimized sparse retrieval.
- Effectiveness metrics computed against BEIR qrels.

## Results

| Engine | Build (ms) | Query Total (ms) | QPS | P50 (ms) | P95 (ms) | P99 (ms) | nDCG@k | MAP@k | Recall@k | Precision@k | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_bm25_okapi | 189.70 | 2430.27 | 123.44 | 7.100 | 15.022 | 20.939 | 0.5597 | 0.5147 | 0.6862 | 0.0763 | 0.5242 |
| retrievalx_exhaustive_daat | 114.47 | 39.29 | 7634.91 | 0.124 | 0.252 | 0.309 | 0.6871 | 0.6427 | 0.8072 | 0.0887 | 0.6561 |
| retrievalx_block_max_wand | 136.45 | 63.96 | 4690.38 | 0.162 | 0.599 | 0.755 | 0.6871 | 0.6427 | 0.8072 | 0.0887 | 0.6561 |

## Reproduce

```bash
python examples/benchmark_retrievalx_vs_rank_bm25.py --dataset scifact --split test --top-k 10
```
