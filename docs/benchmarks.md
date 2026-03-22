# Comprehensive Benchmark Matrix

- Generated (UTC): `2026-03-22T06:47:40.747688+00:00`
- Dataset: `scifact` (split `test`)
- Python: `3.12.9`
- Platform: `macOS-26.3.1-arm64-arm-64bit`
- retrievalx: `0.1.3`
- rank-bm25: `0.2.2`
- Corpus size: `5183`
- Query count: `300`
- Top-k: `10`
- retrievalx WAND/BMW top_k_budget: `1000`
- Tokenization for parity: `lower+whitespace (no stemming, no stopwords)`

## Method

- Real corpus/queries/qrels loaded from BEIR.
- Identical corpus, query set, and tokenization for all systems.
- retrievalx matrix: all 8 scoring variants × all 5 retrieval strategies (40 runs).
- WAND/BlockMaxWAND are exact when `top_k_budget=0`; approximate when `> 0`.
- rank-bm25 matrix: all available variants (Okapi, Plus, L) with exhaustive scoring.
- Effectiveness metrics computed against BEIR qrels.

## Top Highlights

- Fastest query throughput:
  - `retrievalx` | `Plus(k1=1.2,b=0.75,delta=0.5)` | `ExhaustiveTAAT` | `12281.16` QPS
- Strongest effectiveness (nDCG@k):
  - `rank-bm25` | `Plus(k1=1.2,b=0.75,delta=0.5)` | `Exhaustive` | `0.5765`
- Fastest index build:
  - `retrievalx` | `Okapi(k1=1.2,b=0.75)` | `ExhaustiveDAAT` | `121.28 ms`

## rank-bm25 Variant Results

| System | Scoring | Retrieval | Build (ms) | Query Total (ms) | QPS | P50 (ms) | P95 (ms) | P99 (ms) | nDCG@k | MAP@k | Recall@k | Precision@k | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank-bm25 | L(k1=1.2,b=0.75,delta=0.5) | Exhaustive | 207.18 | 2204.51 | 136.08 | 6.710 | 13.302 | 15.434 | 0.3983 | 0.3477 | 0.5431 | 0.0597 | 0.3571 |
| rank-bm25 | Okapi(k1=1.2,b=0.75) | Exhaustive | 194.46 | 2231.67 | 134.43 | 6.964 | 13.268 | 15.631 | 0.5618 | 0.5157 | 0.6922 | 0.0767 | 0.5258 |
| rank-bm25 | Plus(k1=1.2,b=0.75,delta=0.5) | Exhaustive | 142.42 | 2253.26 | 133.14 | 6.828 | 14.281 | 17.647 | 0.5765 | 0.5343 | 0.6937 | 0.0773 | 0.5436 |

## retrievalx Full Configuration Matrix

| System | Scoring | Retrieval | Build (ms) | Query Total (ms) | QPS | P50 (ms) | P95 (ms) | P99 (ms) | nDCG@k | MAP@k | Recall@k | Precision@k | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrievalx | Adpt(b=0.75) | BlockMaxWand(top_k_budget=1000) | 181.86 | 54.22 | 5533.43 | 0.139 | 0.491 | 0.638 | 0.5691 | 0.5280 | 0.6823 | 0.0757 | 0.5379 |
| retrievalx | Adpt(b=0.75) | ExhaustiveDAAT | 133.20 | 44.29 | 6774.09 | 0.146 | 0.262 | 0.312 | 0.5691 | 0.5280 | 0.6823 | 0.0757 | 0.5379 |
| retrievalx | Adpt(b=0.75) | ExhaustiveTAAT | 131.75 | 24.83 | 12083.90 | 0.083 | 0.142 | 0.167 | 0.5691 | 0.5280 | 0.6823 | 0.0757 | 0.5379 |
| retrievalx | Adpt(b=0.75) | MaxScore | 127.02 | 40.51 | 7406.03 | 0.100 | 0.351 | 0.492 | 0.5691 | 0.5280 | 0.6823 | 0.0757 | 0.5379 |
| retrievalx | Adpt(b=0.75) | Wand(top_k_budget=1000) | 129.90 | 52.42 | 5722.67 | 0.132 | 0.492 | 0.628 | 0.5691 | 0.5280 | 0.6823 | 0.0757 | 0.5379 |
| retrievalx | Atire(k1=1.2,b=0.75) | BlockMaxWand(top_k_budget=1000) | 178.58 | 57.78 | 5191.84 | 0.144 | 0.526 | 0.661 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Atire(k1=1.2,b=0.75) | ExhaustiveDAAT | 138.67 | 45.60 | 6578.42 | 0.148 | 0.269 | 0.322 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Atire(k1=1.2,b=0.75) | ExhaustiveTAAT | 143.12 | 24.97 | 12015.16 | 0.083 | 0.141 | 0.170 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Atire(k1=1.2,b=0.75) | MaxScore | 148.85 | 42.00 | 7143.15 | 0.104 | 0.379 | 0.491 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Atire(k1=1.2,b=0.75) | Wand(top_k_budget=1000) | 135.30 | 55.59 | 5396.96 | 0.139 | 0.512 | 0.655 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | F(k1=1.2,b=0.75,fields=[]) | BlockMaxWand(top_k_budget=1000) | 237.65 | 60.19 | 4984.31 | 0.146 | 0.531 | 0.723 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | F(k1=1.2,b=0.75,fields=[]) | ExhaustiveDAAT | 127.00 | 43.14 | 6954.26 | 0.140 | 0.254 | 0.303 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | F(k1=1.2,b=0.75,fields=[]) | ExhaustiveTAAT | 285.06 | 50.73 | 5913.98 | 0.112 | 0.527 | 0.860 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | F(k1=1.2,b=0.75,fields=[]) | MaxScore | 189.63 | 44.23 | 6782.41 | 0.105 | 0.386 | 0.503 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | F(k1=1.2,b=0.75,fields=[]) | Wand(top_k_budget=1000) | 243.15 | 62.21 | 4822.07 | 0.147 | 0.539 | 0.894 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | L(k1=1.2,b=0.75,c=1.0) | BlockMaxWand(top_k_budget=1000) | 177.06 | 47.30 | 6341.94 | 0.123 | 0.422 | 0.539 | 0.5624 | 0.5181 | 0.6873 | 0.0760 | 0.5281 |
| retrievalx | L(k1=1.2,b=0.75,c=1.0) | ExhaustiveDAAT | 130.51 | 43.19 | 6945.94 | 0.142 | 0.253 | 0.299 | 0.5624 | 0.5181 | 0.6873 | 0.0760 | 0.5281 |
| retrievalx | L(k1=1.2,b=0.75,c=1.0) | ExhaustiveTAAT | 125.12 | 24.59 | 12198.86 | 0.082 | 0.141 | 0.169 | 0.5624 | 0.5181 | 0.6873 | 0.0760 | 0.5281 |
| retrievalx | L(k1=1.2,b=0.75,c=1.0) | MaxScore | 130.14 | 37.62 | 7974.43 | 0.092 | 0.334 | 0.457 | 0.5624 | 0.5181 | 0.6873 | 0.0760 | 0.5281 |
| retrievalx | L(k1=1.2,b=0.75,c=1.0) | Wand(top_k_budget=1000) | 125.93 | 46.44 | 6459.54 | 0.118 | 0.417 | 0.526 | 0.5624 | 0.5181 | 0.6873 | 0.0760 | 0.5281 |
| retrievalx | Okapi(k1=1.2,b=0.75) | BlockMaxWand(top_k_budget=1000) | 206.72 | 60.99 | 4919.21 | 0.151 | 0.536 | 0.675 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Okapi(k1=1.2,b=0.75) | ExhaustiveDAAT | 121.28 | 46.12 | 6504.51 | 0.152 | 0.274 | 0.336 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Okapi(k1=1.2,b=0.75) | ExhaustiveTAAT | 169.90 | 25.14 | 11934.97 | 0.083 | 0.142 | 0.170 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Okapi(k1=1.2,b=0.75) | MaxScore | 132.97 | 40.81 | 7351.08 | 0.099 | 0.337 | 0.486 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Okapi(k1=1.2,b=0.75) | Wand(top_k_budget=1000) | 146.53 | 66.87 | 4486.32 | 0.149 | 0.649 | 0.980 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | Plus(k1=1.2,b=0.75,delta=0.5) | BlockMaxWand(top_k_budget=1000) | 179.46 | 51.38 | 5838.42 | 0.131 | 0.462 | 0.581 | 0.5704 | 0.5298 | 0.6823 | 0.0757 | 0.5400 |
| retrievalx | Plus(k1=1.2,b=0.75,delta=0.5) | ExhaustiveDAAT | 128.38 | 42.40 | 7074.78 | 0.137 | 0.248 | 0.296 | 0.5704 | 0.5298 | 0.6823 | 0.0757 | 0.5400 |
| retrievalx | Plus(k1=1.2,b=0.75,delta=0.5) | ExhaustiveTAAT | 124.42 | 24.43 | 12281.16 | 0.082 | 0.139 | 0.159 | 0.5704 | 0.5298 | 0.6823 | 0.0757 | 0.5400 |
| retrievalx | Plus(k1=1.2,b=0.75,delta=0.5) | MaxScore | 132.61 | 38.57 | 7777.15 | 0.095 | 0.334 | 0.455 | 0.5704 | 0.5298 | 0.6823 | 0.0757 | 0.5400 |
| retrievalx | Plus(k1=1.2,b=0.75,delta=0.5) | Wand(top_k_budget=1000) | 127.44 | 49.30 | 6084.87 | 0.124 | 0.440 | 0.564 | 0.5704 | 0.5298 | 0.6823 | 0.0757 | 0.5400 |
| retrievalx | T(default_k1=1.2,b=0.75,term_k1={}) | BlockMaxWand(top_k_budget=1000) | 252.62 | 72.37 | 4145.17 | 0.182 | 0.662 | 0.838 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | T(default_k1=1.2,b=0.75,term_k1={}) | ExhaustiveDAAT | 178.09 | 47.29 | 6344.36 | 0.152 | 0.288 | 0.349 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | T(default_k1=1.2,b=0.75,term_k1={}) | ExhaustiveTAAT | 177.19 | 27.39 | 10951.35 | 0.090 | 0.162 | 0.227 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | T(default_k1=1.2,b=0.75,term_k1={}) | MaxScore | 149.86 | 53.65 | 5592.23 | 0.109 | 0.460 | 0.783 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | T(default_k1=1.2,b=0.75,term_k1={}) | Wand(top_k_budget=1000) | 159.10 | 64.30 | 4665.33 | 0.153 | 0.639 | 0.744 | 0.5723 | 0.5315 | 0.6837 | 0.0763 | 0.5408 |
| retrievalx | TfIdf | BlockMaxWand(top_k_budget=1000) | 236.32 | 40.49 | 7409.50 | 0.124 | 0.285 | 0.317 | 0.1380 | 0.1212 | 0.1839 | 0.0193 | 0.1280 |
| retrievalx | TfIdf | ExhaustiveDAAT | 189.34 | 45.36 | 6614.03 | 0.149 | 0.273 | 0.325 | 0.2780 | 0.2228 | 0.4434 | 0.0483 | 0.2298 |
| retrievalx | TfIdf | ExhaustiveTAAT | 148.83 | 25.04 | 11981.31 | 0.084 | 0.138 | 0.168 | 0.2780 | 0.2228 | 0.4434 | 0.0483 | 0.2298 |
| retrievalx | TfIdf | MaxScore | 154.28 | 56.75 | 5286.63 | 0.164 | 0.365 | 0.822 | 0.2780 | 0.2228 | 0.4434 | 0.0483 | 0.2298 |
| retrievalx | TfIdf | Wand(top_k_budget=1000) | 164.12 | 39.20 | 7652.58 | 0.116 | 0.275 | 0.306 | 0.1380 | 0.1212 | 0.1839 | 0.0193 | 0.1280 |

## Like-for-Like Deltas (retrievalx ExhaustiveDAAT vs rank-bm25)

*Notes:*
- Okapi and Plus are directly comparable.
- BM25L formulas differ between libraries (`retrievalx` uses `c`; `rank-bm25` uses `delta`).

| Scoring | retrievalx QPS | rank-bm25 QPS | Speedup (x) | retrievalx Build (ms) | rank-bm25 Build (ms) | Build Ratio (x) | Δ nDCG@k | Δ MAP@k | Δ Recall@k | Δ Precision@k | Δ MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Okapi(k1=1.2,b=0.75) | 6504.51 | 134.43 | 48.39 | 121.28 | 194.46 | 0.62 | +0.0105 | +0.0158 | -0.0085 | -0.0003 | +0.0150 |
| Plus(k1=1.2,b=0.75,delta=0.5) | 7074.78 | 133.14 | 53.14 | 128.38 | 142.42 | 0.90 | -0.0062 | -0.0045 | -0.0113 | -0.0017 | -0.0036 |
| L(k1=1.2,b=0.75,c=1.0) | 6945.94 | 136.08 | 51.04 | 130.51 | 207.18 | 0.63 | +0.1642 | +0.1703 | +0.1442 | +0.0163 | +0.1710 |

## Reproduce

```bash
python examples/comprehensive_benchmark_matrix.py --dataset scifact --split test --top-k 10 --retrievalx-top-k-budget 1000
```
