# Algorithms

## Scoring

Implemented scoring variants:

- BM25 Okapi
- BM25+
- BM25L
- BM25-Adpt
- BM25F
- BM25T
- ATIRE BM25
- TF-IDF

Common BM25 form used in `retrievalx`:

\[
\text{score}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t,d)(k_1+1)}{tf(t,d)+k_1(1-b+b\cdot |d|/\text{avgdl})}
\]

BM25+ adds \(\delta\) to the TF component to avoid zero lower-bounds for matching terms.
BM25L applies a smoothed c-parameter to stabilize long document behavior.

## Retrieval

Available strategy API:

- Exhaustive DAAT
- Exhaustive TAAT
- WAND
- Block-Max WAND
- MaxScore

Deterministic top-k parity with exhaustive retrieval is guaranteed for:

- Exhaustive DAAT
- Exhaustive TAAT
- WAND with `top_k_budget=0` (unbounded)
- Block-Max WAND with `top_k_budget=0` (unbounded)
- MaxScore (BM25-family scorers); TfIdf requests fall back to Exhaustive DAAT

When `top_k_budget > 0`, WAND/Block-Max WAND are approximate by design and may trade recall for latency.
