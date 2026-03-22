from __future__ import annotations

import math
import re
from collections import defaultdict

from retrievalx import BM25Index, WeightedQuery, ndcg_at_k, recall_at_k

DOCS = [
    ("d1", "Checkout API timeout caused by connection pool exhaustion in postgres"),
    ("d2", "Postgres slow query log analysis for order table index regression"),
    ("d3", "Kubernetes autoscaling guide for stateless ingestion services"),
    ("d4", "Incident report payment gateway retries after downstream timeout"),
    ("d5", "Database failover runbook with replication lag and recovery checks"),
    ("d6", "API gateway TLS certificate rotation and zero downtime deployment"),
    ("d7", "On-call playbook for elevated checkout latency and retry storms"),
    ("d8", "Identity token issuer mismatch troubleshooting"),
]

QUERY = "checkout latency"
RELEVANT = {"d1", "d2", "d7"}
STOPWORDS = {
    "the",
    "for",
    "and",
    "with",
    "after",
    "from",
    "in",
    "of",
    "to",
    "a",
    "on",
    "by",
    "api",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def expand_query(
    original_query: str,
    ranked_doc_ids: list[str],
    docs_by_id: dict[str, str],
    feedback_docs: int = 4,
    expansion_terms: int = 8,
) -> WeightedQuery:
    base_terms = tokenize(original_query)
    base_set = set(base_terms)

    doc_freq: dict[str, int] = defaultdict(int)
    for text in docs_by_id.values():
        for token in set(tokenize(text)):
            doc_freq[token] += 1
    total_docs = max(len(docs_by_id), 1)

    scores: dict[str, float] = defaultdict(float)
    for doc_id in ranked_doc_ids[:feedback_docs]:
        text = docs_by_id[doc_id]
        for token in tokenize(text):
            if token in STOPWORDS or token in base_set:
                continue
            idf = math.log((total_docs + 1.0) / (doc_freq.get(token, 0) + 1.0)) + 1.0
            scores[token] += idf

    weights: dict[str, float] = {term: 1.5 for term in base_terms}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (term, _) in enumerate(ranked[:expansion_terms], start=1):
        weights[term] = 0.9 / rank
    return WeightedQuery(weights=weights)


def main() -> None:
    index = BM25Index.from_documents(DOCS)
    docs_by_id = {doc_id: text for doc_id, text in DOCS}

    baseline_hits = index.search(QUERY, top_k=5)
    baseline_ranked = [hit.doc_id for hit in baseline_hits]

    expanded_query = expand_query(
        original_query=QUERY,
        ranked_doc_ids=baseline_ranked,
        docs_by_id=docs_by_id,
    )
    expanded_hits = index.search_weighted(expanded_query, top_k=5)
    expanded_ranked = [hit.doc_id for hit in expanded_hits]

    print(f"Original query: {QUERY}")
    print(f"Expanded terms: {expanded_query.weights}")
    print("")
    print("Baseline top-5:")
    for hit in baseline_hits:
        print(f"  {hit.doc_id}: {hit.score:.4f}")
    print("")
    print("Expanded top-5:")
    for hit in expanded_hits:
        print(f"  {hit.doc_id}: {hit.score:.4f}")
    print("")
    print(f"Baseline Recall@5: {recall_at_k(baseline_ranked, RELEVANT, 5):.4f}")
    print(f"Expanded Recall@5: {recall_at_k(expanded_ranked, RELEVANT, 5):.4f}")
    print(f"Baseline nDCG@5: {ndcg_at_k(baseline_ranked, RELEVANT, 5):.4f}")
    print(f"Expanded nDCG@5: {ndcg_at_k(expanded_ranked, RELEVANT, 5):.4f}")


if __name__ == "__main__":
    main()
