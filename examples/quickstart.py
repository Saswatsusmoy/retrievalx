from __future__ import annotations

import tempfile
import time
from pathlib import Path

from retrievalx import BM25Config, BM25Index, BooleanQuery, PhraseQuery, RetrievalStrategy

KNOWLEDGE_BASE = [
    (
        "kb-1001",
        "Checkout API timeout troubleshooting guide. Verify database pool saturation, "
        "slow query logs, and gateway retry policy before scaling workers.",
    ),
    (
        "kb-1002",
        "Incident response process for authentication outage. Validate token issuer "
        "rotation cache invalidation and signing key synchronization.",
    ),
    (
        "kb-1003",
        "Runbook for Kubernetes node pressure. Drain node, rebalance stateful sets, "
        "and recover ingestion lag using replay queue.",
    ),
    (
        "kb-1004",
        "Security hardening checklist for API gateways with WAF signatures, TLS policy, "
        "certificate lifecycle automation, and audit logging.",
    ),
    (
        "kb-1005",
        "Production migration checklist for Postgres major upgrade with replication lag "
        "monitoring and rollback slots.",
    ),
]


def main() -> None:
    config = BM25Config(retrieval=RetrievalStrategy.block_max_wand(top_k_budget=1000))

    build_start = time.perf_counter()
    index = BM25Index.from_documents(KNOWLEDGE_BASE, config=config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    print(f"Indexed {len(index)} docs in {build_ms:.2f} ms")
    print(f"Vocabulary size: {len(index.vocabulary)}")
    print(f"Average document length: {index.avgdl:.2f}")
    print("")

    q1 = "checkout timeout slow query"
    print(f"Query: {q1}")
    for hit in index.search(q1, top_k=3):
        print(f"  {hit.doc_id}: {hit.score:.4f}")

    print("")
    bq = BooleanQuery(must=["api"], should=["security"], must_not=["kubernetes"])
    print("Boolean query (must api, should security, must_not kubernetes):")
    for hit in index.search_boolean(bq, top_k=3):
        print(f"  {hit.doc_id}: {hit.score:.4f}")

    print("")
    pq = PhraseQuery(terms=["token", "issuer", "rotation"], window=2)
    print("Phrase query (token issuer rotation):")
    for hit in index.search_phrase(pq, top_k=3):
        print(f"  {hit.doc_id}: {hit.score:.4f}")

    print("")
    sparse = index.sparse_vector_for_query("authentication token outage")
    print(f"Sparse query vector length: {len(sparse)}")

    with tempfile.TemporaryDirectory(prefix="retrievalx-quickstart-") as tmp:
        index_path = Path(tmp) / "kb_index.rtx"
        index.save(str(index_path))
        reloaded = BM25Index.load(str(index_path), mode="mmap")
        reloaded_hits = reloaded.search("gateway tls policy", top_k=2)
        print("Reloaded index (mmap) query results:")
        for hit in reloaded_hits:
            print(f"  {hit.doc_id}: {hit.score:.4f}")


if __name__ == "__main__":
    main()
