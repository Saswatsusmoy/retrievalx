from __future__ import annotations

import time

from retrievalx import BM25Config, BM25Index, FieldConfig, ScoringVariant

CHANGE_REQUESTS = [
    {
        "id": "cr-2101",
        "title": "Rotate API gateway certificates for external ingress",
        "body": "Plan staged certificate rollout, validate mutual TLS to payment providers, "
        "and run zero-downtime handoff during low traffic window.",
        "tags": ["security", "tls", "gateway"],
    },
    {
        "id": "cr-2102",
        "title": "Migrate checkout service to new Postgres primary region",
        "body": "Perform read replica warmup, replay write-ahead logs, and cut over with "
        "feature flag rollback protection.",
        "tags": ["database", "checkout", "migration"],
    },
    {
        "id": "cr-2103",
        "title": "Harden identity provider token issuer validation",
        "body": "Introduce explicit audience checks and key identifier rotation audits "
        "to prevent token acceptance drift.",
        "tags": ["identity", "security", "auth"],
    },
    {
        "id": "cr-2104",
        "title": "Increase ingestion pipeline throughput with autoscaling workers",
        "body": "Tune queue partitioning and apply load-shedding guardrails before peak "
        "retail traffic events.",
        "tags": ["throughput", "autoscaling", "pipeline"],
    },
]


def flatten(record: dict[str, object]) -> str:
    title = str(record["title"])
    body = str(record["body"])
    tags = " ".join(str(tag) for tag in record["tags"])
    # Keep title highly salient for retrieval by repeating it.
    return f"{title}\n{title}\n{body}\n{tags}"


def main() -> None:
    fields = [
        FieldConfig(name="title", weight=2.0, b=0.2).to_dict(),
        FieldConfig(name="body", weight=1.0, b=0.75).to_dict(),
        FieldConfig(name="tags", weight=1.5, b=0.2).to_dict(),
    ]
    config = BM25Config(scoring=ScoringVariant.bm25_f(k1=1.2, b=0.75, fields=fields))

    docs = [(record["id"], flatten(record)) for record in CHANGE_REQUESTS]
    build_start = time.perf_counter()
    index = BM25Index.from_documents(docs, config=config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    query = "token issuer validation security key rotation"
    search_start = time.perf_counter()
    hits = index.search(query, top_k=3)
    search_ms = (time.perf_counter() - search_start) * 1000.0

    print(f"Built BM25F-style index in {build_ms:.2f} ms")
    print(f"Search query: {query}")
    print(f"Query latency: {search_ms:.3f} ms")
    print("Top results:")
    for hit in hits:
        print(f"  {hit.doc_id}: {hit.score:.4f}")


if __name__ == "__main__":
    main()
