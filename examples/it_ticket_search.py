from retrievalx import (
    BM25Config,
    BM25Index,
    BooleanQuery,
    Filter,
    PhraseQuery,
    RetrievalStrategy,
    ScoringVariant,
    Stemmer,
    Tokenizer,
    TokenizerConfig,
)

tickets = [
    (
        "INC-10421",
        "Postgres database timeout in checkout service after connection pool exhaustion",
    ),
    (
        "INC-10422",
        "Redis cluster failover triggered elevated latency in session API",
    ),
    (
        "INC-10423",
        "Kubernetes node disk pressure caused pod eviction for ingestion workers",
    ),
    (
        "INC-10424",
        "Payment gateway webhook retries failed due to TLS certificate mismatch",
    ),
    (
        "INC-10425",
        "Checkout service timeout from slow database query on order table",
    ),
]

config = BM25Config(
    scoring=ScoringVariant.plus(k1=1.4, b=0.65, delta=0.3),
    retrieval=RetrievalStrategy.block_max_wand(top_k_budget=500),
    tokenizer=TokenizerConfig(
        tokenizer=Tokenizer.UNICODE,
        filters=[Filter.LOWERCASE],
        stemmer=Stemmer.NOOP,
    ),
)

index = BM25Index.from_documents(tickets, config=config)

query = BooleanQuery(must=["checkout", "timeout"], must_not=["tls"])
print("Boolean search:")
for hit in index.search_boolean(query, top_k=3):
    print(f"  {hit.doc_id}: {hit.score:.4f}")

phrase_query = PhraseQuery(terms=["checkout", "service", "timeout"], window=2)
print("\nPhrase search:")
for hit in index.search_phrase(phrase_query, top_k=3):
    print(f"  {hit.doc_id}: {hit.score:.4f}")
