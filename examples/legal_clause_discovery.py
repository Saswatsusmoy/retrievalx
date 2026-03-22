from retrievalx import BM25Config, BM25Index, Filter, RetrievalStrategy, Tokenizer, TokenizerConfig

contracts = [
    (
        "MSA-001",
        "Termination for convenience requires ninety day written notice by either party.",
    ),
    (
        "MSA-002",
        "Limitation of liability excludes indirect consequential special and punitive damages.",
    ),
    (
        "MSA-003",
        "Data processing addendum mandates GDPR compliant subprocessors and breach notification.",
    ),
    (
        "MSA-004",
        "Confidential information survives termination for three years except trade secrets.",
    ),
    (
        "MSA-005",
        "Service level agreement includes credits for monthly uptime below ninety nine point nine.",
    ),
]

config = BM25Config(
    retrieval=RetrievalStrategy.exhaustive_taat(),
    tokenizer=TokenizerConfig(
        tokenizer=Tokenizer.WHITESPACE,
        filters=[Filter.LOWERCASE],
    ),
)

index = BM25Index.from_documents(contracts, config=config)
hits = index.search("termination notice convenience", top_k=3)

print("Top contract clauses:")
for hit in hits:
    print(f"  {hit.doc_id}: {hit.score:.4f}")
