from retrievalx import BM25Config, BM25Index, Filter, Stemmer, Tokenizer, TokenizerConfig

articles = [
    (
        "news-1",
        "RBI hints at calibrated rate cuts as inflation cools in urban markets.",
    ),
    (
        "news-2",
        "La banque centrale maintient les taux et surveille la croissance du crédit.",
    ),
    (
        "news-3",
        "El banco central anuncia medidas de liquidez para bancos regionales.",
    ),
    (
        "news-4",
        "Reserve bank extends digital payments framework for cross border settlements.",
    ),
]

config = BM25Config(
    tokenizer=TokenizerConfig(
        tokenizer=Tokenizer.UNICODE,
        filters=[Filter.LOWERCASE],
        stemmer=Stemmer.snowball("en"),
    )
)

index = BM25Index.from_documents(articles, config=config)

for query in ["central bank liquidity", "digital payments framework"]:
    print(f"\nQuery: {query}")
    for hit in index.search(query, top_k=2):
        print(f"  {hit.doc_id}: {hit.score:.4f}")
