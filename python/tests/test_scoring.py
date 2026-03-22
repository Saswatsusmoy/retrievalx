from retrievalx import BM25Config, BM25Index, ScoringVariant


def test_bm25_plus_scores_non_empty() -> None:
    config = BM25Config(scoring=ScoringVariant.plus(delta=0.5))
    index = BM25Index.from_documents(
        ["rust fast", "python batteries", "rust python"],
        config=config,
    )
    results = index.search("rust", top_k=2)
    assert len(results) == 2
    assert results[0].score > 0.0
