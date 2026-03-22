from retrievalx import BM25Config, BM25Index, BooleanQuery, PhraseQuery, RetrievalStrategy


def test_retrieval_strategy_variants_return_results() -> None:
    docs = [
        "rust systems programming",
        "python scripting language",
        "hybrid retrieval with rust and python",
    ]
    for strategy in [
        RetrievalStrategy.exhaustive_daat(),
        RetrievalStrategy.exhaustive_taat(),
        RetrievalStrategy.wand(),
        RetrievalStrategy.block_max_wand(),
        RetrievalStrategy.max_score(),
    ]:
        index = BM25Index.from_documents(docs, config=BM25Config(retrieval=strategy))
        results = index.search("rust", top_k=2)
        assert results


def test_boolean_and_phrase_queries() -> None:
    docs = [
        ("d1", "rust language guide"),
        ("d2", "guide language rust"),
        ("d3", "python language guide"),
    ]
    index = BM25Index.from_documents(docs)

    bq = BooleanQuery(must=["rust"], must_not=["python"])
    b_hits = index.search_boolean(bq, top_k=5)
    assert b_hits
    assert b_hits[0].doc_id in {"d1", "d2"}

    pq = PhraseQuery(terms=["rust", "language"], window=1)
    p_hits = index.search_phrase(pq, top_k=5)
    assert p_hits
    assert p_hits[0].doc_id == "d1"
