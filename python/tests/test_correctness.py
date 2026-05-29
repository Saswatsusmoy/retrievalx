from retrievalx import BM25Config, BM25Index, Filter, RetrievalStrategy, TokenizerConfig


def _check_parity(docs, queries, top_k=5):
    exhaustive = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.exhaustive_daat()),
    )
    wand = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.wand()),
    )
    blockmax = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.block_max_wand()),
    )
    maxscore = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.max_score()),
    )

    for query in queries:
        gt = {hit.doc_id for hit in exhaustive.search(query, top_k=top_k)}
        for candidate in (wand, blockmax, maxscore):
            got = {hit.doc_id for hit in candidate.search(query, top_k=top_k)}
            assert got == gt, (
                f"Parity failure for query={query!r} "
                f"gt={gt} got={got}"
            )


def test_exhaustive_and_blockmax_produce_same_top1() -> None:
    docs = [
        "rust language guide",
        "python language guide",
        "rust retrieval engine",
        "bm25 retrieval ranking",
    ]
    exhaustive = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.exhaustive_daat()),
    )
    blockmax = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.block_max_wand()),
    )

    q = "rust retrieval"
    e = exhaustive.search(q, top_k=1)
    b = blockmax.search(q, top_k=1)

    assert e and b
    assert e[0].doc_id == b[0].doc_id


def test_recall_parity_against_exhaustive() -> None:
    docs = [
        "rust language guide and ownership",
        "python language guide and packaging",
        "bm25 sparse retrieval ranking",
        "hybrid search sparse dense fusion",
        "maxscore and wand pruning methods",
        "block max wand for efficient retrieval",
        "tokenization stemming stopwords",
        "query expansion rm3 rocchio bo1",
        "production retrieval pipelines",
        "retrievalx rust native engine",
    ]
    queries = ["rust retrieval", "wand pruning", "query expansion", "hybrid search"]
    _check_parity(docs, queries)


def test_parity_with_empty_documents() -> None:
    docs = [
        "",
        "rust language guide",
        "   ",
        "python fast",
    ]
    queries = ["rust", "python", "language"]
    _check_parity(docs, queries)


def test_parity_with_unicode_and_unusual_characters() -> None:
    config = BM25Config(
        tokenizer=TokenizerConfig(filters=[Filter.LOWERCASE]),
        retrieval=RetrievalStrategy.exhaustive_daat(),
    )
    docs = [
        "café résumé",
        "über cool naïveté",
        "emoji 😊 test 🚀 rocket",
        "中文 测试 文档",
        "हिन्दी भाषा परीक्षण",
        "emoji rocket fueled",
    ]
    index = BM25Index.from_documents(docs, config=config)

    results = index.search("😊", top_k=5)
    assert len(results) >= 1

    results = index.search("rocket", top_k=5)
    assert len(results) == 2


def test_parity_with_repetitive_content() -> None:
    docs = [
        "rust " * 5000 + "guide",
        "python " * 10 + "guide",
        "rust " * 3 + "engine",
        "bm25 retrieval ranking",
    ]
    queries = ["rust", "guide", "python", "retrieval"]
    _check_parity(docs, queries)


def test_parity_with_very_long_repetitive_terms() -> None:
    long_word = "x" * 120
    docs = [
        f"{long_word} other terms",
        f"{long_word} " * 50 + "guide",
        "other terms here",
    ]
    queries = [long_word, "other", "terms"]
    _check_parity(docs, queries)


def test_parity_with_empty_query() -> None:
    docs = ["rust language", "python fast"]
    exhaustive = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.exhaustive_daat()),
    )
    results = exhaustive.search("", top_k=5)
    assert results == []


def test_score_monotonicity_with_frequency() -> None:
    docs = [
        "rust",
        "rust rust",
        "rust rust rust rust",
    ]
    index = BM25Index.from_documents(
        docs,
        config=BM25Config(retrieval=RetrievalStrategy.exhaustive_daat()),
    )
    results = index.search("rust", top_k=3)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
    assert results[2].score > 0.0
