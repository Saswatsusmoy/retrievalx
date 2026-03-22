from retrievalx import BM25Config, BM25Index, RetrievalStrategy


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
        gt = {hit.doc_id for hit in exhaustive.search(query, top_k=5)}
        for candidate in (wand, blockmax, maxscore):
            got = {hit.doc_id for hit in candidate.search(query, top_k=5)}
            assert got == gt
