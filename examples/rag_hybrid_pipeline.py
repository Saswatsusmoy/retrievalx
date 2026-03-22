from retrievalx import BM25Index, linear_combination

docs = [
    ("d1", "retrieval augmented generation with bm25"),
    ("d2", "vector databases and dense retrieval"),
    ("d3", "hybrid search uses sparse and dense features"),
]

index = BM25Index.from_documents(docs)
bm25 = [(hit.doc_id, hit.score) for hit in index.search("hybrid retrieval", top_k=3)]

dense = [("d3", 0.9), ("d2", 0.85), ("d1", 0.30)]
fused = linear_combination(bm25, dense, alpha=0.6)

for item in fused:
    print(item)
