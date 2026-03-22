from retrievalx import BM25Index, linear_combination, rrf

chunks = [
    ("c1", "SOC2 policy states production secrets rotate every 90 days."),
    ("c2", "Incident response runbook includes severity matrix and paging policy."),
    ("c3", "RBAC standard requires least privilege for service identities."),
    ("c4", "Data retention policy sets 30 day hot storage and 365 day archive."),
]

index = BM25Index.from_documents(chunks)

query = "least privilege identity policy"
bm25_hits = index.search(query, top_k=4)
bm25 = [(hit.doc_id, hit.score) for hit in bm25_hits]

# Simulated dense reranker output from a vector model.
dense = [("c3", 0.94), ("c2", 0.78), ("c1", 0.74), ("c4", 0.62)]

print("RRF fusion:")
for row in rrf(bm25, dense, k=60):
    print(f"  {row.doc_id}: {row.score:.4f}")

print("\nLinear fusion:")
for row in linear_combination(bm25, dense, alpha=0.65):
    print(f"  {row.doc_id}: {row.score:.4f}")
