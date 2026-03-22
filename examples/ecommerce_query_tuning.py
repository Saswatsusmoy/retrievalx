from retrievalx import BM25Config, BM25Index, ScoringVariant, WeightedQuery

catalog = [
    (
        "SKU-100",
        "Wireless noise cancelling headphones with 40 hour battery and USB C fast charge",
    ),
    (
        "SKU-101",
        "Over ear studio headphones open back reference tuning for mixing and mastering",
    ),
    (
        "SKU-102",
        "Bluetooth earbuds with active noise cancellation and transparency mode",
    ),
    (
        "SKU-103",
        "Portable DAC headphone amplifier with balanced output for hi fi listening",
    ),
]

index = BM25Index.from_documents(
    catalog,
    config=BM25Config(scoring=ScoringVariant.okapi(k1=1.2, b=0.75)),
)

query = WeightedQuery(
    weights={
        "headphones": 2.0,
        "noise": 1.5,
        "bluetooth": 0.7,
        "studio": 0.4,
    }
)

hits = index.search_weighted(query, top_k=4)

print("Weighted retrieval for e-commerce ranking:")
for hit in hits:
    print(f"  {hit.doc_id}: {hit.score:.4f}")
