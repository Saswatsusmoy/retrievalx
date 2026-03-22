from retrievalx import BM25Config, BM25Index, Filter, Tokenizer, TokenizerConfig

logs = [
    (
        "LOG-9001",
        "2026-03-21T08:12:00Z auth-service WARN failed login from 10.1.2.7 user=admin",
    ),
    (
        "LOG-9002",
        "2026-03-21T08:13:10Z waf INFO blocked sql injection payload path=/api/login",
    ),
    (
        "LOG-9003",
        "2026-03-21T08:14:44Z gateway ERROR token validation failure issuer mismatch",
    ),
    (
        "LOG-9004",
        "2026-03-21T08:16:02Z auth-service WARN failed login from 10.1.2.7 user=root",
    ),
    (
        "LOG-9005",
        "2026-03-21T08:17:09Z iam INFO privileged role assignment requested by svc-ci",
    ),
]

config = BM25Config(
    tokenizer=TokenizerConfig(
        tokenizer=Tokenizer.regex(r"[A-Za-z0-9._:/-]+"),
        filters=[Filter.LOWERCASE, Filter.duplicate_cap(max_per_doc=8)],
    )
)

index = BM25Index.from_documents(logs, config=config)
hits = index.search("failed login auth-service 10.1.2.7", top_k=3)

print("Security log hunt results:")
for hit in hits:
    print(f"  {hit.doc_id}: {hit.score:.4f}")
