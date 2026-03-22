from __future__ import annotations

import time

from retrievalx import BM25Config, BM25Index, Filter, Tokenizer, TokenizerConfig

LOGS = [
    (
        "log-1",
        "2026-03-21T08:12:00Z auth-service WARN failed_login user=admin src_ip=10.1.2.7",
    ),
    (
        "log-2",
        "2026-03-21T08:13:10Z gateway ERROR token_validation_failed issuer_mismatch kid=rot-71",
    ),
    (
        "log-3",
        "2026-03-21T08:14:44Z waf INFO blocked_sql_injection path=/api/login rule=942100",
    ),
    (
        "log-4",
        "2026-03-21T08:15:31Z auth-service WARN failed_login user=root src_ip=10.1.2.7",
    ),
    (
        "log-5",
        "2026-03-21T08:16:09Z iam INFO privileged_role_assignment actor=svc-ci role=prod-admin",
    ),
]


def run_search(label: str, config: BM25Config, query: str) -> None:
    build_start = time.perf_counter()
    index = BM25Index.from_documents(LOGS, config=config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    query_start = time.perf_counter()
    hits = index.search(query, top_k=3)
    query_ms = (time.perf_counter() - query_start) * 1000.0

    print(f"{label}")
    print(f"  build_ms={build_ms:.2f} query_ms={query_ms:.3f}")
    for hit in hits:
        print(f"  {hit.doc_id}: {hit.score:.4f}")
    print("")


def main() -> None:
    query = "failed_login auth-service src_ip 10.1.2.7"

    baseline = BM25Config()
    regex_config = BM25Config(
        tokenizer=TokenizerConfig(
            tokenizer=Tokenizer.regex(r"[A-Za-z0-9._:/=-]+"),
            filters=[
                Filter.LOWERCASE,
                Filter.duplicate_cap(max_per_doc=6),
                Filter.length(min_len=2, max_len=80),
            ],
        )
    )

    print(f"Query: {query}\n")
    run_search("Default tokenizer", baseline, query)
    run_search("Custom regex tokenizer", regex_config, query)


if __name__ == "__main__":
    main()
