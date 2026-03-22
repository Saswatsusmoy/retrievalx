from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from beir_utils import load_dataset
from retrievalx import (
    BM25Config,
    BM25Index,
    Filter,
    RetrievalStrategy,
    ScoringVariant,
    Stemmer,
    Tokenizer,
    TokenizerConfig,
    average_precision_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrievalx benchmark on a real BEIR dataset."
    )
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/beir"),
        help="Dataset cache directory",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def make_config() -> BM25Config:
    return BM25Config(
        scoring=ScoringVariant.okapi(k1=1.2, b=0.75),
        retrieval=RetrievalStrategy.block_max_wand(top_k_budget=1000),
        tokenizer=TokenizerConfig(
            tokenizer=Tokenizer.UNICODE,
            filters=[Filter.LOWERCASE, Filter.stopwords("en")],
            stemmer=Stemmer.snowball("en"),
        ),
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        name=args.dataset,
        cache_dir=args.cache_dir,
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )

    if not dataset.queries:
        print(f"Dataset: {dataset.name} (split={args.split})")
        print(f"Corpus docs: {len(dataset.corpus)}")
        print("Queries: 0")
        print("No evaluable queries found for selected filters.")
        return

    docs = []
    for doc in dataset.corpus:
        combined = f"{doc.title or ''}\n{doc.text}".strip()
        docs.append((doc.doc_id, combined))

    config = make_config()
    build_start = time.perf_counter()
    index = BM25Index.from_documents(docs, config=config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    ndcg_scores: list[float] = []
    map_scores: list[float] = []
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    mrr_scores: list[float] = []
    latencies_ms: list[float] = []

    run_start = time.perf_counter()
    for query in dataset.queries:
        start = time.perf_counter()
        hits = index.search(query.text, top_k=args.top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        ranked = [hit.doc_id for hit in hits]
        relevant = {
            doc_id
            for doc_id, score in dataset.qrels.get(query.query_id, {}).items()
            if score > 0
        }

        ndcg_scores.append(ndcg_at_k(ranked, relevant, args.top_k))
        map_scores.append(average_precision_at_k(ranked, relevant, args.top_k))
        recall_scores.append(recall_at_k(ranked, relevant, args.top_k))
        precision_scores.append(precision_at_k(ranked, relevant, args.top_k))
        mrr_scores.append(mrr(ranked, relevant))
    total_ms = (time.perf_counter() - run_start) * 1000.0

    qps = len(dataset.queries) / (total_ms / 1000.0) if total_ms > 0 else 0.0
    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = _percentile(latencies_ms, 0.95)
    p99 = _percentile(latencies_ms, 0.99)

    print(f"Dataset: {dataset.name} (split={args.split})")
    print(f"Corpus docs: {len(dataset.corpus)}")
    print(f"Queries: {len(dataset.queries)}")
    print(f"Top-k: {args.top_k}")
    print("")
    print(f"Build time: {build_ms:.2f} ms")
    print(f"Query total: {total_ms:.2f} ms")
    print(f"QPS: {qps:.2f}")
    print(f"Latency p50/p95/p99: {p50:.3f}/{p95:.3f}/{p99:.3f} ms")
    print("")
    print("Effectiveness:")
    print(f"  nDCG@{args.top_k}: {statistics.mean(ndcg_scores):.4f}")
    print(f"  MAP@{args.top_k}: {statistics.mean(map_scores):.4f}")
    print(f"  Recall@{args.top_k}: {statistics.mean(recall_scores):.4f}")
    print(f"  Precision@{args.top_k}: {statistics.mean(precision_scores):.4f}")
    print(f"  MRR: {statistics.mean(mrr_scores):.4f}")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * p)
    return ordered[idx]


if __name__ == "__main__":
    main()
