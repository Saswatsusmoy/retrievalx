from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from beir_utils import load_dataset
from rank_bm25 import BM25Okapi
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


@dataclass
class BenchmarkRow:
    engine: str
    dataset: str
    corpus_size: int
    query_count: int
    build_ms: float
    query_total_ms: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ndcg_at_k: float
    map_at_k: float
    recall_at_k: float
    precision_at_k: float
    mrr: float


@dataclass
class BenchmarkReport:
    generated_at_utc: str
    python: str
    platform: str
    retrievalx_version: str
    rank_bm25_version: str
    dataset: str
    split: str
    top_k: int
    corpus_size: int
    query_count: int
    rows: list[BenchmarkRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark retrievalx vs rank-bm25 on a real BEIR dataset."
    )
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/beir"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/rank_bm25_benchmark.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/rank_bm25_benchmarks.md"),
    )
    return parser.parse_args()


def make_retrievalx_config(strategy: object) -> BM25Config:
    return BM25Config(
        scoring=ScoringVariant.okapi(k1=1.2, b=0.75),
        retrieval=strategy,
        tokenizer=TokenizerConfig(
            tokenizer=Tokenizer.UNICODE,
            filters=[Filter.LOWERCASE, Filter.stopwords("en")],
            stemmer=Stemmer.snowball("en"),
        ),
    )


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def run_rank_bm25(
    doc_ids: list[str],
    corpus_texts: list[str],
    queries: list[str],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
    top_k: int,
) -> BenchmarkRow:
    tokenized = [tokenize(text) for text in corpus_texts]

    build_start = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    for query in queries[:20]:
        _ = bm25.get_scores(tokenize(query))

    ranked_lists: list[list[str]] = []
    latencies_ms: list[float] = []
    query_start = time.perf_counter()
    k = min(top_k, len(doc_ids))
    for query in queries:
        start = time.perf_counter()
        scores = bm25.get_scores(tokenize(query))
        idx = np.argpartition(scores, -k)[-k:]
        ranked_idx = idx[np.argsort(scores[idx])[::-1]]
        ranked_lists.append([doc_ids[int(i)] for i in ranked_idx])
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    query_total_ms = (time.perf_counter() - query_start) * 1000.0

    return _summarize_row(
        engine="rank_bm25_okapi",
        dataset="",
        build_ms=build_ms,
        query_total_ms=query_total_ms,
        latencies_ms=latencies_ms,
        ranked_lists=ranked_lists,
        query_ids=query_ids,
        qrels=qrels,
        top_k=top_k,
    )


def run_retrievalx(
    engine: str,
    strategy: object,
    docs: list[tuple[str, str]],
    queries: list[str],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
    top_k: int,
) -> BenchmarkRow:
    config = make_retrievalx_config(strategy)

    build_start = time.perf_counter()
    index = BM25Index.from_documents(docs, config=config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    for query in queries[:20]:
        _ = index.search(query, top_k=top_k)

    ranked_lists: list[list[str]] = []
    latencies_ms: list[float] = []
    query_start = time.perf_counter()
    for query in queries:
        start = time.perf_counter()
        hits = index.search(query, top_k=top_k)
        ranked_lists.append([hit.doc_id for hit in hits])
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    query_total_ms = (time.perf_counter() - query_start) * 1000.0

    return _summarize_row(
        engine=engine,
        dataset="",
        build_ms=build_ms,
        query_total_ms=query_total_ms,
        latencies_ms=latencies_ms,
        ranked_lists=ranked_lists,
        query_ids=query_ids,
        qrels=qrels,
        top_k=top_k,
    )


def _summarize_row(
    engine: str,
    dataset: str,
    build_ms: float,
    query_total_ms: float,
    latencies_ms: list[float],
    ranked_lists: list[list[str]],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    top_k: int,
) -> BenchmarkRow:
    ndcgs: list[float] = []
    maps: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []

    for query_id, ranked in zip(query_ids, ranked_lists):
        relevant = {
            doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0
        }
        ndcgs.append(ndcg_at_k(ranked, relevant, top_k))
        maps.append(average_precision_at_k(ranked, relevant, top_k))
        recalls.append(recall_at_k(ranked, relevant, top_k))
        precisions.append(precision_at_k(ranked, relevant, top_k))
        mrrs.append(mrr(ranked, relevant))

    query_count = len(query_ids)
    return BenchmarkRow(
        engine=engine,
        dataset=dataset,
        corpus_size=0,
        query_count=query_count,
        build_ms=build_ms,
        query_total_ms=query_total_ms,
        qps=(query_count / (query_total_ms / 1000.0)) if query_total_ms > 0 else 0.0,
        p50_ms=_percentile(latencies_ms, 0.50),
        p95_ms=_percentile(latencies_ms, 0.95),
        p99_ms=_percentile(latencies_ms, 0.99),
        ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
        map_at_k=statistics.mean(maps) if maps else 0.0,
        recall_at_k=statistics.mean(recalls) if recalls else 0.0,
        precision_at_k=statistics.mean(precisions) if precisions else 0.0,
        mrr=statistics.mean(mrrs) if mrrs else 0.0,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * p)
    return ordered[idx]


def render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# retrievalx vs rank-bm25 Benchmark (Real BEIR Data)",
        "",
        f"- Generated (UTC): `{report.generated_at_utc}`",
        f"- Dataset: `{report.dataset}` (split `{report.split}`)",
        f"- Python: `{report.python}`",
        f"- Platform: `{report.platform}`",
        f"- retrievalx: `{report.retrievalx_version}`",
        f"- rank-bm25: `{report.rank_bm25_version}`",
        f"- Corpus size: `{report.corpus_size}`",
        f"- Query count: `{report.query_count}`",
        f"- Top-k: `{report.top_k}`",
        "",
        "## Method",
        "",
        "- Real corpus/queries/qrels loaded from BEIR.",
        "- Same corpus and query set used for all engines.",
        "- `retrievalx_exhaustive_daat` for parity baseline.",
        "- `retrievalx_block_max_wand` for production-optimized sparse retrieval.",
        "- Effectiveness metrics computed against BEIR qrels.",
        "",
        "## Results",
        "",
        (
            "| Engine | Build (ms) | Query Total (ms) | QPS | P50 (ms) | P95 (ms) | "
            "P99 (ms) | nDCG@k | MAP@k | Recall@k | Precision@k | MRR |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.engine} | {row.build_ms:.2f} | {row.query_total_ms:.2f} | {row.qps:.2f} | "
            f"{row.p50_ms:.3f} | {row.p95_ms:.3f} | {row.p99_ms:.3f} | "
            f"{row.ndcg_at_k:.4f} | {row.map_at_k:.4f} | {row.recall_at_k:.4f} | "
            f"{row.precision_at_k:.4f} | {row.mrr:.4f} |"
        )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append(
        "```bash\npython examples/benchmark_retrievalx_vs_rank_bm25.py "
        "--dataset scifact --split test --top-k 10\n```"
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        name=args.dataset,
        cache_dir=args.cache_dir,
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )

    docs = []
    doc_ids: list[str] = []
    corpus_texts: list[str] = []
    for doc in dataset.corpus:
        text = f"{doc.title or ''}\n{doc.text}".strip()
        docs.append((doc.doc_id, text))
        doc_ids.append(doc.doc_id)
        corpus_texts.append(text)

    query_ids = [query.query_id for query in dataset.queries]
    query_texts = [query.text for query in dataset.queries]
    qrels = dataset.qrels

    rows = [
        run_rank_bm25(
            doc_ids=doc_ids,
            corpus_texts=corpus_texts,
            queries=query_texts,
            qrels=qrels,
            query_ids=query_ids,
            top_k=args.top_k,
        ),
        run_retrievalx(
            engine="retrievalx_exhaustive_daat",
            strategy=RetrievalStrategy.exhaustive_daat(),
            docs=docs,
            queries=query_texts,
            qrels=qrels,
            query_ids=query_ids,
            top_k=args.top_k,
        ),
        run_retrievalx(
            engine="retrievalx_block_max_wand",
            strategy=RetrievalStrategy.block_max_wand(top_k_budget=1000),
            docs=docs,
            queries=query_texts,
            qrels=qrels,
            query_ids=query_ids,
            top_k=args.top_k,
        ),
    ]

    for row in rows:
        row.dataset = dataset.name
        row.corpus_size = len(dataset.corpus)

    report = BenchmarkReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        retrievalx_version=metadata.version("retrievalx"),
        rank_bm25_version=metadata.version("rank-bm25"),
        dataset=dataset.name,
        split=args.split,
        top_k=args.top_k,
        corpus_size=len(dataset.corpus),
        query_count=len(dataset.queries),
        rows=rows,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                **asdict(report),
                "rows": [asdict(row) for row in report.rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    markdown = render_markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
