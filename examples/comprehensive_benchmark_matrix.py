from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from beir_utils import load_dataset
from rank_bm25 import BM25L, BM25Okapi, BM25Plus
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
    system: str
    scoring: str
    retrieval: str
    tokenizer: str
    dataset: str
    split: str
    corpus_size: int
    query_count: int
    top_k: int
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
class ComparableDelta:
    scoring: str
    retrievalx_qps: float
    rank_bm25_qps: float
    qps_speedup_x: float
    retrievalx_build_ms: float
    rank_bm25_build_ms: float
    build_time_ratio_x: float
    ndcg_delta: float
    map_delta: float
    recall_delta: float
    precision_delta: float
    mrr_delta: float


@dataclass
class ComprehensiveReport:
    generated_at_utc: str
    python: str
    platform: str
    retrievalx_version: str
    rank_bm25_version: str
    dataset: str
    split: str
    top_k: int
    retrievalx_top_k_budget: int
    tokenizer: str
    corpus_size: int
    query_count: int
    retrievalx_rows: list[BenchmarkRow]
    rank_bm25_rows: list[BenchmarkRow]
    comparable_deltas: list[ComparableDelta]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark matrix for retrievalx vs rank-bm25."
    )
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--retrievalx-top-k-budget",
        type=int,
        default=0,
        help=(
            "WAND/BlockMaxWAND score budget. Use 0 for unbounded exact mode; "
            "values > 0 enable approximate early stopping."
        ),
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/beir"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/comprehensive_benchmark_matrix.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/comprehensive_benchmark_matrix.csv"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/benchmarks.md"),
    )
    return parser.parse_args()


def naive_tokenize(text: str) -> list[str]:
    return text.lower().split()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * p)
    return ordered[idx]


def summarize_row(
    *,
    system: str,
    scoring: str,
    retrieval: str,
    tokenizer: str,
    dataset: str,
    split: str,
    top_k: int,
    corpus_size: int,
    query_count: int,
    build_ms: float,
    query_total_ms: float,
    latencies_ms: list[float],
    ranked_lists: list[list[str]],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
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

    qps = (query_count / (query_total_ms / 1000.0)) if query_total_ms > 0 else 0.0
    return BenchmarkRow(
        system=system,
        scoring=scoring,
        retrieval=retrieval,
        tokenizer=tokenizer,
        dataset=dataset,
        split=split,
        corpus_size=corpus_size,
        query_count=query_count,
        top_k=top_k,
        build_ms=build_ms,
        query_total_ms=query_total_ms,
        qps=qps,
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
        ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
        map_at_k=statistics.mean(maps) if maps else 0.0,
        recall_at_k=statistics.mean(recalls) if recalls else 0.0,
        precision_at_k=statistics.mean(precisions) if precisions else 0.0,
        mrr=statistics.mean(mrrs) if mrrs else 0.0,
    )


def retrievalx_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        tokenizer=Tokenizer.WHITESPACE,
        filters=[Filter.LOWERCASE],
        stemmer=Stemmer.NOOP,
    )


def retrievalx_scoring_variants() -> list[tuple[str, Any]]:
    return [
        ("Okapi(k1=1.2,b=0.75)", ScoringVariant.okapi(k1=1.2, b=0.75)),
        ("Plus(k1=1.2,b=0.75,delta=0.5)", ScoringVariant.plus(k1=1.2, b=0.75, delta=0.5)),
        ("L(k1=1.2,b=0.75,c=1.0)", ScoringVariant.bm25_l(k1=1.2, b=0.75, c=1.0)),
        ("Adpt(b=0.75)", ScoringVariant.adpt(b=0.75)),
        ("F(k1=1.2,b=0.75,fields=[])", ScoringVariant.bm25_f(k1=1.2, b=0.75, fields=[])),
        (
            "T(default_k1=1.2,b=0.75,term_k1={})",
            ScoringVariant.bm25_t(default_k1=1.2, b=0.75, term_k1={}),
        ),
        ("Atire(k1=1.2,b=0.75)", ScoringVariant.atire(k1=1.2, b=0.75)),
        ("TfIdf", ScoringVariant.tfidf()),
    ]


def retrievalx_retrieval_strategies(top_k_budget: int) -> list[tuple[str, Any]]:
    return [
        ("ExhaustiveDAAT", RetrievalStrategy.exhaustive_daat()),
        ("ExhaustiveTAAT", RetrievalStrategy.exhaustive_taat()),
        (
            f"Wand(top_k_budget={top_k_budget})",
            RetrievalStrategy.wand(top_k_budget=top_k_budget),
        ),
        (
            f"BlockMaxWand(top_k_budget={top_k_budget})",
            RetrievalStrategy.block_max_wand(top_k_budget=top_k_budget),
        ),
        ("MaxScore", RetrievalStrategy.max_score()),
    ]


def rank_bm25_variants() -> list[tuple[str, type, dict[str, float]]]:
    return [
        ("Okapi(k1=1.2,b=0.75)", BM25Okapi, {"k1": 1.2, "b": 0.75, "epsilon": 0.25}),
        ("Plus(k1=1.2,b=0.75,delta=0.5)", BM25Plus, {"k1": 1.2, "b": 0.75, "delta": 0.5}),
        ("L(k1=1.2,b=0.75,delta=0.5)", BM25L, {"k1": 1.2, "b": 0.75, "delta": 0.5}),
    ]


def run_retrievalx_matrix(
    *,
    dataset_name: str,
    split: str,
    top_k: int,
    retrievalx_top_k_budget: int,
    docs: list[tuple[str, str]],
    query_texts: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    tokenizer = retrievalx_tokenizer_config()

    for scoring_name, scoring in retrievalx_scoring_variants():
        for retrieval_name, retrieval in retrievalx_retrieval_strategies(retrievalx_top_k_budget):
            config = BM25Config(
                scoring=scoring,
                retrieval=retrieval,
                tokenizer=tokenizer,
            )

            build_start = time.perf_counter()
            index = BM25Index.from_documents(docs, config=config)
            build_ms = (time.perf_counter() - build_start) * 1000.0

            for query in query_texts[:20]:
                _ = index.search(query, top_k=top_k)

            ranked_lists: list[list[str]] = []
            latencies_ms: list[float] = []
            query_start = time.perf_counter()
            for query in query_texts:
                start = time.perf_counter()
                hits = index.search(query, top_k=top_k)
                ranked_lists.append([hit.doc_id for hit in hits])
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
            query_total_ms = (time.perf_counter() - query_start) * 1000.0

            rows.append(
                summarize_row(
                    system="retrievalx",
                    scoring=scoring_name,
                    retrieval=retrieval_name,
                    tokenizer="lower+whitespace (no stemming, no stopwords)",
                    dataset=dataset_name,
                    split=split,
                    top_k=top_k,
                    corpus_size=len(docs),
                    query_count=len(query_ids),
                    build_ms=build_ms,
                    query_total_ms=query_total_ms,
                    latencies_ms=latencies_ms,
                    ranked_lists=ranked_lists,
                    query_ids=query_ids,
                    qrels=qrels,
                )
            )
            print(
                f"[retrievalx] scoring={scoring_name} retrieval={retrieval_name} "
                f"build={build_ms:.2f}ms qps={rows[-1].qps:.2f}"
            )

    return rows


def run_rank_bm25_matrix(
    *,
    dataset_name: str,
    split: str,
    top_k: int,
    doc_ids: list[str],
    corpus_texts: list[str],
    query_texts: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    tokenized = [naive_tokenize(text) for text in corpus_texts]
    k = min(top_k, len(doc_ids))

    for scoring_name, cls, kwargs in rank_bm25_variants():
        build_start = time.perf_counter()
        bm25 = cls(tokenized, **kwargs)
        build_ms = (time.perf_counter() - build_start) * 1000.0

        for query in query_texts[:20]:
            _ = bm25.get_scores(naive_tokenize(query))

        ranked_lists: list[list[str]] = []
        latencies_ms: list[float] = []
        query_start = time.perf_counter()
        for query in query_texts:
            start = time.perf_counter()
            scores = bm25.get_scores(naive_tokenize(query))
            idx = np.argpartition(scores, -k)[-k:]
            ranked_idx = idx[np.argsort(scores[idx])[::-1]]
            ranked_lists.append([doc_ids[int(i)] for i in ranked_idx])
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        query_total_ms = (time.perf_counter() - query_start) * 1000.0

        rows.append(
            summarize_row(
                system="rank-bm25",
                scoring=scoring_name,
                retrieval="Exhaustive",
                tokenizer="lower+whitespace (no stemming, no stopwords)",
                dataset=dataset_name,
                split=split,
                top_k=top_k,
                corpus_size=len(doc_ids),
                query_count=len(query_ids),
                build_ms=build_ms,
                query_total_ms=query_total_ms,
                latencies_ms=latencies_ms,
                ranked_lists=ranked_lists,
                query_ids=query_ids,
                qrels=qrels,
            )
        )
        print(
            f"[rank-bm25] scoring={scoring_name} build={build_ms:.2f}ms "
            f"qps={rows[-1].qps:.2f}"
        )

    return rows


def compute_comparable_deltas(
    retrievalx_rows: list[BenchmarkRow], rank_rows: list[BenchmarkRow]
) -> list[ComparableDelta]:
    comparable_pairs = [
        ("Okapi(k1=1.2,b=0.75)", "Okapi(k1=1.2,b=0.75)"),
        ("Plus(k1=1.2,b=0.75,delta=0.5)", "Plus(k1=1.2,b=0.75,delta=0.5)"),
        ("L(k1=1.2,b=0.75,c=1.0)", "L(k1=1.2,b=0.75,delta=0.5)"),
    ]

    rx_map = {
        (row.scoring, row.retrieval): row
        for row in retrievalx_rows
        if row.retrieval == "ExhaustiveDAAT"
    }
    rank_map = {row.scoring: row for row in rank_rows}

    deltas: list[ComparableDelta] = []
    for rx_scoring, rank_scoring in comparable_pairs:
        rx_row = rx_map.get((rx_scoring, "ExhaustiveDAAT"))
        rank_row = rank_map.get(rank_scoring)
        if rx_row is None or rank_row is None:
            continue

        deltas.append(
            ComparableDelta(
                scoring=rx_scoring,
                retrievalx_qps=rx_row.qps,
                rank_bm25_qps=rank_row.qps,
                qps_speedup_x=(rx_row.qps / rank_row.qps) if rank_row.qps > 0 else 0.0,
                retrievalx_build_ms=rx_row.build_ms,
                rank_bm25_build_ms=rank_row.build_ms,
                build_time_ratio_x=(
                    rx_row.build_ms / rank_row.build_ms if rank_row.build_ms > 0 else 0.0
                ),
                ndcg_delta=rx_row.ndcg_at_k - rank_row.ndcg_at_k,
                map_delta=rx_row.map_at_k - rank_row.map_at_k,
                recall_delta=rx_row.recall_at_k - rank_row.recall_at_k,
                precision_delta=rx_row.precision_at_k - rank_row.precision_at_k,
                mrr_delta=rx_row.mrr - rank_row.mrr,
            )
        )
    return deltas


def render_table(rows: list[BenchmarkRow]) -> list[str]:
    lines = [
        "| System | Scoring | Retrieval | Build (ms) | Query Total (ms) | QPS | "
        "P50 (ms) | P95 (ms) | P99 (ms) | nDCG@k | MAP@k | Recall@k | "
        "Precision@k | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.system} | {row.scoring} | {row.retrieval} | "
            f"{row.build_ms:.2f} | {row.query_total_ms:.2f} | {row.qps:.2f} | "
            f"{row.p50_ms:.3f} | {row.p95_ms:.3f} | {row.p99_ms:.3f} | "
            f"{row.ndcg_at_k:.4f} | {row.map_at_k:.4f} | {row.recall_at_k:.4f} | "
            f"{row.precision_at_k:.4f} | {row.mrr:.4f} |"
        )
    return lines


def render_delta_table(deltas: list[ComparableDelta]) -> list[str]:
    lines = [
        "| Scoring | retrievalx QPS | rank-bm25 QPS | Speedup (x) | retrievalx Build (ms) | "
        "rank-bm25 Build (ms) | Build Ratio (x) | Δ nDCG@k | Δ MAP@k | "
        "Δ Recall@k | Δ Precision@k | Δ MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            "| "
            f"{row.scoring} | {row.retrievalx_qps:.2f} | {row.rank_bm25_qps:.2f} | "
            f"{row.qps_speedup_x:.2f} | {row.retrievalx_build_ms:.2f} | "
            f"{row.rank_bm25_build_ms:.2f} | {row.build_time_ratio_x:.2f} | "
            f"{row.ndcg_delta:+.4f} | {row.map_delta:+.4f} | {row.recall_delta:+.4f} | "
            f"{row.precision_delta:+.4f} | {row.mrr_delta:+.4f} |"
        )
    return lines


def render_markdown(report: ComprehensiveReport) -> str:
    retrievalx_rows_sorted = sorted(
        report.retrievalx_rows, key=lambda row: (row.scoring, row.retrieval)
    )
    rank_rows_sorted = sorted(report.rank_bm25_rows, key=lambda row: row.scoring)
    all_rows_sorted = retrievalx_rows_sorted + rank_rows_sorted

    fastest_query = max(all_rows_sorted, key=lambda row: row.qps)
    strongest_quality = max(all_rows_sorted, key=lambda row: row.ndcg_at_k)
    fastest_build = min(all_rows_sorted, key=lambda row: row.build_ms)

    lines = [
        "# Comprehensive Benchmark Matrix",
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
        f"- retrievalx WAND/BMW top_k_budget: `{report.retrievalx_top_k_budget}`",
        f"- Tokenization for parity: `{report.tokenizer}`",
        "",
        "## Method",
        "",
        "- Real corpus/queries/qrels loaded from BEIR.",
        "- Identical corpus, query set, and tokenization for all systems.",
        "- retrievalx matrix: all 8 scoring variants × all 5 retrieval strategies (40 runs).",
        "- WAND/BlockMaxWAND are exact when `top_k_budget=0`; approximate when `> 0`.",
        "- rank-bm25 matrix: all available variants (Okapi, Plus, L) with exhaustive scoring.",
        "- Effectiveness metrics computed against BEIR qrels.",
        "",
        "## Top Highlights",
        "",
        "- Fastest query throughput:",
        "  - "
        f"`{fastest_query.system}` | `{fastest_query.scoring}` | "
        f"`{fastest_query.retrieval}` | `{fastest_query.qps:.2f}` QPS",
        "- Strongest effectiveness (nDCG@k):",
        "  - "
        f"`{strongest_quality.system}` | `{strongest_quality.scoring}` | "
        f"`{strongest_quality.retrieval}` | `{strongest_quality.ndcg_at_k:.4f}`",
        "- Fastest index build:",
        "  - "
        f"`{fastest_build.system}` | `{fastest_build.scoring}` | "
        f"`{fastest_build.retrieval}` | `{fastest_build.build_ms:.2f} ms`",
        "",
        "## rank-bm25 Variant Results",
        "",
        *render_table(rank_rows_sorted),
        "",
        "## retrievalx Full Configuration Matrix",
        "",
        *render_table(retrievalx_rows_sorted),
        "",
        "## Like-for-Like Deltas (retrievalx ExhaustiveDAAT vs rank-bm25)",
        "",
        "*Notes:*",
        "- Okapi and Plus are directly comparable.",
        "- BM25L formulas differ between libraries (`retrievalx` uses `c`; "
        "`rank-bm25` uses `delta`).",
        "",
        *render_delta_table(report.comparable_deltas),
        "",
        "## Reproduce",
        "",
        "```bash",
        "python examples/comprehensive_benchmark_matrix.py --dataset scifact "
        f"--split test --top-k 10 --retrievalx-top-k-budget {report.retrievalx_top_k_budget}",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    dataset = load_dataset(
        name=args.dataset,
        cache_dir=args.cache_dir,
        split=args.split,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )

    docs: list[tuple[str, str]] = []
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

    retrievalx_rows = run_retrievalx_matrix(
        dataset_name=dataset.name,
        split=args.split,
        top_k=args.top_k,
        retrievalx_top_k_budget=args.retrievalx_top_k_budget,
        docs=docs,
        query_texts=query_texts,
        query_ids=query_ids,
        qrels=qrels,
    )
    rank_rows = run_rank_bm25_matrix(
        dataset_name=dataset.name,
        split=args.split,
        top_k=args.top_k,
        doc_ids=doc_ids,
        corpus_texts=corpus_texts,
        query_texts=query_texts,
        query_ids=query_ids,
        qrels=qrels,
    )
    deltas = compute_comparable_deltas(retrievalx_rows, rank_rows)

    report = ComprehensiveReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        retrievalx_version=metadata.version("retrievalx"),
        rank_bm25_version=metadata.version("rank-bm25"),
        dataset=dataset.name,
        split=args.split,
        top_k=args.top_k,
        retrievalx_top_k_budget=args.retrievalx_top_k_budget,
        tokenizer="lower+whitespace (no stemming, no stopwords)",
        corpus_size=len(docs),
        query_count=len(query_ids),
        retrievalx_rows=retrievalx_rows,
        rank_bm25_rows=rank_rows,
        comparable_deltas=deltas,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                **asdict(report),
                "retrievalx_rows": [asdict(row) for row in report.retrievalx_rows],
                "rank_bm25_rows": [asdict(row) for row in report.rank_bm25_rows],
                "comparable_deltas": [asdict(row) for row in report.comparable_deltas],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(args.output_csv, retrievalx_rows + rank_rows)

    markdown = render_markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
