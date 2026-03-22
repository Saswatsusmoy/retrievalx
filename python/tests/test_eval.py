from retrievalx import (
    LatencyProfiler,
    average_precision_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_eval_metrics_and_profiler() -> None:
    ranked = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}

    assert ndcg_at_k(ranked, relevant, 3) > 0.0
    assert average_precision_at_k(ranked, relevant, 3) > 0.0
    assert recall_at_k(ranked, relevant, 2) > 0.0
    assert precision_at_k(ranked, relevant, 2) > 0.0
    assert mrr(ranked, relevant) == 1.0

    profiler = LatencyProfiler()
    profiler.record(1.0)
    profiler.record(2.5)
    profiler.record(3.0)
    assert len(profiler) == 3
    assert profiler.p50() >= 1.0
