from __future__ import annotations

from collections.abc import Sequence

from ._native import (
    NativeLatencyProfiler,
    native_average_precision_at_k,
    native_mrr,
    native_ndcg_at_k,
    native_precision_at_k,
    native_recall_at_k,
)


class LatencyProfiler:
    def __init__(self) -> None:
        self._native = NativeLatencyProfiler()

    def record(self, total_ms: float) -> None:
        self._native.record(total_ms)

    def p50(self) -> float:
        return float(self._native.p50())

    def p95(self) -> float:
        return float(self._native.p95())

    def p99(self) -> float:
        return float(self._native.p99())

    def p999(self) -> float:
        return float(self._native.p999())

    def __len__(self) -> int:
        return int(self._native.len())


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(native_ndcg_at_k(list(ranked), list(relevant), k))


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(native_average_precision_at_k(list(ranked), list(relevant), k))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(native_recall_at_k(list(ranked), list(relevant), k))


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(native_precision_at_k(list(ranked), list(relevant), k))


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    return float(native_mrr(list(ranked), list(relevant)))
