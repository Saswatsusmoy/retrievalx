from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ._native import (
    native_cdf_normalize,
    native_linear_combination,
    native_min_max_normalize,
    native_rrf,
    native_z_score_normalize,
)


@dataclass
class FusionResult:
    doc_id: str
    score: float


def rrf(
    primary: Sequence[tuple[str, float]],
    secondary: Sequence[tuple[str, float]],
    k: int = 60,
) -> list[FusionResult]:
    scores = native_rrf(list(primary), list(secondary), k)
    return [
        FusionResult(doc_id=doc_id, score=score) for doc_id, score in scores
    ]


def linear_combination(
    primary: Sequence[tuple[str, float]],
    secondary: Sequence[tuple[str, float]],
    alpha: float = 0.5,
) -> list[FusionResult]:
    rows = native_linear_combination(list(primary), list(secondary), alpha)
    return [FusionResult(doc_id=doc_id, score=score) for doc_id, score in rows]


def min_max_normalize(values: Iterable[float]) -> list[float]:
    return list(native_min_max_normalize(list(values)))


def z_score_normalize(values: Iterable[float]) -> list[float]:
    return list(native_z_score_normalize(list(values)))


def cdf_normalize(values: Iterable[float]) -> list[float]:
    return list(native_cdf_normalize(list(values)))
