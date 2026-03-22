from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class WeightedQuery:
    weights: dict[str, float] = field(default_factory=dict)

    def terms(self) -> list[tuple[str, float]]:
        return list(self.weights.items())


@dataclass
class BooleanQuery:
    must: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)


@dataclass
class PhraseQuery:
    terms: Iterable[str]
    window: int = 1

    def __post_init__(self) -> None:
        self.terms = list(self.terms)
