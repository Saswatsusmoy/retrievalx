from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ScoringVariant:
    @staticmethod
    def okapi(k1: float = 1.2, b: float = 0.75) -> dict[str, Any]:
        return {"Okapi": {"k1": k1, "b": b}}

    @staticmethod
    def plus(k1: float = 1.2, b: float = 0.75, delta: float = 0.5) -> dict[str, Any]:
        return {"Plus": {"k1": k1, "b": b, "delta": delta}}

    @staticmethod
    def bm25_l(k1: float = 1.2, b: float = 0.75, c: float = 1.0) -> dict[str, Any]:
        return {"L": {"k1": k1, "b": b, "c": c}}

    @staticmethod
    def adpt(b: float = 0.75) -> dict[str, Any]:
        return {"Adpt": {"b": b}}

    @staticmethod
    def bm25_f(
        k1: float = 1.2,
        b: float = 0.75,
        fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {"F": {"k1": k1, "b": b, "fields": fields or []}}

    @staticmethod
    def bm25_t(
        default_k1: float = 1.2,
        b: float = 0.75,
        term_k1: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        return {"T": {"default_k1": default_k1, "b": b, "term_k1": dict(term_k1 or {})}}

    @staticmethod
    def atire(k1: float = 1.2, b: float = 0.75) -> dict[str, Any]:
        return {"Atire": {"k1": k1, "b": b}}

    @staticmethod
    def tfidf() -> str:
        return "TfIdf"


class RetrievalStrategy:
    @staticmethod
    def exhaustive_daat() -> str:
        return "ExhaustiveDAAT"

    @staticmethod
    def exhaustive_taat() -> str:
        return "ExhaustiveTAAT"

    @staticmethod
    def wand(top_k_budget: int = 0) -> dict[str, Any]:
        return {"Wand": {"top_k_budget": top_k_budget}}

    @staticmethod
    def block_max_wand(top_k_budget: int = 0) -> dict[str, Any]:
        return {"BlockMaxWand": {"top_k_budget": top_k_budget}}

    @staticmethod
    def max_score() -> str:
        return "MaxScore"


class Tokenizer:
    WHITESPACE = "Whitespace"
    UNICODE = "UnicodeWord"

    @staticmethod
    def regex(pattern: str) -> dict[str, Any]:
        return {"Regex": {"pattern": pattern}}

    @staticmethod
    def ngram(min_n: int = 3, max_n: int = 3, mode: str = "Character") -> dict[str, Any]:
        return {"Ngram": {"min_n": min_n, "max_n": max_n, "mode": mode}}


class Filter:
    LOWERCASE = "Lowercase"

    @staticmethod
    def stopwords(lang: str = "en") -> dict[str, Any]:
        return {"Stopwords": {"lang": lang}}

    @staticmethod
    def length(min_len: int = 1, max_len: int = 128) -> dict[str, Any]:
        return {"Length": {"min_len": min_len, "max_len": max_len}}

    @staticmethod
    def duplicate_cap(max_per_doc: int = 32) -> dict[str, Any]:
        return {"DuplicateCap": {"max_per_doc": max_per_doc}}


class Stemmer:
    NOOP = "NoOp"
    PORTER = "Porter"

    @staticmethod
    def snowball(lang: str = "en") -> dict[str, Any]:
        return {"Snowball": {"lang": lang}}


@dataclass
class FieldConfig:
    name: str
    weight: float = 1.0
    b: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "b": self.b}


@dataclass
class ExpansionConfig:
    method: str = "RM3"
    num_feedback_docs: int = 10
    num_expansion_terms: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "num_feedback_docs": self.num_feedback_docs,
            "num_expansion_terms": self.num_expansion_terms,
        }


@dataclass
class FusionConfig:
    method: str = "ReciprocalRankFusion"
    alpha: float = 0.5
    normalizer: str | None = "Cdf"
    rrf_k: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "normalizer": self.normalizer,
            "rrf_k": self.rrf_k,
        }


@dataclass
class PersistConfig:
    load_mode: str = "in_memory"
    enable_wal: bool = False
    compression: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_mode": self.load_mode,
            "enable_wal": self.enable_wal,
            "compression": self.compression,
        }


@dataclass
class TokenizerConfig:
    tokenizer: Any = Tokenizer.WHITESPACE
    filters: list[Any] = field(default_factory=lambda: [Filter.LOWERCASE, Filter.stopwords("en")])
    stemmer: Any = Stemmer.NOOP
    min_token_len: int = 1
    max_token_len: int = 128

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer": self.tokenizer,
            "filters": self.filters,
            "stemmer": self.stemmer,
            "min_token_len": self.min_token_len,
            "max_token_len": self.max_token_len,
        }


@dataclass
class BM25Config:
    scoring: Any = field(default_factory=ScoringVariant.okapi)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    retrieval: Any = field(default_factory=RetrievalStrategy.block_max_wand)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring": self.scoring,
            "tokenizer": self.tokenizer.to_dict(),
            "retrieval": self.retrieval,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> BM25Config:
        tokenizer_map = dict(mapping.get("tokenizer", {}))
        tokenizer = TokenizerConfig(
            tokenizer=tokenizer_map.get("tokenizer", Tokenizer.WHITESPACE),
            filters=list(tokenizer_map.get("filters", [Filter.LOWERCASE])),
            stemmer=tokenizer_map.get("stemmer", Stemmer.NOOP),
            min_token_len=tokenizer_map.get("min_token_len", 1),
            max_token_len=tokenizer_map.get("max_token_len", 128),
        )
        return cls(
            scoring=mapping.get("scoring", ScoringVariant.okapi()),
            tokenizer=tokenizer,
            retrieval=mapping.get("retrieval", RetrievalStrategy.block_max_wand()),
        )
