from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

from ._native import NativeBM25Index
from .config import BM25Config
from .query import BooleanQuery, PhraseQuery, WeightedQuery

DocInput = Union[str, Tuple[str, str]]


@dataclass
class SearchHit:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, config: BM25Config | None = None):
        self._config = config or BM25Config()
        self._native = NativeBM25Index(self._config.to_json())

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[DocInput],
        config: BM25Config | None = None,
    ) -> BM25Index:
        instance = cls(config=config)
        instance.insert_batch(documents)
        return instance

    @classmethod
    def load(cls, path: str, mode: str = "in_memory") -> BM25Index:
        if mode == "in_memory":
            native = NativeBM25Index.load(path)
        elif mode == "mmap":
            native = NativeBM25Index.load_mmap(path)
        else:
            raise ValueError("mode must be one of: in_memory, mmap")

        return cls._from_native(native)

    @classmethod
    def load_with_wal(
        cls,
        path: str,
        wal_path: str,
        mode: str = "in_memory",
    ) -> BM25Index:
        native = NativeBM25Index.load_with_wal(path, wal_path, mode)
        return cls._from_native(native)

    @classmethod
    def _from_native(cls, native: NativeBM25Index) -> BM25Index:
        config = BM25Config.from_mapping(json.loads(native.config_json()))
        instance = cls.__new__(cls)
        instance._config = config
        instance._native = native
        return instance

    def save(self, path: str) -> None:
        self._native.save(path)

    def save_metadata(self, path: str) -> None:
        self._native.save_metadata(path)

    def insert_batch(self, documents: Iterable[DocInput]) -> None:
        plain_docs: list[str] = []
        id_docs: list[tuple[str, str]] = []

        for doc in documents:
            if isinstance(doc, tuple):
                id_docs.append(doc)
            else:
                plain_docs.append(doc)

        if plain_docs:
            self._native.insert_batch(plain_docs)
        if id_docs:
            self._native.insert_documents(id_docs)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        rows = self._native.search(query, top_k)
        return [SearchHit(doc_id=doc_id, score=score) for doc_id, score in rows]

    def search_weighted(self, query: WeightedQuery, top_k: int = 10) -> list[SearchHit]:
        rows = self._native.search_weighted(query.terms(), top_k)
        return [SearchHit(doc_id=doc_id, score=score) for doc_id, score in rows]

    def search_boolean(self, query: BooleanQuery, top_k: int = 10) -> list[SearchHit]:
        rows = self._native.search_boolean(query.must, query.should, query.must_not, top_k)
        return [SearchHit(doc_id=doc_id, score=score) for doc_id, score in rows]

    def search_phrase(self, query: PhraseQuery, top_k: int = 10) -> list[SearchHit]:
        rows = self._native.search_phrase(list(query.terms), int(query.window), top_k)
        return [SearchHit(doc_id=doc_id, score=score) for doc_id, score in rows]

    def delete(self, external_id: str) -> None:
        self._native.delete_by_external_id(external_id)

    def compact(self) -> None:
        self._native.compact()

    def compact_and_flush(self, index_path: str) -> None:
        self._native.compact_and_flush(index_path)

    def enable_wal(self, wal_path: str) -> None:
        self._native.enable_wal(wal_path)

    def disable_wal(self) -> None:
        self._native.disable_wal()

    def replay_wal(self, wal_path: str) -> int:
        return int(self._native.replay_wal(wal_path))

    def postings(self, term: str) -> list[tuple[int, int]]:
        return list(self._native.postings(term))

    def doc_length(self, doc_id: int) -> Optional[int]:
        return self._native.doc_length(doc_id)

    def sparse_vector_for_query(self, query: str) -> list[tuple[int, float]]:
        return list(self._native.sparse_vector_for_query(query))

    def sparse_vector_for_document(self, doc_id: int) -> list[tuple[int, float]]:
        return list(self._native.sparse_vector_for_document(doc_id))

    @property
    def avgdl(self) -> float:
        return float(self._native.avgdl())

    @property
    def vocabulary(self) -> dict[str, dict[str, int]]:
        vocab = {}
        for term, term_id, doc_freq, collection_freq, max_tf in self._native.vocabulary():
            vocab[term] = {
                "term_id": int(term_id),
                "doc_freq": int(doc_freq),
                "collection_freq": int(collection_freq),
                "max_tf": int(max_tf),
            }
        return vocab

    def stats(self) -> dict[str, object]:
        return dict(self._native.stats_dict())

    def __len__(self) -> int:
        value = self.stats().get("num_live_docs", 0)
        return int(cast(Any, value))

    def __iter__(self) -> Iterator[str]:
        yield from self.vocabulary
