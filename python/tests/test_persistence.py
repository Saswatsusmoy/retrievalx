from pathlib import Path

from retrievalx import BM25Index


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    index = BM25Index.from_documents([("a", "rust fast"), ("b", "python fast")])
    index_path = tmp_path / "index.rtx"
    meta_path = tmp_path / "index.meta.json"

    index.save(str(index_path))
    index.save_metadata(str(meta_path))

    loaded = BM25Index.load(str(index_path))
    results = loaded.search("rust", top_k=1)
    assert results[0].doc_id == "a"
    assert meta_path.exists()

    mapped = BM25Index.load(str(index_path), mode="mmap")
    mapped_results = mapped.search("rust", top_k=1)
    assert mapped_results[0].doc_id == "a"


def test_wal_replay_and_sparse_vectors(tmp_path: Path) -> None:
    index = BM25Index.from_documents([("a", "rust fast"), ("b", "python fast")])
    index_path = tmp_path / "index.rtx"
    wal_path = tmp_path / "index.wal"
    index.save(str(index_path))

    index.enable_wal(str(wal_path))
    index.insert_batch([("c", "rust wal replay")])
    index.delete("b")

    restored = BM25Index.load_with_wal(str(index_path), str(wal_path))
    hits = restored.search("wal", top_k=2)
    assert hits
    assert hits[0].doc_id == "c"

    query_vec = restored.sparse_vector_for_query("rust replay")
    doc_vec = restored.sparse_vector_for_document(0)
    assert query_vec
    assert doc_vec
