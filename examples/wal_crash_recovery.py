import tempfile
from pathlib import Path

from retrievalx import BM25Index

with tempfile.TemporaryDirectory(prefix="retrievalx-wal-") as tmp_dir:
    root = Path(tmp_dir)
    index_path = root / "support_index.rtx"
    wal_path = root / "support_index.wal"

    index = BM25Index.from_documents(
        [
            ("doc-1", "how to reset mfa device"),
            ("doc-2", "password reset flow for enterprise sso"),
        ]
    )
    index.save(str(index_path))

    # Turn on WAL so writes are crash-recoverable.
    index.enable_wal(str(wal_path))
    index.insert_batch([("doc-3", "mfa backup codes invalid after rotation")])
    index.delete("doc-2")

    # Simulate process restart and recover from base snapshot + WAL.
    recovered = BM25Index.load_with_wal(str(index_path), str(wal_path))
    hits = recovered.search("mfa reset", top_k=3)

    print("Recovered hits:")
    for hit in hits:
        print(f"  {hit.doc_id}: {hit.score:.4f}")

    # Persist a compacted checkpoint and clear WAL.
    recovered.compact_and_flush(str(index_path))
