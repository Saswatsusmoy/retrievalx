from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_USER_AGENT = "retrievalx-examples/0.1"


@dataclass
class BeirDocument:
    doc_id: str
    text: str
    title: str | None


@dataclass
class BeirQuery:
    query_id: str
    text: str


@dataclass
class BeirDataset:
    name: str
    corpus: list[BeirDocument]
    queries: list[BeirQuery]
    qrels: dict[str, dict[str, int]]


def dataset_url(name: str) -> str:
    return f"{BEIR_BASE_URL}/{name}.zip"


def ensure_dataset(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = cache_dir / name
    if _dataset_ready(dataset_dir):
        return dataset_dir

    archive_path = cache_dir / f"{name}.zip"
    if not archive_path.exists():
        _download(dataset_url(name), archive_path)

    _extract_zip(archive_path, cache_dir)
    resolved = _resolve_dataset_dir(cache_dir, name)
    if not _dataset_ready(resolved):
        raise FileNotFoundError(
            f"dataset '{name}' was downloaded but required files were not found in {resolved}"
        )
    return resolved


def load_dataset(
    name: str,
    cache_dir: Path,
    split: str = "test",
    max_docs: int | None = None,
    max_queries: int | None = None,
) -> BeirDataset:
    dataset_dir = ensure_dataset(name=name, cache_dir=cache_dir)
    corpus = _load_corpus(dataset_dir / "corpus.jsonl", max_docs=max_docs)
    qrels = _load_qrels(dataset_dir, split=split)
    queries = _load_queries(dataset_dir / "queries.jsonl", max_queries=None)

    query_ids = set(qrels.keys())
    filtered_queries = [query for query in queries if query.query_id in query_ids]

    if max_queries is not None:
        filtered_queries = filtered_queries[:max_queries]
        allowed_query_ids = {query.query_id for query in filtered_queries}
        qrels = {qid: rels for qid, rels in qrels.items() if qid in allowed_query_ids}

    return BeirDataset(
        name=name,
        corpus=corpus,
        queries=filtered_queries,
        qrels=qrels,
    )


def _dataset_ready(dataset_dir: Path) -> bool:
    if not dataset_dir.exists():
        return False
    if not (dataset_dir / "corpus.jsonl").exists():
        return False
    if not (dataset_dir / "queries.jsonl").exists():
        return False
    qrels_dir = dataset_dir / "qrels"
    if qrels_dir.exists():
        return True
    return (dataset_dir / "qrels.tsv").exists()


def _download(url: str, output_path: Path) -> None:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request) as response, output_path.open("wb") as sink:
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError(
                    f"archive too large for safe download ({total} bytes > {MAX_ARCHIVE_BYTES})"
                )
            sink.write(chunk)


def _extract_zip(archive_path: Path, output_root: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = Path(member.filename)
            # Zip-slip protection
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(f"unsafe archive member path: {member.filename}")
            resolved_target = output_root / target
            if member.is_dir():
                resolved_target.mkdir(parents=True, exist_ok=True)
                continue
            resolved_target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, resolved_target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _resolve_dataset_dir(cache_dir: Path, name: str) -> Path:
    direct = cache_dir / name
    if direct.exists():
        return direct

    matches = [
        entry
        for entry in cache_dir.iterdir()
        if entry.is_dir() and entry.name.lower() == name.lower()
    ]
    if matches:
        return matches[0]
    return direct


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_corpus(path: Path, max_docs: int | None = None) -> list[BeirDocument]:
    out: list[BeirDocument] = []
    for row in _read_jsonl(path):
        doc_id = str(row.get("_id", ""))
        text = str(row.get("text", ""))
        title = row.get("title")
        title_text = str(title) if title is not None else None
        out.append(BeirDocument(doc_id=doc_id, text=text, title=title_text))
        if max_docs is not None and len(out) >= max_docs:
            break
    return out


def _load_queries(path: Path, max_queries: int | None = None) -> list[BeirQuery]:
    out: list[BeirQuery] = []
    for row in _read_jsonl(path):
        out.append(BeirQuery(query_id=str(row.get("_id", "")), text=str(row.get("text", ""))))
        if max_queries is not None and len(out) >= max_queries:
            break
    return out


def _load_qrels(dataset_dir: Path, split: str) -> dict[str, dict[str, int]]:
    split_path = dataset_dir / "qrels" / f"{split}.tsv"
    fallback = dataset_dir / "qrels.tsv"
    qrels_path = split_path if split_path.exists() else fallback
    if not qrels_path.exists():
        return {}

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if line_no == 0 and "query" in line.lower() and "corpus" in line.lower():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            query_id = parts[0]
            corpus_id = parts[1]
            try:
                score = int(float(parts[2]))
            except ValueError:
                score = 0
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels
