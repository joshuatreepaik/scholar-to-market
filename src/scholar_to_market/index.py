"""Embed ingested works into a persistent Chroma vector store."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from . import config


def load_works(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or config.works_path()
    if not path.exists():
        raise FileNotFoundError(f"No works file at {path}. Run ingestion first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _doc_text(work: dict[str, Any]) -> str:
    """The text we embed and retrieve on: title + abstract."""
    title = work.get("title") or ""
    abstract = work.get("abstract") or ""
    return f"{title}\n\n{abstract}".strip()


def _metadata(work: dict[str, Any]) -> dict[str, Any]:
    # Chroma metadata values must be scalars; flatten lists to strings.
    companies = work.get("companies") or []
    return {
        "title": (work.get("title") or "")[:300],
        "year": work.get("year") or 0,
        "cited_by_count": work.get("cited_by_count", 0),
        "topic": work.get("topic") or "unknown",
        "companies": "; ".join(companies),
        "has_company_author": bool(companies),
        "doi": work.get("doi") or "",
        "is_oa": work.get("is_oa", False),
    }


def _short_id(openalex_id: str) -> str:
    """Turn 'https://openalex.org/W123' into 'W123' for compact citations."""
    return (openalex_id or "").rstrip("/").split("/")[-1] or openalex_id


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBED_MODEL
    )
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    # Cosine space so a query distance of `d` gives cosine similarity `1 - d`.
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(path: Path | None = None, batch_size: int = 256) -> int:
    """(Re)build the vector store from a works JSONL file. Returns doc count."""
    works = load_works(path)
    # Only index works that have text to retrieve on.
    works = [w for w in works if _doc_text(w)]

    collection = get_collection(reset=True)
    total = 0
    for start in range(0, len(works), batch_size):
        batch = works[start:start + batch_size]
        collection.add(
            ids=[_short_id(w["id"]) for w in batch],
            documents=[_doc_text(w) for w in batch],
            metadatas=[_metadata(w) for w in batch],
        )
        total += len(batch)
    print(f"Indexed {total} works into '{config.COLLECTION_NAME}' at {config.CHROMA_PATH}")
    return total


if __name__ == "__main__":
    build_index()
