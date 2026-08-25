"""ChromaDB persistence layer.

Embeddings are always computed by the Ollama embedding service and passed in
explicitly, so Chroma never falls back to downloading its own model. Metadata
travels with every chunk, which is what makes source citations and later
per-company/per-candidate filtering possible.
"""
import logging
import os

# chromadb 0.5.x ships a posthog telemetry hook whose signature no longer matches
# the installed posthog, so every call logs an error. Turn it off and mute it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings

from app import config

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=config.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add(ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
    if not ids:
        return
    collection().upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def delete_document(document_name: str) -> None:
    """Remove every chunk of one document so re-ingesting is idempotent."""
    collection().delete(where={"document_name": document_name})


def query(embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
    col = collection()
    n_results = min(top_k, col.count())
    if n_results == 0:
        return []
    result = col.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": text,
            "distance": round(float(distance), 4),
            # Cosine distance -> a 0..1 similarity that is easier to read.
            "score": round(max(0.0, 1.0 - float(distance)), 4),
            **meta,
        })
    return hits


def stats() -> dict:
    col = collection()
    total = col.count()
    documents: dict[str, dict] = {}
    if total:
        data = col.get(include=["metadatas"])
        for meta in data["metadatas"]:
            name = meta.get("document_name", "unknown")
            entry = documents.setdefault(
                name, {"document_name": name, "source_type": meta.get("source_type"), "chunks": 0}
            )
            entry["chunks"] += 1
    return {
        "total_chunks": total,
        "documents": sorted(documents.values(), key=lambda d: d["document_name"]),
    }


def reset() -> None:
    global _collection
    try:
        _get_client().delete_collection(config.COLLECTION)
    except Exception:
        pass
    _collection = None
