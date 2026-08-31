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


def delete_by_where(where: dict) -> None:
    """Delete chunks matching a metadata filter (e.g. company_id or candidate_id)."""
    if where:
        collection().delete(where=where)


def get_by_where(where: dict) -> list[dict]:
    """Retrieve raw chunks and metadata matching a metadata filter."""
    col = collection()
    if col.count() == 0:
        return []
    data = col.get(where=where, include=["documents", "metadatas"])
    items = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        items.append({"text": text, **meta})
    return items


def query(embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
    col = collection()
    if col.count() == 0:
        return []
    
    # Chroma returns up to count() results
    n_results = min(top_k, col.count())
    result = col.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    
    if not result or not result.get("documents") or not result["documents"][0]:
        return []
        
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
    source_counts: dict[str, int] = {}
    
    if total:
        data = col.get(include=["metadatas"])
        for meta in data["metadatas"]:
            name = meta.get("document_name", "unknown")
            stype = meta.get("source_type", "unknown")
            
            entry = documents.setdefault(
                name, {
                    "document_name": name,
                    "source_type": stype,
                    "company_id": meta.get("company_id"),
                    "candidate_id": meta.get("candidate_id"),
                    "chunks": 0
                }
            )
            entry["chunks"] += 1
            source_counts[stype] = source_counts.get(stype, 0) + 1

    return {
        "total_chunks": total,
        "source_counts": source_counts,
        "documents": sorted(documents.values(), key=lambda d: d["document_name"]),
    }


def reset() -> None:
    global _collection
    try:
        _get_client().delete_collection(config.COLLECTION)
    except Exception:
        pass
    _collection = None

