"""Ingestion pipeline: document -> extract -> clean -> chunk -> embed -> ChromaDB."""
import hashlib
from pathlib import Path

from app import chunking, config, documents, llm, vectorstore

EMBED_BATCH = 32


def _document_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def ingest_file(path: Path, source_type: str = "bmu_policy", extra: dict | None = None) -> dict:
    """Ingest a single document and return what was stored."""
    path = Path(path)
    pages = documents.load(path)
    if not pages:
        return {"document_name": path.name, "chunks": 0, "note": "no extractable text"}

    document_id = _document_id(path)
    document_name = path.name

    # Re-ingesting a document replaces its old chunks rather than duplicating them.
    vectorstore.delete_document(document_name)

    ids, texts, metadatas = [], [], []
    for page_number, page_text in pages:
        for index, piece in enumerate(chunking.chunk(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{document_id}:p{page_number}:c{index}")
            texts.append(piece["text"])
            metadatas.append({
                "document_id": document_id,
                "document_name": document_name,
                "source_type": source_type,
                "page": page_number,
                "section": piece["section"] or "",
                "chunk_index": index,
                **(extra or {}),
            })

    for start in range(0, len(texts), EMBED_BATCH):
        batch = slice(start, start + EMBED_BATCH)
        vectorstore.add(ids[batch], texts[batch], metadatas[batch], llm.embed(texts[batch]))

    return {
        "document_name": document_name,
        "source_type": source_type,
        "pages": len(pages),
        "chunks": len(ids),
    }


def ingest_directory(directory: Path, source_type: str) -> list[dict]:
    return [ingest_file(path, source_type) for path in documents.discover(Path(directory))]


def ingest_knowledge_base() -> dict:
    """Ingest everything currently sitting in knowledge/."""
    results = ingest_directory(config.BMU_DIR, "bmu_policy")
    results += ingest_directory(config.UPLOAD_DIR, "uploaded_document")
    return {
        "documents": results,
        "total_chunks": sum(r.get("chunks", 0) for r in results),
    }
