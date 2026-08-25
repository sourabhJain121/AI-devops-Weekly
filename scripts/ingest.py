"""CLI ingestion: python scripts/ingest.py [--reset]

Reads every supported document under knowledge/ and rebuilds its chunks in
ChromaDB. Safe to re-run: a document's old chunks are replaced, not duplicated.
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, ingest, llm, vectorstore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if not llm.is_available():
        print(f"Ollama is not reachable at {config.OLLAMA_URL}. Start it with `ollama serve`.")
        return 1

    if args.reset:
        vectorstore.reset()
        print("Collection reset.")

    print(f"Scanning {config.KNOWLEDGE_DIR}")
    result = ingest.ingest_knowledge_base()
    if not result["documents"]:
        print(f"No documents found. Put .pdf/.md/.txt files in {config.BMU_DIR} and re-run.")
        return 1

    for doc in result["documents"]:
        print(f"  {doc['document_name']:<45} {doc.get('pages', 0):>3} pages  {doc['chunks']:>4} chunks")
    print(f"\nTotal: {result['total_chunks']} chunks in collection '{config.COLLECTION}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
