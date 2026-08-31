"""Central configuration. Everything environment-driven, nothing hard-coded."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the prototype has no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


_load_dotenv(BASE_DIR / ".env")

# --- Ollama ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "codellama:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# --- Vector store ---
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "storage" / "chroma")))
COLLECTION = os.getenv("COLLECTION", "placement_knowledge")

# --- Chunking / retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))
# Cosine distance above which a chunk is considered too unrelated to trust.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.65"))

# --- Knowledge locations ---
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", str(BASE_DIR / "knowledge")))
BMU_DIR = KNOWLEDGE_DIR / "bmu"
UPLOAD_DIR = KNOWLEDGE_DIR / "uploads"

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
