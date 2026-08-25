"""The only module that talks to Ollama: generation + embeddings.

Keeping this behind one interface is what lets the model be swapped through
configuration later (Code Llama / StarCoder2 / another local model) without
touching the RAG pipeline.
"""
import time

import requests

from app import config


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int) -> dict:
    try:
        resp = requests.post(f"{config.OLLAMA_URL}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is `ollama serve` running?"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise OllamaError(f"Ollama returned an error for {path}: {exc}") from exc


def is_available() -> bool:
    try:
        requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


def available_models() -> list[str]:
    try:
        resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def embed(texts: list[str]) -> list[list[float]]:
    """Embed one or more strings. Ollama's /api/embed accepts a list directly."""
    if not texts:
        return []
    data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts}, timeout=180)
    return data["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def generate(system: str, user: str, model: str | None = None) -> dict:
    """Send prompt to Ollama and return the reply plus measured metrics."""
    model = model or config.LLM_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    }
    started = time.perf_counter()
    data = _post("/api/chat", payload, timeout=600)
    latency = time.perf_counter() - started
    return {
        "reply": data["message"]["content"].strip(),
        "model": model,
        "latency_s": round(latency, 2),
        # Reported by Ollama itself - not estimated.
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
    }
