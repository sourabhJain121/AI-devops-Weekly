"""LLM Service Abstraction & Multi-Model Provider Interface for Ollama."""
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
        if exc.response is not None and exc.response.status_code == 404:
            installed = available_models()
            requested = payload.get("model", "unknown")
            raise OllamaError(
                f"Model '{requested}' is not installed in Ollama. "
                f"Currently installed models: {installed}. "
                f"Please run `ollama pull {requested}` to install it."
            ) from exc
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
    """Embed one or more strings using Ollama's embedding service."""
    if not texts:
        return []
    data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": texts}, timeout=180)
    return data["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


# --- LLM Provider Abstraction Classes ---

class BaseLLMProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model_name,
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
            "model": self.model_name,
            "latency_s": round(latency, 2),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        }


class CodeLlamaProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__("codellama:7b")


class StarCoder2Provider(BaseLLMProvider):
    def __init__(self):
        super().__init__("starcoder2:3b")


class QwenProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__("qwen2.5:0.5b")


class OllamaProviderFactory:
    @staticmethod
    def get_provider(model_name: str | None = None) -> BaseLLMProvider:
        model = model_name or config.LLM_MODEL
        if "codellama" in model:
            return CodeLlamaProvider()
        elif "starcoder" in model:
            return StarCoder2Provider()
        elif "qwen" in model:
            return QwenProvider()
        return BaseLLMProvider(model)


def generate(system: str, user: str, model: str | None = None) -> dict:
    """Send prompt to the selected LLM provider and return the reply plus measured metrics."""
    provider = OllamaProviderFactory.get_provider(model)
    return provider.generate(system, user)
