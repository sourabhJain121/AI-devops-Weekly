"""The RAG pipeline: question -> query embedding -> similarity search ->
relevant context -> grounded prompt -> Ollama -> answer.
"""
from app import config, llm, vectorstore

SYSTEM_PROMPT = """You are the BMU Placement Assistant.

Rules you must follow:
1. Answer ONLY from the CONTEXT below. It is the sole source of truth.
2. Never invent policies, rules, numbers, dates, eligibility criteria or company
   requirements that are not written in the CONTEXT.
3. If the CONTEXT does not contain enough information, reply exactly:
   "I could not find sufficient information in the provided documents to answer
   this reliably." and then say what document would be needed.
4. Cite the source of each fact inline using the [n] markers shown in the CONTEXT.
5. Keep the distinction clear between what the documents state and any advice you
   add. Label advice as "Suggestion:".
6. Be concise and use plain language.

CONTEXT:
{context}"""

INSUFFICIENT = (
    "I could not find sufficient information in the provided documents to answer "
    "this reliably. Please add the relevant document to the knowledge base and "
    "re-run ingestion."
)


def retrieve(question: str, top_k: int | None = None, source_types: list[str] | None = None) -> list[dict]:
    """Embed the question and pull the closest chunks out of ChromaDB."""
    top_k = top_k or config.TOP_K
    where = {"source_type": {"$in": source_types}} if source_types else None
    hits = vectorstore.query(llm.embed_one(question), top_k, where)
    return [h for h in hits if h["distance"] <= config.MAX_DISTANCE]


def build_context(hits: list[dict]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        label = hit.get("document_name", "unknown")
        if hit.get("section"):
            label += f" > {hit['section']}"
        if hit.get("page"):
            label += f" (page {hit['page']})"
        blocks.append(f"[{index}] {label}\n{hit['text']}")
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "n": index,
            "document_name": hit.get("document_name"),
            "source_type": hit.get("source_type"),
            "section": hit.get("section") or None,
            "page": hit.get("page"),
            "score": hit["score"],
            "excerpt": hit["text"][:320] + ("..." if len(hit["text"]) > 320 else ""),
        }
        for index, hit in enumerate(hits, start=1)
    ]


def answer(question: str, top_k: int | None = None, use_rag: bool = True, model: str | None = None) -> dict:
    """Full pipeline. Returns the reply together with everything it was based on."""
    if not use_rag:
        result = llm.generate(
            "You are a helpful assistant. Answer from your own knowledge and say so.",
            question,
            model=model,
        )
        return {**result, "rag": False, "sources": [], "context": ""}

    hits = retrieve(question, top_k)
    if not hits:
        # Nothing passed the relevance threshold - do not let the model guess.
        return {
            "reply": INSUFFICIENT,
            "model": model or config.LLM_MODEL,
            "latency_s": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "rag": True,
            "sources": [],
            "context": "",
            "grounded": False,
        }

    context = build_context(hits)
    result = llm.generate(SYSTEM_PROMPT.format(context=context), question, model=model)
    return {**result, "rag": True, "sources": _sources(hits), "context": context, "grounded": True}
