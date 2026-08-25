"""FastAPI application: chat UI + JSON API over the RAG pipeline."""
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, ingest, llm, rag, vectorstore

app = FastAPI(title="BMU Placement Assistant", version="0.1.0")

STATIC_DIR = config.BASE_DIR / "app" / "static"


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    top_k: int | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ollama_reachable": llm.is_available(),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "models_installed": llm.available_models(),
        "knowledge": vectorstore.stats(),
    }


@app.get("/api/documents")
def list_documents():
    return vectorstore.stats()


@app.post("/api/ingest")
def run_ingest():
    """Re-scan knowledge/ and rebuild the vector store entries for what is there."""
    try:
        return ingest.ingest_knowledge_base()
    except llm.OllamaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), source_type: str = "uploaded_document"):
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in config.SUPPORTED_SUFFIXES:
        return JSONResponse(
            {"error": f"Unsupported file type '{suffix}'. Allowed: .md, .txt, .pdf"},
            status_code=400,
        )
    target_dir = config.BMU_DIR if source_type == "bmu_policy" else config.UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / file.filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        return ingest.ingest_file(destination, source_type)
    except llm.OllamaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        return JSONResponse({"error": "message is empty"}, status_code=400)
    try:
        return rag.answer(req.message, top_k=req.top_k, use_rag=req.use_rag)
    except llm.OllamaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/retrieve")
def retrieve(req: ChatRequest):
    """Retrieval only - useful for inspecting what the LLM was actually given."""
    hits = rag.retrieve(req.message, top_k=req.top_k)
    return {"hits": hits, "context": rag.build_context(hits)}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
