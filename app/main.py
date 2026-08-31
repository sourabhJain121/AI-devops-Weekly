"""FastAPI application: BMU Placement Intelligence & Interview Assistant."""
import json
import shutil
import subprocess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, ingest, llm, orchestrator, rag, vectorstore
from app.services import eligibility, jd, matching, resume, skill_gap, improvement, interview

app = FastAPI(title="BMU Placement Intelligence & Interview Assistant", version="1.0.0")

STATIC_DIR = config.BASE_DIR / "app" / "static"


class ChatRequest(BaseModel):
    message: str
    candidate_id: str = "default_candidate"
    company_id: str | None = None
    use_rag: bool = True
    top_k: int | None = None
    model: str | None = None


class AnalysisRequest(BaseModel):
    candidate_id: str = "default_candidate"
    company_id: str | None = None


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


@app.delete("/api/documents/{document_name}")
def delete_document(document_name: str):
    """Delete all vector store chunks for a given document."""
    try:
        vectorstore.delete_document(document_name)
        return {"status": "deleted", "document_name": document_name, "knowledge": vectorstore.stats()}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


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


# --- TARGET SPECIFICATION API ENDPOINTS ---

@app.post("/api/upload/resume")
async def upload_resume(file: UploadFile = File(...), candidate_id: str = "default_candidate"):
    """Upload candidate resume, extract entities, and store candidate profile."""
    target_dir = config.KNOWLEDGE_DIR / "resumes"
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / file.filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        return resume.process_and_index_resume(destination, candidate_id)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/upload/company-jd")
async def upload_company_jd(file: UploadFile = File(...), company_id: str | None = None, session_id: str = "default_session"):
    """Upload company JD, extract entities, and store with company metadata isolation."""
    target_dir = config.KNOWLEDGE_DIR / "jds"
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / file.filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        return jd.process_and_index_jd(destination, company_id, session_id)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/query")
@app.post("/api/chat")
def query_orchestrator(req: ChatRequest):
    """Unified Orchestrated Assistant Query API with model override support."""
    if not req.message.strip():
        return JSONResponse({"error": "message is empty"}, status_code=400)
    try:
        return orchestrator.route_and_execute(
            message=req.message,
            candidate_id=req.candidate_id,
            company_id=req.company_id,
            use_rag=req.use_rag,
            top_k=req.top_k,
            model=req.model,
        )
    except llm.OllamaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.post("/api/analyze/eligibility")
def analyze_eligibility_endpoint(req: AnalysisRequest):
    """Deterministic Placement Eligibility Analysis."""
    return eligibility.evaluate_eligibility(req.candidate_id, req.company_id)


@app.post("/api/analyze/resume-jd")
def analyze_resume_jd_endpoint(req: AnalysisRequest):
    """Structured Resume vs JD Matching."""
    return matching.compare_resume_to_jd(req.candidate_id, req.company_id)


@app.post("/api/analyze/skill-gap")
def analyze_skill_gap_endpoint(req: AnalysisRequest):
    """Prioritized Skill Gap Analysis."""
    return skill_gap.analyze_skill_gaps(req.candidate_id, req.company_id)


@app.post("/api/analyze/resume-improvement")
def analyze_resume_improvement_endpoint(req: AnalysisRequest):
    """Evidence-Based Resume Improvement Recommendations."""
    return improvement.suggest_resume_improvements(req.candidate_id, req.company_id)


@app.post("/api/interview/prepare")
def interview_prepare_endpoint(req: AnalysisRequest):
    """Company & Role-Tailored Interview Question Generator."""
    return interview.generate_interview_prep(req.candidate_id, req.company_id)


@app.get("/api/session/{session_id}")
def get_session_info(session_id: str):
    """Get active session details, candidate profile, and company JDs."""
    cand = resume.get_candidate_profile("default_candidate")
    jds = jd.list_company_jds()
    return {
        "session_id": session_id,
        "candidate": cand,
        "companies": jds,
        "stats": vectorstore.stats(),
    }


@app.post("/api/retrieve")
def retrieve(req: ChatRequest):
    """Retrieval only - inspect what hits were fetched."""
    hits = rag.retrieve(req.message, top_k=req.top_k)
    return {"hits": hits, "context": rag.build_context(hits)}


# --- MODEL EVALUATION & BENCHMARK API ENDPOINTS ---

@app.get("/api/evaluation/comparison")
def get_evaluation_comparison():
    """Retrieve stored multi-model quantitative evaluation comparison results."""
    comp_file = config.BASE_DIR / "evaluation" / "results" / "comparison.json"
    if comp_file.exists():
        return json.loads(comp_file.read_text(encoding="utf-8"))
    legacy_file = config.BASE_DIR / "evaluation" / "results.json"
    if legacy_file.exists():
        return json.loads(legacy_file.read_text(encoding="utf-8"))
    return JSONResponse({"error": "No evaluation comparison data found. Please run evaluation."}, status_code=444)


@app.get("/api/evaluation/rag-analysis")
def get_rag_analysis():
    """Retrieve stored RAG pipeline trace analysis (Cases A to E)."""
    rag_file = config.BASE_DIR / "evaluation" / "rag_analysis.json"
    if not rag_file.exists():
        rag_file = config.BASE_DIR / "evaluation" / "results" / "rag_analysis.json"
    if rag_file.exists():
        return json.loads(rag_file.read_text(encoding="utf-8"))
    return JSONResponse({"error": "No RAG analysis data found."}, status_code=404)


@app.get("/api/evaluation/repository")
def get_repository_results():
    """Retrieve stored Codebase Repository RAG evaluation benchmark results."""
    repo_file = config.BASE_DIR / "evaluation" / "repository_results.json"
    if repo_file.exists():
        return json.loads(repo_file.read_text(encoding="utf-8"))
    return JSONResponse({"error": "No repository analysis data found."}, status_code=404)


@app.get("/api/evaluation/questions")
def get_evaluation_questions():
    """Retrieve the 25 benchmark questions in questions.json."""
    q_file = config.BASE_DIR / "evaluation" / "questions.json"
    if q_file.exists():
        return json.loads(q_file.read_text(encoding="utf-8"))
    return JSONResponse({"error": "No benchmark questions file found."}, status_code=404)


@app.post("/api/evaluation/run")
def trigger_evaluation_rerun():
    """Trigger a refresh run of the multi-model evaluation harness."""
    try:
        script = config.BASE_DIR / "evaluation" / "eval.py"
        subprocess.run([config.PYTHON_EXE, str(script), "--all"], check=True)
        comp_file = config.BASE_DIR / "evaluation" / "results" / "comparison.json"
        if comp_file.exists():
            return json.loads(comp_file.read_text(encoding="utf-8"))
        return {"status": "complete"}
    except Exception as exc:
        return JSONResponse({"error": f"Evaluation rerun failed: {exc}"}, status_code=500)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
