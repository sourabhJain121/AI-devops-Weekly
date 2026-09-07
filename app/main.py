"""FastAPI application: BMU Placement Intelligence & Interview Assistant."""
import json
import shutil
import subprocess
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, ingest, llm, orchestrator, rag, vectorstore
from app.services import eligibility, jd, matching, resume, skill_gap, improvement, interview

app = FastAPI(title="BMU Placement Intelligence & Interview Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class CompanyIntelligenceRequest(BaseModel):
    company_name: str
    company_id: str | None = None


class QuestionBankSearchRequest(BaseModel):
    query: str | None = None
    category: str | None = None
    subcategory: str | None = None
    difficulty: str | None = None
    skills: str | None = None
    role: str | None = None
    question_type: str | None = None
    tags: str | None = None
    top_k: int = 10


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


# --- COMPANY INTELLIGENCE API ENDPOINTS ---

@app.post("/api/company/intelligence")
def company_intelligence_endpoint(req: CompanyIntelligenceRequest):
    """Live Company Intelligence — Web Research with Verification Gates."""
    from app.services import company_intelligence
    try:
        return company_intelligence.research_company(
            user_query=f"Tell me about {req.company_name}",
            company_id=req.company_id,
        )
    except Exception as exc:
        return JSONResponse({
            "error": str(exc),
            "reply": (
                "I couldn't retrieve live company information right now. "
                "Please try again later or provide an official company source."
            ),
        }, status_code=500)


@app.get("/api/company/search")
def company_search_endpoint(query: str = Query(..., description="Company name to search")):
    """Quick company verification check."""
    from app.services import company_intelligence
    try:
        return company_intelligence.verify_company(query)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# --- INTERVIEW QUESTION BANK API ENDPOINTS ---

@app.post("/api/interview/questions/search")
def search_question_bank(req: QuestionBankSearchRequest):
    """Search the interview question bank with filters and/or semantic search."""
    from app.services import question_bank
    try:
        results = question_bank.retrieve_questions(
            query=req.query,
            category=req.category,
            subcategory=req.subcategory,
            difficulty=req.difficulty,
            skills=req.skills,
            role=req.role,
            question_type=req.question_type,
            tags=req.tags,
            top_k=req.top_k,
        )
        return {
            "results": results,
            "count": len(results),
            "filters": {
                "category": req.category,
                "difficulty": req.difficulty,
                "skills": req.skills,
                "query": req.query,
            },
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/interview/questions")
def list_question_bank(
    query: str | None = None,
    category: str | None = None,
    difficulty: str | None = None,
    skills: str | None = None,
    top_k: int = 30,
):
    """Retrieve interview questions with optional search query and category/difficulty/skills filters."""
    from app.services import question_bank
    try:
        results = question_bank.retrieve_questions(
            query=query,
            category=category,
            difficulty=difficulty,
            skills=skills,
            top_k=top_k,
        )
        return {"results": results, "count": len(results), "total": len(results)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/ingest/question-bank")
def ingest_question_bank_endpoint():
    """Manually trigger interview question bank ingestion."""
    from app.services import question_bank
    try:
        result = question_bank.ingest_question_bank()
        return result
    except llm.OllamaError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/interview/questions/validate")
def validate_question_bank_endpoint():
    """Validate the interview question bank JSON without ingesting."""
    from app.services import question_bank
    return question_bank.validate_question_bank()


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
@app.get("/api/candidate/profile")
def get_candidate_profile_endpoint(candidate_id: str = "default_candidate"):
    """Get active structured candidate profile."""
    prof = resume.get_candidate_profile(candidate_id)
    if not prof:
        resume_files = list((config.KNOWLEDGE_DIR / "resumes").glob("*.pdf"))
        if resume_files:
            try:
                res_data = resume.process_and_index_resume(resume_files[0], candidate_id)
                prof = res_data.get("profile")
            except Exception:
                pass
    return {"candidate": prof}


@app.get("/api/company-jds")
def list_company_jds_endpoint():
    """List all loaded/indexed company JDs with structured details."""
    jds = jd.get_all_company_jds()
    if not jds:
        jd_files = list((config.KNOWLEDGE_DIR / "jds").glob("*.pdf"))
        for f in jd_files:
            try:
                jd.process_and_index_jd(f)
            except Exception:
                pass
        jds = jd.get_all_company_jds()
    return {"jds": jds, "count": len(jds)}


@app.get("/api/company-jds/{company_id}")
def get_company_jd_endpoint(company_id: str):
    """Retrieve specific company JD by id."""
    jd_obj = jd.get_company_jd(company_id)
    if not jd_obj:
        return JSONResponse({"error": "Company JD not found"}, status_code=404)
    return jd_obj


class ReadinessRequest(BaseModel):
    candidate_id: str = "default_candidate"
    company_id: str | None = None


@app.post("/api/placement/readiness")
def calculate_readiness_endpoint(req: ReadinessRequest):
    """Compute deterministic placement readiness across real dimensions."""
    cand = resume.get_candidate_profile(req.candidate_id)
    comp = jd.get_company_jd(req.company_id) if req.company_id else None

    if not cand and not comp:
        return {
            "status": "missing_context",
            "message": "Upload a candidate resume and select a Job Description to calculate readiness.",
            "overall_readiness": None,
            "dimensions": {},
        }

    # 1. Resume Match Dimension
    match_result = matching.compare_resume_to_jd(req.candidate_id, req.company_id) if comp and cand else None
    resume_match = float(match_result.get("match_score", 0.0)) if match_result else (60.0 if cand else 0.0)

    # 2. Institutional Policy Eligibility Dimension
    elig_result = eligibility.evaluate_eligibility(req.candidate_id, req.company_id) if cand else None
    if elig_result:
        status_val = elig_result.get("status", "")
        policy_score = 100.0 if status_val == "Eligible" else (70.0 if "Conditional" in status_val else 45.0)
    else:
        policy_score = 0.0

    # 3. Technical Skill Readiness
    gaps_result = skill_gap.analyze_skill_gaps(req.candidate_id, req.company_id) if comp and cand else None
    if gaps_result:
        high_gaps = gaps_result.get("high_priority_gaps", [])
        tech_score = round(max(0.0, 100.0 - (len(high_gaps) * 20.0)), 1)
    elif cand:
        cand_skills = cand.get("skills", [])
        tech_score = min(100.0, float(len(cand_skills) * 6.0))
    else:
        tech_score = 0.0

    # 4. Interview Readiness
    interview_score = round((resume_match * 0.4 + tech_score * 0.4 + policy_score * 0.2), 1)

    # 5. Overall Placement Readiness
    overall = round((resume_match * 0.35 + tech_score * 0.35 + policy_score * 0.15 + interview_score * 0.15), 1)

    return {
        "status": "ready",
        "overall_readiness": overall,
        "dimensions": {
            "resume_match": resume_match,
            "jd_match": resume_match,
            "technical_readiness": tech_score,
            "interview_readiness": interview_score,
            "policy_eligibility": policy_score,
        },
        "candidate": cand.get("name") if cand else None,
        "company": comp.get("company_name") if comp else None,
        "role": comp.get("role_title") if comp else None,
        "gaps_count": len(gaps_result.get("high_priority_gaps", [])) if gaps_result else 0,
        "matching_skills": match_result.get("matching_skills", []) if match_result else [],
        "missing_skills": match_result.get("missing_skills", []) if match_result else [],
    }


class MockEvaluationRequest(BaseModel):
    question: str
    user_answer: str
    category: str | None = "General"
    expected_topics: str | None = None
    role: str | None = None


@app.post("/api/interview/mock/evaluate")
def evaluate_mock_answer_endpoint(req: MockEvaluationRequest):
    """Evaluate candidate answer against technical question and expected topics."""
    if not req.user_answer.strip():
        return JSONResponse({"error": "Answer cannot be empty"}, status_code=400)

    prompt = f"""You are a Senior Technical Interviewer evaluating a candidate's answer for a placement interview.

QUESTION: {req.question}
CATEGORY: {req.category}
EXPECTED TOPICS: {req.expected_topics or 'Core principles, correct terminology, edge cases, trade-offs'}

CANDIDATE'S ANSWER:
{req.user_answer}

Provide an objective evaluation. Return a JSON object with:
- "technical_accuracy": int (0 to 100)
- "groundedness": int (0 to 100)
- "completeness": int (0 to 100)
- "strengths": list of 2-3 specific things the candidate explained well
- "improvements": list of 2-3 specific topics or concepts the candidate missed or should clarify
- "model_answer_snippet": 2-3 sentence ideal answer summary
- "feedback_summary": 2-3 sentence concise coaching feedback
"""
    try:
        res = llm.generate_json(prompt, "Evaluate mock interview answer.")
        if isinstance(res, dict) and "technical_accuracy" in res:
            return res
    except Exception:
        pass

    ans = req.user_answer.lower()
    topics = [t.strip().lower() for t in (req.expected_topics or "").split(",") if t.strip()]
    found_topics = [t for t in topics if t in ans]
    acc = min(95, max(40, int((len(found_topics) / max(len(topics), 1)) * 100))) if topics else (80 if len(ans.split()) > 25 else 55)

    return {
        "technical_accuracy": acc,
        "groundedness": min(95, acc + 5),
        "completeness": min(95, max(35, int(len(ans.split()) * 1.5))),
        "strengths": [f"Demonstrated awareness of {found_topics[0]}" if found_topics else "Provided structured response", "Clear intent and communication"],
        "improvements": [f"Incorporate discussion of {t}" for t in topics if t not in found_topics][:2] or ["Elaborate on production failure modes or edge cases"],
        "model_answer_snippet": f"A comprehensive answer addresses {req.expected_topics or 'fundamental principles and real-world trade-offs'}.",
        "feedback_summary": "Good initial articulation. Expand on concrete examples and expected architectural keywords."
    }


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


class RagComparisonRequest(BaseModel):
    question: str
    model: str = "codellama:7b"


@app.post("/api/evaluation/rag-comparison")
def rag_comparison_endpoint(req: RagComparisonRequest):
    """Run the SAME question through the selected model in both WITHOUT-RAG and WITH-RAG modes.
    Uses real local Ollama model execution and real ChromaDB vectorstore retrieval.
    """
    clean_q = req.question.strip()
    if not clean_q:
        return JSONResponse({"error": "Question cannot be empty."}, status_code=400)

    model_name = req.model or "codellama:7b"
    start_time = time.perf_counter()

    # 1. Execute WITHOUT RAG
    t_wor0 = time.perf_counter()
    try:
        without_rag = rag.answer(clean_q, use_rag=False, model=model_name)
        without_rag["latency_s"] = round(time.perf_counter() - t_wor0, 2)
    except Exception as exc:
        without_rag = {
            "reply": f"Model execution failed without RAG: {exc}",
            "rag": False,
            "sources": [],
            "context": "",
            "latency_s": round(time.perf_counter() - t_wor0, 2),
            "error": str(exc),
        }

    # 2. Execute WITH RAG
    t_wr0 = time.perf_counter()
    try:
        with_rag = rag.answer(clean_q, use_rag=True, model=model_name)
        with_rag["latency_s"] = round(time.perf_counter() - t_wr0, 2)
    except Exception as exc:
        with_rag = {
            "reply": f"Model execution failed with RAG: {exc}",
            "rag": True,
            "sources": [],
            "context": "",
            "latency_s": round(time.perf_counter() - t_wr0, 2),
            "error": str(exc),
        }

    total_latency = round(time.perf_counter() - start_time, 2)

    return {
        "question": clean_q,
        "model": model_name,
        "without_rag": without_rag,
        "with_rag": with_rag,
        "total_latency_s": total_latency,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
