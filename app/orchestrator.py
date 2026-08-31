"""Orchestrator & Intent Router: Dispatches queries to specialized intelligence services."""
import re
import time
from app import config, llm, rag, vectorstore
from app.services import eligibility, jd, matching, resume, skill_gap, improvement, interview, repo_rag


def detect_intent(message: str) -> str:
    """Classify prompt intent into one of 8 distinct categories."""
    msg = message.lower().strip()

    # Repository / Codebase Q&A
    if any(k in msg for k in ["which file", "how is rag", "codebase", "repository", "architecture", "fastapi route"]):
        return "REPO_RAG"

    # Resume Improvement / Suggestions / Changes
    if any(k in msg for k in ["improve my resume", "resume advice", "enhance resume", "resume suggestion", "changes that i can make", "change in my resume", "resume changes", "edit resume", "feedback on resume", "resume review"]):
        return "RESUME_IMPROVEMENT"

    # Eligibility Analysis
    if any(k in msg for k in ["can i apply", "am i eligible", "eligibility check", "eligible for", "do i meet criteria", "eligible"]):
        return "ELIGIBILITY"

    # Resume vs JD Match
    if any(k in msg for k in ["match my resume", "resume match", "how well do i match", "match score", "fit for this role", "resume fit"]):
        return "RESUME_MATCH"

    # Skill Gap
    if any(k in msg for k in ["skill gap", "missing skills", "what skills am i missing", "what am i missing"]):
        return "SKILL_GAP"

    # Interview Prep
    if any(k in msg for k in ["interview prep", "prepare for interview", "interview questions", "what to prepare"]):
        return "INTERVIEW_PREPARATION"

    # Company JD Specific
    if any(k in msg for k in ["company require", "role requirement", "jd skill", "company jd", "job description"]):
        return "COMPANY_JD"

    # BMU Policy Specific
    if any(k in msg for k in ["bmu policy", "cgpa requirement", "backlog policy", "attendance rule", "placement policy"]):
        return "BMU_POLICY"

    return "BMU_POLICY"


def route_and_execute(
    message: str,
    candidate_id: str = "default_candidate",
    company_id: str | None = None,
    use_rag: bool = True,
    top_k: int | None = None,
    model: str | None = None,
) -> dict:
    """Main Orchestrator Entrypoint: Routes user query to specialized services."""
    start_time = time.perf_counter()
    intent = detect_intent(message)
    top_k = top_k or config.TOP_K
    active_model = model or config.LLM_MODEL

    # 1. REPO RAG
    if intent == "REPO_RAG":
        result = repo_rag.answer_codebase_question(message, model=active_model)
        result["intent"] = intent
        return result

    # 2. ELIGIBILITY ANALYSIS
    if intent == "ELIGIBILITY":
        analysis = eligibility.evaluate_eligibility(candidate_id=candidate_id, company_id=company_id)
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": analysis["explanation"],
            "intent": intent,
            "grounded": True,
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": analysis.get("sources", []),
            "analysis_data": analysis,
        }

    # 3. RESUME vs JD MATCHING
    if intent == "RESUME_MATCH":
        analysis = matching.compare_resume_to_jd(candidate_id=candidate_id, company_id=company_id)
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": analysis["summary"],
            "intent": intent,
            "grounded": True,
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": analysis.get("sources", []),
            "analysis_data": analysis,
        }

    # 4. SKILL GAP ANALYSIS
    if intent == "SKILL_GAP":
        analysis = skill_gap.analyze_skill_gaps(candidate_id=candidate_id, company_id=company_id)
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": analysis["summary"],
            "intent": intent,
            "grounded": True,
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": analysis.get("sources", []),
            "analysis_data": analysis,
        }

    # 5. RESUME IMPROVEMENT
    if intent == "RESUME_IMPROVEMENT":
        analysis = improvement.suggest_resume_improvements(candidate_id=candidate_id, company_id=company_id)
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": analysis["summary"],
            "intent": intent,
            "grounded": True,
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": analysis.get("sources", []),
            "analysis_data": analysis,
        }

    # 6. INTERVIEW PREPARATION
    if intent == "INTERVIEW_PREPARATION":
        analysis = interview.generate_interview_prep(candidate_id=candidate_id, company_id=company_id)
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": analysis["summary"],
            "intent": intent,
            "grounded": True,
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": analysis.get("sources", []),
            "analysis_data": analysis,
        }

    # 7. COMPANY JD SPECIFIC RAG
    if intent == "COMPANY_JD":
        where = {"source_type": "company_jd"}
        if company_id:
            where["company_id"] = company_id
        hits = vectorstore.query(llm.embed_one(message), top_k=top_k, where=where)
        if not hits:
            # Fallback if no specific company chunk found
            hits = rag.retrieve(message, top_k=top_k)
        
        context = rag.build_context(hits)
        if not hits:
            reply = rag.INSUFFICIENT
        else:
            prompt = rag.SYSTEM_PROMPT.format(context=context)
            res = llm.generate(prompt, message, model=active_model)
            reply = res["reply"]
            
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": reply,
            "intent": intent,
            "grounded": bool(hits),
            "model": active_model,
            "latency_s": latency,
            "rag": True,
            "sources": rag._sources(hits),
            "context": context,
        }

    # 8. DEFAULT BMU POLICY RAG
    result = rag.answer(message, top_k=top_k, use_rag=use_rag, model=active_model)
    result["intent"] = intent
    return result
