"""Orchestrator & Intent Router: Dispatches queries to specialized intelligence services."""
import re
import time
from app import config, llm, rag, vectorstore
from app.services import eligibility, jd, matching, resume, skill_gap, improvement, interview, repo_rag


def detect_intent(message: str) -> str:
    """Classify prompt intent into one of 10 distinct categories."""
    msg = message.lower().strip()

    # Repository / Codebase Q&A
    if any(k in msg for k in ["which file", "how is rag", "codebase", "repository", "architecture", "fastapi route"]):
        return "REPO_RAG"

    # Company Intelligence / Web Research
    if any(k in msg for k in ["tell me about", "company info", "company research", "company intelligence",
                               "about the company", "research company", "company overview", "company profile",
                               "what does .* do", "what is .* company"]):
        return "COMPANY_INTELLIGENCE"

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

    # Interview Prep (enhanced — also matches question bank queries)
    if any(k in msg for k in ["interview prep", "prepare for interview", "interview questions", "what to prepare",
                               "question bank", "practice questions", "python questions", "sql questions",
                               "dsa questions", "coding questions", "behavioral questions",
                               "give me .* questions", "prepare me for"]):
        return "INTERVIEW_PREPARATION"

    # Company JD Specific
    if any(k in msg for k in ["company require", "role requirement", "jd skill", "company jd", "job description"]):
        return "COMPANY_JD"

    # BMU Policy Specific
    if any(k in msg for k in ["bmu policy", "cgpa requirement", "backlog policy", "attendance rule", "placement policy"]):
        return "BMU_POLICY"

    # Additional check for company intelligence with regex
    if re.search(r"tell me about\s+\w+", msg) or re.search(r"what (?:is|does)\s+\w+", msg):
        return "COMPANY_INTELLIGENCE"

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

    # 2. COMPANY INTELLIGENCE (Live Web Research)
    if intent == "COMPANY_INTELLIGENCE":
        from app.services import company_intelligence
        result = company_intelligence.research_company(
            user_query=message,
            company_id=company_id,
        )
        latency = round(time.perf_counter() - start_time, 2)
        return {
            "reply": result.get("reply", "Unable to process company intelligence request."),
            "intent": intent,
            "grounded": result.get("grounded", False),
            "model": result.get("model", active_model),
            "latency_s": result.get("latency_s", latency),
            "rag": False,
            "sources": result.get("sources", []),
            "verified": result.get("verified", False),
            "provenance": result.get("provenance", "LIVE_WEB"),
            "company_name": result.get("company_name"),
            "status": result.get("status"),
        }

    # 3. ELIGIBILITY ANALYSIS
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

    # 4. RESUME vs JD MATCHING
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

    # 5. SKILL GAP ANALYSIS
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

    # 6. RESUME IMPROVEMENT
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

    # 7. INTERVIEW PREPARATION (Enhanced with Question Bank)
    if intent == "INTERVIEW_PREPARATION":
        # Check if user is asking for specific question bank queries
        msg_lower = message.lower()
        # Direct question bank retrieval for specific queries
        if any(k in msg_lower for k in ["question bank", "practice questions"]) or \
           re.search(r"(?:give me|show me|get|list)\s+(?:\w+\s+)*questions", msg_lower):
            from app.services import question_bank
            # Try to extract filters from the query
            category = _detect_category(msg_lower)
            difficulty = _detect_difficulty(msg_lower)
            result = question_bank.retrieve_questions(
                query=message,
                category=category,
                difficulty=difficulty,
                top_k=10,
            )
            if result:
                summary = _format_question_bank_results(result, category, difficulty)
                latency = round(time.perf_counter() - start_time, 2)
                return {
                    "reply": summary,
                    "intent": intent,
                    "grounded": True,
                    "model": active_model,
                    "latency_s": latency,
                    "rag": True,
                    "sources": [{"n": i+1, "document_name": "interview_question_bank.json",
                                 "source_type": "interview_question_bank",
                                 "section": q.get("category", ""),
                                 "score": q.get("score", 0),
                                 "excerpt": q.get("question", q.get("text", ""))[:200]}
                                for i, q in enumerate(result)],
                }

        # Full interview prep with candidate+JD context
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

    # 8. COMPANY JD SPECIFIC RAG
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

    # 9. DEFAULT BMU POLICY RAG
    result = rag.answer(message, top_k=top_k, use_rag=use_rag, model=active_model)
    result["intent"] = intent
    return result


# ---------------------------------------------------------------------------
# Helper functions for question bank query parsing
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "python": "Python",
    "dsa": "DSA",
    "data structure": "DSA",
    "algorithm": "DSA",
    "dbms": "DBMS & SQL",
    "sql": "DBMS & SQL",
    "database": "DBMS & SQL",
    "oop": "OOP",
    "object oriented": "OOP",
    "os": "Operating Systems",
    "operating system": "Operating Systems",
    "network": "Computer Networks",
    "computer network": "Computer Networks",
    "backend": "Backend / APIs / FastAPI",
    "api": "Backend / APIs / FastAPI",
    "fastapi": "Backend / APIs / FastAPI",
    "git": "Git / Docker / DevOps",
    "docker": "Git / Docker / DevOps",
    "devops": "Git / Docker / DevOps",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "genai": "GenAI / LLMs",
    "llm": "GenAI / LLMs",
    "rag": "RAG / Embeddings / Vector DB",
    "embedding": "RAG / Embeddings / Vector DB",
    "vector": "RAG / Embeddings / Vector DB",
    "system design": "System Design",
    "data science": "Data Science / Data Engineering",
    "data engineering": "Data Science / Data Engineering",
    "hr": "HR / Behavioral / Placement",
    "behavioral": "HR / Behavioral / Placement",
    "placement": "HR / Behavioral / Placement",
}


def _detect_category(msg: str) -> str | None:
    """Try to detect question category from user message."""
    for keyword, category in _CATEGORY_MAP.items():
        if keyword in msg:
            return category
    return None


def _detect_difficulty(msg: str) -> str | None:
    """Try to detect difficulty level from user message."""
    if "easy" in msg:
        return "easy"
    if "hard" in msg or "difficult" in msg:
        return "hard"
    if "medium" in msg:
        return "medium"
    return None


def _format_question_bank_results(results: list[dict], category: str | None, difficulty: str | None) -> str:
    """Format question bank results for display."""
    parts = ["📝 **Interview Question Bank Results**\n"]

    if category:
        parts.append(f"**Category:** {category}")
    if difficulty:
        parts.append(f"**Difficulty:** {difficulty.capitalize()}")
    parts.append(f"**Questions found:** {len(results)}\n")

    for i, q in enumerate(results, 1):
        cat = q.get("category", "General")
        diff = q.get("difficulty", "medium")
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "⚪")
        question_text = q.get("question", q.get("text", ""))
        topics = q.get("expected_topics", "")

        parts.append(f"**{i}. [{cat}] {diff_icon} {diff.capitalize()}**")
        parts.append(f"{question_text}")
        if topics:
            parts.append(f"*Expected topics: {topics}*")
        parts.append("")

    parts.append(
        "> ℹ️ These are generic interview-preparation questions from the question bank. "
        "They are not claimed to be questions officially asked by any specific company."
    )

    return "\n".join(parts)
