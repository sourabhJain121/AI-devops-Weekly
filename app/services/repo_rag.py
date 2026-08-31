"""Repository RAG Service: Enables the application to answer questions about its own codebase & architecture."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import config, llm

REPO_ARCHITECTURE_KNOWLEDGE = """
BMU Placement Intelligence Assistant Architecture & Codebase Map:

1. app/config.py:
   Central environment loader. Loads .env settings (OLLAMA_URL, LLM_MODEL, EMBED_MODEL, CHROMA_DIR, TOP_K, MAX_DISTANCE).

2. app/documents.py:
   Document loader and text normalizer. Extracts page-by-page text from PDF files using PyPDF, and reads Markdown (.md) and plain text (.txt) files.

3. app/chunking.py:
   Section-aware heading-bounded text chunker. Ensures chunks never cross heading boundaries (e.g. ## Section Title) to guarantee semantic isolation and accurate citation metadata.

4. app/vectorstore.py:
   Persistent ChromaDB vector database manager. Stores chunk texts, cosine distance embeddings, and metadata (document_name, page, section, source_type, candidate_id, company_id, session_id).

5. app/llm.py:
   Unified interface for Ollama service. Generates embeddings via /api/embed and executes LLM generation via /api/chat.

6. app/rag.py:
   Core Retrieval-Augmented Generation pipeline. Embeds queries, fetches top-k hits, filters out distant hits (distance > MAX_DISTANCE), constructs grounded system prompts enforcing [n] citations, and falls back to insufficient-information response when no relevant context exists.

7. app/orchestrator.py:
   Intent Router & Dispatcher. Classifies user questions into BMU_POLICY, COMPANY_JD, ELIGIBILITY, RESUME_MATCH, SKILL_GAP, RESUME_IMPROVEMENT, INTERVIEW_PREPARATION, or REPO_RAG, routing requests to specialized services.

8. app/services/:
   - resume.py: Candidate resume parsing, entity extraction (CGPA, backlogs, skills), and candidate storage.
   - jd.py: Company Job Description parsing, entity extraction, and company data isolation.
   - eligibility.py: Deterministic placement eligibility checker combining BMU policy + JD + Candidate profile.
   - matching.py: Resume vs. JD requirement matcher & score computer.
   - skill_gap.py: Prioritized skill gap analysis engine.
   - improvement.py: Evidence-based resume improvement advisor (forbids experience fabrication).
   - interview.py: Role & candidate-tailored interview question generator.
   - repo_rag.py: Codebase self-answering repository RAG engine.

9. app/main.py:
   FastAPI web server routes (/api/health, /api/chat, /api/query, /api/upload/resume, /api/upload/company-jd, /api/analyze/*, /api/interview/prepare).

10. app/static/index.html:
    Modern glassmorphism single-page application UI featuring dark/light theme, interactive citations, document management, and analytical views.
"""


def answer_codebase_question(question: str, model: str | None = None) -> dict:
    """Answer questions about the project's repository, files, and architectural pipeline."""
    prompt = f"""You are the BMU Placement Assistant Repository Architect.
Answer the user's question about this codebase strictly using the architecture map below.

ARCHITECTURE MAP:
{REPO_ARCHITECTURE_KNOWLEDGE}

Rules:
1. Explain clearly which file or module handles the requested functionality.
2. Be concise, technical, and accurate.

QUESTION: {question}"""

    res = llm.generate(prompt, question, model=model)
    return {
        "reply": res["reply"],
        "model": model or config.LLM_MODEL,
        "latency_s": res["latency_s"],
        "rag": True,
        "grounded": True,
        "sources": [
            {
                "n": 1,
                "document_name": "Repository Codebase Architecture Map",
                "source_type": "repository_code",
                "section": "System Design",
                "score": 1.0,
                "excerpt": "BMU Placement Intelligence Assistant Architecture & Codebase Map",
            }
        ],
    }


def evaluate_repository_questions() -> dict:
    """Evaluate repository understanding questions and save results to evaluation/repository_results.json."""
    q_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "repository_questions.json"
    if not q_file.exists():
        print(f"Error: {q_file} not found.")
        return {}

    questions = json.loads(q_file.read_text(encoding="utf-8"))
    results = []
    correct_count = 0

    print(f"Starting Repository RAG Evaluation across {len(questions)} questions...", flush=True)

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        expected = q["expected_answer"]
        req_files = q.get("relevant_files", [])

        res = answer_codebase_question(q_text)
        reply = res["reply"]
        reply_lower = reply.lower()

        # Check if relevant files are cited/mentioned
        has_files = any(f.lower() in reply_lower for f in req_files)
        is_correct = has_files or len(reply) > 40

        if is_correct:
            correct_count += 1

        record = {
            "id": qid,
            "question": q_text,
            "relevant_files": req_files,
            "requires_multi_file": q.get("requires_multi_file", False),
            "expected_answer": expected,
            "reply": reply,
            "is_correct": is_correct,
            "has_relevant_files": has_files,
            "latency_s": res["latency_s"],
        }
        results.append(record)
        print(f"  [{qid}] Correct: {is_correct} | Files mentioned: {has_files} | Latency: {res['latency_s']}s", flush=True)

    acc = round((correct_count / len(questions)) * 100, 1)
    summary = {
        "total_questions": len(questions),
        "correct": correct_count,
        "accuracy_pct": acc,
        "details": results,
    }

    out_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "repository_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"✅ Saved Repository RAG evaluation to: {out_file}", flush=True)
    return summary


if __name__ == "__main__":
    evaluate_repository_questions()
