# Current Implementation Audit

## 1. Overview
The current repository is a grounded RAG prototype designed to answer institutional policy questions regarding BMU placement rules. It integrates document loading, section-aware chunking, vector embedding generation via Ollama (`nomic-embed-text`), persistent vector storage in ChromaDB, and grounded prompt generation using Code Llama.

---

## 2. File & Module Map

| File Path | Primary Responsibility | Current Behavior | Reusable? | Required Changes |
| --- | --- | --- | --- | --- |
| `app/config.py` | Configuration management | Loads `.env` file variables (`OLLAMA_URL`, `LLM_MODEL`, `EMBED_MODEL`, `CHROMA_DIR`, etc.) | Yes | Add settings for multi-model evaluation, candidate/company session isolation, and docker configs. |
| `app/documents.py` | Document parsing & cleaning | Extracts text page-by-page from `.pdf`, `.md`, `.txt` files | Yes | Extend to support structured resume extraction and JD parsing (and `.docx` text extraction if needed). |
| `app/chunking.py` | Text chunking | Section-aware chunking preserving section headers and paragraph boundaries | Yes | Reusable as-is for BMU policy and company JDs. |
| `app/vectorstore.py` | ChromaDB collection interface | Manages single Chroma collection (`placement_knowledge`) with cosine distance and basic metadata | Yes | Enhance to support multi-source filtering (`candidate_id`, `company_id`, `session_id`, `source_type`). |
| `app/llm.py` | Ollama client interface | Issues embedding and chat requests to Ollama | Yes | Refactor into configurable provider abstraction supporting multiple LLM models and token usage metrics. |
| `app/rag.py` | RAG retrieval & prompt building | Embeds queries, fetches top-k hits, filters by distance threshold, builds grounded prompts | Yes | Integrate intent routing, multi-source context (BMU + Candidate + Company), and deterministic rules. |
| `app/main.py` | FastAPI application routes | Provides REST endpoints (`/api/health`, `/api/ingest`, `/api/upload`, `/api/chat`, `/api/retrieve`, `/api/documents/{name}`) | Yes | Add endpoints for `/api/upload/resume`, `/api/upload/company-jd`, `/api/analyze/*`, `/api/interview/prepare`, and `/api/query`. |
| `app/static/index.html` | Web frontend | Glassmorphism dashboard for uploading files, chatting, and viewing sources | Yes | Expand UI with tabs for Candidate Resume, Company JD, Eligibility Analysis, Skill Gap, Resume Match, and Interview Prep. |
| `scripts/ingest.py` | CLI ingestion runner | Scans `knowledge/` and indexes files into ChromaDB | Yes | Update to support indexing resume and JD directories if needed. |
| `scripts/ask.py` | CLI question runner | Queries `rag.answer` from terminal | Yes | Update to accept session parameters or test multi-source queries. |

---

## 3. Current Architecture Diagram

```text
User Question / API Request
          │
          ▼
      app/main.py (FastAPI Routes)
          │
          ▼
       app/rag.py (RAG Pipeline)
    ┌─────┴─────────────────────────────┐
    ▼                                   ▼
app/llm.py (Embed query)      app/vectorstore.py (Query ChromaDB)
    │                                   │
    └─────────────────┬─────────────────┘
                      ▼
            app/rag.py (Build Grounded Prompt)
                      │
                      ▼
         app/llm.py (Call Ollama Chat)
                      │
                      ▼
         Response + Source Citations
```

---

## 4. Gaps relative to Master Specification

1. **Candidate Knowledge (Resume)**: No structured resume parser, candidate entity extractor, or candidate storage.
2. **Dynamic Company Knowledge (JD)**: No company JD parser, skill/qualification extractor, or company/session data isolation.
3. **Intent Orchestrator & Router**: Every request uses identical RAG path without classifying intent (`BMU_POLICY`, `ELIGIBILITY`, `RESUME_MATCH`, `SKILL_GAP`, etc.).
4. **Deterministic Eligibility Engine**: Missing deterministic CGPA, backlog, and qualification evaluation logic.
5. **Resume-JD Matcher & Skill Gap Analyzer**: Missing structured skill comparison and evidence extraction.
6. **Resume Improvement & Interview Prep**: Missing evidence-based resume advisor and role-specific interview question generator.
7. **Model Abstraction & Evaluation Harness**: Single LLM provider; missing multi-model comparison suite across 3 models (Code Llama, StarCoder2, etc.) and evaluation dataset.
8. **Repository RAG**: Missing codebase self-answering capabilities.
9. **Docker & Testing Suite**: No `docker-compose` setup or automated pytest suite (unit, integration, e2e, failure cases).
