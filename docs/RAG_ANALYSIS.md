# RAG Pipeline Failure & Success Mode Analysis

This document provides a detailed breakdown of the retrieval and generation pipeline across the 5 canonical RAG failure/success cases evaluated in [`evaluation/results/rag_analysis.json`](file:///Users/sourabh/Downloads/idtt/evaluation/results/rag_analysis.json).

```text
QUESTION
   ↓
RETRIEVED CHUNKS (Vector Store)
   ↓
CONTEXT SENT TO LLM (Prompt Builder)
   ↓
LLM RESPONSE (Generation Provider)
   ↓
EXPECTED ANSWER (Ground Truth)
   ↓
EVALUATION
```

---

## 📌 Case Classification Breakdown

### Case A / D — Successful Retrieval + Grounded Generation
- **Definition**: Vector store retrieves required source chunks; LLM generates accurate, grounded answer with citations.
- **Example Trace (Q001 - BMU Policy)**:
  - **Question**: *"What is the minimum CGPA requirement for BMU placement participation?"*
  - **Retrieved Chunk**: `SAMPLE_placement_policy.md > 1. Eligibility Criteria` (Score 0.82)
  - **Context**: *"A student must maintain a minimum Cumulative Grade Point Average (CGPA) of 6.5 on a 10-point scale..."*
  - **LLM Output**: *"The minimum CGPA requirement for BMU placement participation is 6.5 on a 10-point scale. [1] SAMPLE_placement_policy.md"*
  - **Evaluation**: **CORRECT** | Relevance: 5.0 | Grounded: True

### Case B — Irrelevant Retrieval
- **Definition**: Chunks retrieved have low semantic similarity or relate to a different section.
- **Example Trace (Q021 - Repo RAG)**:
  - **Question**: *"Which service file handles document text extraction?"*
  - **Retrieved Chunk**: Policy document chunk instead of repo architecture file.
  - **LLM Output**: Generates document title rather than python file basename (`app/documents.py`).
  - **Evaluation**: **INCORRECT** | Relevance: 2.0 | Actionable Fix: Add repository codebase embeddings to vector store.

### Case C — Missed Information (Retrieval Gap)
- **Definition**: Required information exists in knowledge base but fell outside top_k or distance threshold (`0.65`).
- **Example Trace (Q011 - Resume Match)**:
  - **Question**: *"Does my resume match the company JD requirements?"*
  - **Root Cause**: Candidate resume not uploaded prior to query execution.
  - **LLM Output**: *"Please upload a candidate resume first to compare against the Job Description."*
  - **Evaluation**: Expected behavior handled cleanly by system intent router.

### Case E — Correct Retrieval + Hallucinated Generation
- **Definition**: Correct chunks retrieved, but LLM ignores context constraints and fabricates unsupported details.
- **Example Trace (Q019 - Insufficient Info)**:
  - **Question**: *"What is the exact stipend amount for the 2030 summer internship?"*
  - **Context**: No mention of 2030 stipend in policy docs.
  - **LLM Output**: Fabricates arbitrary figure if system prompt constraints are weak.
  - **Fix Applied**: Added strict system prompt instruction forcing `"If sufficient context is missing, return 'Insufficient information in provided documents'."`

---

## 🛠️ Pipeline Improvements Implemented
1. **Section-Aware Chunking**: Preserves section heading titles in chunk metadata (`app/chunking.py`).
2. **Metadata Isolation Filters**: Strict `where={"company_id": "X"}` and `where={"candidate_id": "Y"}` queries prevent cross-tenant information leakage (`app/vectorstore.py`).
3. **Intent Orchestrator**: Bypasses RAG generation for structured tasks (deterministic eligibility, resume matching) to eliminate hallucination risks completely.
