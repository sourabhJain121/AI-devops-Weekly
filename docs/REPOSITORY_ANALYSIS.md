# Repository & Codebase RAG Understanding Report

## 1. Overview

This document presents the findings from evaluating the system's self-answering Codebase RAG capability using the 10-question benchmark in [`evaluation/repository_questions.json`](file:///Users/sourabh/Downloads/idtt/evaluation/repository_questions.json).

The objective is to investigate how far the local LLM + Architecture RAG system can go toward answering questions requiring understanding of:
- Specific module files
- Single-file component responsibilities
- Multi-file execution flow tracing
- Component-to-component interactions

---

## 2. Evaluation Results Summary

- **Total Repository Benchmark Questions**: 10
- **Single-File Questions**: 5
- **Multi-File Execution Flow Questions**: 5
- **Accuracy**: **90.0%** (9 / 10 correct)
- **Output Artifact**: [`evaluation/repository_results.json`](file:///Users/sourabh/Downloads/idtt/evaluation/repository_results.json)

---

## 3. Sample Case Traces

### Case 1 — Single-File Module Discovery (`REPO_003`)
- **Question**: *"Where is the Ollama LLM called in the codebase?"*
- **Target File**: `app/llm.py`
- **Model Output**: *"Ollama is called in the `app/llm.py` module. This file generates embeddings via `/api/embed` and executes LLM generation via `/api/chat`."*
- **Result**: **SUCCESS** — Precise file and function mapping.

### Case 2 — Multi-File Execution Flow Tracing (`REPO_004`)
- **Question**: *"What happens after a user submits a query to FastAPI?"*
- **Target Files**: `app/main.py` -> `app/orchestrator.py`
- **Model Output**: *"When a query is submitted to FastAPI, `app/main.py` receives the request at `/api/query` and delegates execution to `orchestrator.route_and_execute` in `app/orchestrator.py`."*
- **Result**: **SUCCESS** — Traced cross-module dependency.

### Case 3 — Full Multi-Module Pipeline Tracing (`REPO_010`)
- **Question**: *"What happens from the moment a user submits `/api/query` until the final answer is returned?"*
- **Target Files**: `main.py` -> `orchestrator.py` -> `rag.py` / services -> `vectorstore.py` -> `llm.py`
- **Model Output**: *"1. `app/main.py` receives endpoint request. 2. `app/orchestrator.py` detects intent. 3. Query is routed to `app/rag.py` or specialized service. 4. Chunks retrieved from `app/vectorstore.py`. 5. Prompt formatted and sent to `app/llm.py`."*
- **Result**: **SUCCESS** — Accurately traced full 5-module flow.

---

## 4. Strengths & Known Limitations

### Strengths
1. **Module Responsibility Mapping**: Accurately maps functionality to specific codebase files (`app/documents.py`, `app/chunking.py`, `app/vectorstore.py`, `app/eligibility.py`).
2. **Data Isolation Flow**: Explains how `company_id` and `candidate_id` metadata filters isolate multi-tenant data across `vectorstore.py` and service layers.

### Limitations
- **Granular AST Line Tracing**: While high-level file relationships and execution flows are traced accurately, line-level AST variable mutations require dedicated static analysis tools (e.g. Sourcegraph / LSP indexers) covered in future modules.
