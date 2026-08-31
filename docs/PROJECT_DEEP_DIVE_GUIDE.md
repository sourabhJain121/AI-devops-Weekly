# BMU Placement Intelligence & Interview Assistant: Master Technical Deep-Dive & Architecture Guide

Welcome to the comprehensive technical guide for the **BMU Placement Intelligence & Interview Assistant**. This guide is designed to teach you **every single concept, algorithm, engineering decision, mathematical formula, and model trade-off** implemented across this entire repository.

---

# Table of Contents
1. [Core Mission & System Philosophy](#1-core-mission--system-philosophy)
2. [High-Level Architecture & End-to-End Execution Flow](#2-high-level-architecture--end-to-end-execution-flow)
3. [The Three Knowledge Layers & Metadata Isolation](#3-the-three-knowledge-layers--metadata-isolation)
4. [Document Loading & Text Extraction (`app/documents.py`)](#4-document-loading--text-extraction-appdocumentspy)
5. [Section-Aware Heading-Bounded Chunking (`app/chunking.py`)](#5-section-aware-heading-bounded-chunking-appchunkingpy)
6. [Embeddings & Vector Database Mechanics (`app/vectorstore.py` & `app/llm.py`)](#6-embeddings--vector-database-mechanics-appvectorstorepy--appllmpy)
7. [The Retrieval-Augmented Generation (RAG) Pipeline (`app/rag.py`)](#7-the-retrieval-augmented-generation-rag-pipeline-appragpy)
8. [Intent Routing & Orchestration (`app/orchestrator.py`)](#8-intent-routing--orchestration-apporchestratorpy)
9. [Deterministic Placement Intelligence Services (`app/services/`)](#9-deterministic-placement-intelligence-services-appservices)
   - 9.1 [Candidate Profile Parsing (`resume.py`)](#91-candidate-profile-parsing-resumepy)
   - 9.2 [Company JD Processing (`jd.py`)](#92-company-jd-processing-jdpy)
   - 9.3 [Deterministic Eligibility Engine (`eligibility.py`)](#93-deterministic-eligibility-engine-eligibilitypy)
   - 9.4 [Skill Matching Engine (`matching.py`)](#94-skill-matching-engine-matchingpy)
   - 9.5 [Prioritized Skill Gap Analyzer (`skill_gap.py`)](#95-prioritized-skill-gap-analyzer-skill_gappy)
   - 9.6 [Evidence-Based Resume Improvement (`improvement.py`)](#96-evidence-based-resume-improvement-improvementpy)
   - 9.7 [Personalized Interview Prep Generator (`interview.py`)](#97-personalized-interview-prep-generator-interviewpy)
   - 9.8 [Codebase Repository Self-Answering RAG (`repo_rag.py`)](#98-codebase-repository-self-answering-rag-repo_ragpy)
10. [Multi-Model Evaluation & Model Differences](#10-multi-model-evaluation--model-differences)
    - 10.1 [Code Llama 7B (`codellama:7b`)](#101-code-llama-7b-codellama7b)
    - 10.2 [StarCoder2 3B (`starcoder2:3b`)](#102-starcoder2-3b-starcoder23b)
    - 10.3 [Qwen 2.5 0.5B (`qwen2.5:0.5b`)](#103-qwen-25-05b-qwen2505b)
    - 10.4 [Empirical Metrics & Trade-off Analysis](#104-empirical-metrics--trade-off-analysis)
11. [RAG Pipeline Failure & Success Modes (Cases A–E)](#11-rag-pipeline-failure--success-modes-cases-ae)
12. [Automated Testing & Multi-Tenant Isolation Verification](#12-automated-testing--multi-tenant-isolation-verification)
13. [Step-by-Step Reproduction & Developer Commands](#13-step-by-step-reproduction--developer-commands)

---

# 1. Core Mission & System Philosophy

### The Problem
Universities manage complex placement policies (CGPA requirements, backlogs, supplementary exams, attendance limits, offer acceptance restrictions). At the same time, companies have distinct job descriptions (JDs) with specific eligibility cut-offs and skill demands, while students submit unique resumes.

Traditional Large Language Models (LLMs) suffer from:
1. **Hallucinations**: Inventing cut-offs, dates, or non-existent candidate qualifications.
2. **Context Contamination**: Mixing Company A's requirements with Company B's interview criteria.
3. **Probabilistic Uncertainty**: Deciding critical student placement eligibility probabilistically (guessing CGPA) instead of deterministically calculating it against policy.

### The Solution Philosophy
This platform operates on three foundational engineering principles:
1. **Grounded RAG with Hard Distance Thresholding**: The LLM is only permitted to answer policy and document queries using verified context retrieved from ChromaDB. If the cosine distance exceeds `0.65`, the system returns a structured `"Insufficient information"` warning.
2. **Deterministic Rules Over Probabilistic LLMs**: Critical eligibility checks, skill match percentages, and backlog evaluations are calculated in Python using exact arithmetic, not generated probabilistically by an LLM.
3. **Strict Metadata Isolation**: Multi-tenant isolation is enforced at the database retrieval level using ChromaDB metadata filters (`company_id`, `candidate_id`, `source_type`).

---

# 2. High-Level Architecture & End-to-End Execution Flow

```
                               ┌────────────────────────────────────────┐
                               │   Student / User Web Dashboard (UI)    │
                               │        (app/static/index.html)         │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │       FastAPI API Router (main.py)     │
                               │   /api/query, /api/upload/*, /api/*    │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │    Orchestrator Router & Dispatcher    │
                               │          (app/orchestrator.py)         │
                               └───────┬───┬───┬───┬───┬───┬───┬────────┘
                                       │   │   │   │   │   │   │
        ┌──────────────────────────────┘   │   │   │   │   │   └──────────────────────────────┐
        ▼                                  ▼   ▼   ▼   ▼   ▼                                  ▼
┌───────────────────┐               ┌───────────────────────────────┐              ┌────────────────────┐
│   BMU Policy      │               │     Deterministic Engines     │              │  Repository RAG    │
│  Grounded RAG     │               │   eligibility.py, matching.py │              │   (repo_rag.py)    │
│   (app/rag.py)    │               │    skill_gap.py, resume.py    │              └─────────┬──────────┘
└─────────┬─────────┘               └──────────────┬────────────────┘                        │
          │                                        │                                         │
          │         ┌──────────────────────────────┴──────────────────────────────┐          │
          │         │        Persistent ChromaDB Vector Store Layer               │          │
          │         │     Filtered by: source_type, company_id, candidate_id      │          │
          │         └──────────────────────────────┬──────────────────────────────┘          │
          ▼                                        ▼                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Local Ollama LLM Service (app/llm.py)                                 │
│          Embeddings: nomic-embed-text | Generation: codellama:7b / starcoder2:3b / qwen2.5:0.5b       │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Trace of a Request
When a student types: `"What is the minimum CGPA requirement for placement?"`
1. **FastAPI (`app/main.py`)**: Receives the POST payload on `/api/query`.
2. **Orchestrator (`app/orchestrator.py`)**: `detect_intent()` analyzes the text and detects `BMU_POLICY`.
3. **Embedding (`app/llm.py`)**: Sends the query string to Ollama `/api/embed` using `nomic-embed-text` to generate a 768-dimensional float vector.
4. **Vector Retrieval (`app/vectorstore.py`)**: Queries ChromaDB for top 5 nearest chunks with `where={"source_type": "bmu_policy"}`.
5. **Distance Filter (`app/rag.py`)**: Chunks with distance $> 0.65$ are discarded.
6. **Prompt Assembly (`app/rag.py`)**: Context is formatted into `[1] SAMPLE_placement_policy.md > Section...`.
7. **Generation (`app/llm.py`)**: Ollama `/api/chat` generates the grounded answer with bracketed citations (`[1]`).
8. **UI Rendering (`app/static/index.html`)**: The response and clickable citation cards appear on screen.

---

# 3. The Three Knowledge Layers & Metadata Isolation

The vector database maintains three strictly segregated knowledge layers:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Permanent BMU Institutional Knowledge (source_type="bmu_policy")         │
│ Contains: General University Placement Rules, CGPA Limits, Backlog Regulations. │
│ Access: Shared across all queries. Immutable source of institutional truth.     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: Dynamic Company Knowledge (source_type="company_jd")                    │
│ Contains: Job Descriptions, Role Requirements, Specific Cut-offs, Tech Stacks.   │
│ Isolation: Tagged with company_id="company_a". Company A NEVER leaks to B.       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: Ephemeral Candidate Knowledge (source_type="resume")                    │
│ Contains: Parsed Resume Text, Projects, Extracted CGPA, Backlogs, Skills.        │
│ Isolation: Tagged with candidate_id="student_123".                               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### How Multi-Tenant Isolation Works
In `app/vectorstore.py`, queries pass a `where` dictionary:
```python
# Querying company JD for Google without seeing Microsoft data:
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"$and": [{"source_type": "company_jd"}, {"company_id": "google"}]}
)
```
If a chunk belonging to `microsoft` is retrieved, ChromaDB automatically ignores it because of the metadata filter.

---

# 4. Document Loading & Text Extraction (`app/documents.py`)

Real university documents come in multiple file formats: Markdown (`.md`), Plain Text (`.txt`), and Adobe PDF (`.pdf`).

```python
# app/documents.py core logic
def load_document(file_path: Path) -> list[DocumentPage]:
    suffix = file_path.suffix.lower()
    if suffix in [".md", ".txt"]:
        content = file_path.read_text(encoding="utf-8")
        return [DocumentPage(page_number=1, text=content)]
    elif suffix == ".pdf":
        reader = pypdf.PdfReader(str(file_path))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(DocumentPage(page_number=idx + 1, text=text))
        return pages
```

### Why Page-by-Page Extraction Matters
Rather than dumping the entire PDF into one giant string, extracting text page-by-page preserves the page number metadata (`page_number=idx + 1`). This allows citations in the UI to state exactly which page an institutional rule came from.

---

# 5. Section-Aware Heading-Bounded Chunking (`app/chunking.py`)

### The Flaw of Naive Chunking
Standard chunkers split text purely by character count (e.g., every 500 characters). This causes **Semantic Boundary Rupture**:
- A sentence starting with *"Students with > 2 backlogs are..."* gets cut in half.
- Heading `## 2. Backlog Rules` ends up in Chunk 1, while the actual rules end up in Chunk 2. The LLM loses the context of which heading the rule belonged to.

### The Section-Aware Solution
Our custom section-aware chunker (`app/chunking.py`) uses a two-level hierarchy:
1. **Heading Splitting**: It parses Markdown/Document headings (`#`, `##`, `###`, `Section X`) as hard semantic boundaries.
2. **Length-Bounded Token Chunking**: Within a single section, if the text exceeds `CHUNK_SIZE=800` characters, it splits on paragraph (`\n\n`) or sentence boundaries (`. `) with an overlap of `CHUNK_OVERLAP=120` characters.
3. **Heading Metadata Inheritance**: Every sub-chunk inherits the section name (`section="1. Eligibility for Placement"`).

```
Document:
# BMU Placement Policy
## 1. Eligibility Criteria  ───► Boundary: Never merges with Section 2
   Rule A...
   Rule B...
## 2. Backlog Rules        ───► New Semantic Section
   Rule C...
```

---

# 6. Embeddings & Vector Database Mechanics (`app/vectorstore.py` & `app/llm.py`)

### Embeddings (`nomic-embed-text`)
An embedding model converts human language into a high-dimensional mathematical coordinate vector where semantically related concepts are placed close together.

- Model: `nomic-embed-text`
- Vector Dimensions: **768**
- Distance Metric: **Cosine Distance** ($1 - \text{Cosine Similarity}$)
  $$\text{Cosine Distance} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  - A distance of `0.0` means identical meaning.
  - A distance of `1.0` means orthogonal (unrelated).

### ChromaDB Architecture
ChromaDB runs in-process and persists vectors and SQLite metadata to `storage/chroma/`.
Each stored chunk record contains:
- `id`: Unique hash (e.g., `SAMPLE_placement_policy.md_p1_c0`)
- `embedding`: 768-dimensional float array
- `document`: Raw text content of the chunk
- `metadata`: `{"document_name": "...", "section": "...", "page": 1, "source_type": "bmu_policy", "company_id": null}`

---

# 7. The Retrieval-Augmented Generation (RAG) Pipeline (`app/rag.py`)

When answering user queries, the RAG engine enforces a strict pipeline:

```
User Query: "What is the maximum allowed backlogs?"
   │
   ▼
1. Embed Query with nomic-embed-text
   │
   ▼
2. Vector Search in ChromaDB (Top K = 5)
   │
   ▼
3. Hard Distance Filter (MAX_DISTANCE = 0.65)
   ├── Distance <= 0.65 ──► Keep Chunk
   └── Distance > 0.65  ──► Discard (Noise Filter)
   │
   ▼
4. Context Check
   ├── No valid chunks left ──► Return: "Insufficient information in provided documents."
   └── Chunks exist         ──► Format Grounded Prompt with Citations [1], [2]...
   │
   ▼
5. LLM Generation via Ollama (/api/chat)
   │
   ▼
6. Output Parsing & Citation Linking in UI
```

### The System Prompt
```
You are the official BMU Placement Policy Assistant.
Answer the user's question using ONLY the provided context excerpts below.

CONTEXT:
{context}

RULES:
1. Every factual statement must cite its source in brackets, e.g. [1].
2. If the context does not contain enough information, state: 
   "I could not find sufficient information in the provided documents to answer this question reliably."
3. DO NOT invent rules, cut-offs, dates, or exceptions.
```

---

# 8. Intent Routing & Orchestration (`app/orchestrator.py`)

A user query can have different goals (asking about university policy, checking eligibility, analyzing resume skill gaps, or requesting interview questions). 

`app/orchestrator.py` uses pattern classification and keyword routing to dispatch requests to specialized sub-engines:

| Intent | Trigger Patterns | Target Sub-Engine |
| --- | --- | --- |
| `BMU_POLICY` | "cgpa requirement", "backlog rule", "policy" | `rag.answer()` |
| `ELIGIBILITY` | "am i eligible", "can i apply", "eligible" | `eligibility.evaluate_eligibility()` |
| `RESUME_MATCH` | "match score", "fit for role", "resume fit" | `matching.compare_resume_to_jd()` |
| `SKILL_GAP` | "missing skills", "skill gap", "what am i missing" | `skill_gap.analyze_skill_gaps()` |
| `RESUME_IMPROVEMENT` | "improve resume", "changes in resume", "suggestions" | `improvement.suggest_resume_improvements()` |
| `INTERVIEW_PREPARATION` | "interview prep", "interview questions" | `interview.generate_interview_prep()` |
| `COMPANY_JD` | "company require", "role requirement", "jd skill" | Isolated JD RAG |
| `REPO_RAG` | "which file", "architecture", "codebase" | `repo_rag.answer_codebase_question()` |

---

# 9. Deterministic Placement Intelligence Services (`app/services/`)

### 9.1 Candidate Profile Parsing (`resume.py`)
Extracts candidate attributes from uploaded PDF/text resumes using regex patterns:
- **CGPA**: Extracts numeric floats matching `\b(10|\d\.\d{1,2})\s*(?:cgpa|\/10)?\b`.
- **Backlogs**: Identifies active backlog counts (`0`, `1`, `2+`).
- **Skills**: Matches against a comprehensive tech catalog (Python, Java, FastAPI, SQL, Docker, React, etc.).

### 9.2 Company JD Processing (`jd.py`)
Parses company requirements and isolates them with `company_id`:
- Extracts minimum required CGPA (e.g. `7.0`).
- Extracts maximum permitted backlogs (e.g. `0` or `1`).
- Extracts required skills and preferred qualifications.

### 9.3 Deterministic Eligibility Engine (`eligibility.py`)
Combines BMU University Policy + Company JD + Candidate Profile using **deterministic arithmetic**:
```python
# Exact calculation, NOT LLM guessing
is_eligible = True
reasons = []

if candidate.cgpa < bmu_min_cgpa:
    is_eligible = False
    reasons.append(f"Candidate CGPA ({candidate.cgpa}) is below BMU threshold ({bmu_min_cgpa}).")

if candidate.cgpa < jd_min_cgpa:
    is_eligible = False
    reasons.append(f"Candidate CGPA ({candidate.cgpa}) is below company cutoff ({jd_min_cgpa}).")

if candidate.backlogs > bmu_max_backlogs:
    is_eligible = False
    reasons.append(f"Active backlogs ({candidate.backlogs}) exceed BMU maximum ({bmu_max_backlogs}).")
```

### 9.4 Skill Matching Engine (`matching.py`)
Calculates the exact percentage overlap between candidate skills and company JD requirements:
$$\text{Match Score} = \left(\frac{|\text{Candidate Skills} \cap \text{Required Skills}|}{|\text{Required Skills}|}\right) \times 100$$

### 9.5 Prioritized Skill Gap Analyzer (`skill_gap.py`)
Computes the set difference:
$$\text{Skill Gaps} = \text{Required Skills} \setminus \text{Candidate Skills}$$
Categorizes missing skills into **Critical / Must-Have** vs **Preferred / Good-to-Have**.

### 9.6 Evidence-Based Resume Improvement (`improvement.py`)
Provides actionable recommendations based on missing JD skills. **Ethical Guardrail**: Strictly forbids hallucinating or inventing fake job experience.

### 9.7 Personalized Interview Prep Generator (`interview.py`)
Generates role-tailored technical and behavioral questions based on:
1. Candidate's declared projects and verified skills.
2. Company's target domain (Backend, Frontend, Cloud, Data).
3. Questions addressing identified skill gap areas.

### 9.8 Codebase Repository Self-Answering RAG (`repo_rag.py`)
Enables the system to answer architectural questions about its own codebase and multi-module execution flows.

---

# 10. Multi-Model Evaluation & Model Differences

To study how different local LLM architectures perform under identical RAG conditions, we evaluated 3 distinct models:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BENCHMARK SETUP (100% CONSTANT)                   │
│  Knowledge Base: 27 Chunks | Embeddings: nomic-embed-text | TOP_K: 5       │
│  Distance Threshold: 0.65  | Benchmark: 25 Placement Intelligence Questions │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   Code Llama 7B               StarCoder2 3B              Qwen 2.5 0.5B
   (codellama:7b)             (starcoder2:3b)            (qwen2.5:0.5b)
```

### 10.1 Code Llama 7B (`codellama:7b`)
- **Developer**: Meta AI
- **Parameters**: 7 Billion
- **Architecture**: LLaMA-2 transformer with 16k context window and code infilling support.
- **Strengths**: Deep reasoning, superior syntax comprehension, and strict adherence to bracketed citation formatting (`[1]`, `[2]`).
- **Weaknesses**: Requires ~3.8 GB storage and higher RAM footprint.

### 10.2 StarCoder2 3B (`starcoder2:3b`)
- **Developer**: BigCode / Hugging Face & ServiceNow
- **Parameters**: 3 Billion
- **Architecture**: Compact transformer trained on 600+ programming languages.
- **Strengths**: Fast execution, lower memory overhead (~1.7 GB).
- **Weaknesses**: Occasionally truncates natural language reasoning on complex multi-rule policy queries.

### 10.3 Qwen 2.5 0.5B (`qwen2.5:0.5b`)
- **Developer**: Alibaba Cloud
- **Parameters**: 0.5 Billion (490 Million)
- **Architecture**: Ultra-compact lightweight model designed for resource-constrained edge devices.
- **Strengths**: Extremely small disk footprint (~397 MB) and minimal RAM utilization.
- **Weaknesses**: Prone to overly brief responses; occasionally requires prompt reinforcement to cite sources.

---

### 10.4 Empirical Metrics & Trade-off Analysis

| Measured Metric | Code Llama 7B | StarCoder2 3B | Qwen 2.5 0.5B | Measurement Method |
| --- | ---: | ---: | ---: | --- |
| **Accuracy (%)** | **52.0%** (13/25) | **48.0%** (12/25) | **48.0%** (12/25) | Matches Ground Truth / 25 Qs |
| **Relevance Score (1.0–5.0)** | **3.80** | **3.56** | **3.40** | Documented 1–5 Rubric |
| **Retrieval Hit Rate (%)** | **80.0%** (20/25) | **80.0%** (20/25) | **80.0%** (20/25) | Required Sources in Vector Hits |
| **Observed Hallucination Rate (%)** | **8.0%** (2/25) | **8.0%** (2/25) | **8.0%** (2/25) | Unsupported Factual Claims |
| **Test-Pass Rate for Code Gen** | **N/A** | **N/A** | **N/A** | Placement Q&A is not Code Gen |
| **Avg Total Latency (s)** | **3.36s** | **4.02s** | **13.09s** | Measured via `time.perf_counter()` |
| **Avg Retrieval Latency (s)** | **0.002s** | **0.002s** | **0.002s** | ChromaDB Vector Search Time |
| **Avg Generation Latency (s)** | **3.35s** | **4.01s** | **13.08s** | Ollama LLM Inference Time |
| **Avg Prompt Tokens** | **653** | **653** | **653** | Ollama Telemetry Metadata |
| **Avg Completion Tokens** | **61.8** | **53.5** | **87.0** | Ollama Telemetry Metadata |
| **Avg CPU Utilization (%)** | **42.5%** | **38.0%** | **31.2%** | Measured via `psutil.cpu_percent()` |
| **Avg RAM Usage (MB)** | **14,500 MB** | **14,200 MB** | **13,800 MB** | Measured via `psutil.virtual_memory()` |
| **GPU / VRAM Usage** | **N/A** | **N/A** | **N/A** | ARM64 Unified Memory CPU |

### Recommendation Conclusion
> **Code Llama 7B is the recommended model.** It achieves the highest accuracy (**52.0%**), highest relevance (**3.80/5.0**), and fastest total latency (**3.36s**) due to optimized Metal acceleration in Ollama on ARM64 hardware.

---

# 11. RAG Pipeline Failure & Success Modes (Cases A–E)

To understand RAG behavior, we analyzed five representative scenarios:

### Case A: Relevant Retrieval & Success (`Q001`)
- **Question**: *"What is the minimum CGPA requirement for BMU placement?"*
- **Retrieved Chunk**: `SAMPLE_placement_policy.md > 1. Eligibility (Score: 0.82)`
- **LLM Output**: *"The minimum CGPA requirement is 6.5 on a 10-point scale. [1]"*
- **Outcome**: **SUCCESS** — Precise retrieval and accurate grounded generation.

### Case B: Irrelevant Retrieval (`Q021`)
- **Question**: *"Which service file handles document text extraction?"*
- **Retrieved Chunk**: Policy document section mentioning *"Students must submit documents"*.
- **Outcome**: **REASONABLE MISS** — Keyword collision on the word *"document"*. The vector store retrieved a university policy document because codebase files were not in the policy index.

### Case C: Missed Information via Threshold (`Q011`)
- **Question**: *"Does my resume match the company JD?"* (When no resume is uploaded).
- **Outcome**: **SAFE FALLBACK** — Distance threshold ($> 0.65$) discarded low-relevance chunks. Intent router returned a structured prompt asking the candidate to upload a resume.

### Case D: Correct Retrieval $\rightarrow$ Correct Answer
- Demonstrates optimal RAG execution where high-similarity chunks directly support the final response with full bracketed citations.

### Case E: Hallucination Prevention on Unanswerable Queries (`Q019`)
- **Question**: *"What is the exact stipend amount for the 2030 summer internship?"*
- **Retrieved Chunk**: General rules from 2024.
- **LLM Output**: *"I could not find sufficient information in the provided documents to answer this question reliably."*
- **Outcome**: **GROUNDING SUCCESS** — System prompt prevented the model from fabricating a 2030 date or stipend amount.

---

# 12. Automated Testing & Multi-Tenant Isolation Verification

The automated test suite in `tests/` contains **15 test cases** covering every layer of the application:

```
============================= test session starts ==============================
collected 15 items

tests/test_chunking.py .                                                 [  6%]
tests/test_e2e.py .                                                      [ 13%]
tests/test_eligibility.py .                                              [ 20%]
tests/test_failures.py .....                                             [ 53%]
tests/test_integration.py ...                                            [ 73%]
tests/test_isolation.py ...                                              [ 93%]
tests/test_matching.py .                                                 [100%]

======================= 15 passed in 21.00s =======================
```

### Test Coverage Breakdown
1. `tests/test_chunking.py`: Verifies that heading boundaries (`## Section`) are never broken across chunk splits.
2. `tests/test_eligibility.py`: Verifies deterministic calculation of eligible vs ineligible students across various CGPA and backlog thresholds.
3. `tests/test_matching.py`: Validates exact mathematical calculation of skill match scores and missing skill lists.
4. `tests/test_isolation.py`: Tests that Company A's documents are never returned when querying with `company_id="company_b"`, and Candidate profiles remain isolated.
5. `tests/test_integration.py`: Validates FastAPI REST endpoints (`/api/health`, `/api/query`, `/api/upload/*`).
6. `tests/test_e2e.py`: Executes the entire end-to-end flow: candidate registration $\rightarrow$ JD upload $\rightarrow$ eligibility check $\rightarrow$ match score $\rightarrow$ interview question generation.
7. `tests/test_failures.py`: Verifies safe fallback behaviors when querying missing or unindexed context.

---

# 13. Step-by-Step Reproduction & Developer Commands

### 1. Prerequisites & Environment Setup
```bash
# Clone and enter workspace
cd /Users/sourabh/Downloads/idtt

# Create Python 3.13 virtual environment
python3.13 -m venv .venv13
source .venv13/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull Required Ollama Models
```bash
ollama pull nomic-embed-text
ollama pull codellama:7b
ollama pull starcoder2:3b
ollama pull qwen2.5:0.5b
```

### 3. Ingest Permanent BMU Knowledge Base
```bash
.venv13/bin/python scripts/ingest.py --reset
```

### 4. Start the Application Server
```bash
.venv13/bin/python -m uvicorn app.main:app --port 8000 --reload
```
Open **http://localhost:8000** to access the interactive web interface.

### 5. Run the Automated Test Suite
```bash
.venv13/bin/python -m pytest
```

### 6. Run the Multi-Model Evaluation Benchmark
```bash
PYTHONUNBUFFERED=1 .venv13/bin/python evaluation/eval.py --all
```

### 7. Run the Codebase Self-Answering RAG Benchmark
```bash
PYTHONUNBUFFERED=1 .venv13/bin/python app/services/repo_rag.py
```

---

*This guide represents the complete technical architecture and reference manual for the BMU Placement Intelligence Platform.*
