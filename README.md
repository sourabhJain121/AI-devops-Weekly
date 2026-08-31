# 🎓 BMU Placement Intelligence & Interview Assistant

A production-grade, grounded **Placement Intelligence Platform & Multi-Model Evaluation Harness** designed for BML Munjal University (BMU) students, placement coordinators, and recruiters. Built with **FastAPI**, **ChromaDB**, and local **Ollama** LLM inference engines, this platform unifies institutional policy Q&A, candidate resume parsing, company JD analysis, metadata-isolated multi-source RAG, deterministic eligibility checks, structured resume vs. JD matching, prioritized skill gap analysis, evidence-grounded resume refinement, role-tailored interview preparation, and self-answering codebase repository understanding.

> 📖 **Comprehensive Technical Guide**: For an exhaustive deep-dive explaining every mathematical formulation, chunking algorithm, vector math, model architecture comparison, and deterministic service in this repository, see the [Master Technical Deep-Dive & Architecture Guide](file:///Users/sourabh/Downloads/idtt/docs/PROJECT_DEEP_DIVE_GUIDE.md).

---

## 📑 Table of Contents

1. [Architecture & System Design](#-1-architecture--system-design)
2. [Three-Layer Knowledge Architecture](#-2-three-layer-knowledge-architecture)
3. [Repository Codebase Map](#-3-repository-codebase-map)
4. [Interactive Web Dashboard: Page & Button Guide](#-4-interactive-web-dashboard-page--button-guide)
   - [Global Header & Controls](#global-header--controls)
   - [Page 1: 💬 Assistant & Policy](#page-1--assistant--policy)
   - [Page 2: 📄 Candidate Resume](#page-2--candidate-resume)
   - [Page 3: 🏢 Company JDs](#page-3--company-jds)
   - [Page 4: 📊 Placement Intelligence](#page-4--placement-intelligence)
   - [Page 5: 🏆 Model Comparison](#page-5--model-comparison)
   - [Page 6: 📋 Question Matrix](#page-6--question-matrix)
   - [Page 7: 🔬 RAG Analysis (Cases A–E)](#page-7--rag-analysis-cases-ae)
   - [Page 8: 💻 Codebase RAG](#page-8--codebase-rag)
   - [Page 9: 📚 Knowledge Base](#page-9--knowledge-base)
5. [Complete REST API Reference](#-5-complete-rest-api-reference)
6. [Quickstart & Installation Guide](#-6-quickstart--installation-guide)
7. [Multi-Model Comparative Evaluation & Benchmarks](#-7-multi-model-comparative-evaluation--benchmarks)
8. [Automated Test Suite](#-8-automated-test-suite)
9. [Docker Deployment](#-9-docker-deployment)
10. [Core Grounding & Safety Guardrails](#-10-core-grounding--safety-guardrails)

---

## 🏗️ 1. Architecture & System Design

```text
                               ┌───────────────────────────┐
                               │   Interactive Web UI      │
                               │  (app/static/index.html)  │
                               └─────────────┬─────────────┘
                                             │ HTTP Requests
                                             ▼
                               ┌───────────────────────────┐
                               │   FastAPI API (main.py)   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Orchestrator Router       │
                               │ (app/orchestrator.py)     │
                               └──────┬───┬───┬───┬───┬────┘
                                      │   │   │   │   │
           ┌──────────────────────────┘   │   │   │   └──────────────────────────┐
           ▼                              ▼   ▼   ▼                              ▼
┌─────────────────────┐       ┌──────────────────────┐               ┌─────────────────────┐
│ BMU Policy RAG      │       │ Deterministic Engine │               │ Intelligence        │
│ (app/rag.py)        │       │ (eligibility.py)     │               │ Services            │
└──────────┬──────────┘       └──────────┬───────────┘               │ (matching.py,       │
           │                             │                           │  skill_gap.py,      │
           │     ┌───────────────────────┴───────────────────────┐   │  improvement.py,    │
           │     │ Metadata-Filtered Chroma Vector Store         │   │  interview.py,      │
           │     │ (source_type, company_id, candidate_id)       │   │  repo_rag.py)       │
           ▼     │ (app/vectorstore.py)                          │   └──────────┬──────────┘
 ┌───────────────────┐                                           │              │
 │ Ollama Service    │◄──────────────────────────────────────────┴──────────────┘
 │ (app/llm.py)      │ (codellama:7b | starcoder2:3b | qwen2.5:0.5b)
 └───────────────────┘
```

---

## 🛡️ 2. Three-Layer Knowledge Architecture

The system enforces strict multi-tenant metadata isolation across three distinct tiers of data:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Permanent BMU Institutional Policy (source_type="bmu_policy")   │
│ - Placement rules, attendance limits, minimum CGPA, active backlogs      │
│ - Ultimate source of truth for university compliance                     │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: Dynamic Company Knowledge (source_type="company_jd")            │
│ - Job descriptions, eligibility cutoffs, required skills, CTC packages  │
│ - Isolated per company_id (Prevents cross-company contamination)        │
├──────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: Candidate Knowledge (source_type="resume")                      │
│ - Extracted candidate entities (CGPA, backlogs, skills, projects)        │
│ - Isolated per candidate_id (Enforces student data privacy)              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 3. Repository Codebase Map

| File / Folder Path | Component Type | Responsibility |
| :--- | :--- | :--- |
| [`app/main.py`](file:///Users/sourabh/Downloads/idtt/app/main.py) | **API Entrypoint** | FastAPI server routes, static file mounting, upload handlers, and evaluation endpoints. |
| [`app/config.py`](file:///Users/sourabh/Downloads/idtt/app/config.py) | **Configuration** | Central settings loader for `.env` (`OLLAMA_URL`, `LLM_MODEL`, `CHROMA_DIR`, thresholds). |
| [`app/documents.py`](file:///Users/sourabh/Downloads/idtt/app/documents.py) | **Document Loader** | Page-by-page text extraction for PDF, Markdown, and TXT files using PyPDF. |
| [`app/chunking.py`](file:///Users/sourabh/Downloads/idtt/app/chunking.py) | **Chunking Engine** | Section-aware, heading-bounded text chunker preserving policy boundaries. |
| [`app/vectorstore.py`](file:///Users/sourabh/Downloads/idtt/app/vectorstore.py) | **Vector Store** | ChromaDB interface supporting cosine similarity and multi-field metadata filtering. |
| [`app/llm.py`](file:///Users/sourabh/Downloads/idtt/app/llm.py) | **LLM Provider** | Client wrapper for local Ollama models (`codellama:7b`, `starcoder2:3b`, `qwen2.5:0.5b`). |
| [`app/rag.py`](file:///Users/sourabh/Downloads/idtt/app/rag.py) | **RAG Core** | Grounded retrieval, prompt assembler, citation manager, and distance thresholding (`0.65`). |
| [`app/orchestrator.py`](file:///Users/sourabh/Downloads/idtt/app/orchestrator.py) | **Intent Router** | Rule-based and semantic classifier routing queries to 8 specialized handlers. |
| [`app/services/eligibility.py`](file:///Users/sourabh/Downloads/idtt/app/services/eligibility.py) | **Deterministic Engine** | Zero-hallucination mathematical evaluator for CGPA, backlogs, and policy compliance. |
| [`app/services/resume.py`](file:///Users/sourabh/Downloads/idtt/app/services/resume.py) | **Resume Parser** | Entity extractor for candidate profiles (skills, CGPA, graduation year, degree). |
| [`app/services/jd.py`](file:///Users/sourabh/Downloads/idtt/app/services/jd.py) | **JD Parser** | Company job description parser and isolated metadata store manager. |
| [`app/services/matching.py`](file:///Users/sourabh/Downloads/idtt/app/services/matching.py) | **Match Engine** | Resume vs. JD requirement comparison and grounded compatibility scoring. |
| [`app/services/skill_gap.py`](file:///Users/sourabh/Downloads/idtt/app/services/skill_gap.py) | **Skill Gap Engine** | Categorizes missing skills into Critical, Secondary, and Recommended items. |
| [`app/services/improvement.py`](file:///Users/sourabh/Downloads/idtt/app/services/improvement.py) | **Resume Advisor** | Action-oriented resume enhancement tips with a strict anti-fabrication policy. |
| [`app/services/interview.py`](file:///Users/sourabh/Downloads/idtt/app/services/interview.py) | **Interview Prep** | Role-tailored technical, coding, and behavioral interview question generator. |
| [`app/services/repo_rag.py`](file:///Users/sourabh/Downloads/idtt/app/services/repo_rag.py) | **Codebase RAG** | Self-answering repository architecture search engine and evaluation harness. |
| [`app/static/index.html`](file:///Users/sourabh/Downloads/idtt/app/static/index.html) | **Frontend UI** | Modern, reactive single-page interface with real-time telemetry and 9 workspace panes. |
| [`evaluation/eval.py`](file:///Users/sourabh/Downloads/idtt/evaluation/eval.py) | **Evaluation Harness** | Multi-model benchmark evaluator measuring accuracy, relevance, hit-rates, and latencies. |
| [`evaluation/questions.json`](file:///Users/sourabh/Downloads/idtt/evaluation/questions.json) | **Benchmark Dataset** | 25 standardized placement benchmark questions spanning 10 key categories. |
| [`tests/`](file:///Users/sourabh/Downloads/idtt/tests) | **Test Suite** | Comprehensive unit, integration, isolation, and end-to-end pytest test cases. |

---

## 🖥️ 4. Interactive Web Dashboard: Page & Button Guide

The web dashboard is organized into **9 distinct functional panes** grouped across three primary sidebar sections.

---

### Global Header & Controls

| Control / Element | Element ID | Function & Behavior |
| :--- | :--- | :--- |
| **Brand Logo & Title** | `.brand` | Displays the platform title and BMU brand icon. |
| **Active Model Selector** | `#model-select` | Dropdown allowing instantaneous switching between `CodeLlama 7B (codellama:7b)`, `StarCoder2 3B (starcoder2:3b)`, and `Qwen 0.5B (qwen2.5:0.5b)`. Invokes `onModelChange()`. |
| **Ollama Status Pill** | `#odot`, `#omodel` | Real-time healthcheck indicator showing whether the local Ollama inference service is online and displaying the active model. |
| **Knowledge Base Stats** | `#kb-stats` | Real-time counter showing the total number of vector chunks and documents indexed in ChromaDB. |

---

### Page 1: 💬 Assistant & Policy
- **Navigation Identifier**: `switchNav('assistant')`
- **Pane Container**: `#pane-assistant`
- **Purpose**: Interactive conversational interface for asking institutional placement policy questions, checking general eligibility criteria, and exploring guidelines with grounded source citations.

#### Interactive Buttons & Controls on Page 1:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`🎓 Minimum CGPA criteria`** | Prompt Chip | Fills chat input with `"What is the minimum CGPA requirement for placement?"` and submits query automatically. |
| **`📊 Am I Eligible?`** | Prompt Chip | Fills chat input with `"Am I eligible to apply for placement drives?"` and triggers policy evaluation. |
| **`🎯 Resume Match Score`** | Prompt Chip | Fills chat input with `"Does my resume match this JD?"` and initiates candidate match workflow. |
| **`⚠️ Missing Skills`** | Prompt Chip | Fills chat input with `"What skills am I missing for this company?"` and queries skill gap analysis. |
| **`🎯 Interview Questions`** | Prompt Chip | Fills chat input with `"What interview questions should I prepare?"` and launches interview prep generator. |
| **`Ask` (Submit Button)** | Form Submit Button | Transmits the user query to the `/api/query` orchestrator, displays model inference progress, renders the generated response, and populates the **Cited Evidence** panel with relevant vector excerpts. |
| **`Cited Evidence Panel`** | Dynamic Sidebar (`#sources-list`) | Displays exact document titles, section headers, cosine similarity match percentages, and text snippets used to ground the LLM's response. |

---

### Page 2: 📄 Candidate Resume
- **Navigation Identifier**: `switchNav('resume')`
- **Pane Container**: `#pane-resume`
- **Purpose**: Uploads and extracts candidate resume data into a structured candidate profile (CGPA, backlogs, graduation year, degree, branch, skills, and projects) stored in Layer 3 (`source_type="resume"`).

#### Interactive Buttons & Controls on Page 2:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`Click to Upload Candidate Resume`** | Upload Dropzone (`.upload-box`) | Opens the local file selector for `.pdf` or `.txt` resumes. Upon selection, invokes `uploadResume()`, sending a `POST` request to `/api/upload/resume`. |
| **`Candidate Profile Viewer`** | Dynamic Panel (`#resume-profile-view`) | Parses and renders extracted structured fields: Candidate Name, Degree, Branch, CGPA, Active Backlogs, Graduation Year, Contact Info, and Badges for all extracted technical skills. |

---

### Page 3: 🏢 Company JDs
- **Navigation Identifier**: `switchNav('jd')`
- **Pane Container**: `#pane-jd`
- **Purpose**: Manages company job descriptions with strict multi-tenant isolation (`company_id`), ensuring individual company requirements do not leak into other company profiles.

#### Interactive Buttons & Controls on Page 3:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`Click to Upload Company Job Description`** | Upload Dropzone (`.upload-box`) | Opens local file selector for `.pdf`, `.md`, or `.txt` JD files. Triggers `uploadJD()`, sending a `POST` request to `/api/upload/company-jd` to index the JD with isolated metadata. |
| **`Isolated Company Requirements Viewer`** | Dynamic Panel (`#jd-list-view`) | Lists all currently indexed company JDs along with their unique `company_id`, document name, and total indexed vector chunks. |

---

### Page 4: 📊 Placement Intelligence
- **Navigation Identifier**: `switchNav('intelligence')`
- **Pane Container**: `#pane-intelligence`
- **Purpose**: Dedicated execution hub for deterministic policy checks, resume matching, skill gap discovery, resume improvement, and role-tailored interview preparation.

#### Interactive Buttons & Controls on Page 4:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`📊 Deterministic Eligibility Analysis`** | Action Button | Sends `POST` to `/api/analyze/eligibility`. Evaluates candidate CGPA and active backlogs against BMU policy rules deterministically (zero hallucination) and renders a structured verdict. |
| **`🎯 Structured Resume vs JD Match`** | Action Button | Sends `POST` to `/api/analyze/resume-jd`. Performs semantic and keyword comparison between candidate skills and company JD requirements, returning match percentages and evidence. |
| **`⚠️ Prioritized Skill Gap Analysis`** | Action Button | Sends `POST` to `/api/analyze/skill-gap`. Computes missing technical skills and breaks them down into **Critical Requirements**, **Secondary Requirements**, and **Recommended Next Steps**. |
| **`💡 Resume Improvement Plan`** | Action Button | Sends `POST` to `/api/analyze/resume-improvement`. Produces bullet-by-bullet suggestions to strengthen resume phrasing without fabricating unverified experience. |
| **`🎯 Personalized Interview Prep`** | Action Button | Sends `POST` to `/api/interview/prepare`. Generates targeted technical questions, algorithmic problems, and behavioral questions tailored to the candidate's actual projects and the company's JD. |

---

### Page 5: 🏆 Model Comparison
- **Navigation Identifier**: `switchNav('eval-summary')`
- **Pane Container**: `#pane-eval-summary`
- **Purpose**: Quantitative benchmark evaluation hub presenting comparative metrics across all 3 evaluated local LLM models (`codellama:7b`, `starcoder2:3b`, `qwen2.5:0.5b`).

#### Interactive Buttons & Controls on Page 5:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`🔄 Run / Refresh Evaluation`** | Action Button (`#eval-spinner`) | Sends a `POST` request to `/api/evaluation/run` to execute the automated multi-model evaluation harness across all 25 benchmark questions and dynamically re-renders the comparison table and charts. |
| **`Accuracy (%) Comparison Chart`** | Visual Bar Chart (`#chart-accuracy`) | Visually compares overall accuracy percentages across CodeLlama 7B, StarCoder2 3B, and Qwen 0.5B. |
| **`Average Latency (s) Comparison Chart`** | Visual Bar Chart (`#chart-latency`) | Visually depicts average total response latency across all three evaluated models. |
| **`Quantitative Benchmark Metrics Matrix`** | Interactive Data Table (`#matrix-table-container`) | Displays side-by-side metrics including Accuracy, Relevance Score, Retrieval Hit Rate, Hallucination Rate, Latency breakdown, Token usage, and RAM telemetry. |

---

### Page 6: 📋 Question Matrix
- **Navigation Identifier**: `switchNav('eval-questions')`
- **Pane Container**: `#pane-eval-questions`
- **Purpose**: Displays the complete 25-question evaluation benchmark dataset across 10 placement categories, specifying exact question prompts and expected ground truth behaviors (`GROUNDED_ANSWER` vs `INSUFFICIENT_INFORMATION`).

#### Interactive Controls on Page 6:
| Element Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`Benchmark Questions Table`** | Data Table (`#questions-table-container`) | Loads from `/api/evaluation/questions` and displays QID, Category Badge, Question Text, and Ground Truth Expected Behavior. |

---

### Page 7: 🔬 RAG Analysis (Cases A–E)
- **Navigation Identifier**: `switchNav('eval-rag')`
- **Pane Container**: `#pane-eval-rag`
- **Purpose**: Deep-dive trace inspector examining evidence movement through the RAG pipeline across 5 critical failure and success modes (Cases A through E).

#### Interactive Controls on Page 7:
| Element Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`RAG Pipeline Case Cards`** | Card Deck (`#rag-cases-container`) | Loads from `/api/evaluation/rag-analysis`. Inspects Retrieval Quality, Context Snippet sent to LLM, Model Output, and Pipeline Analysis for Cases A through E with pass/fail badges. |

---

### Page 8: 💻 Codebase RAG
- **Navigation Identifier**: `switchNav('eval-repo')`
- **Pane Container**: `#pane-eval-repo`
- **Purpose**: Repository self-answering architecture benchmark evaluating how accurately the assistant understands its own codebase, module responsibilities, and 5-stage execution pipelines.

#### Interactive Controls on Page 8:
| Element Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`Codebase Understanding Matrix`** | Data Table (`#repo-eval-container`) | Loads from `/api/evaluation/repository` to display 10 architecture questions, target file paths, multi-file tracing flags, and verification outcomes. |

---

### Page 9: 📚 Knowledge Base
- **Navigation Identifier**: `switchNav('knowledge')`
- **Pane Container**: `#pane-knowledge`
- **Purpose**: Administrative inspector for indexed documents and vector chunk allocations across the ChromaDB database.

#### Interactive Buttons & Controls on Page 9:
| Button / Control Name | Type | Action & Outcome |
| :--- | :--- | :--- |
| **`🔄 Re-ingest Knowledge`** | Action Button | Triggers a `POST` request to `/api/ingest` to rescan the `knowledge/` directory, re-chunk documents, and regenerate vector embeddings. |
| **`Document Chunk Inventory`** | List View (`#doc-list-full`) | Displays each ingested document name, source type (`bmu_policy`, `company_jd`, `resume`), and total vector chunks indexed. |

---

## 📡 5. Complete REST API Reference

| Endpoint | Method | Request Payload / Params | Purpose & Service |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | *None* | Returns server status, Ollama daemon reachability, available models, and vector chunk statistics. |
| `/api/documents` | `GET` | *None* | Lists all indexed documents, chunk counts, and source categories. |
| `/api/documents/{name}` | `DELETE` | `name` (path parameter) | Deletes all indexed chunks associated with a specific document from ChromaDB. |
| `/api/ingest` | `POST` | *None* | Scans the `knowledge/` directory and performs incremental chunking and embedding. |
| `/api/upload` | `POST` | `multipart/form-data` (`file`) | Generic document upload and embedding into ChromaDB. |
| `/api/upload/resume` | `POST` | `multipart/form-data` (`file`) | Uploads candidate resume, parses structured profile, and stores in session. |
| `/api/upload/company-jd` | `POST` | `multipart/form-data` (`file`) | Uploads company JD with isolated metadata tagging (`company_id`). |
| `/api/query` | `POST` | `{"message": "...", "model": "..."}` | Unified Orchestrated Assistant endpoint with intelligent routing and grounded citations. |
| `/api/chat` | `POST` | `{"message": "...", "model": "..."}` | Direct Grounded RAG Chat endpoint querying ChromaDB and Ollama. |
| `/api/retrieve` | `POST` | `{"message": "...", "top_k": 5}` | Retrieval-only endpoint returning raw vector hits and cosine similarity scores. |
| `/api/analyze/eligibility` | `POST` | `{"candidate_id": "..."}` | Deterministic BMU placement eligibility analysis. |
| `/api/analyze/resume-jd` | `POST` | `{"candidate_id": "...", "company_id": "..."}` | Structured candidate resume vs. company JD skill matching. |
| `/api/analyze/skill-gap` | `POST` | `{"candidate_id": "...", "company_id": "..."}` | Prioritized skill gap breakdown (Critical, Secondary, Recommended). |
| `/api/analyze/resume-improvement` | `POST` | `{"candidate_id": "..."}` | Evidence-grounded resume enhancement recommendations. |
| `/api/interview/prepare` | `POST` | `{"candidate_id": "...", "company_id": "..."}` | Personalized technical, algorithmic, and behavioral interview questions. |
| `/api/session/{id}` | `GET` | `id` (path parameter) | Retrieves active session state, candidate profile, and indexed JDs. |
| `/api/evaluation/comparison` | `GET` | *None* | Fetches stored multi-model quantitative evaluation comparison matrix. |
| `/api/evaluation/questions` | `GET` | *None* | Fetches the 25 benchmark evaluation questions. |
| `/api/evaluation/rag-analysis` | `GET` | *None* | Fetches RAG pipeline trace analysis across test Cases A through E. |
| `/api/evaluation/repository` | `GET` | *None* | Fetches codebase repository RAG benchmark scores and questions. |
| `/api/evaluation/run` | `POST` | *None* | Triggers a fresh run of the multi-model evaluation benchmark harness. |

---

## 🚀 6. Quickstart & Installation Guide

### Prerequisites
- Python 3.11+ (Python 3.13 tested and supported)
- [Ollama](https://ollama.ai) installed and running locally (`ollama serve`)

### Step 1: Pull Required Ollama Models
```bash
# Embeddings Model
ollama pull nomic-embed-text

# Evaluated LLM Models
ollama pull codellama:7b
ollama pull starcoder2:3b
ollama pull qwen2.5:0.5b
```

### Step 2: Environment Setup
```bash
# Navigate to project directory
cd /Users/sourabh/Downloads/idtt

# Create and activate virtual environment
python3.13 -m venv .venv13
source .venv13/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure Environment (`.env`)
Create or verify `.env` file in the project root:
```env
OLLAMA_URL=http://localhost:11434
LLM_MODEL=codellama:7b
EMBED_MODEL=nomic-embed-text
TEMPERATURE=0.2

CHROMA_DIR=storage/chroma
COLLECTION=placement_knowledge
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5
MAX_DISTANCE=0.65
```

### Step 4: Ingest BMU Knowledge Base
```bash
.venv13/bin/python scripts/ingest.py --reset
```

### Step 5: Start FastAPI Server
```bash
.venv13/bin/python -m uvicorn app.main:app --port 8000 --reload
```
Open your browser and navigate to **http://localhost:8000** to access the Web Dashboard.

---

## 📊 7. Multi-Model Comparative Evaluation & Benchmarks

### Evaluation Methodology
Three local LLM models were benchmarked under **100% identical experimental control conditions**:
- **Same Vector Index**: ChromaDB collection with 27 chunks
- **Same Embedding Model**: `nomic-embed-text`
- **Same Chunking Configuration**: Section-aware (800 char chunk size, 120 char overlap)
- **Same Retrieval Parameters**: `TOP_K = 5`, `MAX_DISTANCE = 0.65`, `Temperature = 0.2`
- **Same Dataset**: Standardized 25 questions across 10 institutional placement categories

### Measured Quantitative Results Matrix

| Benchmark Metric | `codellama:7b` (Recommended) | `starcoder2:3b` | `qwen2.5:0.5b` |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | **52.0%** (13/25) | **48.0%** (12/25) | **48.0%** (12/25) |
| **Relevance Score (1.0–5.0)** | **3.80 / 5.0** | **3.56 / 5.0** | **3.40 / 5.0** |
| **Retrieval Hit Rate (%)** | **80.0%** | **80.0%** | **80.0%** |
| **Observed Hallucination Rate (%)** | **8.0%** | **8.0%** | **8.0%** |
| **Test-Pass Rate for Code Gen** | `N/A` (App generates Q&A) | `N/A` | `N/A` |
| **Average Total Latency (s)** | **3.36s** | **4.02s** | **13.09s** |
| **Average Retrieval Latency (s)** | **0.002s** | **0.002s** | **0.002s** |
| **Average Generation Latency (s)** | **3.35s** | **4.01s** | **13.08s** |
| **Average Prompt Tokens** | **653** | **653** | **653** |
| **Average Completion Tokens** | **61.8** | **53.5** | **87.0** |
| **Average CPU Utilization (%)** | **35.0%** | **32.0%** | **18.0%** |
| **Average RAM Usage (MB)** | **14,500 MB** | **14,200 MB** | **13,800 MB** |

### Model Recommendation & Analysis
- **CodeLlama 7B (`codellama:7b`)** is the recommended model for production. It demonstrates superior instruction-following, the highest accuracy (**52.0%**), highest relevance (**3.80/5.0**), and the lowest generation latency (**3.35s**) on Apple Silicon ARM64 hardware.
- **Qwen 0.5B (`qwen2.5:0.5b`)** exhibited token throttling and repetitive loops on certain negative constraint prompts, leading to higher overall latency (**13.09s**).
- **StarCoder2 3B (`starcoder2:3b`)** offered balanced performance (**48.0%** accuracy, **4.02s** latency) but produced more terse answers compared to CodeLlama.

### Reproduction Commands
```bash
# Run multi-model evaluation harness across all 3 models
PYTHONUNBUFFERED=1 .venv13/bin/python evaluation/eval.py --all

# Run codebase repository RAG benchmark
PYTHONUNBUFFERED=1 .venv13/bin/python app/services/repo_rag.py
```

---

## 🧪 8. Automated Test Suite

The project includes an automated **pytest** test suite with 15 test cases covering all critical paths:

```bash
.venv13/bin/python -m pytest
```

### Test Coverage Breakdown:
- [`tests/test_chunking.py`](file:///Users/sourabh/Downloads/idtt/tests/test_chunking.py): Section-aware chunking boundary preservation and heading retention.
- [`tests/test_eligibility.py`](file:///Users/sourabh/Downloads/idtt/tests/test_eligibility.py): Deterministic eligibility calculation (CGPA thresholds, backlog rules, graduation years).
- [`tests/test_matching.py`](file:///Users/sourabh/Downloads/idtt/tests/test_matching.py): Structured resume vs. JD requirement comparison and scoring.
- [`tests/test_isolation.py`](file:///Users/sourabh/Downloads/idtt/tests/test_isolation.py): Metadata isolation verifying zero data leakage across company JDs and student resumes.
- [`tests/test_integration.py`](file:///Users/sourabh/Downloads/idtt/tests/test_integration.py): FastAPI REST endpoint lifecycle and response validation.
- [`tests/test_e2e.py`](file:///Users/sourabh/Downloads/idtt/tests/test_e2e.py): End-to-end placement workflow from document upload to interview preparation.
- [`tests/test_failures.py`](file:///Users/sourabh/Downloads/idtt/tests/test_failures.py): Negative testing for unanswerable queries and out-of-scope requests.

---

## 🐳 9. Docker Deployment

Deploy the entire stack with Docker Compose:

```bash
docker-compose up --build
```
- **App Service**: Accessible at `http://localhost:8000`
- **Ollama Service**: Accessible at `http://localhost:11434`

---

## ⚠️ 10. Core Grounding & Safety Guardrails

The platform adheres to 5 strict operational guardrails:
1. **Institutional Grounding**: Never invent rules, numbers, dates, or cutoffs not present in ingested BMU policy documents.
2. **Anti-Fabrication Guardrail**: Never invent resume skills, projects, or work history when advising candidates.
3. **Multi-Tenant Isolation**: Never allow Company A data to appear in Company B queries.
4. **Deterministic Preference**: Always execute mathematical and logical policy checks deterministically through code rather than relying on probabilistic LLM inferences.
5. **Explicit Insufficient Information Warning**: Always return a clear notice when required information is absent from the indexed knowledge base.
