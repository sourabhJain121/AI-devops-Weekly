# 🛠️ BMU Placement Intelligence Platform: Master Technical Architecture & Python Codebase Manual

> **Dedicated Technical Reference Manual**  
> Comprehensive specification of **every Python service, module, function, mathematical algorithm, regex heuristic, and test file** across the entire repository.

---

## 📑 Table of Contents

1. [Architecture & Engine Taxonomy](#1-architecture--engine-taxonomy)
2. [Core Framework & Ingestion Modules (`app/`)](#2-core-framework--ingestion-modules-app)
   - [`app/config.py`](#21-appconfigpy)
   - [`app/documents.py`](#22-appdocumentspy)
   - [`app/chunking.py`](#23-appchunkingpy)
   - [`app/ingest.py`](#24-appingestpy)
   - [`app/vectorstore.py`](#25-appvectorstorepy)
   - [`app/llm.py`](#26-appllmpy)
   - [`app/rag.py`](#27-appragpy)
   - [`app/orchestrator.py`](#28-apporchestratorpy)
   - [`app/web_search.py`](#29-appweb_searchpy)
   - [`app/main.py`](#210-appmainpy)
3. [Specialized Placement Services (`app/services/`)](#3-specialized-placement-services-appservices)
   - [`app/services/resume.py`](#31-appservicesresumepy)
   - [`app/services/jd.py`](#32-appservicesjdpy)
   - [`app/services/eligibility.py`](#33-appserviceseligibilitypy)
   - [`app/services/matching.py`](#34-appservicesmatchingpy)
   - [`app/services/skill_gap.py`](#35-appservicesskill_gappy)
   - [`app/services/improvement.py`](#36-appservicesimprovementpy)
   - [`app/services/interview.py`](#37-appservicesinterviewpy)
   - [`app/services/question_bank.py`](#38-appservicesquestion_bankpy)
   - [`app/services/company_intelligence.py`](#39-appservicescompany_intelligencepy)
   - [`app/services/repo_rag.py`](#310-appservicesrepo_ragpy)
4. [Scientific Evaluation Engine (`evaluation/`)](#4-scientific-evaluation-engine-evaluation)
   - [`evaluation/eval.py`](#41-evaluationevalpy)
5. [CLI & Operational Scripts (`scripts/`)](#5-cli--operational-scripts-scripts)
   - [`scripts/ingest.py`](#51-scriptsingestpy)
   - [`scripts/ask.py`](#52-scriptsaskpy)
6. [Automated Test Suite Specification (`tests/`)](#6-automated-test-suite-specification-tests)
   - [`tests/test_chunking.py`](#61-teststest_chunkingpy)
   - [`tests/test_eligibility.py`](#62-teststest_eligibilitypy)
   - [`tests/test_matching.py`](#63-teststest_matchingpy)
   - [`tests/test_isolation.py`](#64-teststest_isolationpy)
   - [`tests/test_orchestration.py`](#65-teststest_orchestrationpy)
   - [`tests/test_rag_comparison.py`](#66-teststest_rag_comparisonpy)
   - [`tests/test_company_intelligence.py`](#67-teststest_company_intelligencepy)
   - [`tests/test_question_bank.py`](#68-teststest_question_bankpy)
   - [`tests/test_integration.py`](#69-teststest_integrationpy)
   - [`tests/test_failures.py`](#610-teststest_failurespy)
   - [`tests/test_e2e.py`](#611-teststest_e2epy)
7. [Complete Module Dependency & Cross-Reference Matrix](#7-complete-module-dependency--cross-reference-matrix)

---

## 1. Architecture & Engine Taxonomy

The platform enforces a strict boundary between **Deterministic Rule Engines** (algorithms, set theory, mathematical cutoffs) and **Probabilistic Generation Pipelines** (vector similarity, sovereign local LLMs).

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI WEB INGRESS                                    │
│                                    (app/main.py)                                       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
               ┌─────────────────────────────────────────────────────────┐
               │              DYNAMIC INTENT ORCHESTRATOR                │
               │                  (app/orchestrator.py)                  │
               └───────────┬─────────────────────────────────┬───────────┘
                           │                                 │
           ┌───────────────┴───────────────┐                 │
           ▼                               ▼                 ▼
┌───────────────────────┐       ┌───────────────────────┐ ┌───────────────────────┐
│ DETERMINISTIC SERVICES│       │  VECTOR RETRIEVAL     │ │  WEB INTELLIGENCE     │
├───────────────────────┤       ├───────────────────────┤ ├───────────────────────┤
│ • eligibility.py      │       │ • rag.py              │ │ • company_            │
│ • matching.py         │       │ • vectorstore.py      │ │   intelligence.py     │
│ • skill_gap.py        │       │ • question_bank.py    │ │ • web_search.py       │
│ • resume.py (Regex)   │       │ • repo_rag.py         │ │ (4 Verification Gates)│
└──────────┬────────────┘       └──────────┬────────────┘ └──────────┬────────────┘
           │                               │                         │
           └───────────────────────┬───────┴─────────────────────────┘
                                   ▼
                   GROUNDED CONTEXT ENVELOPE + CITATIONS
                                   │
                                   ▼
                 LOCAL OLLAMA SOVEREIGN INFERENCE ENGINE
                               (app/llm.py)
                                   │
                                   ▼
                   STRUCTURED DECISION TRACE RESPONSE
```

---

## 2. Core Framework & Ingestion Modules (`app/`)

---

### 2.1 `app/config.py`
- **Role in Architecture**: Central environment configuration loader.
- **Responsibility**: Loads runtime variables from `.env` using `python-dotenv`, parses types, sets defaults, and exports global parameters across all services.
- **Key Variables & Constants**:
  - `OLLAMA_URL`: Default `http://localhost:11434`. Base endpoint for local Ollama HTTP requests.
  - `LLM_MODEL`: Default `codellama:7b`. Primary open-weights LLM checkpoint.
  - `EMBED_MODEL`: Default `nomic-embed-text`. 768-dimensional dense embedding model.
  - `CHROMA_DIR`: Path resolving to `storage/chroma`. On-disk vector database directory.
  - `COLLECTION`: `placement_knowledge`. Primary ChromaDB vector collection.
  - `CHUNK_SIZE`: Default `800` characters. Target size for semantic text chunks.
  - `CHUNK_OVERLAP`: Default `120` characters. Sliding window overlap between chunks.
  - `TOP_K`: Default `5`. Maximum chunks retrieved per vector query.
  - `MAX_DISTANCE`: Default `0.65`. Cosine distance cutoff threshold ($d = 1 - \cos(\theta)$). Chunks with $d > 0.65$ are discarded as irrelevant.
  - `BMU_DIR`: Path resolving to `knowledge/`. Root storage for official policy documents.
  - `RESUMES_DIR`: Path resolving to `data/sample_resumes/`. Benchmark candidate resumes.
- **Underlying Logic**: Ensures fail-safe initialization. If `.env` is absent, the system falls back to tested production defaults without raising unhandled exceptions.

---

### 2.2 `app/documents.py`
- **Role in Architecture**: Document extractor, normalizer, and file discovery service.
- **Responsibility**: Ingests `.pdf`, `.md`, and `.txt` files from disk, extracts page-delimited text, normalizes character encoding, and discovers files while strictly excluding non-content directories.
- **Key Functions**:
  - `load(path: Path) -> list[tuple[int, str]]`: Universal document loader dispatcher. Detects file extension:
    - `.pdf`: Dispatches to `load_pdf()`.
    - `.md`: Dispatches to `load_markdown()`.
    - `.txt`: Dispatches to `load_text()`.
    - Returns a list of `(page_number, page_text)` tuples.
  - `load_pdf(path: Path) -> list[tuple[int, str]]`:
    - *Logic*: Uses `pypdf.PdfReader`. Iterates across `reader.pages`, invoking `page.extract_text()`.
    - Sanitizes whitespace via `_clean_text()`.
    - Page numbering starts at `1` (1-indexed) to maintain honest student-facing citations.
  - `load_markdown(path: Path)` & `load_text(path: Path)`:
    - *Logic*: Reads text with `utf-8` encoding (with `replace` error fallback). Treats entire file as page `1`.
  - `discover(directory: Path) -> list[Path]`:
    - *Logic*: Recursively traverses `directory` using `rglob("*")`.
    - Enforces strict directory exclusion:
      ```python
      EXCLUDED_DIRS = {".venv", ".venv13", "__pycache__", ".pytest_cache", ".git", "storage"}
      ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
      ```
    - Prevents indexing virtual environments, Git databases, or SQLite binaries into ChromaDB.
  - `_clean_text(text: str) -> str`: Normalizes multiple blank lines (`\n{3,}`) into `\n\n`, strips carriage returns (`\r`), and trims leading/trailing whitespace.

---

### 2.3 `app/chunking.py`
- **Role in Architecture**: Section-aware, heading-bounded semantic text chunker.
- **Responsibility**: Partitions long documents into vector-indexable text snippets while strictly respecting section boundaries.
- **Key Functions**:
  - `chunk(text: str, size: int, overlap: int) -> list[dict]`:
    - *Output Schema*: Returns `list[{"text": str, "section": str}]`.
    - *Logic*:
      1. Splits input text into discrete paragraphs using `re.split(r"\n{2,}", text)`.
      2. Evaluates each paragraph with `_is_heading()`.
      3. **Heading Boundary Preservation**: If a heading is detected, the current chunk buffer is immediately flushed to `chunks`. The chunk **never** crosses a section boundary.
      4. The detected heading is stored as the chunk's `section` metadata and prepended as the first line of the new chunk buffer.
      5. **Paragraph Packing**: Subsequent paragraphs are packed into `buffer` as long as `len(buffer) + len(paragraph) + 2 <= size`.
      6. **Hard Splitting for Oversized Paragraphs**: If a single paragraph exceeds `size=800`, it finds the last whitespace before `size`, splits the head into a chunk, flushes, and carries over the tail with `overlap=120` characters to preserve semantic continuity.
  - `_is_heading(paragraph: str) -> bool`:
    - *Logic*: Evaluates the compiled regular expression:
      ```python
      HEADING = re.compile(
          r"^\s*(?:#{1,6}\s+.+"              # Markdown headings: ## Section Name
          r"|\d+(?:\.\d+)*[.)]\s+\S.{0,80}"  # Numbered headings: 1. Eligibility or 2.3)
          r"|[A-Z][A-Z \-/&]{4,60})\s*$"     # All-caps headings: PLACEMENT REGULATIONS
      )
      ```
  - `_heading_text(paragraph: str) -> str`: Strips leading `#` symbols and returns cleaned section titles truncated to 120 characters.

---

### 2.4 `app/ingest.py`
- **Role in Architecture**: Ingestion pipeline coordinator.
- **Responsibility**: Orchestrates the pipeline: `File -> Extraction -> Chunking -> Dense Embedding -> ChromaDB Storage`.
- **Key Functions**:
  - `_document_id(path: Path) -> str`: Computes a deterministic 12-character SHA-1 hash of the resolved file path (`hashlib.sha1(path).hexdigest()[:12]`).
  - `ingest_file(path: Path, source_type: str, extra: dict) -> dict`:
    - *Logic*:
      1. Calls `documents.load(path)`.
      2. Calls `vectorstore.delete_document(document_name)`: **Re-ingesting a document replaces old chunks rather than creating duplicate embeddings**.
      3. Iterates over pages and chunks, constructing unique composite chunk IDs: `f"{document_id}:p{page_number}:c{chunk_index}"`.
      4. Builds rich metadata dictionaries: `document_id`, `document_name`, `source_type`, `page`, `section`, `chunk_index`.
      5. Batch embeds chunks in groups of `EMBED_BATCH = 32` via `llm.embed(texts[batch])`.
      6. Stores chunks in ChromaDB via `vectorstore.add()`.
  - `ingest_directory(directory: Path, source_type: str) -> list[dict]`: Discovers all files in directory and calls `ingest_file()` for each.
  - `ingest_knowledge_base() -> dict`: Ingests all policy files in `knowledge/`.

---

### 2.5 `app/vectorstore.py`
- **Role in Architecture**: Persistent ChromaDB vector database manager.
- **Responsibility**: Provides an interface for cosine similarity searches, idempotent chunk writes, document-level deletions, and metadata tenancy isolation.
- **Key Functions**:
  - `_get_collection()`: Initializes ChromaDB `PersistentClient(path=config.CHROMA_DIR)` with cosine distance settings:
    ```python
    metadata={"hnsw:space": "cosine"}
    ```
  - `add(ids, texts, metadatas, embeddings)`: Upserts chunks into the ChromaDB collection.
  - `query(embedding: list[float], top_k: int, where: dict | None) -> list[dict]`:
    - *Logic*:
      1. Executes collection query with dense vector and optional metadata filter (`where`).
      2. Calculates cosine similarity score from returned cosine distance:
         $$\text{Score} = 1.0 - \text{distance}$$
      3. Unpacks results into structured hit dictionaries: `chunk_id`, `text`, `score`, `distance`, `document_name`, `page`, `section`, `source_type`, `company_id`.
  - `delete_document(document_name: str)`: Executes `collection.delete(where={"document_name": document_name})`.
  - `stats() -> dict`: Returns total indexed chunks and breakdown by document name.

---

### 2.6 `app/llm.py`
- **Role in Architecture**: Local Ollama HTTP inference client.
- **Responsibility**: Dispatches chat completions and embedding requests to the local Ollama daemon, handles connection failures, tracks token usage, and measures execution latency.
- **Key Functions**:
  - `is_available() -> bool`: Sends `GET http://localhost:11434/api/tags`. Returns `True` if Ollama is online.
  - `available_models() -> list[str]`: Queries Ollama and returns list of installed model tags (e.g. `["codellama:7b", "nomic-embed-text"]`).
  - `embed(texts: list[str]) -> list[list[float]]`:
    - Sends `POST /api/embed` with `{"model": config.EMBED_MODEL, "input": texts}`.
    - Returns 768-dimensional dense vectors.
  - `embed_one(text: str) -> list[float]`: Generates a single vector embedding.
  - `generate(prompt: str, user_message: str, model: str, temperature: float) -> dict`:
    - Sends `POST /api/chat` with:
      ```json
      {
        "model": "codellama:7b",
        "messages": [
          {"role": "system", "content": prompt},
          {"role": "user", "content": user_message}
        ],
        "options": {"temperature": 0.2},
        "stream": false
      }
      ```
    - Measures elapsed latency using `time.perf_counter()`.
    - Extracts `prompt_eval_count` and `eval_count` for token telemetry.

---

### 2.7 `app/rag.py`
- **Role in Architecture**: Grounded Retrieval-Augmented Generation pipeline.
- **Responsibility**: Coordinates vector similarity retrieval, discriminative query term filtering, grounded prompt construction, inline citation insertion, and unanswerable query abstention.
- **Key Functions & Logic**:
  - `retrieve(question: str, top_k: int, source_types: list[str]) -> list[dict]`:
    - Generates query vector via `llm.embed_one(question)`.
    - Queries ChromaDB with tenancy filters (`where={"source_type": {"$in": source_types}}`).
    - Discards hits exceeding cutoff: `[h for h in hits if h["distance"] <= config.MAX_DISTANCE]`.
    - Applies `filter_relevant_hits()`.
  - `filter_relevant_hits(hits: list[dict], question: str) -> list[dict]`:
    - *Logic*: Computes set of non-generic query tokens:
      $$\text{Discriminative} = \text{QueryWords} \setminus \text{GENERIC\_SEARCH\_TERMS}$$
    - Checks overlap between discriminative tokens and chunk text/section words.
    - If the top chunk has high similarity ($\ge 0.75$), prunes secondary chunks whose scores trail by $> 0.08$ and lack keyword overlap. Prevents dilution from tangential context.
  - `build_context(hits: list[dict]) -> str`: Formats retrieved chunks with bracketed citation markers:
    ```text
    [1] BMU_Placement_Policy.pdf > 1. Eligibility for Placement (page 1)
    <chunk text>
    ```
  - `answer(question: str, model: str) -> dict`:
    - Evaluates retrieved hits. If `hits` is empty, returns safe fallback without invoking LLM:
      > *"I could not find sufficient information in the provided documents to answer this reliably. Please add the relevant document to the knowledge base and re-run ingestion."*
    - Otherwise formats `SYSTEM_PROMPT` bounding the model to the context, invokes `llm.generate()`, and returns reply with attached source list.

---

### 2.8 `app/orchestrator.py`
- **Role in Architecture**: Autonomous AI Orchestrator & Dynamic Intent Router.
- **Responsibility**: Analyzes user messages, determines execution intent, dispatches requests to deterministic services or vector pipelines, and packages responses with structured decision traces.
- **Key Functions**:
  - `detect_intent(message: str) -> str`:
    - Evaluates message against regular expressions and keyword taxonomies.
    - Returns one of 10 intent codes: `REPO_RAG`, `COMPANY_INTELLIGENCE`, `RESUME_IMPROVEMENT`, `ELIGIBILITY`, `RESUME_MATCH`, `SKILL_GAP`, `INTERVIEW_PREPARATION`, `QUESTION_BANK_DIRECT`, `COMPANY_JD`, `BMU_POLICY`.
  - `route_and_execute(message, candidate_id, company_id, model) -> dict`:
    - Binds session entities and starts stopwatch.
    - Routes `ELIGIBILITY` -> `eligibility.evaluate_eligibility()`.
    - Routes `RESUME_MATCH` -> `matching.compare_resume_to_jd()`.
    - Routes `SKILL_GAP` -> `skill_gap.analyze_skill_gaps()`.
    - Routes `RESUME_IMPROVEMENT` -> `improvement.suggest_resume_improvements()`.
    - Routes `INTERVIEW_PREPARATION` -> `interview.generate_interview_prep()`.
    - Routes `COMPANY_INTELLIGENCE` -> `company_intelligence.research_company()`.
    - Routes `REPO_RAG` -> `repo_rag.answer_codebase_question()`.
    - Routes `BMU_POLICY` -> `rag.answer()`.
    - Returns JSON payload containing `reply`, `intent`, `grounded`, `model`, `latency_s`, `sources`, and `analysis_data`.

---

### 2.9 `app/web_search.py`
- **Role in Architecture**: DuckDuckGo web search provider abstraction.
- **Responsibility**: Executes real-time web searches, enforces rate limits, rotates user agents, cleans raw HTML snippets, and tags domain reliability tiers.
- **Key Functions**:
  - `search(query: str, max_results: int = 5) -> list[dict]`:
    - Uses `duckduckgo_search.DDGS().text(query, max_results=max_results)`.
    - Normalizes outputs into: `title`, `href`, `body`, `domain`, `reliability_tier`.
  - `classify_domain_tier(domain: str) -> str`:
    - *Tier 1 (Authoritative)*: Official corporate domains, `*.edu`, `*.gov`, verified investor/financial portals.
    - *Tier 2 (Established Media)*: Forbes, TechCrunch, Bloomberg, Reuters, Economic Times.
    - *Tier 3 (Community/Forums)*: Reddit, Quora, Medium, personal blogs.

---

### 2.10 `app/main.py`
- **Role in Architecture**: FastAPI application entrypoint and REST API server.
- **Responsibility**: Defines HTTP routing, request/response models, static file hosting, upload processing, session storage, and evaluation endpoints.
- **Key Endpoints & Functions**:
  - `index()`: Serves `app/static/index.html` at `GET /`.
  - `health()`: Returns Ollama status, vector collection counts, and installed models at `GET /api/health`.
  - `query_orchestrator(req)`: Main chat entrypoint at `POST /api/query`.
  - `upload_resume(file)`: Receives multipart PDF/TXT, calls `resume.parse_resume()`, stores candidate profile in memory at `POST /api/upload/resume`.
  - `upload_company_jd(file)`: Receives multipart JD, calls `jd.parse_jd()`, indexes chunks with `company_id` at `POST /api/upload/company-jd`.
  - `analyze_eligibility()`, `analyze_resume_jd()`, `analyze_skill_gap()`: Direct analytical API endpoints.
  - `compare_rag_query(req)`: Live side-by-side RAG Comparison endpoint at `POST /api/evaluation/rag-comparison`. Executes both Without-RAG and With-RAG under identical parameters and returns side-by-side payloads.
  - `run_eval()`: Triggers full evaluation suite at `POST /api/evaluation/run`.

---

## 3. Specialized Placement Services (`app/services/`)

---

### 3.1 `app/services/resume.py`
- **Role in Architecture**: Candidate resume parsing, entity extraction, and profile caching.
- **Responsibility**: Ingests resume files, extracts structured entities (CGPA, backlogs, degree, branch, graduation year, skills), and caches profiles in `_CANDIDATE_PROFILES`.
- **Functions & Logic**:
  - `extract_candidate_profile(text: str, document_name: str) -> dict`:
    - Initializes dictionary: `name`, `email`, `phone`, `cgpa`, `backlogs`, `degree`, `branch`, `graduation_year`, `skills`.
    - **Email Regex**: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
    - **Phone Regex**: `(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}`
    - **CGPA Regex**: Dual patterns:
      1. `(?:CGPA|GPA)\s*(?:[:=]|\bis\b)?\s*([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?`
      2. `([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?\s*(?:CGPA|GPA)`
      Casts match to `float`.
    - **Backlog Regex**: Checks `\b(?:no|zero|0)\s+active\s+backlogs?\b` -> sets `0`. Otherwise matches `(\d+)\s+active\s+backlogs?` -> casts to `int`.
    - **Degree Regex**: Evaluates `B.Tech`, `M.Tech`, `BBA`, `MBA`.
    - **Branch Regex**: Evaluates `Computer Science`, `CSE`, `Mechanical`, `ECE`.
    - **Graduation Year**: Matches `\b(202[3-9])\b`.
    - **Skills Matching**: Matches text against 32 technical keywords using `\b{skill}\b` word boundaries.
  - `parse_resume(file_path: Path, candidate_id: str) -> dict`: Loads PDF text via `documents.load_pdf()`, calls `extract_candidate_profile()`, indexes text chunks into ChromaDB with `source_type="resume"`, and updates `_CANDIDATE_PROFILES[candidate_id]`.
  - `get_candidate_profile(candidate_id: str) -> dict | None`: Retrieves cached candidate profile.

---

### 3.2 `app/services/jd.py`
- **Role in Architecture**: Company Job Description parsing and metadata-isolated indexing.
- **Responsibility**: Extracts company hiring specifications and indexes vector chunks with isolated `company_id`.
- **Functions & Logic**:
  - `parse_jd(file_path: Path, company_id: str) -> dict`:
    - Extracts raw text using `documents.load()`.
    - Role Title: Extracted from heading lines or prefix patterns (`Role:`, `Job Title:`).
    - Required CGPA Cutoff: Evaluates `r"(?:minimum|min|cutoff|cut-off)?\s*cgpa\s*(?:of|is|:)?\s*([0-9]\.[0-9]{1,2})"`.
    - Max Allowed Backlogs: Evaluates `r"max(?:imum)?\s*(?:allowed)?\s*backlogs?\s*(?:of|is|:)?\s*(\d+)"`.
    - Technical Skills: Matches against technical taxonomy, partitioning into required vs preferred skills.
    - Caches structured JD in `_COMPANY_JDS[company_id]`.
    - Calls `ingest_file(file_path, source_type="company_jd", extra={"company_id": company_id})` to guarantee tenancy isolation.
  - `get_company_jd(company_id: str) -> dict | None`: Retrieves cached JD.

---

### 3.3 `app/services/eligibility.py`
- **Role in Architecture**: Deterministic placement eligibility verification engine.
- **Responsibility**: Mathematically compares candidate academic credentials against BMU policy rules and company JD cutoffs with zero hallucination.
- **Functions & Logic**:
  - `evaluate_eligibility(candidate_id: str, company_id: str | None) -> dict`:
    1. Fetches candidate profile (`cand`) and company JD (`comp`).
    2. Queries ChromaDB for BMU policy context: `vectorstore.query("minimum CGPA backlog eligibility rules", where={"source_type": "bmu_policy"})`.
    3. Mathematical Assertions:
       - BMU Minimum CGPA: $C_{\text{BMU}} = 6.5$. If $\text{CGPA} < 6.5$, appends failure reason.
       - BMU Maximum Backlogs: $B_{\text{BMU}} = 2$. If $\text{Backlogs} > 2$, appends failure reason.
       - Company JD CGPA: If $C_{\text{cand}} < C_{\text{JD}}$, appends company-specific failure reason.
       - Company JD Backlogs: If $B_{\text{cand}} > B_{\text{JD}}$, appends company-specific failure reason.
    4. Deterministic Verdict: Sets `status = "Eligible"` if all criteria are satisfied; otherwise `status = "Not Eligible"`.
    5. Prompt Synthesis: Calls `llm.generate()` strictly to summarize the verdict into human prose, passing the precomputed `STATUS DETERMINED BY SYSTEM: {status}` and forbidding the LLM from altering the decision.

---

### 3.4 `app/services/matching.py`
- **Role in Architecture**: Structured Resume vs. JD requirement matching engine.
- **Responsibility**: Computes candidate-to-role compatibility using set theory and keyword tokenization.
- **Functions & Logic**:
  - `compare_resume_to_jd(candidate_id: str, company_id: str | None) -> dict`:
    1. Normalizes candidate skills to lowercase set: $S_{\text{Resume}} = \{s.\text{lower}() \mid s \in \text{cand}["\text{skills}"]\}$.
    2. Normalizes JD required skills: $S_{\text{JD}} = \{s.\text{lower}() \mid s \in \text{comp}["\text{required\_skills}"]\}$.
    3. Computes matching skills: $\text{Matched} = S_{\text{Resume}} \cap S_{\text{JD}}$.
    4. Computes missing skills: $\text{Missing} = S_{\text{JD}} \setminus S_{\text{Resume}}$.
    5. Computes Match Score Percentage:
       $$\text{Match Score} = \text{round}\left( \frac{|\text{Matched}|}{\max(|S_{\text{JD}}|, 1)} \times 100, 1 \right)$$
    6. Queries ChromaDB for additional responsibilities context: `vectorstore.query("skills requirements responsibilities", where={"company_id": comp["company_id"]})`.
    7. Returns score, matched list, missing list, formatted summary, and cited sources.

---

### 3.5 `app/services/skill_gap.py`
- **Role in Architecture**: Prioritized skill gap analysis service.
- **Responsibility**: Identifies missing competencies between candidate resume and target JD, categorizing them into actionable preparation tiers.
- **Functions & Logic**:
  - `analyze_skill_gaps(candidate_id: str, company_id: str | None) -> dict`:
    1. Iterates over $S_{\text{JD}}$.
    2. If skill is present in $S_{\text{Resume}}$: assigns `gap_status: "No Gap"`, `priority: "Low"`.
    3. If skill is missing: assigns `gap_status: "Missing Skill"`, `priority: "High"`.
    4. Formats an explainable Markdown table:
       ```markdown
       | Skill | Evidence in Resume | Status | Priority |
       | **Python** | Present in resume skills list | No Gap | **Low** |
       | **Kubernetes** | Not found in resume | Missing Skill | **High** |
       ```
    5. Returns gaps array, high-priority list, and summary.

---

### 3.6 `app/services/improvement.py`
- **Role in Architecture**: Evidence-grounded resume improvement advisor.
- **Responsibility**: Generates targeted bullet-point resume enhancements tailored to the target JD while adhering to a strict anti-fabrication guardrail.
- **Functions & Logic**:
  - `suggest_resume_improvements(candidate_id: str, company_id: str | None) -> dict`:
    - Extracts candidate skills, parsed projects, and missing JD skills.
    - System Prompt Enforcement:
      > *"Rules: 1. DO NOT fabricate or invent new projects or employment history. 2. Focus on how to highlight existing skills and bridge missing skills through coursework or project extensions. 3. Provide 3-5 concrete, actionable bullet points."*
    - Generates quantified accomplishment suggestions (e.g. converting *"Worked on database"* into *"Optimized PostgreSQL indexing, reducing query latency by 35%"*) without inventing unearned qualifications.

---

### 3.7 `app/services/interview.py`
- **Role in Architecture**: Role-tailored interview preparation engine.
- **Responsibility**: Generates personalized technical, algorithmic, and behavioral interview questions tailored to the candidate's actual projects and the company's JD requirements.
- **Functions & Logic**:
  - `generate_interview_prep(candidate_id: str, company_id: str | None) -> dict`:
    1. Aggregates candidate profile, verified company intelligence, and target JD.
    2. Queries ChromaDB for company role context (`where={"company_id": comp["company_id"]}`).
    3. Prompts LLM across 3 structured domains:
       - **Technical Questions**: Tailored to required JD skills candidate possesses + missing skills they must justify.
       - **Project Deep-Dive Questions**: Probes the candidate's actual named resume projects (architectural tradeoffs, failure handling).
       - **Behavioral Questions**: Company culture questions aligned with corporate values.

---

### 3.8 `app/services/question_bank.py`
- **Role in Architecture**: Curated 120-question technical interview question bank engine.
- **Responsibility**: Validates JSON schema, indexes questions idempotently into ChromaDB with `company_id: null`, and supports hybrid keyword/semantic search.
- **Functions & Logic**:
  - `ingest_question_bank() -> dict`:
    - Reads `knowledge/interview_question_bank/interview_question_bank.json`.
    - Validates schema: `id`, `category`, `difficulty`, `question`, `expected_answer`, `tags`.
    - Enforces metadata tagging: `{"source_type": "interview_question_bank", "company_id": "null"}` to prevent false attribution to any specific hiring company.
    - Embeds and stores questions into ChromaDB.
  - `retrieve_questions(query: str, category: str, difficulty: str, top_k: int) -> list[dict]`:
    - Queries ChromaDB with composite metadata filters: `{"$and": [{"source_type": "interview_question_bank"}, {"category": category}, {"difficulty": difficulty}]}`.
    - Returns structured question cards with sample answers and expected topics.

---

### 3.9 `app/services/company_intelligence.py`
- **Role in Architecture**: Live enterprise web research and verification engine.
- **Responsibility**: Conducts live DuckDuckGo web research on employers, verifies corporate entity existence, classifies source domains into reliability tiers, and suppresses low-quality community speculation.
- **Functions & Logic**:
  - `research_company(user_query: str, company_id: str | None) -> dict`:
    1. Extracts target company name using regex heuristics.
    2. Calls `web_search.search(f"{company_name} company overview tech stack business domain")`.
    3. Evaluates the **4 Verification Gates**:
       - `● VERIFIED`: Found authoritative match on official corporate portal (Tier 1).
       - `⚠ AMBIGUOUS`: Detected multiple distinct corporate entities with the same name.
       - `⚠ UNVERIFIED`: Information available only through community forums (Tier 3).
       - `○ LIVE UNAVAILABLE`: Search provider timed out; falls back to cached profile.
    4. Assembles context, weighting Tier 1 sources heavily, and generates grounded company brief.

---

### 3.10 `app/services/repo_rag.py`
- **Role in Architecture**: Source Graph / Codebase Understanding RAG engine.
- **Responsibility**: Encapsulates the application's architectural blueprint and enables the platform to answer self-referential questions about its own code.
- **Functions & Logic**:
  - `REPO_ARCHITECTURE_KNOWLEDGE`: Comprehensive multi-line knowledge string documenting all 10 modules, file paths, and component responsibilities.
  - `answer_codebase_question(question: str, model: str) -> dict`:
    - Binds LLM to `REPO_ARCHITECTURE_KNOWLEDGE`.
    - Generates concise technical response citing exact filenames and functions.
  - `evaluate_repository_questions() -> dict`:
    - Reads `evaluation/repository_questions.json`.
    - Runs automated benchmark across 10 architectural questions, scoring correctness (achieving 100% accuracy).

---

## 4. Scientific Evaluation Engine (`evaluation/`)

---

### 4.1 `evaluation/eval.py`
- **Role in Architecture**: Multi-model quantitative evaluation harness.
- **Responsibility**: Evaluates 30 standardized benchmark questions across 3 local models in both WITH-RAG and WITHOUT-RAG modes (6 configurations).
- **Functions & Mathematical Formulations**:
  - `run_evaluation(models_to_eval: list[str]) -> dict`:
    - Iterates over `["codellama:7b", "starcoder2:3b", "qwen2.5:0.5b"]`.
    - Evaluates all 30 questions in `evaluation/questions.json`.
    - Measures separate retrieval latency ($t_{\text{ret}}$) and generation latency ($t_{\text{gen}}$) using `time.perf_counter()`.
    - Computes:
      1. **Answer Correctness (%)**:
         $$\text{Correctness} = \left( \frac{\sum \mathbb{I}(\text{is\_correct})}{N} \right) \times 100$$
         Checks keyword presence or valid abstention on unanswerable queries.
      2. **Groundedness (%)**:
         $$\text{Groundedness} = \left( \frac{\sum \mathbb{I}(\text{len}(\text{sources}) > 0 \land \text{is\_correct})}{N_{\text{answerable}}} \right) \times 100$$
      3. **Hallucination Rate (%)**:
         $$\text{Hallucination Rate} = \left( \frac{\sum \mathbb{I}(\text{expected}=\text{"INSUFFICIENT"} \land \neg \text{abstained})}{N_{\text{unanswerable}}} \right) \times 100$$
      4. **Relevance Score (1.0 to 5.0)**: Evaluates semantic overlap between question and response via `compute_relevance_score()`.
    - Persists summary matrix to `evaluation/results/comparison.json`.

---

## 5. CLI & Operational Scripts (`scripts/`)

---

### 5.1 `scripts/ingest.py`
- **Role in Architecture**: Command-line interface for knowledge base ingestion.
- **Responsibility**: Parses CLI flags (`--reset`), scans `knowledge/`, chunks documents, generates dense embeddings, and populates ChromaDB.
- **Usage**:
  ```bash
  # Incremental ingestion
  .venv13/bin/python scripts/ingest.py
  
  # Wipe collection and re-ingest from scratch
  .venv13/bin/python scripts/ingest.py --reset
  ```

---

### 5.2 `scripts/ask.py`
- **Role in Architecture**: Terminal-based RAG query tool.
- **Responsibility**: Accepts a question as a CLI argument, calls `app.rag.answer()`, and prints the grounded reply with inline citations directly to the console.
- **Usage**:
  ```bash
  .venv13/bin/python scripts/ask.py "What is the minimum CGPA required for campus placement?"
  ```

---

## 6. Automated Test Suite Specification (`tests/`)

Every test file is written with **pytest** and validates specific invariants:

### 6.1 `tests/test_chunking.py`
- `test_heading_boundary_preservation()`: Verifies that chunks never span across markdown headings (`## Section`).
- `test_heading_retention_in_metadata()`: Verifies that the heading title is correctly saved in the chunk's `section` attribute.
- `test_sliding_window_overlap()`: Asserts that paragraphs exceeding 800 characters repeat 120 characters across split points.

### 6.2 `tests/test_eligibility.py`
- `test_eligible_candidate()`: Verifies that a student with CGPA 8.0 and 0 backlogs receives `status: "Eligible"`.
- `test_ineligible_cgpa()`: Verifies that a student with CGPA 6.2 (< 6.5) receives `status: "Not Eligible"` with exact BMU policy clause cited.
- `test_ineligible_backlogs()`: Verifies that a student with 3 backlogs (> 2) is rejected.
- `test_company_specific_rejection()`: Verifies that a student meeting BMU rules (CGPA 6.8) is rejected if company JD requires CGPA 7.0.

### 6.3 `tests/test_matching.py`
- `test_perfect_match()`: Asserts 100% match score when candidate skills contain all JD skills.
- `test_partial_match()`: Asserts correct mathematical ratio when subset of skills matches.
- `test_missing_skills_extraction()`: Asserts that unmentioned skills appear in the `missing_skills` list.

### 6.4 `tests/test_isolation.py`
- `test_company_jd_isolation()`: Verifies that querying ChromaDB with `company_id="company_a"` returns 0 chunks from `company_b`.
- `test_policy_isolation()`: Verifies that policy queries do not leak student resume text.

### 6.5 `tests/test_orchestration.py`
- `test_intent_classification()`: Asserts that 10 distinct test prompts correctly map to their expected intent codes.
- `test_decision_trace_structure()`: Verifies that `route_and_execute()` emits structured decision trace dictionaries.

### 6.6 `tests/test_rag_comparison.py`
- `test_identical_generation_parameters()`: Asserts that Without-RAG and With-RAG use identical temperature (`0.2`) and model checkpoints.
- `test_context_isolation()`: Verifies that Without-RAG receives an empty context envelope (`context=""`).

### 6.7 `tests/test_company_intelligence.py`
- `test_verification_gates()`: Tests all 4 verification states (`VERIFIED`, `AMBIGUOUS`, `UNVERIFIED`, `LIVE_UNAVAILABLE`).
- `test_domain_reliability_tiering()`: Asserts that official corporate domains are classified as Tier 1.

### 6.8 `tests/test_question_bank.py`
- `test_question_bank_schema()`: Validates that all 120 questions possess valid IDs, categories, and answers.
- `test_company_id_null_guardrail()`: Asserts that every question bank item maintains `company_id: null` to prevent false attribution.

### 6.9 `tests/test_integration.py`
- `test_health_endpoint()`: Tests `GET /api/health`.
- `test_query_endpoint()`: Tests `POST /api/query`.
- `test_resume_upload_lifecycle()`: Uploads sample PDF and validates extracted profile in session.

### 6.10 `tests/test_failures.py`
- `test_unanswerable_policy_query()`: Verifies that the model explicitly abstains on questions not covered in the placement policy.
- `test_missing_profile_handling()`: Verifies graceful degradation when analyzing eligibility without an uploaded resume.

### 6.11 `tests/test_e2e.py`
- `test_full_e2e_placement_pipeline()`: Executes the entire workflow: Resume Registration $\to$ JD Registration $\to$ Deterministic Eligibility $\to$ Orchestrated Policy Chat.

---

## 7. Complete Module Dependency & Cross-Reference Matrix

| Module Path | Imported By | Imports From |
| :--- | :--- | :--- |
| [`app/config.py`](file:///Users/sourabh/Downloads/idtt/app/config.py) | All modules across `app/`, `scripts/`, `evaluation/` | `dotenv`, `os`, `pathlib` |
| [`app/documents.py`](file:///Users/sourabh/Downloads/idtt/app/documents.py) | `app/ingest.py`, `app/services/resume.py` | `pypdf`, `pathlib` |
| [`app/chunking.py`](file:///Users/sourabh/Downloads/idtt/app/chunking.py) | `app/ingest.py`, `app/services/jd.py` | `re` |
| [`app/ingest.py`](file:///Users/sourabh/Downloads/idtt/app/ingest.py) | `app/main.py`, `scripts/ingest.py` | `app/documents`, `app/chunking`, `app/vectorstore`, `app/llm` |
| [`app/vectorstore.py`](file:///Users/sourabh/Downloads/idtt/app/vectorstore.py) | `app/rag.py`, `app/ingest.py`, `app/services/*` | `chromadb`, `app/config` |
| [`app/llm.py`](file:///Users/sourabh/Downloads/idtt/app/llm.py) | `app/rag.py`, `app/ingest.py`, `app/services/*` | `requests`, `app/config` |
| [`app/rag.py`](file:///Users/sourabh/Downloads/idtt/app/rag.py) | `app/orchestrator.py`, `app/main.py`, `scripts/ask.py` | `app/vectorstore`, `app/llm`, `app/config` |
| [`app/orchestrator.py`](file:///Users/sourabh/Downloads/idtt/app/orchestrator.py) | `app/main.py`, `evaluation/eval.py` | `app/services/*`, `app/rag`, `app/vectorstore`, `app/llm` |
| [`app/web_search.py`](file:///Users/sourabh/Downloads/idtt/app/web_search.py) | `app/services/company_intelligence.py` | `duckduckgo_search`, `re` |
| [`app/main.py`](file:///Users/sourabh/Downloads/idtt/app/main.py) | Uvicorn Server | `fastapi`, `app/orchestrator`, `app/services/*`, `app/rag` |
| [`app/services/resume.py`](file:///Users/sourabh/Downloads/idtt/app/services/resume.py) | `app/orchestrator.py`, `app/main.py` | `app/documents`, `app/chunking`, `app/vectorstore` |
| [`app/services/jd.py`](file:///Users/sourabh/Downloads/idtt/app/services/jd.py) | `app/orchestrator.py`, `app/main.py` | `app/documents`, `app/chunking`, `app/vectorstore` |
| [`app/services/eligibility.py`](file:///Users/sourabh/Downloads/idtt/app/services/eligibility.py) | `app/orchestrator.py`, `app/main.py` | `app/services/resume`, `app/services/jd`, `app/vectorstore`, `app/llm` |
| [`app/services/matching.py`](file:///Users/sourabh/Downloads/idtt/app/services/matching.py) | `app/orchestrator.py`, `app/main.py` | `app/services/resume`, `app/services/jd`, `app/vectorstore` |
| [`app/services/skill_gap.py`](file:///Users/sourabh/Downloads/idtt/app/services/skill_gap.py) | `app/orchestrator.py`, `app/main.py` | `app/services/resume`, `app/services/jd` |
| [`app/services/improvement.py`](file:///Users/sourabh/Downloads/idtt/app/services/improvement.py) | `app/orchestrator.py`, `app/main.py` | `app/services/resume`, `app/services/jd`, `app/llm` |
| [`app/services/interview.py`](file:///Users/sourabh/Downloads/idtt/app/services/interview.py) | `app/orchestrator.py`, `app/main.py` | `app/services/resume`, `app/services/jd`, `app/vectorstore`, `app/llm` |
| [`app/services/question_bank.py`](file:///Users/sourabh/Downloads/idtt/app/services/question_bank.py) | `app/orchestrator.py`, `app/main.py` | `app/vectorstore`, `app/llm`, `json` |
| [`app/services/company_intelligence.py`](file:///Users/sourabh/Downloads/idtt/app/services/company_intelligence.py) | `app/orchestrator.py`, `app/main.py` | `app/web_search`, `app/llm` |
| [`app/services/repo_rag.py`](file:///Users/sourabh/Downloads/idtt/app/services/repo_rag.py) | `app/orchestrator.py`, `app/main.py` | `app/llm`, `app/config` |
| [`evaluation/eval.py`](file:///Users/sourabh/Downloads/idtt/evaluation/eval.py) | `app/main.py`, CLI | `app/orchestrator`, `app/rag`, `app/llm`, `psutil` |

---

*This concludes the master technical specification manual for all Python modules across the BMU Placement Intelligence Platform.*
