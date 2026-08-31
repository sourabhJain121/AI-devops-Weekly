# Week 4 Completion Matrix

This document provides explicit evidence and status tracking for all Week 4 requirements.

| Exercise | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| **1** | **3 LLM models** | `ollama list` (`codellama:7b`, `starcoder2:3b`, `qwen2.5:0.5b`) | **VERIFIED** |
| **1** | **Same application** | `app/main.py` (FastAPI backend remains identical) | **VERIFIED** |
| **1** | **Same prompts** | `app/rag.py` (Identical `SYSTEM_PROMPT` across all runs) | **VERIFIED** |
| **1** | **Same questions** | `evaluation/questions.json` (25 questions used for all models) | **VERIFIED** |
| **1** | **Same KB** | `storage/chroma` (Identical 27 vector chunks) | **VERIFIED** |
| **2** | **20–30 questions** | `evaluation/questions.json` (25 structured questions) | **VERIFIED** |
| **2** | **Actual use-case** | Placement Q&A, eligibility, JD match, skill gaps | **VERIFIED** |
| **2** | **Same dataset** | Single benchmark file used for CodeLlama, StarCoder2, Qwen | **VERIFIED** |
| **3** | **Accuracy** | Measured accuracy % (`codellama_results.json`) | **VERIFIED** |
| **3** | **Relevance** | 1.0 to 5.0 scale evaluator (`evaluation/eval.py`) | **VERIFIED** |
| **3** | **Retrieval Quality** | Independent hit rate (80.0%) measured on vector hits | **VERIFIED** |
| **3** | **Hallucination** | Unsupported claim tracking (8.0% rate) | **VERIFIED** |
| **3** | **Test-Pass Rate/N/A** | Marked `N/A` (Placement app is not code generator) | **N/A** |
| **3** | **Latency** | Retrieval + Generation latency breakdown | **VERIFIED** |
| **3** | **Token Usage** | Input prompt & completion tokens tracked via Ollama | **VERIFIED** |
| **3** | **CPU** | Average CPU % tracked via `psutil` | **VERIFIED** |
| **3** | **GPU/VRAM** | Marked `N/A` (ARM64 unified memory CPU architecture) | **N/A** |
| **3** | **Memory** | RAM MB tracked via `psutil` | **VERIFIED** |
| **4** | **Quantitative comparison** | `docs/MODEL_COMPARISON.md` comparison table | **VERIFIED** |
| **4** | **Quality analysis** | Quality vs latency trade-off analysis | **VERIFIED** |
| **4** | **Performance analysis** | Resource consumption analysis | **VERIFIED** |
| **5** | **Relevant retrieval** | `docs/RAG_ANALYSIS.md` (Case A / D trace) | **VERIFIED** |
| **5** | **Irrelevant retrieval** | `docs/RAG_ANALYSIS.md` (Case B trace) | **VERIFIED** |
| **5** | **Missed information** | `docs/RAG_ANALYSIS.md` (Case C trace) | **VERIFIED** |
| **5** | **Correct RAG response** | `docs/RAG_ANALYSIS.md` (Case D trace) | **VERIFIED** |
| **5** | **Hallucination despite context** | `docs/RAG_ANALYSIS.md` (Case E trace) | **VERIFIED** |
| **5** | **Retrieval→Context→Response** | `docs/RAG_ANALYSIS.md` pipeline analysis | **VERIFIED** |
| **6** | **Repository benchmark** | `evaluation/repository_questions.json` (10 questions) | **VERIFIED** |
| **6** | **Multi-file questions** | 5 execution flow tracing questions across multiple modules | **VERIFIED** |
| **6** | **Component relationships** | `docs/REPOSITORY_ANALYSIS.md` | **VERIFIED** |

*Allowed Statuses*: `NOT STARTED`, `PARTIAL`, `IMPLEMENTED`, `VERIFIED`, `N/A`. (Status is `VERIFIED` only when actual empirical evidence exists).
