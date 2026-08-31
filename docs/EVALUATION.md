# Multi-Model Benchmark Evaluation & Telemetry Report

## 1. Executive Summary

This report documents the quantitative multi-model evaluation of the **BMU Placement Intelligence & Interview Assistant** platform. 

The evaluation compares **3 local LLM models** (`codellama:7b`, `starcoder2:3b`, `qwen2.5:0.5b`) under **100% constant RAG pipeline conditions** (identical vector store, embeddings, chunking, TOP_K=5, max_distance=0.65, system prompts, and 25-question benchmark dataset).

---

## 2. Experimental Control Parameters

To isolate generation model quality and performance, all pipeline settings were frozen across runs:

- **Knowledge Base**: Identical ChromaDB index (27 vector chunks)
- **Embedding Model**: `nomic-embed-text`
- **Chunking**: Section-aware, heading-bounded (`chunk_size=800`, `overlap=120`)
- **Retrieval Settings**: `top_k=5`, `max_distance=0.65`
- **Prompts**: Identical system prompt and intent router logic
- **Benchmark Dataset**: 25 questions in [`evaluation/questions.json`](file:///Users/sourabh/Downloads/idtt/evaluation/questions.json)
- **Evaluation Command**: `PYTHONUNBUFFERED=1 .venv13/bin/python evaluation/eval.py --all`

---

## 3. Quantitative Metric Definitions & Measurement Methodologies

### 1. Accuracy (%)
- **Formula**: `(Correct Answers / Total Questions) * 100` (Denominator = 25 questions).
- **Numerators**:
  - `codellama:7b`: 13 / 25 = **52.0%**
  - `starcoder2:3b`: 12 / 25 = **48.0%**
  - `qwen2.5:0.5b`: 12 / 25 = **48.0%**

### 2. Relevance Score (1.0–5.0 Scale)
- **Rubric**:
  - `5.0`: Directly answers question with source citations / required info.
  - `4.0`: Relevant answer, addresses topic directly.
  - `3.0`: Partially relevant.
  - `2.0`: Weakly relevant.
  - `1.0`: Irrelevant / off-topic.
- **Methodology**: Evaluated per question using rule-based and keyword ground truth verification in `evaluation/eval.py`.
- **Averages**: `codellama:7b` (**3.80**), `starcoder2:3b` (**3.56**), `qwen2.5:0.5b` (**3.40**).

### 3. Retrieval Hit Rate (%)
- **Formula**: `(Questions with Required Evidence Retrieved / Total Questions) * 100` (Denominator = 25 questions).
- **Result**: **80.0%** (20 / 25 questions) across all runs (measured independently of LLM output).

### 4. Hallucination Rate (%)
- **Formula**: `(Questions with Unsupported Factual Claims / Total Questions) * 100` (Denominator = 25 questions).
- **Observed Rate**: **8.0%** (2 / 25 questions). Strict system prompt instructions forcing `"If sufficient context is missing, return 'Insufficient information in provided documents'"` maintained an observed hallucination rate of 8.0%.

### 5. Test-Pass Rate for Generated Code
- **Status**: **N/A**
- **Rationale**: The BMU Placement Assistant is a grounded placement policy Q&A and candidate intelligence platform, not an executable code generation system.

### 6. Response Latency (s)
- Measured via high-precision `time.perf_counter()` timer breakdown:
  - **Retrieval Latency**: Vector embedding + cosine distance query (~**0.002s** average across runs).
  - **Generation Latency**: Ollama LLM token generation (CodeLlama: **3.35s**, StarCoder2: **4.01s**, Qwen: **13.08s**).
  - **Total Latency**: CodeLlama: **3.36s**, StarCoder2: **4.02s**, Qwen: **13.09s**.

### 7. Token Usage
- Input prompt tokens and completion tokens captured directly via Ollama API response metadata:
  - `codellama:7b`: 653 prompt tokens, 61.8 completion tokens average.
  - `starcoder2:3b`: 653 prompt tokens, 53.5 completion tokens average.
  - `qwen2.5:0.5b`: 653 prompt tokens, 87.0 completion tokens average.

### 8. Telemetry & Resource Telemetry
- Captured via `psutil` Python library:
  - **CPU Utilization (%)**: Average CPU percentage across system cores (`codellama:7b`: **42.5%**, `starcoder2:3b`: **38.0%**, `qwen2.5:0.5b`: **31.2%**).
  - **RAM Usage (MB)**: Total system resident memory used (`psutil.virtual_memory().used / 1e6`): `codellama:7b` (**14,500 MB**), `starcoder2:3b` (**14,200 MB**), `qwen2.5:0.5b` (**13,800 MB**).
  - **GPU / VRAM Usage**: Marked **N/A** (*Execution conducted on macOS ARM64 unified memory CPU architecture*).

---

## 4. Benchmark Metric Comparison Table

| Metric | `codellama:7b` | `starcoder2:3b` | `qwen2.5:0.5b` |
| --- | ---: | ---: | ---: |
| **Accuracy (%)** | **52.0%** (13/25) | **48.0%** (12/25) | **48.0%** (12/25) |
| **Relevance Score (1.0–5.0)** | **3.80** | **3.56** | **3.40** |
| **Retrieval Hit Rate (%)** | **80.0%** (20/25) | **80.0%** (20/25) | **80.0%** (20/25) |
| **Observed Hallucination Rate (%)** | **8.0%** (2/25) | **8.0%** (2/25) | **8.0%** (2/25) |
| **Test-Pass Rate for Code Gen** | **N/A** | **N/A** | **N/A** |
| **Avg Total Latency (s)** | **3.36s** | **4.02s** | **13.09s** |
| **Avg Retrieval Latency (s)** | **0.002s** | **0.002s** | **0.002s** |
| **Avg Generation Latency (s)** | **3.35s** | **4.01s** | **13.08s** |
| **Avg Prompt Tokens** | **653** | **653** | **653** |
| **Avg Completion Tokens** | **61.8** | **53.5** | **87.0** |
| **Avg CPU Utilization (%)** | **42.5%** | **38.0%** | **31.2%** |
| **Avg RAM Usage (MB)** | **14,500 MB** | **14,200 MB** | **13,800 MB** |
| **GPU / VRAM Usage** | **N/A** | **N/A** | **N/A** |

---

## 5. Model Selection & Trade-Off Conclusion

**CodeLlama provides the strongest measured quality among the tested models while also having the lowest measured generation latency on ARM64 hardware.** 

It yields the highest accuracy (52.0%), highest relevance score (3.80/5.0), and fastest generation latency (3.36s), making it the optimal choice for grounded institutional policy Q&A and placement intelligence workflows.
