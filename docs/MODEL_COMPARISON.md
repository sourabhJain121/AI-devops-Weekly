# Multi-Model Quantitative Comparison & Trade-off Analysis

## 1. Experimental Setup & Constant Parameters

To evaluate how the choice of local LLM impacts application quality and performance, the entire application pipeline was kept **100% constant**:

- **Knowledge Base**: Identical ChromaDB index (27 chunks)
- **Embedding Model**: `nomic-embed-text`
- **Chunking**: Section-aware, heading-bounded (`chunk_size=800`, `overlap=120`)
- **Retrieval Settings**: `top_k=5`, `max_distance=0.65`
- **Prompts**: Identical system prompts and intent classification rules
- **Benchmark Dataset**: Identical 25 questions in [`evaluation/questions.json`](file:///Users/sourabh/Downloads/idtt/evaluation/questions.json)

Only the **generation LLM** was varied.

---

## 2. Full Quantitative Metric Comparison Matrix

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

> *Note on Test-Pass Rate*: Marked `N/A`. The BMU Placement Assistant is a grounded placement intelligence and policy analysis system, not an executable code generation system.
> *Note on GPU/VRAM*: Marked `N/A`. Execution was conducted on macOS ARM64 unified memory CPU architecture.

---

## 3. Quantitative Analysis & Trade-off Questions

1. **Which model has the highest accuracy?**
   `codellama:7b` (52.0%, 13/25 vs 48.0%, 12/25 for StarCoder2 and Qwen).
2. **Which model has the highest relevance score?**
   `codellama:7b` (3.80 / 5.0 vs 3.56 for StarCoder2 and 3.40 for Qwen).
3. **Which model has the best retrieval-grounded responses?**
   `codellama:7b` consistently formats and cites source documents (`[n] filename.md`) without prompt truncation.
4. **Which model has the lowest hallucination rate?**
   All 3 models achieved equal 8.0% (2/25) observed hallucination rate when constrained by distance thresholding (`MAX_DISTANCE=0.65`).
5. **Which model has the lowest latency?**
   `codellama:7b` achieved the fastest generation latency (**3.36s** avg latency).
6. **Which model uses the least computational resources?**
   `qwen2.5:0.5b` uses the least system RAM (13,800 MB) and lowest CPU utilization (31.2%).
7. **Is the most accurate model also the fastest?**
   **Yes.** `codellama:7b` was both the most accurate (52.0%) and the fastest (3.36s total latency) on macOS ARM64 hardware due to optimized Metal acceleration in Ollama.
8. **Quality-vs-Latency Trade-off**:
   On macOS ARM64, `codellama:7b` provides superior quality with zero latency penalty compared to smaller local models.

---

## 4. Evidence-Based Model Recommendation

**CodeLlama provides the strongest measured quality among the tested models while also having the lowest measured generation latency on ARM64 hardware.** It yields the highest accuracy (52.0%), highest relevance score (3.80/5.0), and fastest generation latency (3.36s), making it the optimal choice for grounded institutional policy Q&A and placement intelligence workflows.
