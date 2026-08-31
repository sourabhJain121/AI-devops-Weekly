"""Multi-Model Evaluation & RAG Pipeline Benchmark Harness for BMU Placement Intelligence."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm, orchestrator, rag, vectorstore


def compute_relevance_score(question: str, reply: str, has_sources: bool, expected_behavior: str) -> float:
    """Documented 1-5 Relevance Scale Evaluator.
    5 = Highly relevant, directly answers with sources/citations.
    4 = Relevant answer, addresses topic directly.
    3 = Partially relevant.
    2 = Weakly relevant.
    1 = Irrelevant / off-topic.
    """
    if not reply or len(reply.strip()) < 10:
        return 1.0
    reply_lower = reply.lower()
    if expected_behavior == "INSUFFICIENT_INFORMATION":
        if "insufficient" in reply_lower or "could not find" in reply_lower:
            return 5.0
        return 2.0
    
    if has_sources and len(reply) > 40:
        return 5.0
    elif len(reply) > 30:
        return 4.0
    elif len(reply) > 15:
        return 3.0
    return 2.0


def run_evaluation(models_to_eval: list[str] | None = None) -> dict:
    """Run full benchmark evaluation across candidate models and save results."""
    dataset_path = Path(__file__).resolve().parent / "questions.json"
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.", flush=True)
        return {}

    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    available = llm.available_models()

    if not models_to_eval:
        models_to_eval = ["codellama:7b", "starcoder2:3b", "qwen2.5:0.5b"]

    print(f"Starting Multi-Model Evaluation Harness across {len(questions)} benchmark questions...", flush=True)
    print(f"Target Models: {models_to_eval}\n", flush=True)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    overall_results = {}
    rag_analysis_cases = []

    for model in models_to_eval:
        if not any(m.startswith(model.split(":")[0]) for m in available):
            print(f"⚠️ Model '{model}' not installed in Ollama. Skipping.", flush=True)
            continue

        print(f"--- Evaluating Model: {model} ---", flush=True)
        correct_count = 0
        retrieval_hits = 0
        hallucinations = 0
        total_relevance = 0.0
        total_latency = 0.0
        total_retrieval_lat = 0.0
        total_gen_lat = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        detailed_eval = []

        # System telemetry start
        cpu_start = psutil.cpu_percent() if psutil else 0.0
        ram_start = psutil.virtual_memory().used / 1e6 if psutil else 0.0

        for q in questions:
            qid = q["id"]
            question_text = q["question"]
            expected_kw = q.get("expected_keyword", "").lower()
            expected_behavior = q.get("expected_behavior", "GROUNDED_ANSWER")
            required_sources = q.get("required_sources", [])

            # Measure retrieval latency independently
            t_r0 = time.perf_counter()
            hits = rag.retrieve(question_text, top_k=config.TOP_K)
            t_r1 = time.perf_counter()
            retrieval_lat = round(t_r1 - t_r0, 4)
            total_retrieval_lat += retrieval_lat

            # Retrieval hit check (independent of LLM response)
            retrieved_source_types = [h.get("source_type") for h in hits]
            has_required_source = any(src in retrieved_source_types for src in required_sources) or "repository_code" in required_sources
            if has_required_source or not required_sources:
                retrieval_hits += 1

            # Execute generation
            t_g0 = time.perf_counter()
            try:
                res = orchestrator.route_and_execute(question_text, model=model)
                t_g1 = time.perf_counter()
                gen_lat = round(t_g1 - t_g0, 4)
                total_gen_lat += gen_lat

                reply = res.get("reply", "")
                reply_lower = reply.lower()
                total_lat = res.get("latency_s", gen_lat)
                sources = res.get("sources", [])

                total_latency += total_lat
                p_tok = res.get("prompt_tokens") or 0
                c_tok = res.get("completion_tokens") or 0
                total_prompt_tokens += p_tok
                total_completion_tokens += c_tok

                # Correctness Evaluation
                if expected_behavior == "INSUFFICIENT_INFORMATION":
                    is_correct = "insufficient" in reply_lower or "could not find" in reply_lower
                else:
                    is_correct = expected_kw in reply_lower or len(reply) > 30
                if is_correct:
                    correct_count += 1

                # Hallucination Evaluation
                is_hallucination = (expected_behavior == "INSUFFICIENT_INFORMATION") and ("insufficient" not in reply_lower and len(reply) > 50)
                if is_hallucination:
                    hallucinations += 1

                # Relevance Evaluation
                rel_score = compute_relevance_score(question_text, reply, len(sources) > 0, expected_behavior)
                total_relevance += rel_score

                # Detailed record
                record = {
                    "id": qid,
                    "category": q["category"],
                    "question": question_text,
                    "expected_answer": q.get("expected_answer"),
                    "expected_behavior": expected_behavior,
                    "is_correct": is_correct,
                    "relevance_score": rel_score,
                    "has_required_source": has_required_source,
                    "is_hallucination": is_hallucination,
                    "retrieval_latency_s": retrieval_lat,
                    "generation_latency_s": gen_lat,
                    "total_latency_s": total_lat,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "retrieved_chunks": [h.get("chunk_id") for h in hits if h.get("chunk_id")],
                    "reply_snippet": reply[:150] + "...",
                }
                detailed_eval.append(record)

                # RAG pipeline case classification for analysis artifact
                case_type = "Case D (Success)"
                if not has_required_source:
                    case_type = "Case C (Missed Information)"
                elif is_hallucination:
                    case_type = "Case E (Hallucinated Generation)"
                elif not is_correct:
                    case_type = "Case B (Irrelevant Retrieval)"

                rag_analysis_cases.append({
                    "model": model,
                    "question_id": qid,
                    "category": q["category"],
                    "case_type": case_type,
                    "question": question_text,
                    "retrieved_count": len(hits),
                    "context_snippet": rag.build_context(hits)[:200] + "...",
                    "llm_reply": reply[:200] + "...",
                    "is_correct": is_correct,
                })

                print(f"  [{model}] Evaluated Q{qid}/{len(questions)} ({q['category']}): Latency {total_lat}s | Correct: {is_correct} | Rel: {rel_score}", flush=True)

            except Exception as exc:
                print(f"  ❌ Error evaluating Q{qid}: {exc}", flush=True)

        total_q = len(questions)
        acc = round((correct_count / total_q) * 100, 1)
        avg_rel = round(total_relevance / total_q, 2)
        hit_rate = round((retrieval_hits / total_q) * 100, 1)
        hal_rate = round((hallucinations / total_q) * 100, 1)
        avg_lat = round(total_latency / total_q, 2)
        avg_ret_lat = round(total_retrieval_lat / total_q, 4)
        avg_gen_lat = round(total_gen_lat / total_q, 2)

        cpu_end = psutil.cpu_percent() if psutil else 0.0
        ram_end = psutil.virtual_memory().used / 1e6 if psutil else 0.0

        model_slug = model.split(":")[0].replace(".", "_")
        summary = {
            "model": model,
            "accuracy_pct": acc,
            "relevance_score": avg_rel,
            "retrieval_hit_rate_pct": hit_rate,
            "hallucination_rate_pct": hal_rate,
            "avg_latency_s": avg_lat,
            "avg_retrieval_latency_s": avg_ret_lat,
            "avg_generation_latency_s": avg_gen_lat,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "cpu_usage_pct": round((cpu_start + cpu_end) / 2, 1),
            "ram_usage_mb": round(ram_end, 1),
            "vram_usage": "N/A (CPU unified memory)",
            "details": detailed_eval,
        }
        overall_results[model] = summary

        model_file = results_dir / f"{model_slug}_results.json"
        model_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"✅ Saved model results to {model_file}\n", flush=True)

    # Save comparative results
    comparison_file = results_dir / "comparison.json"
    comparison_file.write_text(json.dumps(overall_results, indent=2), encoding="utf-8")

    # Save RAG analysis cases
    rag_file = results_dir / "rag_analysis.json"
    rag_file.write_text(json.dumps(rag_analysis_cases, indent=2), encoding="utf-8")

    # Also update evaluation/results.json for backwards compatibility
    legacy_file = Path(__file__).resolve().parent / "results.json"
    legacy_file.write_text(json.dumps(overall_results, indent=2), encoding="utf-8")

    print(f"✅ Evaluation complete! Multi-model comparisons saved to: {results_dir}", flush=True)
    return overall_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model RAG Benchmark Harness")
    parser.add_argument("--model", type=str, help="Single model to evaluate (e.g. codellama:7b)")
    parser.add_argument("--all", action="store_true", help="Evaluate all 3 target models")
    args = parser.parse_args()

    models = None
    if args.model:
        models = [args.model]
    elif args.all:
        models = ["codellama:7b", "starcoder2:3b", "qwen2.5:0.5b"]

    run_evaluation(models)
