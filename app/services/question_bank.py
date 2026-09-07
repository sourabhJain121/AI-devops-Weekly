"""Interview Question Bank Service: Validation, Ingestion, Retrieval & Personalization.

The authoritative dataset is:
  knowledge/interview_question_bank/interview_question_bank.json

Questions are ingested into the EXISTING ChromaDB collection with
source_type = 'interview_question_bank' and deterministic IDs of the form
  interview_question_bank:<question_id>
so re-ingestion is idempotent and never creates duplicates.
"""
import json
import logging
import re
from pathlib import Path

from app import config, llm, vectorstore

log = logging.getLogger(__name__)

REQUIRED_FIELDS = {"question_id", "question", "question_type", "category",
                   "difficulty", "source"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
EMBED_BATCH = 32


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_question_bank(path: Path | None = None) -> dict:
    """Validate the question bank JSON file.

    Returns a report dict with status, record_count, errors.
    """
    path = Path(path or config.INTERVIEW_QUESTION_BANK_PATH)
    report: dict = {
        "path": str(path),
        "valid": False,
        "record_count": 0,
        "errors": [],
        "warnings": [],
    }

    if not path.exists():
        report["errors"].append(f"File not found: {path}")
        return report

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["errors"].append(f"Invalid JSON: {exc}")
        return report

    if not isinstance(data, list):
        report["errors"].append("Root element must be a JSON array")
        return report

    report["record_count"] = len(data)

    if len(data) != 120:
        report["errors"].append(f"Expected exactly 120 records, found {len(data)}")

    seen_ids: set[str] = set()
    for i, record in enumerate(data):
        qid = record.get("question_id", f"<missing at index {i}>")

        # Unique ID check
        if qid in seen_ids:
            report["errors"].append(f"Duplicate question_id: {qid}")
        seen_ids.add(qid)

        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in record or record[field] is None:
                report["errors"].append(f"[{qid}] missing required field: {field}")

        # Source check
        if record.get("source") != "interview_question_bank":
            report["errors"].append(
                f"[{qid}] source must be 'interview_question_bank', got '{record.get('source')}'"
            )

        # company_id must be null
        if record.get("company_id") is not None:
            report["warnings"].append(
                f"[{qid}] company_id should be null for generic questions, got '{record.get('company_id')}'"
            )

        # Difficulty validation
        diff = record.get("difficulty", "")
        if diff and diff.lower() not in VALID_DIFFICULTIES:
            report["warnings"].append(f"[{qid}] unusual difficulty value: '{diff}'")

    report["valid"] = len(report["errors"]) == 0
    return report


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_question_bank(path: Path | None = None) -> dict:
    """Ingest the question bank into the existing ChromaDB vector store.

    Uses deterministic IDs (interview_question_bank:<question_id>) for
    idempotent upsert — running twice never creates duplicates.
    """
    path = Path(path or config.INTERVIEW_QUESTION_BANK_PATH)

    # Validate first
    validation = validate_question_bank(path)
    if not validation["valid"]:
        log.error("Question bank validation failed: %s", validation["errors"])
        return {
            "status": "validation_failed",
            "validation": validation,
            "ingested": 0,
            "failed": 0,
            "skipped": 0,
        }

    data = json.loads(path.read_text(encoding="utf-8"))

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    skipped = 0

    for record in data:
        qid = record["question_id"]
        question_text = record.get("question", "")

        if not question_text.strip():
            log.warning("Skipping question %s: empty question text", qid)
            skipped += 1
            continue

        # Build rich text for embedding: question + answer hints + topics
        embedding_text_parts = [question_text]
        if record.get("answer"):
            embedding_text_parts.append(record["answer"])
        if record.get("explanation"):
            embedding_text_parts.append(record["explanation"])
        if record.get("expected_topics"):
            embedding_text_parts.append("Topics: " + ", ".join(record["expected_topics"]))
        embedding_text = "\n".join(embedding_text_parts)

        # Deterministic ID
        doc_id = f"interview_question_bank:{qid}"

        # Flatten metadata for ChromaDB (no nested objects)
        meta = {
            "document_id": doc_id,
            "document_name": "interview_question_bank.json",
            "source_type": "interview_question_bank",
            "question_id": qid,
            "question": question_text,
            "question_type": record.get("question_type", ""),
            "category": record.get("category", ""),
            "subcategory": record.get("subcategory", ""),
            "difficulty": record.get("difficulty", ""),
            "skills": ", ".join(record.get("skills", [])),
            "role": record.get("role", ""),
            "company_id": "",  # ChromaDB doesn't support None in metadata
            "source": record.get("source", "interview_question_bank"),
            "answer": record.get("answer", ""),
            "explanation": record.get("explanation", ""),
            "expected_topics": ", ".join(record.get("expected_topics", [])),
            "tags": ", ".join(record.get("tags", [])),
            "section": record.get("category", "Interview Question"),
            "page": 1,
            "chunk_index": 0,
        }

        ids.append(doc_id)
        texts.append(embedding_text)
        metadatas.append(meta)

    # Embed and upsert in batches
    ingested = 0
    failed = 0
    try:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = slice(start, start + EMBED_BATCH)
            batch_texts = texts[batch]
            batch_ids = ids[batch]
            batch_metas = metadatas[batch]

            try:
                embeddings = llm.embed(batch_texts)
                vectorstore.add(batch_ids, batch_texts, batch_metas, embeddings)
                ingested += len(batch_texts)
            except Exception as exc:
                log.error("Failed to ingest batch starting at %d: %s", start, exc)
                failed += len(batch_texts)
    except Exception as exc:
        log.error("Question bank ingestion failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "ingested": ingested,
            "failed": failed + (len(texts) - ingested),
            "skipped": skipped,
        }

    log.info(
        "Question bank ingestion complete: %d ingested, %d failed, %d skipped",
        ingested, failed, skipped,
    )

    return {
        "status": "success",
        "total_records": len(data),
        "ingested": ingested,
        "failed": failed,
        "skipped": skipped,
        "duplicates": 0,  # upsert ensures no duplicates
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_json_question_bank(
    query: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    difficulty: str | None = None,
    skills: str | None = None,
    role: str | None = None,
    question_type: str | None = None,
    tags: str | None = None,
    top_k: int = 30,
) -> list[dict]:
    """Direct, high-speed keyword search over the authoritative 120-record JSON question bank."""
    path = Path(config.INTERVIEW_QUESTION_BANK_PATH)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    query_clean = (query or "").strip().lower()
    q_words = [w for w in re.findall(r"[a-z0-9]+", query_clean) if len(w) > 1 or len(query_clean) <= 2]
    cat_lower = category.lower().strip() if category else None
    diff_lower = difficulty.lower().strip() if difficulty else None
    skill_lower = skills.lower().strip() if skills else None
    subcat_lower = subcategory.lower().strip() if subcategory else None
    role_lower = role.lower().strip() if role else None
    qtype_lower = question_type.lower().strip() if question_type else None

    scored = []
    for r in data:
        # Category check
        if cat_lower and cat_lower != r.get("category", "").lower().strip():
            continue
        # Subcategory check
        if subcat_lower and subcat_lower != r.get("subcategory", "").lower().strip():
            continue
        # Difficulty check
        if diff_lower and diff_lower != r.get("difficulty", "").lower().strip():
            continue
        # Role check
        if role_lower and role_lower != r.get("role", "").lower().strip():
            continue
        # Question type check
        if qtype_lower and qtype_lower != r.get("question_type", "").lower().strip():
            continue
        # Skills check
        if skill_lower:
            r_skills = [s.lower() for s in r.get("skills", [])]
            if not any(skill_lower in s for s in r_skills):
                continue

        # If search words or phrase provided, score relevance
        if query_clean:
            qid_lower = (r.get("question_id") or "").lower()
            q_text_lower = (r.get("question") or "").lower()
            expected = r.get("expected_topics") or []
            expected_str = ", ".join(expected) if isinstance(expected, list) else str(expected)

            haystack = (
                qid_lower + " " +
                q_text_lower + " " +
                (r.get("category") or "") + " " +
                (r.get("subcategory") or "") + " " +
                " ".join(r.get("skills") or []) + " " +
                " ".join(r.get("tags") or []) + " " +
                expected_str + " " +
                (r.get("answer") or "") + " " +
                (r.get("explanation") or "")
            ).lower()

            tokens_set = set(re.findall(r"[a-z0-9]+", haystack))
            q_tokens_set = set(re.findall(r"[a-z0-9]+", q_text_lower))

            score = 0
            # Direct ID match
            if query_clean == qid_lower:
                score += 50
            elif query_clean in qid_lower:
                score += 25

            # Exact phrase match in question
            if query_clean in q_text_lower:
                score += 20
            elif query_clean in haystack:
                score += 10

            # Whole-word token overlap
            for w in q_words:
                if w in q_tokens_set:
                    score += 5
                elif w in tokens_set:
                    score += 2

            if score > 0:
                item = dict(r)
                item["_is_generic"] = True
                item["_company_attribution"] = None
                scored.append((score, item))
        else:
            item = dict(r)
            item["_is_generic"] = True
            item["_company_attribution"] = None
            scored.append((1, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def retrieve_questions(
    query: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    difficulty: str | None = None,
    skills: str | None = None,
    role: str | None = None,
    question_type: str | None = None,
    tags: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve interview questions with metadata filtering and/or semantic search.

    Supports both filtered retrieval and semantic search.
    """
    # Build metadata filter
    where_conditions: list[dict] = [{"source_type": "interview_question_bank"}]

    if category:
        where_conditions.append({"category": category})
    if subcategory:
        where_conditions.append({"subcategory": subcategory})
    if difficulty:
        where_conditions.append({"difficulty": difficulty.lower()})
    if question_type:
        where_conditions.append({"question_type": question_type})
    if role:
        where_conditions.append({"role": role})

    # Combine filters
    if len(where_conditions) == 1:
        where = where_conditions[0]
    else:
        where = {"$and": where_conditions}

    # 1. Authoritative question bank lookup (handles both query search and category/difficulty/skills filters)
    direct = search_json_question_bank(
        query=query,
        category=category,
        subcategory=subcategory,
        difficulty=difficulty,
        skills=skills,
        role=role,
        question_type=question_type,
        tags=tags,
        top_k=top_k,
    )
    if direct:
        return direct

    # 2. Vectorstore fallback
    if query:
        try:
            embedding = llm.embed_one(query)
            hits = vectorstore.query(embedding, top_k=top_k, where=where)
        except Exception as exc:
            log.warning("Semantic search fallback to keyword filtering: %s", exc)
            hits = []
        if not hits:
            try:
                candidates = vectorstore.get_by_where(where)
                q_words = set(re.findall(r"\w+", query.lower()))
                scored = []
                for item in candidates:
                    text_lower = (
                        (item.get("text") or "") + " " +
                        (item.get("category") or "") + " " +
                        str(item.get("skills") or "")
                    ).lower()
                    overlap = sum(1 for w in q_words if w in text_lower)
                    scored.append((overlap, item))
                scored.sort(key=lambda x: x[0], reverse=True)
                hits = [item for _, item in scored[:top_k]]
            except Exception as exc:
                log.error("Fallback filtering failed: %s", exc)
                hits = []
    else:
        try:
            items = vectorstore.get_by_where(where)
            hits = items[:top_k]
        except Exception as exc:
            log.error("Metadata filtering failed: %s", exc)
            hits = []

    # Normalize fields and apply generic question guardrail
    for hit in hits:
        if isinstance(hit.get("skills"), str):
            hit["skills"] = [s.strip() for s in hit["skills"].split(",") if s.strip()]
        elif not hit.get("skills"):
            hit["skills"] = []

        if isinstance(hit.get("tags"), str):
            hit["tags"] = [t.strip() for t in hit["tags"].split(",") if t.strip()]
        elif not hit.get("tags"):
            hit["tags"] = []

        if isinstance(hit.get("expected_topics"), str):
            hit["expected_topics"] = [t.strip() for t in hit["expected_topics"].split(",") if t.strip()]
        elif not hit.get("expected_topics"):
            hit["expected_topics"] = []

        if not hit.get("company_id") or hit.get("company_id") == "":
            hit["_is_generic"] = True
            hit["_company_attribution"] = None
        else:
            hit["_is_generic"] = False
            hit["_company_attribution"] = hit.get("company_id")

    return hits


# ---------------------------------------------------------------------------
# Personalized interview preparation
# ---------------------------------------------------------------------------

def personalized_interview_prep(
    candidate_id: str = "default_candidate",
    company_id: str | None = None,
    top_k: int = 10,
) -> dict:
    """Combine resume + JD + skill gaps + question bank for personalized prep.

    Questions are prioritized by:
      1. Skill gap topics (highest priority)
      2. JD-required skills
      3. Candidate's existing skills (for depth testing)
    """
    from app.services import resume, jd as jd_service, skill_gap

    cand = resume.get_candidate_profile(candidate_id)
    comp = jd_service.get_company_jd(company_id) if company_id else None

    if not cand and not comp:
        return {
            "questions": [],
            "summary": "Upload both candidate resume and company JD to generate personalized interview preparation.",
            "personalized": False,
        }

    # Get skill gaps
    gaps_data = {}
    high_gaps: list[str] = []
    if cand and comp:
        gaps_data = skill_gap.analyze_skill_gaps(candidate_id=candidate_id, company_id=company_id)
        high_gaps = gaps_data.get("high_priority_gaps", [])

    # Build search queries based on priority
    queries: list[str] = []
    query_labels: list[str] = []

    # Priority 1: Skill gap questions
    for gap_skill in high_gaps[:3]:
        queries.append(f"{gap_skill} interview questions")
        query_labels.append(f"skill_gap:{gap_skill}")

    # Priority 2: JD-required skills
    if comp:
        req_skills = comp.get("required_skills", [])
        for skill in req_skills[:3]:
            if skill not in high_gaps:
                queries.append(f"{skill} interview questions")
                query_labels.append(f"jd_skill:{skill}")

    # Priority 3: Candidate skills (for depth)
    if cand:
        for skill in cand.get("skills", [])[:2]:
            queries.append(f"{skill} technical interview")
            query_labels.append(f"candidate_skill:{skill}")

    # Fallback: general questions
    if not queries:
        queries.append("software engineer interview questions")
        query_labels.append("general")

    # Retrieve questions for each priority
    all_questions: list[dict] = []
    seen_ids: set[str] = set()

    for query, label in zip(queries, query_labels):
        hits = retrieve_questions(query=query, top_k=5)
        for hit in hits:
            qid = hit.get("question_id", hit.get("document_id", ""))
            if qid not in seen_ids:
                seen_ids.add(qid)
                hit["_retrieval_reason"] = label
                all_questions.append(hit)

    # Limit total
    all_questions = all_questions[:top_k]

    # Apply generic question guardrail
    for q in all_questions:
        q["_is_generic"] = True
        q["_company_attribution"] = None

    # Build summary
    summary_parts = []
    if comp:
        summary_parts.append(
            f"🎯 **Personalized Interview Prep for {comp.get('company_name', 'Target Company')} "
            f"({comp.get('role_title', 'Role')})**\n"
        )
    else:
        summary_parts.append("🎯 **Interview Preparation Questions**\n")

    if high_gaps:
        summary_parts.append(
            f"⚠️ **Skill Gap Focus Areas:** {', '.join(high_gaps)}\n"
        )

    summary_parts.append(
        f"📝 Retrieved **{len(all_questions)}** relevant questions from the interview question bank.\n"
    )

    # Format questions
    for i, q in enumerate(all_questions, 1):
        reason = q.get("_retrieval_reason", "")
        category = q.get("category", "General")
        difficulty = q.get("difficulty", "medium")
        difficulty_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "⚪")
        question_text = q.get("question", q.get("text", ""))

        summary_parts.append(
            f"\n**{i}. [{category}] {difficulty_icon} {difficulty.capitalize()}**\n"
            f"{question_text}\n"
            f"*Expected topics:* {q.get('expected_topics', 'N/A')}"
        )

        if reason.startswith("skill_gap:"):
            summary_parts.append(f"\n💡 *This question targets your preparation gap in {reason.split(':')[1]}.*")

    summary_parts.append(
        "\n\n> ℹ️ These are generic interview-preparation questions from the question bank. "
        "They are not claimed to be questions officially asked by any specific company "
        "unless separately verified."
    )

    return {
        "questions": all_questions,
        "summary": "\n".join(summary_parts),
        "personalized": bool(cand and comp),
        "skill_gaps": high_gaps,
        "total_retrieved": len(all_questions),
    }
