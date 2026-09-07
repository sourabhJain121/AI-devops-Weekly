"""Tests for Interview Question Bank: validation, ingestion, retrieval, guardrails."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app import config, vectorstore
from app.services import question_bank


# ---------------------------------------------------------------------------
# Test 9 — JSON validation (120 questions)
# ---------------------------------------------------------------------------

def test_question_bank_json_validation():
    """Dataset must contain exactly 120 valid records."""
    report = question_bank.validate_question_bank()
    assert report["valid"] is True, f"Validation errors: {report['errors']}"
    assert report["record_count"] == 120


# ---------------------------------------------------------------------------
# Test 10 — Unique question IDs
# ---------------------------------------------------------------------------

def test_question_ids_unique():
    """No duplicate question_id values."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [q["question_id"] for q in data]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {[i for i in ids if ids.count(i) > 1]}"


# ---------------------------------------------------------------------------
# Test 11 — Metadata preservation
# ---------------------------------------------------------------------------

def test_metadata_fields_present():
    """All required metadata fields must exist in every record."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"question_id", "question", "question_type", "category",
                "subcategory", "difficulty", "skills", "role", "company_id",
                "source", "answer", "explanation", "expected_topics", "tags"}
    for record in data:
        for field in required:
            assert field in record, f"Missing field '{field}' in record {record.get('question_id', '?')}"


# ---------------------------------------------------------------------------
# Test 12 — Source metadata
# ---------------------------------------------------------------------------

def test_source_is_interview_question_bank():
    """Every record must have source = 'interview_question_bank'."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        assert record["source"] == "interview_question_bank", \
            f"Record {record['question_id']} has source='{record['source']}'"


# ---------------------------------------------------------------------------
# Test 13 — company_id is null
# ---------------------------------------------------------------------------

def test_company_id_is_null():
    """Generic questions must have company_id = null."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        assert record["company_id"] is None, \
            f"Record {record['question_id']} has company_id='{record['company_id']}' (expected null)"


# ---------------------------------------------------------------------------
# Test 14 — Idempotent ingestion
# ---------------------------------------------------------------------------

def _make_dummy_vec(seed: str) -> list[float]:
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(768)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [_make_dummy_vec(t) for t in texts]


def _mock_embed_one(text: str) -> list[float]:
    return _make_dummy_vec(text)


# ---------------------------------------------------------------------------
# Test 14 — Idempotent ingestion
# ---------------------------------------------------------------------------

def test_idempotent_ingestion():
    """Running ingestion twice must NOT create duplicate records.
    This test uses mock embeddings to avoid requiring Ollama."""
    with patch.object(question_bank.llm, "embed", side_effect=_mock_embed):
        # First ingestion
        result1 = question_bank.ingest_question_bank()
        assert result1["status"] == "success"
        assert result1["ingested"] == 120

        # Count questions after first ingestion
        items1 = vectorstore.get_by_where({"source_type": "interview_question_bank"})
        count1 = len(items1)

        # Second ingestion (should upsert, not duplicate)
        result2 = question_bank.ingest_question_bank()
        assert result2["status"] == "success"
        assert result2["ingested"] == 120

        # Count should be the same
        items2 = vectorstore.get_by_where({"source_type": "interview_question_bank"})
        count2 = len(items2)
        assert count2 == count1, f"Duplicates created: {count1} -> {count2}"


# ---------------------------------------------------------------------------
# Test 15 — Category filtering
# ---------------------------------------------------------------------------

def test_category_filtering():
    """Filtering by category='Python' returns Python questions."""
    with patch.object(question_bank.llm, "embed", side_effect=_mock_embed):
        question_bank.ingest_question_bank()

    results = question_bank.retrieve_questions(category="Python", top_k=20)
    assert len(results) > 0
    for r in results:
        assert r.get("category") == "Python", f"Expected Python, got {r.get('category')}"


# ---------------------------------------------------------------------------
# Test 16 — Difficulty filtering
# ---------------------------------------------------------------------------

def test_difficulty_filtering():
    """Filtering by difficulty='hard' returns hard questions."""
    with patch.object(question_bank.llm, "embed", side_effect=_mock_embed):
        question_bank.ingest_question_bank()

    results = question_bank.retrieve_questions(difficulty="hard", top_k=20)
    assert len(results) > 0
    for r in results:
        assert r.get("difficulty") == "hard", f"Expected hard, got {r.get('difficulty')}"


# ---------------------------------------------------------------------------
# Test 17 — Generic question guardrail
# ---------------------------------------------------------------------------

def test_generic_question_guardrail():
    """Generic questions (company_id=null) must never be labeled as company-specific."""
    with patch.object(question_bank.llm, "embed", side_effect=_mock_embed):
        question_bank.ingest_question_bank()

    results = question_bank.retrieve_questions(category="Python", top_k=5)
    for r in results:
        # company_id should be empty string (ChromaDB representation of null)
        assert r.get("company_id") in (None, "", "null"), \
            f"Generic question {r.get('question_id')} should not have company_id"
        assert r.get("_is_generic") is True, \
            f"Generic question {r.get('question_id')} not flagged as generic"
        assert r.get("_company_attribution") is None


# ---------------------------------------------------------------------------
# Test 18 — Personalization
# ---------------------------------------------------------------------------

def test_personalized_interview_prep():
    """Given mock resume + JD + skill gaps, system retrieves relevant questions."""
    from app.services import resume as resume_svc, jd as jd_svc

    # Set up mock candidate
    resume_svc._CANDIDATE_PROFILES["test_personal"] = {
        "candidate_id": "test_personal",
        "name": "Test Candidate",
        "cgpa": 8.0,
        "backlogs": 0,
        "degree": "B.Tech",
        "skills": ["Python", "SQL", "RAG"],
    }

    # Set up mock JD
    jd_svc._COMPANY_JDS["test_comp_personal"] = {
        "company_id": "test_comp_personal",
        "company_name": "TestCorp",
        "role_title": "Software Engineer",
        "required_skills": ["Python", "FastAPI", "Docker", "SQL", "RAG"],
    }

    with patch.object(question_bank.llm, "embed", side_effect=_mock_embed):
        question_bank.ingest_question_bank()

    with patch.object(question_bank.llm, "embed_one", side_effect=_mock_embed_one):
        result = question_bank.personalized_interview_prep(
            candidate_id="test_personal",
            company_id="test_comp_personal",
        )

    assert result["personalized"] is True
    assert len(result["questions"]) > 0
    assert "summary" in result


# ---------------------------------------------------------------------------
# Additional validation tests
# ---------------------------------------------------------------------------

def test_valid_difficulty_values():
    """All difficulty values must be easy, medium, or hard."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = {"easy", "medium", "hard"}
    for record in data:
        assert record["difficulty"] in valid, \
            f"Invalid difficulty '{record['difficulty']}' for {record['question_id']}"


def test_categories_preserved():
    """All expected categories are represented."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = set(r["category"] for r in data)
    expected = {
        "Python", "DSA", "DBMS & SQL", "OOP", "Operating Systems",
        "Computer Networks", "Backend / APIs / FastAPI",
        "Git / Docker / DevOps", "Machine Learning", "GenAI / LLMs",
        "RAG / Embeddings / Vector DB", "System Design",
        "Data Science / Data Engineering", "HR / Behavioral / Placement",
    }
    for cat in expected:
        assert cat in categories, f"Missing category: {cat}"


def test_skills_and_tags_preserved():
    """Skills and tags fields are non-empty lists."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        assert isinstance(record["skills"], list), f"{record['question_id']}: skills must be list"
        assert len(record["skills"]) > 0, f"{record['question_id']}: skills must not be empty"
        assert isinstance(record["tags"], list), f"{record['question_id']}: tags must be list"
        assert len(record["tags"]) > 0, f"{record['question_id']}: tags must not be empty"


def test_expected_topics_preserved():
    """expected_topics field is a non-empty list."""
    path = config.INTERVIEW_QUESTION_BANK_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        assert isinstance(record["expected_topics"], list), \
            f"{record['question_id']}: expected_topics must be list"
        assert len(record["expected_topics"]) > 0, \
            f"{record['question_id']}: expected_topics must not be empty"


# ---------------------------------------------------------------------------
# Test 21 — Question Bank Keyword Search & API Endpoint
# ---------------------------------------------------------------------------

def test_question_bank_keyword_search():
    """Keyword search correctly retrieves matching questions with top relevance."""
    # Test topic searches
    deadlock_res = question_bank.retrieve_questions(query="deadlock")
    assert len(deadlock_res) > 0
    assert any("deadlock" in q["question"].lower() for q in deadlock_res)
    assert deadlock_res[0]["question_id"] == "OS-003"

    indexing_res = question_bank.retrieve_questions(query="indexing")
    assert len(indexing_res) > 0
    assert indexing_res[0]["question_id"] == "DB-005"

    fastapi_res = question_bank.retrieve_questions(query="fastapi")
    assert len(fastapi_res) > 0
    assert any("fastapi" in q["question"].lower() or "fastapi" in " ".join(q.get("skills", [])).lower() for q in fastapi_res)

    # Test exact question_id lookup
    id_res = question_bank.retrieve_questions(query="OS-003")
    assert len(id_res) > 0
    assert id_res[0]["question_id"] == "OS-003"


def test_question_bank_api_search_endpoint():
    """GET /api/interview/questions with query parameter returns filtered results."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/interview/questions?query=deadlock")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert data["results"][0]["question_id"] == "OS-003"

    # Category + query combined
    resp2 = client.get("/api/interview/questions?category=Operating+Systems&query=deadlock")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["results"]) > 0
    assert data2["results"][0]["question_id"] == "OS-003"

    # Category + Difficulty combined (e.g. Python + easy)
    resp3 = client.get("/api/interview/questions?category=Python&difficulty=easy")
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["total"] == 7
    assert all(q["category"] == "Python" and q["difficulty"] == "easy" for q in data3["results"])
    assert all(isinstance(q["skills"], list) for q in data3["results"])


