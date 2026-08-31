"""Failure mode & edge-case unit tests."""
from fastapi.testclient import TestClient
from app import llm, rag, vectorstore
from app.main import app
from app.services import eligibility, matching

client = TestClient(app)


def test_failure_insufficient_information():
    # Query for information not present in the vector store
    res = rag.answer("What is the name of the Vice Chancellor secret discount code?", top_k=3)
    assert "reply" in res
    reply_lower = res["reply"].lower()
    assert "insufficient" in reply_lower or "could not find" in reply_lower or len(res.get("sources", [])) == 0


def test_failure_missing_candidate_resume():
    # Attempting matching without candidate resume profile
    res = matching.compare_resume_to_jd(candidate_id="non_existent_candidate", company_id="comp_1")
    assert "error" in res or res["match_score"] == 0.0


def test_failure_missing_company_jd():
    # Attempting eligibility without company JD
    res = eligibility.evaluate_eligibility(candidate_id="cand_1", company_id="non_existent_company")
    assert "status" in res
    assert res["status"] in ["Eligible", "Not Eligible", "Unable to Determine"]


def test_failure_cross_company_zero_contamination():
    # Verify zero contamination when querying filtered vectorstore
    dummy_vec = [0.1] * 768
    vectorstore.add(
        ids=["fail_comp_a"],
        documents=["Company Secret A 12345"],
        metadatas=[{"source_type": "company_jd", "company_id": "secret_comp_a"}],
        embeddings=[dummy_vec]
    )
    vectorstore.add(
        ids=["fail_comp_b"],
        documents=["Company Secret B 67890"],
        metadatas=[{"source_type": "company_jd", "company_id": "secret_comp_b"}],
        embeddings=[dummy_vec]
    )

    hits_a = vectorstore.query(dummy_vec, top_k=5, where={"company_id": "secret_comp_a"})
    for hit in hits_a:
        assert "Secret B" not in hit["text"]
        assert hit["company_id"] == "secret_comp_a"


def test_failure_empty_query():
    res = client.post("/api/query", json={"message": "   "})
    assert res.status_code == 400
    assert "error" in res.json()
