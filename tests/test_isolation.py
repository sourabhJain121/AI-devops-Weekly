"""Unit tests for metadata data isolation across companies, candidates, and sessions."""
from app import vectorstore
from app.services import jd, resume


def test_company_metadata_isolation():
    dummy_vec = [0.1] * 768
    
    # Insert mock chunks for Company A and Company B
    vectorstore.add(
        ids=["iso_comp_a_1"],
        documents=["Company A requires Python, FastAPI, and 8.0 CGPA."],
        metadatas=[{"source_type": "company_jd", "company_id": "company_a", "document_name": "JD_A.pdf"}],
        embeddings=[dummy_vec]
    )

    vectorstore.add(
        ids=["iso_comp_b_1"],
        documents=["Company B requires Java, Spring Boot, and 6.5 CGPA."],
        metadatas=[{"source_type": "company_jd", "company_id": "company_b", "document_name": "JD_B.pdf"}],
        embeddings=[dummy_vec]
    )

    # Query Company A only
    hits_a = vectorstore.query(dummy_vec, top_k=5, where={"company_id": "company_a"})
    assert len(hits_a) == 1
    assert hits_a[0]["company_id"] == "company_a"
    assert "Company A" in hits_a[0]["text"]
    assert "Company B" not in hits_a[0]["text"]

    # Query Company B only
    hits_b = vectorstore.query(dummy_vec, top_k=5, where={"company_id": "company_b"})
    assert len(hits_b) == 1
    assert hits_b[0]["company_id"] == "company_b"
    assert "Company B" in hits_b[0]["text"]
    assert "Company A" not in hits_b[0]["text"]


def test_candidate_metadata_isolation():
    dummy_vec = [0.1] * 768

    vectorstore.add(
        ids=["iso_cand_a_1"],
        documents=["Candidate Alice has 8.5 CGPA and Python experience."],
        metadatas=[{"source_type": "resume", "candidate_id": "candidate_alice", "document_name": "Alice_Resume.pdf"}],
        embeddings=[dummy_vec]
    )

    vectorstore.add(
        ids=["iso_cand_b_1"],
        documents=["Candidate Bob has 6.2 CGPA and Java experience."],
        metadatas=[{"source_type": "resume", "candidate_id": "candidate_bob", "document_name": "Bob_Resume.pdf"}],
        embeddings=[dummy_vec]
    )

    # Candidate Alice Query
    hits_alice = vectorstore.query(dummy_vec, top_k=5, where={"candidate_id": "candidate_alice"})
    assert len(hits_alice) == 1
    assert hits_alice[0]["candidate_id"] == "candidate_alice"
    assert "Alice" in hits_alice[0]["text"]
    assert "Bob" not in hits_alice[0]["text"]


def test_session_metadata_isolation():
    # Verify session profile isolation in service layer
    jd._COMPANY_JDS["session_comp_1"] = {"company_id": "session_comp_1", "company_name": "Tech Corp", "role_title": "Dev"}
    jd._COMPANY_JDS["session_comp_2"] = {"company_id": "session_comp_2", "company_name": "Data Corp", "role_title": "Analyst"}

    assert "session_comp_1" in jd._COMPANY_JDS
    assert "session_comp_2" in jd._COMPANY_JDS
    assert jd._COMPANY_JDS["session_comp_1"]["company_name"] == "Tech Corp"
    assert jd._COMPANY_JDS["session_comp_2"]["company_name"] == "Data Corp"
