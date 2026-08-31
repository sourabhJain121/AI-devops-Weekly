"""Unit tests for deterministic placement eligibility evaluator."""
from app.services import eligibility, resume, jd


def test_deterministic_eligibility_evaluator():
    # Mock candidate profile
    resume._CANDIDATE_PROFILES["test_cand_pass"] = {
        "candidate_id": "test_cand_pass",
        "name": "Jane Doe",
        "cgpa": 8.5,
        "backlogs": 0,
        "degree": "B.Tech",
        "skills": ["Python", "SQL"]
    }
    
    resume._CANDIDATE_PROFILES["test_cand_fail"] = {
        "candidate_id": "test_cand_fail",
        "name": "John Smith",
        "cgpa": 6.0,
        "backlogs": 3,
        "degree": "B.Tech",
        "skills": ["C++"]
    }

    # Test Candidate 1 (Pass)
    eval_pass = eligibility.evaluate_eligibility(candidate_id="test_cand_pass")
    assert eval_pass["status"] == "Eligible"

    # Test Candidate 2 (Fail CGPA & Backlogs)
    eval_fail = eligibility.evaluate_eligibility(candidate_id="test_cand_fail")
    assert eval_fail["status"] == "Not Eligible"
    assert len(eval_fail["reasons"]) >= 1
