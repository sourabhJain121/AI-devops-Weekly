"""End-to-End Workflow Test: Policy + Resume + JD -> Eligibility -> Grounded RAG."""
from app.services import eligibility, jd, resume
from app import orchestrator


def test_full_e2e_placement_pipeline():
    # 1. Register candidate profile
    cand_id = "e2e_cand_1"
    resume._CANDIDATE_PROFILES[cand_id] = {
        "candidate_id": cand_id,
        "name": "Alex Smith",
        "cgpa": 8.2,
        "backlogs": 0,
        "degree": "B.Tech",
        "skills": ["Python", "FastAPI", "SQL", "Docker"]
    }

    # 2. Register company JD
    comp_id = "e2e_comp_1"
    jd._COMPANY_JDS[comp_id] = {
        "company_id": comp_id,
        "company_name": "CloudTech Systems",
        "role_title": "Backend Software Engineer",
        "required_cgpa": 7.5,
        "max_backlogs": 1,
        "required_skills": ["Python", "FastAPI", "Docker", "Kubernetes"]
    }

    # 3. Deterministic Eligibility Check
    elig_result = eligibility.evaluate_eligibility(candidate_id=cand_id, company_id=comp_id)
    assert elig_result["status"] == "Eligible"
    assert "explanation" in elig_result

    # 4. Orchestrator Query Dispatching
    query_res = orchestrator.route_and_execute(
        message="Am I eligible for CloudTech Systems campus drive?",
        candidate_id=cand_id,
        company_id=comp_id
    )
    assert query_res["intent"] in ["ELIGIBILITY", "BMU_POLICY"]
    assert "reply" in query_res
    assert len(query_res["reply"]) > 10
