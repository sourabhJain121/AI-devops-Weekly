"""Unit tests for Resume vs JD Skill Matching."""
from app.services import matching, resume, jd


def test_resume_jd_matching():
    resume._CANDIDATE_PROFILES["cand_match"] = {
        "candidate_id": "cand_match",
        "skills": ["Python", "FastAPI", "Docker", "Git"]
    }

    jd._COMPANY_JDS["comp_match"] = {
        "company_id": "comp_match",
        "company_name": "Tech Corp",
        "role_title": "Backend Dev",
        "required_skills": ["Python", "FastAPI", "Kubernetes"]
    }

    res = matching.compare_resume_to_jd(candidate_id="cand_match", company_id="comp_match")
    assert res["match_score"] == 66.7
    assert "Python" in res["matching_skills"]
    assert "FastAPI" in res["matching_skills"]
    assert "Kubernetes" in res["missing_skills"]
