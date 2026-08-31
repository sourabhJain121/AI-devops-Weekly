"""Resume vs Company JD Matching Service."""
from app import config, llm, vectorstore
from app.services import jd, resume


def compare_resume_to_jd(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Perform a structured, explainable comparison of candidate resume vs company JD requirements."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    if not cand:
        return {
            "match_score": 0,
            "status": "No Resume Uploaded",
            "summary": "Please upload a candidate resume first to compare against the Job Description.",
            "matching_skills": [],
            "missing_skills": [],
            "sources": [],
        }

    if not comp:
        return {
            "match_score": 0,
            "status": "No Company JD Provided",
            "summary": "Please upload a Company Job Description to compare with your resume.",
            "matching_skills": [],
            "missing_skills": [],
            "sources": [],
        }

    cand_skills = set(s.lower() for s in cand.get("skills", []))
    req_skills = comp.get("required_skills", [])

    matched = []
    missing = []

    for skill in req_skills:
        if skill.lower() in cand_skills:
            matched.append(skill)
        else:
            missing.append(skill)

    total_req = len(req_skills) if req_skills else 1
    match_score = round((len(matched) / total_req) * 100, 1)

    # Retrieve vector hits for JD context
    jd_hits = vectorstore.query(
        llm.embed_one("skills requirements responsibilities"),
        top_k=3,
        where={"company_id": comp["company_id"]},
    )

    summary_text = (
        f"**Resume vs JD Match Score: {match_score}%** for **{comp['company_name']}** ({comp['role_title']})\n\n"
        f"✅ **Matching Skills ({len(matched)}):** {', '.join(matched) if matched else 'None'}\n"
        f"⚠️ **Missing / Unmatched Skills ({len(missing)}):** {', '.join(missing) if missing else 'None'}\n"
    )

    return {
        "match_score": match_score,
        "company_name": comp["company_name"],
        "role_title": comp["role_title"],
        "matching_skills": matched,
        "missing_skills": missing,
        "summary": summary_text,
        "sources": jd_hits,
    }
