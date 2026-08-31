"""Evidence-Based Resume Improvement Advisor."""
from app import config, llm
from app.services import jd, resume


def suggest_resume_improvements(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Generate evidence-based resume suggestions without fabricating experience."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    if not cand or not comp:
        return {
            "suggestions": [],
            "summary": "Please upload both candidate resume and company JD to generate targeted resume improvements.",
        }

    cand_skills = set(s.lower() for s in cand.get("skills", []))
    req_skills = comp.get("required_skills", [])
    missing_skills = [s for s in req_skills if s.lower() not in cand_skills]

    prompt = f"""You are a professional Resume Coach for BMU placement candidates.
Analyze the candidate's profile and company requirements below to provide actionable resume improvements.

CRITICAL RULES:
1. NEVER invent or fabricate experience, projects, skills, or employment that the candidate does not have.
2. Recommend emphasizing existing candidate skills ({', '.join(cand.get('skills', []))}).
3. For missing skills ({', '.join(missing_skills)}), explicitly say: "Consider highlighting X *if* you genuinely have experience with it."
4. Suggest bullet point action verbs and structural improvements.

CANDIDATE PROFILE:
Name: {cand.get('name')}
Degree: {cand.get('degree')}
Skills: {', '.join(cand.get('skills', []))}

TARGET COMPANY & ROLE:
Company: {comp.get('company_name')}
Role: {comp.get('role_title')}
Required Skills: {', '.join(req_skills)}

Provide 3-4 concise, bulleted recommendations."""

    res = llm.generate(prompt, "Provide resume improvement recommendations.")
    
    summary_text = f"💡 **Resume Improvement Plan for {comp['company_name']} ({comp['role_title']})**\n\n" + res["reply"]

    return {
        "company_name": comp["company_name"],
        "role_title": comp["role_title"],
        "summary": summary_text,
        "missing_keywords": missing_skills,
    }
