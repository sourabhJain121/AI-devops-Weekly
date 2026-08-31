"""Company-Specific & Role-Tailored Interview Preparation Generator."""
from app import config, llm
from app.services import jd, resume, skill_gap


def generate_interview_prep(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Generate personalized interview questions based on candidate resume, company JD, and skill gaps."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    if not cand or not comp:
        return {
            "questions": [],
            "summary": "Upload both candidate resume and company JD to generate personalized interview preparation.",
        }

    gaps = skill_gap.analyze_skill_gaps(candidate_id=candidate_id, company_id=company_id)
    high_gaps = gaps.get("high_priority_gaps", [])

    prompt = f"""You are an Expert Technical Interviewer preparing a BMU candidate for an upcoming company interview.

CANDIDATE PROFILE:
Name: {cand.get('name')}
Degree: {cand.get('degree')}
Skills: {', '.join(cand.get('skills', []))}

TARGET ROLE & COMPANY:
Company: {comp.get('company_name')}
Role: {comp.get('role_title')}
Required Skills: {', '.join(comp.get('required_skills', []))}
Identified Preparation Gaps: {', '.join(high_gaps)}

Generate 5 personalized, highly realistic interview questions in the following format:
1. 🛠️ **Resume Technical Question** (testing declared skills)
2. 🎯 **Role-Specific Scenario Question** (based on job requirements)
3. ⚠️ **Skill-Gap Probe Question** (addressing preparation gaps)
4. 💻 **System Design / Coding Question**
5. 🤝 **Behavioral / HR Question**

Keep explanations concise and include brief tips on how the candidate should structure their answer."""

    res = llm.generate(prompt, "Generate personalized interview prep questions.")

    summary_text = (
        f"🎯 **Personalized Interview Prep Guide for {comp['company_name']} ({comp['role_title']})**\n\n"
        + res["reply"]
    )

    return {
        "company_name": comp["company_name"],
        "role_title": comp["role_title"],
        "summary": summary_text,
    }
