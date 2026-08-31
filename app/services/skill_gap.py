"""Prioritized Skill Gap Analysis Service."""
from app.services import jd, resume


def analyze_skill_gaps(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Analyze missing skills and prioritize preparation gaps for the target role."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    if not cand or not comp:
        return {
            "gaps": [],
            "summary": "Upload both candidate resume and company JD to compute prioritized skill gaps.",
        }

    cand_skills = set(s.lower() for s in cand.get("skills", []))
    req_skills = comp.get("required_skills", [])

    gaps = []
    for skill in req_skills:
        if skill.lower() in cand_skills:
            gaps.append({
                "skill": skill,
                "evidence": "Present in resume skills list",
                "gap_status": "No Gap",
                "priority": "Low",
            })
        else:
            gaps.append({
                "skill": skill,
                "evidence": "Not found in resume",
                "gap_status": "Missing Skill",
                "priority": "High",
            })

    high_gaps = [g["skill"] for g in gaps if g["priority"] == "High"]

    summary_lines = [
        f"**Skill Gap Analysis for {comp['company_name']} ({comp['role_title']})**\n",
        f"🔥 **High Priority Preparation Gaps ({len(high_gaps)}):** {', '.join(high_gaps) if high_gaps else 'None! All key skills present.'}\n",
        "| Skill | Evidence in Resume | Status | Priority |",
        "| --- | --- | --- | --- |",
    ]

    for item in gaps:
        summary_lines.append(f"| **{item['skill']}** | {item['evidence']} | {item['gap_status']} | **{item['priority']}** |")

    return {
        "company_name": comp["company_name"],
        "role_title": comp["role_title"],
        "gaps": gaps,
        "high_priority_gaps": high_gaps,
        "summary": "\n".join(summary_lines),
    }
