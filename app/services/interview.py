"""Company-Specific & Role-Tailored Interview Preparation Generator.

Enhanced to integrate the Interview Question Bank and verified company
intelligence.  Generic questions from the bank are never labeled as
company-specific unless separately verified.
"""
from app import config, llm
from app.services import jd, resume, skill_gap, question_bank


def generate_interview_prep(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Generate personalized interview questions based on candidate resume, company JD, skill gaps,
    and the interview question bank."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    if not cand or not comp:
        # Fall back to question bank personalized prep if we have at least some context
        if cand or comp:
            return question_bank.personalized_interview_prep(
                candidate_id=candidate_id,
                company_id=company_id,
                top_k=10,
            )
        return {
            "questions": [],
            "summary": "Upload both candidate resume and company JD to generate personalized interview preparation.",
        }

    gaps = skill_gap.analyze_skill_gaps(candidate_id=candidate_id, company_id=company_id)
    high_gaps = gaps.get("high_priority_gaps", [])

    # Retrieve relevant questions from the question bank
    bank_questions = []
    for skill in (high_gaps + comp.get("required_skills", []))[:5]:
        hits = question_bank.retrieve_questions(
            query=f"{skill} interview question",
            top_k=3,
        )
        for h in hits:
            qid = h.get("question_id", h.get("document_id", ""))
            if qid and not any(q.get("question_id") == qid for q in bank_questions):
                bank_questions.append(h)
    bank_questions = bank_questions[:5]

    # Format bank questions for context
    bank_context = ""
    if bank_questions:
        bank_lines = ["\n\nRELEVANT QUESTIONS FROM INTERVIEW QUESTION BANK:"]
        for i, q in enumerate(bank_questions, 1):
            question_text = q.get("question", q.get("text", ""))
            cat = q.get("category", "General")
            diff = q.get("difficulty", "medium")
            topics = q.get("expected_topics", "")
            bank_lines.append(f"{i}. [{cat} | {diff}] {question_text}")
            if topics:
                bank_lines.append(f"   Expected topics: {topics}")
        bank_context = "\n".join(bank_lines)

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
{bank_context}

Generate 5 personalized, highly realistic interview questions in the following format:
1. 🛠️ **Resume Technical Question** (testing declared skills)
2. 🎯 **Role-Specific Scenario Question** (based on job requirements)
3. ⚠️ **Skill-Gap Probe Question** (addressing preparation gaps)
4. 💻 **System Design / Coding Question**
5. 🤝 **Behavioral / HR Question**

IMPORTANT RULES:
- Ground resume questions in ACTUAL candidate skills and projects. Do NOT invent projects.
- If relevant questions from the question bank are provided above, incorporate or reference them.
- Generic questions from the question bank are NOT company-specific questions.
  Do NOT claim that the company asks these specific questions unless you have verified evidence.
- Keep explanations concise and include brief tips on how the candidate should structure their answer."""

    res = llm.generate(prompt, "Generate personalized interview prep questions.")

    # Build summary with question bank attribution
    summary_text = (
        f"🎯 **Personalized Interview Prep Guide for {comp['company_name']} ({comp['role_title']})**\n\n"
        + res["reply"]
    )

    if bank_questions:
        summary_text += (
            "\n\n---\n"
            "📝 **Additional Practice Questions from Interview Question Bank:**\n\n"
        )
        for i, q in enumerate(bank_questions, 1):
            question_text = q.get("question", q.get("text", ""))
            cat = q.get("category", "General")
            diff = q.get("difficulty", "medium")
            diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "⚪")
            topics = q.get("expected_topics", "")
            summary_text += f"**{i}. [{cat}] {diff_icon} {diff.capitalize()}:** {question_text}\n"
            if topics:
                summary_text += f"   *Expected topics: {topics}*\n"

        summary_text += (
            "\n> ℹ️ These are generic interview-preparation questions and "
            "are not claimed to be questions officially asked by "
            f"{comp['company_name']} unless separately verified."
        )

    if high_gaps:
        summary_text += f"\n\n⚠️ **Focus on skill gaps:** {', '.join(high_gaps)}"

    return {
        "company_name": comp["company_name"],
        "role_title": comp["role_title"],
        "summary": summary_text,
        "bank_questions": len(bank_questions),
        "skill_gaps": high_gaps,
    }
