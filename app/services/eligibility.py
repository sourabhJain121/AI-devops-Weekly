"""Deterministic Placement Eligibility Engine: BMU Policy + Company JD + Resume."""
from app import config, llm, rag, vectorstore
from app.services import jd, resume


def evaluate_eligibility(candidate_id: str = "default_candidate", company_id: str | None = None) -> dict:
    """Evaluate candidate eligibility deterministically and generate a grounded explanation."""
    cand = resume.get_candidate_profile(candidate_id)
    comp = jd.get_company_jd(company_id) if company_id else None

    # Retrieve BMU Policy Eligibility chunks
    policy_hits = vectorstore.query(
        llm.embed_one("minimum CGPA backlog eligibility rules"),
        top_k=3,
        where={"source_type": "bmu_policy"},
    )

    reasons = []
    status = "Eligible"
    deterministic_checks = {}

    # Check 1: Candidate Profile Presence
    if not cand or cand.get("cgpa") is None:
        status = "Unable to Determine"
        reasons.append("Candidate resume does not state CGPA or is not uploaded.")
        deterministic_checks["candidate_cgpa"] = "Missing"
    else:
        cgpa = cand["cgpa"]
        backlogs = cand.get("backlogs", 0)
        deterministic_checks["candidate_cgpa"] = cgpa
        deterministic_checks["candidate_backlogs"] = backlogs

        # BMU Policy Default Thresholds (Fallback if not found in RAG)
        bmu_min_cgpa = 6.5
        bmu_max_backlogs = 2

        # Check BMU Policy Thresholds
        if cgpa < bmu_min_cgpa:
            status = "Not Eligible"
            reasons.append(f"Candidate CGPA ({cgpa}) is below BMU minimum policy threshold ({bmu_min_cgpa}).")
        if backlogs > bmu_max_backlogs:
            status = "Not Eligible"
            reasons.append(f"Candidate active backlogs ({backlogs}) exceed BMU maximum policy limit ({bmu_max_backlogs}).")

        # Check Company Specific Thresholds (if JD provided)
        if comp:
            req_cgpa = comp.get("required_cgpa")
            max_backlogs = comp.get("max_backlogs")
            deterministic_checks["company_req_cgpa"] = req_cgpa
            deterministic_checks["company_max_backlogs"] = max_backlogs

            if req_cgpa is not None and cgpa < req_cgpa:
                status = "Not Eligible"
                reasons.append(f"Candidate CGPA ({cgpa}) is below company required cut-off ({req_cgpa}).")
            if max_backlogs is not None and backlogs > max_backlogs:
                status = "Not Eligible"
                reasons.append(f"Candidate active backlogs ({backlogs}) exceed company maximum allowed ({max_backlogs}).")

    # Construct Grounded Context
    context_blocks = []
    if policy_hits:
        context_blocks.append("BMU POLICY CONTEXT:\n" + rag.build_context(policy_hits))
    if comp:
        context_blocks.append(f"COMPANY JD ({comp['company_name']} - {comp['role_title']}):\n"
                              f"Required CGPA: {comp.get('required_cgpa')}, Max Backlogs: {comp.get('max_backlogs')}, Required Skills: {', '.join(comp.get('required_skills', []))}")
    if cand:
        context_blocks.append(f"CANDIDATE RESUME:\nName: {cand.get('name')}, CGPA: {cand.get('cgpa')}, Backlogs: {cand.get('backlogs')}, Degree: {cand.get('degree')}, Skills: {', '.join(cand.get('skills', []))}")

    full_context = "\n\n".join(context_blocks)

    prompt = f"""You are the BMU Placement Eligibility Evaluator.
Summarize the placement eligibility status clearly based on the context below.

STATUS DETERMINED BY SYSTEM: {status}
SYSTEM REASONS: {'; '.join(reasons) if reasons else 'All CGPA and backlog criteria satisfied.'}

Rules:
1. Explain the eligibility status concisely in 2-3 sentences.
2. Quote the exact candidate CGPA ({cand.get('cgpa') if cand else 'N/A'}) and backlog status vs the requirements.
3. Keep the system's status ({status}) unchanged.

CONTEXT:
{full_context}"""

    llm_res = llm.generate(prompt, "Provide eligibility breakdown.")
    explanation = llm_res["reply"]

    return {
        "status": status,
        "reasons": reasons,
        "explanation": f"**Eligibility Result: {status}**\n\n{explanation}",
        "deterministic_checks": deterministic_checks,
        "sources": rag._sources(policy_hits),
    }
