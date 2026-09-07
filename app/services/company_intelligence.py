"""Live Company Intelligence Service: Web Research with Verification Gates.

Pipeline: Entity Resolution → Company Verification → Web Search → Source
Reliability Filtering → Evidence Extraction → Grounded Answer + Citations.

Live web data is NEVER automatically embedded into ChromaDB.  It is used
transiently, cited, and discarded.
"""
import logging
import re
from datetime import datetime, timezone

from app import config, web_search
from app.services import jd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

def resolve_company_name(
    user_query: str,
    jd_data: dict | None = None,
    filename: str | None = None,
) -> dict:
    """Determine the company name from available signals.

    Priority (deterministic):
      1. Explicit company name in user query
      2. Company name from JD content (passed directly or from active JD store)
      3. Filename-derived (LOW confidence fallback)
    """
    # 1. Attempt extraction from user query
    explicit = _extract_company_from_query(user_query)
    if explicit:
        return {"name": explicit, "confidence": "high", "source": "user_query"}

    def _is_placeholder(name: str | None) -> bool:
        if not name:
            return True
        return name.strip().lower() in ("jd", "target company", "company", "unknown", "sample", "sample jd")

    # 2. From explicitly passed JD data
    if jd_data and not _is_placeholder(jd_data.get("company_name")):
        return {"name": jd_data["company_name"], "confidence": "medium", "source": "jd_content"}

    # 3. Filename fallback
    if filename:
        name = _extract_company_from_filename(filename)
        if name and not _is_placeholder(name):
            return {"name": name, "confidence": "low", "source": "filename_fallback"}

    # 4. From active loaded company JDs in memory/vectorstore
    from app.services import jd as jd_svc
    all_jds = jd_svc.get_all_company_jds()
    valid_jds = [j for j in all_jds if not _is_placeholder(j.get("company_name"))]
    if valid_jds:
        return {"name": valid_jds[0]["company_name"], "confidence": "medium", "source": "jd_content"}

    # 5. Contextual explanation if user asked about "this company" but no JD is loaded
    if any(phrase in user_query.lower() for phrase in ["this company", "the company", "target company", "this role"]):
        return {
            "name": None,
            "confidence": "none",
            "source": "missing_jd_context",
            "message": (
                "You asked about 'this company', but no Job Description (JD) is currently "
                "active in your workspace.\n\n"
                "To get company intelligence, you can:\n"
                "1. Ask directly with the company name (e.g., **'Tell me about Google'** or **'Tell me about Microsoft'**).\n"
                "2. Upload a company Job Description on the **Company JDs** tab, then ask **'Tell me about this company'**."
            ),
        }

    return {"name": None, "confidence": "none", "source": "unresolved"}


def _extract_company_from_query(query: str) -> str | None:
    """Extract company name from natural-language queries."""
    q = query.strip()
    _STOP_WORDS = {"this", "that", "the", "a", "an", "my", "our", "their", "its",
                   "company", "firm", "organization", "role", "job", "position", "target"}

    patterns = [
        r"(?:tell me about|research|look up|search for|company info(?:rmation)? (?:for|on|about)?)\s+(.+?)(?:\.|$|\?)",
        r"(?:what (?:is|does|do you know about))\s+(.+?)(?:\s+(?:do|company|hire|interview)|\.|$|\?)",
        r"(?:tell me about)\s+(.+?)$",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip().strip('"\'')
            # Clean prefixes like "the company", "this company", "company", ":"
            name = re.sub(r"^(?:this|that|the|a|an)\b\s*", "", name, flags=re.IGNORECASE)
            name = re.sub(r"^(?:company|firm|org|organization)\b\s*[:\-]?\s*", "", name, flags=re.IGNORECASE)
            name = name.strip().strip('"\'')
            name_words = set(name.lower().split())
            if name and len(name) < 100 and not name_words.issubset(_STOP_WORDS):
                return name
    # If query is short and looks like a company name (no question words)
    if len(q.split()) <= 4 and not any(w in q.lower() for w in
            ["what", "how", "why", "when", "where", "do", "does", "can", "is", "this", "that"]):
        return q
    return None


def _extract_company_from_filename(filename: str) -> str | None:
    """Low-confidence fallback: guess company from filename."""
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[_\-]", " ", name).strip()
    parts = name.split()
    if parts:
        return parts[0]
    return None


# ---------------------------------------------------------------------------
# Company verification gate
# ---------------------------------------------------------------------------

def verify_company(company_name: str) -> dict:
    """Anti-hallucination guardrail: verify company exists in credible public sources.

    Returns:
        {
            "verified": bool,
            "ambiguous": bool,
            "candidates": [str, ...],
            "sources": [SearchResult.to_dict(), ...],
            "verification_query": str,
        }
    """
    log.info("Verifying company: '%s'", company_name)

    # Search for the company
    query = f"{company_name} company official website"
    results = web_search.search(query, max_results=config.WEB_SEARCH_MAX_RESULTS, company_name=company_name)

    if not results:
        log.info("Verification failed for '%s': no search results", company_name)
        return {
            "verified": False,
            "ambiguous": False,
            "candidates": [],
            "sources": [],
            "verification_query": query,
            "reason": "no_search_results",
        }

    # Check for tier-1 or tier-2 sources
    credible = [r for r in results if r.reliability_tier <= 2]

    if not credible:
        log.info("Verification failed for '%s': no credible sources", company_name)
        return {
            "verified": False,
            "ambiguous": False,
            "candidates": [],
            "sources": [r.to_dict(company_name) for r in results],
            "verification_query": query,
            "reason": "no_credible_sources",
        }

    # Check for ambiguity: if credible sources point to very different entities
    unique_domains = set()
    company_names_found = set()
    for r in credible:
        unique_domains.add(r.domain)
        # Extract possible company name variations from titles
        if r.title:
            company_names_found.add(r.title.split(" - ")[0].split(" | ")[0].strip()[:60])

    # Simple ambiguity heuristic: if we see very different titles and no
    # official (tier-1) source, it may be ambiguous
    tier1_sources = [r for r in results if r.reliability_tier == 1]
    name_lower = company_name.lower()
    is_ambiguous = (
        len(company_name) <= 4
        and not tier1_sources
        and len(company_names_found) > 3
        and not any(name_lower in d for d in unique_domains)
    )

    if is_ambiguous:
        log.info("Company '%s' is ambiguous: %d candidates", company_name, len(company_names_found))
        return {
            "verified": False,
            "ambiguous": True,
            "candidates": sorted(list(company_names_found))[:5],
            "sources": [r.to_dict(company_name) for r in credible],
            "verification_query": query,
            "reason": "ambiguous",
        }

    log.info("Company '%s' verified with %d credible sources", company_name, len(credible))
    return {
        "verified": True,
        "ambiguous": False,
        "candidates": [],
        "sources": [r.to_dict(company_name) for r in credible],
        "verification_query": query,
        "reason": "verified",
    }


# ---------------------------------------------------------------------------
# Full company intelligence pipeline
# ---------------------------------------------------------------------------

def get_company_intelligence(
    company_name: str,
    user_query: str = "",
) -> dict:
    """Full pipeline: resolve → verify → search → filter → extract → cite.

    Live web results are NOT stored in ChromaDB.
    """
    log.info("Company intelligence request for: '%s'", company_name)
    started = datetime.now(timezone.utc)

    # Step 1: Verification gate
    verification = verify_company(company_name)

    if not verification["verified"]:
        if verification.get("ambiguous"):
            candidates = verification.get("candidates", [])
            cand_list = "\n".join(f"  - {c}" for c in candidates) if candidates else ""
            message = (
                f"I found multiple companies matching \"{company_name}\".\n"
                f"Please specify the company or provide its official website.\n"
                f"\nPossible matches:\n{cand_list}" if cand_list else
                f"I found multiple companies matching \"{company_name}\".\n"
                f"Please specify the company or provide its official website."
            )
            return {
                "status": "ambiguous",
                "company_name": company_name,
                "reply": message,
                "sources": verification.get("sources", []),
                "verified": False,
                "provenance": "LIVE_WEB",
            }

        reason = verification.get("reason", "unknown")
        if reason == "no_search_results":
            message = (
                f"I couldn't verify a company named \"{company_name}\" from reliable "
                f"public sources. I don't want to provide potentially fabricated "
                f"information.\n\nPlease check the company name or provide an "
                f"official company URL."
            )
        else:
            message = (
                f"I found references to \"{company_name}\", but I couldn't verify "
                f"enough information from reliable public sources to provide a "
                f"grounded company profile."
            )

        return {
            "status": "unverified",
            "company_name": company_name,
            "reply": message,
            "sources": verification.get("sources", []),
            "verified": False,
            "provenance": "LIVE_WEB",
        }

    # Step 2: Gather detailed intelligence from verified sources
    detail_query = f"{company_name} company overview products services headquarters careers hiring"
    detail_results = web_search.search(
        detail_query,
        max_results=config.WEB_SEARCH_MAX_RESULTS,
        company_name=company_name,
    )

    # Step 3: Filter by reliability — prefer tier 1 & 2
    all_sources = verification["sources"]  # already dicts
    for r in detail_results:
        src_dict = r.to_dict(company_name)
        # Deduplicate by URL
        if not any(s["source_url"] == src_dict["source_url"] for s in all_sources):
            all_sources.append(src_dict)

    # Sort: tier 1 first, then tier 2, then tier 3
    all_sources.sort(key=lambda s: s.get("reliability_tier", 3))

    # Step 4: Extract content from top credible sources (tier 1 & 2 only)
    evidence_texts: list[str] = []
    used_sources: list[dict] = []
    for src in all_sources:
        if src.get("reliability_tier", 3) > 2:
            continue
        if len(evidence_texts) >= 3:
            break  # Limit to 3 source extractions
        text = web_search.fetch_page_text(src["source_url"])
        if text and len(text) > 100:
            evidence_texts.append(text[:5000])
            src["content_extracted"] = True
            used_sources.append(src)
        elif src.get("snippet"):
            evidence_texts.append(src["snippet"])
            src["content_extracted"] = False
            used_sources.append(src)

    if not evidence_texts:
        return {
            "status": "insufficient_evidence",
            "company_name": company_name,
            "reply": (
                f"I found references to \"{company_name}\", but I couldn't extract "
                f"enough information from reliable public sources to provide a "
                f"grounded company profile."
            ),
            "sources": all_sources,
            "verified": True,
            "provenance": "LIVE_WEB",
        }

    # Step 5: Build evidence block for citations
    evidence_block = _build_evidence_block(used_sources, evidence_texts, company_name)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    return {
        "status": "verified",
        "company_name": company_name,
        "evidence": evidence_block,
        "evidence_texts": evidence_texts,
        "sources": used_sources,
        "all_sources": all_sources,
        "verified": True,
        "retrieval_time_s": round(elapsed, 2),
        "provenance": "LIVE_WEB",
    }


def _build_evidence_block(sources: list[dict], texts: list[str], company_name: str) -> str:
    """Build a formatted evidence block with citations."""
    blocks = []
    for i, (src, text) in enumerate(zip(sources, texts), 1):
        tier_label = {1: "OFFICIAL", 2: "CREDIBLE SECONDARY", 3: "USER-GENERATED"}.get(
            src.get("reliability_tier", 3), "UNKNOWN"
        )
        blocks.append(
            f"[{i}] [{tier_label}] {src.get('source_title', 'Untitled')} "
            f"({src.get('source_domain', 'unknown')})\n"
            f"{text[:3000]}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM-grounded company answer generation
# ---------------------------------------------------------------------------

def generate_company_answer(
    company_name: str,
    intelligence: dict,
    user_query: str = "",
) -> dict:
    """Generate a grounded answer about a company using LLM + live evidence.

    The LLM is strictly instructed to answer ONLY from the provided evidence.
    """
    from app import llm

    if intelligence.get("status") != "verified":
        return intelligence  # Already contains the reply (abstention / ambiguity)

    evidence = intelligence.get("evidence", "")
    sources = intelligence.get("sources", [])

    system_prompt = f"""You are the BMU Placement Intelligence Assistant providing company research.

Rules you MUST follow:
1. Answer ONLY from the EVIDENCE below. It is the sole source of truth.
2. NEVER invent company information (history, founders, products, revenue, employees,
   offices, hiring process, interview process, technologies, clients, salary) that is
   not in the EVIDENCE.
3. If the EVIDENCE does not contain information about a topic, say so explicitly.
   Do NOT fill gaps with your general knowledge.
4. Cite sources using [n] markers matching the evidence blocks.
5. Clearly label the provenance:
   [LIVE WEB] for information from web sources.
6. Distinguish between:
   - VERIFIED COMPANY FACT (from official sources)
   - PUBLICLY REPORTED INFORMATION (from credible secondary sources)
   - USER-GENERATED INFORMATION (if any tier 3 sources are used)
7. Be concise and use clear, professional language.

EVIDENCE:
{evidence}"""

    question = user_query or f"Provide a comprehensive overview of {company_name}, including what the company does, its key products/services, and any relevant information for a placement candidate."

    try:
        result = llm.generate(system_prompt, question)
        # Build citation references
        citation_refs = []
        for i, src in enumerate(sources, 1):
            tier_label = {1: "Official", 2: "Credible Secondary", 3: "User-Generated"}.get(
                src.get("reliability_tier", 3), "Unknown"
            )
            citation_refs.append(
                f"[{i}] {src.get('source_title', 'Untitled')} — "
                f"{src.get('source_domain', 'unknown')} "
                f"({tier_label}, retrieved {src.get('retrieved_at', 'unknown')})"
            )

        citations_block = "\n".join(citation_refs) if citation_refs else ""
        full_reply = result["reply"]
        if citations_block:
            full_reply += f"\n\n---\n**Sources:**\n{citations_block}"

        return {
            "status": "verified",
            "company_name": company_name,
            "reply": full_reply,
            "raw_reply": result["reply"],
            "sources": sources,
            "all_sources": intelligence.get("all_sources", sources),
            "verified": True,
            "model": result.get("model"),
            "latency_s": result.get("latency_s"),
            "retrieval_time_s": intelligence.get("retrieval_time_s"),
            "provenance": "LIVE_WEB",
            "grounded": True,
        }
    except Exception as exc:
        log.error("LLM generation failed for company '%s': %s", company_name, exc)
        return {
            "status": "error",
            "company_name": company_name,
            "reply": (
                "I couldn't retrieve live company information right now. "
                "Please try again later or provide an official company source."
            ),
            "sources": sources,
            "verified": True,
            "provenance": "LIVE_WEB",
            "grounded": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Convenience: full end-to-end pipeline
# ---------------------------------------------------------------------------

def research_company(
    user_query: str,
    company_id: str | None = None,
    filename: str | None = None,
) -> dict:
    """End-to-end company intelligence: resolve → verify → search → answer."""
    # Try to get JD data if company_id is provided
    jd_data = None
    if company_id:
        jd_data = jd.get_company_jd(company_id)

    # Step 1: Entity resolution
    resolution = resolve_company_name(user_query, jd_data=jd_data, filename=filename)
    company_name = resolution.get("name")

    if not company_name:
        default_reply = (
            "I couldn't determine which company you're asking about. "
            "Please specify the company name clearly (e.g. \"Tell me about Google\")."
        )
        return {
            "status": "unresolved",
            "reply": resolution.get("message", default_reply),
            "sources": [],
            "verified": False,
            "provenance": "LIVE_WEB",
        }

    # Step 2: Get intelligence (verify + search + filter)
    intelligence = get_company_intelligence(company_name, user_query)

    # Step 3: Generate grounded answer
    return generate_company_answer(company_name, intelligence, user_query)
