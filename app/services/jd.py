"""Company Job Description Service: Upload, Entity Extraction, Data Isolation & Indexing."""
import re
from pathlib import Path
from app import chunking, config, documents, llm, vectorstore

# In-memory store for company JDs (keyed by company_id)
_COMPANY_JDS: dict[str, dict] = {}


def extract_company_entities(text: str, document_name: str = "") -> dict:
    """Extract structured job entities from company JD text."""
    jd_data = {
        "document_name": document_name,
        "company_name": None,
        "role_title": None,
        "required_cgpa": None,
        "max_backlogs": None,
        "eligible_degrees": [],
        "eligible_branches": [],
        "graduation_years": [],
        "required_skills": [],
        "preferred_skills": [],
        "responsibilities": [],
    }

    # Company Name extraction from document name or heading
    name_clean = document_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    jd_data["company_name"] = name_clean.split()[0] if name_clean else "Target Company"

    # Role Title
    if re.search(r"Software\s+(?:Development|Developer|Engineer)|SWE\b", text, re.IGNORECASE):
        jd_data["role_title"] = "Software Engineer"
    elif re.search(r"Data\s+(?:Analyst|Scientist|Engineer)", text, re.IGNORECASE):
        jd_data["role_title"] = "Data Specialist"
    elif re.search(r"Product\s+Manager|PM\b", text, re.IGNORECASE):
        jd_data["role_title"] = "Product Manager"
    elif re.search(r"Consultant|Analyst\b", text, re.IGNORECASE):
        jd_data["role_title"] = "Analyst / Consultant"
    else:
        jd_data["role_title"] = name_clean if name_clean else "Graduate Role"

    # Required CGPA
    cgpa_match = re.search(
        r"(?:CGPA|GPA|cut-off|cutoff)\s*(?:of|>=|:|\bis\b)?\s*([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?",
        text,
        re.IGNORECASE,
    )
    if not cgpa_match:
        cgpa_match = re.search(r"([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?\s*(?:CGPA|GPA|minimum)", text, re.IGNORECASE)
    if cgpa_match:
        try:
            jd_data["required_cgpa"] = float(cgpa_match.group(1))
        except ValueError:
            pass

    # Max Backlogs
    if re.search(r"\bno\s+active\s+backlogs?\b|\b0\s+backlogs?\b|\bzero\s+backlogs?\b", text, re.IGNORECASE):
        jd_data["max_backlogs"] = 0
    else:
        backlog_match = re.search(r"(?:max|maximum|up to)\s*(\d+)\s*active?\s*backlogs?", text, re.IGNORECASE)
        if backlog_match:
            try:
                jd_data["max_backlogs"] = int(backlog_match.group(1))
            except ValueError:
                pass

    # Graduation Year
    years = re.findall(r"\b(202[3-9])\b", text)
    if years:
        jd_data["graduation_years"] = sorted(list(set(int(y) for y in years)))

    # Required Skills Extractor
    common_skills = [
        "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "HTML", "CSS", "SQL",
        "React", "Node.js", "Express", "FastAPI", "Flask", "Django", "Docker", "Kubernetes",
        "AWS", "Git", "GitHub", "Linux", "REST API", "Machine Learning", "Data Analysis",
        "MongoDB", "PostgreSQL", "MySQL", "Tailwind", "System Design", "Algorithms", "Data Structures"
    ]
    found_skills = set()
    for skill in common_skills:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.add(skill)
    jd_data["required_skills"] = sorted(list(found_skills))

    return jd_data


def process_and_index_jd(path: Path, company_id: str | None = None, session_id: str = "default_session") -> dict:
    """Read Company JD file, extract entities, and store in vectorstore with isolated company_id metadata."""
    path = Path(path)
    pages = documents.load(path)
    if not pages:
        return {"error": "Could not extract text from Company JD file"}

    full_text = "\n\n".join(text for _, text in pages)
    entities = extract_company_entities(full_text, path.name)

    if not company_id:
        slug_name = re.sub(r"[^a-zA-Z0-9]", "_", path.name.rsplit(".", 1)[0]).lower()
        company_id = f"company_{slug_name}"

    entities["company_id"] = company_id
    entities["session_id"] = session_id
    _COMPANY_JDS[company_id] = entities

    # Ensure company isolation: delete old chunks for this company_id before inserting
    vectorstore.delete_by_where({"company_id": company_id})

    # Chunk and store in vectorstore
    doc_id = f"jd_{company_id}"
    ids, texts, metadatas = [], [], []

    for page_num, page_text in pages:
        chunks = chunking.chunk(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for idx, item in enumerate(chunks):
            ids.append(f"{doc_id}:p{page_num}:c{idx}")
            texts.append(item["text"])
            metadatas.append({
                "document_id": doc_id,
                "document_name": path.name,
                "source_type": "company_jd",
                "company_id": company_id,
                "session_id": session_id,
                "page": page_num,
                "section": item["section"] or "Job Specification",
                "chunk_index": idx,
            })

    if texts:
        embeddings = llm.embed(texts)
        vectorstore.add(ids, texts, metadatas, embeddings)

    return {
        "status": "success",
        "company_id": company_id,
        "session_id": session_id,
        "document_name": path.name,
        "pages": len(pages),
        "chunks": len(ids),
        "entities": entities,
    }


def get_company_jd(company_id: str) -> dict | None:
    """Retrieve structured company JD from memory cache or vectorstore."""
    if company_id in _COMPANY_JDS:
        return _COMPANY_JDS[company_id]

    chunks = vectorstore.get_by_where({"company_id": company_id})
    if not chunks:
        return None

    full_text = "\n\n".join(c["text"] for c in chunks)
    doc_name = chunks[0].get("document_name", "JD.pdf")
    entities = extract_company_entities(full_text, doc_name)
    entities["company_id"] = company_id
    _COMPANY_JDS[company_id] = entities
    return entities


def get_all_company_jds() -> list[dict]:
    """Retrieve all loaded/cached company JD entities."""
    if _COMPANY_JDS:
        return list(_COMPANY_JDS.values())
    for item in list_company_jds():
        cid = item.get("company_id")
        if cid:
            jd_obj = get_company_jd(cid)
            if jd_obj:
                _COMPANY_JDS[cid] = jd_obj
    return list(_COMPANY_JDS.values())


def list_company_jds() -> list[dict]:
    """List all indexed company JDs."""
    stats = vectorstore.stats()
    jds = []
    for doc in stats.get("documents", []):
        if doc.get("source_type") == "company_jd" and doc.get("company_id"):
            jds.append({
                "company_id": doc.get("company_id"),
                "document_name": doc.get("document_name"),
                "chunks": doc.get("chunks"),
            })
    return jds
