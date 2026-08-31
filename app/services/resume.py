"""Candidate Resume Service: Upload, Parsing, Extraction & Vector Indexing."""
import json
import re
from pathlib import Path
from app import chunking, config, documents, llm, vectorstore

# In-memory candidate profile cache (keyed by candidate_id)
_CANDIDATE_PROFILES: dict[str, dict] = {}


def extract_candidate_profile(text: str, document_name: str = "") -> dict:
    """Extract structured candidate entities from raw resume text."""
    profile = {
        "document_name": document_name,
        "name": None,
        "email": None,
        "phone": None,
        "cgpa": None,
        "backlogs": 0,
        "degree": None,
        "branch": None,
        "graduation_year": None,
        "skills": [],
        "projects": [],
        "experience": [],
        "certifications": [],
    }

    # Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        profile["email"] = email_match.group(0)

    # Phone
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    if phone_match:
        profile["phone"] = phone_match.group(0)

    # CGPA Extraction (e.g., "CGPA: 8.5", "8.2 / 10", "8.4 CGPA", "GPA: 7.8")
    cgpa_match = re.search(
        r"(?:CGPA|GPA)\s*(?:[:=]|\bis\b)?\s*([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?",
        text,
        re.IGNORECASE,
    )
    if not cgpa_match:
        cgpa_match = re.search(r"([0-9]\.[0-9]{1,2})\s*(?:\/\s*10)?\s*(?:CGPA|GPA)", text, re.IGNORECASE)
    if cgpa_match:
        try:
            profile["cgpa"] = float(cgpa_match.group(1))
        except ValueError:
            pass

    # Backlogs Extraction (e.g., "0 backlogs", "No active backlogs", "1 active backlog")
    if re.search(r"\b(?:no|zero|0)\s+active\s+backlogs?\b|\bno\s+backlogs?\b", text, re.IGNORECASE):
        profile["backlogs"] = 0
    else:
        backlog_match = re.search(r"(\d+)\s+active\s+backlogs?", text, re.IGNORECASE)
        if backlog_match:
            try:
                profile["backlogs"] = int(backlog_match.group(1))
            except ValueError:
                pass

    # Degree & Branch Regex Patterns
    if re.search(r"\bB\.?Tech\b|\bBachelor of Technology\b", text, re.IGNORECASE):
        profile["degree"] = "B.Tech"
    elif re.search(r"\bM\.?Tech\b|\bMaster of Technology\b", text, re.IGNORECASE):
        profile["degree"] = "M.Tech"
    elif re.search(r"\bBBA\b", text, re.IGNORECASE):
        profile["degree"] = "BBA"
    elif re.search(r"\bMBA\b", text, re.IGNORECASE):
        profile["degree"] = "MBA"

    if re.search(r"\bComputer Science\b|\bCSE\b", text, re.IGNORECASE):
        profile["branch"] = "Computer Science"
    elif re.search(r"\bMechanical\b", text, re.IGNORECASE):
        profile["branch"] = "Mechanical Engineering"
    elif re.search(r"\bElectronics\b|\bECE\b", text, re.IGNORECASE):
        profile["branch"] = "Electronics & Communication"

    # Graduation Year (e.g., "2024", "2025", "2026", "2027")
    year_match = re.search(r"\b(202[3-9])\b", text)
    if year_match:
        profile["graduation_year"] = int(year_match.group(1))

    # Tech Skills Identification (keyword extraction)
    common_skills = [
        "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "HTML", "CSS", "SQL",
        "React", "Node.js", "Express", "FastAPI", "Flask", "Django", "Docker", "Kubernetes",
        "AWS", "Git", "GitHub", "Linux", "REST API", "Machine Learning", "Data Analysis",
        "MongoDB", "PostgreSQL", "MySQL", "Tailwind", "Pandas", "NumPy", "PyTorch", "TensorFlow"
    ]
    found_skills = set()
    for skill in common_skills:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.add(skill)
    profile["skills"] = sorted(list(found_skills))

    # Name fallback (First non-empty heading/line)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        possible_name = lines[0]
        if len(possible_name) < 40 and not re.search(r"@|http|resume|cv", possible_name, re.IGNORECASE):
            profile["name"] = possible_name

    return profile


def process_and_index_resume(path: Path, candidate_id: str = "default_candidate") -> dict:
    """Read candidate resume file, extract profile, and store in vectorstore with metadata."""
    path = Path(path)
    pages = documents.load(path)
    if not pages:
        return {"error": "Could not extract text from resume file"}

    full_text = "\n\n".join(text for _, text in pages)
    profile = extract_candidate_profile(full_text, path.name)
    profile["candidate_id"] = candidate_id

    # Cache structured profile
    _CANDIDATE_PROFILES[candidate_id] = profile

    # Clear previous chunks for candidate
    vectorstore.delete_by_where({"candidate_id": candidate_id})

    # Chunk and store in vectorstore
    doc_id = f"resume_{candidate_id}"
    ids, texts, metadatas = [], [], []

    for page_num, page_text in pages:
        chunks = chunking.chunk(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for idx, item in enumerate(chunks):
            ids.append(f"{doc_id}:p{page_num}:c{idx}")
            texts.append(item["text"])
            metadatas.append({
                "document_id": doc_id,
                "document_name": path.name,
                "source_type": "resume",
                "candidate_id": candidate_id,
                "page": page_num,
                "section": item["section"] or "Resume Content",
                "chunk_index": idx,
            })

    if texts:
        embeddings = llm.embed(texts)
        vectorstore.add(ids, texts, metadatas, embeddings)

    return {
        "status": "success",
        "candidate_id": candidate_id,
        "document_name": path.name,
        "pages": len(pages),
        "chunks": len(ids),
        "profile": profile,
    }


def get_candidate_profile(candidate_id: str = "default_candidate") -> dict | None:
    """Get candidate profile from memory cache or vector store metadata."""
    if candidate_id in _CANDIDATE_PROFILES:
        return _CANDIDATE_PROFILES[candidate_id]
    
    # Try reconstructing from vectorstore
    chunks = vectorstore.get_by_where({"candidate_id": candidate_id})
    if not chunks:
        return None
    
    full_text = "\n\n".join(c["text"] for c in chunks)
    doc_name = chunks[0].get("document_name", "Resume.pdf")
    profile = extract_candidate_profile(full_text, doc_name)
    profile["candidate_id"] = candidate_id
    _CANDIDATE_PROFILES[candidate_id] = profile
    return profile
