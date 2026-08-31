"""Unit tests for section-aware chunking boundaries."""
from app import chunking


def test_chunking_heading_boundaries():
    text = (
        "## 1. Eligibility Rules\n\n"
        "A student must have a CGPA of 6.5 or higher to participate.\n"
        "No more than two active backlogs.\n\n"
        "## 2. Document Submission\n\n"
        "Students must submit their updated resume in PDF format.\n"
        "Consolidated marksheet must be attached."
    )

    chunks = chunking.chunk(text, size=800, overlap=120)
    assert len(chunks) == 2
    assert "1. Eligibility Rules" in chunks[0]["section"]
    assert "2. Document Submission" in chunks[1]["section"]
    assert "6.5 or higher" in chunks[0]["text"]
    assert "resume in PDF" in chunks[1]["text"]
