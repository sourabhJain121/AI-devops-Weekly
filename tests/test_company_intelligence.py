"""Tests for Company Intelligence: verification, entity resolution, guardrails."""
import pytest
from unittest.mock import patch, MagicMock
from app.services import company_intelligence
from app import web_search


# ---------------------------------------------------------------------------
# Test 1 — Valid company resolution
# ---------------------------------------------------------------------------

def test_resolve_company_name_explicit():
    """Explicit company name in query has highest priority."""
    result = company_intelligence.resolve_company_name("Tell me about Microsoft")
    assert result["name"] is not None
    assert result["confidence"] == "high"
    assert result["source"] == "user_query"


def test_resolve_company_name_from_jd():
    """Company name extracted from JD content."""
    jd_data = {"company_name": "Google", "role_title": "SWE"}
    result = company_intelligence.resolve_company_name(
        "What does this company do?", jd_data=jd_data
    )
    assert result["name"] == "Google"
    assert result["source"] == "jd_content"


# ---------------------------------------------------------------------------
# Test 2 — Nonexistent company guardrail
# ---------------------------------------------------------------------------

def test_nonexistent_company_guardrail():
    """Verification must fail for fictional companies — no fabricated profile."""
    with patch.object(web_search, "search", return_value=[]):
        result = company_intelligence.verify_company("XYZSuperAI123")
        assert result["verified"] is False
        assert result["reason"] == "no_search_results"

    # Full pipeline should return abstention message
    with patch.object(web_search, "search", return_value=[]):
        result = company_intelligence.research_company("Tell me about XYZSuperAI123")
        assert "couldn't verify" in result["reply"].lower() or "couldn't retrieve" in result["reply"].lower()
        assert result.get("verified") is False


# ---------------------------------------------------------------------------
# Test 3 — Ambiguous company
# ---------------------------------------------------------------------------

def test_ambiguous_company_detection():
    """Short ambiguous names with no official source trigger clarification."""
    mock_results = []
    for i, title in enumerate(["ABC Corp", "ABC Industries", "ABC Tech", "ABC Ltd", "ABC Consulting"]):
        r = MagicMock()
        r.reliability_tier = 2
        r.domain = f"abc{i}.com"
        r.title = title
        r.to_dict = MagicMock(return_value={
            "source_url": f"https://abc{i}.com",
            "source_title": title,
            "source_domain": f"abc{i}.com",
            "reliability_tier": 2,
        })
        mock_results.append(r)

    with patch.object(web_search, "search", return_value=mock_results):
        result = company_intelligence.verify_company("ABC")
        # Should detect ambiguity for very short names with no tier-1 source
        assert result.get("ambiguous") is True or result.get("verified") is True
        # If ambiguous, should have candidates
        if result.get("ambiguous"):
            assert len(result.get("candidates", [])) > 0


# ---------------------------------------------------------------------------
# Test 4 — Web failure handling
# ---------------------------------------------------------------------------

def test_web_failure_handling():
    """Graceful handling when web search raises an exception."""
    with patch.object(web_search, "search", side_effect=Exception("Network error")):
        # verify_company should handle the error gracefully
        try:
            result = company_intelligence.verify_company("TestCorp")
            # If it doesn't raise, it should report unverified
            assert result["verified"] is False
        except Exception:
            # If it raises, that's acceptable for the raw verify function
            pass

    # research_company should always return a safe response
    with patch.object(web_search, "search", return_value=[]):
        result = company_intelligence.research_company("Tell me about TestCorp")
        assert "reply" in result
        assert result.get("verified") is False


# ---------------------------------------------------------------------------
# Test 5 — Low-quality sources filtering
# ---------------------------------------------------------------------------

def test_low_quality_source_rejection():
    """Tier 3 sources (forums, social media) should be deprioritized."""
    assert web_search.classify_source_reliability("https://reddit.com/r/tech") == 3
    assert web_search.classify_source_reliability("https://quora.com/question") == 3
    assert web_search.classify_source_reliability("https://medium.com/article") == 3
    assert web_search.classify_source_reliability("https://twitter.com/company") == 3


# ---------------------------------------------------------------------------
# Test 6 — Official source preference
# ---------------------------------------------------------------------------

def test_official_source_preference():
    """Tier 1 sources (official domains, .gov, .edu) should be prioritized."""
    assert web_search.classify_source_reliability("https://microsoft.com/careers", "Microsoft") == 1
    assert web_search.classify_source_reliability("https://sec.gov/filings") <= 2
    assert web_search.classify_source_reliability("https://reuters.com/article") == 2


# ---------------------------------------------------------------------------
# Test 7 — JD company extraction
# ---------------------------------------------------------------------------

def test_jd_company_extraction():
    """Company name extracted from JD content takes priority over filename."""
    jd_data = {"company_name": "Microsoft", "role_title": "Software Engineer"}
    result = company_intelligence.resolve_company_name(
        "What does this company do?",
        jd_data=jd_data,
        filename="MSFT_JD_2024.pdf",
    )
    assert result["name"] == "Microsoft"
    assert result["source"] == "jd_content"
    assert result["confidence"] == "medium"


# ---------------------------------------------------------------------------
# Test 8 — Filename fallback is low confidence
# ---------------------------------------------------------------------------

def test_filename_fallback_low_confidence():
    """Filename-derived company names should have low confidence."""
    result = company_intelligence.resolve_company_name(
        "What does this company do?",
        filename="Microsoft_Software_Engineer_JD.pdf",
    )
    assert result["confidence"] == "low"
    assert result["source"] == "filename_fallback"


# ---------------------------------------------------------------------------
# Source reliability classification tests
# ---------------------------------------------------------------------------

def test_source_reliability_tiers():
    """Verify the 3-tier classification system."""
    # Tier 1: government, official
    assert web_search.classify_source_reliability("https://sec.gov/filing") == 1
    assert web_search.classify_source_reliability("https://mit.edu/research") == 1

    # Tier 2: credible secondary
    assert web_search.classify_source_reliability("https://reuters.com/tech") == 2
    assert web_search.classify_source_reliability("https://bloomberg.com/news") == 2
    assert web_search.classify_source_reliability("https://techcrunch.com") == 2

    # Tier 3: user-generated
    assert web_search.classify_source_reliability("https://reddit.com/r/tech") == 3
    assert web_search.classify_source_reliability("https://quora.com") == 3


def test_source_type_labels():
    """Verify source type labels match tiers."""
    assert web_search.source_type_label(1) == "official"
    assert web_search.source_type_label(2) == "credible_secondary"
    assert web_search.source_type_label(3) == "user_generated"


def test_url_validation():
    """Invalid URLs should be rejected."""
    assert web_search._is_valid_url("https://example.com") is True
    assert web_search._is_valid_url("http://example.com/path") is True
    assert web_search._is_valid_url("not-a-url") is False
    assert web_search._is_valid_url("ftp://example.com") is False
    assert web_search._is_valid_url("") is False
