"""Web search provider abstraction with source reliability classification.

Provider-agnostic: swap the concrete provider via WEB_SEARCH_PROVIDER env var.
DuckDuckGo is the default — no API key needed.  All failures return empty
results; callers never see raised exceptions from the search layer.
"""
import logging
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source reliability classification
# ---------------------------------------------------------------------------

# Tier 1 patterns — official / government / regulatory
_TIER1_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.gov(\.\w+)?$"),       # government
    re.compile(r"\.edu(\.\w+)?$"),        # educational institutions
]

# Tier 2 domains — established news / business publications
_TIER2_DOMAINS: set[str] = {
    "reuters.com", "bloomberg.com", "cnbc.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "wsj.com", "ft.com", "forbes.com", "fortune.com",
    "techcrunch.com", "wired.com", "theverge.com", "arstechnica.com",
    "businessinsider.com", "economist.com", "inc.com", "hbr.org",
    "zdnet.com", "cnet.com", "venturebeat.com", "crunchbase.com",
    "glassdoor.com", "linkedin.com", "indeed.com", "wikipedia.org",
    "sec.gov", "pitchbook.com", "owler.com", "cbinsights.com",
}

# Tier 3 patterns — low-confidence user-generated
_TIER3_DOMAINS: set[str] = {
    "reddit.com", "quora.com", "medium.com", "tumblr.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "pinterest.com", "4chan.org",
}


def classify_source_reliability(url: str, company_name: str = "") -> int:
    """Return reliability tier (1 = highest, 3 = lowest) for a URL."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 3

    # Tier 1: official company domain
    if company_name:
        slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_slug = re.sub(r"[^a-z0-9]", "", domain.split(".")[0])
        if slug and domain_slug and (slug in domain_slug or domain_slug in slug):
            return 1

    for pat in _TIER1_PATTERNS:
        if pat.search(domain):
            return 1

    if domain in _TIER2_DOMAINS:
        return 2

    if domain in _TIER3_DOMAINS:
        return 3

    # Default to tier 2 for unknown domains (not tier 3, to avoid over-filtering)
    return 2


def source_type_label(tier: int) -> str:
    """Human-readable label for a reliability tier."""
    return {1: "official", 2: "credible_secondary", 3: "user_generated"}.get(tier, "unknown")


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def fetch_page_text(url: str, timeout: int | None = None, max_size: int | None = None) -> str:
    """Fetch a URL and return cleaned text content.  Never raises."""
    timeout = timeout or config.WEB_SEARCH_TIMEOUT
    max_size = max_size or config.WEB_CONTENT_MAX_SIZE
    if not _is_valid_url(url):
        return ""
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "BMU-PlacementAssistant/1.0"},
            allow_redirects=True,
        )
        resp.raise_for_status()
        content = resp.text[:max_size]
        soup = BeautifulSoup(content, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Normalise whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_size]
    except Exception as exc:
        log.warning("fetch_page_text failed for %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Search provider interface
# ---------------------------------------------------------------------------

class SearchResult:
    """Single search result."""
    __slots__ = ("title", "url", "snippet", "domain", "reliability_tier",
                 "source_type", "retrieved_at")

    def __init__(self, title: str, url: str, snippet: str, company_name: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        try:
            self.domain = urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            self.domain = ""
        self.reliability_tier = classify_source_reliability(url, company_name)
        self.source_type = source_type_label(self.reliability_tier)
        self.retrieved_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self, company_name: str = "") -> dict:
        return {
            "company_name": company_name,
            "source_url": self.url,
            "source_title": self.title,
            "source_domain": self.domain,
            "retrieved_at": self.retrieved_at,
            "source_type": self.source_type,
            "reliability_tier": self.reliability_tier,
            "snippet": self.snippet,
        }


def search(query: str, max_results: int | None = None, company_name: str = "") -> list[SearchResult]:
    """Run a resilient web search across authoritative public providers.

    Combines:
      1. DuckDuckGo Instant Answer API (official free API)
      2. Wikipedia Summary & Search API (authoritative company encyclopedia)
      3. DuckDuckGo search library (fallback)

    Returns an empty list on any total failure; callers never see raised exceptions.
    """
    max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    def _add_result(r: SearchResult):
        if r.url and r.url not in seen_urls and _is_valid_url(r.url):
            seen_urls.add(r.url)
            results.append(r)

    # Provider 1: Wikipedia company encyclopedia lookup (fast, authoritative, never rate-limited)
    wiki_results = _search_wikipedia(query, max_results, company_name)
    for r in wiki_results:
        _add_result(r)

    # Provider 2: DuckDuckGo Instant Answer API (official free API)
    if len(results) < max_results:
        ddg_api_results = _search_duckduckgo_instant_api(query, max_results - len(results), company_name)
        for r in ddg_api_results:
            _add_result(r)

    # Provider 3: duckduckgo_search package fallback
    if len(results) < max_results:
        ddg_pkg_results = _search_duckduckgo(query, max_results - len(results), company_name)
        for r in ddg_pkg_results:
            _add_result(r)

    log.info("Web search returned %d results for '%s' (company='%s')", len(results), query, company_name)
    return results[:max_results]


def _search_wikipedia(query: str, max_results: int, company_name: str) -> list[SearchResult]:
    """Search Wikipedia REST and Search APIs for verified company facts."""
    results: list[SearchResult] = []
    target = company_name or query
    clean_target = re.sub(
        r"\b(?:company|firm|official|website|overview|products|services|headquarters|careers|hiring)\b",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip()
    if not clean_target:
        clean_target = target.strip()

    headers = {"User-Agent": "BMUPlacementAssistant/1.0 (contact@bmu.edu.in)"}

    # 1. Direct page summary
    try:
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean_target)}"
        resp = requests.get(wiki_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("type") != "disambiguation" and data.get("extract"):
                page_url = data.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{requests.utils.quote(clean_target)}"
                results.append(SearchResult(
                    title=f"{data.get('title', clean_target)} — Overview",
                    url=page_url,
                    snippet=data.get("extract", "")[:1000],
                    company_name=company_name,
                ))
    except Exception as exc:
        log.debug("Wikipedia summary lookup failed for '%s': %s", clean_target, exc)

    # 2. Wikipedia search API
    try:
        srch_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": f"{clean_target} company", "format": "json"}
        resp = requests.get(srch_url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            for item in resp.json().get("query", {}).get("search", [])[:max_results]:
                title = item.get("title", "")
                url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title)}"
                if not any(r.url == url for r in results):
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=clean_snippet,
                        company_name=company_name,
                    ))
    except Exception as exc:
        log.debug("Wikipedia search API failed for '%s': %s", clean_target, exc)

    return results


def _search_duckduckgo_instant_api(query: str, max_results: int, company_name: str) -> list[SearchResult]:
    """DuckDuckGo official Instant Answer API (free, zero API key)."""
    results: list[SearchResult] = []
    target = company_name or query
    clean_target = re.sub(
        r"\b(?:company|firm|official|website|overview|products|services|headquarters|careers|hiring)\b",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip()
    if not clean_target:
        clean_target = target.strip()

    headers = {"User-Agent": "BMUPlacementAssistant/1.0 (contact@bmu.edu.in)"}
    try:
        ddg_api = f"https://api.duckduckgo.com/?q={requests.utils.quote(clean_target)}+company&format=json"
        resp = requests.get(ddg_api, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("AbstractText") and data.get("AbstractURL"):
                results.append(SearchResult(
                    title=data.get("Heading", clean_target),
                    url=data.get("AbstractURL"),
                    snippet=data.get("AbstractText"),
                    company_name=company_name,
                ))
            for topic in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                    results.append(SearchResult(
                        title=topic.get("Text", "")[:80],
                        url=topic.get("FirstURL"),
                        snippet=topic.get("Text", ""),
                        company_name=company_name,
                    ))
    except Exception as exc:
        log.debug("DuckDuckGo Instant Answer API failed for '%s': %s", clean_target, exc)

    return results


def _search_duckduckgo(query: str, max_results: int, company_name: str) -> list[SearchResult]:
    """DuckDuckGo search via the duckduckgo-search package (if available)."""
    try:
        from duckduckgo_search import DDGS
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href", r.get("link", ""))
                if not _is_valid_url(url):
                    continue
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("body", r.get("snippet", "")),
                    company_name=company_name,
                ))
        return results
    except Exception as exc:
        log.debug("DuckDuckGo package search failed for '%s': %s", query, exc)
        return []
