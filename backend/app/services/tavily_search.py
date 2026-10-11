"""Tavily web search for academic paper candidates."""

from __future__ import annotations

import hashlib
import random
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urlparse

from backend.app.config import get_settings
from backend.app.models.schemas import QueryUnderstanding


_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_INVALID_TITLE_RE = re.compile(
    r"(?:\barticle\s+in\s+press\b|^untitled$|^articles?\s+in\s+(?:19|20)\d{2}\b)",
    re.IGNORECASE,
)
_PAPER_LANDING_PATH_RE = re.compile(
    r"(?:"
    r"^/abs/[^/]+|"  # arXiv abstract
    r"^/articles/[^/]+|"  # Nature and similar article pages
    r"/article/(?:pii/)?[^/]+|"  # Springer / ScienceDirect
    r"/doi/(?:abs/|full/|pdf/)?[^/]+|"  # DOI-backed publisher pages
    r"^/\d+/?$|"  # PubMed PMID
    r"/document/\d+|"  # IEEE document
    r"/paper/[^/]+|"  # generic repository paper page
    r"/publication/[^/]+"
    r")",
    re.IGNORECASE,
)


class TavilyRequestError(RuntimeError):
    def __init__(self, error_type: str, *, attempts: int) -> None:
        super().__init__(f"Tavily request failed: {error_type}")
        self.error_type = error_type
        self.attempts = attempts


@lru_cache(maxsize=2)
def _tavily_client(api_key: str):
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def _is_transient_tavily_error(exc: Exception) -> bool:
    name = type(exc).__name__.casefold()
    if any(value in name for value in ("connection", "connect", "timeout", "ssl")):
        return True
    response = getattr(exc, "response", None)
    status_code = int(getattr(response, "status_code", 0) or 0)
    return status_code in {408, 425, 429, 500, 502, 503, 504}


def build_tavily_query(query: str, understanding: QueryUnderstanding) -> str:
    parts: list[str] = []

    # Prefer bilingual rewritten queries.
    rewritten = list(understanding.rewritten_queries or [])
    if rewritten:
        parts.append(" OR ".join(f"({q})" for q in rewritten[:2]))
    elif understanding.keywords:
        parts.append(" ".join(understanding.keywords[:8]))
    else:
        parts.append(query)

    if understanding.time_range:
        parts.append(understanding.time_range)
    if understanding.domain:
        parts.append(understanding.domain)

    lang = (understanding.language or "zh+en").lower()
    if lang in {"zh+en", "both", "bilingual"}:
        parts.append("(中文 OR English OR Chinese)")
    elif lang == "zh":
        parts.append("中文")
    else:
        parts.append("English")

    parts.append("(academic paper OR 学术论文 OR review) (site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov OR site:nature.com OR site:science.org OR filetype:pdf)")
    return " ".join(p for p in parts if p).strip()


def _guess_year(*texts: str) -> int | None:
    for text in texts:
        years = [int(y) for y in _YEAR_RE.findall(text or "")]
        if years:
            years = [y for y in years if 1990 <= y <= 2030]
            if years:
                return max(years)
    return None


def _parse_time_bounds(time_range: str | None) -> tuple[int | None, int | None]:
    if not time_range:
        return None, None
    years = [int(y) for y in re.findall(r"(20\d{2})", time_range)]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], years[0]
    return min(years), max(years)


def _paper_id(url: str, title: str) -> str:
    basis = (url or title or "").encode("utf-8")
    return hashlib.md5(basis).hexdigest()[:16]


def is_concrete_paper_resource(title: str, url: str | None) -> bool:
    """Return whether a search hit points to a concrete paper page or paper PDF."""
    clean_title = " ".join((title or "").split()).strip()
    if not clean_title or _INVALID_TITLE_RE.search(clean_title):
        return False
    if not url:
        return False

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False

    path = unquote(parsed.path).lower().rstrip("/")
    query = parsed.query.lower()
    full_url = unquote(url).lower()

    # Publisher-generated reference files are not paper full texts.
    if "_reference.pdf" in full_url:
        return False

    # Collection, archive and year-index pages are not individual papers.
    if "/list/" in path:
        return False
    if path.endswith("/articles") and ("year=" in query or not query):
        return False

    # A direct PDF is accepted after the reference/list exclusions above.
    if path.endswith(".pdf"):
        return True

    host = parsed.netloc.lower().removeprefix("www.")
    if host == "doi.org" and path not in {"", "/"}:
        return True
    return bool(_PAPER_LANDING_PATH_RE.search(path))


def search_papers_with_tavily(
    query: str,
    understanding: QueryUnderstanding,
    max_results: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY")

    client = _tavily_client(settings.tavily_api_key)
    search_q = build_tavily_query(query, understanding)
    limit = max_results or settings.tavily_max_results

    attempts = 0
    last_error: Exception | None = None
    for attempts in range(1, max(1, settings.academic_request_max_attempts) + 1):
        try:
            raw = client.search(
                query=search_q,
                max_results=limit,
                search_depth=settings.tavily_search_depth,
                include_answer=False,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not _is_transient_tavily_error(exc) or attempts >= settings.academic_request_max_attempts:
                raise TavilyRequestError(type(exc).__name__, attempts=attempts) from exc
            delay = settings.academic_retry_backoff_seconds * (2 ** (attempts - 1))
            time.sleep(delay + random.uniform(0.0, max(0.0, delay * 0.2)))
    else:  # pragma: no cover - loop always breaks or raises
        raise TavilyRequestError(type(last_error).__name__, attempts=attempts)
    results = raw.get("results") or []
    start_year, end_year = _parse_time_bounds(understanding.time_range)

    papers: list[dict[str, Any]] = []
    time_filtered_out = 0
    invalid_resource_filtered_out = 0
    for item in results:
        title = (item.get("title") or "Untitled").strip()
        content = (item.get("content") or item.get("snippet") or "").strip()
        url = item.get("url")
        if not is_concrete_paper_resource(title, url):
            invalid_resource_filtered_out += 1
            continue
        year = _guess_year(title, content, url or "")
        if start_year is not None and year is not None and year < start_year:
            time_filtered_out += 1
            continue
        if end_year is not None and year is not None and year > end_year:
            time_filtered_out += 1
            continue
        host = urlparse(url or "").netloc.replace("www.", "") if url else None
        papers.append(
            {
                "id": _paper_id(url or "", title),
                "title": title,
                "abstract": content[:1200],
                "authors": [],
                "year": year,
                "published_at": str(year) if year else None,
                "venue": host,
                "citation_count": 0,
                "url": url,
                "tags": list(understanding.topics[:3] or understanding.keywords[:3]),
                "source": "tavily",
                "score": float(item.get("score") or 0.0),
            }
        )

    meta = {
        "tavily_query": search_q,
        "tavily_raw_count": len(results),
        "tavily_filtered_out": time_filtered_out + invalid_resource_filtered_out,
        "tavily_time_filtered_out": time_filtered_out,
        "tavily_invalid_resource_filtered_out": invalid_resource_filtered_out,
        "api_calls": attempts,
    }
    return papers, meta
