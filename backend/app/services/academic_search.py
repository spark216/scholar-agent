"""Multi-provider academic paper retrieval and canonical candidate merging."""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable

from backend.app.config import get_settings
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.local_vector_corpus import search_local_vector_corpus
from backend.app.services.academic_http import (
    ProviderRequestError,
    academic_http_request,
    request_attempts,
)
from backend.app.services.local_pasa_corpus import (
    resolve_pasa_title_corpus,
    search_local_pasa_corpus,
)
from backend.app.services.persistent_cache import (
    FrozenProviderCacheMiss,
    cache_payload_sha256,
    load_cached_json,
    save_cached_json,
)


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)([^?#\s]+?)(?:\.pdf)?$", re.I)
_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
_SEMANTIC_SCHOLAR_FIELDS = ",".join(
    (
        "paperId",
        "corpusId",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "externalIds",
        "citationCount",
        "publicationDate",
        "fieldsOfStudy",
    )
)
_SEMANTIC_SCHOLAR_RATE_LOCK = Lock()
_semantic_scholar_next_request_at = 0.0

_ACADEMIC_QUERY_STOPWORDS = {
    "a", "about", "an", "and", "any", "are", "article", "articles", "be", "can",
    "contain", "could", "find", "for", "from", "give", "have", "how", "in", "insight",
    "insights", "into", "is", "list", "me", "mention", "of", "on", "paper", "papers",
    "capability", "process", "provide", "publication", "publications", "research", "share", "show",
    "some", "studies", "study", "that", "the", "their", "there", "these", "to", "using", "what",
    "which", "with", "work", "works", "would", "you",
}

_BIOMEDICAL_CUES = {
    "biomedical", "biology", "cancer", "clinical", "covid", "disease", "drug", "gene",
    "genome", "genomic", "health", "medicine", "medical", "neural disorder", "neuroscience",
    "patient", "pharmacology", "protein", "therapy", "treatment",
}


def _relation_expansion_terms(text: str) -> list[str]:
    searchable = text.casefold()
    terms: list[str] = []
    if any(cue in searchable for cue in ("negative impact", "negatively", "harm", "degrad")):
        terms.extend(["performance degradation", "adverse effect", "reduced performance"])
    if any(cue in searchable for cue in ("cannot surpass", "underperform", "fall short", "compared to")):
        terms.extend(["performance limitation", "underperformance", "comparative evaluation"])
    if any(cue in searchable for cue in ("experiment", "evaluate", "benchmark", "dataset")):
        terms.extend(["benchmark evaluation", "experimental results"])
    if any(cue in searchable for cue in ("advantage", "benefit", "improve")):
        terms.extend(["performance improvement", "benefit", "efficient"])
    return list(dict.fromkeys(terms))


def _structured_relation_phrases(understanding: QueryUnderstanding) -> list[str]:
    raw = understanding.raw_json or {}
    values: list[str] = []
    for relation in raw.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        phrase = " ".join(
            str(relation.get(field) or "").strip()
            for field in ("source", "relation", "target")
        ).strip()
        if phrase:
            values.append(phrase)
    for constraint in raw.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        phrase = " ".join(
            str(constraint.get(field) or "").strip()
            for field in ("facet", "operator", "value")
        ).strip()
        if phrase:
            values.append(phrase)
    return list(dict.fromkeys(values))


def _plain_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _normalize_title(title: str) -> str:
    return "".join(ch.lower() for ch in (title or "") if ch.isalnum())[:160]


def _parse_year(value: Any) -> int | None:
    match = _YEAR_RE.search(str(value or ""))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1900 <= year <= 2100 else None


def _parse_time_bounds(time_range: str | None) -> tuple[int | None, int | None]:
    years = [int(item) for item in _YEAR_RE.findall(time_range or "")]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], years[0]
    return min(years), max(years)


def _preferred_query(query: str, understanding: QueryUnderstanding) -> str:
    rewritten = [item.strip() for item in understanding.rewritten_queries if item.strip()]
    if rewritten:
        # arXiv/PubMed generally retrieve English terminology more reliably.
        rewritten.sort(key=lambda item: sum(ch.isascii() and ch.isalpha() for ch in item), reverse=True)
        return rewritten[0]
    terms = [
        *understanding.topics,
        *understanding.methods,
        *understanding.datasets,
        *understanding.keywords,
    ]
    return " ".join(dict.fromkeys(item.strip() for item in terms if item.strip())) or query


def _multi_query_texts(
    query: str,
    understanding: QueryUnderstanding,
    max_extra: int = 2,
) -> list[str]:
    """Build a bounded query list for multi-angle retrieval (PaSA Crawler
    style): the preferred main query plus up to ``max_extra`` independent
    sub-problems from the intent parser's decomposition."""
    texts = [_preferred_query(query, understanding)]
    for sub_query in (understanding.sub_queries or []):
        clean = " ".join(str(sub_query or "").strip().split())
        if not clean:
            continue
        if clean.casefold() in {text.casefold() for text in texts}:
            continue
        texts.append(clean)
        if len(texts) >= 1 + max_extra:
            break
    return texts


def _stable_id(external_ids: dict[str, str], title: str, year: int | None = None) -> str:
    for name in ("doi", "pmid", "arxiv", "openalex", "semantic_scholar", "corpus_id"):
        value = external_ids.get(name)
        if value:
            normalized = _normalize_external_id(name, value)
            return hashlib.sha256(f"{name}:{normalized}".encode()).hexdigest()[:16]
    fallback = f"title:{_normalize_title(title)}:year:{year or 'unknown'}"
    return hashlib.sha256(fallback.encode()).hexdigest()[:16]


def _normalize_external_id(name: str, value: Any) -> str:
    normalized = str(value or "").strip()
    key = name.casefold()
    if key == "doi":
        match = _DOI_RE.search(normalized)
        return (match.group(1) if match else normalized).rstrip(".,;)").casefold()
    if key == "arxiv":
        normalized = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", normalized, flags=re.I)
        normalized = re.sub(r"^arxiv:\s*", "", normalized, flags=re.I)
        normalized = normalized.removesuffix(".pdf")
        return re.sub(r"v\d+$", "", normalized, flags=re.I).casefold()
    if key in {"openalex", "semantic_scholar", "corpus_id", "pmid"}:
        return normalized.rstrip("/").rsplit("/", 1)[-1].casefold()
    return normalized.casefold()


def _extract_doi(*values: Any) -> str | None:
    for value in values:
        match = _DOI_RE.search(str(value or ""))
        if match:
            return match.group(1).rstrip(".,;)").lower()
    return None


def _candidate(
    *,
    title: str,
    abstract: str,
    authors: list[str],
    year: int | None,
    published_at: str | None,
    venue: str | None,
    citation_count: int,
    url: str | None,
    source: str,
    score: float,
    external_ids: dict[str, str | None],
    tags: list[str] | None = None,
) -> dict[str, Any] | None:
    clean_title = " ".join((title or "").split()).strip()
    if not clean_title:
        return None
    clean_ids = {
        str(key).casefold(): _normalize_external_id(str(key), value)
        for key, value in external_ids.items()
        if value
    }
    return {
        "id": _stable_id(clean_ids, clean_title, year),
        "title": clean_title,
        "abstract": " ".join((abstract or "").split())[:12000],
        "authors": [name for name in authors if name],
        "year": year,
        "published_at": published_at or (str(year) if year else None),
        "venue": venue,
        "citation_count": max(0, int(citation_count or 0)),
        "url": url,
        "tags": list(tags or []),
        "source": source,
        "sources": [source],
        "external_ids": clean_ids,
        "score": float(score or 0.0),
    }


def search_serper_scholar(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    if not settings.serper_api_key:
        raise RuntimeError("未配置 SERPER_API_KEY")

    search_query = _preferred_query(query, understanding)
    page_size = min(max(limit, 1), 100)
    page = (offset // page_size) + 1
    response = academic_http_request(
        "serper-scholar",
        "POST",
        "https://google.serper.dev/scholar",
        headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
        json={"q": search_query, "num": page_size, "page": page},
    )
    try:
        payload = response.json()
        api_calls = request_attempts(response)
    finally:
        response.close()
    raw_results = payload.get("organic") or payload.get("scholar") or []

    papers: list[dict[str, Any]] = []
    for index, item in enumerate(raw_results):
        title = str(item.get("title") or "").strip()
        link = item.get("link")
        publication_info = item.get("publicationInfo") or item.get("publication_info") or ""
        year = _parse_year(item.get("year") or publication_info)
        start_year, end_year = _parse_time_bounds(understanding.time_range)
        if year and ((start_year and year < start_year) or (end_year and year > end_year)):
            continue
        cited_by = item.get("citedBy") or item.get("cited_by") or 0
        citation_match = re.search(r"\d+", str(cited_by))
        authors_raw = item.get("authors") or []
        authors = [
            str(author.get("name") if isinstance(author, dict) else author)
            for author in authors_raw
        ]
        doi = _extract_doi(link, item.get("snippet"))
        arxiv_match = _ARXIV_ID_RE.search(str(link or ""))
        paper = _candidate(
            title=title,
            abstract=str(item.get("snippet") or ""),
            authors=authors,
            year=year,
            published_at=str(year) if year else None,
            venue=str(publication_info)[:500] or None,
            citation_count=int(citation_match.group()) if citation_match else 0,
            url=link,
            source="serper-scholar",
            score=1.0 / (1.0 + offset + index),
            external_ids={
                "doi": doi,
                "arxiv": arxiv_match.group(1).removesuffix(".pdf") if arxiv_match else None,
            },
            tags=understanding.topics[:5],
        )
        if paper:
            papers.append(paper)
    return papers, {
        "provider": "serper-scholar",
        "query": search_query,
        "raw_count": len(raw_results),
        "returned_count": len(papers),
        "api_calls": api_calls,
    }


def _wait_for_semantic_scholar_slot() -> None:
    """Apply the provider's global request cadence without blocking peers."""
    global _semantic_scholar_next_request_at

    interval = max(0.0, get_settings().semantic_scholar_min_interval_seconds)
    if interval <= 0.0:
        return
    with _SEMANTIC_SCHOLAR_RATE_LOCK:
        now = time.monotonic()
        delay = _semantic_scholar_next_request_at - now
        if delay > 0.0:
            time.sleep(delay)
        _semantic_scholar_next_request_at = time.monotonic() + interval


def search_semantic_scholar(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Search Semantic Scholar as a first-round candidate provider."""
    settings = get_settings()
    search_query = _preferred_query(query, understanding)
    page_size = min(max(limit, 1), 100)
    params: dict[str, Any] = {
        "query": search_query,
        "offset": max(0, offset),
        "limit": page_size,
        "fields": _SEMANTIC_SCHOLAR_FIELDS,
    }
    start_year, end_year = _parse_time_bounds(understanding.time_range)
    if start_year or end_year:
        params["year"] = (
            str(start_year)
            if start_year == end_year
            else f"{start_year or ''}-{end_year or ''}"
        )
    headers = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    _wait_for_semantic_scholar_slot()
    response = academic_http_request(
        "semantic-scholar",
        "GET",
        _SEMANTIC_SCHOLAR_SEARCH_URL,
        params=params,
        headers=headers,
    )
    try:
        payload = response.json()
        api_calls = request_attempts(response)
    finally:
        response.close()
    results = payload.get("data") or []

    papers: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        paper = _semantic_scholar_candidate(
            item,
            source="semantic-scholar",
            score=1.0 / (1.0 + max(0, offset) + index),
        )
        if paper:
            papers.append(paper)
    return papers, {
        "provider": "semantic-scholar",
        "query": search_query,
        "raw_count": len(results),
        "returned_count": len(papers),
        "total": int(payload.get("total") or 0),
        "api_calls": api_calls,
    }


def _semantic_scholar_candidate(
    item: dict[str, Any],
    *,
    source: str,
    score: float,
) -> dict[str, Any] | None:
    """Normalize one Graph API paper from search or citation endpoints."""
    external = item.get("externalIds") or {}
    paper_id = str(item.get("paperId") or "").strip() or None
    corpus_id = item.get("corpusId")
    return _candidate(
        title=str(item.get("title") or ""),
        abstract=str(item.get("abstract") or ""),
        authors=[
            str(author.get("name") or "")
            for author in item.get("authors") or []
            if isinstance(author, dict)
        ],
        year=_parse_year(item.get("year") or item.get("publicationDate")),
        published_at=str(item.get("publicationDate") or item.get("year") or "") or None,
        venue=str(item.get("venue") or "") or None,
        citation_count=int(item.get("citationCount") or 0),
        url=(
            item.get("url")
            or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)
        ),
        source=source,
        score=score,
        external_ids={
            "doi": _extract_doi(external.get("DOI"), external.get("doi")),
            "arxiv": external.get("ArXiv") or external.get("ARXIV"),
            "pmid": external.get("PubMed") or external.get("PMID"),
            "semantic_scholar": paper_id,
            "corpus_id": str(corpus_id) if corpus_id is not None else None,
        },
        tags=[str(value) for value in (item.get("fieldsOfStudy") or [])[:5] if value],
    )


def _citation_directions(
    query: str,
    understanding: QueryUnderstanding,
    *,
    both: bool = False,
) -> list[str]:
    """Choose citation directions for a query.

    ``both=True`` expands along citations (newer work) AND references
    (foundational/closely related work) so the agent is not forced to pick
    one based on query wording alone.
    """
    if both:
        return ["citations", "references"]
    searchable = " ".join(
        [query, understanding.time_range or "", *understanding.keywords]
    ).casefold()
    recent_cues = ("recent", "latest", "newest", "state of the art", "最新", "近年", "近五年")
    return ["citations"] if any(cue in searchable for cue in recent_cues) else ["references"]


def _fetch_semantic_scholar_neighbors(
    seed_paper_id: str,
    direction: str,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """One Semantic Scholar /papers/{id}/{direction} call for one seed."""
    settings = get_settings()
    item_key = "citingPaper" if direction == "citations" else "citedPaper"
    source = f"semantic-scholar-{direction}"
    headers = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    _wait_for_semantic_scholar_slot()
    response = academic_http_request(
        source,
        "GET",
        f"{_SEMANTIC_SCHOLAR_PAPER_URL}/{seed_paper_id}/{direction}",
        params={
            "limit": min(max(1, limit), 100),
            "fields": _SEMANTIC_SCHOLAR_FIELDS,
        },
        headers=headers,
    )
    try:
        payload = response.json()
        api_calls = request_attempts(response)
    finally:
        response.close()
    papers: list[dict[str, Any]] = []
    rows = payload.get("data") or []
    for index, row in enumerate(rows):
        item = row.get(item_key) if isinstance(row, dict) else None
        if not isinstance(item, dict):
            continue
        paper = _semantic_scholar_candidate(
            item,
            source=source,
            score=1.0 / (1.0 + index),
        )
        if paper:
            paper["citation_seed_id"] = seed_paper_id
            papers.append(paper)
    return papers, api_calls


def search_semantic_scholar_neighborhood(
    seed_paper_id: str,
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand one high-confidence seed through a bounded citation hop.

    Recent/frontier queries follow citations to newer work; other queries
    follow references to foundational/closely related work. One seed and one
    request are enforced by the controller, so this remains cost- and
    memory-bounded.
    """
    settings = get_settings()
    if not settings.semantic_scholar_citation_expansion_enabled:
        raise RuntimeError("Semantic Scholar citation expansion is disabled")
    if (
        settings.semantic_scholar_citation_expansion_require_api_key
        and not settings.semantic_scholar_api_key
    ):
        raise RuntimeError("Semantic Scholar citation expansion requires an API key")

    direction = _citation_directions(query, understanding)[0]
    papers, api_calls = _fetch_semantic_scholar_neighbors(
        seed_paper_id, direction, limit=limit
    )
    return papers, {
        "provider": f"semantic-scholar-{direction}",
        "direction": direction,
        "seed_paper_id": seed_paper_id,
        "raw_count": len(papers),
        "returned_count": len(papers),
        "api_calls": api_calls,
    }


def search_semantic_scholar_neighborhood_multi(
    seed_paper_ids: list[str],
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    directions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand multiple seeds along citations AND/OR references (PaSa-style).

    Returns merged, deduplicated candidate papers across all (seed, direction)
    pairs, bounded by ``len(seed_paper_ids) * len(directions)`` API calls and
    ``limit`` per call. Every paper records which seed and direction it came
    from via ``citation_seed_id`` / ``citation_direction``.
    """
    settings = get_settings()
    if not settings.semantic_scholar_citation_expansion_enabled:
        raise RuntimeError("Semantic Scholar citation expansion is disabled")
    if (
        settings.semantic_scholar_citation_expansion_require_api_key
        and not settings.semantic_scholar_api_key
    ):
        raise RuntimeError("Semantic Scholar citation expansion requires an API key")

    directions = directions or _citation_directions(
        query, understanding, both=True
    )
    all_papers: list[dict[str, Any]] = []
    api_calls = 0
    seen: set[str] = set()
    per_call: list[dict[str, Any]] = []
    for seed_id in seed_paper_ids:
        for direction in directions:
            try:
                papers, calls = _fetch_semantic_scholar_neighbors(
                    seed_id, direction, limit=limit
                )
                api_calls += calls
                for paper in papers:
                    paper["citation_direction"] = direction
                    identity = str(
                        paper.get("canonical_paper_id")
                        or paper.get("id")
                        or paper.get("title")
                        or ""
                    ).casefold()
                    if identity and identity in seen:
                        continue
                    if identity:
                        seen.add(identity)
                    all_papers.append(paper)
                per_call.append(
                    {
                        "seed": seed_id,
                        "direction": direction,
                        "returned": len(papers),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                per_call.append(
                    {
                        "seed": seed_id,
                        "direction": direction,
                        "error": (
                            exc.error_type
                            if hasattr(exc, "error_type")
                            else type(exc).__name__
                        ),
                    }
                )
    return all_papers, {
        "provider": "semantic-scholar-neighborhood",
        "seeds": list(seed_paper_ids),
        "directions": list(directions),
        "returned_count": len(all_papers),
        "api_calls": api_calls,
        "per_call": per_call,
    }


def _meaningful_query_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", text)
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        clean = token.strip("._-")
        key = clean.casefold()
        if len(clean) <= 1 or key in _ACADEMIC_QUERY_STOPWORDS or key in seen:
            continue
        result.append(clean)
        seen.add(key)
    return result


def build_query_variants(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int = 4,
) -> list[str]:
    """Build bounded, entity-preserving variants for adaptive retrieval.

    The list deliberately stays small: the controller executes at most one
    variant per round, so recall improves without multiplying every provider
    request for every query.
    """
    result: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        clean = " ".join(str(value or "").split()).strip()
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)

    for value in [*understanding.rewritten_queries, *understanding.sub_queries]:
        add(value)

    facets = [
        *understanding.datasets,
        *understanding.entities,
        *understanding.methods,
        *understanding.topics,
        *understanding.keywords,
        *_structured_relation_phrases(understanding),
    ]
    exact_facets = [" ".join(str(value).split()) for value in facets if str(value).strip()]
    if exact_facets:
        # Preserve dataset names/acronyms verbatim.  These are often the most
        # discriminative terms in benchmark and real research queries.
        add(" ".join(dict.fromkeys(exact_facets[:6])))

    terms = _meaningful_query_terms(" ".join([query, *exact_facets]))
    ranked = sorted(terms, key=_term_priority, reverse=True)
    if ranked:
        technical = [term for term in ranked if _term_priority(term)[0]]
        anchors = list(dict.fromkeys([*technical[:4], *ranked[:6]]))[:8]
        add(" ".join(anchors))
    expansions = _relation_expansion_terms(" ".join([query, *exact_facets]))
    if expansions:
        add(" ".join([*exact_facets[:3], *expansions]))
    add(" ".join(terms[:8]))
    add(query)
    return result[: max(1, limit)]


def _term_priority(term: str) -> tuple[int, int]:
    technical = int(
        any(char.isdigit() for char in term)
        or any(char in term for char in "+._-")
        or (term.isupper() and len(term) >= 2)
    )
    return technical, len(term)


def _arxiv_atom(value: str) -> str:
    escaped = " ".join(str(value).replace('"', " ").split())
    return f'all:"{escaped}"'


def _compatible_arxiv_facet(query: str, facet: str) -> bool:
    """Avoid injecting a pure-CJK LLM facet into an English arXiv query.

    Mixed facets such as ``大语言模型 LLM`` are retained because their Latin
    acronym remains searchable in English metadata.
    """
    query_has_latin = bool(re.search(r"[A-Za-z][A-Za-z0-9+._-]*", query))
    facet_has_latin = bool(re.search(r"[A-Za-z][A-Za-z0-9+._-]*", facet))
    return not query_has_latin or facet_has_latin


def _arxiv_query(text: str, understanding: QueryUnderstanding | None = None) -> str:
    """Compile one recall-oriented arXiv DSL with bounded OR branches.

    The previous query turned the first eight words into mandatory ``AND``
    terms. Natural-language question words then suppressed otherwise relevant
    papers. This compiler keeps one precise branch and several broader entity /
    method branches inside a single API request, preserving latency and budget.
    """
    terms = _meaningful_query_terms(text)
    if not terms:
        return _arxiv_atom(text)

    ranked_terms = sorted(
        enumerate(terms),
        key=lambda value: (*_term_priority(value[1]), -value[0]),
        reverse=True,
    )
    selected_indexes = sorted(index for index, _term in ranked_terms[:6])
    selected = [terms[index] for index in selected_indexes]

    branches: list[str] = []
    if understanding is not None:
        facet_phrases = [
            *understanding.entities,
            *understanding.methods,
            *understanding.datasets,
            *understanding.topics,
        ]
        for phrase in facet_phrases[:4]:
            clean = " ".join(str(phrase).split()).strip()
            if (
                clean
                and _compatible_arxiv_facet(text, clean)
                and len(_meaningful_query_terms(clean)) <= 6
            ):
                branches.append(_arxiv_atom(clean))

    # A compact phrase branch rewards papers using the same formulation, while
    # the remaining branches prevent the phrase from becoming a hard filter.
    if 2 <= len(selected) <= 6:
        branches.append(_arxiv_atom(" ".join(selected)))
    if len(selected) >= 2:
        strongest = sorted(selected, key=_term_priority, reverse=True)[:3]
        branches.append("(" + " AND ".join(_arxiv_atom(term) for term in strongest[:2]) + ")")
    branches.append("(" + " OR ".join(_arxiv_atom(term) for term in selected[:5]) + ")")
    return "(" + " OR ".join(dict.fromkeys(branches)) + ")"


def _use_pubmed(query: str, understanding: QueryUnderstanding) -> bool:
    searchable = " ".join(
        [
            query,
            understanding.domain or "",
            *understanding.topics,
            *understanding.entities,
            *understanding.methods,
            *understanding.keywords,
        ]
    ).casefold()
    return any(
        re.search(rf"\b{re.escape(cue)}\b", searchable)
        for cue in _BIOMEDICAL_CUES
    )


def search_arxiv(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_year, end_year = _parse_time_bounds(understanding.time_range)
    settings = get_settings()
    # Multi-angle retrieval (PaSA Crawler style): the preferred query plus
    # independent sub-problems, each issued as its own arXiv API request and
    # merged with the main query ranked first. Disabled by default: measured
    # neutral on pasa-real (no new gold, +API cost).
    if settings.academic_arxiv_multi_query:
        texts = _multi_query_texts(query, understanding)
    else:
        texts = [_preferred_query(query, understanding)]
    all_papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_query: list[dict[str, Any]] = []
    total_api_calls = 0
    for text_index, search_text in enumerate(texts):
        query_dsl = _arxiv_query(search_text, understanding)
        if start_year or end_year:
            first = start_year or 1900
            last = end_year or 2100
            query_dsl += (
                f" AND submittedDate:[{first:04d}01010000 TO {last:04d}12312359]"
            )
        query_limit = limit if text_index == 0 else max(10, limit // 2)
        response = academic_http_request(
            "arxiv",
            "GET",
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query_dsl,
                "start": offset if text_index == 0 else 0,
                "max_results": query_limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers={"User-Agent": f"{settings.ncbi_tool}/1.0"},
        )
        try:
            content = response.content
            api_calls = request_attempts(response)
        finally:
            response.close()
        total_api_calls += api_calls
        root = ET.fromstring(content)
        entries = root.findall("atom:entry", _ATOM)
        per_query.append({"query": search_text, "raw_count": len(entries)})

        for index, entry in enumerate(entries):
            entry_id = _plain_text(entry.find("atom:id", _ATOM))
            arxiv_id = entry_id.rsplit("/", 1)[-1]
            links = {
                link.attrib.get("title") or link.attrib.get("rel") or "": link.attrib.get("href")
                for link in entry.findall("atom:link", _ATOM)
            }
            doi = _plain_text(entry.find("arxiv:doi", _ATOM)) or None
            published_at = _plain_text(entry.find("atom:published", _ATOM)) or None
            year = _parse_year(published_at)
            categories = [
                item.attrib.get("term", "")
                for item in entry.findall("atom:category", _ATOM)
                if item.attrib.get("term")
            ]
            paper = _candidate(
                title=_plain_text(entry.find("atom:title", _ATOM)),
                abstract=_plain_text(entry.find("atom:summary", _ATOM)),
                authors=[
                    _plain_text(author.find("atom:name", _ATOM))
                    for author in entry.findall("atom:author", _ATOM)
                ],
                year=year,
                published_at=published_at,
                venue=_plain_text(entry.find("arxiv:journal_ref", _ATOM)) or "arXiv",
                citation_count=0,
                url=links.get("alternate") or entry_id,
                source="arxiv",
                # Main query ranks first; later angles get a small penalty so
                # duplicates keep the main-query ordering.
                score=(1.0 / (1.0 + (offset if text_index == 0 else 0) + index))
                * (1.0 - 0.05 * text_index),
                external_ids={"doi": doi, "arxiv": arxiv_id},
                tags=categories,
            )
            if paper:
                key = str(arxiv_id) or str(paper.get("title"))
                if key not in seen:
                    seen.add(key)
                    all_papers.append(paper)

    all_papers.sort(key=lambda paper: -float(paper.get("score") or 0.0))
    all_papers = all_papers[:limit]
    return all_papers, {
        "provider": "arxiv",
        "query": texts[0],
        "queries": [item["query"] for item in per_query],
        "raw_count": sum(item["raw_count"] for item in per_query),
        "returned_count": len(all_papers),
        "api_calls": total_api_calls,
    }


def _openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    positions: list[tuple[int, str]] = []
    for word, indexes in (inverted_index or {}).items():
        positions.extend((int(index), word) for index in indexes)
    return " ".join(word for _, word in sorted(positions))


def search_openalex(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    if not settings.openalex_api_key:
        raise RuntimeError("未配置 OPENALEX_API_KEY")

    search_query = _preferred_query(query, understanding)
    page_size = min(max(limit, 1), 100)
    page = (offset // page_size) + 1
    params: dict[str, Any] = {
        "search": search_query,
        "per_page": page_size,
        "page": page,
        "api_key": settings.openalex_api_key,
    }
    start_year, end_year = _parse_time_bounds(understanding.time_range)
    filters = []
    if start_year:
        filters.append(f"from_publication_date:{start_year:04d}-01-01")
    if end_year:
        filters.append(f"to_publication_date:{end_year:04d}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    response = academic_http_request(
        "openalex",
        "GET",
        "https://api.openalex.org/works",
        params=params,
    )
    try:
        payload = response.json()
        api_calls = request_attempts(response)
    finally:
        response.close()
    results = payload.get("results") or []

    papers: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        ids = item.get("ids") or {}
        primary_location = item.get("primary_location") or {}
        best_oa = item.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        doi = _extract_doi(item.get("doi"), ids.get("doi"))
        pmid = str(ids.get("pmid") or "").rsplit("/", 1)[-1] or None
        arxiv_id = str(ids.get("arxiv") or "").rsplit("/", 1)[-1] or None
        openalex_id = str(item.get("id") or "").rsplit("/", 1)[-1] or None
        paper = _candidate(
            title=str(item.get("display_name") or item.get("title") or ""),
            abstract=_openalex_abstract(item.get("abstract_inverted_index")),
            authors=[
                str((authorship.get("author") or {}).get("display_name") or "")
                for authorship in item.get("authorships") or []
            ],
            year=item.get("publication_year"),
            published_at=item.get("publication_date"),
            venue=source.get("display_name"),
            citation_count=int(item.get("cited_by_count") or 0),
            url=(
                best_oa.get("landing_page_url")
                or primary_location.get("landing_page_url")
                or item.get("doi")
                or item.get("id")
            ),
            source="openalex",
            score=float(item.get("relevance_score") or (1.0 / (1.0 + offset + index))),
            external_ids={
                "doi": doi,
                "pmid": pmid,
                "arxiv": arxiv_id,
                "openalex": openalex_id,
            },
            tags=[
                str(topic.get("display_name"))
                for topic in (item.get("topics") or [])[:5]
                if topic.get("display_name")
            ],
        )
        if paper:
            papers.append(paper)
    return papers, {
        "provider": "openalex",
        "query": search_query,
        "raw_count": len(results),
        "returned_count": len(papers),
        "api_calls": api_calls,
    }


def _resolve_openalex_work_ids(
    external_ids_list: list[dict[str, Any]],
) -> dict[str, str]:
    """Map candidate external_ids to OpenAlex work IDs.

    Candidates may already carry an ``openalex`` id; otherwise a DOI is used to
    look the work up on OpenAlex (anonymous, with mailto for a higher pool).
    One request per unresolved DOI, bounded by the caller's budget.
    """
    settings = get_settings()
    resolved: dict[str, str] = {}
    mailto = settings.ncbi_email or "scholar-agent@example.com"
    headers = {}
    params = {"mailto": mailto, "select": "id"}
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    for external_ids in external_ids_list:
        ext = external_ids or {}
        openalex_id = str(ext.get("openalex") or "").strip()
        if openalex_id:
            resolved[str(ext.get("doi") or ext.get("arxiv") or "") or ""] = openalex_id
            resolved["openalex:" + openalex_id] = openalex_id
            continue
        # arXiv papers carry the official DOI 10.48550/arXiv.<id>, which OpenAlex
        # resolves reliably. First-round candidates are mostly arXiv rows, so
        # accepting arXiv IDs here makes seed selection stable.
        arxiv_id = str(ext.get("arxiv") or "").strip().removesuffix(".pdf")
        doi = str(ext.get("doi") or "").strip()
        candidates = []
        if doi:
            candidates.append(doi)
        if arxiv_id:
            candidates.append(f"10.48550/arXiv.{arxiv_id}")
        for lookup in candidates:
            try:
                response = academic_http_request(
                    "openalex",
                    "GET",
                    f"https://api.openalex.org/works/https://doi.org/{lookup}",
                    params=params,
                    headers=headers,
                )
                try:
                    payload = response.json()
                finally:
                    response.close()
                work_id = str(payload.get("id") or "").rsplit("/", 1)[-1]
                if work_id:
                    resolved[lookup] = work_id
                    resolved["openalex:" + work_id] = work_id
                    break
            except Exception:  # noqa: BLE001 - a missing DOI must not abort expansion
                continue
    return resolved


def _fetch_openalex_neighbors(
    work_id: str,
    direction: str,
    *,
    limit: int,
    mailto: str,
) -> tuple[list[dict[str, Any]], int]:
    """One OpenAlex works?filter=cites:|referenced_works: call for one work."""
    settings = get_settings()
    params: dict[str, Any] = {
        "filter": f"{direction}:{work_id}",
        "per_page": min(max(1, limit), 100),
        "select": (
            "id,display_name,abstract_inverted_index,publication_year,"
            "publication_date,cited_by_count,doi,ids,primary_location"
        ),
        "mailto": mailto,
    }
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    response = academic_http_request(
        "openalex",
        "GET",
        "https://api.openalex.org/works",
        params=params,
    )
    try:
        payload = response.json()
        api_calls = request_attempts(response)
    finally:
        response.close()
    source = f"openalex-{direction}"
    papers: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("results") or []):
        ids = item.get("ids") or {}
        primary_location = item.get("primary_location") or {}
        best_oa = item.get("best_oa_location") or {}
        src = primary_location.get("source") or {}
        paper = _candidate(
            title=str(item.get("display_name") or ""),
            abstract=_openalex_abstract(item.get("abstract_inverted_index")),
            authors=[
                str((authorship.get("author") or {}).get("display_name") or "")
                for authorship in item.get("authorships") or []
            ],
            year=item.get("publication_year"),
            published_at=item.get("publication_date"),
            venue=src.get("display_name"),
            citation_count=int(item.get("cited_by_count") or 0),
            url=best_oa.get("landing_page_url") or item.get("doi") or item.get("id"),
            source=source,
            score=1.0 / (1.0 + index),
            external_ids={
                "doi": _extract_doi(item.get("doi"), ids.get("doi")),
                "arxiv": str(ids.get("arxiv") or "").rsplit("/", 1)[-1] or None,
                "openalex": str(item.get("id") or "").rsplit("/", 1)[-1] or None,
            },
        )
        if paper:
            paper["citation_seed_id"] = work_id
            paper["citation_direction"] = direction
            papers.append(paper)
    return papers, api_calls


def search_openalex_neighborhood_multi(
    external_ids_list: list[dict[str, Any]],
    *,
    limit: int,
    directions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """PaSa-style citation expansion via OpenAlex (anonymous-compatible).

    Resolves each seed's DOI/OpenAlex ID to a work ID, then expands along
    ``cites`` (newer work) and ``referenced_works`` (foundational work).
    Falls back to the caller-provided OpenAlex ID when resolution is not
    needed. Bounded by len(seeds) * len(directions) API calls.
    """
    settings = get_settings()
    directions = directions or ["cites", "referenced_works"]
    mailto = settings.ncbi_email or "scholar-agent@example.com"
    all_papers: list[dict[str, Any]] = []
    api_calls = 0
    seen: set[str] = set()
    per_call: list[dict[str, Any]] = []
    for ext in external_ids_list:
        work_ids = [
            value
            for key, value in _resolve_openalex_work_ids([ext]).items()
            if key.startswith("openalex:")
        ]
        if not work_ids:
            per_call.append({"seed": ext, "error": "no_openalex_id_or_doi"})
            continue
        work_id = work_ids[0]
        for direction in directions:
            try:
                papers, calls = _fetch_openalex_neighbors(
                    work_id, direction, limit=limit, mailto=mailto
                )
                api_calls += calls
                for paper in papers:
                    identity = str(
                        paper.get("canonical_paper_id")
                        or paper.get("id")
                        or paper.get("title")
                        or ""
                    ).casefold()
                    if identity and identity in seen:
                        continue
                    if identity:
                        seen.add(identity)
                    all_papers.append(paper)
                per_call.append(
                    {"seed": work_id, "direction": direction, "returned": len(papers)}
                )
            except Exception as exc:  # noqa: BLE001
                per_call.append(
                    {
                        "seed": work_id,
                        "direction": direction,
                        "error": (
                            exc.error_type
                            if hasattr(exc, "error_type")
                            else type(exc).__name__
                        ),
                    }
                )
    return all_papers, {
        "provider": "openalex-neighborhood",
        "directions": list(directions),
        "returned_count": len(all_papers),
        "api_calls": api_calls,
        "per_call": per_call,
    }


def _pubmed_query(text: str, understanding: QueryUnderstanding) -> str:
    start_year, end_year = _parse_time_bounds(understanding.time_range)
    query = text
    if start_year or end_year:
        first = start_year or 1900
        last = end_year or 2100
        query += f' AND ("{first}/01/01"[Date - Publication] : "{last}/12/31"[Date - Publication])'
    return query


def search_pubmed(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    search_query = _pubmed_query(_preferred_query(query, understanding), understanding)
    common_params = {
        "api_key": settings.ncbi_api_key or None,
        "tool": settings.ncbi_tool,
        "email": settings.ncbi_email or None,
    }
    common_params = {key: value for key, value in common_params.items() if value}

    search_response = academic_http_request(
        "pubmed",
        "GET",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            **common_params,
            "db": "pubmed",
            "term": search_query,
            "retmode": "json",
            "retstart": offset,
            "retmax": limit,
            "sort": "relevance",
        },
    )
    try:
        ids = search_response.json().get("esearchresult", {}).get("idlist", [])
        search_api_calls = request_attempts(search_response)
    finally:
        search_response.close()
    if not ids:
        return [], {
            "provider": "pubmed",
            "query": search_query,
            "raw_count": 0,
            "returned_count": 0,
            "api_calls": search_api_calls,
        }

    fetch_response = academic_http_request(
        "pubmed",
        "GET",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            **common_params,
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        },
    )
    try:
        fetch_content = fetch_response.content
        fetch_api_calls = request_attempts(fetch_response)
    finally:
        fetch_response.close()
    root = ET.fromstring(fetch_content)

    papers: list[dict[str, Any]] = []
    for index, article in enumerate(root.findall(".//PubmedArticle")):
        citation = article.find("MedlineCitation")
        article_node = citation.find("Article") if citation is not None else None
        if citation is None or article_node is None:
            continue
        pmid = _plain_text(citation.find("PMID"))
        article_ids = {
            item.attrib.get("IdType", ""): _plain_text(item)
            for item in article.findall("./PubmedData/ArticleIdList/ArticleId")
        }
        doi = article_ids.get("doi") or next(
            (
                _plain_text(item)
                for item in article_node.findall("ELocationID")
                if item.attrib.get("EIdType") == "doi"
            ),
            None,
        )
        abstract_parts = []
        for part in article_node.findall("./Abstract/AbstractText"):
            text = _plain_text(part)
            label = part.attrib.get("Label")
            abstract_parts.append(f"{label}: {text}" if label and text else text)
        authors = []
        for author in article_node.findall("./AuthorList/Author"):
            collective = _plain_text(author.find("CollectiveName"))
            personal = " ".join(
                item for item in (
                    _plain_text(author.find("ForeName")),
                    _plain_text(author.find("LastName")),
                )
                if item
            )
            authors.append(collective or personal)
        date_node = article_node.find("ArticleDate")
        pub_date = article_node.find("./Journal/JournalIssue/PubDate")
        date_source = date_node if date_node is not None else pub_date
        published_at = "-".join(
            item
            for item in (
                _plain_text(date_source.find("Year")) if date_source is not None else "",
                _plain_text(date_source.find("Month")) if date_source is not None else "",
                _plain_text(date_source.find("Day")) if date_source is not None else "",
            )
            if item
        )
        if not published_at and pub_date is not None:
            published_at = _plain_text(pub_date.find("MedlineDate"))
        paper = _candidate(
            title=_plain_text(article_node.find("ArticleTitle")),
            abstract=" ".join(abstract_parts),
            authors=authors,
            year=_parse_year(published_at),
            published_at=published_at or None,
            venue=_plain_text(article_node.find("./Journal/Title")) or "PubMed",
            citation_count=0,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
            score=1.0 / (1.0 + offset + index),
            external_ids={"doi": doi, "pmid": pmid},
            tags=[
                _plain_text(heading.find("DescriptorName"))
                for heading in citation.findall("./MeshHeadingList/MeshHeading")[:5]
            ],
        )
        if paper:
            papers.append(paper)
    return papers, {
        "provider": "pubmed",
        "query": search_query,
        "raw_count": len(ids),
        "returned_count": len(papers),
        "api_calls": search_api_calls + fetch_api_calls,
    }


def _identity_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    # Provider rows normally expose ``id`` while Candidate Registry snapshots
    # may already expose ``canonical_paper_id``. Treat both as identity keys so
    # equal canonical IDs can never survive as two rows and abort EvidRank.
    for field in ("canonical_paper_id", "id"):
        value = str(item.get(field) or "").strip().casefold()
        if value:
            keys.append(f"id:{value}")
    for name, value in (item.get("external_ids") or {}).items():
        if value:
            normalized_name = str(name).casefold()
            keys.append(f"{normalized_name}:{_normalize_external_id(normalized_name, value)}")
    # Some providers supply arbitrary row IDs while others use our stable ID.
    # Index the strong-ID-derived value as well so a later DOI/arXiv bridge
    # joins both components before the merged row receives its final stable ID.
    external_ids = {
        str(name).casefold(): _normalize_external_id(str(name), value)
        for name, value in (item.get("external_ids") or {}).items()
        if value
    }
    if external_ids:
        keys.append(
            f"id:{_stable_id(external_ids, str(item.get('title') or ''), item.get('year'))}"
        )
    return list(dict.fromkeys(keys))


def _weak_title_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_title = _normalize_title(str(left.get("title") or ""))
    right_title = _normalize_title(str(right.get("title") or ""))
    if not left_title or left_title != right_title:
        return False
    left_year = left.get("year")
    right_year = right.get("year")
    if left_year and right_year and int(left_year) != int(right_year):
        return False
    left_authors = {str(value).casefold().strip() for value in left.get("authors") or [] if value}
    right_authors = {str(value).casefold().strip() for value in right.get("authors") or [] if value}
    return not left_authors or not right_authors or bool(left_authors & right_authors)


def _merge_candidate_rows(item: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge one connected identity row into ``item`` in place."""
    if len(str(incoming.get("abstract") or "")) > len(str(item.get("abstract") or "")):
        item["abstract"] = incoming.get("abstract")
    if len(incoming.get("authors") or []) > len(item.get("authors") or []):
        item["authors"] = incoming.get("authors")
    item["citation_count"] = max(
        int(item.get("citation_count") or 0),
        int(incoming.get("citation_count") or 0),
    )
    for field in ("year", "published_at", "venue", "url"):
        if not item.get(field) and incoming.get(field):
            item[field] = incoming[field]
    item["tags"] = list(dict.fromkeys([*(item.get("tags") or []), *(incoming.get("tags") or [])]))
    item["external_ids"] = {
        **(item.get("external_ids") or {}),
        **(incoming.get("external_ids") or {}),
    }
    item["sources"] = list(
        dict.fromkeys(
            [
                *(item.get("sources") or [item.get("source")]),
                *(incoming.get("sources") or [incoming.get("source")]),
            ]
        )
    )
    item["sources"] = [value for value in item["sources"] if value]
    item["source_queries"] = list(
        dict.fromkeys(
            [
                *(item.get("source_queries") or []),
                *(incoming.get("source_queries") or []),
            ]
        )
    )
    item["source"] = "+".join(item["sources"])
    item["score"] = max(float(item.get("score") or 0.0), float(incoming.get("score") or 0.0))
    item["provider_scores"] = {
        **(item.get("provider_scores") or {}),
        **(incoming.get("provider_scores") or {}),
    }
    merged_ranks = {
        str(name): int(rank)
        for name, rank in (item.get("provider_ranks") or {}).items()
    }
    for name, rank in (incoming.get("provider_ranks") or {}).items():
        key = str(name)
        merged_ranks[key] = min(merged_ranks.get(key, int(rank)), int(rank))
    item["provider_ranks"] = merged_ranks


def _canonicalize_merged_id(item: dict[str, Any]) -> None:
    item["external_ids"] = {
        str(name).casefold(): _normalize_external_id(str(name), value)
        for name, value in (item.get("external_ids") or {}).items()
        if value
    }
    item["id"] = _stable_id(
        item.get("external_ids") or {},
        str(item.get("title") or ""),
        item.get("year"),
    )
    if "canonical_paper_id" in item:
        item["canonical_paper_id"] = item["id"]


def merge_academic_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the transitive closure of all canonical identity matches.

    A paper can be connected indirectly, for example title A ↔ DOI bridge ↔
    provider ID B. The old single-pass index merged only the first matching row,
    which could leave two rows that later canonicalized to the same ID.
    """
    if not candidates:
        return []

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    identity_owner: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for key in _identity_keys(candidate):
            owner = identity_owner.get(key)
            if owner is None:
                identity_owner[key] = index
            else:
                union(index, owner)

    # Exact titles are weak identity evidence: merge only when bibliographic
    # context is compatible, so same-title papers from different years/authors
    # remain separate candidates.
    title_owners: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        title_key = _normalize_title(str(candidate.get("title") or ""))
        if not title_key:
            continue
        for owner in title_owners.get(title_key, []):
            if _weak_title_compatible(candidate, candidates[owner]):
                union(index, owner)
        title_owners.setdefault(title_key, []).append(index)

    components: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        components.setdefault(find(index), []).append(index)

    merged: list[dict[str, Any]] = []
    for indexes in sorted(components.values(), key=lambda values: values[0]):
        item = dict(candidates[indexes[0]])
        item["sources"] = list(dict.fromkeys(item.get("sources") or [item.get("source")]))
        item["sources"] = [value for value in item["sources"] if value]
        for index in indexes[1:]:
            _merge_candidate_rows(item, candidates[index])
        _canonicalize_merged_id(item)
        merged.append(item)

    # Defensive invariant: EvidRank must never receive duplicate canonical IDs.
    canonical_ids = [str(item.get("canonical_paper_id") or item["id"]) for item in merged]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise RuntimeError("candidate identity merge produced duplicate canonical IDs")
    return merged


Provider = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]


def search_academic_papers(
    query: str,
    understanding: QueryUnderstanding,
    *,
    target: int | None = None,
    provider_limit: int | None = None,
    allow_second_page: bool = True,
    api_call_budget: int | None = None,
    use_cache: bool = False,
    cache_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query routed academic providers and target the configured unique count."""
    settings = get_settings()
    target_count = max(1, target or settings.academic_min_candidates)
    base_provider_limit = (
        settings.academic_provider_limit
        if provider_limit is None
        else provider_limit
    )
    per_provider = max(base_provider_limit, (target_count + 1) // 2)
    providers: dict[str, Provider] = {
        "serper-scholar": search_serper_scholar,
        "arxiv": search_arxiv,
        "openalex": search_openalex,
    }
    if settings.semantic_scholar_enabled:
        # Keep this in the first round: unlike the AstaBench batch resolver,
        # this endpoint discovers new candidates from the natural-language query.
        providers["semantic-scholar"] = search_semantic_scholar
    if resolve_pasa_title_corpus() is not None:
        providers["pasa-local-corpus"] = search_local_pasa_corpus
    if settings.academic_local_vector_enabled:
        providers["pasa-local-vector"] = search_local_vector_corpus
    if _use_pubmed(query, understanding):
        providers["pubmed"] = search_pubmed
    all_candidates: list[dict[str, Any]] = []
    provider_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def call_provider(
        name: str,
        provider: Provider,
        offset: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        provider_limit_value = per_provider
        provider_offset = offset
        if name in {"pasa-local-corpus", "pasa-local-vector"}:
            provider_limit_value = max(
                per_provider,
                (
                    settings.academic_local_provider_limit
                    if name == "pasa-local-corpus"
                    else settings.academic_local_vector_provider_limit
                ),
            )
            # ``offset`` is the online-provider page boundary.  The local
            # source may use a wider first page, so advance by its own width
            # to avoid returning an overlapping second page.
            if offset > 0:
                provider_offset = provider_limit_value
        cache_key = {
            "provider": name,
            "query": query,
            "understanding": understanding.model_dump(mode="json"),
            "limit": provider_limit_value,
            "offset": provider_offset,
        }
        cache_allowed = (
            use_cache
            and settings.academic_provider_cache_enabled
            and name not in {"pasa-local-corpus", "pasa-local-vector"}
        )
        if cache_allowed:
            cached = load_cached_json(
                f"academic:{name}",
                cache_key,
                ttl_seconds=settings.academic_provider_cache_ttl_seconds,
                allow_expired=cache_only,
            )
            if isinstance(cached, dict) and isinstance(cached.get("papers"), list):
                cached_meta = dict(cached.get("meta") or {})
                return list(cached["papers"]), {
                    **cached_meta,
                    "api_calls": 0,
                    "cached_api_calls": int(cached_meta.get("api_calls") or 0),
                    "cache_hit": True,
                    "latency_ms": 0.0,
                    "offset": provider_offset,
                    "cache_entry_sha256": cache_payload_sha256(cached),
                }
        if cache_only and cache_allowed:
            raise FrozenProviderCacheMiss(
                f"frozen provider snapshot missing academic:{name} "
                f"for query={query!r}, offset={provider_offset}"
            )
        started = time.perf_counter()
        papers, meta = provider(
            query,
            understanding,
            limit=provider_limit_value,
            offset=provider_offset,
        )
        if cache_allowed:
            cached_payload = {"papers": papers, "meta": meta}
            save_cached_json(
                f"academic:{name}",
                cache_key,
                cached_payload,
            )
        return papers, {
            **meta,
            "cache_hit": False,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "offset": provider_offset,
            "cache_entry_sha256": (
                cache_payload_sha256(cached_payload)
                if cache_allowed
                else None
            ),
        }

    def run_round(offset: int, provider_names: list[str] | None = None) -> None:
        selected = {
            name: provider
            for name, provider in providers.items()
            if provider_names is None or name in provider_names
        }
        if not selected:
            return
        started_at = {name: time.perf_counter() for name in selected}
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(
                    call_provider,
                    name,
                    provider,
                    offset,
                ): name
                for name, provider in selected.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    papers, meta = future.result()
                    provider_offset = max(0, int(meta.get("offset") or 0))
                    for provider_rank, paper in enumerate(
                        papers,
                        start=provider_offset + 1,
                    ):
                        paper["provider_ranks"] = {
                            **(paper.get("provider_ranks") or {}),
                            name: provider_rank,
                        }
                        paper["provider_scores"] = {
                            **(paper.get("provider_scores") or {}),
                            name: float(paper.get("score") or 0.0),
                        }
                    all_candidates.extend(papers)
                    previous = provider_meta.get(name) or {}
                    provider_meta[name] = {
                        **meta,
                        "raw_count": int(previous.get("raw_count") or 0)
                        + int(meta.get("raw_count") or 0),
                        "returned_count": int(previous.get("returned_count") or 0)
                        + int(meta.get("returned_count") or 0),
                        "api_calls": int(previous.get("api_calls") or 0)
                        + int(meta.get("api_calls") or 0),
                        "latency_ms": float(previous.get("latency_ms") or 0.0)
                        + float(meta.get("latency_ms") or 0.0),
                        "attempts": [
                            *(previous.get("attempts") or []),
                            {
                                "offset": int(meta.get("offset") or 0),
                                "query": meta.get("query"),
                                "queries": meta.get("queries"),
                                "raw_count": int(meta.get("raw_count") or 0),
                                "returned_count": int(meta.get("returned_count") or 0),
                                "api_calls": int(meta.get("api_calls") or 0),
                                "latency_ms": float(meta.get("latency_ms") or 0.0),
                                "cache_hit": bool(meta.get("cache_hit")),
                                "cache_entry_sha256": meta.get("cache_entry_sha256"),
                                "error": None,
                            },
                        ],
                    }
                except Exception as exc:  # noqa: BLE001
                    # A frozen benchmark must never silently degrade to fewer
                    # providers.  Propagate the miss to the run boundary
                    # instead of recording it as an ordinary provider error.
                    if isinstance(exc, FrozenProviderCacheMiss):
                        raise
                    # Do not surface exception URLs because query parameters may contain API keys.
                    error_type = (
                        exc.error_type
                        if isinstance(exc, ProviderRequestError)
                        else type(exc).__name__
                    )
                    errors[name] = error_type
                    previous = provider_meta.get(name) or {}
                    failed_calls = (
                        exc.attempts
                        if isinstance(exc, ProviderRequestError)
                        else (
                            0
                            if name in {"pasa-local-corpus", "pasa-local-vector"}
                            else 1
                        )
                    )
                    failed_latency_ms = (time.perf_counter() - started_at[name]) * 1000
                    provider_meta[name] = {
                        **previous,
                        "provider": name,
                        "raw_count": int(previous.get("raw_count") or 0),
                        "returned_count": int(previous.get("returned_count") or 0),
                        "api_calls": int(previous.get("api_calls") or 0) + failed_calls,
                        "latency_ms": float(previous.get("latency_ms") or 0.0)
                        + failed_latency_ms,
                        "error": error_type,
                        "circuit_open": bool(
                            isinstance(exc, ProviderRequestError) and exc.circuit_open
                        ),
                        "attempts": [
                            *(previous.get("attempts") or []),
                            {
                                "offset": offset,
                                "query": query,
                                "queries": None,
                                "raw_count": 0,
                                "returned_count": 0,
                                "api_calls": failed_calls,
                                "latency_ms": failed_latency_ms,
                                "error": error_type,
                                "circuit_open": bool(
                                    isinstance(exc, ProviderRequestError) and exc.circuit_open
                                ),
                            },
                        ],
                    }

    run_round(0)
    unique = merge_academic_candidates(all_candidates)
    if allow_second_page and len(unique) < target_count:
        used_calls = sum(int(meta.get("api_calls") or 0) for meta in provider_meta.values())
        remaining_calls = (
            max(0, api_call_budget - used_calls)
            if api_call_budget is not None
            else 10**9
        )
        # Retry the weakest sources first. Local FTS pagination costs no API;
        # online providers are admitted only while the controller budget fits.
        second_page: list[str] = []
        for name in sorted(
            providers,
            key=lambda value: int((provider_meta.get(value) or {}).get("returned_count") or 0),
        ):
            provider_state = provider_meta.get(name) or {}
            if name != "pasa-local-corpus" and provider_state.get("error"):
                # Retrying the same failed provider on page two wastes budget
                # and was a major source of cascading connection failures.
                continue
            estimated_calls = max(
                1,
                int((provider_meta.get(name) or {}).get("api_calls") or 0),
            )
            if name in {"pasa-local-corpus", "pasa-local-vector"}:
                estimated_calls = 0
            if estimated_calls <= remaining_calls:
                second_page.append(name)
                remaining_calls -= estimated_calls
        run_round(per_provider, second_page)
        unique = merge_academic_candidates(all_candidates)

    source_counts = {
        name: int((provider_meta.get(name) or {}).get("returned_count") or 0)
        for name in providers
    }
    network_provider_names = [
        name
        for name in providers
        if name not in {"pasa-local-corpus", "pasa-local-vector"}
    ]
    network_errors = {
        name: str((provider_meta.get(name) or {}).get("error"))
        for name in network_provider_names
        if (provider_meta.get(name) or {}).get("error")
    }
    local_rows = sum(
        source_counts.get(name, 0)
        for name in ("pasa-local-corpus", "pasa-local-vector")
    )
    return unique, {
        "target": target_count,
        "target_met": len(unique) >= target_count,
        "unique_count": len(unique),
        "raw_count": len(all_candidates),
        "source_counts": source_counts,
        "routed_providers": list(providers),
        "providers": provider_meta,
        "errors": errors,
        "network_errors": network_errors,
        "offline_fallback_available": bool(local_rows or any(
            name in providers for name in ("pasa-local-corpus", "pasa-local-vector")
        )),
        "offline_fallback_used": bool(network_errors and local_rows),
        "offline_fallback_candidate_count": local_rows,
        "api_calls": sum(
            int(meta.get("api_calls") or 0)
            for meta in provider_meta.values()
        ),
        "second_page_used": any(
            len(meta.get("attempts") or []) > 1
            for meta in provider_meta.values()
        ),
    }
