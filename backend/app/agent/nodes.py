"""Executable nodes for the adaptive academic-search controller."""

from __future__ import annotations

import time
from collections import Counter
import re
from typing import Any

from backend.app.agent.state import PaperSearchState
from backend.app.config import get_settings
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.academic_search import (
    build_query_variants,
    merge_academic_candidates,
    search_academic_papers,
    search_openalex_neighborhood_multi,
)
from backend.app.services.llm_intent import evolve_query_with_llm
from backend.app.services.tavily_search import TavilyRequestError, search_papers_with_tavily
from backend.app.services.persistent_cache import (
    FrozenProviderCacheMiss,
    cache_payload_sha256,
    load_cached_json,
    save_cached_json,
)


def _trace(
    step: str,
    detail: str,
    *,
    status: str = "done",
    phase: str = "tool",
    tool: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    iteration: int | None = None,
    api_calls: int | None = None,
    token_usage: int | None = None,
    cumulative_api_calls: int | None = None,
    cumulative_token_usage: int | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "detail": detail,
        "status": status,
        "phase": phase,
        "tool": tool,
        "input": input_text,
        "output": output_text,
        "iteration": iteration,
        "api_calls": api_calls,
        "token_usage": token_usage,
        "cumulative_api_calls": cumulative_api_calls,
        "cumulative_token_usage": cumulative_token_usage,
    }


def _log_citation_step(
    *,
    query: str,
    seed_external_ids: list[dict[str, Any]],
    returned: int,
    titles: list[str],
    meta: dict[str, Any],
) -> None:
    """Append one citation-expansion step to the audit log.

    Records which seeds/directions fired, how many papers were returned and
    their titles, so later frozen runs can reuse the snapshot without
    re-calling the citation APIs (saves quota and latency).
    """
    import json as _json
    from datetime import UTC, datetime
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    log_path = root / ".modeling-audit" / "citation_steps.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                _json.dumps(
                    {
                        "logged_at": datetime.now(UTC).isoformat(),
                        "query": query,
                        "seeds": seed_external_ids,
                        "returned_count": returned,
                        "titles": titles,
                        "api_calls": int(meta.get("api_calls") or 0),
                        "directions": list(meta.get("directions") or []),
                        "cache_entry_sha256": meta.get("cache_entry_sha256"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001 - logging is best-effort
        pass


def _unique_queries(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = " ".join(str(item or "").split()).strip()
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def _facet_values(understanding: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("topics", "entities", "methods", "datasets", "venues"):
        values.extend(str(item).strip() for item in understanding.get(field) or [])
    return _unique_queries([item for item in values if len(item) > 1])


def _normalized_words(value: str) -> set[str]:
    words: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", value.casefold()):
        clean = token.strip("._-")
        if len(clean) > 3 and clean.endswith("ies"):
            clean = f"{clean[:-3]}y"
        elif len(clean) > 4 and clean.endswith("ing"):
            clean = clean[:-3]
        elif len(clean) > 4 and clean.endswith("ed"):
            clean = clean[:-2]
        elif len(clean) > 3 and clean.endswith("s") and not clean.endswith("ss"):
            clean = clean[:-1]
        if clean:
            words.add(clean)
    return words


def _facet_is_covered(facet: str, searchable_text: str) -> bool:
    folded = facet.casefold()
    if folded in searchable_text:
        return True
    facet_words = _normalized_words(facet)
    if not facet_words:
        return False
    document_words = _normalized_words(searchable_text)
    # Require every short/acronym facet token and most tokens for longer
    # phrases. This recognizes harmless morphology without treating a single
    # broad word as evidence that a multi-condition query is covered.
    required = len(facet_words) if len(facet_words) <= 2 else len(facet_words) - 1
    return len(facet_words & document_words) >= required


def _coverage_report(
    understanding: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    facets = _facet_values(understanding)
    if not facets:
        return {
            "rate": 1.0,
            "covered": [],
            "total": 0,
            "per_facet": {},
        }, []

    searchable_texts = [
        " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("abstract") or ""),
                str(item.get("venue") or ""),
                " ".join(str(tag) for tag in item.get("tags") or []),
            ]
        ).casefold()
        for item in candidates
    ]
    per_facet = {
        facet: any(_facet_is_covered(facet, text) for text in searchable_texts)
        for facet in facets
    }
    covered = [facet for facet, present in per_facet.items() if present]
    gaps = [facet for facet, present in per_facet.items() if not present]
    return {
        "rate": len(covered) / len(facets),
        "covered": covered,
        "total": len(facets),
        "per_facet": per_facet,
    }, gaps


def _candidate_topic_cues(
    candidates: list[dict[str, Any]],
    *,
    original_query: str = "",
) -> list[str]:
    tag_counts = Counter(
        str(tag).strip()
        for candidate in candidates
        for tag in candidate.get("tags") or []
        if len(str(tag).strip()) > 2
    )
    query_words = _normalized_words(original_query)
    stopwords = {
        "about", "approach", "based", "method", "model", "paper", "research",
        "result", "study", "system", "using", "work",
    }
    title_document_frequency: Counter[str] = Counter()
    for candidate in candidates[:24]:
        words = {
            word
            for word in _normalized_words(str(candidate.get("title") or ""))
            if len(word) >= 4 and word not in query_words and word not in stopwords
        }
        title_document_frequency.update(words)

    cues = [term for term, _count in tag_counts.most_common(6)]
    # Terms repeated across multiple independently retrieved titles are safer
    # evolution cues than a token seen in only one top result.
    cues.extend(
        term
        for term, count in title_document_frequency.most_common(8)
        if count >= 2
    )
    return _unique_queries(cues)[:6]


def _prf_keywords(
    candidates: list[dict[str, Any]],
    *,
    original_query: str = "",
    top_n: int = 6,
) -> list[str]:
    """Pseudo-Relevance-Feedback keyword extraction from high-confidence seeds.

    Unlike ``_candidate_topic_cues`` (which leans on provider tags + title
    word frequency), this looks at title+abstract of the seed pool and scores
    terms by cross-document frequency (a term must recur across several
    high-confidence papers to be a genuine shared topic) while excluding
    original-query words and generic stopwords. The result drives a
    query-plus-seed-keywords second-round search that is far more precise
    than citation tracing for finding the actual target papers.
    """
    query_words = _normalized_words(original_query)
    stopwords = {
        "about", "approach", "based", "been", "between", "different", "during",
        "large", "method", "methods", "model", "models", "new", "paper",
        "papers", "present", "propose", "proposed", "research", "result",
        "results", "show", "shows", "study", "system", "systems", "task",
        "tasks", "using", "work", "works",
    }
    from collections import defaultdict as _defaultdict

    doc_frequency: Counter[str] = Counter()
    total_seeds = 0
    for candidate in candidates[:24]:
        text = " ".join(
            [
                str(candidate.get("title") or ""),
                str(candidate.get("abstract") or ""),
                " ".join(str(tag) for tag in candidate.get("tags") or []),
            ]
        )
        words = {
            word
            for word in _normalized_words(text)
            if len(word) >= 4
            and word not in query_words
            and word not in stopwords
            and not word.isdigit()
        }
        if not words:
            continue
        total_seeds += 1
        for word in words:
            doc_frequency[word] += 1
    if total_seeds < 2:
        # Not enough seeds to infer shared topics; fall back to topic cues.
        return _candidate_topic_cues(
            candidates, original_query=original_query
        )[:top_n]

    # A term must appear in at least 2 seeds to be a shared topic, and rarer
    # (more discriminative) terms score higher within the recurring set.
    recurring = {
        word: count
        for word, count in doc_frequency.items()
        if count >= 2
    }
    ranked = sorted(
        recurring,
        key=lambda word: (
            -recurring[word],
            -len(word),
            word.casefold(),
        ),
    )
    return _unique_queries(ranked)[:top_n]


def _filter_citation_papers(
    papers: list[dict[str, Any]],
    *,
    query: str,
    prf_keywords: list[str],
    min_overlap: int = 1,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop citation-expansion papers that are not related to the query.

    Citation tracing returns many same-field but off-topic surveys (the
    precision problem we measured: 57 citation papers, 0 gold hits). Before
    merging them into the candidate pool, require each paper to share at
    least ``min_overlap`` tokens with the query and/or the PRF seed keywords;
    otherwise it is discarded as noise. Returns (kept, dropped_titles).
    """
    query_words = _normalized_words(query)
    seed_words: set[str] = set()
    for keyword in prf_keywords:
        seed_words |= _normalized_words(keyword)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for paper in papers:
        text = " ".join(
            [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
            ]
        )
        paper_words = _normalized_words(text)
        overlap = len((paper_words & query_words) | (paper_words & seed_words))
        if overlap >= min_overlap:
            kept.append(paper)
        else:
            dropped.append(str(paper.get("title") or "")[:80])
    return kept, dropped


def _select_citation_seeds(
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Select the most confident candidate papers as citation-expansion seeds.

    PaSa-style: instead of one seed, take the top-N papers that carry an
    OpenAlex ID or a DOI (both resolve to OpenAlex citation edges), ranked by
    best provider rank, source count and score. The controller bounds the
    number actually used by the API budget.
    """
    eligible = [
        candidate
        for candidate in candidates
        if (candidate.get("external_ids") or {}).get("openalex")
        or (candidate.get("external_ids") or {}).get("doi")
        or (candidate.get("external_ids") or {}).get("arxiv")
    ]
    if not eligible:
        return []

    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        ranks = [
            int(value)
            for value in (candidate.get("provider_ranks") or {}).values()
            if int(value) > 0
        ]
        # Mature papers (citation_count > 0) first: a brand-new arXiv preprint
        # has no cites/references to expand and wastes the API budget. Within
        # each maturity tier, fall back to provider rank / source count / score.
        return (
            0 if int(candidate.get("citation_count") or 0) > 0 else 1,
            min(ranks, default=10**6),
            -int(candidate.get("citation_count") or 0),
            -len(candidate.get("sources") or [candidate.get("source")]),
            -float(candidate.get("score") or 0.0),
            str(candidate.get("id") or ""),
        )

    return sorted(eligible, key=key)[: max(1, top_n)]


def _candidate_identity_snapshot(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep compact pre-merge identities for evaluation diagnostics."""
    return [
        {
            "id": str(candidate.get("canonical_paper_id") or candidate.get("id") or ""),
            "title": str(candidate.get("title") or ""),
            "external_ids": dict(candidate.get("external_ids") or {}),
            "source": str(candidate.get("source") or "unknown"),
        }
        for candidate in candidates
    ]


def plan_search(state: PaperSearchState) -> dict[str, Any]:
    """Build the initial query queue from the parsed research intent."""
    settings = get_settings()
    understanding = state.get("understanding") or {}
    raw_query = str(state.get("raw_query") or "")
    parsed = QueryUnderstanding.model_validate(understanding)
    plan = _unique_queries(
        build_query_variants(raw_query, parsed, limit=6)
    )
    if not plan:
        plan = [raw_query]
    max_iterations = max(
        1,
        int(state.get("max_iterations") or settings.agent_search_max_iterations),
    )
    current_query = plan[0]
    return {
        "search_plan": plan,
        "current_query": current_query,
        "executed_queries": [],
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "target_candidates": int(
            state.get("target_candidates") or settings.academic_min_candidates
        ),
        "round_candidate_target": int(
            state.get("round_candidate_target") or settings.agent_search_round_target
        ),
        "min_new_candidates": int(
            state.get("min_new_candidates")
            or settings.agent_search_min_new_candidates
        ),
        "min_coverage": float(
            state.get("min_coverage") or settings.agent_search_min_coverage
        ),
        "max_api_calls": int(
            state.get("max_api_calls") or settings.agent_search_max_api_calls
        ),
        "llm_evolution_enabled": bool(
            state.get("llm_evolution_enabled", settings.agent_search_llm_evolution_enabled)
        ),
        "llm_evolution_calls": int(state.get("llm_evolution_calls") or 0),
        "should_continue": True,
        "agent_trace": [
            _trace(
                "agent:plan",
                f"建立自适应检索计划：{len(plan)} 个候选查询，最多 {max_iterations} 轮",
                phase="plan",
                tool="adaptive-search-controller",
                input_text=raw_query,
                output_text=f"round_1={current_query}",
            )
        ],
    }


def retrieve_papers(state: PaperSearchState) -> dict[str, Any]:
    """Execute one retrieval round against academic providers and Tavily."""
    settings = get_settings()
    current_query = str(state.get("current_query") or state.get("raw_query") or "")
    iteration = int(state.get("current_iteration") or 0)
    understanding = QueryUnderstanding.model_validate(state.get("understanding") or {})
    evaluation_fast_mode = bool(state.get("evaluation_fast_mode"))
    provider_cache_mode = str(state.get("provider_cache_mode") or "record")
    cache_only = evaluation_fast_mode and provider_cache_mode == "frozen"
    round_understanding = understanding.model_copy(
        update={
            "rewritten_queries": [current_query],
            # Network providers use the evolved query, while the zero-cost
            # local FTS source also keeps the user's exact wording. LLM
            # rewrites occasionally remove the decisive relation or dataset.
            # Preserve the intent parser's independent decomposition. The old
            # update replaced it, so method/dataset/relation subqueries never
            # reached the cheap local FTS provider.
            "sub_queries": _unique_queries(
                [
                    current_query,
                    *understanding.sub_queries,
                    str(state.get("raw_query") or "").strip(),
                ]
            )[:6],
        }
    )

    academic_candidates: list[dict[str, Any]] = []
    academic_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        remaining_api_calls = max(
            0,
            int(state.get("max_api_calls") or settings.agent_search_max_api_calls)
            - int(state.get("api_calls") or 0),
        )
        academic_candidates, academic_meta = search_academic_papers(
            current_query,
            round_understanding,
            target=int(state.get("round_candidate_target") or 40),
            provider_limit=max(
                10,
                int(state.get("round_candidate_target") or 40) // 2,
            ),
            # First round stays cheap. A second-page fallback is enabled only
            # after the controller has observed a real recall/coverage gap.
            allow_second_page=iteration > 0,
            # Reserve one request for Tavily when the API budget permits it.
            api_call_budget=max(0, remaining_api_calls - 1),
            use_cache=evaluation_fast_mode,
            cache_only=cache_only,
        )
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, FrozenProviderCacheMiss):
            raise
        errors["academic"] = type(exc).__name__

    citation_candidates: list[dict[str, Any]] = []
    citation_meta: dict[str, Any] = {"api_calls": 0, "skipped": "not_needed"}
    seeds = _select_citation_seeds(
        list(state.get("candidates") or []),
        top_n=settings.semantic_scholar_citation_expansion_seed_top_n,
    )
    # OpenAlex citation expansion works anonymously (mailto raises the pool),
    # so no API key is required; the toggle just controls whether the feature
    # is active at all.
    citation_configured = bool(
        settings.semantic_scholar_citation_expansion_enabled
    )
    # PaSa-style: expand along citation edges every round after the first,
    # not only when coverage is below target. The API budget still bounds how
    # many (seed, direction) calls actually run.
    citation_needed = (
        iteration > 0
        and bool(seeds)
        and citation_configured
        and settings.semantic_scholar_citation_expansion_every_round
    )
    academic_api_calls = int(academic_meta.get("api_calls") or 0)
    remaining_for_citation = max(
        0,
        int(state.get("max_api_calls") or settings.agent_search_max_api_calls)
        - int(state.get("api_calls") or 0)
        - academic_api_calls,
    )
    budgeted_seeds = seeds[: max(1, min(len(seeds), remaining_for_citation // 2))]
    citation_allowed = citation_needed and bool(budgeted_seeds)
    if citation_allowed:
        seed_external_ids = [
            dict(seed.get("external_ids") or {}) for seed in budgeted_seeds
        ]
        cache_key = {
            "seed_external_ids": seed_external_ids,
            "query": current_query,
            "understanding": round_understanding.model_dump(mode="json"),
            "limit": settings.semantic_scholar_citation_expansion_limit,
            "directions": sorted(settings.semantic_scholar_citation_expansion_directions),
        }
        cached = (
            load_cached_json(
                "academic:openalex-neighborhood",
                cache_key,
                ttl_seconds=settings.academic_provider_cache_ttl_seconds,
                allow_expired=cache_only,
            )
            if evaluation_fast_mode and settings.academic_provider_cache_enabled
            else None
        )
        if isinstance(cached, dict) and isinstance(cached.get("papers"), list):
            citation_candidates = list(cached["papers"])
            stored_meta = dict(cached.get("meta") or {})
            citation_meta = {
                **stored_meta,
                "api_calls": 0,
                "cached_api_calls": int(stored_meta.get("api_calls") or 0),
                "cache_hit": True,
                "cache_entry_sha256": cache_payload_sha256(cached),
            }
        elif cache_only and settings.academic_provider_cache_enabled:
            raise FrozenProviderCacheMiss(
                f"frozen citation snapshot missing for seeds={seed_external_ids!r}"
            )
        else:
            try:
                citation_started = time.perf_counter()
                citation_candidates, citation_meta = search_openalex_neighborhood_multi(
                    seed_external_ids,
                    limit=settings.semantic_scholar_citation_expansion_limit,
                    directions=list(
                        settings.semantic_scholar_citation_expansion_directions
                    ),
                )
                citation_meta = {
                    **citation_meta,
                    "latency_ms": (time.perf_counter() - citation_started) * 1000,
                    "cache_hit": False,
                }
                if evaluation_fast_mode and settings.academic_provider_cache_enabled:
                    payload = {"papers": citation_candidates, "meta": citation_meta}
                    save_cached_json(
                        "academic:openalex-neighborhood",
                        cache_key,
                        payload,
                    )
                    citation_meta["cache_entry_sha256"] = cache_payload_sha256(payload)
                # Record every citation-expansion step for audit/reporting and
                # to let frozen runs reuse snapshots without re-calling APIs.
                try:
                    _log_citation_step(
                        query=current_query,
                        seed_external_ids=seed_external_ids,
                        returned=len(citation_candidates),
                        titles=[str(p.get("title")) for p in citation_candidates[:20]],
                        meta=citation_meta,
                    )
                except Exception:  # noqa: BLE001 - logging must never break search
                    pass
                # Precision guard: citation tracing pulls many same-field but
                # off-topic surveys. Drop papers with no token overlap with
                # the query or the PRF seed keywords before merging.
                prf = (
                    _prf_keywords(
                        list(state.get("candidates") or []),
                        original_query=str(state.get("raw_query") or ""),
                        top_n=6,
                    )
                    if get_settings().agent_search_prf_enabled
                    else _candidate_topic_cues(
                        list(state.get("candidates") or []),
                        original_query=str(state.get("raw_query") or ""),
                    )
                )
                citation_candidates, dropped_titles = _filter_citation_papers(
                    citation_candidates,
                    query=current_query,
                    prf_keywords=prf,
                    min_overlap=1,
                )
                citation_meta = {
                    **citation_meta,
                    "filtered_count": len(citation_candidates),
                    "filtered_dropped": dropped_titles[:10],
                }
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, FrozenProviderCacheMiss):
                    raise
                error_type = (
                    exc.error_type
                    if hasattr(exc, "error_type")
                    else type(exc).__name__
                )
                errors["semantic-scholar-neighborhood"] = str(error_type)
                citation_meta = {
                    "api_calls": int(getattr(exc, "attempts", 1)),
                    "returned_count": 0,
                    "error": str(error_type),
                    "seed_paper_ids": seed_ids,
                }
        citation_source = str(
            citation_meta.get("provider") or "semantic-scholar-neighborhood"
        )
        for provider_rank, paper in enumerate(citation_candidates, start=1):
            paper["provider_ranks"] = {
                **(paper.get("provider_ranks") or {}),
                citation_source: provider_rank,
            }
            paper["provider_scores"] = {
                **(paper.get("provider_scores") or {}),
                citation_source: float(paper.get("score") or 0.0),
            }
        academic_candidates = merge_academic_candidates(
            [*academic_candidates, *citation_candidates]
        )
        academic_meta = {
            **academic_meta,
            "api_calls": int(academic_meta.get("api_calls") or 0)
            + int(citation_meta.get("api_calls") or 0),
            "citation_expansion": citation_meta,
        }
    else:
        citation_meta = {
            "api_calls": 0,
            "skipped": (
                "api_budget_exhausted"
                if citation_needed
                else (
                    "api_key_required"
                    if iteration > 0 and bool(seeds) and not citation_configured
                    else "no_seed_or_config"
                )
            ),
        }
        academic_meta = {**academic_meta, "citation_expansion": citation_meta}

    tavily_candidates: list[dict[str, Any]] = []
    tavily_meta: dict[str, Any] = {}
    academic_api_calls = int(academic_meta.get("api_calls") or 0)
    previous_tavily_failed = any(
        (round_info.get("errors") or {}).get("tavily")
        for round_info in state.get("round_history") or []
    )
    academic_coverage, _academic_gaps = _coverage_report(
        understanding.model_dump(),
        academic_candidates,
    )
    tavily_needed = (
        not settings.agent_search_tavily_conditional
        or iteration > 0
        or len(academic_candidates) < settings.agent_search_tavily_min_academic_candidates
        or float(academic_coverage.get("rate") or 0.0)
        < settings.agent_search_tavily_min_coverage
    )
    tavily_allowed = (
        int(state.get("api_calls") or 0) + academic_api_calls
        < int(state.get("max_api_calls") or settings.agent_search_max_api_calls)
    ) and not previous_tavily_failed and tavily_needed
    if tavily_allowed:
        try:
            tavily_started = time.perf_counter()
            tavily_cache_key = {
                "query": current_query,
                "understanding": round_understanding.model_dump(mode="json"),
                "max_results": settings.agent_search_tavily_limit,
            }
            cached_tavily = (
                load_cached_json(
                    "tavily",
                    tavily_cache_key,
                    ttl_seconds=settings.academic_provider_cache_ttl_seconds,
                    allow_expired=cache_only,
                )
                if evaluation_fast_mode and settings.academic_provider_cache_enabled
                else None
            )
            if isinstance(cached_tavily, dict) and isinstance(
                cached_tavily.get("papers"), list
            ):
                tavily_candidates = list(cached_tavily["papers"])
                original_meta = dict(cached_tavily.get("meta") or {})
                tavily_meta = {
                    **original_meta,
                    "api_calls": 0,
                    "cached_api_calls": int(original_meta.get("api_calls") or 0),
                    "cache_hit": True,
                    "cache_entry_sha256": cache_payload_sha256(cached_tavily),
                }
            else:
                if cache_only and settings.academic_provider_cache_enabled:
                    raise FrozenProviderCacheMiss(
                        f"frozen provider snapshot missing tavily for query={current_query!r}"
                    )
                tavily_candidates, tavily_meta = search_papers_with_tavily(
                    current_query,
                    round_understanding,
                    max_results=settings.agent_search_tavily_limit,
                )
                if evaluation_fast_mode and settings.academic_provider_cache_enabled:
                    cached_payload = {
                        "papers": tavily_candidates,
                        "meta": tavily_meta,
                    }
                    save_cached_json(
                        "tavily",
                        tavily_cache_key,
                        cached_payload,
                    )
                tavily_meta = {
                    **tavily_meta,
                    "cache_hit": False,
                    "cache_entry_sha256": (
                        cache_payload_sha256(cached_payload)
                        if evaluation_fast_mode
                        and settings.academic_provider_cache_enabled
                        else None
                    ),
                }
            tavily_meta = {
                **tavily_meta,
                "returned_count": len(tavily_candidates),
                "latency_ms": (time.perf_counter() - tavily_started) * 1000,
            }
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, FrozenProviderCacheMiss):
                raise
            error_type = (
                exc.error_type if isinstance(exc, TavilyRequestError) else type(exc).__name__
            )
            errors["tavily"] = error_type
            tavily_meta = {
                "api_calls": exc.attempts if isinstance(exc, TavilyRequestError) else 1,
                "returned_count": 0,
                "latency_ms": (time.perf_counter() - tavily_started) * 1000,
                "error": error_type,
            }
    else:
        tavily_meta = {
            "api_calls": 0,
            "returned_count": 0,
            "skipped": (
                "previous_round_provider_failure"
                if previous_tavily_failed
                else (
                    "academic_recall_and_coverage_sufficient"
                    if not tavily_needed
                    else "api_budget_reserved_for_academic_recall"
                )
            ),
            "academic_candidate_count": len(academic_candidates),
            "academic_coverage": float(academic_coverage.get("rate") or 0.0),
        }

    round_candidates = merge_academic_candidates(
        [*academic_candidates, *tavily_candidates]
    )
    for candidate in round_candidates:
        candidate["source_query"] = current_query
        candidate["source_queries"] = [current_query]
        candidate["source_iteration"] = iteration

    previous = list(state.get("candidates") or [])
    combined = merge_academic_candidates([*previous, *round_candidates])
    new_count = max(0, len(combined) - len(previous))
    round_api_calls = int(academic_meta.get("api_calls") or 0) + int(
        tavily_meta.get("api_calls") or 0
    )
    # Retrieval providers and rule-based query evolution do not call an LLM.
    round_token_usage = 0
    api_calls = int(state.get("api_calls") or 0) + round_api_calls
    token_usage = int(state.get("token_usage") or 0) + round_token_usage
    executed_queries = _unique_queries(
        [*(state.get("executed_queries") or []), current_query]
    )
    history = {
        "iteration": iteration + 1,
        "query": current_query,
        "retrieved_count": len(round_candidates),
        "new_unique_count": new_count,
        "total_unique_count": len(combined),
        "api_calls": round_api_calls,
        "token_usage": round_token_usage,
        "cumulative_api_calls": api_calls,
        "cumulative_token_usage": token_usage,
        "academic": academic_meta,
        "tavily": tavily_meta,
        "provider_candidates": _candidate_identity_snapshot(
            [*academic_candidates, *tavily_candidates]
        ),
        "errors": errors,
    }
    round_trace = [
        _trace(
            f"agent:retrieve:{iteration + 1}",
            (
                f"第 {iteration + 1} 轮召回 {len(round_candidates)} 篇，"
                f"新增唯一论文 {new_count} 篇，累计 {len(combined)} 篇"
            ),
            status="done" if round_candidates else "error",
            phase="tool",
            tool="academic-multisource+tavily",
            input_text=current_query,
            output_text=f"api_calls={round_api_calls}, errors={list(errors)}",
            iteration=iteration + 1,
            api_calls=round_api_calls,
            token_usage=round_token_usage,
            cumulative_api_calls=api_calls,
            cumulative_token_usage=token_usage,
        )
    ]
    if citation_candidates:
        round_trace.append(
            _trace(
                f"agent:citation-hop:{iteration + 1}",
                (
                    "沿高置信种子论文执行有界引文扩展："
                    f"{len(citation_candidates)} 篇候选"
                ),
                phase="tool",
                tool=str(citation_meta.get("provider") or "semantic-scholar-neighborhood"),
                input_text=",".join(citation_meta.get("seeds") or []),
                output_text=(
                    f"directions={','.join(citation_meta.get('directions') or [])}, "
                    f"cache_hit={bool(citation_meta.get('cache_hit'))}"
                ),
                iteration=iteration + 1,
                api_calls=int(citation_meta.get("api_calls") or 0),
                cumulative_api_calls=api_calls,
                cumulative_token_usage=token_usage,
            )
        )
    return {
        "candidates": combined,
        "round_candidates": round_candidates,
        "round_new_count": new_count,
        "round_history": [history],
        "executed_queries": executed_queries,
        "api_calls": api_calls,
        "token_usage": token_usage,
        "agent_trace": round_trace,
    }


def analyze_coverage(state: PaperSearchState) -> dict[str, Any]:
    """Measure constraint coverage and marginal gain after a retrieval round."""
    candidates = list(state.get("candidates") or [])
    coverage, gaps = _coverage_report(
        state.get("understanding") or {},
        candidates,
    )
    iteration = int(state.get("current_iteration") or 0)
    return {
        "coverage": coverage,
        "coverage_gaps": gaps,
        "agent_trace": [
            _trace(
                f"agent:analyze:{iteration + 1}",
                (
                    f"约束覆盖率 {coverage['rate']:.0%}，"
                    f"未覆盖维度 {len(gaps)} 个，"
                    f"本轮新增 {int(state.get('round_new_count') or 0)} 篇"
                ),
                phase="observe",
                tool="coverage-gap-analyzer",
                output_text=f"gaps={gaps[:8]}",
                iteration=iteration + 1,
            )
        ],
    }


def decide_iteration(state: PaperSearchState) -> dict[str, Any]:
    """Stop on sufficient coverage, low marginal gain, max rounds, or budget."""
    iteration = int(state.get("current_iteration") or 0)
    rounds_done = iteration + 1
    total = len(state.get("candidates") or [])
    target = int(state.get("target_candidates") or 80)
    coverage_rate = float((state.get("coverage") or {}).get("rate") or 0.0)
    min_coverage = float(state.get("min_coverage") or 0.6)
    api_calls = int(state.get("api_calls") or 0)
    max_api_calls = int(state.get("max_api_calls") or 12)
    new_count = int(state.get("round_new_count") or 0)
    min_new = int(state.get("min_new_candidates") or 5)
    max_iterations = int(state.get("max_iterations") or 2)
    executed = {
        str(item).casefold()
        for item in state.get("executed_queries") or []
        if str(item).strip()
    }
    has_unexecuted_variant = any(
        str(item).strip() and str(item).casefold() not in executed
        for item in state.get("search_plan") or []
    )
    minimum_evaluation_rounds = min(
        max_iterations,
        max(1, get_settings().paper_test_min_query_variants),
    )
    force_distinct_evaluation_variant = (
        bool(state.get("evaluation_fast_mode"))
        and rounds_done < minimum_evaluation_rounds
        and has_unexecuted_variant
        and api_calls < max_api_calls
    )

    if force_distinct_evaluation_variant:
        should_continue = True
        stop_reason = "evaluation_minimum_distinct_variants"
    elif total >= target and coverage_rate >= min_coverage:
        should_continue = False
        stop_reason = "target_and_coverage_met"
    elif rounds_done >= max_iterations:
        should_continue = False
        stop_reason = "max_iterations_reached"
    elif api_calls >= max_api_calls:
        should_continue = False
        stop_reason = "api_budget_exhausted"
    elif iteration > 0 and new_count < min_new:
        should_continue = False
        stop_reason = "low_marginal_gain"
    else:
        should_continue = True
        stop_reason = "coverage_or_recall_gap"

    return {
        "should_continue": should_continue,
        "stop_reason": stop_reason,
        "agent_trace": [
            _trace(
                f"agent:decide:{rounds_done}",
                (
                    "继续检索并演化查询"
                    if should_continue
                    else f"停止检索：{stop_reason}"
                ),
                phase="plan",
                tool="adaptive-stop-controller",
                output_text=(
                    f"unique={total}/{target}, coverage={coverage_rate:.2f}/"
                    f"{min_coverage:.2f}, api_calls={api_calls}/{max_api_calls}"
                ),
                iteration=rounds_done,
                cumulative_api_calls=api_calls,
                cumulative_token_usage=int(state.get("token_usage") or 0),
            )
        ],
    }


def evolve_query(state: PaperSearchState) -> dict[str, Any]:
    """Choose the next planned query and enrich it with uncovered facets."""
    next_iteration = int(state.get("current_iteration") or 0) + 1
    executed = {item.casefold() for item in state.get("executed_queries") or []}
    plan = list(state.get("search_plan") or [])
    base_query = next(
        (item for item in plan if item.casefold() not in executed),
        str(state.get("raw_query") or ""),
    )
    gaps = list(state.get("coverage_gaps") or [])
    # PRF: pull high-discrimination keywords from the high-confidence seed
    # pool (title+abstract) and combine them with the user query for the
    # second-round search. Fall back to generic topic cues when PRF is
    # disabled (baseline) or the seed pool is too thin.
    prf_enabled = bool(get_settings().agent_search_prf_enabled)
    prf_keywords = (
        _prf_keywords(
            list(state.get("candidates") or []),
            original_query=str(state.get("raw_query") or ""),
            top_n=6,
        )
        if prf_enabled
        else []
    )
    cues = _candidate_topic_cues(
        list(state.get("candidates") or []),
        original_query=str(state.get("raw_query") or ""),
    )
    expansion_terms = gaps[:4] if gaps else prf_keywords[:4] or cues[:3]
    additions = [
        term
        for term in expansion_terms
        if term.casefold() not in base_query.casefold()
    ]
    deterministic = " ".join([base_query, *additions]).strip()
    if deterministic.casefold() in executed:
        deterministic = f"{base_query} recent methods evidence".strip()
    deterministic_reason = (
        f"补齐未覆盖约束：{', '.join(gaps[:4])}"
        if gaps
        else (
            f"根据高质量种子论文关键词二次检索："
            f"{', '.join(prf_keywords[:4]) or '增加检索多样性'}"
        )
    )
    evolved = deterministic
    reason = deterministic_reason
    settings = get_settings()
    llm_calls = int(state.get("llm_evolution_calls") or 0)
    use_llm = (
        bool(state.get("llm_evolution_enabled"))
        and str(state.get("stop_reason") or "") == "coverage_or_recall_gap"
        and llm_calls < max(0, settings.agent_search_llm_evolution_max_calls)
        and int(state.get("api_calls") or 0) < int(state.get("max_api_calls") or 0)
    )
    llm_meta: dict[str, Any] = {"api_calls": 0, "token_usage": 0}
    cache_hit = False
    if use_llm:
        cache_key = {
            "prompt_version": settings.agent_search_llm_evolution_prompt_version,
            "model": settings.zhipu_intent_model,
            "raw_query": state.get("raw_query"),
            "understanding": state.get("understanding") or {},
            "base_query": base_query,
            "coverage_gaps": gaps[:6],
            "candidate_cues": cues[:6],
            "executed_queries": list(state.get("executed_queries") or []),
        }
        evaluation = bool(state.get("evaluation_fast_mode"))
        frozen = evaluation and str(state.get("provider_cache_mode") or "") == "frozen"
        cached = (
            load_cached_json(
                "adaptive-query",
                cache_key,
                ttl_seconds=settings.academic_provider_cache_ttl_seconds,
                allow_expired=frozen,
            )
            if evaluation and settings.academic_provider_cache_enabled
            else None
        )
        if isinstance(cached, dict) and cached.get("query"):
            evolved = str(cached["query"])
            reason = str(cached.get("reason") or deterministic_reason)
            llm_meta = {"api_calls": 0, "token_usage": 0, "cached": True}
            cache_hit = True
        elif frozen and settings.academic_provider_cache_enabled:
            raise FrozenProviderCacheMiss(
                f"frozen provider snapshot missing adaptive-query for query={base_query!r}"
            )
        else:
            parsed = QueryUnderstanding.model_validate(state.get("understanding") or {})
            candidate_query, candidate_reason, llm_meta = evolve_query_with_llm(
                raw_query=str(state.get("raw_query") or ""),
                understanding=parsed,
                base_query=base_query,
                coverage_gaps=gaps,
                candidate_cues=cues,
                executed_queries=list(state.get("executed_queries") or []),
            )
            if candidate_query:
                evolved = candidate_query
                reason = candidate_reason or deterministic_reason
                if evaluation and settings.academic_provider_cache_enabled:
                    save_cached_json(
                        "adaptive-query",
                        cache_key,
                        {"query": evolved, "reason": reason},
                    )
    api_calls = int(state.get("api_calls") or 0) + int(llm_meta.get("api_calls") or 0)
    token_usage = int(state.get("token_usage") or 0) + int(llm_meta.get("token_usage") or 0)
    used_llm = use_llm and (cache_hit or not llm_meta.get("error"))
    return {
        "current_iteration": next_iteration,
        "current_query": evolved,
        "api_calls": api_calls,
        "token_usage": token_usage,
        "llm_evolution_calls": llm_calls + int(llm_meta.get("api_calls") or 0),
        "agent_trace": [
            _trace(
                f"agent:evolve:{next_iteration + 1}",
                reason,
                phase="plan",
                tool="llm-query-evolution" if used_llm else "query-evolution-fallback",
                input_text=base_query,
                output_text=(
                    f"{evolved} (cache)" if cache_hit else evolved
                ),
                iteration=next_iteration + 1,
                api_calls=int(llm_meta.get("api_calls") or 0),
                token_usage=int(llm_meta.get("token_usage") or 0),
                cumulative_api_calls=api_calls,
                cumulative_token_usage=token_usage,
            )
        ],
    }
