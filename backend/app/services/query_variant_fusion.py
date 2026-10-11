"""Fuse pre-generated PaSa Crawler search queries into the LLM-intent
understanding as AUXILIARY query variants.

The crawler queries are never a replacement for the intent parser — they add
extra FTS / provider / vector variants.  ``include_adaptive`` additionally
merges the coverage-gap follow-up queries (direction 1: adaptive retrieval).

The cache is produced offline by backend/scripts/generate_crawler_queries.py
(the crawler 7B is too large to share the GPU with the selector, so query
generation is a pre-pass, keyed by the raw query string).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.config import ROOT_DIR, get_settings

# By construction Qwen2-style queries are ASCII; keep a plain normalize for
# cache lookup so the same query string always hits the same entry.
def _key(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _load_cache(path: str) -> dict:
    cache_path = Path(path)
    if not cache_path.is_absolute():
        cache_path = ROOT_DIR / cache_path
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fuse_crawler_queries(query: str, understanding) -> "object":
    """Return ``understanding`` with crawler variants appended (no-op when the
    cache has no entry for this query or the feature is disabled)."""
    settings = get_settings()
    if not settings.crawler_query_fusion_enabled:
        return understanding

    cache = _load_cache(settings.crawler_query_cache_path)
    entry = None
    wanted = _key(query)
    # Exact-normalized lookup first, then prefix match on the first 40 chars.
    for key, value in cache.items():
        if _key(value.get("query") or "") == wanted:
            entry = value
            break
    if entry is None:
        for value in cache.values():
            if wanted[:40] and _key(value.get("query") or "")[:40] == wanted[:40]:
                entry = value
                break
    if entry is None:
        return understanding

    extra: list[str] = []
    extra.extend(str(q) for q in entry.get("initial") or [])
    if settings.crawler_query_include_adaptive:
        extra.extend(str(q) for q in entry.get("adaptive") or [])
    if not extra:
        return understanding

    seen = {
        _key(q)
        for q in [
            *(understanding.rewritten_queries or []),
            *(understanding.sub_queries or []),
            *(understanding.topics or []),
            *(understanding.keywords or []),
        ]
    }
    merged = [
        q for q in extra if _key(q) not in seen
    ][: max(1, settings.crawler_query_max_variants)]
    if not merged:
        return understanding

    understanding.rewritten_queries = [
        *(understanding.rewritten_queries or []),
        *merged,
    ]
    understanding.sub_queries = [*(understanding.sub_queries or []), *merged]
    # Deliberately NOT added to keywords/topics: those feed the single
    # OR-joined precise FTS query, and extra OR terms dilute the exact-match
    # BM25 signal (measured: Q7 gold dropped 19->8 when crawler phrases were
    # OR-joined). sub_queries become independent FTS branches instead.
    understanding.raw_json = {
        **(understanding.raw_json or {}),
        "crawler_query_fusion": {
            "source": "pasa-7b-crawler",
            "include_adaptive": settings.crawler_query_include_adaptive,
            "added_variants": merged,
        },
    }
    return understanding


def crawler_fusion_digest(query: str) -> str | None:
    """Stable digest of the fused variants for this query (audit/trace)."""
    settings = get_settings()
    if not settings.crawler_query_fusion_enabled:
        return None
    cache = _load_cache(settings.crawler_query_cache_path)
    wanted = _key(query)
    for value in cache.values():
        if _key(value.get("query") or "") == wanted:
            payload = json.dumps(
                {
                    "initial": value.get("initial"),
                    "adaptive": value.get("adaptive")
                    if settings.crawler_query_include_adaptive
                    else [],
                },
                sort_keys=True,
            )
            return hashlib.sha256(payload.encode()).hexdigest()[:12]
    return None
