"""Build the SPARBench baseline and fused candidate pools for ranking eval.

For each of the 50 SPAR queries:
  * baseline pool = FTS + faiss-vector with the lexical-fallback understanding
    (identical to the diagnostic's baseline in eval_spar_crawler_vector.py)
  * fused pool   = baseline pool UNION crawler-understanding pool
    (the "fused" gold-in-pool pool from the diagnostic: 78 -> 93 gold)

Every pool entry carries the REAL arXiv id (FTS papers expose it via
external_ids.arxiv, vector papers via arxiv_id -- the "id" field is a stable
sha256 and must never be used as an identity key).

Gold answers are paired per source_meta.answers[] entry: gold_titles[i] and
gold_arxiv_ids[i] refer to the SAME answer, so title-only and arxiv-id matches
never double count (mirrors eval_pasa_selector_rerank.relevance_flags).

Output (same shape as build_selector_pool.py):
  .modeling-audit/spar_baseline_pool.json
  .modeling-audit/spar_fused_pool.json
  { "pool": 200, "queries": [{qid, query, gold_titles, gold_arxiv_ids,
                               pool: [{arxiv_id, title, abstract, sources}]}],
    "gold_total", "gold_in_pool" }

Run: python -m backend.scripts.build_spar_pools
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.schemas import QueryUnderstanding  # noqa: E402
from backend.app.services.local_pasa_corpus import search_local_pasa_corpus  # noqa: E402
from backend.app.services.local_vector_corpus import search_local_vector_corpus  # noqa: E402

SPAR_PATH = ROOT / "dataset" / "SPAR_Benchmark.jsonl"
CRAWLER_CACHE = ROOT / ".modeling-audit" / "spar_crawler_queries_cache.json"
OUT_BASELINE = ROOT / ".modeling-audit" / "spar_baseline_pool.json"
OUT_FUSED = ROOT / ".modeling-audit" / "spar_fused_pool.json"
_POOL_LIMIT = 200
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _norm_arxiv(pid: str) -> str:
    p = str(pid or "").strip().casefold()
    p = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", p)
    p = p.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", p)


def _paper_arxiv_id(paper: dict) -> str:
    return str(
        paper.get("arxiv_id")
        or (paper.get("external_ids") or {}).get("arxiv")
        or ""
    ).casefold()


def _fallback_understanding(query: str) -> QueryUnderstanding:
    return QueryUnderstanding(
        intent=f"检索与 {query} 相关的学术论文",
        is_paper_search=True,
        language="zh+en",
        topics=[query],
        keywords=[query],
        sub_queries=[query],
        rewritten_queries=[query, f"{query} academic papers"],
    )


def _crawler_understanding(query: str, crawler_queries: list[str]) -> QueryUnderstanding:
    queries = list(crawler_queries) or [query]
    return QueryUnderstanding(
        intent=f"检索与 {query} 相关的学术论文",
        is_paper_search=True,
        language="zh+en",
        topics=[query],
        keywords=queries,
        sub_queries=queries,
        rewritten_queries=queries,
    )


def _search_pool(query: str, understanding: QueryUnderstanding) -> list[dict]:
    fts_papers, _ = search_local_pasa_corpus(
        query, understanding, limit=_POOL_LIMIT, offset=0
    )
    vec_papers, _ = search_local_vector_corpus(
        query, understanding, limit=_POOL_LIMIT, offset=0
    )
    merged: dict[str, dict] = {}
    for source, papers in (("fts", fts_papers), ("vec", vec_papers)):
        for paper in papers:
            key = _paper_arxiv_id(paper)
            if not key:
                continue
            entry = merged.get(key)
            if entry is None:
                merged[key] = {
                    "arxiv_id": key,
                    "title": str(paper.get("title") or ""),
                    "abstract": str(paper.get("abstract") or ""),
                    "sources": [source],
                }
            else:
                entry["sources"].append(source)
                if not entry["abstract"]:
                    entry["abstract"] = str(paper.get("abstract") or "")
    return list(merged.values())


def _merge_pools(*pools: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for pool in pools:
        for paper in pool:
            key = paper["arxiv_id"]
            if key not in merged:
                merged[key] = dict(paper)
    return list(merged.values())


def _gold_answers(row: dict) -> tuple[list[str], list[str]]:
    """Index-aligned gold title / arxiv-id lists (per answer, no cross-pairing)."""
    titles: list[str] = []
    arxiv_ids: list[str] = []
    for answer in (row.get("source_meta") or {}).get("answers") or []:
        titles.append(str(answer.get("title") or ""))
        pid = str(answer.get("paperID") or "")
        if "arxiv" in pid.lower() or _ARXIV_RE.match(pid):
            arxiv_ids.append(_norm_arxiv(pid))
        else:
            arxiv_ids.append("")
    return titles, arxiv_ids


def main() -> None:
    _safe_stdout()
    cache = json.loads(CRAWLER_CACHE.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in SPAR_PATH.open(encoding="utf-8") if line.strip()]
    print(f"loaded {len(rows)} SPAR queries, crawler cache has {len(cache)}", flush=True)

    out = {"baseline": [], "fused": []}
    totals = {"baseline": {"gold": 0, "in_pool": 0}, "fused": {"gold": 0, "in_pool": 0}}
    pool_gold_keys = {"baseline": set(), "fused": set()}  # arxiv-id keys of gold found

    for index, row in enumerate(rows, start=1):
        qid = f"SPARBench_{index}"
        query = str(row.get("question") or "")
        gold_titles, gold_arxiv_ids = _gold_answers(row)
        crawler_queries = (cache.get(qid) or {}).get("initial") or []

        base_pool = _search_pool(query, _fallback_understanding(query))
        fused_pool = base_pool
        if crawler_queries:
            crawler_pool = _search_pool(query, _crawler_understanding(query, crawler_queries))
            fused_pool = _merge_pools(base_pool, crawler_pool)

        gold_ids = {a for a in gold_arxiv_ids if a}
        for kind, pool in (("baseline", base_pool), ("fused", fused_pool)):
            found = sum(
                1
                for title, arxiv_id in zip(gold_titles, gold_arxiv_ids, strict=True)
                if any(
                    (arxiv_id and p["arxiv_id"] == arxiv_id)
                    or (title and p["title"].strip().casefold() == title.strip().casefold())
                    for p in pool
                )
            )
            totals[kind]["gold"] += len(gold_titles)
            totals[kind]["in_pool"] += found
            out[kind].append(
                {
                    "qid": qid,
                    "query": query,
                    "gold_titles": gold_titles,
                    "gold_arxiv_ids": gold_arxiv_ids,
                    "pool": pool,
                }
            )
        print(
            f"{qid}: base_pool={len(base_pool)} fused_pool={len(fused_pool)} "
            f"gold={len(gold_titles)}",
            flush=True,
        )

    # re-print clean per-query line (the above expression is only for the summary)
    for kind, path in (("baseline", OUT_BASELINE), ("fused", OUT_FUSED)):
        payload = {
            "pool": _POOL_LIMIT,
            "queries": out[kind],
            "gold_total": totals[kind]["gold"],
            "gold_in_pool": totals[kind]["in_pool"],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(
            f"[{kind}] gold_in_pool={totals[kind]['in_pool']}/{totals[kind]['gold']} "
            f"({totals[kind]['in_pool'] / totals[kind]['gold'] * 100:.1f}%) -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
