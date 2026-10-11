"""Diagnose whether citation-expansion papers enter the candidate pool and
whether they match the gold answers for a PaSa query.

Minimal: run ONE query through the agent graph with citation expansion on,
then inspect:
  - how many candidates carry the openalex-neighborhood citation source
  - whether those citation papers are ranked into the final top-K
  - whether any gold answer title overlaps the citation papers

Run: python -m backend.scripts.diagnose_citation_recall --qid RealScholarQuery_1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.search_service import _get_paper_search_graph  # noqa: E402


def normalize_title(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def load_qrels() -> dict[str, list[str]]:
    path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    gold: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold[str(row.get("qid") or "")] = [
                normalize_title(t) for t in (row.get("answer") or [])
            ]
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", default="RealScholarQuery_1")
    parser.add_argument("--query", default="")
    args = parser.parse_args()

    gold_by_qid = load_qrels()
    if args.qid not in gold_by_qid:
        print(f"qid {args.qid} not in RealScholarQuery test.jsonl")
        return
    gold_titles = gold_by_qid[args.qid]
    query = args.query or " ".join(str(t) for t in gold_titles[:0]) or (
        # fall back to the first gold answer as a rough query if none passed
        " ".join(str(g) for g in []) or " ".join(gold_titles[:1])
    )
    # Prefer the stored question text from the jsonl for a realistic run.
    path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row.get("qid") or "") == args.qid:
                query = str(row.get("question") or query)
                break

    print(f"=== 诊断 qid={args.qid} ===")
    print(f"query: {query[:90]}")
    print(f"gold 标题数: {len(gold_titles)}")

    graph = _get_paper_search_graph()
    initial = {
        "raw_query": query,
        "max_papers": 5,
        "use_mock": False,
        "evaluation_fast_mode": True,
        "provider_cache_mode": "record",
    }
    final_state = {}
    for state in graph.stream(initial, stream_mode="values"):
        final_state = state

    candidates = list(final_state.get("candidates") or [])
    executed = list(final_state.get("executed_queries") or [])
    rounds = list(final_state.get("round_history") or [])
    print(f"检索轮次: {len(rounds)} | 执行查询: {executed}")

    # Count citation-sourced candidates (source names: openalex-cites /
    # openalex-referenced_works, or a citation_seed_id marker)
    citation_candidates = [
        c for c in candidates
        if c.get("citation_seed_id")
        or any(
            "openalex-" in str(s)
            for s in (c.get("sources") or [c.get("source") or ""])
        )
    ]
    print(f"候选池总数: {len(candidates)}")
    print(f"引文来源论文数: {len(citation_candidates)}")
    for c in citation_candidates[:5]:
        print(f"  [引文] {str(c.get('title'))[:60]} | seed={c.get('citation_seed_id')} dir={c.get('citation_direction')}")

    # Check how many citation papers are in the final top-K projection
    top_k_ids = {p.get("id") for p in final_state.get("ranked_papers") or []}
    cit_in_topk = [c for c in citation_candidates if c.get("id") in top_k_ids or c.get("canonical_paper_id") in top_k_ids]
    print(f"引文论文进入最终 top-k: {len(cit_in_topk)}/{len(citation_candidates)}")

    # Gold coverage: do gold titles match candidates? among citation papers?
    norm_candidates = {
        normalize_title(c.get("title")): c for c in candidates
    }
    gold_in_candidates = [g for g in gold_titles if g in norm_candidates]
    gold_in_citation = [
        g for g in gold_titles
        if any(normalize_title(c.get("title")) == g for c in citation_candidates)
    ]
    print(f"gold 命中候选池: {len(gold_in_candidates)}/{len(gold_titles)}")
    print(f"gold 命中引文论文: {len(gold_in_citation)}/{len(gold_titles)}")
    print(f"  (引文论文对 gold 的贡献: {len(gold_in_citation)} 篇)")
    if gold_in_citation:
        for g in gold_in_citation:
            print(f"    ✓ {g[:70]}")

    # Citation quality: how many citation papers are in the full candidate pool
    # (i.e. were they even retained after merge/budget?)
    print(f"\n引文论文在候选池中的保留率: {len(citation_candidates)}/57 (round history 报告的返回数)")

    # Persist the round citation detail for logging
    detail = []
    for i, rnd in enumerate(rounds):
        cit = (rnd.get("academic") or {}).get("citation_expansion") or {}
        detail.append({"round": i + 1, "citation": cit})
    out = ROOT / ".modeling-audit" / f"citation_diag_{args.qid}.json"
    out.write_text(
        json.dumps(
            {
                "qid": args.qid,
                "query": query,
                "candidate_total": len(candidates),
                "citation_count": len(citation_candidates),
                "citation_in_topk": len(cit_in_topk),
                "gold_total": len(gold_titles),
                "gold_in_candidates": len(gold_in_candidates),
                "gold_in_citation": len(gold_in_citation),
                "round_citation_detail": detail,
                "citation_titles": [str(c.get("title")) for c in citation_candidates[:20]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n诊断已保存: {out}")


if __name__ == "__main__":
    main()
