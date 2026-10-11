"""Build the candidate paper pool (FTS + full-corpus faiss vector, union) for
the PaSa benchmark queries, capturing title + abstract for every candidate.

This is the retrieval-stage input that a reranker (bge-reranker today, or the
official pasa-7b-selector under test) would rank.  The pool is built the same
way as eval_pasa_real_candidate_recall.py (lexical-fallback understanding) so
the "gold-in-pool" ceiling is comparable to the documented candidate-recall
numbers.

Output: .modeling-audit/selector_test_pool.json
  { "queries": [ {qid, query, gold_titles, gold_arxiv_ids,
                   pool: [{arxiv_id, title, abstract}] } ] }

Run: python -m backend.scripts.build_selector_pool --limit 5 --seed 42 --pool 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.schemas import QueryUnderstanding  # noqa: E402
from backend.app.services.local_pasa_corpus import search_local_pasa_corpus  # noqa: E402
from backend.app.services.local_vector_corpus import search_local_vector_corpus  # noqa: E402


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _key(value: str) -> str:
    """NFKD + alnum-normalized key (mirrors eval_pasa_real_candidate_recall)."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(c for c in normalized if c.isalnum())


def _stable_id(paper: dict) -> str:
    return str(paper.get("id") or paper.get("arxiv_id") or "").casefold()


def build_pool(query: str, pool: int) -> list[dict]:
    understanding = QueryUnderstanding(
        intent=f"检索与 {query} 相关的学术论文",
        is_paper_search=True,
        language="zh+en",
        topics=[query],
        keywords=[query],
        sub_queries=[query],
        rewritten_queries=[query, f"{query} academic papers"],
    )
    fts_papers, fts_meta = search_local_pasa_corpus(
        query, understanding, limit=pool, offset=0
    )
    vec_papers, vec_meta = search_local_vector_corpus(
        query, understanding, limit=pool, offset=0
    )
    merged: dict[str, dict] = {}
    for source, papers in (("fts", fts_papers), ("vec", vec_papers)):
        for paper in papers:
            key = _stable_id(paper)
            if not key:
                continue
            if key not in merged:
                merged[key] = {
                    "arxiv_id": key,
                    "title": str(paper.get("title") or ""),
                    "abstract": str(paper.get("abstract") or ""),
                    "sources": [source],
                }
            else:
                merged[key]["sources"].append(source)
                if not merged[key]["abstract"]:
                    merged[key]["abstract"] = str(paper.get("abstract") or "")
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", type=int, default=200)
    args = parser.parse_args()
    _safe_stdout()

    test_path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    samples = [json.loads(line) for line in test_path.open(encoding="utf-8") if line.strip()]
    if args.limit < len(samples):
        indices = sorted(random.Random(args.seed).sample(range(len(samples)), args.limit))
        samples = [samples[i] for i in indices]

    queries_out = []
    gold_total = 0
    gold_in_pool = 0
    for row in samples:
        qid = row["qid"]
        query = row["question"]
        gold_titles = [g for g in (row.get("answer") or []) if g]
        gold_arxiv_ids = [g for g in (row.get("answer_arxiv_id") or []) if g]
        pool = build_pool(query, args.pool)
        gold_total += len(gold_titles)
        gold_title_keys = {_key(t) for t in gold_titles}
        gold_id_keys = {(str(a or "").casefold()) for a in gold_arxiv_ids}
        pool_title_keys = {_key(p["title"]) for p in pool}
        pool_id_keys = {p["arxiv_id"] for p in pool}
        found = len(gold_title_keys & pool_title_keys) + len(gold_id_keys & pool_id_keys)
        gold_in_pool += found
        print(
            f"{qid}: pool={len(pool)} gold={len(gold_titles)} gold_in_pool={found} "
            f"({found / len(gold_titles) * 100:.1f}%)",
            flush=True,
        )
        queries_out.append(
            {
                "qid": qid,
                "query": query,
                "gold_titles": gold_titles,
                "gold_arxiv_ids": gold_arxiv_ids,
                "pool": pool,
            }
        )

    print(
        f"=== pool recall ceiling: {gold_in_pool}/{gold_total} = "
        f"{gold_in_pool / gold_total * 100:.1f}% ===",
        flush=True,
    )
    out = ROOT / ".modeling-audit" / "selector_test_pool.json"
    out.write_text(
        json.dumps(
            {"pool": args.pool, "queries": queries_out, "gold_total": gold_total,
             "gold_in_pool": gold_in_pool},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
