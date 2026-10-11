"""Compare candidate recall (gold papers entering the top-K pool) between the
FTS local corpus and the new full-corpus faiss vector index, on the real
PaSa evaluation queries.

For each query this checks whether the gold answer titles appear in the
top-K pool returned by each retrieval source. This is the "retrieval stage"
diagnostic: it isolates whether the correct papers are being found at all
(before ranking/reranking).

Run: python -m backend.scripts.eval_pasa_real_candidate_recall --limit 50 --pool 200
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
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
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pasa-real")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--pool", type=int, default=200)
    args = parser.parse_args()
    _safe_stdout()

    dataset_root = ROOT / "backend" / "paper-test" / "datasets"
    test_path = dataset_root / "RealScholarQuery" / "test.jsonl"
    if args.dataset == "pasa-auto":
        test_path = dataset_root / "AutoScholarQuery" / "test.jsonl"
    rows = []
    with test_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]

    # Use a lexical-fallback understanding (like the benchmark path in
    # search_service) so the multi-variant vector retrieval is exercised the
    # same way the real evaluation does -- an empty understanding would
    # silently degrade multi-variant retrieval to the single raw query.
    fts_hits = 0
    vec_hits = 0
    gold_total = 0
    queries_with_gold = 0
    details: list[dict] = []

    for row in rows:
        query = row.get("question") or row.get("query") or ""
        gold = [g for g in (row.get("answer") or []) if g]
        if not gold:
            continue
        gold_keys = {_key(g) for g in gold}
        queries_with_gold += 1
        gold_total += len(gold)

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
            query, understanding, limit=args.pool, offset=0
        )
        vec_papers, vec_meta = search_local_vector_corpus(
            query, understanding, limit=args.pool, offset=0
        )
        fts_keys = {_key(p.get("title")) for p in fts_papers}
        vec_keys = {_key(p.get("title")) for p in vec_papers}
        fts_found = len(gold_keys & fts_keys)
        vec_found = len(gold_keys & vec_keys)
        both = len(gold_keys & fts_keys & vec_keys)
        fts_hits += fts_found
        vec_hits += vec_found
        details.append(
            {
                "qid": row.get("qid") or row.get("id"),
                "query": query[:60],
                "gold_count": len(gold),
                "fts_found": fts_found,
                "vec_found": vec_found,
                "gold_in_both": both,
                "vec_only": vec_found - both,
                "fts_only": fts_found - both,
            }
        )

    print(f"=== {args.dataset} candidate recall (pool={args.pool}, "
          f"{queries_with_gold} queries, {gold_total} gold papers) ===")
    print(f"FTS   gold-in-pool: {fts_hits}/{gold_total} = {fts_hits / gold_total * 100:.1f}%")
    print(f"VEC   gold-in-pool: {vec_hits}/{gold_total} = {vec_hits / gold_total * 100:.1f}%")
    print()
    print("per-query:")
    for d in details:
        flag = "VEC" if d["vec_found"] > d["fts_found"] else ("FTS" if d["fts_found"] > d["vec_found"] else "=")
        print(f"  {str(d['qid']):<20} gold={d['gold_count']} fts={d['fts_found']} vec={d['vec_found']} "
              f"vec_only={d['vec_only']} fts_only={d['fts_only']}  [{flag}]")

    out = ROOT / ".modeling-audit" / f"{args.dataset}_candidate_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "pool": args.pool,
                "queries": queries_with_gold,
                "gold_total": gold_total,
                "fts_found": fts_hits,
                "vec_found": vec_hits,
                "fts_recall": round(fts_hits / gold_total, 4) if gold_total else 0,
                "vec_recall": round(vec_hits / gold_total, 4) if gold_total else 0,
                "per_query": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
