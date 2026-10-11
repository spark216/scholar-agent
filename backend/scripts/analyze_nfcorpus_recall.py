"""Analyze why relevant NFCorpus documents miss the FTS top-K candidate pool.

Classifies each missed relevant document into:
  - lexical_gap:   relevant doc shares ZERO query tokens in title+text
  - low_overlap:   relevant doc shares few query tokens (ranked below pool)
  - truncated:     shares tokens but BM25 ranked it past the candidate pool
  - no_query_terms: query had no searchable terms after stopword filtering

Run: python -m backend.scripts.analyze_nfcorpus_recall --split test --pool 200
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import ir_datasets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "paper", "papers", "research", "that",
    "the", "their", "this", "to", "using", "what", "which", "with",
}


def _tokens(value: str) -> set[str]:
    return {
        t.casefold()
        for t in _TOKEN_RE.findall(value)
        if t.casefold() not in _STOPWORDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--pool", type=int, default=200)
    args = parser.parse_args()

    data_dir = ROOT / "backend" / "paper-test" / "datasets" / "nfcorpus-fixed"
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / f"queries-{args.split}.jsonl"
    qrels_path = data_dir / f"qrels-{args.split}.tsv"

    # Build corpus token sets lazily (only for relevant docs + fetched docs).
    corpus: dict[str, dict[str, object]] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = row

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["query"]

    # Build FTS index like evaluate_fixed_corpus.
    index_path = data_dir / f"_fts_{args.split}.sqlite3"
    if index_path.exists():
        index_path.unlink()
    conn = sqlite3.connect(index_path)
    conn.execute(
        "CREATE VIRTUAL TABLE documents_fts USING fts5("
        "doc_id UNINDEXED, title, body, tokenize='porter unicode61')"
    )
    for row in corpus.values():
        conn.execute(
            "INSERT INTO documents_fts(doc_id,title,body) VALUES(?,?,?)",
            (row["_id"], str(row.get("title") or ""), str(row.get("text") or "")),
        )
    conn.commit()

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_path.open(encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[2].casefold() in {"relevance", "score"}:
                continue
            qid, docid, rel = parts[0], parts[1], parts[2]
            qrels[qid][docid] = max(int(rel), qrels[qid].get(docid, int(rel)))

    category_counter: Counter[str] = Counter()
    missed_total = 0
    relevant_total = 0
    examples: dict[str, list[dict]] = defaultdict(list)

    for qid, query_text in queries.items():
        relevant_docs = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        if not relevant_docs:
            continue
        relevant_total += len(relevant_docs)

        terms = [
            t.casefold() for t in _TOKEN_RE.findall(query_text)
            if t.casefold() not in _STOPWORDS
        ]
        terms = list(dict.fromkeys(terms))[:24]
        if not terms:
            for docid in relevant_docs:
                missed_total += 1
                category_counter["no_query_terms"] += 1
                examples["no_query_terms"].append({"qid": qid, "doc": docid})
            continue
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        fetched = conn.execute(
            "SELECT doc_id FROM documents_fts WHERE documents_fts MATCH ? "
            "ORDER BY bm25(documents_fts,0.0,2.0,1.0) LIMIT ?",
            (fts_query, args.pool),
        ).fetchall()
        fetched_ids = {row[0] for row in fetched}
        query_tokens = set(terms)

        for docid in relevant_docs:
            if docid in fetched_ids:
                continue
            missed_total += 1
            doc = corpus.get(docid, {})
            doc_tokens = _tokens(
                f"{doc.get('title') or ''} {doc.get('text') or ''}"
            )
            overlap = doc_tokens & query_tokens
            if not overlap:
                category_counter["lexical_gap"] += 1
                examples["lexical_gap"].append(
                    {"qid": qid, "doc": docid, "overlap": sorted(overlap), "query": query_text[:80]}
                )
            elif len(overlap) < max(1, len(query_tokens) // 2):
                category_counter["low_overlap"] += 1
                examples["low_overlap"].append(
                    {"qid": qid, "doc": docid, "overlap": sorted(overlap), "query": query_text[:80]}
                )
            else:
                category_counter["truncated"] += 1
                examples["truncated"].append(
                    {"qid": qid, "doc": docid, "overlap": sorted(overlap), "query": query_text[:80]}
                )

    print(f"=== NFCorpus {args.split} 召回失败分析 (pool={args.pool}) ===")
    print(f"相关文档总数: {relevant_total} | 未进入候选池: {missed_total} "
          f"({missed_total / relevant_total * 100:.1f}%)")
    print(f"进入候选池: {relevant_total - missed_total} "
          f"({(relevant_total - missed_total) / relevant_total * 100:.1f}%)")
    print()
    print("=== 失败原因分类 ===")
    for cat, count in category_counter.most_common():
        print(f"  {cat:15s} {count:5d} ({count / max(1, missed_total) * 100:.1f}%)")

    # Show a few concrete examples per category
    for cat in ("lexical_gap", "low_overlap", "truncated", "no_query_terms"):
        ex = examples.get(cat, [])[:3]
        if ex:
            print(f"\n  [{cat} 示例]")
            for e in ex:
                print(f"    qid={e['qid']} doc={e['doc']} overlap={e.get('overlap', [])}")

    # Persist analysis
    out = ROOT / ".modeling-audit" / f"nfcorpus_{args.split}_recall_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "pool": args.pool,
        "relevant_total": relevant_total,
        "missed": missed_total,
        "recall": (relevant_total - missed_total) / max(1, relevant_total),
        "categories": dict(category_counter),
        "examples": {k: v[:10] for k, v in examples.items()},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分析已保存: {out}")


if __name__ == "__main__":
    main()
