"""Leakage analysis for the PaSa evaluation with the PaSa corpus.

Two experiments:

1. Masking / content-driven check: the faiss index is a brute-force scan, so
   every paper's score is independent of the others. Retrieving then masking
   the query's gold papers is therefore equivalent to removing them from the
   index. If retrieval is genuinely content-driven, the masked results should
   still be topically relevant papers (not noise) -- and gold papers were
   recalled purely because of their own similarity score.

2. Query-answer lexical overlap: how much of each test query's vocabulary is
   literally covered by its gold papers' titles/abstracts, vs random papers.
   High gold coverage means the answers are easy to find lexically, i.e. the
   dataset's construction inflates retrieval numbers.

Run: python -m backend.scripts.leakage_analysis --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.schemas import QueryUnderstanding  # noqa: E402
from backend.app.services.local_vector_corpus import PaSaVectorCorpusStore  # noqa: E402


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+._-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "paper", "papers", "research", "that",
    "the", "their", "this", "to", "using", "what", "which", "with",
    "how", "can", "show", "me", "provide", "list", "give", "some",
}


def _tokens(value: str) -> list[str]:
    return [
        t.casefold()
        for t in _WORD_RE.findall(value)
        if t.casefold() not in _STOPWORDS and len(t) > 2
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--pool", type=int, default=200)
    args = parser.parse_args()
    _safe_stdout()

    test_path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    rows = [json.loads(line) for line in test_path.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    store = PaSaVectorCorpusStore()
    store.ensure_collection()
    fb = store._faiss
    print(f"index: {len(fb.docs)} docs", flush=True)
    doc_keys = [_key(row.get("title")) for row in fb.docs]

    # Collect gold doc indices per query (by title key).
    query_gold: list[tuple[dict, set[int]]] = []
    for row in rows:
        gold_titles = [g for g in (row.get("answer") or []) if g]
        if not gold_titles:
            continue
        gold_keys = {_key(g) for g in gold_titles}
        gold_idx = {i for i, k in enumerate(doc_keys) if k in gold_keys}
        query_gold.append((row, gold_idx))

    encoder = store.encoder

    # ------------------------------------------------------------------
    print("\n=== 实验1: masking / 内容驱动验证 (pool=200, 屏蔽 gold 后 top-5) ===")
    print(f"{'qid':<20} gold_in_pool  masked_top5_still_relevant")
    for row, gold_idx in query_gold:
        query = row.get("question") or ""
        dense_q, _sparse_q, _ = encoder.encode_query(query)
        vector = np.asarray([dense_q], dtype=np.float32)
        import faiss
        faiss.normalize_L2(vector)
        distances, indices = fb.dense.search(vector, args.pool)
        pool_idx = [int(i) for i in indices[0] if i >= 0]
        gold_hits = len(set(pool_idx) & gold_idx)
        masked = [i for i in pool_idx if i not in gold_idx][:5]
        print(f"{str(row.get('qid')):<20} {gold_hits:>5}    " + " | ".join(fb.docs[i]["title"][:38] for i in masked))

    # ------------------------------------------------------------------
    print("\n=== 实验2: 查询-gold 字面重合度 (查询 token 被标题/摘要覆盖的比例) ===")
    print("(gold=测试答案; rand=随机论文对照; overlap = |query_tokens & doc_tokens| / |query_tokens|)")
    print(f"{'qid':<20} {'q_terms':>8} {'gold_t':>8} {'gold_a':>8} {'rand_t':>8}")

    # Random baseline: sample 200 random docs once, reuse for every query.
    rng = np.random.RandomState(7)
    random_indices = rng.choice(len(fb.docs), size=200, replace=False)
    random_tokens = [_tokens(fb.docs[i]["title"]) for i in random_indices]

    gold_title_coverage: list[float] = []
    gold_abs_coverage: list[float] = []
    rand_coverage: list[float] = []

    for row, gold_idx in query_gold:
        query = row.get("question") or ""
        query_tokens = set(_tokens(query))
        if not query_tokens:
            continue
        gold_abs = set()
        for i in gold_idx:
            gold_abs.update(_tokens(fb.docs[i].get("abstract") or ""))
        gold_t = set()
        for i in gold_idx:
            gold_t.update(_tokens(fb.docs[i].get("title") or ""))
        g_t_cov = len(query_tokens & gold_t) / len(query_tokens)
        g_a_cov = len(query_tokens & gold_abs) / len(query_tokens)
        r_cov = sum(
            len(query_tokens & set(rt)) / len(query_tokens) for rt in random_tokens
        ) / len(random_tokens)
        gold_title_coverage.append(g_t_cov)
        gold_abs_coverage.append(g_a_cov)
        rand_coverage.append(r_cov)
        print(
            f"{str(row.get('qid')):<20} {len(query_tokens):>8} {g_t_cov:>8.2f} {g_a_cov:>8.2f} {r_cov:>8.2f}"
        )

    if gold_title_coverage:
        print("\n汇总 (平均):")
        print(f"  gold 标题覆盖查询词汇: {sum(gold_title_coverage)/len(gold_title_coverage)*100:.1f}%")
        print(f"  gold 摘要覆盖查询词汇: {sum(gold_abs_coverage)/len(gold_abs_coverage)*100:.1f}%")
        print(f"  随机论文标题覆盖      : {sum(rand_coverage)/len(rand_coverage)*100:.1f}%")
        ratio = (sum(gold_title_coverage) / len(gold_title_coverage)) / max(
            1e-9, sum(rand_coverage) / len(rand_coverage)
        )
        print(f"  gold/随机 比值        : {ratio:.1f}x")

    out = ROOT / ".modeling-audit" / "leakage_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "gold_title_coverage_avg": (
                    sum(gold_title_coverage) / len(gold_title_coverage) if gold_title_coverage else 0
                ),
                "gold_abs_coverage_avg": (
                    sum(gold_abs_coverage) / len(gold_abs_coverage) if gold_abs_coverage else 0
                ),
                "random_coverage_avg": (
                    sum(rand_coverage) / len(rand_coverage) if rand_coverage else 0
                ),
                "queries": len(query_gold),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
