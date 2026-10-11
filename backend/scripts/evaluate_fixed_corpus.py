"""Evaluate retrieval/ranking on a frozen corpus with exact document qrels.

This evaluator is deliberately separate from the open-Provider paper test.
It never contacts academic Providers and never matches by title.  Every hit is
matched by the corpus document ID from qrels, so its scores are suitable for
controlled ablations on one immutable candidate universe.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.config import ROOT_DIR, get_settings
from backend.app.models.evidrank import RankingLevel
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.embeddings import get_bge_m3_encoder
from backend.app.services.ranking.adapters import candidates_from_rows
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.hybrid_ranker import HybridCandidateRanker
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.app.services.reranker import get_bge_reranker


PROFILES = ("fts", "bge_dense", "bge_rrf", "reranker", "evidrank")
PROFILE_SCOPES = {
    "fts": "full-corpus SQLite FTS5 retrieval",
    "bge_dense": "BGE-M3 dense reranking inside the FTS candidate pool",
    "bge_rrf": "FTS + BGE-M3 dense/sparse RRF inside the FTS candidate pool",
    "reranker": "BGE Reranker v2 M3 after in-pool BGE RRF",
    "evidrank": "full EvidRank pipeline inside the same FTS candidate pool",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "paper", "papers", "research", "that",
    "the", "their", "this", "to", "using", "what", "which", "with",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT_DIR, check=True, capture_output=True, text=True,
                timeout=3,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    digest = hashlib.sha256()
    code_files = sorted((ROOT_DIR / "backend" / "app").rglob("*.py"))
    code_files.append(Path(__file__).resolve())
    for path in code_files:
        digest.update(str(path.relative_to(ROOT_DIR)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"git_commit": commit, "git_dirty": dirty, "code_tree_sha256": digest.hexdigest()}


def _load_jsonl(path: Path, *, id_field: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get(id_field) or "").strip()
            if not identity:
                raise ValueError(f"{path.name} line {line_number}: missing {id_field}")
            if identity in values:
                raise ValueError(f"{path.name} line {line_number}: duplicate ID {identity}")
            values[identity] = row
    return values


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 3:
                raise ValueError(f"{path.name} line {line_number}: expected qid, docid, relevance")
            qid, doc_id, relevance = row[0].strip(), row[1].strip(), row[2].strip()
            if line_number == 1 and relevance.casefold() in {"score", "relevance"}:
                continue
            if not qid or not doc_id:
                raise ValueError(f"{path.name} line {line_number}: empty qid or docid")
            try:
                grade = int(float(relevance))
            except ValueError as exc:
                raise ValueError(f"{path.name} line {line_number}: invalid relevance") from exc
            qrels[qid][doc_id] = max(grade, qrels[qid].get(doc_id, grade))
    return dict(qrels)


class FixedCorpusFTS:
    def __init__(self, corpus_path: Path, index_path: Path) -> None:
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.corpus_sha256 = _sha256_file(corpus_path)
        self._ensure_index()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='corpus_sha256'"
            ).fetchone()
            ready = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents_fts'"
            ).fetchone()
            if row and row[0] == self.corpus_sha256 and ready:
                return
            connection.execute("DROP TABLE IF EXISTS documents_fts")
            connection.execute(
                "CREATE VIRTUAL TABLE documents_fts USING fts5("
                "doc_id UNINDEXED, title, body, tokenize='porter unicode61')"
            )
            batch: list[tuple[str, str, str]] = []
            for doc_id, row_data in _load_jsonl(self.corpus_path, id_field="_id").items():
                batch.append((doc_id, str(row_data.get("title") or ""), str(row_data.get("text") or "")))
                if len(batch) >= 1000:
                    connection.executemany(
                        "INSERT INTO documents_fts(doc_id,title,body) VALUES(?,?,?)", batch
                    )
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO documents_fts(doc_id,title,body) VALUES(?,?,?)", batch
                )
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('corpus_sha256',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.corpus_sha256,),
            )
            connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")

    @staticmethod
    def _query(value: str) -> str:
        terms = [
            token.casefold() for token in _TOKEN_RE.findall(value)
            if token.casefold() not in _STOPWORDS
        ]
        terms = list(dict.fromkeys(terms))[:24]
        if not terms:
            raise ValueError("query has no searchable terms")
        return " OR ".join(f'"{term}"' for term in terms)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id,title,body,bm25(documents_fts,0.0,2.0,1.0) AS score "
                "FROM documents_fts WHERE documents_fts MATCH ? ORDER BY score LIMIT ?",
                (self._query(query), limit),
            ).fetchall()
        return [dict(row) for row in rows]


def _rrf(paths: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for path in paths:
        for rank, doc_id in enumerate(dict.fromkeys(path), start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def _metric(ranked: list[str], qrels: dict[str, int], k: int) -> dict[str, float | int]:
    # A duplicated document is not a second retrieval hit.  Deduplicate before
    # truncating so malformed ranker output cannot inflate any metric.
    selected = list(dict.fromkeys(ranked))[:k]
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    hits = sum(doc_id in relevant for doc_id in selected)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    gains = [max(0, qrels.get(doc_id, 0)) for doc_id in selected]
    ideal = sorted((max(0, value) for value in qrels.values()), reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(values, 1))

    first = next((rank for rank, doc_id in enumerate(selected, 1) if doc_id in relevant), None)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ndcg": dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "hits": hits,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["doc_id"],
            "canonical_paper_id": row["doc_id"],
            "title": row["title"],
            "abstract": row["body"],
            "source": "fixed-corpus-fts",
            "sources": ["fixed-corpus-fts"],
            "provider_ranks": {"fixed-corpus-fts": rank},
            "provider_scores": {"fixed-corpus-fts": -float(row["score"])},
            "external_ids": {"fixed_corpus_id": row["doc_id"]},
        }
        for rank, row in enumerate(rows, start=1)
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    corpus_path = args.corpus.resolve()
    queries_path = args.queries.resolve()
    qrels_path = args.qrels.resolve()
    for path in (corpus_path, queries_path, qrels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    profiles = list(dict.fromkeys(args.profiles))
    invalid = sorted(set(profiles) - set(PROFILES))
    if invalid:
        raise ValueError(f"unsupported profiles: {invalid}")
    queries = _load_jsonl(queries_path, id_field="query_id")
    qrels = _load_qrels(qrels_path)
    eligible = sorted(set(queries) & set(qrels))
    if not eligible:
        raise ValueError("queries and qrels have no shared query IDs")
    if args.limit and args.limit < len(eligible):
        eligible = sorted(random.Random(args.seed).sample(eligible, args.limit))
    index_path = args.index or (
        ROOT_DIR / "backend" / "paper-test" / "index" /
        f"fixed_{hashlib.sha256(str(corpus_path).encode()).hexdigest()[:12]}.sqlite3"
    )
    fts = FixedCorpusFTS(corpus_path, index_path.resolve())
    needs_encoder = any(profile != "fts" for profile in profiles)
    needs_standalone_hybrid = any(
        profile in profiles for profile in ("bge_dense", "bge_rrf", "reranker")
    )
    # Load the PaSa selector BEFORE the BGE encoder/reranker: its 4-bit load
    # materialises the bf16 checkpoint (RAM spike bounded by config) and must
    # happen while memory is still free — loading it after the BGE models
    # segfaults on 16GB machines. No-op when disabled or already loaded.
    if "evidrank" in profiles and get_settings().pasa_selector_enabled:
        from backend.app.services.pasa_selector import get_pasa_selector

        get_pasa_selector()
    encoder = get_bge_m3_encoder() if needs_encoder else None
    hybrid = (
        HybridCandidateRanker(encoder)
        if encoder is not None and needs_standalone_hybrid
        else None
    )
    reranker = get_bge_reranker() if any(p in profiles for p in ("reranker", "evidrank")) else None
    pipeline = (
        EvidRankPipeline(encoder=encoder, reranker=reranker, evidence_judge=DeterministicEvidenceJudge())
        if "evidrank" in profiles else None
    )
    results: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    failures: list[dict[str, str]] = []
    for qid in eligible:
        query = str(queries[qid].get("query") or "").strip()
        started = time.perf_counter()
        try:
            fts_rows = fts.search(query, args.candidate_pool)
        except Exception as exc:  # FTS failure invalidates every in-pool profile.
            elapsed = (time.perf_counter() - started) * 1000
            failures.append(
                {
                    "qid": qid, "profile": "all", "stage": "fts",
                    "error_type": type(exc).__name__, "error": str(exc),
                }
            )
            for profile in profiles:
                results[profile].append(
                    {
                        "qid": qid, "query": query, "latency_ms": elapsed,
                        "candidate_recall": 0.0,
                        "metrics": {str(k): _metric([], qrels[qid], k) for k in args.ks},
                        "returned_ids": [], "error": str(exc), "error_stage": "fts",
                    }
                )
            continue

        fts_elapsed = (time.perf_counter() - started) * 1000
        profile_latency = {"fts": fts_elapsed}
        profile_errors: dict[str, tuple[str, Exception]] = {}
        candidates = _candidate_rows(fts_rows)
        fts_ids = [row["doc_id"] for row in fts_rows]
        fts_id_set = set(fts_ids)
        orders: dict[str, list[str]] = {"fts": fts_ids}
        if needs_standalone_hybrid:
            stage_started = time.perf_counter()
            try:
                if needs_standalone_hybrid:
                    assert hybrid is not None
                    papers = candidates_from_rows(candidates)
                    rrf_order, score_map, _meta = hybrid.rank(query, papers)
                    bge_elapsed = (time.perf_counter() - stage_started) * 1000
                    profile_latency["bge_dense"] = fts_elapsed + bge_elapsed
                    profile_latency["bge_rrf"] = fts_elapsed + bge_elapsed
                    orders["bge_dense"] = [
                        item.canonical_paper_id
                        for item in sorted(
                            papers,
                            key=lambda item: (
                                score_map[item.canonical_paper_id].dense_rank or 10**9,
                                item.canonical_paper_id,
                            ),
                        )
                    ]
                    orders["bge_rrf"] = [item.canonical_paper_id for item in rrf_order]
            except Exception as exc:
                elapsed = fts_elapsed + (time.perf_counter() - stage_started) * 1000
                for profile in ("bge_dense", "bge_rrf", "reranker"):
                    if profile in profiles:
                        profile_latency[profile] = elapsed
                        profile_errors[profile] = ("bge_hybrid", exc)

        if "reranker" in profiles and "reranker" not in profile_errors:
            assert reranker is not None
            stage_started = time.perf_counter()
            try:
                candidate_by_id = {row["id"]: row for row in candidates}
                rerank_input = [
                    {
                        "id": doc_id,
                        "title": candidate_by_id[doc_id]["title"],
                        "abstract": candidate_by_id[doc_id]["abstract"],
                        "rrf_score": 1.0 / (60 + rank),
                    }
                    for rank, doc_id in enumerate(orders["bge_rrf"], start=1)
                ]
                reranked, _meta = reranker.rerank(
                    query, rerank_input, len(rerank_input), candidate_limit=args.reranker_pool
                )
                orders["reranker"] = [str(row["id"]) for row in reranked]
                profile_latency["reranker"] = (
                    profile_latency["bge_rrf"]
                    + (time.perf_counter() - stage_started) * 1000
                )
            except Exception as exc:
                profile_latency["reranker"] = (
                    profile_latency["bge_rrf"]
                    + (time.perf_counter() - stage_started) * 1000
                )
                profile_errors["reranker"] = ("reranker", exc)

        if "evidrank" in profiles:
            assert pipeline is not None
            stage_started = time.perf_counter()
            try:
                understanding = QueryUnderstanding(
                    intent=f"Retrieve papers relevant to {query}", topics=[query],
                    keywords=_TOKEN_RE.findall(query), sub_queries=[query],
                    rewritten_queries=[query],
                    raw_json={"parser_version": "fixed-corpus-eval-v1"},
                )
                bundle = pipeline.rank(
                    session_id=f"fixed-{qid}", query=query, understanding=understanding,
                    candidate_rows=candidates, top_k=max(args.ks), fast_evaluation=True,
                )
                orders["evidrank"] = [
                    item.paper_id for item in bundle.items
                    if item.level != RankingLevel.EXCLUDED
                ]
                # EvidRank performs its own hybrid/reranker stages.  Do not add
                # time spent evaluating sibling ablation profiles.
                profile_latency["evidrank"] = (
                    fts_elapsed + (time.perf_counter() - stage_started) * 1000
                )
            except Exception as exc:
                profile_latency["evidrank"] = (
                    fts_elapsed + (time.perf_counter() - stage_started) * 1000
                )
                profile_errors["evidrank"] = ("evidrank", exc)

        relevant = qrels[qid]
        candidate_recall = (
            sum(doc_id in fts_id_set for doc_id, grade in relevant.items() if grade > 0)
            / max(1, sum(grade > 0 for grade in relevant.values()))
        )
        for profile in profiles:
            if profile in profile_errors:
                stage, exc = profile_errors[profile]
                failures.append(
                    {
                        "qid": qid, "profile": profile, "stage": stage,
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                )
                results[profile].append(
                    {
                        "qid": qid, "query": query, "latency_ms": profile_latency[profile],
                        "candidate_recall": candidate_recall,
                        "metrics": {str(k): _metric([], relevant, k) for k in args.ks},
                        "returned_ids": [], "error": str(exc), "error_stage": stage,
                    }
                )
            else:
                results[profile].append(
                    {
                        "qid": qid,
                        "query": query,
                        "latency_ms": profile_latency[profile],
                        "candidate_recall": candidate_recall,
                        "metrics": {str(k): _metric(orders[profile], relevant, k) for k in args.ks},
                        "returned_ids": orders[profile][: max(args.ks)],
                    }
                )
    aggregate: dict[str, Any] = {}
    for profile, rows in results.items():
        latencies = [float(row["latency_ms"]) for row in rows]
        aggregate[profile] = {
            "query_count": len(rows),
            "failed_queries": sum("error" in row for row in rows),
            "candidate_recall": statistics.fmean(float(row["candidate_recall"]) for row in rows),
            "mean_latency_ms": statistics.fmean(latencies),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "metrics": {
                str(k): {
                    name: (
                        sum(float(row["metrics"][str(k)][name]) for row in rows) / len(rows)
                        if name != "hits" else sum(int(row["metrics"][str(k)][name]) for row in rows)
                    )
                    for name in ("precision", "recall", "f1", "ndcg", "mrr", "hits")
                }
                for k in args.ks
            },
        }
    settings = get_settings()
    return {
        "schema_version": "fixed-corpus-evaluation-v2",
        "evaluation_protocol": "frozen-corpus-exact-document-id-qrels",
        "leaderboard_comparable": False,
        "leaderboard_note": "Comparable only when corpus/query/qrels preprocessing matches the target benchmark exactly.",
        "tuning_allowed": args.split != "test",
        "split": args.split,
        "seed": args.seed,
        "ks": args.ks,
        "candidate_pool": args.candidate_pool,
        "reranker_pool": args.reranker_pool,
        "profile_scopes": {profile: PROFILE_SCOPES[profile] for profile in profiles},
        "retrieval_scope_note": (
            "Only FTS searches the complete corpus in this evaluator. BGE and EvidRank profiles "
            "rerank that frozen FTS pool; use build_pasa_vector_index.py for full-corpus BGE recall."
        ),
        "sample_qids_sha256": hashlib.sha256(json.dumps(eligible).encode()).hexdigest(),
        "data": {
            "corpus": str(corpus_path), "corpus_sha256": _sha256_file(corpus_path),
            "queries": str(queries_path), "queries_sha256": _sha256_file(queries_path),
            "qrels": str(qrels_path), "qrels_sha256": _sha256_file(qrels_path),
            "index": str(index_path.resolve()),
        },
        "models": {
            "bge_m3": {"path": settings.bge_m3_model_path, "revision": settings.bge_m3_model_revision},
            "reranker": {"path": settings.bge_reranker_model_path, "revision": settings.bge_reranker_model_revision},
        },
        "code": _code_state(),
        "aggregate": aggregate,
        "failures": failures,
        "queries": results,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--profiles", nargs="+", default=["fts"], choices=PROFILES)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every qrel query")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", dest="ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--reranker-pool", type=int, default=80)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.ks = sorted(set(args.ks))
    if any(k <= 0 for k in args.ks):
        parser.error("--k values must be positive")
    if args.limit < 0 or args.candidate_pool < max(args.ks) or args.reranker_pool <= 0:
        parser.error("invalid limit/candidate/reranker pool")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
