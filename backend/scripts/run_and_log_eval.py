"""Run an open-provider paper test and append a detailed record to the eval log.

Every run is logged as one JSON line in .modeling-audit/eval_log.jsonl with
timestamps, configuration, per-query metrics, API/token usage, latency and
citation-expansion detail, so results can be reused as report evidence.

Usage (minimal):
    python -m backend.scripts.run_and_log_eval --dataset pasa-real --limit 3
    python -m backend.scripts.run_and_log_eval --dataset pasa-real --limit 3 --citation-off
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.models.paper_test import PaperTestRequest  # noqa: E402
from backend.app.services.paper_test import run_paper_test  # noqa: E402

LOG_PATH = ROOT / ".modeling-audit" / "eval_log.jsonl"


def _git_state() -> tuple[str | None, bool | None]:
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT, check=True, capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        )
        return commit or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pasa-real")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--citation-off", action="store_true",
                        help="disable citation expansion for this run")
    parser.add_argument("--prf-off", action="store_true",
                        help="disable PRF second-round keyword search (baseline)")
    parser.add_argument("--tag", default="", help="free-text tag for the log")
    args = parser.parse_args()

    settings = get_settings()
    if args.citation_off:
        # Temporarily disable citation expansion for the A/B comparison.
        settings.semantic_scholar_citation_expansion_every_round = False
    if args.prf_off:
        settings.agent_search_prf_enabled = False

    started = time.time()
    report = run_paper_test(
        PaperTestRequest(
            dataset=args.dataset,
            sample_limit=args.limit,
            ks=sorted(set(args.k)),
            seed=args.seed,
            use_mock=False,
            provider_cache_mode="record",
        )
    )
    elapsed = time.time() - started

    commit, dirty = _git_state()
    entry = {
        "logged_at": datetime.now(UTC).isoformat(),
        "elapsed_s": round(elapsed, 2),
        "dataset": args.dataset,
        "sample_limit": args.limit,
        "ks": sorted(set(args.k)),
        "seed": args.seed,
        "tag": args.tag,
        "citation_enabled": not args.citation_off,
        "run_id": report.run_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "model": {"bge_m3": settings.bge_m3_model_path, "reranker": settings.bge_reranker_model_path},
        "config_fingerprint": (report.run_metadata.config_fingerprint
                               if getattr(report, "run_metadata", None) else None),
        "summary": {
            "successful": report.successful_queries,
            "failed": report.failed_queries,
            "total_relevant": report.total_relevant,
            "mean_latency_ms": round(report.mean_latency_ms, 1),
            "p50_latency_ms": round(report.p50_latency_ms or 0, 1),
            "p95_latency_ms": round(report.p95_latency_ms or 0, 1),
            "total_api_calls": report.total_api_calls,
            "total_token_usage": report.total_token_usage,
        },
        "metrics": {
            str(k): {
                "precision": round(v.precision, 4),
                "recall": round(v.recall, 4),
                "f1": round(v.f1, 4),
                "ndcg": round(v.ndcg, 4),
                "mrr": round(v.mrr, 4),
            }
            for k, v in report.aggregate.items()
        },
        "per_query": [
            {
                "qid": q.qid,
                "query": q.query,
                "relevant_count": q.relevant_count,
                "returned_count": q.returned_count,
                "matched": q.matched_titles,
                "latency_ms": round(q.latency_ms, 1),
                "api_calls": q.api_calls,
                "token_usage": q.token_usage,
                "metrics": {
                    str(k): {
                        "precision": round(v.precision, 4),
                        "recall": round(v.recall, 4),
                        "f1": round(v.f1, 4),
                        "ndcg": round(v.ndcg, 4),
                        "mrr": round(v.mrr, 4),
                    }
                    for k, v in q.metrics.items()
                },
            }
            for q in report.queries
        ],
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry["summary"], ensure_ascii=False, indent=2))
    print("LOGGED_TO:", LOG_PATH)
    print("run_id:", report.run_id)


if __name__ == "__main__":
    main()
