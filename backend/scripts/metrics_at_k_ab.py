"""Recompute metrics at arbitrary k from the A/B run JSONs using the host's
matching + metrics logic (ranking trace = non-EXCLUDED items by final_rank).

Usage:
    python -m backend.scripts.metrics_at_k_ab --runs <a.json> <b.json> [--ks 5 10 20 25 30]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def normalize_arxiv_id(value) -> str:
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", normalized)
    normalized = re.sub(r"^arxiv:\s*", "", normalized)
    normalized = normalized.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", normalized)


def normalize_title(value) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def gold_records(gold_arxiv_ids, gold_titles):
    records = []
    for index in range(max(len(gold_arxiv_ids), len(gold_titles))):
        raw_id = gold_arxiv_ids[index] if index < len(gold_arxiv_ids) else ""
        raw_title = gold_titles[index] if index < len(gold_titles) else ""
        arxiv_id = normalize_arxiv_id(raw_id)
        title = normalize_title(raw_title)
        if arxiv_id or title:
            records.append((arxiv_id, title, str(raw_title or raw_id)))
    return records


def relevance_flags(papers, gold_arxiv_ids, gold_titles):
    gold = gold_records(gold_arxiv_ids, gold_titles)
    unmatched = set(range(len(gold)))
    flags, matches = [], []
    for paper in papers:
        arxiv_id = normalize_arxiv_id(paper.get("arxiv_id") or paper.get("id"))
        title = normalize_title(paper.get("title"))
        matched_index = next(
            (i for i in unmatched if arxiv_id and gold[i][0] and arxiv_id == gold[i][0]),
            None,
        )
        if matched_index is None:
            matched_index = next(
                (i for i in unmatched if title and gold[i][1] and title == gold[i][1]),
                None,
            )
        flags.append(1 if matched_index is not None else 0)
        if matched_index is not None:
            unmatched.remove(matched_index)
            matches.append(gold[matched_index][2])
    return flags, matches


def metric_at_k(flags, relevant_count, k):
    selected = flags[:k]
    hits = sum(selected)
    precision = hits / k
    recall = hits / relevant_count if relevant_count else 0.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom else 0.0
    dcg = sum((1 / math.log2(rank + 1)) for rank, flag in enumerate(selected, start=1) if flag)
    ideal_hits = min(relevant_count, k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    first_hit = next((rank for rank, flag in enumerate(selected, start=1) if flag), None)
    mrr = (1 / first_hit) if first_hit else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "ndcg": ndcg, "mrr": mrr, "hits": hits}


def ranked_papers_from_run(run):
    """Per-query ranked non-EXCLUDED paper list from candidate_trace.ranking."""
    out = {}
    for q in run.get("queries", []):
        trace = (q.get("candidate_trace") or {}).get("ranking") or []
        rows = []
        for r in sorted(trace, key=lambda r: int(r.get("rank") or 0)):
            if str(r.get("level") or "").lower() == "excluded":
                continue
            item = dict(r)
            item["arxiv_id"] = (r.get("external_ids") or {}).get("arxiv") or r.get("arxiv_id")
            rows.append(item)
        out[q["qid"]] = rows
    return out


def load_samples():
    path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    samples = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        samples[row["qid"]] = row
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 25, 30])
    args = parser.parse_args()
    _safe_stdout()

    samples = load_samples()
    runs = []
    for path in args.runs:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        ranked = ranked_papers_from_run(run)
        runs.append((run["run_id"][:12], ranked))

    # validate recomputation against the run's own @5/@10/@20 aggregates
    for label, ranked in runs:
        run = json.loads(Path(next(p for p in args.runs if label in p)).read_text(encoding="utf-8"))
        ok = True
        for k in ("5", "10", "20"):
            agg_hits = run.get("aggregate", {}).get(k, {}).get("hits")
            if agg_hits is None:
                continue
            total = 0
            for qid, rows in ranked.items():
                gold = samples.get(qid, {})
                flags, _ = relevance_flags(rows, gold.get("answer_arxiv_id", []), gold.get("answer", []))
                rc = len(gold_records(gold.get("answer_arxiv_id", []), gold.get("answer", [])))
                total += metric_at_k(flags, rc, int(k))["hits"]
            match = "OK" if total == agg_hits else f"MISMATCH(run={agg_hits})"
            print(f"[validate] {label} @{k} recompute={total} {match}", flush=True)

    print()
    for label, ranked in runs:
        print(f"===== {label} =====", flush=True)
        header = "k   | " + " | ".join(f"{m:>6}" for m in ("hits", "P", "R", "F1", "nDCG", "MRR"))
        print(header, flush=True)
        for k in args.ks:
            per_query = []
            total_relevant = 0
            for qid, rows in ranked.items():
                gold = samples.get(qid, {})
                flags, _ = relevance_flags(rows, gold.get("answer_arxiv_id", []), gold.get("answer", []))
                rc = len(gold_records(gold.get("answer_arxiv_id", []), gold.get("answer", [])))
                total_relevant += rc
                per_query.append(metric_at_k(flags, rc, k))
            agg = {m: sum(x[m] for x in per_query) / len(per_query) for m in ("precision", "recall", "f1", "ndcg", "mrr")}
            agg["hits"] = sum(x["hits"] for x in per_query)
            print(
                f"{k:>2} | " + " | ".join(
                    f"{agg['hits']:>6} {agg['precision']:>6.3f} {agg['recall']:>6.3f} "
                    f"{agg['f1']:>6.3f} {agg['ndcg']:>6.3f} {agg['mrr']:>6.3f}"
                ).replace("  ", " "),
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
