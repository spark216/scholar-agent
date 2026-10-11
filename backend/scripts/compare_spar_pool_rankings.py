"""Compare SPAR ranking results on the baseline pool vs the fused pool.

Reads the two reports produced by eval_pasa_selector_rerank.py:
  .modeling-audit/spar_baseline_ranking.json   (baseline pool)
  .modeling-audit/spar_fused_ranking.json      (baseline UNION crawler pool)

For every ranking variant and every k in {5,10,20} it prints the A/B table
(hits / P / R / F1 / nDCG / MRR and the fused-minus-baseline delta), plus the
per-query list of queries where the fused pool gains top-k hits.

Run: python -m backend.scripts.compare_spar_pool_rankings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["hits", "precision", "recall", "f1", "ndcg", "mrr"]


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=str(ROOT / ".modeling-audit" / "spar_baseline_ranking.json"))
    parser.add_argument("--fused", type=str, default=str(ROOT / ".modeling-audit" / "spar_fused_ranking.json"))
    parser.add_argument("--out", type=str, default=str(ROOT / ".modeling-audit" / "spar_pool_ranking_ab.json"))
    args = parser.parse_args()
    _safe_stdout()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    fused = json.loads(Path(args.fused).read_text(encoding="utf-8"))
    variants = list(base.keys())
    report = {}
    print(f"{'variant':<26} {'k':>3} {'hits B/F':>10} {'R B/F':>12} {'F1 B/F':>12} {'nDCG B/F':>12} {'MRR B/F':>12}")
    for variant in variants:
        agg_b = base[variant]["agg"]
        agg_f = fused[variant]["agg"]
        report[variant] = {}
        for k in ("5", "10", "20"):
            b, f = agg_b[k], agg_f[k]
            row = {
                "baseline": b,
                "fused": f,
                "delta": {m: round(f[m] - b[m], 4) for m in METRICS},
            }
            report[variant][k] = row
            print(
                f"{variant:<26} {k:>3} {b['hits']:>4}/{f['hits']:<4} "
                f"{b['recall']:.3f}/{f['recall']:.3f}  "
                f"{b['f1']:.3f}/{f['f1']:.3f}  "
                f"{b['ndcg']:.3f}/{f['ndcg']:.3f}  "
                f"{b['mrr']:.3f}/{f['mrr']:.3f}",
            )
        # per-query gains (fused pool surfaces gold in top-k that baseline pool misses)
        by_qid_b = {q["qid"]: q for q in base[variant]["per_query"]}
        by_qid_f = {q["qid"]: q for q in fused[variant]["per_query"]}
        report[variant]["gains"] = {}
        for k in ("5", "10", "20"):
            gained = [
                {"qid": qid, "baseline_hits": by_qid_b[qid]["metrics"][k]["hits"],
                 "fused_hits": by_qid_f[qid]["metrics"][k]["hits"]}
                for qid in sorted(by_qid_b)
                if by_qid_f[qid]["metrics"][k]["hits"] > by_qid_b[qid]["metrics"][k]["hits"]
            ]
            lost = [
                {"qid": qid, "baseline_hits": by_qid_b[qid]["metrics"][k]["hits"],
                 "fused_hits": by_qid_f[qid]["metrics"][k]["hits"]}
                for qid in sorted(by_qid_b)
                if by_qid_f[qid]["metrics"][k]["hits"] < by_qid_b[qid]["metrics"][k]["hits"]
            ]
            report[variant]["gains"][k] = {"gained": gained, "lost": lost}
            if gained or lost:
                print(f"  [{variant} @{k}] gained {len(gained)}q: {[g['qid'] for g in gained]}")
                if lost:
                    print(f"  [{variant} @{k}] lost  {len(lost)}q: {[g['qid'] for g in lost]}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
