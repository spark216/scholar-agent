"""Compare evaluation runs: baseline, planC and the latest (vector-enabled)
run, using the persisted run reports under backend/paper-test/results/runs/.

Run: python -m backend.scripts.compare_eval_runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _metrics(report: dict, k: int) -> dict:
    agg = report.get("aggregate") or {}
    return agg.get(str(k)) or {}


def main() -> None:
    _safe_stdout()
    runs_dir = ROOT / "backend" / "paper-test" / "results" / "runs"
    reports = {}
    for path in sorted(runs_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        run_id = report.get("run_id") or path.stem.split("_", 1)[-1].rsplit(".", 1)[0]
        reports[run_id] = report

    labels = {
        "493d12fe1be3402f8ce1f67719ea7a5a": "baseline (PRF off, cit on? FTS only)",
        "23f91983bc294841812fed021aaa7169": "planC (PRF on, FTS only)",
    }
    # Latest run by file mtime.
    latest_path = max(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    latest_report = json.loads(latest_path.read_text(encoding="utf-8"))
    latest_id = latest_report.get("run_id") or latest_path.stem
    labels[latest_id] = "latest (vector-enabled)"

    ordered = [rid for rid in labels if rid in reports]
    ordered = sorted(
        ordered,
        key=lambda rid: (
            labels[rid].startswith("latest"),
            0 if rid == latest_id else (1 if rid == "23f91983bc294841812fed021aaa7169" else 2),
        ),
    )

    print(f"latest run file: {latest_path.name}")
    print(f"latest run id  : {latest_id}")
    print(f"latest samples : {latest_report.get('sample_count')}, "
          f"seed {latest_report.get('seed')}, ks {latest_report.get('ks')}")
    print()

    for k in (5, 10):
        print(f"=== @{k} ===")
        print(f"{'run':<46} {'P@':>7} {'R@':>7} {'F1':>7} {'nDCG':>7} {'MRR':>7} {'hits':>5}")
        for rid in ordered:
            m = _metrics(reports[rid], k)
            if not m:
                print(f"{labels[rid]:<46}  (no @{k} aggregate)")
                continue
            print(
                f"{labels[rid]:<46} {m.get('precision', 0):>7.3f} {m.get('recall', 0):>7.3f} "
                f"{m.get('f1', 0):>7.3f} {m.get('ndcg', 0):>7.3f} {m.get('mrr', 0):>7.3f} "
                f"{m.get('hits', 0):>5}"
            )
        print()

    print("=== latest per-query ===")
    for item in latest_report.get("queries") or []:
        print(f"  {str(item.get('qid')):<20} matched={item.get('matched')} "
              f"mrr5={item.get('mrr_at_5') or item.get('mrr5')}")


if __name__ == "__main__":
    main()
