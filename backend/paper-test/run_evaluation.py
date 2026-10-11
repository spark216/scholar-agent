"""CLI entry point for the ScholarAgent PaSa benchmark.

Run from the repository root:
    python backend/paper-test/run_evaluation.py --dataset pasa-real --limit 3 --k 5 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.paper_test import PaperTestRequest  # noqa: E402
from backend.app.services.paper_test import run_paper_test  # noqa: E402


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    _safe_stdout()
    parser = argparse.ArgumentParser(description="Evaluate ScholarAgent on PaSa")
    parser.add_argument(
        "--dataset",
        choices=[
            "pasa-real",
            "pasa-auto",
            "pasa-auto-dev",
            "spar",
            "asta-paper-finder-validation",
            "asta-paper-finder-test",
            "trec-pm-2017",
            "trec-pm-2018",
            "trec-pm-2020",
            "beir-scifact",
            "beir-nfcorpus",
            "beir-arguana",
            "beir-trec-covid",
            "beir-scidocs",
        ],
        default="pasa-real",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--provider-cache-mode",
        choices=["record", "frozen"],
        default="record",
    )
    args = parser.parse_args()
    report = run_paper_test(
        PaperTestRequest(
            dataset=args.dataset,
            sample_limit=args.limit,
            ks=args.k,
            seed=args.seed,
            use_mock=args.mock,
            provider_cache_mode=args.provider_cache_mode,
        )
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
