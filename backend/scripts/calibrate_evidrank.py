"""Create a versioned Cross-logit threshold profile from labelled dev JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.ranking.rank_calibrator import (
    CalibrationExample,
    calibrate_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL development examples")
    parser.add_argument("output", type=Path, help="output threshold profile JSON")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    examples = [
        CalibrationExample.model_validate(json.loads(line))
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    profile = calibrate_threshold(
        examples,
        model_revision=args.model_revision,
        version=args.version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
