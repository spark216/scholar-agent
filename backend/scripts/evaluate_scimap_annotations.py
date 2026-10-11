"""Initialize and evaluate double-blind SciMap quality annotations.

The script never creates labels. ``init`` expands source records into rows with
blank annotator fields; ``score`` refuses incomplete or invalid annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


LABEL_FIELDS = (
    "paper_relevant",
    "evidence_supports_claim",
    "claim_overstated",
    "comparison_comparable",
    "citation_accurate",
)
VALID_LABELS = {"0", "1"}
IDENTITY_FIELDS = ("query_id", "query", "paper_id", "claim_id", "evidence_id")


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_rate = sum(value == "1" for value in left) / len(left)
    right_rate = sum(value == "1" for value in right) / len(right)
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def _wilson_interval(positive: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = positive / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            yield value


def initialize(source: Path, output_a: Path, output_b: Path, limit: int) -> None:
    if limit < 100:
        raise ValueError("--limit must be at least 100 for the required blind audit")
    rows: list[dict[str, str]] = []
    query_ids: set[str] = set()
    for index, value in enumerate(_iter_jsonl(source)):
        query_id = str(value.get("qid") or value.get("query_id") or "").strip()
        query = str(value.get("question") or value.get("query") or "").strip()
        paper_id = str(value.get("paper_id") or "").strip()
        claim_id = str(value.get("claim_id") or "").strip()
        evidence_id = str(value.get("evidence_id") or "").strip()
        if not all((query_id, query, paper_id, claim_id, evidence_id)):
            raise ValueError(
                f"source row {index + 1} lacks query/paper/claim/evidence identity"
            )
        if query_id not in query_ids and len(query_ids) >= limit:
            continue
        query_ids.add(query_id)
        row = {
            "query_id": query_id,
            "query": query,
            "paper_id": paper_id,
            "claim_id": claim_id,
            "evidence_id": evidence_id,
            "annotator": "",
            "notes": "",
        }
        for field in LABEL_FIELDS:
            row[field] = ""
        rows.append(row)
    if len(query_ids) < min(100, limit):
        raise ValueError(
            f"only {len(query_ids)} source queries available; at least {min(100, limit)} required"
        )
    if output_a.resolve() == output_b.resolve():
        raise ValueError("A/B annotation outputs must be different files")
    fieldnames = list(rows[0])
    for output in (output_a, output_b):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _annotation_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "").strip() for field in IDENTITY_FIELDS)


def merge_annotations(source_a: Path, source_b: Path, output: Path) -> None:
    sheets: list[list[dict[str, str]]] = []
    for source in (source_a, source_b):
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty annotation sheet: {source}")
        sheets.append(rows)
    left, right = sheets
    left_by_key = {_annotation_key(row): row for row in left}
    right_by_key = {_annotation_key(row): row for row in right}
    if len(left_by_key) != len(left) or len(right_by_key) != len(right):
        raise ValueError("annotation sheets contain duplicate review units")
    if set(left_by_key) != set(right_by_key):
        raise ValueError("A/B annotation sheets do not contain the same review units")

    merged: list[dict[str, str]] = []
    for key in sorted(left_by_key):
        row_a, row_b = left_by_key[key], right_by_key[key]
        annotator_a = str(row_a.get("annotator") or "").strip()
        annotator_b = str(row_b.get("annotator") or "").strip()
        if not annotator_a or not annotator_b:
            raise ValueError(f"review unit {key}: both annotator names are required")
        if annotator_a.casefold() == annotator_b.casefold():
            raise ValueError(f"review unit {key}: A/B annotators must be different")
        row = {
            **dict(zip(IDENTITY_FIELDS, key, strict=True)),
            "annotator_a": annotator_a,
            "annotator_b": annotator_b,
            "adjudicator": "",
            "notes": " | ".join(
                value for value in (row_a.get("notes", ""), row_b.get("notes", ""))
                if value
            ),
        }
        for field in LABEL_FIELDS:
            value_a = str(row_a.get(field) or "").strip()
            value_b = str(row_b.get(field) or "").strip()
            if value_a not in VALID_LABELS or value_b not in VALID_LABELS:
                raise ValueError(f"review unit {key}: incomplete A/B label for {field}")
            row[f"{field}_a"] = value_a
            row[f"{field}_b"] = value_b
            row[f"{field}_final"] = ""
        merged.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)


def score(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({row.get("query_id", "") for row in rows if row.get("query_id")}) < 100:
        raise ValueError("fewer than 100 distinct annotated queries")

    metrics: dict[str, object] = {
        "schema_version": "scimap-human-eval-v1",
        "row_count": len(rows),
        "query_count": len({row["query_id"] for row in rows}),
        "agreement": {},
        "annotator_pairs": [],
    }
    final_values: dict[str, list[str]] = {field: [] for field in LABEL_FIELDS}
    annotator_pairs: set[tuple[str, str]] = set()
    for field in LABEL_FIELDS:
        left: list[str] = []
        right: list[str] = []
        for line_number, row in enumerate(rows, start=2):
            annotator_a = str(row.get("annotator_a") or "").strip()
            annotator_b = str(row.get("annotator_b") or "").strip()
            if not annotator_a or not annotator_b:
                raise ValueError(
                    f"line {line_number}: annotator_a and annotator_b are required"
                )
            if annotator_a.casefold() == annotator_b.casefold():
                raise ValueError(
                    f"line {line_number}: A/B labels require two different annotators"
                )
            annotator_pairs.add((annotator_a, annotator_b))
            a = str(row.get(f"{field}_a") or "").strip()
            b = str(row.get(f"{field}_b") or "").strip()
            final = str(row.get(f"{field}_final") or "").strip()
            if a not in VALID_LABELS or b not in VALID_LABELS:
                raise ValueError(f"line {line_number}: incomplete {field} A/B labels")
            if a != b:
                adjudicator = str(row.get("adjudicator") or "").strip()
                if final not in VALID_LABELS or not adjudicator:
                    raise ValueError(
                        f"line {line_number}: disagreement requires a final label and adjudicator for {field}"
                    )
                if adjudicator.casefold() in {
                    annotator_a.casefold(),
                    annotator_b.casefold(),
                }:
                    raise ValueError(
                        f"line {line_number}: adjudicator must be a third person for {field}"
                    )
            elif final in VALID_LABELS and final != a:
                raise ValueError(
                    f"line {line_number}: agreed A/B label cannot be overridden for {field}"
                )
            if final not in VALID_LABELS:
                final = a
            left.append(a)
            right.append(b)
            final_values[field].append(final)
        metrics["agreement"][field] = {
            "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(left),
            "cohen_kappa": _cohen_kappa(left, right),
            "disagreements": sum(a != b for a, b in zip(left, right, strict=True)),
        }

    def rate_with_ci(field: str, positive: str = "1") -> dict[str, object]:
        values = final_values[field]
        count = sum(value == positive for value in values)
        return {
            "value": count / len(values),
            "positive": count,
            "total": len(values),
            "wilson_95ci": _wilson_interval(count, len(values)),
        }

    def rate_with_ci_composite() -> dict[str, object]:
        values = [
            final_values["evidence_supports_claim"][index] == "1"
            and final_values["citation_accurate"][index] == "1"
            for index in range(len(rows))
        ]
        count = sum(values)
        return {
            "value": count / len(values),
            "positive": count,
            "total": len(values),
            "wilson_95ci": _wilson_interval(count, len(values)),
        }

    metrics.update(
        {
            "paper_relevance_rate": rate_with_ci("paper_relevant"),
            # A citation is precise only when it points to the right paper/span
            # and that span supports the emitted claim.
            "claim_citation_precision": rate_with_ci_composite(),
            "unsupported_strong_assertion_rate": rate_with_ci("claim_overstated"),
            "comparability_accuracy": rate_with_ci("comparison_comparable"),
            "citation_accuracy": rate_with_ci("citation_accurate"),
            "label_counts": {
                field: dict(Counter(values)) for field, values in final_values.items()
            },
        }
    )
    metrics["annotator_pairs"] = [
        {"annotator_a": left, "annotator_b": right}
        for left, right in sorted(annotator_pairs)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--source", type=Path, required=True)
    init_parser.add_argument("--output-a", type=Path, required=True)
    init_parser.add_argument("--output-b", type=Path, required=True)
    init_parser.add_argument("--limit", type=int, default=100)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--source-a", type=Path, required=True)
    merge_parser.add_argument("--source-b", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--source", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.source, args.output_a, args.output_b, args.limit)
    elif args.command == "merge":
        merge_annotations(args.source_a, args.source_b, args.output)
    else:
        score(args.source, args.output)


if __name__ == "__main__":
    main()
