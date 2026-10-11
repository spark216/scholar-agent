from __future__ import annotations

import csv
import json

import pytest

from backend.scripts.evaluate_scimap_annotations import LABEL_FIELDS, score


def _write_rows(path, *, disagreement: bool, adjudicator: str) -> None:
    fieldnames = [
        "query_id",
        "query",
        "paper_id",
        "claim_id",
        "evidence_id",
        "annotator_a",
        "annotator_b",
        "adjudicator",
        "notes",
        *[
            name
            for field in LABEL_FIELDS
            for name in (f"{field}_a", f"{field}_b", f"{field}_final")
        ],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(100):
            row = {
                "query_id": f"q{index}",
                "query": f"query {index}",
                "paper_id": f"p{index}",
                "claim_id": f"c{index}",
                "evidence_id": f"e{index}",
                "annotator_a": "member-a",
                "annotator_b": "member-b",
                "adjudicator": adjudicator,
                "notes": "",
            }
            for field in LABEL_FIELDS:
                row[f"{field}_a"] = "1"
                row[f"{field}_b"] = "0" if disagreement else "1"
                row[f"{field}_final"] = "1" if disagreement else ""
            writer.writerow(row)


def test_scimap_annotation_score_requires_third_person_for_disagreement(tmp_path):
    source = tmp_path / "labels.csv"
    _write_rows(source, disagreement=True, adjudicator="member-a")

    with pytest.raises(ValueError, match="third person"):
        score(source, tmp_path / "metrics.json")


def test_scimap_annotation_score_records_valid_annotators(tmp_path):
    source = tmp_path / "labels.csv"
    output = tmp_path / "metrics.json"
    _write_rows(source, disagreement=True, adjudicator="member-c")

    score(source, output)

    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 100
    assert metrics["annotator_pairs"] == [
        {"annotator_a": "member-a", "annotator_b": "member-b"}
    ]
