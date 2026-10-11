"""Development-set calibration for transparent V1 ranking thresholds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class CalibrationExample(BaseModel):
    query_id: str
    paper_id: str
    cross_logit: float
    relevant: bool


class ThresholdProfile(BaseModel):
    version: str
    model_revision: str
    min_cross_logit: float
    development_data_hash: str
    example_count: int = Field(gt=0)
    objective: str = "macro_f1"


def _data_hash(examples: list[CalibrationExample]) -> str:
    payload = [
        value.model_dump(mode="json")
        for value in sorted(examples, key=lambda item: (item.query_id, item.paper_id))
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _macro_f1(examples: list[CalibrationExample], threshold: float) -> float:
    by_query: dict[str, list[CalibrationExample]] = {}
    for example in examples:
        by_query.setdefault(example.query_id, []).append(example)
    scores: list[float] = []
    for values in by_query.values():
        tp = sum(value.relevant and value.cross_logit >= threshold for value in values)
        fp = sum(not value.relevant and value.cross_logit >= threshold for value in values)
        fn = sum(value.relevant and value.cross_logit < threshold for value in values)
        denominator = 2 * tp + fp + fn
        scores.append((2 * tp / denominator) if denominator else 1.0)
    return sum(scores) / len(scores)


def calibrate_threshold(
    examples: list[CalibrationExample],
    *,
    model_revision: str,
    version: str,
) -> ThresholdProfile:
    if not examples:
        raise ValueError("at least one calibration example is required")
    if len({value.query_id for value in examples}) < 2:
        raise ValueError("calibration requires examples from at least two query_id groups")
    if not any(value.relevant for value in examples):
        raise ValueError("calibration data has no relevant examples")
    if all(value.relevant for value in examples):
        raise ValueError("calibration data has no negative examples")

    logits = sorted({value.cross_logit for value in examples})
    epsilon = 1e-9
    candidates = [logits[0] - epsilon, *logits, logits[-1] + epsilon]
    # On equal macro F1, select the higher threshold to prefer precision.
    threshold = max(candidates, key=lambda value: (_macro_f1(examples, value), value))
    return ThresholdProfile(
        version=version,
        model_revision=model_revision,
        min_cross_logit=threshold,
        development_data_hash=_data_hash(examples),
        example_count=len(examples),
    )


def load_threshold_profile(path: str | Path | None) -> ThresholdProfile | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"EvidRank threshold profile not found: {resolved}")
    return ThresholdProfile.model_validate_json(resolved.read_text(encoding="utf-8"))
