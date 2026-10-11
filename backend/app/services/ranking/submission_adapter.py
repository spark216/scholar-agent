"""Deterministic projection from a frozen bundle to the official prediction set."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.models.evidrank import FinalRankingBundle, RankingLevel
from backend.app.services.ranking.bundle_builder import compute_bundle_hash


class SubmissionPolicy(BaseModel):
    adapter_version: str
    include_levels: list[Literal["high", "partial"]] = Field(default_factory=lambda: ["high"])
    max_results: int | None = Field(default=None, gt=0)


class SubmissionPrediction(BaseModel):
    ranking_run_id: str
    bundle_hash: str
    adapter_version: str
    predicted_relevant_set: list[str]
    prediction_hash: str


def build_submission_prediction(
    bundle: FinalRankingBundle,
    policy: SubmissionPolicy,
) -> SubmissionPrediction:
    if not bundle.is_frozen or bundle.frozen_at is None:
        raise ValueError("Submission Adapter only accepts a frozen FinalRankingBundle")
    if compute_bundle_hash(bundle) != bundle.bundle_hash:
        raise ValueError("FinalRankingBundle hash verification failed")
    included = {RankingLevel(value) for value in policy.include_levels}
    paper_ids = [item.paper_id for item in bundle.items if item.level in included]
    if policy.max_results is not None:
        paper_ids = paper_ids[: policy.max_results]
    raw = json.dumps(
        {
            "ranking_run_id": bundle.ranking_run_id,
            "bundle_hash": bundle.bundle_hash,
            "adapter_version": policy.adapter_version,
            "predicted_relevant_set": paper_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return SubmissionPrediction(
        ranking_run_id=bundle.ranking_run_id,
        bundle_hash=bundle.bundle_hash,
        adapter_version=policy.adapter_version,
        predicted_relevant_set=paper_ids,
        prediction_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
