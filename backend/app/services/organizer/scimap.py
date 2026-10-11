"""Assemble feature-four SciMap output from one frozen bundle."""

from __future__ import annotations

import hashlib
import json

from backend.app.models.evidrank import FinalRankingBundle, ResearchIntent
from backend.app.models.scimap import OutputArtifact
from backend.app.services.organizer.comparison_matrix import build_comparison_matrix
from backend.app.services.organizer.comparability import build_comparability_assessments
from backend.app.services.organizer.evidence_graph import build_evidence_graph
from backend.app.services.organizer.ranked_list import build_ranked_list
from backend.app.services.organizer.synthesis import build_synthesis
from backend.app.services.organizer.timeline import build_timeline
from backend.app.services.ranking.bundle_builder import compute_bundle_hash


def compute_artifact_hash(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_output_artifact(
    bundle: FinalRankingBundle,
    intent: ResearchIntent,
) -> OutputArtifact:
    if not bundle.is_frozen or bundle.frozen_at is None:
        raise ValueError("SciMap only accepts a frozen FinalRankingBundle")
    if compute_bundle_hash(bundle) != bundle.bundle_hash:
        raise ValueError("FinalRankingBundle hash verification failed")
    if intent.intent_hash != bundle.intent_hash:
        raise ValueError("ResearchIntent does not match FinalRankingBundle")

    synthesis, validation = build_synthesis(bundle, intent)
    data = {
        "artifact_version": "scimap-v1",
        "ranking_run_id": bundle.ranking_run_id,
        "bundle_hash": bundle.bundle_hash,
        "intent_hash": bundle.intent_hash,
        "ranked_list": [value.model_dump(mode="json") for value in build_ranked_list(bundle)],
        "comparison_matrix": build_comparison_matrix(bundle, intent).model_dump(mode="json"),
        "timeline": [value.model_dump(mode="json") for value in build_timeline(bundle)],
        "evidence_graph": build_evidence_graph(bundle).model_dump(mode="json"),
        "comparability_assessments": [
            value.model_dump(mode="json")
            for value in build_comparability_assessments(bundle, intent)
        ],
        "synthesis": synthesis.model_dump(mode="json"),
        "claim_validation": validation.model_dump(mode="json"),
    }
    return OutputArtifact.model_validate({**data, "artifact_hash": compute_artifact_hash(data)})


def verify_output_artifact_hash(artifact: OutputArtifact) -> bool:
    payload = artifact.model_dump(mode="json", exclude={"artifact_hash"})
    return compute_artifact_hash(payload) == artifact.artifact_hash
