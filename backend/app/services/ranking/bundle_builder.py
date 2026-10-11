"""Reproducible RankingBundle construction and controller-owned freezing."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from backend.app.models.evidrank import FinalRankingBundle, RankingBundle, RankingLevel


def compute_bundle_hash(bundle: RankingBundle) -> str:
    """Hash semantic content, excluding run ids, timings, and freeze metadata."""
    payload = {
        "session_id": bundle.session_id,
        "intent_hash": bundle.intent_hash,
        "items": [item.model_dump(mode="json") for item in bundle.items],
        "evidence_matrix": [item.model_dump(mode="json") for item in bundle.evidence_matrix],
        "coverage_gaps": [item.model_dump(mode="json") for item in bundle.coverage_gaps],
        "uncertainty_queue": [item.model_dump(mode="json") for item in bundle.uncertainty_queue],
        "model_versions": bundle.model_versions,
        "threshold_version": bundle.threshold_version,
        "degraded": bundle.degraded,
        "degradation_reasons": bundle.degradation_reasons,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def populate_level_lists(bundle: RankingBundle) -> RankingBundle:
    return bundle.model_copy(
        update={
            "high": [item.paper_id for item in bundle.items if item.level == RankingLevel.HIGH],
            "partial": [item.paper_id for item in bundle.items if item.level == RankingLevel.PARTIAL],
            "excluded": [item.paper_id for item in bundle.items if item.level == RankingLevel.EXCLUDED],
            "verify": [item.paper_id for item in bundle.items if item.level == RankingLevel.VERIFY],
        }
    )


def finalize_draft_hash(bundle: RankingBundle) -> RankingBundle:
    draft = populate_level_lists(bundle)
    return draft.model_copy(update={"bundle_hash": compute_bundle_hash(draft)})


def freeze_bundle_for_controller(bundle: RankingBundle) -> FinalRankingBundle:
    """Controller helper; EvidRank itself never calls this function."""
    if bundle.is_frozen and not bundle.frozen_at:
        raise ValueError("frozen bundle is missing frozen_at")
    frozen_at = bundle.frozen_at or datetime.now(UTC)
    payload = bundle.model_dump(mode="python")
    payload.update({"is_frozen": True, "frozen_at": frozen_at})
    frozen = FinalRankingBundle.model_validate(payload)
    return frozen.model_copy(update={"bundle_hash": compute_bundle_hash(frozen)})
