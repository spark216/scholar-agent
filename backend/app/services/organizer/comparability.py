"""Conservative pairwise comparability checks over evidence-backed facets."""

from __future__ import annotations

import hashlib
from itertools import combinations

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle, ResearchIntent
from backend.app.models.scimap import ComparabilityAssessment


_CONTEXT_TERMS = {
    "task",
    "dataset",
    "split",
    "platform",
    "metric",
    "protocol",
    "data_scale",
    "setting",
    "environment",
    "real_or_sim",
    "real_robot",
    "population",
    "intervention",
    "outcome",
    "sample_size",
    "study_type",
}


def _is_context_facet(facet_id: str) -> bool:
    normalized = facet_id.casefold().replace("-", "_")
    return any(term in normalized for term in _CONTEXT_TERMS)


def _assessment_id(left: str, right: str, facets: list[str]) -> str:
    raw = "\x1f".join([left, right, *facets])
    return "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]


def build_comparability_assessments(
    bundle: FinalRankingBundle,
    intent: ResearchIntent,
) -> list[ComparabilityAssessment]:
    requested_facets = list(intent.dynamic_facets)
    requested_facets.extend(constraint.facet for constraint in intent.constraints)
    context_facets = sorted(
        dict.fromkeys(facet for facet in requested_facets if _is_context_facet(facet))
    )
    if not context_facets:
        return []

    visible_ids = set(bundle.high + bundle.partial)
    visible_items = sorted(
        (item for item in bundle.items if item.paper_id in visible_ids),
        key=lambda item: (item.final_rank or 10**9, item.paper_id),
    )
    evidence_index: dict[tuple[str, str], list[str]] = {}
    for record in bundle.evidence_matrix:
        if record.validation_status == "valid" and record.label != EvidenceLabel.UNKNOWN:
            evidence_index.setdefault((record.paper_id, record.facet_id), []).append(
                record.evidence_id
            )

    assessments: list[ComparabilityAssessment] = []
    for left, right in combinations(visible_items, 2):
        matched: list[str] = []
        conflicting: list[str] = []
        unknown: list[str] = []
        evidence_ids: list[str] = []
        for facet_id in context_facets:
            left_label = left.facet_labels.get(facet_id, EvidenceLabel.UNKNOWN)
            right_label = right.facet_labels.get(facet_id, EvidenceLabel.UNKNOWN)
            if EvidenceLabel.UNKNOWN in {left_label, right_label}:
                unknown.append(facet_id)
                continue
            if left_label != right_label:
                conflicting.append(facet_id)
            elif left_label == EvidenceLabel.SUPPORT:
                matched.append(facet_id)
            else:
                # Two violations do not establish that the papers share the
                # same experimental context.
                unknown.append(facet_id)
                continue
            evidence_ids.extend(evidence_index.get((left.paper_id, facet_id), []))
            evidence_ids.extend(evidence_index.get((right.paper_id, facet_id), []))

        if conflicting:
            status = "not_comparable"
            allowed_claims: list[str] = []
            reasons = ["CONFLICTING_EXPERIMENT_CONTEXT"]
        elif unknown:
            status = "insufficient_evidence"
            allowed_claims = []
            reasons = ["EXPERIMENT_CONTEXT_INCOMPLETE"]
        else:
            # Facet agreement supports qualified trend comparison only. Direct
            # numerical superiority needs normalized values and protocols,
            # which are intentionally not inferred from three-state labels.
            status = "partially_comparable"
            allowed_claims = ["QUALIFIED_TREND_COMPARISON"]
            reasons = ["QUERY_FACETS_MATCH_BUT_PROTOCOL_VALUES_UNAVAILABLE"]

        assessments.append(
            ComparabilityAssessment(
                assessment_id=_assessment_id(left.paper_id, right.paper_id, context_facets),
                left_paper_id=left.paper_id,
                right_paper_id=right.paper_id,
                status=status,
                matched_facets=matched,
                conflicting_facets=conflicting,
                unknown_facets=unknown,
                evidence_ids=sorted(set(evidence_ids)),
                allowed_claims=allowed_claims,
                reason_codes=reasons,
            )
        )
    return assessments
