"""V1 rule-based ranking levels; no score is interpreted as a probability."""

from __future__ import annotations

from backend.app.models.evidrank import (
    ConstraintHardness,
    EvidenceLabel,
    GateDecision,
    IdentityStatus,
    RankingItem,
    RankingLevel,
    ResearchIntent,
)


def decide_level(
    item: RankingItem,
    intent: ResearchIntent,
    *,
    high_rank_cutoff: int,
    min_cross_logit: float | None = None,
    recall_first: bool = False,
) -> tuple[RankingLevel, list[str], list[str]]:
    if item.gate.decision == GateDecision.REJECT:
        return RankingLevel.EXCLUDED, ["NON_COMPENSATORY_HARD_REJECT"], []

    hard_facets = {
        constraint.facet
        for constraint in intent.constraints
        if constraint.hardness == ConstraintHardness.HARD
    }
    required_relation_facets = {
        f"relation:{relation.relation}:{relation.source}:{relation.target}"
        for relation in intent.relations
        if relation.required
    }
    core_facets = {
        constraint.facet
        for constraint in intent.constraints
        if constraint.scope in {"core", "required_for_high"}
    }
    required_facets = hard_facets | required_relation_facets | core_facets
    critical_unknown = sorted(
        facet
        for facet in required_facets
        if item.facet_labels.get(facet, EvidenceLabel.UNKNOWN) == EvidenceLabel.UNKNOWN
    )
    hard_violations = [
        facet
        for facet in hard_facets
        if item.facet_labels.get(facet) == EvidenceLabel.VIOLATE
    ]
    if hard_violations:
        if not recall_first:
            return RankingLevel.EXCLUDED, ["EVIDENCE_CONFIRMED_HARD_VIOLATION"], critical_unknown

    relation_violations = [
        facet
        for facet in required_relation_facets
        if item.facet_labels.get(facet) == EvidenceLabel.VIOLATE
    ]
    if relation_violations:
        if not recall_first:
            return RankingLevel.EXCLUDED, ["REQUIRED_RELATION_VIOLATED"], critical_unknown

    rank = (
        item.scores.fusion_rank
        or item.scores.cross_rank
        or item.scores.rrf_rank
        or 10**9
    )
    supported = sum(1 for value in item.facet_labels.values() if value == EvidenceLabel.SUPPORT)
    required_count = len(intent.constraints) + len([r for r in intent.relations if r.required])
    required_evidence_present = all(
        item.facet_labels.get(facet) == EvidenceLabel.SUPPORT
        for facet in required_facets
    )

    verification_reasons: list[str] = []
    identity_conflict = item.identity_status in {
        IdentityStatus.CONFLICT,
        IdentityStatus.VERSION_UNRESOLVED,
    }
    if item.identity_status != IdentityStatus.VALID and (not recall_first or identity_conflict):
        verification_reasons.append("IDENTITY_REQUIRES_VERIFICATION")
    if item.metadata_validation.conflict_fields:
        verification_reasons.append("METADATA_CONFLICT_REQUIRES_VERIFICATION")
    if item.metadata_validation.missing_fields and not recall_first:
        verification_reasons.append("REQUIRED_METADATA_MISSING")
    if item.evidence_conflict_facets:
        verification_reasons.append("EVIDENCE_CONFLICT_REQUIRES_VERIFICATION")
    if critical_unknown and not recall_first:
        verification_reasons.append("CRITICAL_EVIDENCE_UNKNOWN")
    if verification_reasons and rank <= high_rank_cutoff:
        return (
            RankingLevel.VERIFY,
            ["TOP_RANKED_REQUIRES_VERIFICATION", "VERIFY_BEFORE_PARTIAL", *verification_reasons],
            critical_unknown,
        )
    threshold_passed = (
        min_cross_logit is None
        or (
            item.scores.cross_logit is not None
            and item.scores.cross_logit >= min_cross_logit
        )
    )

    if recall_first:
        # Recall-first policy: an LLM judgement alone cannot remove a paper
        # from the visible result set. Confirmed semantic violations and weak
        # bibliographic records remain auditable as partial results.
        if hard_violations:
            return (
                RankingLevel.PARTIAL,
                ["RECALL_FIRST_RETAINED_HARD_EVIDENCE_VIOLATION"],
                critical_unknown,
            )
        if relation_violations:
            return (
                RankingLevel.PARTIAL,
                ["RECALL_FIRST_RETAINED_RELATION_VIOLATION"],
                critical_unknown,
            )
        soft_violations = [
            facet
            for facet, label in item.facet_labels.items()
            if label == EvidenceLabel.VIOLATE
        ]
        incomplete_metadata = (
            item.identity_status == IdentityStatus.METADATA_INCOMPLETE
            or bool(item.metadata_validation.missing_fields)
        )
        if rank <= high_rank_cutoff and threshold_passed and not soft_violations and not incomplete_metadata:
            reasons = ["TOP_RANKED", "RECALL_FIRST_POLICY", "NO_CONFIRMED_VIOLATION"]
            if critical_unknown:
                reasons.append("UNRESOLVED_EVIDENCE_ALLOWED")
            return RankingLevel.HIGH, reasons, critical_unknown
        partial_reasons = ["RECALL_FIRST_RETAINED"]
        if soft_violations:
            partial_reasons.append("SOFT_EVIDENCE_VIOLATION")
        if incomplete_metadata:
            partial_reasons.append("INCOMPLETE_METADATA_RETAINED")
        if not threshold_passed:
            partial_reasons.append("BELOW_CALIBRATED_CROSS_LOGIT_THRESHOLD")
        return RankingLevel.PARTIAL, partial_reasons, critical_unknown

    if (
        rank <= high_rank_cutoff
        and threshold_passed
        and not critical_unknown
        and not verification_reasons
        and required_evidence_present
        and (required_count == 0 or supported > 0)
    ):
        return RankingLevel.HIGH, ["TOP_RANKED", "NO_HARD_VIOLATION", "REQUIRED_EVIDENCE_PRESENT"], []
    if rank <= high_rank_cutoff and not threshold_passed:
        return RankingLevel.PARTIAL, ["BELOW_CALIBRATED_CROSS_LOGIT_THRESHOLD"], critical_unknown
    return RankingLevel.PARTIAL, ["IMPORTANT_BUT_INCOMPLETE"], critical_unknown
