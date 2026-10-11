"""Coverage gaps, uncertainty reasons, and ordinal VOI Proxy."""

from __future__ import annotations

import hashlib

from backend.app.models.evidrank import (
    CoverageGap,
    EvidenceLabel,
    IdentityStatus,
    RankingItem,
    RankingLevel,
    ResearchIntent,
    UncertaintyItem,
    UncertaintyReason,
    VoiProxyLevel,
)


def _short_id(prefix: str, *parts: str) -> str:
    return prefix + hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def build_coverage_gaps(
    intent: ResearchIntent,
    items: list[RankingItem],
    *,
    default_target_count: int,
) -> list[CoverageGap]:
    targets: dict[str, int] = {}
    for constraint in intent.constraints:
        targets.setdefault(constraint.facet, default_target_count)
    for relation in intent.relations:
        if relation.required:
            targets.setdefault(
                f"relation:{relation.relation}:{relation.source}:{relation.target}",
                default_target_count,
            )
    for facet_id, definition in intent.dynamic_facets.items():
        targets[facet_id] = definition.target_count

    gaps: list[CoverageGap] = []
    for facet_id, target in targets.items():
        count = sum(
            1
            for item in items
            if item.level in {RankingLevel.HIGH, RankingLevel.PARTIAL}
            and item.facet_labels.get(facet_id) == EvidenceLabel.SUPPORT
        )
        if count >= target:
            continue
        priority = VoiProxyLevel.HIGH if count == 0 else VoiProxyLevel.MEDIUM
        gaps.append(
            CoverageGap(
                gap_id=_short_id("gap-", intent.intent_hash, facet_id),
                facet_id=facet_id,
                supported_count=count,
                target_count=target,
                priority=priority,
                reason_code="FACET_SUPPORT_BELOW_TARGET",
                suggested_actions=["SEARCH_FACET", "SEARCH_EXACT"],
            )
        )
    return gaps


def _voi_proxy(
    item: RankingItem,
    *,
    facet_is_hard: bool,
    high_rank_cutoff: int,
    contributes_to_gap: bool,
    fulltext_budget_exhausted: bool,
) -> tuple[VoiProxyLevel, list[str]]:
    score = 0
    reasons: list[str] = []
    if facet_is_hard:
        score += 3
        reasons.append("HARD_FACET_UNKNOWN")
    rank = (
        item.scores.fusion_rank
        or item.scores.cross_rank
        or item.scores.rrf_rank
        or 10**9
    )
    if rank <= high_rank_cutoff:
        score += 2
        reasons.append("TOP_RANKED")
    if item.level == RankingLevel.VERIFY:
        score += 2
        reasons.append("POTENTIAL_LEVEL_CHANGE")
    if contributes_to_gap:
        score += 2
        reasons.append("CONTRIBUTES_TO_COVERAGE_GAP")
    if item.fulltext_available:
        score += 1
        reasons.append("OA_PDF_AVAILABLE")
    if fulltext_budget_exhausted:
        score -= 100
        reasons.append("FULLTEXT_BUDGET_EXHAUSTED")
    if score >= 6:
        return VoiProxyLevel.HIGH, reasons
    if score >= 3:
        return VoiProxyLevel.MEDIUM, reasons
    return VoiProxyLevel.LOW, reasons


def build_uncertainty_queue(
    intent: ResearchIntent,
    items: list[RankingItem],
    gaps: list[CoverageGap],
    *,
    high_rank_cutoff: int,
    fulltext_budget_exhausted: bool = False,
) -> list[UncertaintyItem]:
    hard_facets = {
        constraint.facet
        for constraint in intent.constraints
        if constraint.hardness.value == "hard"
    }
    gap_facets = {gap.facet_id for gap in gaps}
    queue: list[UncertaintyItem] = []
    for item in items:
        if item.level == RankingLevel.EXCLUDED:
            continue
        unknown_facets = [
            facet_id
            for facet_id, label in item.facet_labels.items()
            if label == EvidenceLabel.UNKNOWN
        ]
        unknown_facets = list(dict.fromkeys(item.evidence_conflict_facets + unknown_facets))
        if item.identity_status != IdentityStatus.VALID:
            unknown_facets = list(dict.fromkeys(["identity"] + unknown_facets))
        if item.metadata_validation.conflict_fields:
            unknown_facets = list(dict.fromkeys(item.metadata_validation.conflict_fields + unknown_facets))
        if item.metadata_validation.missing_fields:
            unknown_facets = list(dict.fromkeys(item.metadata_validation.missing_fields + unknown_facets))

        for facet_id in unknown_facets:
            if facet_id in item.evidence_conflict_facets:
                reason = UncertaintyReason.EVIDENCE_CONFLICT
                actions = ["VERIFY_FULLTEXT", "REVIEW_CONFLICTING_EVIDENCE"]
            elif item.identity_status != IdentityStatus.VALID and facet_id == "identity":
                reason = UncertaintyReason.IDENTITY_UNRESOLVED
                actions = ["VERIFY_METADATA"]
            elif facet_id in item.metadata_validation.conflict_fields:
                reason = UncertaintyReason.METADATA_CONFLICT
                actions = ["VERIFY_METADATA"]
            elif facet_id in item.metadata_validation.missing_fields:
                reason = UncertaintyReason.MISSING_METADATA
                actions = ["VERIFY_METADATA"]
            elif item.fulltext_available:
                reason = UncertaintyReason.FULLTEXT_REQUIRED
                actions = ["FETCH_FULLTEXT"]
            else:
                reason = UncertaintyReason.ABSTRACT_INSUFFICIENT
                actions = ["VERIFY_METADATA", "FETCH_FULLTEXT_IF_AVAILABLE"]
            voi, reasons = _voi_proxy(
                item,
                facet_is_hard=facet_id in hard_facets,
                high_rank_cutoff=high_rank_cutoff,
                contributes_to_gap=facet_id in gap_facets,
                fulltext_budget_exhausted=fulltext_budget_exhausted,
            )
            queue.append(
                UncertaintyItem(
                    uncertainty_id=_short_id("unc-", item.paper_id, facet_id, reason.value),
                    paper_id=item.paper_id,
                    facet_id=facet_id,
                    reason_code=reason,
                    current_level=item.level,
                    potential_level_change="verify->high" if item.level == RankingLevel.VERIFY else None,
                    fulltext_available=item.fulltext_available,
                    suggested_actions=actions,
                    voi_proxy_level=voi,
                    reason_codes=reasons,
                )
            )
    return sorted(
        queue,
        key=lambda value: (
            {VoiProxyLevel.HIGH: 0, VoiProxyLevel.MEDIUM: 1, VoiProxyLevel.LOW: 2}[value.voi_proxy_level],
            value.paper_id,
            value.facet_id,
        ),
    )
