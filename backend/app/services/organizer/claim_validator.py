"""Validate that synthesis claims only reference frozen bundle evidence."""

from __future__ import annotations

from backend.app.models.evidrank import EvidenceLabel, FinalRankingBundle
from backend.app.models.scimap import ClaimValidationSummary, SynthesisClaim


def validate_claims(
    claims: list[SynthesisClaim],
    bundle: FinalRankingBundle,
) -> tuple[list[SynthesisClaim], ClaimValidationSummary]:
    paper_ids = {item.paper_id for item in bundle.items}
    evidence_by_id = {record.evidence_id: record for record in bundle.evidence_matrix}
    validated: list[SynthesisClaim] = []
    invalid_ids: list[str] = []
    unsupported_strong = 0

    for claim in claims:
        reasons: list[str] = []
        missing_papers = sorted(set(claim.paper_ids) - paper_ids)
        missing_evidence = sorted(set(claim.evidence_ids) - set(evidence_by_id))
        if missing_papers:
            reasons.append("UNKNOWN_PAPER_REFERENCE")
        if missing_evidence:
            reasons.append("UNKNOWN_EVIDENCE_REFERENCE")
        if claim.strong_assertion and not claim.evidence_ids:
            reasons.append("UNSUPPORTED_STRONG_ASSERTION")
            unsupported_strong += 1
        if claim.assertion_level == "A" and (
            not claim.paper_ids or not claim.evidence_ids
        ):
            reasons.append("LEVEL_A_REQUIRES_DIRECT_EVIDENCE")
        if claim.assertion_level == "B" and (
            len(set(claim.paper_ids)) < 2 or not claim.evidence_ids
        ):
            reasons.append("LEVEL_B_REQUIRES_MULTIPLE_PAPERS")
        if claim.assertion_level == "C" and not claim.evidence_ids:
            reasons.append("LEVEL_C_REQUIRES_QUALIFYING_EVIDENCE")
        if claim.assertion_level == "D" and claim.strong_assertion:
            reasons.append("INFERENCE_CANNOT_BE_STRONG_ASSERTION")
        for evidence_id in claim.evidence_ids:
            record = evidence_by_id.get(evidence_id)
            if record is None:
                continue
            if record.validation_status != "valid":
                reasons.append("INVALID_EVIDENCE_REFERENCE")
            if claim.paper_ids and record.paper_id not in claim.paper_ids:
                reasons.append("EVIDENCE_PAPER_MISMATCH")
        referenced_records = [
            evidence_by_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        referenced_labels = {record.label for record in referenced_records}
        if claim.claim_type in {"evidence", "consensus", "trend"} and any(
            record.label != EvidenceLabel.SUPPORT for record in referenced_records
        ):
            reasons.append("CLAIM_REQUIRES_SUPPORT_EVIDENCE")
        if claim.claim_type == "disagreement" and not {
            EvidenceLabel.SUPPORT,
            EvidenceLabel.VIOLATE,
        }.issubset(referenced_labels):
            reasons.append("DISAGREEMENT_REQUIRES_OPPOSING_EVIDENCE")
        if claim.claim_type in {"evidence", "consensus", "trend", "disagreement"}:
            evidenced_papers = {record.paper_id for record in referenced_records}
            if set(claim.paper_ids) != evidenced_papers:
                reasons.append("CLAIM_PAPER_SET_NOT_FULLY_EVIDENCED")
        reasons = sorted(set(reasons))
        status = "invalid" if reasons else "valid"
        checked = claim.model_copy(
            update={"validation_status": status, "reason_codes": reasons}
        )
        validated.append(checked)
        if status == "invalid":
            invalid_ids.append(claim.claim_id)

    total = len(validated)
    return validated, ClaimValidationSummary(
        total_claims=total,
        valid_claims=total - len(invalid_ids),
        invalid_claims=len(invalid_ids),
        invalid_claim_ids=invalid_ids,
        unsupported_strong_assertion_rate=(unsupported_strong / total if total else 0.0),
    )
