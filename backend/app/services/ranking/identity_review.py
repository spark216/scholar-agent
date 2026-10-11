"""Consistency review of identities produced by the shared Candidate Registry."""

from __future__ import annotations

from backend.app.models.evidrank import CandidatePaper, IdentityReview, IdentityStatus


def review_identity(paper: CandidatePaper) -> IdentityReview:
    reasons: list[str] = []
    if not paper.canonical_paper_id:
        return IdentityReview(
            paper_id=paper.canonical_paper_id or "missing",
            status=IdentityStatus.CONFLICT,
            reason_codes=["MISSING_CANONICAL_ID"],
        )

    preferred = [v for v in paper.versions if v.version_id == paper.preferred_version_id]
    if paper.versions and not preferred:
        reasons.append("PREFERRED_VERSION_NOT_IN_VERSION_SET")
    identifiers = {k.lower(): v.lower().strip() for k, v in paper.identifiers.items() if v}
    if len(identifiers.values()) != len(set(identifiers.values())):
        reasons.append("DUPLICATE_IDENTIFIER_VALUES")
    if paper.metadata_conflicts:
        reasons.append("IDENTITY_METADATA_CONFLICT")
    if paper.provenance and all(source.confidence_level == "weak" for source in paper.provenance):
        reasons.append("PROVISIONAL_IDENTITY_WEAK_SOURCE")

    if "PREFERRED_VERSION_NOT_IN_VERSION_SET" in reasons:
        status = IdentityStatus.VERSION_UNRESOLVED
    elif "IDENTITY_METADATA_CONFLICT" in reasons:
        status = IdentityStatus.CONFLICT
    elif not paper.title or paper.title == "Untitled" or "PROVISIONAL_IDENTITY_WEAK_SOURCE" in reasons:
        status = IdentityStatus.METADATA_INCOMPLETE
        reasons.append("MISSING_TITLE")
    else:
        status = IdentityStatus.VALID
    return IdentityReview(
        paper_id=paper.canonical_paper_id,
        status=status,
        reason_codes=reasons,
    )
