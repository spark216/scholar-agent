"""Metadata trust validation without guessing missing bibliographic fields."""

from __future__ import annotations

from backend.app.models.evidrank import CandidatePaper, MetadataValidation


_REQUIRED_FIELDS = ("title", "abstract", "year")


def validate_metadata(paper: CandidatePaper) -> MetadataValidation:
    trusted: list[str] = []
    weak: list[str] = []
    field_strength: dict[str, str] = {}
    for item in paper.field_provenance:
        current = field_strength.get(item.field)
        if item.confidence_level in {"authoritative", "trusted"}:
            field_strength[item.field] = "trusted"
        elif current is None:
            field_strength[item.field] = "weak"

    for field, strength in field_strength.items():
        (trusted if strength == "trusted" else weak).append(field)

    missing = [
        field
        for field in _REQUIRED_FIELDS
        if getattr(paper, field, None) in (None, "", [])
    ]
    conflict_fields = sorted({item.field for item in paper.metadata_conflicts})
    reasons: list[str] = []
    if missing:
        reasons.append("REQUIRED_METADATA_MISSING")
    if conflict_fields:
        reasons.append("METADATA_CONFLICT")
    if weak:
        reasons.append("WEAK_METADATA_PROVENANCE")
    if paper.provenance and all(item.provider == "tavily" for item in paper.provenance):
        reasons.append("WEB_SNIPPET_NOT_SCHOLARLY_ABSTRACT")

    return MetadataValidation(
        paper_id=paper.canonical_paper_id,
        trusted_fields=sorted(trusted),
        weak_fields=sorted(weak),
        missing_fields=missing,
        conflict_fields=conflict_fields,
        reason_codes=reasons,
    )
