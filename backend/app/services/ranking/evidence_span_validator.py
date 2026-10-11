"""Independent validation of evidence spans."""

from __future__ import annotations

import hashlib

from backend.app.models.evidrank import EvidenceLabel, EvidenceRecord


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_evidence_span(record: EvidenceRecord, document: str) -> EvidenceRecord:
    """Reject invalid text spans and conservatively fall back to unknown."""
    if record.source_type == "metadata":
        return record
    if record.label == EvidenceLabel.UNKNOWN:
        return record
    valid = (
        record.char_start is not None
        and record.char_end is not None
        and record.quoted_text is not None
        and 0 <= record.char_start <= record.char_end <= len(document)
        and document[record.char_start:record.char_end] == record.quoted_text
        and record.normalized_text_hash == normalized_text_hash(document)
    )
    if valid:
        return record
    return record.model_copy(
        update={
            "label": EvidenceLabel.UNKNOWN,
            "confidence": 0.0,
            "quoted_text": None,
            "char_start": None,
            "char_end": None,
            "validation_status": "invalid",
            "reason_codes": list(dict.fromkeys(record.reason_codes + ["INVALID_EVIDENCE_SPAN"])),
        }
    )
