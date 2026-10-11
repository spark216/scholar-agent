"""Budgeted verification over controller-provided, legally fetched full text."""

from __future__ import annotations

import hashlib

from backend.app.models.evidrank import (
    CandidatePaper,
    EvidenceLabel,
    EvidenceRecord,
    FieldProvenance,
    FulltextDocument,
    ResearchIntent,
    UncertaintyItem,
    UncertaintyReason,
    VoiProxyLevel,
)
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.evidence_span_validator import validate_evidence_span


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_fulltext_document(
    document: FulltextDocument,
    paper: CandidatePaper,
) -> None:
    expected_hash = document.content_hash.removeprefix("sha256:")
    if not document.license_allowed:
        raise ValueError(f"full text license is not approved for {document.paper_id}")
    if document.paper_id != paper.canonical_paper_id:
        raise ValueError(f"full text paper id does not match candidate {paper.canonical_paper_id}")
    if document.paper_version_id != paper.preferred_version_id:
        raise ValueError(f"full text version does not match preferred version for {document.paper_id}")
    if _sha256(document.text) != expected_hash:
        raise ValueError(f"full text content hash mismatch for {document.paper_id}")


def verify_fulltext(
    *,
    session_id: str,
    intent: ResearchIntent,
    papers: dict[str, CandidatePaper],
    uncertainty_queue: list[UncertaintyItem],
    documents: dict[str, FulltextDocument],
    max_documents: int,
) -> tuple[list[EvidenceRecord], list[str]]:
    if max_documents <= 0 or not documents:
        return [], []

    priority = {VoiProxyLevel.HIGH: 0, VoiProxyLevel.MEDIUM: 1, VoiProxyLevel.LOW: 2}
    eligible = sorted(
        (
            item
            for item in uncertainty_queue
            if item.paper_id in documents
            and item.reason_code in {
                UncertaintyReason.FULLTEXT_REQUIRED,
                UncertaintyReason.ABSTRACT_INSUFFICIENT,
            }
        ),
        key=lambda item: (priority[item.voi_proxy_level], item.paper_id, item.facet_id),
    )
    selected_ids: list[str] = []
    for item in eligible:
        if item.paper_id not in selected_ids:
            selected_ids.append(item.paper_id)
        if len(selected_ids) >= max_documents:
            break

    judge = DeterministicEvidenceJudge()
    records: list[EvidenceRecord] = []
    for paper_id in selected_ids:
        paper = papers[paper_id]
        document = documents[paper_id]
        validate_fulltext_document(document, paper)
        fulltext_paper = paper.model_copy(
            update={
                "abstract": document.text,
                "field_provenance": [
                    *paper.field_provenance,
                    FieldProvenance(
                        field="abstract",
                        value=document.content_hash,
                        provider="controller_fulltext_service",
                        response_hash=document.content_hash,
                        confidence_level="authoritative",
                    ),
                ],
            }
        )
        for record in judge.judge(session_id=session_id, paper=fulltext_paper, intent=intent):
            if record.source_type != "abstract":
                continue
            evidence_id = "ev-" + hashlib.sha256(
                (
                    f"{session_id}\x1f{paper_id}\x1f{record.constraint_id or record.facet_id}\x1f"
                    f"fulltext\x1f{document.content_hash}\x1f{document.parser_version}\x1f"
                    f"{record.quoted_text or 'unknown'}"
                ).encode("utf-8")
            ).hexdigest()[:20]
            upgraded = record.model_copy(
                update={
                    "evidence_id": evidence_id,
                    "paper_version_id": document.paper_version_id,
                    "source_type": "fulltext",
                    "document_version_id": document.paper_version_id,
                    "content_hash": document.content_hash.removeprefix("sha256:"),
                    "normalized_text_hash": _sha256(document.text),
                    "parser_version": document.parser_version,
                    "chunk_id": document.chunk_id,
                    "judge_model": "deterministic-fulltext-evidence-v1",
                    "reason_codes": list(
                        dict.fromkeys(record.reason_codes + ["CONTROLLER_PROVIDED_FULLTEXT"])
                    ),
                }
            )
            if upgraded.label != EvidenceLabel.UNKNOWN:
                upgraded = validate_evidence_span(upgraded, document.text)
            records.append(upgraded)
    return records, selected_ids
