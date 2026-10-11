from backend.app.models.evidrank import Constraint, EvidenceLabel, EvidenceRecord
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.adapters import (
    candidates_from_rows,
    research_intent_from_understanding,
)
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.evidence_span_validator import (
    normalized_text_hash,
    validate_evidence_span,
)


def test_exact_negative_span_becomes_violation(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    intent = research_intent_from_understanding(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
    )
    papers = {p.canonical_paper_id: p for p in candidates_from_rows(candidate_fixture)}
    records = DeterministicEvidenceJudge().judge(
        session_id="s1",
        paper=papers["p-alpha-sim"],
        intent=intent,
    )
    record = next(item for item in records if item.constraint_id == "C-no-sim")
    assert record.label == EvidenceLabel.VIOLATE
    assert record.quoted_text == "simulation only"
    assert record.validation_status == "valid"


def test_invalid_span_falls_back_to_unknown():
    document = "A real robot experiment."
    record = EvidenceRecord(
        evidence_id="ev1",
        session_id="s1",
        paper_id="p1",
        paper_version_id="v1",
        facet_id="real_robot",
        label=EvidenceLabel.SUPPORT,
        confidence=0.9,
        source_type="abstract",
        quoted_text="wrong",
        document_version_id="v1",
        content_hash=normalized_text_hash(document),
        normalized_text_hash=normalized_text_hash(document),
        parser_version="test",
        char_start=2,
        char_end=7,
    )
    validated = validate_evidence_span(record, document)
    assert validated.label == EvidenceLabel.UNKNOWN
    assert validated.validation_status == "invalid"
    assert "INVALID_EVIDENCE_SPAN" in validated.reason_codes


def test_missing_provider_language_is_inferred_with_auditable_provenance():
    paper = candidates_from_rows(
        [
            {
                "id": "rice-en",
                "title": "Optimization of sowing dates for enhanced rice yield",
                "abstract": "Field experiments evaluate planting strategies for rice production.",
                "year": 2024,
                "source": "openalex",
            }
        ]
    )[0]
    assert paper.language == "en"
    language_provenance = next(
        value for value in paper.field_provenance if value.field == "language"
    )
    assert language_provenance.provider == "local-language-heuristic-v1"
    assert language_provenance.confidence_level == "trusted"

    record = DeterministicEvidenceJudge().judge_constraint(
        session_id="language",
        paper=paper,
        constraint=Constraint(
            id="language-en",
            facet="language",
            operator="eq",
            value="en",
            hardness="hard",
        ),
    )
    assert record.label == EvidenceLabel.SUPPORT


def test_generic_method_request_uses_exact_method_cue_from_abstract():
    paper = candidates_from_rows(
        [
            {
                "id": "rice-method",
                "title": "Rice yield improvement",
                "abstract": "We optimize sowing dates and nutritional management to improve yield.",
                "year": 2024,
                "source": "openalex",
            }
        ]
    )[0]
    record = DeterministicEvidenceJudge().judge_constraint(
        session_id="method",
        paper=paper,
        constraint=Constraint(
            id="method-generic",
            facet="method",
            operator="contains",
            value="Agricultural techniques",
            hardness="soft",
        ),
    )
    assert record.label == EvidenceLabel.SUPPORT
    assert record.quoted_text in {"optimization", "sowing", "management"}
    assert "GENERIC_METHOD_CUE" in record.reason_codes
