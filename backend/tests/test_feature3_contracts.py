from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from backend.app.models.evidrank import (
    Constraint,
    ConstraintHardness,
    EvidenceLabel,
    EvidenceRecord,
    FinalRankingBundle,
    FulltextDocument,
    GateDecision,
    RankingLevel,
    ResearchIntent,
)
from backend.app.models.schemas import QueryUnderstanding, SearchRequest
from backend.app.api.routes import _local_model_snapshot_ready, _model_reference_ready
from backend.app.services.ranking.adapters import (
    candidates_from_rows,
    research_intent_from_understanding,
)
from backend.app.services.ranking.bundle_builder import freeze_bundle_for_controller
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.hard_gate import apply_hard_gate
from backend.app.services.ranking.identity_review import review_identity
from backend.app.services.ranking.metadata_validator import validate_metadata
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.app.services.ranking.rank_calibrator import (
    CalibrationExample,
    calibrate_threshold,
)
from backend.app.services.ranking.submission_adapter import (
    SubmissionPolicy,
    build_submission_prediction,
)
from backend.app.services.postgres_store import validate_ranking_bundle_envelope
from backend.tests.conftest import FakeEncoder, FakeReranker


def _pipeline() -> EvidRankPipeline:
    return EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=DeterministicEvidenceJudge(),
        decision_policy="strict",
    )


def test_duplicate_canonical_ids_fail_snapshot_validation(candidate_fixture):
    rows = copy.deepcopy(candidate_fixture)
    rows[1]["canonical_paper_id"] = rows[0]["canonical_paper_id"]
    with pytest.raises(ValueError, match="duplicate canonical_paper_id"):
        candidates_from_rows(rows)


def test_incomplete_model_directory_is_not_reported_ready():
    class FakeFile:
        def __init__(self, *, exists=True, size=0):
            self.exists = exists
            self.size = size

        def is_file(self):
            return self.exists

        def stat(self):
            return SimpleNamespace(st_size=self.size)

    class FakeModelDirectory:
        def is_dir(self):
            return True

        def __truediv__(self, name):
            assert name == "config.json"
            return FakeFile()

        def rglob(self, pattern):
            return [FakeFile(size=1024)] if pattern == "*.safetensors" else []

    assert _local_model_snapshot_ready(FakeModelDirectory()) is False  # type: ignore[arg-type]


def test_cached_hub_model_reference_is_resolved_without_download(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "cached-snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "backend.app.api.routes._local_model_snapshot_ready",
        lambda path: str(path) == "cached-snapshot",
    )
    assert _model_reference_ready(
        "BAAI/example",
        revision="fixed-revision",
        cache_dir="custom-cache",
    )
    assert calls == [
        {
            "repo_id": "BAAI/example",
            "revision": "fixed-revision",
            "cache_dir": "custom-cache",
            "local_files_only": True,
        }
    ]


def test_ranking_persistence_envelope_rejects_cross_session_or_tampering(
    intent_fixture,
    candidate_fixture,
):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    draft = _pipeline().rank(
        session_id="persist-session",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    frozen = freeze_bundle_for_controller(draft)
    payload = frozen.model_dump(mode="json")

    assert validate_ranking_bundle_envelope(
        session_id="persist-session",
        ranking_bundle=payload,
    )["bundle_hash"] == frozen.bundle_hash
    with pytest.raises(ValueError, match="session_id"):
        validate_ranking_bundle_envelope(
            session_id="another-session",
            ranking_bundle=payload,
        )
    with pytest.raises(ValueError, match="hash verification"):
        validate_ranking_bundle_envelope(
            session_id="persist-session",
            ranking_bundle={**payload, "bundle_hash": "tampered"},
        )


def test_fulltext_request_is_bounded_before_pipeline_execution():
    document = {
        "paper_id": "p",
        "paper_version_id": "v",
        "text": "evidence",
        "content_hash": hashlib.sha256(b"evidence").hexdigest(),
        "parser_version": "test",
        "license_allowed": True,
    }
    with pytest.raises(ValueError, match="at most 10"):
        SearchRequest(
            query="papers",
            controller_fulltext_documents={
                f"p{index}": {**document, "paper_id": f"p{index}"}
                for index in range(11)
            },
        )


def test_invalid_numeric_hard_constraint_defers_instead_of_crashing(candidate_fixture):
    paper = candidates_from_rows(candidate_fixture)[0]
    intent = ResearchIntent(
        session_id="bad-number",
        raw_query="recent papers",
        constraints=[
            Constraint(
                id="bad-year",
                facet="year",
                operator="gte",
                value="recent",
                hardness=ConstraintHardness.HARD,
            )
        ],
        intent_hash="intent",
    )
    gate = apply_hard_gate(paper, intent, review_identity(paper), validate_metadata(paper))
    assert gate.decision == GateDecision.DEFER
    assert gate.deferred_constraint_ids == ["bad-year"]


def test_single_year_direction_is_not_misread_as_exact_year(intent_fixture):
    payload = copy.deepcopy(intent_fixture["understanding"])
    payload["time_range"] = "2022"
    payload["raw_json"]["constraints"] = []
    understanding = QueryUnderstanding.model_validate(payload)
    intent = research_intent_from_understanding(
        session_id="year-direction",
        query="查找 2022 年以后发表的机器人论文",
        understanding=understanding,
    )
    year_constraints = [value for value in intent.constraints if value.facet == "year"]
    assert [(value.operator, value.value) for value in year_constraints] == [("gte", 2022)]
    assert year_constraints[0].source_span == {"quoted_text": "2022 年以后"}


def test_malformed_constraint_is_not_silently_dropped(intent_fixture):
    payload = copy.deepcopy(intent_fixture["understanding"])
    payload["raw_json"]["constraints"] = [
        {
            "id": "bad",
            "facet": "year",
            "value": 2022,
            "hardness": "definitely-hard",
        }
    ]
    understanding = QueryUnderstanding.model_validate(payload)
    with pytest.raises(ValueError, match="invalid ResearchIntent constraint"):
        research_intent_from_understanding(
            session_id="malformed",
            query=intent_fixture["query"],
            understanding=understanding,
        )


def test_dynamic_facet_enters_evidence_contract(intent_fixture, candidate_fixture):
    payload = copy.deepcopy(intent_fixture["understanding"])
    payload["raw_json"]["dynamic_facets"] = {
        "benchmark": {
            "label": "Benchmark",
            "values": ["RobotSet"],
            "is_hard": True,
            "target_count": 1,
        }
    }
    understanding = QueryUnderstanding.model_validate(payload)
    bundle = _pipeline().rank(
        session_id="dynamic",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    real = next(item for item in bundle.items if item.paper_id == "p-alpha-real")
    assert real.facet_labels["benchmark"] == EvidenceLabel.SUPPORT
    assert any(record.constraint_id == "dynamic:benchmark" for record in bundle.evidence_matrix)


def test_weak_web_snippet_cannot_become_formal_semantic_evidence(
    intent_fixture,
    candidate_fixture,
):
    row = copy.deepcopy(candidate_fixture[0])
    row["source"] = "tavily"
    row.pop("field_provenance")
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="weak-snippet",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=[row],
        top_k=1,
    )
    item = bundle.items[0]
    assert item.level == RankingLevel.VERIFY
    semantic = [record for record in bundle.evidence_matrix if record.source_type == "abstract"]
    assert semantic
    assert all(record.label == EvidenceLabel.UNKNOWN for record in semantic)
    assert all(record.validation_status == "untrusted_source" for record in semantic)


def test_missing_core_method_or_dataset_cannot_enter_high(
    intent_fixture,
    candidate_fixture,
):
    row = copy.deepcopy(candidate_fixture[0])
    row["abstract"] = "We validate a baseline in real robot experiments."
    for provenance in row["field_provenance"]:
        if provenance["field"] == "abstract":
            provenance["value"] = row["abstract"]
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="missing-core",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=[row],
        top_k=1,
    )
    assert bundle.items[0].level == RankingLevel.VERIFY
    assert {"method", "dataset"}.issubset(bundle.items[0].critical_unknown_facets)


def test_conflicting_evidence_is_unknown_and_auditable():
    base = dict(
        evidence_id="abstract",
        session_id="s",
        paper_id="p",
        paper_version_id="v",
        facet_id="robot",
        constraint_id="c",
        confidence=1.0,
        quoted_text="robot",
        document_version_id="v",
        content_hash="h",
        normalized_text_hash="h",
        parser_version="v1",
        char_start=0,
        char_end=5,
    )
    abstract = EvidenceRecord(
        **base,
        label=EvidenceLabel.SUPPORT,
        source_type="abstract",
    )
    fulltext = EvidenceRecord(
        **{**base, "evidence_id": "fulltext"},
        label=EvidenceLabel.VIOLATE,
        source_type="fulltext",
    )
    labels, conflicts = EvidRankPipeline._effective_labels([abstract, fulltext])
    assert labels["robot"] == EvidenceLabel.UNKNOWN
    assert conflicts == ["robot"]


def test_budgeted_fulltext_can_resolve_high_value_unknown(
    intent_fixture,
    candidate_fixture,
):
    rows = copy.deepcopy(candidate_fixture)
    unknown = next(row for row in rows if row["canonical_paper_id"] == "p-alpha-unknown")
    unknown["oa_status"] = "open"
    unknown["pdf_url"] = "https://example.org/p-alpha-unknown.pdf"
    text = "We use Alpha on RobotSet and validate it in real robot experiments."
    document = FulltextDocument(
        paper_id="p-alpha-unknown",
        paper_version_id="p-alpha-unknown:v1",
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        parser_version="fixture-parser-v1",
        license_allowed=True,
    )
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="fulltext",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=rows,
        top_k=4,
        fulltext_documents={document.paper_id: document},
        fulltext_budget=1,
    )
    item = next(value for value in bundle.items if value.paper_id == document.paper_id)
    assert item.level == RankingLevel.HIGH
    assert bundle.costs.fulltext_documents == 1
    assert any(
        record.paper_id == document.paper_id and record.source_type == "fulltext"
        for record in bundle.evidence_matrix
    )


def test_fulltext_hash_mismatch_is_rejected(intent_fixture, candidate_fixture):
    rows = copy.deepcopy(candidate_fixture)
    unknown = next(row for row in rows if row["canonical_paper_id"] == "p-alpha-unknown")
    unknown["oa_status"] = "open"
    unknown["pdf_url"] = "https://example.org/p-alpha-unknown.pdf"
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    with pytest.raises(ValueError, match="content hash mismatch"):
        _pipeline().rank(
            session_id="bad-fulltext",
            query=intent_fixture["query"],
            understanding=understanding,
            candidate_rows=rows,
            top_k=4,
            fulltext_documents={
                "p-alpha-unknown": FulltextDocument(
                    paper_id="p-alpha-unknown",
                    paper_version_id="p-alpha-unknown:v1",
                    text="real robot",
                    content_hash="0" * 64,
                    parser_version="fixture-parser-v1",
                    license_allowed=True,
                )
            },
            fulltext_budget=1,
        )


def test_calibration_is_query_grouped_and_prefers_precision_on_tie():
    examples = [
        CalibrationExample(query_id="q1", paper_id="a", cross_logit=3.0, relevant=True),
        CalibrationExample(query_id="q1", paper_id="b", cross_logit=1.0, relevant=False),
        CalibrationExample(query_id="q2", paper_id="c", cross_logit=2.0, relevant=True),
        CalibrationExample(query_id="q2", paper_id="d", cross_logit=0.0, relevant=False),
    ]
    profile = calibrate_threshold(examples, model_revision="rev", version="dev-v1")
    assert profile.min_cross_logit == 2.0
    assert profile.example_count == 4
    assert len(profile.development_data_hash) == 64


def test_submission_adapter_requires_and_verifies_frozen_bundle(
    intent_fixture,
    candidate_fixture,
):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    draft = _pipeline().rank(
        session_id="freeze",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    with pytest.raises(ValueError, match="only accepts a frozen"):
        build_submission_prediction(
            draft,  # type: ignore[arg-type]
            SubmissionPolicy(adapter_version="submission-v1"),
        )
    frozen = freeze_bundle_for_controller(draft)
    assert isinstance(frozen, FinalRankingBundle)
    prediction = build_submission_prediction(
        frozen,
        SubmissionPolicy(adapter_version="submission-v1"),
    )
    assert prediction.predicted_relevant_set == draft.high
    assert len(prediction.prediction_hash) == 64
