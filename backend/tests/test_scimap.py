from __future__ import annotations

from backend.app.models.evidrank import EvidenceLabel, RankingLevel
from backend.app.models.scimap import SynthesisClaim
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.organizer.claim_validator import validate_claims
from backend.app.services.organizer.comparability import build_comparability_assessments
from backend.app.services.organizer.comparison_matrix import build_comparison_matrix
from backend.app.services.organizer.evidence_graph import build_evidence_graph
from backend.app.services.organizer.ranked_list import build_ranked_list
from backend.app.services.organizer.scimap import build_output_artifact
from backend.app.services import output_artifact_store
from backend.app.services.output_artifact_store import validate_artifact_envelope
from backend.app.services.ranking.adapters import research_intent_from_understanding
from backend.app.services.ranking.bundle_builder import freeze_bundle_for_controller
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.app.services.ranking.submission_adapter import (
    SubmissionPolicy,
    build_submission_prediction,
)
from backend.tests.conftest import FakeEncoder, FakeReranker


def _draft_and_intent(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=DeterministicEvidenceJudge(),
    )
    draft = pipeline.rank(
        session_id="scimap-test",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    intent = research_intent_from_understanding(
        session_id="scimap-test",
        query=intent_fixture["query"],
        understanding=understanding,
    )
    return draft, intent


def test_scimap_requires_frozen_bundle_and_hash_is_deterministic(
    intent_fixture,
    candidate_fixture,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    try:
        build_output_artifact(draft, intent)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("SciMap accepted a draft bundle")

    frozen = freeze_bundle_for_controller(draft)
    first = build_output_artifact(frozen, intent)
    second = build_output_artifact(frozen, intent)
    assert first.artifact_hash == second.artifact_hash


def test_unknown_or_conflicting_facet_never_creates_fact_edge_or_citation(
    intent_fixture,
    candidate_fixture,
):
    draft, _ = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    paper_id = "p-alpha-real"
    facet_id = "real_robot"
    support = next(
        record
        for record in frozen.evidence_matrix
        if record.paper_id == paper_id
        and record.facet_id == facet_id
        and record.label == EvidenceLabel.SUPPORT
    )
    violation = support.model_copy(
        update={
            "evidence_id": "conflicting-fulltext-evidence",
            "label": EvidenceLabel.VIOLATE,
            "source_type": "fulltext",
        }
    )
    items = []
    for item in frozen.items:
        if item.paper_id == paper_id:
            labels = dict(item.facet_labels)
            labels[facet_id] = EvidenceLabel.UNKNOWN
            item = item.model_copy(
                update={
                    "facet_labels": labels,
                    "evidence_conflict_facets": [facet_id],
                }
            )
        items.append(item)
    conflicted = frozen.model_copy(
        update={
            "items": items,
            "evidence_matrix": [*frozen.evidence_matrix, violation],
        }
    )

    graph = build_evidence_graph(conflicted)
    assert not any(
        edge.source == f"paper:{paper_id}" and edge.target == f"facet:{facet_id}"
        for edge in graph.edges
    )
    card = next(value for value in build_ranked_list(conflicted) if value.paper_id == paper_id)
    assert not any(citation.facet_id == facet_id for citation in card.citations)


def test_dynamic_facet_is_present_and_unknown_is_not_filled(
    intent_fixture,
    candidate_fixture,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    intent = intent.model_copy(
        update={
            "dynamic_facets": {
                **intent.dynamic_facets,
                "evaluation_protocol": {
                    "facet_id": "evaluation_protocol",
                    "label": "Evaluation protocol",
                    "target_count": 1,
                },
            }
        }
    )
    matrix = build_comparison_matrix(frozen, intent)
    assert "evaluation_protocol" in matrix.facets
    assert all(
        row.cells["evaluation_protocol"].status == EvidenceLabel.UNKNOWN
        for row in matrix.rows
    )


def test_claim_validator_rejects_hallucinated_references(
    intent_fixture,
    candidate_fixture,
):
    draft, _ = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    claims = [
        SynthesisClaim(
            claim_id="bad-claim",
            claim_type="evidence",
            text="Unsupported claim",
            paper_ids=["missing-paper"],
            evidence_ids=["missing-evidence"],
            strong_assertion=True,
            assertion_level="A",
        )
    ]
    checked, summary = validate_claims(claims, frozen)
    assert checked[0].validation_status == "invalid"
    assert {
        "UNKNOWN_PAPER_REFERENCE",
        "UNKNOWN_EVIDENCE_REFERENCE",
    }.issubset(checked[0].reason_codes)
    assert summary.invalid_claims == 1


def test_claim_validator_rejects_semantically_mismatched_evidence(
    intent_fixture,
    candidate_fixture,
):
    draft, _ = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    support = next(
        record
        for record in frozen.evidence_matrix
        if record.label == EvidenceLabel.SUPPORT and record.validation_status == "valid"
    )
    claim = SynthesisClaim(
        claim_id="false-disagreement",
        claim_type="disagreement",
        text="Claims disagreement from support evidence only",
        paper_ids=[support.paper_id],
        evidence_ids=[support.evidence_id],
        strong_assertion=True,
        assertion_level="C",
    )

    checked, summary = validate_claims([claim], frozen)

    assert checked[0].validation_status == "invalid"
    assert "DISAGREEMENT_REQUIRES_OPPOSING_EVIDENCE" in checked[0].reason_codes
    assert summary.invalid_claims == 1


def test_scimap_does_not_change_submission_set_or_infer_chronology_relations(
    intent_fixture,
    candidate_fixture,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    before = build_submission_prediction(
        frozen,
        SubmissionPolicy(adapter_version="test-v1"),
    ).predicted_relevant_set
    artifact = build_output_artifact(frozen, intent)
    after = build_submission_prediction(
        frozen,
        SubmissionPolicy(adapter_version="test-v1"),
    ).predicted_relevant_set
    assert before == after
    assert all(edge.relation in {"supports", "violates"} for edge in artifact.evidence_graph.edges)


def test_output_artifact_persistence_envelope_rejects_tampering(
    intent_fixture,
    candidate_fixture,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    artifact = build_output_artifact(frozen, intent)
    validate_artifact_envelope(
        session_id=frozen.session_id,
        bundle=frozen,
        artifact=artifact,
    )
    tampered = artifact.model_copy(update={"artifact_hash": "tampered"})
    try:
        validate_artifact_envelope(
            session_id=frozen.session_id,
            bundle=frozen,
            artifact=tampered,
        )
    except ValueError as exc:
        assert "hash verification" in str(exc)
    else:
        raise AssertionError("tampered OutputArtifact was accepted")


def test_comparability_defaults_to_insufficient_when_context_is_unknown(
    intent_fixture,
    candidate_fixture,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    items = []
    for item in draft.items:
        if item.paper_id == "p-alpha-unknown":
            item = item.model_copy(update={"level": RankingLevel.PARTIAL})
        items.append(item)
    draft = draft.model_copy(
        update={
            "items": items,
            "partial": [*draft.partial, "p-alpha-unknown"],
            "verify": [value for value in draft.verify if value != "p-alpha-unknown"],
        }
    )
    frozen = freeze_bundle_for_controller(draft)
    assessments = build_comparability_assessments(frozen, intent)
    assert assessments
    pair = next(
        value
        for value in assessments
        if {value.left_paper_id, value.right_paper_id}
        == {"p-alpha-real", "p-alpha-unknown"}
    )
    assert pair.status == "insufficient_evidence"
    assert pair.allowed_claims == []


def test_persistence_conflict_is_idempotent_only_for_the_same_hash(
    intent_fixture,
    candidate_fixture,
    monkeypatch,
):
    draft, intent = _draft_and_intent(intent_fixture, candidate_fixture)
    frozen = freeze_bundle_for_controller(draft)
    artifact = build_output_artifact(frozen, intent)

    class FakeCursor:
        def __init__(self, existing_hash):
            self.existing_hash = existing_hash
            self.next_row = None
            self.statements = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, statement, parameters=None):
            sql = str(statement)
            self.statements.append(sql)
            if "INSERT INTO output_artifacts" in sql:
                self.next_row = None
            elif "SELECT artifact_hash" in sql:
                self.next_row = (self.existing_hash,)

        def fetchone(self):
            row = self.next_row
            self.next_row = None
            return row

    class FakeConnection:
        def __init__(self, existing_hash):
            self.fake_cursor = FakeCursor(existing_hash)
            self.commits = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def cursor(self):
            return self.fake_cursor

        def commit(self):
            self.commits += 1

    same = FakeConnection(artifact.artifact_hash)
    monkeypatch.setattr(output_artifact_store, "_connect", lambda: same)
    output_artifact_store.save_output_artifact(
        session_id=frozen.session_id,
        bundle=frozen,
        artifact=artifact,
    )
    assert same.commits == 1
    assert any("ON CONFLICT" in sql for sql in same.fake_cursor.statements)

    different = FakeConnection("different-hash")
    monkeypatch.setattr(output_artifact_store, "_connect", lambda: different)
    try:
        output_artifact_store.save_output_artifact(
            session_id=frozen.session_id,
            bundle=frozen,
            artifact=artifact,
        )
    except RuntimeError as exc:
        assert "cannot overwrite immutable OutputArtifact" in str(exc)
    else:
        raise AssertionError("different hash overwrote an immutable artifact")
