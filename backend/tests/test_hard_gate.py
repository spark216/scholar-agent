from backend.app.models.evidrank import GateDecision
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.adapters import (
    candidates_from_rows,
    research_intent_from_understanding,
)
from backend.app.services.ranking.hard_gate import apply_hard_gate
from backend.app.services.ranking.identity_review import review_identity
from backend.app.services.ranking.metadata_validator import validate_metadata


def test_reliable_year_violation_rejects(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    intent = research_intent_from_understanding(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
    )
    papers = {p.canonical_paper_id: p for p in candidates_from_rows(candidate_fixture)}
    old = papers["p-alpha-old"]
    gate = apply_hard_gate(old, intent, review_identity(old), validate_metadata(old))
    assert gate.decision == GateDecision.REJECT
    assert "legacy-year-min" in gate.violated_constraint_ids


def test_semantic_hard_constraint_is_deferred_not_keyword_rejected(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    intent = research_intent_from_understanding(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
    )
    papers = {p.canonical_paper_id: p for p in candidates_from_rows(candidate_fixture)}
    simulated = papers["p-alpha-sim"]
    gate = apply_hard_gate(
        simulated,
        intent,
        review_identity(simulated),
        validate_metadata(simulated),
    )
    assert gate.decision == GateDecision.DEFER
    assert "C-no-sim" in gate.deferred_constraint_ids
