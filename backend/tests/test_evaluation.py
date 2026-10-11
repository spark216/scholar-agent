from backend.scripts.evaluate_evidrank import evaluate_bundle
from backend.tests.test_pipeline import _pipeline


def test_evaluation_reports_quality_and_safety_metrics(intent_fixture, candidate_fixture):
    from backend.app.models.schemas import QueryUnderstanding

    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="evaluation-test",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    report = evaluate_bundle(
        bundle,
        {
            "p-alpha-real": 2,
            "p-alpha-unknown": 1,
            "p-alpha-old": 0,
            "p-alpha-sim": 0,
        },
        k=3,
    )
    assert report["ndcg_at_k"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mrr_at_k"] == 1.0
    assert report["hard_violation_rate_at_k"] == 0.0
