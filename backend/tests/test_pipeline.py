from backend.app.models.evidrank import RankingLevel
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.llm_evidence_judge import ZhipuEvidenceJudge
from backend.app.services.search_service import _to_paper_items
from backend.tests.conftest import (
    DisabledFakeSelector,
    FakeEncoder,
    FakeReranker,
    FakeSelector,
)


def _pipeline() -> EvidRankPipeline:
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        selector=DisabledFakeSelector(),
        evidence_judge=DeterministicEvidenceJudge(),
        decision_policy="strict",
    )
    pipeline.evidence_top_k = 20
    pipeline.high_rank_cutoff = 5
    return pipeline


def test_recall_first_policy_reduces_verify_and_semantic_exclusions(
    intent_fixture,
    candidate_fixture,
):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=DeterministicEvidenceJudge(),
        decision_policy="recall_first",
    )
    pipeline.evidence_top_k = 20
    pipeline.high_rank_cutoff = 5

    bundle = pipeline.rank(
        session_id="recall-first",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )

    levels = {item.paper_id: item.level for item in bundle.items}
    assert levels["p-alpha-real"] == RankingLevel.HIGH
    assert levels["p-alpha-unknown"] == RankingLevel.HIGH
    assert levels["p-alpha-sim"] == RankingLevel.PARTIAL
    # A trusted metadata violation remains a non-compensatory exclusion.
    assert levels["p-alpha-old"] == RankingLevel.EXCLUDED
    assert bundle.verify == []
    assert bundle.execution_metadata["decision_policy"] == "recall_first"
    visible = [item for item in bundle.items if item.level != RankingLevel.EXCLUDED]
    relevance_ranks = [
        item.scores.fusion_rank or item.scores.cross_rank or item.scores.rrf_rank
        for item in visible
    ]
    assert relevance_ranks == sorted(relevance_ranks)
    assert bundle.execution_metadata["final_sort_policy"] == "hard-exclusion-then-relevance"
    assert (
        bundle.execution_metadata["rank_fusion"]["policy"]
        == "weighted-cross-rrf-facet-preference-v3"
    )
    assert all(item.scores.fusion_rank is not None for item in visible)
    assert all(
        item.scores.fusion_rank == item.scores.cross_rank
        for item in visible
        if item.scores.cross_rank and item.scores.cross_rank <= pipeline.head_protect
    )


def test_rerank_query_omits_internal_policy_labels():
    class Intent:
        raw_query = "Papers evaluated on HotPotQA"
        datasets = ["HotPotQA"]
        methods = ["chain-of-thought"]
        tasks = ["question answering"]

    query = EvidRankPipeline._rerank_query(Intent())

    assert query.startswith("Papers evaluated on HotPotQA")
    assert "Background policy" not in query
    assert "chain-of-thought" in query
    assert query.casefold().count("hotpotqa") == 1


def test_exact_dataset_signal_can_enter_head_without_overriding_cross_top_three():
    class OrderedReranker:
        model_revision = "ordered-test"

        def rerank(self, query, papers, top_k, *, candidate_limit=None):
            del candidate_limit
            del query
            order = {
                "broad-1": 1,
                "broad-2": 2,
                "broad-3": 3,
                "broad-4": 4,
                "broad-5": 5,
                "exact": 6,
            }
            values = []
            for paper in papers:
                item = dict(paper)
                rank = order[item["id"]]
                item["cross_logit"] = float(10 - rank)
                item["cross_sigmoid"] = 0.9
                values.append(item)
            values.sort(key=lambda item: order[item["id"]])
            return values[:top_k], {
                "reranker": "ordered-test",
                "model_revision": self.model_revision,
                "applied": True,
                "cross_input_count": len(values),
            }

    candidates = [
        {
            "id": f"broad-{index}",
            "title": f"General language model study {index}",
            "abstract": "A broadly related language model paper.",
            "year": 2024,
            "source": "fixture",
        }
        for index in range(1, 6)
    ]
    candidates.append(
        {
            "id": "exact",
            "title": "Reasoning Evaluation on HotPotQA",
            "abstract": "We evaluate the method on the HotPotQA dataset.",
            "year": 2024,
            "source": "fixture",
        }
    )
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=OrderedReranker(),
        evidence_judge=DeterministicEvidenceJudge(),
        decision_policy="recall_first",
    )
    pipeline.head_protect = 3
    pipeline.facet_rank_weight = 0.15
    bundle = pipeline.rank(
        session_id="facet-head",
        query="Find papers evaluated on HotPotQA",
        understanding=QueryUnderstanding(datasets=["HotPotQA"]),
        candidate_rows=candidates,
        top_k=5,
        fast_evaluation=True,
    )

    exact = next(item for item in bundle.items if item.paper_id == "exact")
    assert exact.scores.cross_rank == 6
    assert exact.scores.facet_rank == 1
    assert exact.scores.fusion_rank == 4
    assert bundle.execution_metadata["rank_fusion"]["structured_signal_active"] is True


def test_fast_evaluation_skips_remote_evidence_judge(intent_fixture, candidate_fixture):
    class RemoteJudgeMustNotRun:
        provider = "remote-test"
        batch_size = 5

        def judge_batch(self, **_kwargs):
            raise AssertionError("remote evidence judge should be skipped")

    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=RemoteJudgeMustNotRun(),
        decision_policy="recall_first",
    )
    bundle = pipeline.rank(
        session_id="fast-eval",
        query=intent_fixture["query"],
        understanding=QueryUnderstanding.model_validate(intent_fixture["understanding"]),
        candidate_rows=candidate_fixture,
        top_k=4,
        fast_evaluation=True,
    )

    assert bundle.execution_metadata["fast_evaluation"] is True
    assert bundle.costs.evidence_api_calls == 0


def test_pipeline_non_compensatory_levels_and_draft_contract(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    levels = {item.paper_id: item.level for item in bundle.items}
    assert levels["p-alpha-real"] == RankingLevel.HIGH
    assert levels["p-alpha-old"] == RankingLevel.EXCLUDED
    assert levels["p-alpha-sim"] == RankingLevel.EXCLUDED
    assert levels["p-alpha-unknown"] == RankingLevel.VERIFY
    assert bundle.is_frozen is False
    assert bundle.frozen_at is None
    assert bundle.bundle_hash != "pending"
    assert bundle.verify == ["p-alpha-unknown"]
    assert any(item.reason_code.value == "FULLTEXT_REQUIRED" for item in bundle.uncertainty_queue) is False


def test_bundle_hash_is_reproducible_for_same_semantic_output(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    first = _pipeline().rank(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    second = _pipeline().rank(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    assert first.ranking_run_id != second.ranking_run_id
    assert first.bundle_hash == second.bundle_hash


def test_all_raw_score_families_remain_auditable(intent_fixture, candidate_fixture):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = _pipeline().rank(
        session_id="s1",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    item = next(value for value in bundle.items if value.paper_id == "p-alpha-real")
    assert item.scores.dense_score is not None
    assert item.scores.sparse_score is not None
    assert item.scores.rrf_score is not None
    assert item.scores.cross_logit == 8.0
    assert item.scores.cross_sigmoid is not None


def test_semantic_evidence_judge_runs_only_for_configured_top_pool(
    intent_fixture,
    candidate_fixture,
):
    class CountingJudge:
        provider = "counting-judge"

        def __init__(self):
            self.paper_ids = []
            self.base = DeterministicEvidenceJudge()

        def judge(self, *, session_id, paper, intent):
            self.paper_ids.append(paper.canonical_paper_id)
            return self.base.judge(session_id=session_id, paper=paper, intent=intent)

    judge = CountingJudge()
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=judge,
    )
    pipeline.evidence_top_k = 1
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    pipeline.rank(
        session_id="evidence-budget",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    assert judge.paper_ids == ["p-alpha-real"]


def test_semantic_evidence_pool_scales_with_requested_result_count(
    intent_fixture,
    candidate_fixture,
):
    class CountingJudge:
        provider = "counting-judge"

        def __init__(self):
            self.paper_ids = []
            self.base = DeterministicEvidenceJudge()

        def judge(self, *, session_id, paper, intent):
            self.paper_ids.append(paper.canonical_paper_id)
            return self.base.judge(session_id=session_id, paper=paper, intent=intent)

    judge = CountingJudge()
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=judge,
    )
    pipeline.evidence_top_k = 20
    pipeline.evidence_pool_multiplier = 2
    pipeline.high_rank_cutoff = 1
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])

    pipeline.rank(
        session_id="scaled-evidence-budget",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=1,
        include_rejected_in_scoring=True,
    )

    assert len(judge.paper_ids) == 2


def test_english_agricultural_method_query_resolves_locally_without_llm_call():
    understanding = QueryUnderstanding(
        intent="检索2019年后水稻增产方法的英文论文",
        topics=["Rice yield increase"],
        methods=["Agricultural techniques"],
        time_range="2019-2026",
        language="en",
        raw_json={
            "constraints": [
                {
                    "id": "year-min",
                    "facet": "year",
                    "operator": "gte",
                    "value": 2019,
                    "hardness": "hard",
                    "polarity": "positive",
                },
                {
                    "id": "language-en",
                    "facet": "language",
                    "operator": "eq",
                    "value": "en",
                    "hardness": "hard",
                    "polarity": "positive",
                },
            ]
        },
    )
    judge = ZhipuEvidenceJudge()
    judge.enabled = True
    judge.api_key = "must-not-be-used"

    def fail_request(*_args, **_kwargs):
        raise AssertionError("locally resolved evidence must not call the LLM")

    judge._request = fail_request
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=judge,
    )
    pipeline.high_rank_cutoff = 1
    bundle = pipeline.rank(
        session_id="rice-local-evidence",
        query="19年后，水稻增产的方法，英文论文",
        understanding=understanding,
        candidate_rows=[
            {
                "id": "rice-paper",
                "title": "Optimization of sowing dates for enhanced rice yield",
                "abstract": "Field experiments evaluate sowing and management strategies for rice yield.",
                "year": 2024,
                "source": "openalex",
            }
        ],
        top_k=1,
    )
    assert bundle.high == ["rice-paper"]
    assert bundle.costs.evidence_api_calls == 0


def test_missing_cross_score_is_not_replaced_by_fake_half_score(
    intent_fixture,
    candidate_fixture,
):
    class OptionalUnavailableReranker:
        model_revision = "unavailable"

        def rerank(self, query, papers, top_k, *, candidate_limit=None):
            del candidate_limit
            return papers[:top_k], {
                "reranker": "unavailable",
                "model_revision": self.model_revision,
                "applied": False,
            }

    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=OptionalUnavailableReranker(),
        evidence_judge=DeterministicEvidenceJudge(),
    )
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = pipeline.rank(
        session_id="no-fake-score",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
    )
    papers = _to_paper_items(bundle, 4)
    assert papers
    assert all(paper.relevance_score == 0.0 for paper in papers)
    assert all(paper.score_kind == "not_available" for paper in papers)
    assert bundle.degraded is True
    assert "RERANKER_NOT_APPLIED" in bundle.degradation_reasons


def test_selector_fusion_writes_scores_and_protects_selector_head(
    intent_fixture,
    candidate_fixture,
):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        selector=FakeSelector(),
        evidence_judge=DeterministicEvidenceJudge(),
        decision_policy="strict",
    )
    pipeline.evidence_top_k = 20
    pipeline.high_rank_cutoff = 5

    bundle = pipeline.rank(
        session_id="selector-on",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
        ablation={"selector": True},
    )

    sel_meta = bundle.execution_metadata["selector"]
    assert sel_meta["applied"] is True
    assert sel_meta["scored_count"] > 0
    assert bundle.stage_counts["selector_scored"] == sel_meta["scored_count"]
    assert bundle.execution_metadata["rank_fusion"]["selector_rank_weight"] > 0.0
    assert bundle.execution_metadata["rank_fusion"]["protected_selector_head"] > 0
    assert bundle.model_versions["selector"] == "fake-pasa-selector"
    # The selector's top-ranked papers are protected at the fusion head.
    items_by_id = {item.paper_id: item for item in bundle.items}
    for item in bundle.items:
        if item.scores.sel_score is not None:
            assert item.scores.sel_rank is not None
    protected = bundle.execution_metadata["rank_fusion"]["protected_selector_head"]
    assert protected >= 1


def test_selector_disabled_leaves_pipeline_unchanged(
    intent_fixture,
    candidate_fixture,
):
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    pipeline = _pipeline()  # uses DisabledFakeSelector

    bundle = pipeline.rank(
        session_id="selector-off",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
        ablation={"selector": True},
    )

    sel_meta = bundle.execution_metadata["selector"]
    assert sel_meta["applied"] is False
    assert bundle.execution_metadata["rank_fusion"]["selector_rank_weight"] == 0.0
    assert bundle.execution_metadata["rank_fusion"]["protected_selector_head"] == 0
    assert bundle.stage_counts["selector_scored"] == 0
    assert "SELECTOR_NOT_APPLIED" not in bundle.degradation_reasons
    assert bundle.model_versions["selector"] == "disabled"
