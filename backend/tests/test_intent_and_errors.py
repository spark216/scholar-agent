"""Regression tests for academic intent detection and API error mapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from backend.app.api.routes import health, search_papers
from backend.app.models.schemas import QueryUnderstanding, SearchRequest
from backend.app.services.llm_intent import (
    IntentClassificationError,
    evolve_query_with_llm,
    parse_query_intent,
)
from backend.app.services.search_service import InvalidPaperQueryError, run_paper_search


def _intent_settings(api_key="test-key"):
    return SimpleNamespace(
        zhipuai_api_key=api_key,
        zhipu_intent_model="test-intent-model",
    )


def _intent_client(content):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content))
        ],
        usage=SimpleNamespace(total_tokens=10),
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)
        )
    )


class LLMIntentClassificationTests(unittest.TestCase):
    def test_adaptive_query_uses_coverage_gaps_and_reports_cost(self):
        settings = SimpleNamespace(
            zhipuai_api_key="test-key",
            zhipu_intent_model="test-intent-model",
            agent_search_llm_evolution_prompt_version="adaptive-query-v1",
            agent_search_llm_evolution_max_tokens=240,
        )
        content = (
            '{"query":"HotPotQA large language model experimental evaluation",'
            '"reason":"补充数据集实验表达"}'
        )
        with (
            patch("backend.app.services.llm_intent.get_settings", return_value=settings),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            query, reason, meta = evolve_query_with_llm(
                raw_query="LLM experiments on HotPotQA",
                understanding=QueryUnderstanding(
                    methods=["large language model"],
                    datasets=["HotPotQA"],
                ),
                base_query="LLM HotPotQA",
                coverage_gaps=["experimental evaluation"],
                candidate_cues=["question answering"],
                executed_queries=["LLM HotPotQA"],
            )

        self.assertIn("HotPotQA", query)
        self.assertIn("实验", reason)
        self.assertEqual(meta["api_calls"], 1)
        self.assertEqual(meta["token_usage"], 10)

    def test_adaptive_query_rejects_topic_drift(self):
        settings = SimpleNamespace(
            zhipuai_api_key="test-key",
            zhipu_intent_model="test-intent-model",
            agent_search_llm_evolution_prompt_version="adaptive-query-v1",
            agent_search_llm_evolution_max_tokens=240,
        )
        with (
            patch("backend.app.services.llm_intent.get_settings", return_value=settings),
            patch(
                "zhipuai.ZhipuAI",
                return_value=_intent_client(
                    '{"query":"protein folding clinical trial","reason":"扩展"}'
                ),
            ),
        ):
            query, _reason, meta = evolve_query_with_llm(
                raw_query="HotPotQA language model evaluation",
                understanding=QueryUnderstanding(datasets=["HotPotQA"]),
                base_query="HotPotQA evaluation",
                coverage_gaps=["results"],
                candidate_cues=[],
                executed_queries=["HotPotQA evaluation"],
            )

        self.assertIsNone(query)
        self.assertEqual(meta["error"], "query_drift")

    def test_explicit_two_digit_year_repairs_inconsistent_hard_constraint(self):
        content = (
            '{"is_paper_search": true,'
            '"intent": "检索2019年后水稻增产方法的英文论文",'
            '"time_range": "2019-2026",'
            '"language": "en",'
            '"methods": ["Agricultural techniques"],'
            '"constraints": ['
            '{"id": "C1", "facet": "year", "operator": "gte",'
            '"value": 2017, "hardness": "hard", "polarity": "positive",'
            '"source_span": {"quoted_text": "19年后"}},'
            '{"id": "C2", "facet": "language", "operator": "eq",'
            '"value": "zh", "hardness": "hard", "polarity": "positive",'
            '"source_span": {"quoted_text": "英文论文"}}],'
            '"rewritten_queries": ["rice yield methods after 2017"]}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            understanding, _meta = parse_query_intent(
                "19年后，水稻增产的方法，英文论文"
            )

        constraints = {
            value["facet"]: value
            for value in understanding.raw_json["constraints"]
        }
        self.assertEqual(understanding.time_range, "2019-2026")
        self.assertEqual(constraints["year"]["value"], 2019)
        self.assertEqual(constraints["language"]["value"], "en")
        self.assertIn("2019", understanding.rewritten_queries[0])

    def test_untraceable_hard_constraint_is_downgraded(self):
        content = (
            '{"is_paper_search": true,'
            '"intent": "检索机器人论文",'
            '"constraints": ['
            '{"id": "C1", "facet": "language", "operator": "eq",'
            '"value": "en", "hardness": "hard", "polarity": "positive",'
            '"source_span": {"quoted_text": "只要英文论文"}},'
            '{"id": "C2", "facet": "year", "operator": "gte",'
            '"value": 2020, "hardness": "hard", "polarity": "positive",'
            '"source_span": {"quoted_text": "2020年后"}}]}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            understanding, _meta = parse_query_intent("查找2020年后的机器人论文")

        constraints = understanding.raw_json["constraints"]
        assert constraints[0]["hardness"] == "soft"
        assert constraints[0]["source_span"] == {}
        assert constraints[1]["hardness"] == "hard"
        assert constraints[1]["source_span"]["quoted_text"] == "2020年后"

    def test_explicit_chinese_paper_constraint_is_preserved(self):
        content = (
            '{"is_paper_search": true,'
            '"reject_reason": null,'
            '"intent": "检索2020年后玉米产量增值方法的中文论文",'
            '"time_range": "2020-2026",'
            '"language": "zh",'
            '"rewritten_queries": ["玉米产量增值方法 中文论文",'
            '"maize yield improvement academic papers"]}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            understanding, _meta = parse_query_intent(
                "20年后玉米产量增值的方法，中文论文"
            )

        self.assertEqual(understanding.language, "zh")
        self.assertEqual(understanding.raw_json["language"], "zh")

    def test_llm_accepts_weather_research_query_without_local_blacklist(self):
        content = (
            '{"is_paper_search": true,'
            '"reject_reason": null,'
            '"intent": "检索上海天气变化研究论文",'
            '"domain": "Climate Science",'
            '"topics": ["上海天气变化"],'
            '"rewritten_queries": ["上海天气变化研究论文",'
            '"Shanghai weather change research papers"]}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            understanding, meta = parse_query_intent("上海天气变化相关研究论文")

        self.assertTrue(understanding.is_paper_search)
        self.assertIsNone(understanding.reject_reason)
        self.assertEqual(meta["intent_provider"], "zhipu")

    def test_llm_rejection_is_not_overridden_by_local_keywords(self):
        content = (
            '{"is_paper_search": false,'
            '"reject_reason": "用户要求的是种植建议而不是论文检索",'
            '"intent": "普通农业建议"}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=_intent_client(content)),
        ):
            understanding, _meta = parse_query_intent("西瓜增产的方法")

        self.assertFalse(understanding.is_paper_search)
        self.assertIn("种植建议", understanding.reject_reason)

    def test_missing_llm_key_fails_instead_of_using_local_keyword_rules(self):
        with patch(
            "backend.app.services.llm_intent.get_settings",
            return_value=_intent_settings(api_key=""),
        ):
            with self.assertRaisesRegex(
                IntentClassificationError,
                "无法由 LLM 判断",
            ):
                parse_query_intent("上海天气变化相关研究论文")

    def test_requires_explicit_boolean_decision_from_llm(self):
        client = _intent_client(
            '{"is_paper_search": "true", "intent": "检索论文"}'
        )
        with (
            patch(
                "backend.app.services.llm_intent.get_settings",
                return_value=_intent_settings(),
            ),
            patch("zhipuai.ZhipuAI", return_value=client),
        ):
            with self.assertRaisesRegex(
                IntentClassificationError,
                "is_paper_search",
            ):
                parse_query_intent("上海天气变化相关研究论文")


class SynchronousSearchErrorTests(unittest.TestCase):
    def test_health_checks_milvus_without_initializing_embedding_store(self):
        client = SimpleNamespace(
            list_collections=lambda: [],
            close=lambda: None,
        )
        with (
            patch("pymilvus.MilvusClient", return_value=client) as milvus_client,
            patch("backend.app.api.routes.pg_health", return_value="ok"),
            patch(
                "backend.app.services.milvus_store.get_milvus_store",
            ) as embedding_store,
        ):
            response = health()

        self.assertEqual(response.milvus, "ok")
        milvus_client.assert_called_once()
        embedding_store.assert_not_called()

    def test_invalid_event_raises_specific_service_error(self):
        events = iter(
            [
                {
                    "type": "invalid_query",
                    "message": "请输入查找论文相关问题",
                    "reject_reason": "这不是论文检索问题",
                }
            ]
        )
        with patch(
            "backend.app.services.search_service.iter_paper_search_events",
            return_value=events,
        ):
            with self.assertRaisesRegex(InvalidPaperQueryError, "这不是论文检索问题"):
                run_paper_search("你是谁")

    def test_api_maps_invalid_query_to_422(self):
        with patch(
            "backend.app.api.routes.run_paper_search",
            side_effect=InvalidPaperQueryError("请输入查找论文相关问题"),
        ):
            with self.assertRaises(HTTPException) as caught:
                search_papers(SearchRequest(query="你是谁"))

        self.assertEqual(caught.exception.status_code, 422)
        self.assertEqual(caught.exception.detail, "请输入查找论文相关问题")


if __name__ == "__main__":
    unittest.main()
