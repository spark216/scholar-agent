"""LLM-based academic query intent parsing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from backend.app.config import get_settings
from backend.app.models.schemas import QueryUnderstanding


class IntentClassificationError(RuntimeError):
    """Raised when the configured LLM cannot provide a valid intent decision."""


def _adaptive_query_system_prompt() -> str:
    return """你是学术检索查询规划器。根据原问题、尚未覆盖的约束和已找到论文的主题线索，生成下一轮唯一检索式。
只输出严格 JSON：
{"query":"英文或中英文学术检索式","reason":"不超过40字的调整原因"}
要求：
1. 保留原问题的核心任务、方法、数据集和明确时间/语言约束；
2. 针对覆盖缺口换用术语、缩写或论文常见表达，不要重复已执行查询；
3. 不加入原问题没有要求的会议、期刊、年份或研究领域；
4. 查询不超过300字符，不输出布尔数组、Markdown或解释。"""


def evolve_query_with_llm(
    *,
    raw_query: str,
    understanding: QueryUnderstanding,
    base_query: str,
    coverage_gaps: list[str],
    candidate_cues: list[str],
    executed_queries: list[str],
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Generate one bounded, drift-checked adaptive query.

    Failure is non-fatal: the graph retains its deterministic evolution path.
    """
    settings = get_settings()
    metrics: dict[str, Any] = {
        "api_calls": 0,
        "token_usage": 0,
        "model": settings.zhipu_intent_model,
        "prompt_version": settings.agent_search_llm_evolution_prompt_version,
    }
    if not settings.zhipuai_api_key:
        return None, None, {**metrics, "error": "missing_api_key"}

    payload = {
        "raw_query": raw_query,
        "intent": understanding.intent,
        "topics": understanding.topics,
        "entities": understanding.entities,
        "methods": understanding.methods,
        "datasets": understanding.datasets,
        "venues": understanding.venues,
        "time_range": understanding.time_range,
        "language": understanding.language,
        "base_query": base_query,
        "coverage_gaps": coverage_gaps[:6],
        "candidate_cues": candidate_cues[:6],
        "executed_queries": executed_queries[-6:],
    }
    try:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=settings.zhipuai_api_key)
        response = client.chat.completions.create(
            model=settings.zhipu_intent_model,
            messages=[
                {"role": "system", "content": _adaptive_query_system_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=settings.agent_search_llm_evolution_max_tokens,
        )
        metrics["api_calls"] = 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            metrics["token_usage"] = int(getattr(usage, "total_tokens", 0) or 0)
        raw = _extract_json(response.choices[0].message.content or "")
        query = " ".join(str(raw.get("query") or "").split()).strip()
        reason = " ".join(str(raw.get("reason") or "").split()).strip()
    except Exception as exc:  # noqa: BLE001
        return None, None, {**metrics, "error": type(exc).__name__}

    if not query or len(query) > 300:
        return None, None, {**metrics, "error": "invalid_query_length"}
    executed = {" ".join(value.split()).casefold() for value in executed_queries if value}
    if query.casefold() in executed:
        return None, None, {**metrics, "error": "duplicate_query"}

    # Require at least one explicit intent anchor to prevent an unconstrained
    # model rewrite from drifting into a nearby but different research topic.
    anchors = [
        *understanding.datasets,
        *understanding.methods,
        *understanding.entities,
        *understanding.topics,
        *understanding.keywords,
    ]
    anchor_terms = {
        token.casefold()
        for value in [raw_query, *anchors]
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]{1,}", str(value))
        if len(token) >= 3
    }
    query_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]{1,}", query)
    }
    if anchor_terms and not anchor_terms.intersection(query_terms):
        return None, None, {**metrics, "error": "query_drift"}
    return query, reason or "LLM 根据覆盖缺口调整检索式", metrics


def _current_year() -> int:
    return datetime.now().year


def _intent_system_prompt() -> str:
    year = _current_year()
    return f"""你是学术检索意图解析器。请将用户输入解析为严格 JSON（不要 markdown，不要多余解释）。
当前本地年份是 {year}。
JSON schema:
{{
  "is_paper_search": true,
  "reject_reason": null,
  "intent": "一句话概括用户检索意图",
  "domain": "研究领域，如 NLP / CV / Health",
  "topics": ["主题1", "主题2"],
  "entities": ["关键实体"],
  "methods": ["方法约束"],
  "datasets": ["数据集约束"],
  "venues": ["会议/期刊约束"],
  "time_range": "时间范围，如 2018-{year}；没有则 null",
  "keywords": ["中英文关键词均可"],
  "language": "zh+en / zh / en",
  "sub_queries": ["2-3 个可独立检索的子问题，覆盖查询的不同方面（方法/数据集/任务/对比角度），不是对原查询的简单改写"],
  "rewritten_queries": ["改写后的检索词，建议同时包含中英文表达"],
  "relations": [
    {{"source": "实体/任务", "relation": "关系", "target": "实体/任务", "required": true}}
  ],
  "constraints": [
    {{
      "id": "C001",
      "facet": "year/language/method/dataset/venue/publication_type 或查询特有维度",
      "operator": "eq/gte/lte/contains/not_contains/in",
      "value": "约束值",
      "hardness": "hard/soft/preference",
      "polarity": "positive/negative",
      "scope": "global/core/background",
      "source_span": {{"quoted_text": "用户原话中的精确片段"}}
    }}
  ],
  "dynamic_facets": {{
    "自定义维度ID": {{
      "label": "展示名称",
      "values": ["期望值"],
      "is_hard": false,
      "target_count": 1
    }}
  }},
  "background_policy": "allow_partial",
  "output_intent": ["ranked_list", "comparison_matrix", "evidence_synthesis"],
  "parser_version": "query-understanding-v3-contract-aligned"
}}
合法性判断（最重要，先判断）：
- 必须根据完整语义判断，禁止因为某一个词（例如“天气”“代码”“翻译”）直接分类
- 若用户是在查找/推荐/综述学术论文、文献、研究工作，或在本学术搜索产品中输入一个明确研究主题，
  则 is_paper_search=true，reject_reason=null
- 若用户是在询问即时信息、闲聊、要求完成非文献任务或无关问答，则 is_paper_search=false，
  reject_reason 用中文简要说明，其余字段可置空/null
对比例子：
- “今天天气怎么样” => false（即时天气查询）
- “上海天气变化相关研究论文” => true（明确检索研究论文）
- “西瓜增产的方法” => true（在学术搜索产品中表达研究主题）
- “帮我写一个天气 App” => false（代码生成任务）
- “weather forecasting papers using deep learning” => true（明确检索论文）
- “你是什么模型” => false（闲聊/身份问题）
其他要求（仅当 is_paper_search=true 时）：
1. 时间范围必须基于“当前本地年份 {year}”理解相对时间：
   - “2018年后 / since 2018 / 2018以后” => "2018-{year}"
   - “19年后 / 17年后”这类两位年份 => "2019-{year}" / "2017-{year}"
   - “近三年” => "{year-2}-{year}"
2. rewritten_queries 中的年份必须与 time_range 完全一致；禁止把“19年后”改写成“17年后”等错误年份
3. language 表示用户要求的论文语言，而不是检索词语言：
   - “中文论文 / 中文文献 / 只要中文” => "zh"
   - “英文论文 / 英文文献 / English papers only” => "en"
   - 用户没有明确限定论文语言 => "zh+en"
4. rewritten_queries 至少 2 条。即使论文语言被限定，也可生成中英文检索词扩大召回，
   但不得因此把 language 从 "zh" 或 "en" 改成 "zh+en"
5. sub_queries 必须 2-3 条，且必须覆盖查询的不同方面（例如按方法、数据集、任务、
   对比角度、子主题各自拆出一个可独立检索的子问题），不能只是 rewritten_queries 的重复；
   复杂组合查询尤其要分解（示例："用 QAT 做低比特量化且与 SFT 对比" => 子问题1 "QAT 低比特量化的优势"、
   子问题2 "QAT 与 SFT 的对比"）
6. 只输出 JSON 对象
7. 只有用户明确表达“必须、只要、排除、不能、年份上下界、语言限定”等条件时，
   才能生成 hardness="hard"；不得把模型自行推测的偏好升级为硬约束
8. negative 只表示用户明确排除的内容；证据不足必须保留为 unknown，不能解释成违反
9. source_span.quoted_text 必须逐字来自用户输入；无法定位时留空对象
10. 没有明确 relation、constraint 或 dynamic facet 时分别输出空数组/空对象，不得补造
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError("LLM 未返回合法 JSON")


def _extract_start_year_from_query(query: str) -> int | None:
    """Prefer year cues in the raw user query over model guesses."""
    year_now = _current_year()
    m = re.search(
        r"(20\d{2})\s*年?\s*(?:后|以后|之后|以来)|(?:since|after)\s+(20\d{2})",
        query,
        re.I,
    )
    if m:
        return int(m.group(1) or m.group(2))

    m2 = re.search(r"(?<!\d)(\d{2})\s*年?\s*(?:后|以后|之后|以来)", query)
    if m2:
        yy = int(m2.group(1))
        century = (year_now // 100) * 100
        start = century + yy
        if start > year_now:
            start -= 100
        return start
    return None


def _normalize_time_range(time_range: str | None, query: str) -> str | None:
    year_now = _current_year()

    # 1) Query year cues always win (avoid model rewriting 19年后 -> 2017).
    query_start = _extract_start_year_from_query(query)
    if query_start is not None:
        return f"{query_start}-{year_now}"

    # 2) Fall back to model-provided time_range.
    if time_range:
        years = [int(y) for y in re.findall(r"(20\d{2})", time_range)]
        if len(years) >= 2:
            start, end = min(years), max(years)
            if end > year_now:
                end = year_now
            return f"{start}-{end}"
        if len(years) == 1:
            return f"{years[0]}-{year_now}"

    years = [int(y) for y in re.findall(r"(20\d{2})", query)]
    if len(years) == 1:
        return f"{years[0]}-{year_now}"
    if len(years) >= 2:
        return f"{min(years)}-{min(max(years), year_now)}"
    return time_range


def _parse_time_bounds(time_range: str | None) -> tuple[int | None, int | None]:
    if not time_range:
        return None, None
    years = [int(y) for y in re.findall(r"(20\d{2})", time_range)]
    if len(years) >= 2:
        return min(years), max(years)
    if len(years) == 1:
        return years[0], years[0]
    return None, None


def _align_text_to_time_range(text: str, start: int, end: int | None) -> str:
    """Force year mentions in rewritten text to match normalized time_range."""
    # 2017年后 / 2017年以后 -> 2019年后
    text = re.sub(
        r"(20\d{2})(\s*年?\s*(?:后|以后|之后|以来))",
        lambda m: f"{start}{m.group(2)}",
        text,
    )
    # since/after 2017 -> since/after 2019
    text = re.sub(
        r"((?:since|after)\s+)(20\d{2})",
        lambda m: f"{m.group(1)}{start}",
        text,
        flags=re.I,
    )
    # 17年后 -> 2019年后
    text = re.sub(
        r"(?<!\d)(\d{2})(\s*年?\s*(?:后|以后|之后|以来))",
        lambda m: f"{start}{m.group(2)}",
        text,
    )
    # "past/last 19 years" mistaken for "19年后" -> after 2019
    text = re.sub(
        r"(?:over\s+the\s+)?(?:past|last)\s+\d{1,2}\s+years?",
        f"after {start}",
        text,
        flags=re.I,
    )
    text = re.sub(r"近\d{1,2}年", f"{start}年后", text)
    # 2017-2025 / 2017至2025 -> 2019-2026
    if end is not None:
        text = re.sub(
            r"20\d{2}\s*[-–—至到]\s*20\d{2}",
            f"{start}-{end}",
            text,
        )
    return text


def _align_rewritten_queries(
    rewritten: list[str],
    time_range: str | None,
    query: str,
) -> list[str]:
    start, end = _parse_time_bounds(time_range)
    if start is None:
        return rewritten

    aligned = [_align_text_to_time_range(item, start, end) for item in rewritten]
    deduped: list[str] = []
    for item in aligned:
        item = item.strip()
        if item and item not in deduped:
            deduped.append(item)

    topic = re.sub(r"(?<!\d)\d{2}\s*年?\s*(?:后|以后|之后|以来)[，,\s]*", "", query).strip("，, ")
    topic = re.sub(r"20\d{2}\s*年?\s*(?:后|以后|之后|以来)[，,\s]*", "", topic).strip("，, ")
    zh_fallback = f"{start}年后{topic}" if topic else f"{start}年后相关研究"
    en_fallback = f"{topic} after {start} academic paper" if topic else f"research after {start}"

    if not any(str(start) in item for item in deduped):
        deduped = [zh_fallback, en_fallback] + deduped
    if len(deduped) < 2:
        deduped = list(dict.fromkeys([*deduped, zh_fallback, en_fallback]))
    return deduped[:6]


def _normalize_language(language: str | None, query: str) -> str:
    raw = (language or "").strip().lower()
    q = query.lower()
    explicit_zh = bool(
        re.search(
            r"(?:只要|仅限?|只看|限定|要求)?\s*中文\s*(?:论文|文献|期刊|文章|研究|资料)?",
            query,
        )
    )
    explicit_en = bool(
        re.search(
            r"(?:只要|仅限?|只看|限定|要求)?\s*英文\s*(?:论文|文献|期刊|文章|研究|资料)?",
            query,
        )
        or re.search(
            r"\b(?:english\s+(?:papers?|literature|articles?)|"
            r"(?:papers?|literature|articles?)\s+in\s+english|english\s+only)\b",
            q,
        )
    )

    if explicit_zh and not explicit_en:
        return "zh"
    if explicit_en and not explicit_zh:
        return "en"
    if explicit_zh and explicit_en:
        return "zh+en"

    if raw in {"zh", "en"}:
        # A single-language model answer without a matching user constraint must
        # not silently discard papers in the other language.
        return "zh+en"
    if raw in {"zh+en", "zh-en", "both", "bilingual", "中英", "中英文"}:
        return "zh+en"
    # Default: bilingual when unspecified.
    return "zh+en"


def _normalize_evidrank_contract(raw: dict[str, Any], query: str) -> None:
    """Sanitize LLM facets and prevent untraceable hard constraints."""
    normalized_constraints: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    allowed_operators = {"eq", "gte", "lte", "contains", "not_contains", "in"}
    for index, value in enumerate(raw.get("constraints") or []):
        if not isinstance(value, dict) or not value.get("facet") or value.get("value") is None:
            continue
        constraint_id = str(value.get("id") or f"C{index + 1:03d}")
        if constraint_id in seen_ids:
            constraint_id = f"C{index + 1:03d}"
        seen_ids.add(constraint_id)
        source_span = value.get("source_span") if isinstance(value.get("source_span"), dict) else {}
        quoted_text = str(source_span.get("quoted_text") or "").strip()
        if not quoted_text or quoted_text not in query:
            source_span = {}
        hardness = str(value.get("hardness") or "soft").lower()
        if hardness not in {"hard", "soft", "preference"}:
            hardness = "soft"
        if hardness == "hard" and not source_span:
            hardness = "soft"
        polarity = str(value.get("polarity") or "positive").lower()
        if polarity not in {"positive", "negative"}:
            polarity = "positive"
        operator = str(value.get("operator") or "contains").lower()
        if operator not in allowed_operators:
            operator = "contains"
        normalized_constraints.append(
            {
                "id": constraint_id,
                "facet": str(value["facet"]),
                "operator": operator,
                "value": value["value"],
                "hardness": hardness,
                "polarity": polarity,
                "scope": str(value.get("scope") or "global"),
                "source_span": source_span,
            }
        )
    raw["constraints"] = normalized_constraints

    raw["relations"] = [
        {
            "source": str(value["source"]),
            "relation": str(value.get("relation") or value.get("type")),
            "target": str(value["target"]),
            "required": bool(value.get("required", True)),
        }
        for value in (raw.get("relations") or [])
        if isinstance(value, dict)
        and value.get("source")
        and (value.get("relation") or value.get("type"))
        and value.get("target")
    ]
    if not isinstance(raw.get("dynamic_facets"), dict):
        raw["dynamic_facets"] = {}
    raw["background_policy"] = str(raw.get("background_policy") or "allow_partial")
    raw["output_intent"] = [
        str(value)
        for value in (raw.get("output_intent") or ["ranked_list"])
        if str(value).strip()
    ]
    raw["parser_version"] = "query-understanding-v3-contract-aligned"


def _align_explicit_constraints(
    raw: dict[str, Any],
    query: str,
    *,
    time_range: str | None,
    language: str,
) -> None:
    """Make formal hard constraints agree with normalized user-authored cues."""
    start_year = _extract_start_year_from_query(query)
    range_start, _range_end = _parse_time_bounds(time_range)
    if start_year is None:
        start_year = range_start

    for constraint in raw.get("constraints") or []:
        if not isinstance(constraint, dict):
            continue
        facet = str(constraint.get("facet") or "").casefold()
        source_span = constraint.get("source_span") or {}
        quoted_text = str(source_span.get("quoted_text") or "")

        # A traceable raw-query year wins over an inconsistent model value.
        if (
            facet == "year"
            and start_year is not None
            and quoted_text
            and quoted_text in query
            and re.search(r"\d{2,4}\s*年?", quoted_text)
        ):
            constraint["value"] = start_year
            if re.search(r"后|以后|之后|以来|起|since|after", quoted_text, re.I):
                constraint["operator"] = "gte"

        # Likewise, an explicit language request is authoritative over aliases
        # or accidental language drift in the model-generated contract.
        if (
            facet == "language"
            and language in {"zh", "en"}
            and quoted_text
            and quoted_text in query
        ):
            constraint["value"] = language
            constraint["operator"] = "eq"


def parse_query_intent(query: str) -> tuple[QueryUnderstanding, dict[str, Any]]:
    """Use the configured LLM as the single authority for intent classification."""
    settings = get_settings()
    meta: dict[str, Any] = {
        "intent_provider": "zhipu",
        "api_calls": 0,
        "token_usage": 0,
        "local_year": _current_year(),
    }
    if not settings.zhipuai_api_key:
        raise IntentClassificationError(
            "未配置 ZHIPUAI_API_KEY，无法由 LLM 判断是否为论文检索"
        )

    try:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=settings.zhipuai_api_key)
        resp = client.chat.completions.create(
            model=settings.zhipu_intent_model,
            messages=[
                {"role": "system", "content": _intent_system_prompt()},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or ""
        raw = _extract_json(content)
        meta["api_calls"] = 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["token_usage"] = int(getattr(usage, "total_tokens", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        raise IntentClassificationError(
            f"LLM 意图识别失败（{type(exc).__name__}），未执行本地关键词猜测"
        ) from exc

    if not isinstance(raw.get("is_paper_search"), bool):
        raise IntentClassificationError(
            "LLM 意图识别结果缺少布尔字段 is_paper_search"
        )

    is_paper_search = raw["is_paper_search"]
    reject_reason = (
        None
        if is_paper_search
        else str(raw.get("reject_reason") or "请输入查找论文相关问题")
    )

    if not is_paper_search:
        understanding = QueryUnderstanding(
            intent=str(raw.get("intent") or "非论文检索问题"),
            is_paper_search=False,
            reject_reason=reject_reason,
            language="zh+en",
            raw_json={**raw, "is_paper_search": False, "reject_reason": reject_reason},
        )
        meta["is_paper_search"] = False
        return understanding, meta

    _normalize_evidrank_contract(raw, query)

    time_range = _normalize_time_range(raw.get("time_range"), query)
    language = _normalize_language(raw.get("language"), query)
    _align_explicit_constraints(
        raw,
        query,
        time_range=time_range,
        language=language,
    )
    rewritten = list(raw.get("rewritten_queries") or [])
    if language == "zh+en" and len(rewritten) < 2:
        rewritten = list(dict.fromkeys(rewritten + [query, f"{query} academic paper"]))
    rewritten = _align_rewritten_queries(rewritten, time_range, query)
    # Also fix year drift in other free-text fields.
    start, end = _parse_time_bounds(time_range)
    intent = str(raw.get("intent") or "")
    if start is not None:
        intent = _align_text_to_time_range(intent, start, end)
        sub_queries = [
            _align_text_to_time_range(str(q), start, end)
            for q in list(raw.get("sub_queries") or [])
        ]
        keywords = [
            _align_text_to_time_range(str(k), start, end)
            for k in list(raw.get("keywords") or [])
        ]
    else:
        sub_queries = list(raw.get("sub_queries") or [])
        keywords = list(raw.get("keywords") or [])

    # Fallback: if the parser did not decompose the query into independent
    # sub-problems, derive them from the rewritten queries so the local and
    # network retrieval still get multiple search angles.
    if not sub_queries and rewritten:
        sub_queries = [str(value) for value in rewritten if str(value).strip()][:3]
    sub_queries = list(dict.fromkeys(sub_queries))[:6]

    raw["is_paper_search"] = True
    raw["reject_reason"] = None
    raw["time_range"] = time_range
    raw["language"] = language
    raw["rewritten_queries"] = rewritten
    raw["sub_queries"] = sub_queries
    raw["keywords"] = keywords
    raw["intent"] = intent
    raw["local_year"] = _current_year()

    understanding = QueryUnderstanding(
        intent=intent,
        is_paper_search=True,
        reject_reason=None,
        domain=raw.get("domain"),
        topics=list(raw.get("topics") or []),
        entities=list(raw.get("entities") or []),
        methods=list(raw.get("methods") or []),
        datasets=list(raw.get("datasets") or []),
        venues=list(raw.get("venues") or []),
        time_range=time_range,
        keywords=keywords,
        language=language,
        sub_queries=sub_queries,
        rewritten_queries=rewritten,
        raw_json=raw,
    )
    meta["is_paper_search"] = True
    return understanding, meta
