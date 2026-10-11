"""Mock academic search data for local development and demos."""

from backend.app.models.schemas import (
    AgentTraceStep,
    GraphEdge,
    GraphNode,
    PaperItem,
    QueryUnderstanding,
    RelationGraph,
    RelevanceLevel,
    SearchResponse,
    SearchSummary,
)


def build_mock_search_response(query: str, max_papers: int = 12) -> SearchResponse:
    understanding = QueryUnderstanding(
        intent="复杂学术查询：智能论文搜索、查询分解与多策略检索推荐",
        domain="NLP / Information Retrieval",
        topics=["学术论文搜索", "LLM Agent"],
        entities=["学术论文搜索", "LLM Agent", "查询分解", "引文网络"],
        methods=["多 Agent 协作", "查询改写", "迭代检索", "相关性重排序"],
        datasets=["RealScholarQuery", "AutoScholar", "PaperFindingBench", "LitSearch"],
        venues=["ACL", "EMNLP", "NeurIPS", "arXiv"],
        time_range="2022-2025",
        keywords=["paper search", "LLM agent", "query decomposition"],
        language="zh",
        sub_queries=[
            "LLM agent for academic paper search",
            "query decomposition for scholarly retrieval",
            "citation network exploration for literature discovery",
            "ranking relevance of scientific papers with LLMs",
        ],
        rewritten_queries=[
            "comprehensive academic paper search agent crawler selector",
            "scholarly retrieval query reformulation RefChain",
            "multi-index semantic paper finder citation chasing",
        ],
        raw_json={"mode": "mock"},
    )

    papers = [
        PaperItem(
            id="p1",
            title="PaSa: An LLM Agent for Comprehensive Academic Paper Search",
            abstract=(
                "提出 Crawler+Selector 双 Agent 架构，结合强化学习优化学术检索，"
                "在 RealScholarQuery 上显著超越 Google+GPT-4o 基线。"
            ),
            authors=["He Y.", "Huang G.", "Feng P.", "et al."],
            year=2025,
            published_at="2025",
            venue="ACL 2025",
            citation_count=186,
            relevance=RelevanceLevel.HIGH,
            relevance_score=0.96,
            url="https://arxiv.org/abs/2501.10120",
            tags=["Agent", "RL", "Paper Search"],
            source="mock",
        ),
        PaperItem(
            id="p2",
            title="SPAR: Scholar Paper Retrieval with LLM-based Agents",
            abstract=(
                "通过 RefChain 查询分解与查询演化，构建多 Agent 学术检索流程，"
                "在 AutoScholar 上取得更高 F1。"
            ),
            authors=["Shi X.", "Li Y.", "Kou Q.", "et al."],
            year=2025,
            venue="arXiv",
            citation_count=42,
            relevance=RelevanceLevel.HIGH,
            relevance_score=0.93,
            url="https://arxiv.org/abs/2507.15245",
            tags=["Query Decomposition", "Multi-Agent"],
        ),
        PaperItem(
            id="p3",
            title="LitSearch: A Retrieval Benchmark for Scientific Literature Search",
            abstract=(
                "面向科学文献检索构建评测基准，系统分析检索模型在细粒度研究问题上的表现。"
            ),
            authors=["Ajith A.", "Xia M.", "Chevalier A.", "et al."],
            year=2024,
            venue="EMNLP 2024",
            citation_count=128,
            relevance=RelevanceLevel.HIGH,
            relevance_score=0.88,
            url="https://arxiv.org/abs/2407.18940",
            tags=["Benchmark", "IR"],
        ),
        PaperItem(
            id="p4",
            title="AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite",
            abstract=(
                "构建科学科研场景下的 Agent 评测套件，覆盖文献搜索、推理与科研工作流任务。"
            ),
            authors=["Feldman S.", "et al."],
            year=2025,
            venue="arXiv",
            citation_count=31,
            relevance=RelevanceLevel.PARTIAL,
            relevance_score=0.74,
            url="https://arxiv.org/abs/2510.21652",
            tags=["Benchmark", "Agent Eval"],
        ),
        PaperItem(
            id="p5",
            title="Demonstrate-Search-Predict: Composing Retrieval and Language Models",
            abstract=(
                "提出 DSP 框架，将检索与语言模型组合用于知识密集型 NLP，"
                "为学术检索中的多跳推理提供方法论基础。"
            ),
            authors=["Khattab O.", "Santhanam K.", "Li X.", "et al."],
            year=2022,
            venue="arXiv",
            citation_count=920,
            relevance=RelevanceLevel.PARTIAL,
            relevance_score=0.71,
            url="https://arxiv.org/abs/2212.14024",
            tags=["RAG", "Composition"],
        ),
        PaperItem(
            id="p6",
            title="Language Agents for Answering Questions from Scientific Literature",
            abstract=(
                "研究面向科学文献问答的语言 Agent，强调证据检索、阅读与答案生成闭环。"
            ),
            authors=["Skarlinski M.", "et al."],
            year=2024,
            venue="NeurIPS 2024",
            citation_count=67,
            relevance=RelevanceLevel.PARTIAL,
            relevance_score=0.69,
            url=None,
            tags=["QA", "Scientific Agent"],
        ),
        PaperItem(
            id="p7",
            title="AGILE: A Novel Framework of LLM Agents",
            abstract=(
                "提出 AGILE Agent 框架，强调规划、工具调用与反馈迭代，"
                "可迁移到学术搜索策略优化场景。"
            ),
            authors=["Feng P.", "He Y.", "Huang G.", "et al."],
            year=2024,
            venue="NeurIPS 2024",
            citation_count=55,
            relevance=RelevanceLevel.PARTIAL,
            relevance_score=0.66,
            url=None,
            tags=["Agent Framework"],
        ),
        PaperItem(
            id="p8",
            title="GritLM: Generative Representational Instruction Tuning",
            abstract=(
                "统一生成与表征学习，可用于学术文献的语义检索与查询理解增强。"
            ),
            authors=["Muennighoff N.", "et al."],
            year=2024,
            venue="arXiv",
            citation_count=410,
            relevance=RelevanceLevel.PARTIAL,
            relevance_score=0.61,
            url="https://arxiv.org/abs/2402.09906",
            tags=["Embedding", "Retrieval"],
        ),
    ][:max_papers]

    graph = RelationGraph(
        nodes=[
            GraphNode(id="q", name="用户查询", category="query", value=18),
            GraphNode(id="sq1", name="查询分解", category="subquery", value=10),
            GraphNode(id="sq2", name="查询改写", category="subquery", value=10),
            GraphNode(id="sq3", name="引文追踪", category="subquery", value=10),
            GraphNode(id="m1", name="双Agent检索", category="method", value=8),
            GraphNode(id="m2", name="RefChain", category="method", value=8),
            GraphNode(id="m3", name="多索引语义检索", category="method", value=8),
            GraphNode(id="d1", name="RealScholarQuery", category="dataset", value=7),
            GraphNode(id="d2", name="AutoScholar", category="dataset", value=7),
            GraphNode(id="p1", name="PaSa", category="paper", value=14, paper_id="p1"),
            GraphNode(id="p2", name="SPAR", category="paper", value=13, paper_id="p2"),
            GraphNode(id="p3", name="LitSearch", category="paper", value=11, paper_id="p3"),
            GraphNode(id="p4", name="AstaBench", category="paper", value=9, paper_id="p4"),
            GraphNode(id="p5", name="DSP", category="paper", value=8, paper_id="p5"),
        ],
        edges=[
            GraphEdge(source="q", target="sq1", relation="分解", weight=1.0),
            GraphEdge(source="q", target="sq2", relation="改写", weight=1.0),
            GraphEdge(source="q", target="sq3", relation="扩展", weight=0.9),
            GraphEdge(source="sq1", target="p1", relation="检索命中", weight=1.0),
            GraphEdge(source="sq1", target="p2", relation="检索命中", weight=0.95),
            GraphEdge(source="sq2", target="p3", relation="检索命中", weight=0.9),
            GraphEdge(source="sq3", target="p4", relation="引文扩展", weight=0.8),
            GraphEdge(source="p1", target="m1", relation="方法", weight=1.0),
            GraphEdge(source="p2", target="m2", relation="方法", weight=1.0),
            GraphEdge(source="p3", target="d1", relation="评测", weight=0.85),
            GraphEdge(source="p1", target="d1", relation="评测", weight=1.0),
            GraphEdge(source="p2", target="d2", relation="评测", weight=1.0),
            GraphEdge(source="p5", target="m3", relation="启发", weight=0.7),
            GraphEdge(source="p1", target="p2", relation="主题相关", weight=0.75),
            GraphEdge(source="p2", target="p3", relation="主题相关", weight=0.7),
            GraphEdge(source="p3", target="p4", relation="评测相关", weight=0.65),
        ],
    )

    summary = SearchSummary(
        overview=(
            f"针对查询「{query}」，系统完成了意图解析、子查询分解、多轮检索与综合排序。"
            "当前结果以 LLM Agent 学术搜索、查询演化与评测基准为主线，覆盖高相关与部分相关论文。"
        ),
        key_findings=[
            "双 Agent（Crawler+Selector）与查询分解是提升召回的主流路径",
            "查询改写/演化能显著改善复杂细粒度科研问题的检索质量",
            "评测集正从通用 IR 转向 RealScholarQuery / AutoScholar 等学术专用基准",
        ],
        research_gaps=[
            "多维约束（方法/数据集/venue/时间）的联合解析仍不足",
            "权威性、时效性、多样性之间的动态权衡缺少统一目标函数",
            "低成本迭代检索策略（控制 API/Token）仍有优化空间",
        ],
    )

    agent_trace = [
        AgentTraceStep(step="understand", detail="解析核心意图、实体与约束条件", status="done"),
        AgentTraceStep(step="decompose", detail="拆解为 4 个子查询并生成改写查询", status="done"),
        AgentTraceStep(step="retrieve", detail="按子查询执行多策略检索（mock）", status="done"),
        AgentTraceStep(step="filter", detail="过滤低相关/低质量候选", status="done"),
        AgentTraceStep(step="rank", detail="基于标题与摘要进行细粒度相关性排序", status="done"),
        AgentTraceStep(step="synthesize", detail="归纳结构化结果并构建关系图", status="done"),
    ]

    return SearchResponse(
        query=query,
        understanding=understanding,
        papers=papers,
        graph=graph,
        summary=summary,
        agent_trace=agent_trace,
        meta={
            "mode": "mock",
            "model": "mock-local",
            "candidate_count": len(papers),
            "high_relevance_count": sum(1 for p in papers if p.relevance == RelevanceLevel.HIGH),
        },
    )
