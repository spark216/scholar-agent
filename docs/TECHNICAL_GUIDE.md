# ScholarAgent：可审计论文检索、排序与证据化归纳

本仓库使用 Vue 3、FastAPI 与 LangGraph 实现论文检索闭环。当前默认主链为：

```text
查询理解与约束抽取
  → 自适应多源召回与候选去重
  → 功能3 EvidRank（身份/元数据/硬约束/混合排序/证据判定）
  → 主控制器冻结 FinalRankingBundle
  → 功能4 SciMap（证据卡片/比较矩阵/可比性/时间线/证据图/保守归纳）
  → PostgreSQL 持久化与 Vue 展示
```

数字模型分数均保留原始语义：Dense、Sparse、RRF、Cross-Encoder logit 和 sigmoid 展示值不是相关概率。最终 `high / partial / verify / excluded` 等级由排序位置、决策策略、硬约束和可定位证据共同决定。

## 功能3：EvidRank 论文综合排序

功能3源码位于 `backend/app/services/ranking/`，核心合同位于 `backend/app/models/evidrank.py`。

执行顺序：

1. 将功能1/2输出适配为 `ResearchIntent` 与去重后的 `CandidatePaper` 快照。
2. 核对论文规范身份、版本、来源和字段级 provenance；冲突或弱来源不会被伪装成可靠事实。
3. 对可由可信元数据证明的硬约束执行非补偿门控；语义约束证据不足时进入 `DEFER`，不靠关键词误杀。
4. 使用固定 revision 的 BGE-M3 对查询以及标题、摘要生成 Dense + Sparse 分数，再进行 RRF 融合。
5. 使用固定 revision 的 BGE Reranker v2 M3 对候选池前部进行 Cross-Encoder 精排；未精排尾部保留 RRF 次序。
6. 对 Top-K 证据池执行确定性精确片段判定，并可用智谱模型升级 `unknown`；LLM 输出必须通过结构和原文片段校验。
7. 控制器可提供已经合法获取、版本和哈希均匹配的全文，在预算内处理高价值不确定项；功能3本身不下载 PDF。
8. 生成完整 `RankingBundle`、覆盖缺口和不确定性队列。控制器确认检索图结束后才冻结为带哈希的 `FinalRankingBundle`。

`max_papers` 只限制公开列表的 UI 投影，不截断用于审计的完整 RankingBundle。

默认使用 `EVIDRANK_DECISION_POLICY=recall_first`：证据未知不会自动进入待核验，弱来源或元数据不完整降为部分相关，只有身份、版本、元数据或证据发生明确冲突才待核验；撤稿、重复版本和可信元数据证明的硬条件违反仍然排除。需要保守审计时可切换为 `strict`，恢复“关键证据未知即待核验”的原始规则。两种策略都保留完整证据标签、不确定性队列和决策策略元数据。

### 检索召回与效率策略

- 每轮选中的 Serper Scholar、Semantic Scholar、arXiv、OpenAlex、本地 PaSa 语料和可选 PubMed 并行调用；Semantic Scholar 使用论文搜索接口发现新候选，API Key 可选但建议配置，并通过独立请求节流、公共重试、熔断与持久化缓存控制 429 和延迟；
- 若存在 PaSa `paper_database/id2paper.json`，首次使用会在
  `backend/paper-test/index/` 建立 SQLite FTS5 索引；若同目录存在 `cs_paper_2nd.zip`，
  索引同时纳入摘要，并使用 Porter 词干支持 `evaluate/evaluated/evaluating` 等英文词形。
  它不读取逐查询 gold/qrels，首次构建后直接复用；
- 可选运行 `python -m backend.scripts.build_pasa_vector_index`，将约 57 万篇
  PaSa 标题与摘要离线写入独立的 Milvus BGE-M3 dense+sparse 集合。构建完成后设置
  `ACADEMIC_LOCAL_VECTOR_ENABLED=1`，该全库索引会作为零 API 成本的首轮 Provider；
  未构建时保持关闭，不影响现有 FTS 和网络来源；
- 网络来源共用 HTTP keep-alive 连接池，对瞬时连接错误执行一次有界重试；同一来源连续失败时
  暂时熔断，避免测试集运行中反复等待相同超时；
- 查询编译器保留数据集、缩写、实体和方法的精确分支；正常搜索仍按覆盖情况早停，
  评测模式在预算允许时至少执行 2 个不同 query variant，且第二页请求受 Agent 总 API 预算约束；
- arXiv 使用停用词过滤后的短语、核心术语和实体/方法 OR 分支，不再把自然语言前 8 个词全部设为硬 AND；
- PubMed 仅路由给医学、生物、药物、疾病等查询，通用 CS 查询减少一次无效 API 调用；
- Tavily 默认作为覆盖不足时的回退来源；首轮学术来源数量与约束覆盖充分时跳过，减少慢请求；
- 正常搜索的 BGE-M3 编码候选默认 80 篇、最多 100 篇；评测使用独立的
  120/160 候选档位，不牺牲交互检索速度。
  其中 60% 名额优先保留数据集/方法精确匹配、查询词覆盖和多来源共同命中的论文，
  其余名额按真实 Provider 排名保护来源与检索轮次多样性；选择结果不再受并发接口返回顺序影响；
- Reranker 正常搜索处理 RRF 前 40 篇，评测处理前 80 篇，并融合
  Cross-Encoder、Dense/Sparse RRF 与明确的
  数据集/方法/正向关系匹配；Cross 前 3 名受保护，让第 6–30 位的强条件命中有机会进入前 5；
- Dense/Sparse 编码和 Cross-Encoder 结果使用进程内 LRU 缓存；评测模式还把意图与网络
  Provider 响应写入带版本和 TTL 的本地 SQLite 缓存，重复测试不再重复消耗 API；
- 评测面板可先用“建立/更新快照”联网记录 Provider 返回，再用“冻结快照复测”在完全
  相同的 Provider 候选上比较排序改动；冻结模式缺少缓存时会停止，不会降级成少来源低分报告；
- 候选预算先用 `BGE_CANDIDATE_PROVIDER_HEAD_FRACTION` 和
  `BGE_CANDIDATE_PROVIDER_HEAD_PER_SOURCE` 保护各 Provider 头部结果，再执行全局相关性与
  来源/轮次多样性选择，降低高流量来源挤掉其他来源正确论文的概率；
- `recall_first` 模式下，明确排除仍置底，
  其余论文主要按 Cross-Encoder/RRF 相关性排序，避免“证据未知”压过高相关论文；
- LLM 证据判定池默认按 `max_papers × 1` 动态缩放，同时受
  `EVIDRANK_EVIDENCE_TOP_K` 绝对上限保护；
- 全文核验默认预算为 0，只有控制器显式提供合法全文和预算时才执行；
- `PAPER_TEST_FAST_MODE=1` 时评测保留确定性证据和全部检索排序，但跳过不会改变 ID
  匹配指标的远程 LLM 证据升级；正常论文搜索仍执行完整 EvidRank；
- 每次评测报告同时保存数据集 SHA-256、Provider 候选快照、配置指纹、Git revision/dirty
  状态、模型 revision、候选预算、检索轮数和阈值版本；前端只对同数据、同配置且同
  Provider 快照的历史运行计算 ΔF1；
- 评测报告同时保存平均值、P50/P95 端到端耗时及 API/LLM Token 总量，便于按赛题的
  运行效率权重比较优化前后结果；
- ECharts 关系图与 SciMap 面板使用动态导入，不阻塞首页主包加载。

## 功能4：SciMap 搜索结果归纳整理

功能4源码位于 `backend/app/services/organizer/`，核心合同位于 `backend/app/models/scimap.py`。

SciMap 只接受已经冻结且哈希校验通过的 FinalRankingBundle，并生成：

- 带论文、分面和证据引用的排序卡片；
- 查询动态分面的比较矩阵，未知值保持 `unknown`；
- 实验上下文可比性审查，证据不足时禁止直接优劣结论；
- 仅描述论文年份和证据分布的保守时间线，不臆造技术演进关系；
- 只含 `supports / violates` 事实边的证据图；
- 支持、冲突、缺口和不可比结论及 claim-level 校验结果；
- 带 `artifact_hash` 的 `OutputArtifact`，与输入 `bundle_hash / intent_hash` 绑定。

## 模型与 hf-xet

模型权重不进入 Git 仓库，默认固定为：

| 用途 | Hugging Face 模型 | revision |
|---|---|---|
| Dense + Sparse 编码 | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| Cross-Encoder 精排 | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

`hf-xet` 不是模型，而是 Hugging Face Hub 用于大文件传输和缓存的客户端依赖。联网部署首次运行会下载权重；Docker 使用 `hf_cache` 数据卷复用缓存。离线部署应挂载已下载的模型目录，并设置：

```env
BGE_M3_MODEL_PATH=/models/bge-m3
BGE_RERANKER_MODEL_PATH=/models/bge-reranker-v2-m3
BGE_LOCAL_FILES_ONLY=1
```

不要把 `.bin`、`.safetensors` 或 Hugging Face 缓存提交到 Git。

## 本地启动

Python 3.11/3.12 均可。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

前端：

```powershell
Set-Location frontend
npm ci
npm run dev
```

- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>
- 同步搜索：`POST /api/search`
- 流式搜索：`POST /api/search/stream`
- 前端开发地址：<http://127.0.0.1:5173>

## PostgreSQL 迁移

应用当前会在首次持久化时以幂等 DDL 建表。正式集成和多实例部署建议由 Git/部署负责人按顺序执行：

```text
backend/migrations/000_base.sql
backend/migrations/001_evidrank.sql
backend/migrations/002_output_artifacts.sql
```

顺序不能颠倒，因为后两份迁移引用基础会话表和排序运行表。

## Docker

```bash
cp .env.example .env
docker compose up -d
docker compose --profile app up -d --build
```

基础设施模式启动 PostgreSQL、Redis、Milvus、etcd 和 MinIO；`app` profile 额外启动前后端。具体说明见 `DOCKER.md`。

## 验证

```powershell
python -m pytest backend\tests -q
Set-Location frontend
npm ci
npm run build
```

当前自动测试覆盖规则、合同、哈希、证据片段、全文校验、缓存、检索代理、API错误处理、证据预算和 SciMap 产物。自动测试通过不等于真实效果已经达到竞赛一等奖：合并后仍必须用真实固定 revision 权重、真实 PostgreSQL/Milvus、真实外部 API 和人工标注查询集完成端到端评测。详细交付顺序和验收边界见 `FEATURE_3_4_HANDOFF.md`。

## 竞赛评测口径

项目必须分开报告三条证据线，不能互相替代：

1. 固定语料精确 ID 评测：完全离线，使用 corpus document ID 与 qrels 精确匹配，用于召回/排序消融；
2. 冻结 Provider 评测：先 `record` 再 `frozen`，在同一候选快照上比较代码改动；
3. 开放网络压力测试：报告外部 API 失败率、延迟、调用量和 Token，不冒充固定 benchmark 成绩。

固定语料评测示例（`$DATASET_ROOT` 指向一个含 `corpus.jsonl`、`queries.jsonl` 和 qrels 的数据目录）：

```powershell
python -m backend.scripts.evaluate_fixed_corpus `
  --corpus "$DATASET_ROOT/corpus.jsonl" `
  --queries "$DATASET_ROOT/queries.jsonl" `
  --qrels "$DATASET_ROOT/qrels-dev.tsv" `
  --split dev `
  --profiles fts bge_dense bge_rrf reranker evidrank `
  --candidate-pool 200 --reranker-pool 80 --k 5 10 20 `
  --output ".modeling-audit/fixed_dev_ablation.json"
```

测试集只能使用开发集已冻结的配置：

```powershell
python -m backend.scripts.evaluate_fixed_corpus `
  --corpus "$DATASET_ROOT/corpus.jsonl" `
  --queries "$DATASET_ROOT/queries.jsonl" `
  --qrels "$DATASET_ROOT/qrels-test.tsv" `
  --split test `
  --profiles fts bge_dense bge_rrf reranker evidrank `
  --candidate-pool 200 --reranker-pool 80 --k 5 10 20 `
  --output ".modeling-audit/fixed_test_frozen.json"
```

注意：在这个评测器中只有 FTS 搜索全库，其余 profile 都在同一个 FTS 候选池内重排。因此它能判断排序增益，但不能证明全库 BGE-M3 召回增益。全库 dense+sparse 召回必须先运行 `backend/scripts/build_pasa_vector_index.py` 构建 Milvus 索引，再单独做启用/禁用对照。

当前已核实的 NFCorpus test FTS 基线（323 条、candidate pool=200、精确 doc ID qrels）为：候选 Recall 27.93%、F1@5 11.90%、nDCG@5 35.33%、MRR@5 51.89%、F1@20 11.40%、P95 31.05 ms，失败查询为 0。该结果是内部固定语料诊断，不是比赛排行榜成绩；它表明当前主要上限来自候选召回，而不是仅靠调整 Reranker 即可解决。

## SciMap 人工验收

功能4必须从冻结输出抽取至少100个不同查询，生成 A/B 两份隔离文件，由两名成员独立标注，分歧由第三人裁决：

```powershell
python -m backend.scripts.evaluate_scimap_annotations init --source backend/paper-test/annotations/scimap_units.jsonl --output-a backend/paper-test/annotations/scimap_annotator_a.csv --output-b backend/paper-test/annotations/scimap_annotator_b.csv --limit 100
python -m backend.scripts.evaluate_scimap_annotations merge --source-a backend/paper-test/annotations/scimap_annotator_a.csv --source-b backend/paper-test/annotations/scimap_annotator_b.csv --output backend/paper-test/annotations/scimap_adjudication.csv
python -m backend.scripts.evaluate_scimap_annotations score --source backend/paper-test/annotations/scimap_adjudication.csv --output backend/paper-test/annotations/scimap_human_metrics.json
```

输出必须同时报告样本设计、一致性、Cohen's kappa、置信区间、citation precision、unsupported assertion rate、comparability accuracy 和 citation accuracy。自动生成标签或合成压力测试不能写成真实人工验收成绩。
