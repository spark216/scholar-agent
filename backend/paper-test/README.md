# ScholarAgent Paper Test

这个目录保存 ScholarAgent 的论文检索评测配置和本地报告。后端通过 PaSa、
AstaBench PaperFinding、TREC Precision Medicine 和 BEIR 读取查询及评分标准，调用与网页检索相同的
`run_paper_search` 链路，并报告：

- Precision@K
- Recall@K
- F1@K
- nDCG@K
- MRR@K
- 单查询和整体运行时间

当前支持：

- PaSa RealScholarQuery Test（50 条）；
- PaSa AutoScholarQuery Test（1,000 条）；
- PaSa AutoScholarQuery Dev（1,000 条）；
- AstaBench PaperFinding Validation（66 条）；
- AstaBench PaperFinding Test（267 条）；
- TREC Precision Medicine 2017/2018/2020（30/50/31 条）；
- BEIR SciFact、NFCorpus、ArguAna、TREC-COVID、SciDocs（共 3,074 条）。

TREC PM 使用 `answer_pmid` 与检索结果 PMID 一对一精确匹配。BEIR 当前运行的是
开放 Provider 泛化评测，通过规范化标题匹配；它不是限定在原始 `corpus.jsonl` 内的
BEIR 官方封闭语料成绩，不能直接与 BEIR leaderboard 数值比较。ArguAna 只作为论证
检索压力测试，FiQA 因不是论文语料且标题为空而不接入。

AstaBench 数据固定在 revision
`a600dc767f850385f4664772e3ba7a7f8be17d5e`。Semantic 查询使用相关性 Judge 和
官方 normalizer 计算 adjusted F1、estimated recall 与 rank/nDCG；Specific/Metadata
查询使用 Semantic Scholar Corpus ID 精确匹配。Corpus ID 通过 ScholarAgent 已有的
DOI/arXiv/PMID 一次批量解析，避免逐篇 API 请求。

## 数据集位置

后端按以下顺序查找数据集：

1. 环境变量 `PAPER_TEST_DATASET_ROOT`；
2. `backend/paper-test/datasets/`；
3. `~/Downloads/paper-dataset/pasa-dataset/`。
4. `~/Downloads/dataset/`（自动读取其中的 `eval/*.jsonl`）。

`PAPER_TEST_DATASET_ROOT` 可以指向 `paper-dataset`，也可以直接指向
`pasa-dataset`、新的 `dataset` 包或直接指向它的 `eval` 目录。大体积数据集不应提交到 Git。

如果数据中存在 `paper_database/id2paper.json`，检索后端会将这份约 57 万篇的
arXiv ID—标题语料构建成本地 SQLite FTS5 索引；同目录存在 `cs_paper_2nd.zip`
时还会匹配并索引摘要。它是候选语料而不是逐查询答案，不会读取 `answer` 或
`answer_arxiv_id`。索引位于 `backend/paper-test/index/`、已被 Git 忽略，可通过
`ACADEMIC_LOCAL_CORPUS_PATH`、`ACADEMIC_LOCAL_DOCUMENTS_PATH` 和
`ACADEMIC_LOCAL_INDEX_PATH` 显式指定源文件、论文正文归档及缓存位置。

AstaBench 首次运行会使用 `HF_ACCESS_TOKEN` 或 `HF_TOKEN` 下载固定版本数据；也会
读取 `paper-dataset/asta-bench-main/.env` 中已存在的对应配置。语义 Judge 使用项目
现有的智谱配置，Corpus ID 解析优先使用 `ASTA_TOOL_KEY`。

## API

- `GET /api/paper-test/datasets`：查看可用测试集；
- `GET /api/paper-test/latest`：读取最近一次报告；
- `GET /api/paper-test/history?limit=20&offset=0`：分页读取全部本地历史摘要；
- `GET /api/paper-test/history/{run_id}`：读取某次完整历史报告；
- `POST /api/paper-test/run`：同步运行；
- `POST /api/paper-test/run/stream`：SSE 实时运行，网页使用该接口。

请求示例：

```json
{
  "dataset": "pasa-real",
  "sample_limit": 3,
  "ks": [5, 10],
  "seed": 42,
  "use_mock": false
}
```

每次完成的报告独立写入 `results/runs/<timestamp>_<run_id>.json`，并同步更新
`results/latest.json`。这些本地运行产物被 Git 忽略；旧版本已有的 `latest.json` 会在
首次读取历史时自动归档，不会被后续运行覆盖丢失。

新报告还会记录每条 gold 在 `provider -> merged -> budget -> ranking -> final`
中的首次出现阶段、损失阶段、完整排序命中位置和各阶段 Candidate Recall。报告也保存
每轮 Provider 的查询表达式、返回数量、耗时与错误，用于区分来源召回、合并、候选裁剪、
排序和外部 API 故障。

历史摘要同时返回最大 K、随机种子、来源阶段 Candidate Recall 和平均耗时；前端只在
测试集、样本数、随机种子与最大 K 相同时计算相对上一轮的 F1 变化，避免比较不同实验配置。

默认 `PAPER_TEST_FAST_MODE=1`：评测保留意图解析、全部检索、BGE-M3、Reranker、
确定性硬条件和指标计算，只跳过远程 LLM 证据升级。意图与网络 Provider 响应缓存在
`backend/paper-test/cache/`，TTL 默认 7 天；缓存目录已被 Git 忽略。报告中的 Provider
诊断会标记“缓存”或“按覆盖情况跳过”。关闭快速模式即可按正常搜索执行完整 EvidRank。
