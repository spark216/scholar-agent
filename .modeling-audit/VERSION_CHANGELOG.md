# ScholarAgent 版本改动说明

- 第一版：用户最初解压、首次成功运行的状态（2026-08-20）
- 当前版：2026-08-21 引文追踪改造 + 评测机制完成后的状态
- 说明：项目无 git 仓库，本次改动记录基于代码 diff 与功能验证

## 一、环境与依赖（第一版 → 当前版）

| 项 | 第一版 | 当前版 |
|---|---|---|
| Python 环境 | conda `agent`（3.11.15） | 不变 |
| torch | 2.13.0+cpu（清华源，CPU 版） | **2.6.0+cu124（CUDA，RTX 3060）** |
| BGE 设备 | CPU | **GPU（BGE_M3_DEVICE=auto, FP16=1）** |
| BGE 模型 | 未下载（HuggingFace 被墙） | **models/ 本地（hf-mirror 下载，6.5GB）** |
| 引文源 | 无（S2 无 key 429） | **OpenAlex（匿名+mailto）** |

## 二、代码改动清单（第一版 → 当前版）

### 1. models 目录恢复
- 缺失的 `backend/app/models/`（evidrank/scimap/schemas/paper_test）从备份恢复
- 补 `PaperTestQueryResult.error_category` 字段（新版 paper_test.py 依赖）

### 2. 引文追踪改造（PaSa 式）⭐ 核心
- `academic_search.py`：
  - 新增 `search_openalex_neighborhood_multi`（OpenAlex 引文扩展，filter=cites:/referenced_works:）
  - 新增 `_resolve_openalex_work_ids`（DOI + arXiv DOI `10.48550/arXiv.xxx` → OpenAlex work ID）
  - 新增 `_fetch_openalex_neighbors`（单方向引文拉取）
  - 原 `search_semantic_scholar_neighborhood`（S2 版）保留
- `nodes.py`：
  - `_select_citation_seed`（单种子）→ `_select_citation_seeds`（多种子）
  - 种子资格：semantic_scholar id → **openalex/doi/arxiv id**
  - 种子排序：**citation_count>0 的成熟论文优先**（避免孤立新论文）
  - 触发条件：`coverage_or_recall_gap 才做` → **每轮都做**（every_round）
  - 引文调用：S2 → **OpenAlex multi**
  - 新增 `_log_citation_step`：每步引文写入 `.modeling-audit/citation_steps.jsonl`
- `config.py` / `.env`：
  - 新增 `SEMANTIC_SCHOLAR_CITATION_EXPANSION_EVERY_ROUND/SEED_TOP_N/DIRECTIONS`
  - DIRECTIONS 从 `[citations,references]` → `[cites,referenced_works]`（OpenAlex 命名）
  - `BGE_M3_DEVICE=cpu` → `auto`，`USE_FP16=0` → `1`

### 3. 评测机制
- 新增 `backend/scripts/run_and_log_eval.py`：评测 + 完整日志（eval_log.jsonl）
- 新增 `backend/scripts/diagnose_citation_recall.py`：引文召回诊断
- 新增 `backend/scripts/build_nfcorpus_fixed_corpus.py`：NFCorpus 固定语料构建
- 新增 `backend/scripts/build_beir_eval_files.py`：BEIR 数据集转换
- 新增 `backend/scripts/analyze_nfcorpus_recall.py`：召回失败分析
- 评测日志：`.modeling-audit/eval_log.jsonl`、`citation_steps.jsonl`

### 4. 评测数据集（新增）
- PaSa（RealScholarQuery/AutoScholarQuery，补 qid）→ `backend/paper-test/datasets/`
- BEIR 5 数据集 → `backend/paper-test/datasets/eval/`
- NFCorpus 固定语料（corpus/queries/qrels）→ `backend/paper-test/datasets/nfcorpus-fixed/`

### 5. Bug 修复
- persist 超时：`postgres_store.py`/`output_artifact_store.py` 加 `connect_timeout=3s`
- `config.py` 新增 `postgres_connect_timeout`
- nodes.py `seed`→`seeds` NameError（引文改造引入）

## 三、评测结论快照（当前版基线）

| 评测 | 结果 |
|---|---|
| NFCorpus test FTS 召回@200 | 27.93%（词法鸿沟 89.6%） |
| NFCorpus test evidrank F1@5 | 13.91%（vs fts 11.90%） |
| NFCorpus test evidrank nDCG@5 | 39.09% |
| NFCorpus test evidrank MRR@5 | 56.63% |
| pasa-real 引文开/关对比 | MRR 无差异（样本小） |

## 四、已保存的版本快照

- 当前版代码改动文件备份：见 `version_backup/current_2026-08-21/`
- 评测数据：`.modeling-audit/` 各报告

## 五、2026-08-22 版本快照（方案1 落地后，优化前）

**已备份**：`version_backup/current_2026-08-22/`（backend/app + backend/scripts + .modeling-audit + .env + config.py + 全库向量索引 3.1GB）

### 本快照包含的方案1改动
- `local_vector_corpus.py`：faiss 后端（dense+sparse+RRF），原子写 + 自愈加载 + array 紧凑结构，接口兼容
- `build_pasa_vector_index.py`：--batch-size 1024 / --checkpoint-interval 20000 / build_state.json 断点续传
- `config.py`：academic_vector_backend（faiss/milvus）、academic_local_vector_index_dir
- 新增脚本：eval_nfcorpus_vector_recall / eval_pasa_real_candidate_recall / eval_beir_vector_recall / leakage_analysis / compare_eval_runs
- 全库向量索引：568,059 篇 @ backend/paper-test/index/vector/pasa_full_corpus_bge_m3/
- .env：ACADEMIC_LOCAL_VECTOR_ENABLED=1、BGE_M3_BATCH_SIZE=128

### 关键评测快照
| 评测 | 结果 |
|---|---|
| pasa-real 5条 MRR@5（baseline→planC→向量版） | 0.267 → 0.300 → **0.667**（hits 2→5） |
| pasa-real 候选召回（FTS/VEC/融合@200） | 25.0% / 29.5% / 39.3% |
| BEIR arguana 召回@200（FTS→dense） | 1.25% → 17.79%（14x） |
| BEIR scidocs 召回@200（FTS→dense） | 0.05% → 3.46%（69x） |
| BEIR nfcorpus 召回@200（FTS→dense） | 27.93% → 34.25% |
| 泄露分析 | 无代码泄露；PaSa 答案摘要覆盖查询 72%（数据集虚高已确认） |

### 下一步待办（优化方向）
1. 查询侧：LLM 语义查询改写/多查询变体
2. 通道侧：启用 S2 神经相关性搜索（解决 429）
3. 召回上限：加大候选池 + RRF 调参 + 引文扩展配 Selector 过滤

### 六、2026-08-22 方案A+B 对照实验（零 token 优化）
- 方案A：`local_vector_corpus.py` 多查询变体检索（`_query_variants` + `_rrf_merge_variant_rankings`），开关 `ACADEMIC_LOCAL_VECTOR_MULTIVARIANT`
- 方案B：`PAPER_TEST_BGE_RERANKER_TOP_N=200`（评测精排 80→200）
- 对照（pasa-real 5条 seed42）：
  | 配置 | hits@5 | R@5 | nDCG@5 | API |
  |---|---|---|---|---|
  | 向量版基线（reranker80+单查询，643851ad） | 5 | 0.126 | 0.282 | 47 |
  | B-only（reranker200+单查询，71749503） | 5 | 0.126 | 0.282 | - |
  | **A+B（多查询变体+reranker200，f3557136）** | **6** | **0.176** | **0.312** | **34** |
- 结论：B 单独无效（=基线），A 是召回来源、B 是排序配合（协同）。保留 A+B。
- A 效果依赖真实 LLM 改写（弱回退改写无增益，候选召回 29.3%）。
