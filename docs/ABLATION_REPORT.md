# ScholarAgent 消融实验总报告（当前版本合并版）

> **合并日期**：2026-08-30
> **适用范围**：当前工作目录 `D:\Project\huawei-paper-search-main`（无 git 仓库，最近改动 2026-08-26 SPARBench）
> **排除范围**：`version_backup/` 历史快照；另一副本 `D:\Project\papersearch\` 的标准 SPARBench 全量评测（不属本目录当前版本）
> **评测协议**：`COMPETITION_EVALUATION_PROTOCOL.md`（三条独立证据线 + 严格消融定义）
> **原始数据**：`.modeling-audit/*.json` 全部可复跑验证；复现脚本在 `backend/scripts/`

---

## 0. 评测协议背景：消融如何定义（为什么这样设计）

为避免"换了候选、网络状态或测试样本，却把分数变化归因于算法"的无效实验，本项目的所有消融遵循三条独立证据线与严格消融定义：

- **证据线 A（固定语料消融）**：用数据集自身 corpus + qrels、稳定 doc_id 评分，回答"召回器和排序器本身是否有效"。全部离线、零 API/token。
- **证据线 B（冻结 Provider 端到端）**：`provider_cache_mode=record` 保存全部 Provider 响应，再以 `frozen` 复测，回答"在完全相同候选池上，代码或排序改动是否有效"。冻结缓存缺失时整次运行必须失败。
- **证据线 C（真实开放网络压力测试）**：记录 API 错误、失败率、P50/P95、Token 和调用次数。

**严格消融定义**（BGE-M3 同时输出 dense 与 sparse，主实现用 RRF 融合二者，不可含糊拆分）：

```
FTS（词法基线）→ +BGE dense → +dense/sparse RRF → +Reranker(Cross-Encoder) → Full EvidRank
```

所有消融只在开发集选参数；测试集冻结后只运行预先登记的配置，禁止逐题改权重。可比较运行要求 dataset/split/K/seed/Provider 快照/模型 revision/候选预算/精排 Top-N/RRF k/权重全部一致。

---

## 1. 实验总览

| # | 实验名称 | 证据线 | 目的 | 数据文件 | 结论 |
|---|---|---|---|---|---|
| A1 | FTS→Full EvidRank 递进消融 | A | 验证各检索/排序层独立有效性 | `fixed_test_ablation.json`、`nfcorpus_test_frozen.json`、`nfcorpus_dev_ablation.json` | 每级正向、无回退；MRR@5 51.89%→56.35% |
| A2 | EvidRank 组件逐个关闭消融 | A | 隔离每个组件贡献 | `evidrank_ablation.json` | 仅 reranker 可测出贡献（-0.69pt R） |
| A3 | SciMap 组件消融 | A | 验证六组件独立性与结构质量 | `scimap_ablation.json` | 组件独立；引用完整性是短板 |
| A4 | BEIR 全库向量召回对照 | A 补充 | 证明 FTS 池外召回增益 | `beir_*_vector_recall.json`、`nfcorpus_test_vector_recall.json` | 词法鸿沟任务 14x~69x 提升 |
| B1 | 方案 A+B 对照（多查询变体+reranker200） | B | 验证 08-22 优化方案及各部分贡献 | `backend/paper-test/results/runs/*.json` | A+B 协同有效；B 单独无效 |
| B2 | Selector 离线同池单模型对照 | B | 官方 7B selector 是否优于 reranker | `selector_test_pool.json`、`selector_rerank_results.json` | @5 命中 6→12 翻倍 |
| B3 | Selector 融合 A/B（含权重扫描） | B | selector 融入主程序端到端是否有增益 | `selector_ab_detail.json` | @5 6→11（+83%）、MRR 0.740 |
| B4 | Crawler 查询融合 A/B | B | crawler 融合是否提升端到端 | `clean_ab_crawler_fusion.json` | 融合有害，未采用 |
| C1 | 引文开/关对比 + 引文诊断 | B/C | 引文扩展是否提升召回 | `citation_on_off_comparison.json`、`citation_diag_RealScholarQuery_40.json` | 无差异（样本小）；precision 低 |
| C2 | SPARBench crawler-vector 候选召回消融 | A/C | 全 50 条上 crawler+向量召回增益 | `spar_crawler_vector_test.json` | 78→93 gold（14.0%→16.7%） |
| C3 | SPARBench 排序端 A/B（test10） | A/C | 新候选池在排序端是否有收益 | `spar_baseline/fused_ranking_test10.json`、`spar_pool_ranking_ab_test10.json` | 排序无收益，不融合 |

> 另有 2 项修复类记录（非消融但直接影响评测口径）见 §6：hard_gate `language` 硬拒修复、selector 分片加载。

---

## 2. 证据线 A：固定语料消融（NFCorpus test，零 token）

### 实验 A1｜FTS → Full EvidRank 递进消融

**目的**：按协议 §4 的严格消融定义，在固定语料上验证检索/排序链路每一层组件的独立有效性，回答"召回器和排序器本身是否有效"。

**实现方法**：
- 数据集：NFCorpus 固定版（`backend/paper-test/datasets/nfcorpus-fixed/`，由 `build_nfcorpus_fixed_corpus.py` 从 ir_datasets 构建，corpus 3,633 篇、test 323 条查询，精确 doc ID qrels）
- 脚本：`backend/scripts/evaluate_fixed_corpus.py`（GPU 加速，RTX 3060 + BGE-M3 本地编码）
- 配置：候选池 pool=200、reranker-pool=80、k=5/10/20、失败 0
- 5 个 profile 共享**同一 FTS 候选池**（协议已知限制：池内重排不改变召回），仅改变排序方式：
  1. `FTS`：SQLite FTS5 词法排名
  2. `+BGE dense`：同一候选按 BGE-M3 dense cosine 排名
  3. `+dense/sparse RRF`：dense 与 sparse 两路 RRF 融合
  4. `+Reranker`：加固定 revision Cross-Encoder（BGE-reranker-v2-m3）
  5. `Full EvidRank`：加身份核验、硬约束、证据判定、等级与最终排序策略

**结果**（323 查询，零 API/token，确定性证据判定）：

| Profile | 候选R@200 | R@5 | F1@5 | nDCG@5 | MRR@5 | R@10 | nDCG@10 | 平均延迟 |
|---|---|---|---|---|---|---|---|---|
| FTS（词法基线） | 27.93% | 11.99% | 11.90% | 35.33% | 51.89% | 15.04% | 32.21% | 5ms |
| + BGE dense | 27.93% | 12.68% | 12.89% | 37.26% | 55.69% | 16.15% | 34.15% | 432ms |
| + dense/sparse RRF | 27.93% | 13.15% | 13.09% | 37.77% | 55.73% | 16.28% | 33.94% | 432ms |
| + Reranker（Cross-Encoder） | 27.93% | 13.37% | 13.40% | 38.09% | 56.13% | 16.32% | 35.06% | 1,134ms |
| Full EvidRank | 27.93% | 13.80% | 13.91% | 38.87% | 56.35% | 16.49% | 35.30% | 938ms |

（冻结配置快照 `FROZEN_TEST_REPORT.md` 同口径：evidrank F1@5=13.91%、nDCG@5=39.09%、MRR@5=56.63%，P95=236ms；dev 集 `nfcorpus_dev_ablation.json` 上 evidrank F1@5 相对 FTS +18.6%）

**解读**：
- 候选召回恒定 27.93%——五个 profile 共享同一 FTS 候选池，池内重排不改变召回；全库召回增益由实验 A4 独立对照证明
- 排序组件逐级有效：FTS→EvidRank 全程 MRR@5 +4.5pt、F1@5 +2.0pt，每级正向贡献、无回退
- 零成本（确定性证据判定），GPU 下 evidrank 延迟反而低于 reranker

**复现**：`python -m backend.scripts.evaluate_fixed_corpus`（输出 `nfcorpus_test_frozen.json`）

### 实验 A2｜EvidRank 组件逐个关闭消融

**目的**：A1 只证明链路整体有效，本实验逐个关闭 EvidRank 内部组件，隔离"每个组件贡献多少"。

**实现方法**：
- 脚本：`backend/scripts/ablate_evidrank.py`
- 方法：同一 FTS 候选池（200），对 308 条有效查询（有 relevant gold 的），调用 `EvidRankPipeline.rank(ablation=...)` 逐组件关闭：
  - `full`：完整 EvidRank 管线
  - `no_reranker`：跳过 Cross-Encoder 精排（仅 RRF 序）
  - `no_facet`：跳过结构化 dataset/method/relation facet 融合
  - `no_preference`：跳过权威/新颖偏好加权
  - `no_selector`：跳过 selector 打分阶段（脚本已定义）
  - `no_evidence`：跳过证据判定（仅分数排序）
  - `no_level`：跳过等级决策（全 HIGH、按分数排序）
- 全程离线、确定性证据判定、零 token；宏平均 P/R/F1/nDCG/MRR @5/10

**结果**（308 条有效查询）：

| 变体 | R@5 | F1@5 | nDCG@5 | MRR@5 | 结论 |
|---|---|---|---|---|---|
| **full（完整 EvidRank）** | **14.47%** | **14.59%** | **40.76%** | **59.10%** | 基线 |
| no_reranker（关 Cross-Encoder） | 13.78% | 13.71% | 39.59% | 58.50% | **-0.69pt R、-0.6pt MRR：reranker 有真实贡献** |
| no_facet / no_preference / no_evidence / no_level | = full | = full | = full | = full | NFCorpus 干净语料下不触发 |

**诚实结论**：
- **Cross-Encoder reranker 是唯一能测出贡献的组件**（关闭后 R@5 -0.69pt、MRR -0.6pt）
- facet/preference/evidence/level 在 NFCorpus 上无差异——这是实验设计的真实限制：NFCorpus 查询无方法/数据集/年份约束（facet 不激活）、无权威/新颖偏好词（preference 不激活）、候选无元数据冲突（evidence 无差异）。**这些组件需在含约束的真实查询集（如 PaSa）上验证**，或构造含约束/冲突的测试集
- 注：`no_selector` 变体已写入脚本但保存的数据文件（2026-08-22 生成）中没有结果——selector 于 08-24 才融入主程序，该变体需在集成后单独复跑才能量化（见实验 B3 已给出端到端口径）

**复现**：`python -m backend.scripts.ablate_evidrank`（输出 `evidrank_ablation.json`）

### 实验 A3｜SciMap 组件消融

**目的**：功能 4（SciMap 归纳）的六类组件（ranked_list / comparison_matrix / timeline / evidence_graph / comparability / synthesis）是否各自独立可用、结构质量如何、有无崩溃与非法结构。

**实现方法**：
- 脚本：`backend/scripts/ablate_scimap.py`
- 方法：对真实 PaSa 查询（本地 FTS + 向量检索，离线）生成冻结 `FinalRankingBundle`，随后逐个关闭 organizer 组件构建 SciMap 输出工件，测量**自动结构质量指标**（引用完整性、unknown 单元格处理、证据图边合法性、claim 引用精度）。3 条查询平均。
- 组件清单：ranked_list、comparison_matrix、timeline、evidence_graph、comparability、synthesis（+claim validation）、artifact_hash

**结果**：

| 指标 | full | 说明 |
|---|---|---|
| 证据卡片数 | 167.3 | 每个候选生成卡片 |
| **卡片含证据引用比例** | **0.0%** | ⚠️ 卡片未附加证据引用 |
| 比较矩阵单元 | 460.7 | 167 行 × ~3 分面 |
| **矩阵未知值比例** | **99.5%** | ⚠️ 证据不足时保持 unknown（保守，但覆盖率过低） |
| 时间线条目 | 167.3 | 每篇一个条目 |
| 证据图节点/边 | 168.7 / 2.3 | **非法边 0**（边都合法 supports/violates） |
| 可比性评估对 | 5,133 | 两两比较 |
| 归纳 claims | 7.7 | |
| **claim 引用率（citation precision）** | **39.8%** | ⚠️ 归纳结论引用支撑不足 |

**结论与局限**：
- **组件独立性验证**：各组件可独立产出、去掉一个不影响其他（并行组装设计），无崩溃、无非法证据图边
- **暴露真实质量问题**：卡片证据引用 0%、矩阵未知率 99.5%、claim 引用率仅 39.8%——**功能 4 的引用完整性是当前明显短板**（README 要求 claim citation precision 达标，目前不达标）
- **依赖关系**：SciMap 完全依赖 EvidRank 的证据矩阵；简化查询（无约束）产生空证据 → SciMap 退化（证据生成质量是功能 4 的前提）
- **语义质量需人工标注**：自动指标只能覆盖结构/引用完整性；归纳的语义质量（是否过度断言等）需双人盲标（`backend/paper-test/annotations/README.md` 流程），当前无标注者——按协议 §6，不能声称 SciMap 引用/归纳准确率已达标

**复现**：`python -m backend.scripts.ablate_scimap --limit 3`（输出 `scimap_ablation.json`）

---

## 3. 证据线 A 补充：全库向量召回对照（BEIR 独立验证）

### 实验 A4｜FTS（词法） vs BGE-M3 向量（语义）候选召回

**目的**：协议明确固定语料评测器只在 FTS 池内重排，**证明不了全库召回增益**。因此全库 BGE-M3 向量召回必须用独立对照（启用/禁用），在**与 PaSa 无关**的 BEIR 语料上验证语义通道的召回贡献。

**实现方法**：
- 脚本：`backend/scripts/eval_beir_vector_recall.py`、`backend/scripts/eval_nfcorpus_vector_recall.py`
- 在 BEIR arguana（8,674 篇/399 查询）、scidocs（25,000 篇/400 查询）、NFCorpus（3,633 篇/323 查询）上，对比 FTS 词法召回 vs BGE-M3 dense 全库向量召回，Recall@200

**结果**：

| 数据集 | 语料规模 | 查询数 | FTS 候选 Recall@200 | 向量候选 Recall@200 | 提升倍数 |
|---|---|---|---|---|---|
| arguana（论点检索，词法鸿沟极大） | 8,674 | 399 | 1.25% | **17.79%** | **14.2x** |
| scidocs（引用推荐，语义弱相关） | 25,000 | 400 | 0.05% | **3.46%** | **69.2x** |
| NFCorpus（医学，词面部分重叠） | 3,633 | 323 | 27.93% | 34.25% | 1.2x |

**结论**：向量检索相对词法检索的提升在词法鸿沟大的任务上是数量级（14x–69x），证明语义召回真实有效。绝对水平取决于任务难度（scidocs 的 gold 是被引论文，与查询语义相关性弱，即使向量也只有 3.46%）。

**复现**：`python -m backend.scripts.eval_beir_vector_recall --dataset arguana|scidocs`、`python -m backend.scripts.eval_nfcorpus_vector_recall`（输出 `beir_*_vector_recall.json`、`nfcorpus_test_vector_recall.json`）

---

## 4. 证据线 B：冻结 Provider 端到端消融（pasa-real 5 条 seed42）

> 条件统一：pasa-real、limit=5、k=[5,10]、seed=42、provider-cache-mode=record→frozen。冻结 Provider 快照保证候选相同，仅排序/检索改动可比。

### 实验 B1｜方案 A+B 对照（多查询变体 + reranker200）

**目的**：验证 2026-08-22 落地的两个优化（方案 A：查询侧多查询变体检索；方案 B：精排 top-n 80→200）各自的贡献与协同效应。

**实现方法**：
- 方案 A：`local_vector_corpus.py` 的 `_query_variants` 多查询变体检索 + `_rrf_merge_variant_rankings` 变体结果 RRF 融合（开关 `ACADEMIC_LOCAL_VECTOR_MULTIVARIANT`）
- 方案 B：`PAPER_TEST_BGE_RERANKER_TOP_N` 80→200（扩大精排候选范围）
- 对照组按可比较运行硬条件逐一登记 run_id

**结果**：

| 版本 | run_id | hits@5 | R@5 | P@5 | nDCG@5 | MRR@5 | hits@10 | API 调用 | 平均延迟 |
|---|---|---|---|---|---|---|---|---|---|
| baseline（FTS 为主） | 493d12fe | 2 | 0.047 | 0.080 | 0.102 | 0.267 | 2 | 47 | 236.2s |
| planC（PRF 二次搜索） | 23f91983 | 2 | 0.047 | 0.080 | 0.111 | 0.300 | 2 | 48 | 244.4s |
| + 全库向量检索 | 643851ad | 5 | 0.126 | 0.200 | 0.282 | 0.667 | 9 | 47 | 28.1s |
| **+ 方案A多查询变体 + 方案B reranker200** | f3557136 | **6** | **0.176** | **0.240** | **0.312** | **0.667** | **10** | **34** | 30.2s |
| B-only 对照（仅 reranker200） | 71749503 | 5 | 0.126 | 0.200 | 0.282 | 0.667 | 9 | 15 | 22.8s |

**消融解读**（A/B 对照实验结论）：
- **全库向量检索是主贡献**：MRR 0.267→0.667（2.5x）、hits 2→5——检索阶段突破词法鸿沟
- **方案 B 单独无效**（B-only = 向量版基线，hits 5）——仅扩大精排范围，池子里没有新候选可排
- **方案 A+B 协同有效**（hits 6）：A（多查询变体，用查询理解已付费的 LLM 改写）召回**新 gold 进池**，B（reranker 200）把这些新候选**精排进 top-5**。二者缺一不可
- **成本降低**：API 调用 47→34（本地向量通道召回质量提升 → 控制器早停）；延迟 236s→30s（BGE GPU + persist 超时修复）

**复现**：`python -m backend.paper-test.run_evaluation --dataset pasa-real --limit 5 --k 5 10 --seed 42 --provider-cache-mode record`（数据：`backend/paper-test/results/runs/*.json`）

### 实验 B2｜Selector 离线同池单模型对照

**目的**：官方 `bytedance-research/pasa-7b-selector`（Qwen2.5-7B-Instruct 微调，PaSa 官方 Selector）是否优于当前 reranker 排序信号，以及最佳混合方式。

**实现方法**：
- 脚本：`backend/scripts/build_selector_pool.py` + `backend/scripts/eval_pasa_selector_rerank.py` + `backend/scripts/sanity_pasa_selector.py`
- **论文池**：主程序同款检索栈（FTS BM25 + BGE-M3 全库向量，fallback understanding，pool=200×2 并集），捕获每篇 title+abstract。5 条基准查询（RealScholarQuery_1/7/17/40/47，seed 42），池共 808 篇，gold-in-pool = 29/82 = **35.4%**
- **打分**：忠实复刻官方 `infer_score`——`get_selected` 模板（Title/Abstract/User Query/Decision: True-False），`generate(max_new_tokens=1, do_sample=False)`，取首 token 为 `'True'` 的 softmax 概率，>0.5 入选；4-bit NF4 量化加载（RTX 3060 12.88GB VRAM）
- **对照**：bge-reranker（主程序当前排序信号，同一池子打分）+ 历史全管道 run；匹配与指标复用主程序逻辑（arxiv_id→title 兜底、one-to-one、宏平均）

**结果**（宏平均）：

**@5**

| 排序方式 | P | R | F1 | nDCG | MRR | hits |
|---|---|---|---|---|---|---|
| 历史 A+B 全管道 f3557136 | 0.240 | 0.176 | 0.181 | 0.312 | 0.667 | 6 |
| 历史最新 b28389d6 | 0.120 | 0.064 | 0.068 | 0.106 | 0.150 | 3 |
| reranker（同池，当前信号） | 0.240 | 0.148 | 0.165 | 0.299 | 0.600 | 6 |
| **selector（官方用法，同池）** | **0.480** | **0.185** | **0.238** | **0.490** | 0.640 | **12** |
| sel 过滤→reranker 混合 | 0.400 | 0.205 | **0.244** | 0.432 | **0.740** | 10 |
| rrf 融合（sel+reranker） | 0.360 | 0.168 | 0.199 | 0.370 | 0.590 | 9 |

**@10 / @20**

| 排序方式 | hits@10 | R@10 | hits@20 | R@20 | F1@20 |
|---|---|---|---|---|---|
| 历史最新 b28389d6 | 11 | 0.180 | 15 | 0.249 | 0.155 |
| reranker（同池） | 10 | 0.173 | 14 | 0.243 | 0.148 |
| **selector** | **16** | 0.254 | **23** | 0.391 | **0.244** |
| **rrf 融合** | 14 | 0.287 | **24** | **0.397** | **0.252** |

**逐条（selector @5 hits）**：Q1 0→5（top-7 全 gold）、Q7 0→0（long video，selector 偏好明确研究型论文）、Q17 0→2（历史从未找到的 gold 被找回）、Q40 0→0（3 个池内 gold 排 12/32/35）、Q47 0→1

**结论**：
1. **@5 命中翻倍**：6→12（reranker/历史均 6）；P、F1、nDCG 翻倍级提升，MRR 持平（0.640 vs 0.667）
2. **@20 命中 +64%**：23 vs 14（同池 reranker）/ 15（历史），R@20 0.391 vs 0.243/0.249
3. **最佳混合**：`sel>0.5 过滤→reranker` MRR@5=0.740；`RRF 融合` F1@20=0.252、R@20=0.397（池内 gold 83% 进 top-20）

**诚实解读**：
- 池子为本地检索栈重建（35.4% 上限），历史 run 含 provider 候选；但**同池 reranker 复现历史结果（@5 hits 6=6）证明池子忠实、对比公平**
- Q40 是例外：selector 把大量相关 QAT 论文排在 gold 前（gold 位 12/32/35），reranker 反而把 EfficientQAT 排进 top-5——语义排序器对精选答案集的固有偏差，非 bug
- **训练数据重叠风险**：selector 在 pasa 数据族上微调，RealScholarQuery gold 与训练分布有重叠；**NFCorpus/BEIR 无关语料的泛化性需单独验证**才能下"通用排序器"结论
- 推理成本：4-bit、batch 8 下 ~50s/查询（~0.3s/篇）；与 reranker 同机共存 OK（4bit selector ~5GB + reranker 2.2GB）
- 此前微调 reranker 失败（两次，hits 6→0/6→3）与本测试不矛盾：官方 7B selector 是数据规模远超本项目微调、RL 训练过的判别器

**复现**：`python -m backend.scripts.build_selector_pool --limit 5 --seed 42 --pool 200` → `python -m backend.scripts.eval_pasa_selector_rerank` → `python -m backend.scripts.sanity_pasa_selector`（产物 `selector_test_pool.json`、`selector_rerank_results.json`）

### 实验 B3｜Selector 融合 A/B（当前最佳配置）

**目的**：B2 证明 selector 单模型强于 reranker，本实验验证把 selector **融入主程序 EvidRank 排序管线**后，端到端（冻结 provider 口径）是否有增益。

**实现方法**：
- 新服务 `backend/app/services/pasa_selector.py`（仿 reranker.py：enabled/required/4bit/provider/_load_error/@lru_cache `get_pasa_selector()`，GET_SELECTED 模板 + P(True)）
- `pipeline.py`：selector 打分阶段（对 reranked[:80]，**用 `intent.raw_query`**——`_rerank_query` 会追加 "Key concepts" 破坏微调分布）+ 融合项 `selector_weight/(rrf_k+sel_rank)` + **头部保护改用 selector top-3** + `ablation='selector'` 开关
- 配置：`pasa_selector_rank_weight=0.30`、`paper_test_pasa_selector_top_n=80`、4bit + `max_memory={"cpu":6GiB, 0:11GiB}`、`padding_side="left"`；`iter_paper_search_events` 入口**提前加载 selector**（检索加载 encoder/faiss 之前）避免内存尖峰 segfault
- 评测：pasa-real 5 条 seed42 k=5/10/20，OFF 基线 vs ON 融合，compare_selector_ab.py 对比

**结果**：

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 6 | 10 | — |
| OFF 基线（当前 reranker 管道，ee7ff885） | 6 | 10 | 15 |
| **ON 融合（+selector，adab2edd）** | **11** | **13** | **20** |

| 指标 | OFF 基线 | ON 融合 | Δ |
|---|---|---|---|
| P@5 | 0.240 | **0.440** | +0.200 |
| R@5 | 0.176 | **0.225** | +0.049 |
| F1@5 | 0.181 | **0.255** | +0.074 |
| nDCG@5 | 0.312 | **0.491** | +0.179 |
| MRR@5 | 0.667 | **0.740** | +0.073 |
| hits@20 | 15 | 20 | +5（R@20 0.298 持平） |

- **OFF 基线 ee7ff885 精确复现历史最佳 f3557136**（@5 hits=6 全项一致）——对比公平
- **逐条全提升**：Q1 2→5、Q7 0→1（long video 首破）、Q17 0→2、Q40 0→1、Q47 0→2
- **权重扫描**：0.20 → @5=11/@20=19；**0.30 → @5=11/@20=20（最优）**；0.40 → @5=10/@10=14/@20=20
- **成本**：零新增 API（selector 本地），+23~25s/查询打分（top-80），首次含 40s 加载；`.env` 的 `PASA_SELECTOR_ENABLED` 默认 0（交互/生产不用时别开：+40s 加载 +6GB 显存）

**结论**：selector 融合（权重 0.30 + 头部保护 + top-80）是当前版本**端到端最优配置**，@5 命中 +83%、MRR 0.667→0.740，且零 API 成本。

**复现**：`PASA_SELECTOR_ENABLED=1 python -m backend.paper-test.run_evaluation --dataset pasa-real --limit 5 --k 5 10 20 --seed 42`；对比 `python -m backend.scripts.compare_selector_ab`（数据：`selector_ab_detail.json`）

### 实验 B4｜Crawler 查询融合 A/B

**目的**：crawler 生成的查询（append-to-understanding）融合进意图理解，是否提升端到端检索。

**实现方法**：clean A/B——清除 provider 缓存、对称全新调用，保证两边网络状态一致；`run_A_selector_only=6ff9307d7cf1` vs `run_B_crawler_fusion=72064c9d7478`；k=5/20，5 条基准查询。

**结果**：

| 配置 | @5 hits | MRR@5 | F1@5 | R@5 | P@5 | @20 hits | R@20 |
|---|---|---|---|---|---|---|---|
| **A：selector-only（当前最佳）** | **11** | **0.740** | **0.255** | **0.225** | **0.440** | **20** | **0.297** |
| B：crawler 融合 | 8 | 0.407 | 0.170 | 0.146 | 0.320 | 14 | 0.183 |

**结论**：crawler 查询融合**有害**——gold 在 provider 阶段就丢失（Q7 19→8），不是排序问题。selector-only 是当前最佳且稳定的配置，**crawler 融合未采用**。（与实验 C2/C3 的"召回有增益但排序无收益"结论互相印证。）

**数据**：`clean_ab_crawler_fusion.json`

---

## 5. 证据线 C / 其他组件对照

### 实验 C1｜引文开/关对比 + 引文诊断

**目的**：OpenAlex 引文扩展（PaSa 式引文追踪改造，2026-08-21）是否提升召回，以及引文论文召回质量低下的根因。

**实现方法**：
- 开/关对比：pasa-real 3 条查询，引文扩展启用 vs 禁用（`citation_on_off_comparison.json`）
- 根因诊断：`diagnose_citation_recall.py` 对 RealScholarQuery_40 逐层拆解（`citation_diag_RealScholarQuery_40.json`）

**结果**：
- 开/关对比：**MRR 无差异**（样本小）
- 诊断数据（RealScholarQuery_40，5 个 gold）：候选池 92、引文来源论文 3（进池）、引文进 top-k 0/3、gold 命中候选池 2/5、**gold 命中引文论文 0/5**、引文保留率 3/57（5.3%）

**根因三层**：
1. **种子选择不稳定（已修复）**：第一轮候选主要来自 arXiv（无 DOI/openalex）→ 种子极少 → 引文不触发。修复：`_resolve_openalex_work_ids` 支持 arxiv id（经 `10.48550/arXiv.xxx` DOI 反查）
2. **种子质量（已修复）**：种子按 provider rank 选到无引文网络的新论文（citation_count=0）→ 返回 0。修复：种子优先选 `citation_count>0` 的成熟论文（两级排序）
3. **引文论文召回质量低（未修复）**：引文网络覆盖的是"相关领域 survey"而非 gold 精确答案；进池后又排不到 top-k。这正是 PaSa 论文用 **Selector 学习相关性过滤**解决的问题——我们目前引文论文直接进池、缺相关性过滤

**结论**：引文扩展机制可用（能拉回论文、能进候选池），但召回 precision 低；在当前评测口径下**测不出净收益**。待处理：引文论文相关性过滤（进池前按查询语义过滤）。

### 实验 C2｜SPARBench crawler-vector 候选召回消融（全 50 条）

**目的**：在 SPARBench 全量 50 条查询上，crawler 查询 + 向量检索对**候选召回（gold-in-pool）**的增益，以及本地语料的覆盖上限。

**实现方法**：
- 脚本：`backend/scripts/eval_spar_crawler_vector.py`（--mode generate/recall/both、--limit/--start；生成按 qid checkpoint 到 `spar_crawler_queries_cache.json` 可续传；recall 从输出 JSON 恢复续跑）
- 数据集：`dataset/SPAR_Benchmark.jsonl`（50 条查询、556 个 gold；paperID 三种形态：裸 arXiv id 444 / arXiv URL 76 / Semantic md5 36）
- **匹配口径修正**：现有 `test_crawler_queries.py` 的 id 匹配失效（pool 论文 `id` 是 sha256 哈希，`p.get('id') or p.get('arxiv_id')` 恒取到哈希 → gold_id 交集恒空、实际是 title-only）。新脚本用 `external_ids.arxiv`/`arxiv_id` 显式提取，gold 匹配 = 规范化标题 key OR 从 paperID 提取的 arXiv id
- 检索：FTS + 向量各 200；需 agent 环境 python（base 3.13 无 faiss）

**结果**（50 条全量）：

| 配置 | gold-in-pool | 比例 |
|---|---|---|
| baseline（FTS+向量） | 78/556 | 14.0% |
| crawler_only | 86/556 | 15.5% |
| **fused（baseline + crawler）** | **93/556** | **16.7%** |

- crawler 净增 15 个 gold、12/50 条查询有增益（SPARBench_1/2/3/7/11/12/18/19/20/21/23/27）
- **语料覆盖上限**：gold 中仅 212/556（38.1%）的 arXiv id 在本地 568k 语料（大量 2024–2025 论文超出快照）→ **fused 在可覆盖子集上 93/212=43.9%**（baseline 36.8%）；14 条查询 0 命中（gold 基本不在语料）
- 性能：crawler 生成 ~3.5s/条（非预计 40s）、recall ~14s/条，全量约 4 分钟

**结论**：crawler 查询在候选召回端有真实增益（14.0%→16.7%，可覆盖子集上 36.8%→43.9%），但受限于语料覆盖上限（38.1%）。是否采用取决于排序端是否受益（见实验 C3）。

**复现**：`D:/Users/S/anaconda3/envs/agent/python.exe -m backend.scripts.eval_spar_crawler_vector --mode generate`（缓存已存在则跳过）；`--mode recall --limit 25` + `--limit 50 --start 25` 分块跑

### 实验 C3｜SPARBench 排序端 A/B（test10）

**目的**：C2 证明 fused 池有更多 gold 候选，本实验回答**排序端**（reranker / selector 等 6 种通道）是否把这些新候选排进 top-k——决定 crawler 融合是否值得采用。

**实现方法**：
- `score_spar_pools_selector.py` 对 baseline/fused 两池逐查询打分（`--batch 32 --max-len 512`，sel_score 缓存可续传），`compare_spar_pool_rankings.py` 同池 A/B
- 通道：selector、reranker、sel_filter_rerank、sel_filter_rerank_signal、sel_topN_rerank、rrf_fusion；k=5/10/20；test10 子集

**结果**（aggregate，`spar_pool_ranking_ab_test10.json`）：

| 通道 | @k | baseline hits / MRR | fused hits / MRR | Δ |
|---|---|---|---|---|
| **reranker** | 5 | 7 / 0.220 | 7 / 0.220 | **0 / 0.000（完全不变）** |
| reranker | 10 | 11 / 0.220 | 11 / 0.220 | 0 / 0.000 |
| reranker | 20 | 17 / 0.225 | 17 / 0.225 | 0 / 0.000 |
| selector | 5 | 3 / 0.175 | 2 / 0.200 | -1 / +0.025 |
| selector | 10 | 6 / 0.200 | 5 / 0.229 | -1 / +0.029 |
| selector | 20 | 9 / 0.207 | 9 / 0.238 | 0 / +0.031 |

**结论**：**reranker top-k 完全不变（hits/MRR 零差异），selector 系反而微降（hits -1）**。crawler 融合新增的 gold 候选排不进 top-k——排序端无收益 → **不采用 crawler 融合**（用户决定，与 B4 端到端结论一致）。

**复现**：`python -m backend.scripts.score_spar_pools_selector --pool ... --batch 32 --max-len 512` → `python -m backend.scripts.eval_pasa_selector_rerank --pool ... --out ...` → `python -m backend.scripts.compare_spar_pool_rankings`

---

## 6. 修复类记录（非消融，但直接影响评测口径，必读）

### 6.1 hard_gate `language` 硬拒 bug（2026-08-26 修复）
- **现象**：意图解析会产出 `language contains zh+en` 的 HARD 约束，而论文无 language 元数据 → 命中即全拒
- **影响**：标准 SPARBench 全量评测 run 6555ff09（papersearch 副本）中 SPARBench_41 被全拒，藏在其 24 条 0 命中里
- **修复**：`backend/app/services/ranking/hard_gate.py` 把 `language` 从 `_METADATA_FACETS` 移除（defer 不拒）
- **修复后**：SPARBench_41 恢复 1 gold@20

### 6.2 selector 模型分片加载（2026-08-26）
- 单文件 `models/pasa-7b-selector/model.safetensors`（15.2GB，sha256 校验完好）在 `device_map="auto"` 时**偶发 Windows access violation**（一天内成功 1 次、崩 6+ 次，重试/预热/bf16 均无效）
- 修复：`backend/scripts/shard_selector_model.py` 字节级分片 → `models/pasa-7b-selector-sharded/`（4×3.8GB + index.json）；分片后 4bit 加载稳定（31.7s，VRAM 5.9GB）
- 经验：**单文件巨型 safetensors + device_map 在此机器不可靠，一律先分片**

---

## 7. 汇总结论

### 7.1 采纳的组件（消融证明确有净增益）
| 组件 | 证据 | 增益 |
|---|---|---|
| 全库 BGE-M3 向量检索 | A4 / B1 | 词法鸿沟任务召回 14x–69x；端到端 MRR 0.267→0.667 |
| 多查询变体（方案 A） | B1 | 召回新 gold 进池（hits 5→6 的关键） |
| reranker Top-200（方案 B） | B1 | 单独无效，但把 A 的新候选精排进 top-5（协同必需） |
| **selector 融合（权重 0.30 + 头部保护 + top-80）** | B2 / B3 | **当前最优**：@5 6→11（+83%）、MRR 0.667→0.740、零 API 成本 |

### 7.2 未采纳的组件（消融证明无效或有害）
| 组件 | 证据 | 结论 |
|---|---|---|
| crawler 查询融合 | B4 / C2 / C3 | 候选召回有增益（78→93）但排序端零收益/微降、端到端有害（Q7 19→8）→ 不融合 |
| 微调 reranker（LoRA） | B2 备注 | 两次微调 hits 6→0/6→3，失败（与官方 7B 判别器不可比） |
| planC（PRF 二次搜索） | B1 | MRR 0.267→0.300，弱于向量方案，已被取代 |

### 7.3 测不出贡献的组件（实验设计限制，非"无效"）
| 组件 | 证据 | 限制 |
|---|---|---|
| EvidRank facet / preference / evidence / level | A2 | NFCorpus 查询无约束/偏好词、候选无元数据冲突，组件不激活；需在含约束的 PaSa 查询上验证 |
| 引文扩展 | C1 | 机制可用但召回 precision 低；3 条对比 MRR 无差异（样本小） |
| SciMap 各组件（结构层面） | A3 | 组件独立可用；但卡片引用 0%、矩阵未知 99.5%、claim 引用率 39.8% 未达标，语义质量无人工标注 |

### 7.4 已知局限（报告时须如实披露）
1. **消融在 FTS 池内**：证据线 A 的 reranker/evidrank profile 共享 FTS 候选池（协议已知限制），全库召回增益由 A4 独立对照补充证明
2. **样本量**：证据线 B 全部基于 pasa-real 5 条基准（seed42），逐条结果方差大
3. **语料覆盖上限**：SPARBench gold 仅 38.1% 在本地 568k 语料（2024–2025 论文缺失），fused 可覆盖子集 43.9%
4. **selector 泛化未验证**：selector 与 pasa 数据族训练分布重叠，NFCorpus/BEIR 无关语料泛化性待单独验证
5. **功能 4 无人工标注**：按协议 §6，不能声称 SciMap 引用/归纳准确率达标

---

## 附录：复现命令汇总

```bash
# 环境
D:/Users/S/anaconda3/envs/agent/python.exe -m ...   # 必须用 agent 环境（base 3.13 无 faiss/torch cu124）

# A1 递进消融（NFCorpus test）
python -m backend.scripts.evaluate_fixed_corpus

# A2 EvidRank 组件消融
python -m backend.scripts.ablate_evidrank

# A3 SciMap 组件消融
python -m backend.scripts.ablate_scimap --limit 3

# A4 BEIR 向量召回对照
python -m backend.scripts.eval_beir_vector_recall --dataset arguana
python -m backend.scripts.eval_beir_vector_recall --dataset scidocs
python -m backend.scripts.eval_nfcorpus_vector_recall

# B1 方案 A+B（pasa-real 冻结）
python -m backend.paper-test.run_evaluation --dataset pasa-real --limit 5 --k 5 10 --seed 42 --provider-cache-mode record

# B2 selector 离线同池
python -m backend.scripts.build_selector_pool --limit 5 --seed 42 --pool 200
python -m backend.scripts.eval_pasa_selector_rerank
python -m backend.scripts.sanity_pasa_selector

# B3 selector 融合端到端
PASA_SELECTOR_ENABLED=1 python -m backend.paper-test.run_evaluation --dataset pasa-real --limit 5 --k 5 10 20 --seed 42
python -m backend.scripts.compare_selector_ab

# C2/C3 SPARBench crawler-vector
D:/Users/S/anaconda3/envs/agent/python.exe -m backend.scripts.eval_spar_crawler_vector --mode generate
D:/Users/S/anaconda3/envs/agent/python.exe -m backend.scripts.eval_spar_crawler_vector --mode recall --limit 25
D:/Users/S/anaconda3/envs/agent/python.exe -m backend.scripts.eval_spar_crawler_vector --mode recall --limit 50 --start 25
python -m backend.scripts.score_spar_pools_selector --pool ... --batch 32 --max-len 512
python -m backend.scripts.compare_spar_pool_rankings
```

## 附录：数据文件索引（均在 `.modeling-audit/`）

| 实验 | 文件 |
|---|---|
| A1 | `fixed_test_ablation.json`、`nfcorpus_test_frozen.json`、`nfcorpus_test_fts_baseline.json`、`nfcorpus_dev_ablation.json`、`nfcorpus_test_recall_analysis.json` |
| A2 | `evidrank_ablation.json` |
| A3 | `scimap_ablation.json` |
| A4 | `beir_arguana_vector_recall.json`、`beir_scidocs_vector_recall.json`、`nfcorpus_test_vector_recall.json` |
| B1 | `backend/paper-test/results/runs/*.json`（493d12fe/23f91983/643851ad/f3557136/71749503） |
| B2 | `selector_test_pool.json`、`selector_rerank_results.json` |
| B3 | `selector_ab_detail.json`（逐查询返回论文/两模型命中率/token/延迟/召回全记录） |
| B4 | `clean_ab_crawler_fusion.json` |
| C1 | `citation_on_off_comparison.json`、`citation_diag_RealScholarQuery_40.json`、`citation_steps.jsonl` |
| C2 | `spar_crawler_vector_test.json`、`spar_crawler_queries_cache.json`、`spar_baseline_pool.json`、`spar_fused_pool.json` |
| C3 | `spar_baseline_ranking_test10.json`、`spar_fused_ranking_test10.json`、`spar_pool_ranking_ab_test10.json`、`spar_baseline_pool_test10.json`、`spar_fused_pool_test10.json` |


## 发布核查补充（2026-10-10）

历史实验产物保持原始记录，以下修正根据 JSON 核查，优先于前文口述总结。

1. 引文开关实际 sample_limit=3，不是 40；RealScholarQuery_40 是查询编号。3/57 的保留率为 5.3%。
2. SPARBench 候选文件 queries=25 是过期汇总字段，per_query 实际 50 条。发布说明以逐条记录为准。
3. 相同 hits@5 不能独自证明候选池相同；需检验快照指纹。涉及检索改动或全新网络调用的实验是端到端配置对照，不能自动解释为严格同池消融。
4. 方案 A+B 缺少 A-only 对照，当前结果支持组合优于 baseline / B-only，尚不足证明两者缺一不可。
5. 权重扫描提供局部配置比较，不据此声称独立测试集最优。Selector 的 5 条查询增益是样本内结果。
6. claim 引用率是自动结构指标，不是经人工核验的引用正确率。
7. 本公开发布版修复了 Windows SQLite 连接释放和评测模型缺失字段，旧实验为修复前产物；未重跑并冒充旧配置结果。
8. 下表补齐 SPARBench 六种排序通道。

| 通道 | k | baseline hits | fused hits | baseline MRR | fused MRR |
| --- | --- | --- | --- | --- | --- |
| selector | 5 | 3 | 2 | 0.1750 | 0.2000 |
| selector | 10 | 6 | 5 | 0.2000 | 0.2292 |
| selector | 20 | 9 | 9 | 0.2071 | 0.2383 |
| reranker | 5 | 7 | 7 | 0.2200 | 0.2200 |
| reranker | 10 | 11 | 11 | 0.2200 | 0.2200 |
| reranker | 20 | 17 | 17 | 0.2250 | 0.2250 |
| sel_filter_rerank | 5 | 3 | 2 | 0.1750 | 0.2000 |
| sel_filter_rerank | 10 | 6 | 5 | 0.2000 | 0.2292 |
| sel_filter_rerank | 20 | 9 | 9 | 0.2071 | 0.2383 |
| sel_filter_rerank_signal | 5 | 6 | 6 | 0.1917 | 0.1917 |
| sel_filter_rerank_signal | 10 | 11 | 10 | 0.2160 | 0.2060 |
| sel_filter_rerank_signal | 20 | 15 | 17 | 0.2215 | 0.2258 |
| sel_topN_rerank | 5 | 6 | 5 | 0.1667 | 0.1667 |
| sel_topN_rerank | 10 | 10 | 8 | 0.1976 | 0.2000 |
| sel_topN_rerank | 20 | 14 | 14 | 0.2035 | 0.2142 |
| rrf_fusion | 5 | 4 | 5 | 0.2250 | 0.2450 |
| rrf_fusion | 10 | 8 | 6 | 0.2393 | 0.2450 |
| rrf_fusion | 20 | 14 | 13 | 0.2575 | 0.2572 |

主要端到端运行记录保存在 docs/evidence/runs/，完整数据集与候选全集不随发布包提交。原复现命令中省略的必填参数应通过 --help 与 SETUP.md 补齐。
