# ScholarAgent 评测总结（2026-08-20）

## 一、固定语料评测（NFCorpus，精确 doc ID qrels）

### 数据集
- 构建自 ir_datasets `beir/nfcorpus`：corpus 3633 篇，dev 324 条 / test 323 条查询
- 位置：`backend/paper-test/datasets/nfcorpus-fixed/`
- 构建脚本：`backend/scripts/build_nfcorpus_fixed_corpus.py`

### 功能2 召回指标（test，pool=200）
| 指标 | 值 |
|---|---|
| FTS Candidate Recall@200 | **27.93%** |
| 失败分析 | 词法鸿沟 89.6% / 低重叠 5.2% / 截断 5.2% |
| 分析脚本 | `backend/scripts/analyze_nfcorpus_recall.py` |

### 功能3 排序消融（test，冻结配置）
| profile | F1@5 | nDCG@5 | MRR@5 | P95(ms) |
|---|---|---|---|---|
| fts | 11.90% | 35.33% | 51.89% | 13 |
| bge_dense | 12.89% | 37.29% | 55.69% | 1254 |
| bge_rrf | 13.07% | 37.72% | 55.73% | 1254 |
| reranker | 13.40% | 38.09% | 56.13% | 2164 |
| **evidrank** | **13.91%** | **39.09%** | **56.63%** | **236** |

- dev 集消融见 `.modeling-audit/nfcorpus_dev_ablation.json`（evidrank F1@5 +18.6% vs fts）
- 测试集 FTS 基线与 README 记录精确一致（F1@5 11.90% / nDCG@5 35.33% / MRR@5 51.89%），验证评测口径正确

### 结果文件
- `.modeling-audit/nfcorpus_test_frozen.json`（完整报告）
- `.modeling-audit/nfcorpus_dev_ablation.json`
- `.modeling-audit/nfcorpus_test_fts_baseline.json`
- `.modeling-audit/nfcorpus_test_recall_analysis.json`
- `.modeling-audit/FROZEN_TEST_REPORT.md`

## 二、真实开放评测（pasa-real）

- 基线 run：`e14f531c0b6e4c808be450c6ed31d999`（2 条查询）
- MRR@5=0.25 / nDCG@5=0.107 / P@5=0.1，每条约 450s（当时 CPU），样本极小仅参考
- 报告：`backend/paper-test/results/runs/20260820T065557071477Z_e14f531c0b6e4c808be450c6ed31d999.json`
- record 模式 Provider 快照在 `backend/paper-test/cache/search_cache.sqlite3`，可 frozen 复测

## 三、环境与性能

- **GPU 加速**：torch 2.6.0+cu124（RTX 3060），BGE-M3 编码 150s→1.3s（快 100 倍），评测从小时级降为分钟级
- 配置：`BGE_M3_DEVICE=auto`、`BGE_M3_USE_FP16=1`（.env）

## 四、待办/已知限制

1. **功能2 召回瓶颈**：27.93% 候选召回，89.6% 为词法鸿沟 → 需全库 BGE dense/sparse 召回（build_pasa_vector_index.py）或引文追踪
2. **引文追踪改造**：代码已完成（多种子+双向+每轮触发，`_select_citation_seeds` / `search_semantic_scholar_neighborhood_multi`），但 S2/OpenAlex 被限流无法验证，待 API key
3. **功能4 SciMap 人工标注**：流水线待搭（100 条冻结查询双人盲标），当前无人工标注者
4. **TREC-PM**：topics 源被墙无法下载
