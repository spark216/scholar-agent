# ScholarAgent 竞赛评测与优化协议

本协议用于避免“换了候选、网络状态或测试样本，却把分数变化归因于算法”的无效实验。

## 1. 三条独立证据线

### A. 固定语料检索/排序实验

使用数据集自身 corpus 和 qrels，按稳定 doc_id 评分。该实验回答“召回器和排序器本身是否有效”。
BEIR 只有按此方式运行时才允许与相同版本、相同 split 的公开结果比较。

### B. 冻结 Provider 端到端实验

先以 `provider_cache_mode=record` 保存所有查询的 Provider 响应，再以 `frozen` 复测。
该实验回答“在完全相同候选池上，代码或排序改动是否有效”。冻结缓存缺失时整次运行必须失败，
不能退化成少 Provider 的零分报告。

### C. 真实开放网络压力测试

记录 API 错误、失败率、P50/P95、Token 和调用次数。该实验回答真实部署可靠性，不能冒充
固定语料排行榜成绩。

## 2. F1 报告规则

正式总体指标包含失败查询为零分；同时报告 `aggregate_successful` 和 `failure_breakdown`，
用于区分检索/排序效果与外部服务故障。不得只展示成功查询子集，也不得把开放 Provider 的
BEIR 标题代理分数写成官方 BEIR 成绩。

## 3. 可比较运行的硬条件

下列字段全部一致才允许计算算法差值：

- dataset id、split、SHA-256、sample qid 列表；
- K、seed；
- Provider snapshot SHA-256；
- 模型 path 与 revision；
- 候选预算、精排 Top-N、RRF k、权重和阈值版本；
- Fast Mode、决策策略和匹配策略；
- 代码 commit 已记录，工作树状态已披露。

## 4. 消融定义

“BGE-M3”和“RRF”不能含糊拆分：当前 BGE-M3 同时输出 dense 与 sparse，主实现使用 RRF
融合二者。正式消融采用以下可复现定义：

1. `FTS`：只用冻结的 PaSa FTS 排名；
2. `FTS + BGE dense`：同一 FTS 候选，只按 BGE-M3 dense cosine 排名；
3. `FTS + BGE dense/sparse RRF`：同一候选，按 dense 和 sparse 两路 RRF；
4. `FTS + BGE RRF + Reranker`：在第3项基础上加入固定 revision Cross-Encoder；
5. `Full EvidRank`：加入身份核验、硬约束、证据判定、等级和最终排序策略。

所有消融只在开发集选参数。测试集冻结后只能运行预先登记的配置，禁止逐题改权重。

## 5. 四功能验收

- 功能1：意图字段准确率、硬约束精确率/召回率、查询分解有效率和短查询误拒率；
- 功能2：Provider/merged/budget Candidate Recall、失败率、调用次数和P95；
- 功能3：Recall/F1/nDCG/MRR、硬约束违反率及每个阶段的首次命中/损失位置；
- 功能4：双人盲审一致性、claim citation precision、unsupported strong assertion rate、
  comparability accuracy 和 citation accuracy。

功能4双人标注流程见 `backend/paper-test/annotations/README.md`。

## 6. 当前不能直接声称的内容

- 未构建且未启用的全库向量索引不能写成“当前系统已使用”；
- 代码存在不等于真实模型、Docker、PostgreSQL、Milvus和外部API已经端到端通过；
- 20条辅助冒烟测试不能代表总体泛化性能；
- 没有人工标注结果时不能声称SciMap的引用和归纳准确率已经达到目标；
- 任何奖项只能作为质量目标，不能由当前代码或一次本地结果保证。
