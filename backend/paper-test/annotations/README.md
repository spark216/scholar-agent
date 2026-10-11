# SciMap 双人盲审验收

该目录只保存标注规范和小型模板，不提交含个人身份信息的工作文件。代码不能替代人工判断，
也不得用模型输出自动填充“人工标签”。

## 抽样要求

1. 从冻结测试集中抽取至少 100 个不同 `query_id`，按领域、查询长度、约束数量和结果质量分层。
2. 每个查询固定同一 `FinalRankingBundle` 和 `OutputArtifact`，记录代码 commit、数据 SHA、
   Provider snapshot、模型 revision、配置指纹和 artifact hash。
3. 标注员 A、B 独立工作，互不可见；分歧由第三人裁决。
4. 开发集可用于修规则，测试集只允许一次冻结验收，不得逐题调参。

## 标注单位

每行对应一个已冻结的 `query-paper-claim-evidence` 审查单位，五个二元字段统一采用 `1/0`：

- `paper_relevant`：论文是否满足原始查询；
- `evidence_supports_claim`：引用片段是否支持当前 claim；
- `claim_overstated`：claim 是否超出证据边界，1 表示过强；
- `comparison_comparable`：系统给出的可比性判断是否正确；
- `citation_accurate`：引用是否指向正确论文、版本和片段。

每行还必须填写不同的 `annotator_a`、`annotator_b`；若有分歧，`adjudicator` 必须是与
A/B 都不同的第三人。若 A/B 标签一致，`*_final` 可留空且不得改写一致标签；若不一致，
第三人必须填写 `*_final`。不能用空值、
“不确定”或模型预测绕过分歧裁决；确实无法判断的样本应记录原因并在冻结抽样规则下替换。

## 命令

从已经整理为 JSONL 的冻结审查单位初始化两份互相隔离的空表。A、B 标注员只能收到
各自文件，不能访问另一份：

```powershell
python -m backend.scripts.evaluate_scimap_annotations init `
  --source backend/paper-test/annotations/scimap_units.jsonl `
  --output-a backend/paper-test/annotations/scimap_annotator_a.csv `
  --output-b backend/paper-test/annotations/scimap_annotator_b.csv `
  --limit 100
```

两人独立完成后，由负责人合并成裁决表：

```powershell
python -m backend.scripts.evaluate_scimap_annotations merge `
  --source-a backend/paper-test/annotations/scimap_annotator_a.csv `
  --source-b backend/paper-test/annotations/scimap_annotator_b.csv `
  --output backend/paper-test/annotations/scimap_adjudication.csv
```

第三人只处理合并表中的分歧，填写 `adjudicator` 和对应 `*_final`，然后计算指标：

```powershell
python -m backend.scripts.evaluate_scimap_annotations score `
  --source backend/paper-test/annotations/scimap_adjudication.csv `
  --output backend/paper-test/annotations/scimap_human_metrics.json
```

脚本会拒绝少于 100 个不同查询、A/B 标签缺失以及分歧未裁决的文件，并输出：

- 每个维度的 raw agreement 和 Cohen's kappa；
- claim citation precision；
- unsupported strong assertion rate；
- comparability accuracy；
- citation accuracy。

这些指标必须同时报告样本数、抽样方法和置信区间；不得只报告最好的一次运行。
