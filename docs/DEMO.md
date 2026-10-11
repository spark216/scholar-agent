# 真实运行演示

日期：2026 年 10 月 11 日。通过本地网页提交查询，完成真实检索、模型排序和 SciMap 研究结果整理。

## 查询内容

> Find papers explaining how large language models acquire in-context learning during pre-training.

## 演示配置

| 组件 | 配置 |
| --- | --- |
| 前端 | Vue 3 与 Vite，本地端口 5173 |
| 后端 | FastAPI 与 Uvicorn，本地端口 8000 |
| 学术检索 | 实际配置的在线学术服务与本地语料 |
| 语义模型 | BGE-M3 |
| 精细排序 | BGE-reranker-v2-m3 |
| 本地向量检索 | FAISS，1024 维，568,059 篇论文索引 |
| PaSa Selector | 本次演示关闭 |
| PostgreSQL | 本次演示未启动，网页保留持久化异常提示 |

## 网页返回结果

论文列表展示 5 篇论文；SciMap 为参与整理的候选生成 80 张证据卡片、2 条证据结论、3 条不确定性提示和 2 条事实关系边。论文列表数量与证据卡片数量分别对应展示结果与候选整理范围。

| 排名 | 论文 |
| --- | --- |
| 1 | An Explanation of In-context Learning as Implicit Bayesian Inference |
| 2 | Spin glass model of in-context learning |
| 3 | Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning |
| 4 | Explanations from Large Language Models Make Small Reasoners Better |
| 5 | What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning |

## 页面截图

### 查询入口

![查询入口](assets/demo-home.jpg)

### 执行记录

![执行记录](assets/demo-execution.jpg)

### 论文与关系图

![论文与关系图](assets/demo-papers.jpg)

### 证据比较矩阵

![证据比较矩阵](assets/demo-scimap.jpg)

## 复现操作

按照 [运行指南](SETUP.md) 准备 API、模型和本地索引，启动后端与前端。在查询框输入上述研究需求，点击“开始智能检索”。完成后查看执行记录、论文列表、关系图和可审计结果地图。在线服务响应与候选内容可能随运行时间变化。
