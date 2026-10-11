# 第三方模型与数据

项目使用 Vue、FastAPI、LangGraph、FAISS、Milvus、FlagEmbedding 等开源依赖，及 BAAI/bge-m3、BAAI/bge-reranker-v2-m3、bytedance-research/pasa-7b-selector 模型。

本工程提供检索、编排、推理接入、融合与评测实现，不主张训练了这些基础模型。各依赖、权重和数据集分别遵循其上游许可。

数据来源包括 PaSa、BEIR NFCorpus / ArguAna / SciDocs 与 SPARBench；语料与模型需另行获取，公开包保留选取的实验指标和报告。API 的访问和使用遵循对应服务条款。

仓库未追加未经核实的统一开源授权；若需要正式确定代码许可证，应先确认原始项目与贡献者授权。
