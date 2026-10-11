# 运行与模型准备

## 基础环境

建议 Python 3.11、Node.js 22.12+，Windows 或 Linux。安装 backend/requirements.txt，frontend 使用 npm ci。FAISS 与 NumPy 已显式列入发布版依赖。

复制根目录 .env.example 为 .env，配置使用的智谱与学术 API。不要提交 .env。前端开发端口 5173，API 8000；Vite 将 /api 转发至后端。公开包不携带原机器的模型路径和真实密钥。

## 模型

| 用途 | 模型 | revision |
| --- | --- | --- |
| 编码 | BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 |
| 精排 | BAAI/bge-reranker-v2-m3 | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| 可选 Selector | bytedance-research/pasa-7b-selector | 按下载工具准备，记录所用版本 |

```bash
python -m backend.scripts.download_evidrank_models --help
python -m backend.scripts.download_pasa_selector_model --help
python -m backend.scripts.shard_selector_model --help
```

离线部署配置 BGE_M3_MODEL_PATH、BGE_RERANKER_MODEL_PATH 与 BGE_LOCAL_FILES_ONLY=1。GPU 使用 BGE_M3_DEVICE=auto、BGE_M3_USE_FP16=1。Selector 需要支持的 bitsandbytes / Accelerate、足够显存和对应权重；分片目录通过 PASA_SELECTOR_MODEL_PATH 指定。

## 本地语料与索引

从 PaSa 官方数据来源准备 paper_database/id2paper.json，按需准备摘要包。设置 ACADEMIC_LOCAL_CORPUS_PATH / ACADEMIC_LOCAL_DOCUMENTS_PATH。FTS 首次使用自动建库。

```bash
python -m backend.scripts.build_pasa_vector_index --help
python -m backend.scripts.build_pasa_vector_index
```

设置 ACADEMIC_VECTOR_BACKEND=faiss，在构建完成后启用 ACADEMIC_LOCAL_VECTOR_ENABLED=1。索引根路径通过 ACADEMIC_LOCAL_VECTOR_INDEX_DIR 指定。公开仓库不包含 568k 语料和索引。

## 最佳实验配置与交互配置

| 配置 | 作用 |
| --- | --- |
| ACADEMIC_LOCAL_VECTOR_MULTIVARIANT=1 | 多查询变体融合 |
| PAPER_TEST_BGE_RERANKER_TOP_N=200 | 实验扩大精排 |
| PASA_SELECTOR_ENABLED=1 | 可选高质量判别器 |
| PASA_SELECTOR_RANK_WEIGHT=0.30 | 已保存对照中的融合权重 |
| PAPER_TEST_PASA_SELECTOR_TOP_N=80 | 限制 Selector 评测池 |
| CRAWLER_QUERY_FUSION_ENABLED=0 | 保持关闭，负向对照无稳定收益 |

默认配置为资源保守入口。最佳实验配置需要模型、全库索引和 GPU；CPU Docker 不复现其延迟。

## 数据集与评测

数据集不随发布包上传。按各数据集许可从官方来源下载，将 PAPER_TEST_DATASET_ROOT 指向数据根路径。NFCorpus 的固定 corpus / qrels 可以通过 build_nfcorpus_fixed_corpus 准备。

```bash
python -m backend.scripts.build_nfcorpus_fixed_corpus --help
python -m backend.scripts.evaluate_fixed_corpus --help
python -m backend.scripts.ablate_evidrank --help
python -m backend.scripts.ablate_scimap --help
```

历史实验结果为发布前保存的证据；若重跑当前发布代码，另存报告并记录 revision，不能覆盖旧结果后仍声称同一实验。

## API

- GET /api/health：依赖状态，部分可选基础设施未运行时可能返回 degraded。
- POST /api/search：同步检索。
- POST /api/search/stream：SSE 实时轨迹与最终结果。
- /api/paper-test/...：数据集、运行与历史评测接口，具体见 /docs。

数据库迁移顺序：000_base.sql → 001_evidrank.sql → 002_output_artifacts.sql。Docker 说明见根目录 DOCKER.md。开发默认端口与密码仅供本地运行，外部部署需自行配置。
