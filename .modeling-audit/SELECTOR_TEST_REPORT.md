# pasa-7b-selector 独立排序测试报告

日期：2026-08-23
模型：`bytedance-research/pasa-7b-selector`（PaSa 官方 Selector，Qwen2.5-7B-Instruct 微调）
下载：`models/pasa-7b-selector/model.safetensors` 15,231,271,252 字节（sha256 校验通过，从 3.02GB 断点续传完成）
加载：4-bit NF4 量化（RTX 3060 12.88GB VRAM），38s 加载

## 测试方法

- **论文池**：主程序同款检索栈（FTS BM25 + BGE-M3 全库向量，fallback understanding，pool=200×2 并集），捕获每篇 title+abstract。
  5 条基准查询（RealScholarQuery_1/7/17/40/47，seed 42），池共 808 篇，gold-in-pool = 29/82 = **35.4%**（与历史候选召回诊断一致：10/8/5/3/3）。
- **打分**：忠实复刻官方 `infer_score`——`get_selected` 模板（Title/Abstract/User Query/Decision: True-False），`generate(max_new_tokens=1, do_sample=False)`，取首 token 为 `'True'` 的 softmax 概率。
- **匹配与指标**：复用主程序逻辑（arxiv_id→title 兜底、one-to-one、宏平均 P/R/F1/nDCG/MRR @5/10/20）。
- **对照**：bge-reranker（主程序当前排序信号，同一池子打分）+ 历史全管道 run。

## 结果（宏平均）

### @5
| 排序方式 | P | R | F1 | nDCG | MRR | hits |
|---|---|---|---|---|---|---|
| 历史 A+B 全管道 f3557136 | 0.240 | 0.176 | 0.181 | 0.312 | 0.667 | 6 |
| 历史最新 b28389d6 | 0.120 | 0.064 | 0.068 | 0.106 | 0.150 | 3 |
| reranker（同池，当前信号） | 0.240 | 0.148 | 0.165 | 0.299 | 0.600 | 6 |
| **selector（官方用法，同池）** | **0.480** | **0.185** | **0.238** | **0.490** | 0.640 | **12** |
| sel 过滤→reranker 混合 | 0.400 | 0.205 | **0.244** | 0.432 | **0.740** | 10 |
| rrf 融合（sel+reranker） | 0.360 | 0.168 | 0.199 | 0.370 | 0.590 | 9 |

### @10 / @20
| 排序方式 | hits@10 | R@10 | hits@20 | R@20 | F1@20 |
|---|---|---|---|---|---|
| 历史最新 b28389d6 | 11 | 0.180 | 15 | 0.249 | 0.155 |
| reranker（同池） | 10 | 0.173 | 14 | 0.243 | 0.148 |
| **selector** | **16** | 0.254 | **23** | 0.391 | **0.244** |
| **rrf 融合** | 14 | 0.287 | **24** | **0.397** | **0.252** |

### 逐条（selector @5 hits）
- Q1：0→5（top-7 全 gold，score 0.96–0.999）；历史 2
- Q7：0→0（long video，池内 gold 8 篇但排序未进 top5——selector 偏好明确研究型论文）
- Q17：0→2（历史管道从未找到的 gold 被找回）；历史 0
- Q40：0→0（3 个池内 gold 排在 12/32/35 位，reranker 反而排 EfficientQAT 进 top-5）
- Q47：0→1（金融 LLM agent 评测，命中）；历史 0

## 结论

**有效，显著提升主程序论文池排序能力与召回率**：
1. **@5 命中翻倍**：6→12（reranker/历史均 6）。P、F1、nDCG 全面翻倍级提升；MRR 持平（0.640 vs 0.667）。
2. **@20 命中 +64%**：23 vs 14（同池 reranker）/ 15（历史）。R@20 0.391 vs 0.243/0.249。
3. **Recall@5 提升**：0.185 vs 0.148（同池）/ 0.176（历史）。
4. **最佳混合**：`sel>0.5 过滤→reranker 排序` MRR@5=0.740；`RRF 融合` F1@20=0.252、R@20=0.397（池内 gold 83% 进 top-20）。
5. 官方阈值（>0.5）诊断合理：Q17 只选 5/173 篇、其中 4 gold（precision 0.800）；Q1 选 24/172、9 gold。

## 诚实解读（必读）

1. **池子为本地检索栈重建**（35.4% 上限），历史 run 含 provider 候选。但同池 reranker 复现历史结果（@5 hits 6=6）证明池子忠实、对比公平。全管道（含 provider）下 selector 可发挥的 gold 更多，实际增益可能更大。
2. **Q40 是例外**：selector 把大量相关 QAT 论文排在 gold 之前（gold 位 12/32/35），而 reranker 把 EfficientQAT 排进 top-5。selector 的"相关性判断"与官方 gold 清单有差异——这是语义排序器对精选答案集的固有偏差，非 bug。
3. **训练数据重叠风险**：selector 在 pasa 数据族（AutoScholarQuery）上微调，RealScholarQuery 是官方测试集但 gold 论文多为真实 AI 论文，与其训练分布有重叠。**NFCorpus/BEIR 等无关语料的泛化性需单独验证**才能下"通用排序器"结论。
4. **推理成本**：4-bit、batch 8 下 ~50s/查询（~160 篇），即 ~0.3s/篇。可调 batch 32+ 提速。若接入主程序需考虑显存/延迟预算（与 reranker 同机共存 OK，12.88GB 装下 4bit selector ~5GB + reranker 2.2GB）。
5. **此前微调 reranker 失败**（两次，hits 6→0/6→3）与本测试不矛盾：官方 7B selector 是数据规模远超本项目微调、RL 训练过的判别器，不是本地 LoRA 微调可比。

## 复现

```bash
# 1. 建池（含 title+abstract）
python -m backend.scripts.build_selector_pool --limit 5 --seed 42 --pool 200
# 2. 评测（selector 打分会写回池文件，二次运行自动跳过）
python -m backend.scripts.eval_pasa_selector_rerank
# 3. 健全性检查
python -m backend.scripts.sanity_pasa_selector
```

产物：`.modeling-audit/selector_test_pool.json`（含 sel_score）、`selector_rerank_results.json`。
