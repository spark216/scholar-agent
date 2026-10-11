# Selector 融入主程序：A/B 评测报告

生成时间：2026-08-24  |  pasa-real 5 条基准（seed 42，k 5/10/20）

## 1. 命中率（hits，两模型对比）

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 6 | 10 | — |
| OFF 基线（当前 reranker 管道） | 6 | 10 | 15 |
| ON 融合（+selector） | 11 | 13 | 20 |

### 离线同池单模型命中率（参照）

| 模型 @k | @5 | @10 | @20 |
|---|---|---|---|
| selector | 12 | 16 | 23 |
| reranker | 6 | 10 | 14 |
| rrf_fusion | 9 | 14 | 24 |

## 2. PRECISION @5/10/20

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0.240 | 0.200 | — |
| OFF 基线（当前 reranker 管道） | 0.240 | 0.200 | 0.150 |
| ON 融合（+selector） | 0.440 | 0.260 | 0.200 |

## 3. RECALL @5/10/20

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0.176 | 0.202 | — |
| OFF 基线（当前 reranker 管道） | 0.176 | 0.202 | 0.298 |
| ON 融合（+selector） | 0.225 | 0.254 | 0.297 |

## 4. F1 @5/10/20

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0.181 | 0.165 | — |
| OFF 基线（当前 reranker 管道） | 0.181 | 0.165 | 0.169 |
| ON 融合（+selector） | 0.255 | 0.217 | 0.201 |

## 5. NDCG @5/10/20

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0.312 | 0.309 | — |
| OFF 基线（当前 reranker 管道） | 0.312 | 0.309 | 0.318 |
| ON 融合（+selector） | 0.491 | 0.400 | 0.390 |

## 6. MRR @5/10/20

| 配置 @k | @5 | @10 | @20 |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0.667 | 0.687 | — |
| OFF 基线（当前 reranker 管道） | 0.667 | 0.687 | 0.687 |
| ON 融合（+selector） | 0.740 | 0.740 | 0.740 |

## 7. 成本（token / API 调用 / 延迟）

| 配置 | token 总量 | API 调用 | 总延迟 (s) |
|---|---|---|---|
| 历史 A+B 全管道 (f3557136) | 0 | 34 | 151 |
| OFF 基线（当前 reranker 管道） | 0 | 14 | 133 |
| ON 融合（+selector） | 0 | 14 | 295 |

## 8. 逐查询命中与返回明细

### RealScholarQuery_1

| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |
|---|---|---|---|---|---|---|---|
| 历史 | 2 | 5 | — | — | 1.000 | 10 | 5 |
| OFF | 2 | 5 | 5 | 0.1724137931034483 | 1.000 | 20 | 5 |
| ON | 5 | 6 | 9 | 0.3103448275862069 | 1.000 | 20 | 9 |

返回论文（ranking 阶段，rank/title/level/sel_score）:

| rank | level | sel_score | title |
|---|---|---|---|
| 1 | high | 0.998 | Parallel Structures in Pre-training Data Yield In-Context Learning |
| 2 | high | 0.998 | Understanding in-context learning via supportive pretraining data |
| 3 | high | 0.998 | A Theory of Emergent In-Context Learning as Implicit Structure Induction |
| 4 | high | 0.993 | An Explanation of In-context Learning as Implicit Bayesian Inference |
| 5 | high | 0.985 | Pre-training to learn in context |
| 6 | partial | 0.962 | Training Trajectories of Language Models Across Scales |
| 7 | partial | 0.995 | Cross-Lingual Supervision improves Large Language Models Pre-training |
| 8 | partial | 0.853 | Emergent Abilities in Reduced-Scale Generative Language Models |
| 9 | partial | 0.953 | In-context pretraining: Language modeling beyond document boundaries |
| 10 | partial | 0.629 | A Framework for Quantifying How Pre-Training and Context Benefit In-Context Lear |
| 11 | partial | 0.974 | The Learnability of In-Context Learning |
| 12 | partial | 0.939 | Learning to grok: Emergence of in-context learning and skill composition in modu |
| 13 | partial | 0.362 | Towards the Effect of Examples on In-Context Learning: A Theoretical Case Study |
| 14 | partial | 0.034 | Large language models are latent variable models: Explaining and finding good de |
| 15 | partial | 0.305 | Can Large Language Models Understand Context? |
| 16 | partial | 0.685 | Explaining Emergent In-Context Learning as Kernel Regression |
| 17 | partial | 0.191 | A survey on large language models with some insights on their capabilities and l |
| 18 | partial | 0.822 | On the effect of pretraining corpora on in-context learning by a large-scale lan |
| 19 | partial | 0.146 | A Survey of Large Language Models |
| 20 | partial | 0.997 | Language Models "Grok" to Copy |

### RealScholarQuery_17

| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |
|---|---|---|---|---|---|---|---|
| 历史 | 1 | 1 | — | — | 1.000 | 10 | 1 |
| OFF | 1 | 1 | 3 | 0.3333333333333333 | 1.000 | 20 | 3 |
| ON | 2 | 3 | 3 | 0.3333333333333333 | 1.000 | 20 | 3 |

返回论文（ranking 阶段，rank/title/level/sel_score）:

| rank | level | sel_score | title |
|---|---|---|---|
| 1 | high | 0.996 | Thinking about GPT-3 In-Context Learning for Biomedical IE? Think Again |
| 2 | high | 0.990 | Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Re |
| 3 | high | 0.520 | Large Language Models are Zero-Shot Relation Extractors with Aligned Instruction |
| 4 | partial | 0.146 | LLM's Weakness in NER Doesn't Stop It from Enhancing a ... |
| 5 | high | 0.220 | Recall, Retrieve and Reason: Towards Better In-Context Relation Extraction |
| 6 | partial | 0.036 | LinkNER: Linking Local Named Entity Recognition Models to Large Language Models  |
| 7 | partial | 0.050 | Do LLMs Surpass Encoders for Biomedical NER? |
| 8 | partial | 0.078 | Guideline Learning for In-Context Information Extraction |
| 9 | partial | 0.038 | Supervised Fine-Tuning or In-Context Learning? Evaluating LLMs for Clinical NER |
| 10 | partial | 0.146 | LLMs are not zero-shot reasoners for biomedical information extraction |
| 11 | partial | 0.003 | Is In-Context Learning Sufficient for Instruction Following in LLMs? |
| 12 | partial | 0.066 | Investigating LLM Applications in E-Commerce |
| 13 | partial | 0.005 | How Good are LLMs at Relation Extraction under Low-Resource Scenario? Comprehens |
| 14 | partial | 0.008 | How far is Language Model from 100% Few-shot Named Entity Recognition in Medical |
| 15 | partial | 0.261 | Mining experimental data from Materials Science literature with Large Language M |
| 16 | partial | 0.002 | Large language models meet nlp: A survey |
| 17 | partial | 0.006 | Assessing the Performance of Chinese Open Source Large Language Models in Inform |
| 18 | partial | 0.203 | GPT-NER: Named Entity Recognition via Large Language Models |
| 19 | partial | 0.003 | c-ICL: Contrastive In-context Learning for Information Extraction |
| 20 | partial | 0.004 | Is 'Hope' a person or an idea? A pilot benchmark for NER: comparing traditional  |

### RealScholarQuery_40

| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |
|---|---|---|---|---|---|---|---|
| 历史 | 1 | 1 | — | — | 0.333 | 10 | 1 |
| OFF | 1 | 1 | 2 | 0.4 | 0.333 | 20 | 2 |
| ON | 1 | 1 | 1 | 0.2 | 0.200 | 20 | 1 |

返回论文（ranking 阶段，rank/title/level/sel_score）:

| rank | level | sel_score | title |
|---|---|---|---|
| 1 | high | 0.998 | HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework fo |
| 2 | high | 0.997 | MEC-Quant: Maximum Entropy Coding for Extremely Low Bit Quantization-Aware Train |
| 3 | high | 0.995 | StableQAT: Stable Quantization-Aware Training at Ultra-Low Bitwidths |
| 4 | high | 0.987 | DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Languag |
| 5 | high | 0.971 | EfficientQAT: Efficient Quantization-Aware Training for Large Language Models |
| 6 | partial | 0.982 | OneBit: Towards Extremely Low-bit Large Language Models |
| 7 | partial | 0.995 | Quantized Approximately Orthogonal Recurrent Neural Networks |
| 8 | partial | 0.987 | What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A System |
| 9 | partial | 0.995 | SQuAT: Sharpness- and Quantization-Aware Training for BERT |
| 10 | partial | 0.448 | Compute-Optimal Quantization-Aware Training | alphaXiv |
| 11 | partial | 0.983 | Power-of-Two Quantization for Low Bitwidth and Hardware Compliant Neural Network |
| 12 | partial | 0.687 | Compute-Optimal Quantization-Aware Training |
| 13 | partial | 0.940 | GAQAT: gradient-adaptive quantization-aware training for domain generalization |
| 14 | partial | 0.987 | pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantizat |
| 15 | partial | 0.687 | Once quantization-aware training: High performance extremely low-bit architectur |
| 16 | partial | 0.984 | Error-aware Quantization through Noise Tempering |
| 17 | partial | 0.995 | BinaryDM: Accurate Weight Binarization for Efficient Diffusion Models |
| 18 | partial | 0.711 | EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Model |
| 19 | partial | 0.856 | Poster: Self-Supervised Quantization-Aware Knowledge Distillation |
| 20 | partial | 0.979 | PB-LLM: Partially Binarized Large Language Models |

### RealScholarQuery_47

| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |
|---|---|---|---|---|---|---|---|
| 历史 | 2 | 2 | — | — | 1.000 | 10 | 2 |
| OFF | 2 | 2 | 2 | 0.5 | 1.000 | 20 | 2 |
| ON | 2 | 2 | 2 | 0.5 | 1.000 | 20 | 2 |

返回论文（ranking 阶段，rank/title/level/sel_score）:

| rank | level | sel_score | title |
|---|---|---|---|
| 1 | high | 0.999 | The FinBen: An Holistic Financial Benchmark for Large Language Models |
| 2 | high | 0.999 | UCFE: A User-Centric Financial Expertise Benchmark for Large Language Models |
| 3 | high | 0.998 | PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Fin |
| 4 | high | 0.991 | Towards a Realistic Long-Term Benchmark for Open-Web Research Agents |
| 5 | partial | 0.998 | Finance agent benchmark: Benchmarking llms on real-world financial research task |
| 6 | partial | 0.997 | Investorbench: A benchmark for financial decision-making tasks with llm-based ag |
| 7 | partial | 0.998 | FinToolBench: Evaluating LLM Agents for Real-World Financial Tool Use |
| 8 | partial | 0.427 | AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Ag |
| 9 | partial | 0.967 | Dólares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between S |
| 10 | partial | 0.885 | Standard Benchmarks Fail--Auditing LLM Agents in Finance Must Prioritize Risk |
| 11 | partial | 0.008 | LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners |
| 12 | partial | 0.012 | A Survey of Useful LLM Evaluation |
| 13 | partial | 0.163 | AgentSims: An Open-Source Sandbox for Large Language Model Evaluation |
| 14 | partial | 0.003 | GABench: A Comprehensive Benchmark for Evaluating LLM Agents on Graph Analysis T |
| 15 | partial | 0.660 | AgentBench: Evaluating LLMs as Agents |
| 16 | partial | 0.984 | CFBenchmark: Chinese Financial Assistant Benchmark for Large Language Model |
| 17 | partial | 0.007 | \benchmarkname : A Benchmark for LLMs as Intelligent Agents |
| 18 | partial | 0.134 | InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks |
| 19 | partial | 0.004 | LegalAgentBench: Evaluating LLM Agents in Legal Domain |
| 20 | partial | 0.078 | MIRAI: Evaluating LLM Agents for Event Forecasting |

### RealScholarQuery_7

| 查询/配置 | hits@5 | hits@10 | hits@20 | R@20 | MRR@5 | 返回数 | 匹配 |
|---|---|---|---|---|---|---|---|
| 历史 | 0 | 1 | — | — | 0.000 | 10 | 1 |
| OFF | 0 | 1 | 3 | 0.08571428571428572 | 0.000 | 20 | 3 |
| ON | 1 | 1 | 5 | 0.14285714285714285 | 0.500 | 20 | 5 |

返回论文（ranking 阶段，rank/title/level/sel_score）:

| rank | level | sel_score | title |
|---|---|---|---|
| 1 | high | 0.999 | Storyteller: Improving long video description through global audio-visual charac |
| 2 | high | 0.999 | Video ReCap: Recursive Captioning of Hour-Long Videos |
| 3 | high | 0.997 | Text Synopsis Generation for Egocentric Videos |
| 4 | high | 0.996 | Multi Semantic Feature Fusion Framework for Video Segmentation and Description |
| 5 | high | 0.994 | Goldfish: Vision-Language Understanding of Arbitrarily Long Videos |
| 6 | partial | 0.967 | InfiniBench: A Comprehensive Benchmark for Large Multimodal Models in Very Long  |
| 7 | partial | 0.991 | A Video Is Worth 4096 Tokens: Verbalize Videos To Understand Them In Zero Shot |
| 8 | partial | 0.995 | Videoclip-xl: Advancing long description understanding for video clip models |
| 9 | partial | 0.984 | From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensi |
| 10 | partial | 0.877 | VISTA: Enhancing Long-Duration and High-Resolution Video ... |
| 11 | partial | 0.974 | LLMs Meet Long Video: Advancing Long Video Comprehension with An Interactive Vis |
| 12 | partial | 0.995 | Longvlm: Efficient long video understanding via large language models |
| 13 | partial | 0.971 | VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal  |
| 14 | partial | 0.953 | Towards Long-Form Video Understanding |
| 15 | partial | 0.984 | A Video is Worth 10,000 Words: Training and Benchmarking with Diverse Captions f |
| 16 | partial | 0.986 | Koala: Key frame-conditioned long video-LLM |
| 17 | partial | 0.020 | A Comprehensive Review on Recent Methods and Challenges of Video Description |
| 18 | partial | 0.991 | Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding |
| 19 | partial | 0.449 | How to Make a BLT Sandwich? Learning to Reason towards Understanding Web Instruc |
| 20 | partial | 0.993 | TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Unders |

## 9. 逐查询成本

| qid | 配置 | token | API | 延迟 (s) |
|---|---|---|---|---|
| RealScholarQuery_1 | OFF | 0 | 4 | 65 |
| RealScholarQuery_1 | ON | 0 | 4 | 131 |
| RealScholarQuery_17 | OFF | 0 | 4 | 18 |
| RealScholarQuery_17 | ON | 0 | 4 | 43 |
| RealScholarQuery_40 | OFF | 0 | 2 | 16 |
| RealScholarQuery_40 | ON | 0 | 2 | 41 |
| RealScholarQuery_47 | OFF | 0 | 2 | 17 |
| RealScholarQuery_47 | ON | 0 | 2 | 40 |
| RealScholarQuery_7 | OFF | 0 | 2 | 17 |
| RealScholarQuery_7 | ON | 0 | 2 | 40 |

## 10. 结论

- 命中率：@5 6 → 11（提升），@20 15 → 20（提升）
- F1@5 0.181 → 0.255；MRR@5 0.667 → 0.740

> 注意：本报告的 ON 为融合模式（selector 权重 0.30，头部保护开，打分 top-80）。
> 两个模型的取舍请结合『离线同池单模型命中率』与生产延迟预算决策。