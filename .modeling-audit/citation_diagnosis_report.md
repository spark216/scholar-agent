# 引文扩展诊断分析（2026-08-21）

## 一、为什么引文拉入后召回率没变化（根因分析）

### 诊断数据（RealScholarQuery_40，5 个 gold）
```
候选池总数:      92
引文来源论文:      3（进入候选池）
引文进 top-k:     0/3
gold 命中候选池:  2/5
gold 命中引文论文: 0/5
引文保留率:        3/57（引文扩展返回 57 篇，仅 3 篇进最终候选池）
```

### 三层原因

**1. 种子选择不稳定（已修复）**
- 引文扩展第二轮触发，但种子来自第一轮候选
- 第一轮主要来源是 arXiv（候选只有 arxiv id，无 DOI/openalex），导致带 DOI 种子极少甚至 0 → 引文经常不触发（`no_seed_or_config`）
- ✅ 修复：`_resolve_openalex_work_ids` 支持 arxiv id（经 `10.48550/arXiv.xxx` DOI 反查）；`_select_citation_seeds` 接受 arxiv id

**2. 种子质量（已修复）**
- 种子按 provider rank 选，可能选到**无引文网络的新论文**（citation_count=0，如 arXiv 2505 的新稿）
- 这类种子 cites/referenced_works 全空，引文扩展返回 0，浪费预算
- ✅ 修复：种子优先选 `citation_count>0` 的成熟论文（两级排序：成熟度优先，再按 provider rank）

**3. 引文论文召回质量低（未修复，需进一步处理）**
- 引文扩展返回 57 篇，但**仅 3 篇进最终候选池**（3.3% 保留率）
- 这 3 篇是"相关领域 survey"（AI edge devices、Vision-Language Models 综述），**不是 gold 具体论文**
- **gold 命中引文论文 = 0/5**：引文网络（种子论文的引用/被引）≠ gold 答案集合
- 引文论文进池后又被 BGE/reranker 排到 top-k 之外（0/3 进 top-k）

### 结论
引文扩展**机制可用**（能拉回论文、能进候选池），但**召回质量（precision）低**：种子论文的引文网络覆盖的是"相关领域"而非"gold 精确答案"。这正是 PaSa 论文用 **Selector 学习相关性过滤**解决的问题——我们目前引文论文直接进池，缺乏相关性过滤，效果打折。

## 二、已修复问题清单
1. ✅ OpenAlex 引文源（替代被限流的 S2）
2. ✅ arxiv id 种子支持（第一轮 arXiv 候选可做种子）
3. ✅ 种子成熟度优先（citation_count>0，避免孤立新论文）
4. ✅ 引文步骤日志（`.modeling-audit/citation_steps.jsonl`，每步种子/方向/返回/标题）
5. ✅ 引文扩展结果缓存（`search_cache.sqlite3`，frozen 复用免重复调用）

## 三、待处理
- 引文论文相关性过滤（进池前按查询语义过滤，减少噪声）
- 引文论文与 gold 的匹配率验证（更大样本）
