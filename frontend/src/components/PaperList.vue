<script setup>
defineProps({
  papers: {
    type: Array,
    default: () => [],
  },
  decisionPolicy: {
    type: String,
    default: 'strict',
  },
})

function scorePercent(score) {
  return Math.round((score || 0) * 100)
}

const levelLabel = {
  high: '高度相关',
  partial: '部分相关',
  verify: '待核验',
  excluded: '已排除',
}

const facetLabel = {
  year: '年份',
  language: '语言',
  method: '方法',
  dataset: '数据集',
  venue: '期刊/会议',
  publication_type: '论文类型',
}

const evidenceLabel = {
  support: '支持',
  violate: '违反',
  unknown: '未知',
}

const reasonLabel = {
  CRITICAL_EVIDENCE_UNKNOWN: '关键条件证据不足',
  REQUIRED_METADATA_MISSING: '必要元数据缺失',
  METADATA_CONFLICT_REQUIRES_VERIFICATION: '元数据存在冲突',
  IDENTITY_REQUIRES_VERIFICATION: '论文身份或版本待确认',
}

function verificationReasons(paper) {
  return (paper.reason_codes || [])
    .map((reason) => reasonLabel[reason])
    .filter(Boolean)
}
</script>

<template>
  <section class="panel">
    <div class="head">
      <h3>论文列表</h3>
      <p v-if="decisionPolicy === 'recall_first'">
        召回优先：未知证据继续保留，弱元数据降为部分相关，仅明确冲突进入待核验
      </p>
      <p v-else>按证据化相关性排序，区分高度相关、部分相关与待核验结果</p>
    </div>

    <div class="list">
      <article v-for="paper in papers" :key="paper.id" class="paper">
        <div class="top">
          <span
            class="badge"
            :data-level="paper.relevance"
            :title="paper.relevance === 'verify' ? '排序靠前，但关键条件尚无可靠证据' : ''"
          >
            {{ levelLabel[paper.relevance] || paper.relevance }}
          </span>
          <span v-if="paper.score_kind !== 'not_available'" class="score" title="未校准展示值，不是相关概率">
            展示值 {{ scorePercent(paper.relevance_score) }}%
          </span>
          <span v-else class="score score--na">无可用分数</span>
        </div>

        <h4>
          <a v-if="paper.url" :href="paper.url" target="_blank" rel="noreferrer">{{ paper.title }}</a>
          <template v-else>{{ paper.title }}</template>
        </h4>

        <p class="meta">
          <span v-if="paper.published_at || paper.year">发表时间 {{ paper.published_at || paper.year }}</span>
          <span v-if="paper.venue">· {{ paper.venue }}</span>
          <span v-if="paper.source">· 来源 {{ paper.source }}</span>
          <span v-if="paper.authors?.length">· {{ paper.authors.join(', ') }}</span>
        </p>

        <p class="abstract">{{ paper.abstract }}</p>

        <p v-if="paper.relevance === 'verify'" class="verification-note">
          待核验：{{ verificationReasons(paper).join('；') || '关键条件尚无可靠证据，不能直接标为高度相关' }}
        </p>

        <div class="tags">
          <span v-for="tag in paper.tags" :key="tag">{{ tag }}</span>
        </div>

        <div v-if="Object.keys(paper.evidence_status || {}).length" class="evidence">
          <span
            v-for="(status, facet) in paper.evidence_status"
            :key="facet"
            :data-status="status"
          >
            {{ facetLabel[facet] || facet }}：{{ evidenceLabel[status] || status }}
          </span>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.panel {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem;
  backdrop-filter: blur(8px);
}

.head {
  margin-bottom: 1rem;
}

h3 {
  margin: 0;
  font-family: 'Fraunces', serif;
  font-size: 1.25rem;
}

.head p {
  margin: 0.35rem 0 0;
  color: var(--muted);
  font-size: 0.88rem;
}

.list {
  display: grid;
  gap: 0.85rem;
}

.paper {
  padding: 1rem 1.05rem;
  border-radius: 14px;
  border: 1px solid rgba(15, 42, 51, 0.08);
  background: rgba(247, 243, 234, 0.55);
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.paper:hover {
  transform: translateY(-2px);
  border-color: rgba(31, 111, 120, 0.28);
}

.top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.55rem;
}

.badge {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
}

.badge[data-level='high'] {
  color: var(--high);
  background: rgba(31, 122, 77, 0.12);
}

.badge[data-level='partial'] {
  color: var(--partial);
  background: rgba(154, 107, 47, 0.12);
}

.badge[data-level='verify'] {
  color: #805b1b;
  background: rgba(173, 126, 43, 0.14);
}

.badge[data-level='excluded'] {
  color: #9b3f35;
  background: rgba(155, 63, 53, 0.12);
}

.score {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: var(--sea);
}

.score--na { color: var(--muted); font-weight: 500; }

h4 {
  margin: 0;
  font-size: 1.02rem;
  line-height: 1.45;
}

h4 a {
  text-decoration: none;
  color: var(--ink);
}

h4 a:hover {
  color: var(--sea);
}

.meta {
  margin: 0.45rem 0 0.65rem;
  color: var(--muted);
  font-size: 0.82rem;
}

.abstract {
  margin: 0;
  color: var(--ink-soft);
  line-height: 1.6;
  font-size: 0.92rem;
}

.verification-note {
  margin: 0.65rem 0 0;
  padding: 0.55rem 0.7rem;
  border-radius: 9px;
  color: #7b5a22;
  background: rgba(154, 107, 47, 0.09);
  font-size: 0.8rem;
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.75rem;
}

.tags span {
  font-size: 0.74rem;
  padding: 0.22rem 0.5rem;
  border-radius: 999px;
  background: rgba(31, 111, 120, 0.1);
  color: var(--sea);
}

.evidence { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.65rem; }
.evidence span { font-size: 0.7rem; padding: 0.2rem 0.42rem; border-radius: 6px; background: rgba(80,90,90,.08); }
.evidence span[data-status='support'] { color: #176c46; background: rgba(23,108,70,.1); }
.evidence span[data-status='violate'] { color: #a53d31; background: rgba(165,61,49,.1); }
.evidence span[data-status='unknown'] { color: #866021; background: rgba(134,96,33,.1); }
</style>
