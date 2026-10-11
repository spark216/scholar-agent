<script setup>
import { computed } from 'vue'

const props = defineProps({
  artifact: {
    type: Object,
    required: true,
  },
})

const facets = computed(() => props.artifact?.comparison_matrix?.facets || [])
const rows = computed(() => props.artifact?.comparison_matrix?.rows || [])
const claims = computed(() => props.artifact?.synthesis?.claims || [])
const validClaims = computed(() => claims.value.filter((claim) => claim.validation_status === 'valid'))
const invalidClaims = computed(() => claims.value.filter((claim) => claim.validation_status !== 'valid'))
const uncertaintyTypes = new Set(['gap', 'incomparable'])
const evidenceClaims = computed(() => validClaims.value.filter((claim) => !uncertaintyTypes.has(claim.claim_type)))
const uncertaintyClaims = computed(() => validClaims.value.filter((claim) => uncertaintyTypes.has(claim.claim_type)))
const visibleUncertaintyClaims = computed(() => uncertaintyClaims.value.slice(0, 20))
const cards = computed(() => props.artifact?.ranked_list || [])
const edges = computed(() => props.artifact?.evidence_graph?.edges || [])
const titleByPaper = computed(() => Object.fromEntries(cards.value.map((card) => [card.paper_id, card.title])))

const statusLabel = {
  support: '支持',
  violate: '违反',
  unknown: '未知',
  partially_comparable: '有限可比',
  not_comparable: '不可比',
  insufficient_evidence: '证据不足',
}

const facetLabel = {
  year: '年份',
  language: '语言',
  method: '方法',
  dataset: '数据集',
  venue: '期刊/会议',
  publication_type: '论文类型',
}

function shortHash(value) {
  return value ? value.slice(0, 12) : '-'
}

function paperLabel(paperId) {
  return titleByPaper.value[paperId] || paperId
}
</script>

<template>
  <section class="scimap panel">
    <div class="head">
      <div>
        <p class="eyebrow">SCIMAP · 证据化科研归纳</p>
        <h3>可审计结果地图</h3>
      </div>
      <div class="hashes">
        <span>Bundle {{ shortHash(artifact.bundle_hash) }}</span>
        <span>Artifact {{ shortHash(artifact.artifact_hash) }}</span>
      </div>
    </div>

    <div class="stats">
      <span>论文证据卡 {{ cards.length }}</span>
      <span>证据结论 {{ evidenceClaims.length }}</span>
      <span>不确定性提示 {{ uncertaintyClaims.length }}</span>
      <span>已拦截结论 {{ invalidClaims.length }}</span>
      <span>事实边 {{ edges.length }}</span>
    </div>

    <div v-if="rows.length" class="block">
      <h4>查询自适应比较矩阵</h4>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>论文</th>
              <th v-for="facet in facets" :key="facet">{{ facetLabel[facet] || facet }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.paper_id">
              <td>#{{ row.final_rank }} {{ paperLabel(row.paper_id) }}</td>
              <td v-for="facet in facets" :key="facet">
                <span class="state" :data-state="row.cells?.[facet]?.status || 'unknown'">
                  {{ statusLabel[row.cells?.[facet]?.status] || '未知' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="two-col">
      <div class="block">
        <h4>证据绑定结论</h4>
        <ul v-if="evidenceClaims.length" class="claim-list">
          <li v-for="claim in evidenceClaims" :key="claim.claim_id" data-valid="valid">
            <div>
              <b>{{ claim.assertion_level }}级</b>
              <span>{{ claim.claim_type }}</span>
              <span>已校验</span>
            </div>
            <p>{{ claim.text }}</p>
            <small>论文 {{ claim.paper_ids.length }} · 证据 {{ claim.evidence_ids.length }}</small>
          </li>
        </ul>
        <p v-else class="empty">当前证据不足，系统没有生成可发布结论。</p>
      </div>

      <div class="block">
        <h4>可比性审查</h4>
        <ul v-if="artifact.comparability_assessments?.length" class="compact-list">
          <li v-for="item in artifact.comparability_assessments" :key="item.assessment_id">
            <span class="state" :data-state="item.status">
              {{ statusLabel[item.status] || item.status }}
            </span>
            <b>{{ paperLabel(item.left_paper_id) }}</b> ↔ <b>{{ paperLabel(item.right_paper_id) }}</b>
            <small v-if="item.reason_codes?.length">{{ item.reason_codes.join(' · ') }}</small>
          </li>
        </ul>
        <p v-else class="empty">不足两篇可比较论文，未生成成对审查。</p>
      </div>
    </div>

    <div v-if="uncertaintyClaims.length" class="block">
      <h4>不确定性与证据缺口</h4>
      <ul class="claim-list">
        <li v-for="claim in visibleUncertaintyClaims" :key="claim.claim_id">
          <p>{{ claim.text }}</p>
        </li>
      </ul>
      <p v-if="uncertaintyClaims.length > visibleUncertaintyClaims.length" class="empty">
        当前仅展示前 {{ visibleUncertaintyClaims.length }} 条，共 {{ uncertaintyClaims.length }} 条。
      </p>
    </div>

    <div class="two-col">
      <div class="block">
        <h4>确定性时间线</h4>
        <ol class="compact-list timeline">
          <li v-for="item in artifact.timeline || []" :key="item.paper_id">
            <b>{{ item.year || '年份未知' }}</b>
            <span>#{{ item.final_rank }} {{ item.title }}</span>
          </li>
        </ol>
      </div>

      <div class="block">
        <h4>证据关系边</h4>
        <ul class="compact-list">
          <li v-for="edge in edges.slice(0, 12)" :key="`${edge.source}-${edge.target}-${edge.relation}`">
            <span class="state" :data-state="edge.relation">
              {{ edge.relation === 'supports' ? '支持' : '违反' }}
            </span>
            <span>{{ edge.source }} → {{ edge.target }}</span>
            <small>证据 {{ edge.evidence_ids.length }}</small>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  background: rgba(255, 255, 255, 0.68);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1.2rem;
  backdrop-filter: blur(8px);
}

.head, .stats, .two-col, .claim-list li > div, .compact-list li {
  display: flex;
  align-items: center;
}

.head { justify-content: space-between; gap: 1rem; }
.eyebrow { margin: 0 0 0.25rem; color: var(--sea); font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; }
h3, h4, p { margin-top: 0; }
h3 { margin-bottom: 0; font-family: 'Fraunces', serif; }
h4 { margin-bottom: 0.7rem; font-size: 0.92rem; }
.hashes, .stats { display: flex; flex-wrap: wrap; gap: 0.45rem; }
.hashes span, .stats span { padding: 0.3rem 0.55rem; border-radius: 999px; background: rgba(31, 111, 120, 0.08); font-size: 0.72rem; }
.stats { margin: 1rem 0; }
.block { margin-top: 1rem; padding: 0.9rem; border: 1px solid rgba(15, 42, 51, 0.08); border-radius: 12px; background: rgba(247, 243, 234, 0.45); min-width: 0; }
.two-col { align-items: stretch; display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
th, td { padding: 0.5rem; border-bottom: 1px solid rgba(15, 42, 51, 0.08); text-align: left; white-space: nowrap; }
.state { display: inline-block; padding: 0.18rem 0.42rem; border-radius: 999px; font-size: 0.68rem; background: rgba(80, 90, 90, 0.1); }
.state[data-state='support'], .state[data-state='supports'], .state[data-state='partially_comparable'] { color: #176c46; background: rgba(23, 108, 70, 0.11); }
.state[data-state='violate'], .state[data-state='violates'], .state[data-state='not_comparable'] { color: #a53d31; background: rgba(165, 61, 49, 0.1); }
.state[data-state='unknown'], .state[data-state='insufficient_evidence'] { color: #866021; background: rgba(134, 96, 33, 0.11); }
.claim-list, .compact-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.55rem; }
.claim-list li { padding: 0.65rem; border-left: 3px solid #1f6f78; background: rgba(255,255,255,.45); }
.claim-list li[data-valid='invalid'] { border-color: #a53d31; opacity: 0.75; }
.claim-list li > div { gap: 0.45rem; font-size: 0.72rem; }
.claim-list p { margin: 0.4rem 0; line-height: 1.45; font-size: 0.86rem; }
small { display: block; color: var(--muted); }
.compact-list li { align-items: flex-start; flex-wrap: wrap; gap: 0.4rem; font-size: 0.78rem; }
.compact-list small { width: 100%; padding-left: 0.2rem; }
.timeline li { display: grid; grid-template-columns: 5rem 1fr; }
.empty { color: var(--muted); font-size: 0.82rem; }

@media (max-width: 800px) {
  .head { align-items: flex-start; flex-direction: column; }
  .two-col { grid-template-columns: 1fr; }
}
</style>
