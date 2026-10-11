<script setup>
import { computed, defineAsyncComponent, onMounted, ref } from 'vue'
import { fetchHealth, searchPapersStream } from './api/search'
import SearchHero from './components/SearchHero.vue'
import UnderstandingPanel from './components/UnderstandingPanel.vue'
import PaperList from './components/PaperList.vue'
import AgentTrace from './components/AgentTrace.vue'
import SummaryPanel from './components/SummaryPanel.vue'

const RelationGraphView = defineAsyncComponent(
  () => import('./components/RelationGraphView.vue'),
)
const SciMapPanel = defineAsyncComponent(
  () => import('./components/SciMapPanel.vue'),
)
const PaperTestPanel = defineAsyncComponent(
  () => import('./components/PaperTestPanel.vue'),
)

const query = ref(
  '面向复杂学术查询的 LLM Agent 论文搜索：查询分解、迭代检索与综合排序',
)
const loading = ref(false)
const error = ref('')
const result = ref(null)
const liveTrace = ref([])
const health = ref(null)
const hasSearched = ref(false)
const showInvalidModal = ref(false)
const showPaperTest = ref(false)
const invalidMessage = ref('请输入查找论文相关问题')

const emptyUnderstanding = {
  intent: '正在解析查询意图…',
  is_paper_search: true,
  reject_reason: null,
  domain: null,
  topics: [],
  entities: [],
  methods: [],
  datasets: [],
  venues: [],
  time_range: null,
  keywords: [],
  language: 'zh+en',
  sub_queries: [],
  rewritten_queries: [],
  raw_json: {},
}

const highCount = computed(
  () => result.value?.papers?.filter((p) => p.relevance === 'high').length ?? 0,
)
const partialCount = computed(
  () => result.value?.papers?.filter((p) => p.relevance === 'partial').length ?? 0,
)
const verifyCount = computed(
  () => result.value?.meta?.ranking_levels?.verify
    ?? result.value?.papers?.filter((p) => p.relevance === 'verify').length
    ?? 0,
)
const excludedCount = computed(() => result.value?.meta?.ranking_levels?.excluded ?? 0)

const showWorkspace = computed(() => hasSearched.value && (loading.value || result.value))

const healthText = computed(() => {
  if (health.value?.status === 'ok') return '后端已连接'
  if (health.value?.status === 'degraded') return '后端已连接（部分依赖异常）'
  return '后端未连接'
})

const currentUnderstanding = computed(
  () => result.value?.understanding || emptyUnderstanding,
)

async function runSearch() {
  const q = query.value.trim()
  if (!q || loading.value) return
  loading.value = true
  error.value = ''
  showInvalidModal.value = false
  hasSearched.value = true
  liveTrace.value = []
  result.value = {
    query: q,
    understanding: { ...emptyUnderstanding },
    papers: [],
    graph: { nodes: [], edges: [] },
    summary: {
      overview: 'Agent 正在执行中，请查看下方实时流程…',
      key_findings: [],
      research_gaps: [],
    },
    agent_trace: [],
    meta: { mode: 'streaming' },
    session_id: null,
    ranking_bundle: null,
    output_artifact: null,
  }

  try {
    const final = await searchPapersStream(
      q,
      { useMock: false, maxPapers: 5 },
      {
        onStep: (data) => {
          if (Array.isArray(data.agent_trace)) {
            liveTrace.value = data.agent_trace
            if (result.value) result.value.agent_trace = data.agent_trace
          } else if (data.step) {
            const next = [...liveTrace.value]
            const idx = next.findIndex((s) => s.step === data.step.step)
            if (idx >= 0) next[idx] = data.step
            else next.push(data.step)
            liveTrace.value = next
            if (result.value) result.value.agent_trace = next
          }
        },
        onUnderstanding: (data) => {
          if (result.value && data.understanding) {
            result.value.understanding = data.understanding
          }
        },
        onInvalidQuery: (data) => {
          invalidMessage.value = data?.message || data?.reject_reason || '请输入查找论文相关问题'
          showInvalidModal.value = true
          hasSearched.value = false
          result.value = null
          liveTrace.value = []
        },
        onResult: (data) => {
          result.value = data.result
          liveTrace.value = data.result?.agent_trace || liveTrace.value
        },
        onError: (data) => {
          error.value = data?.detail || '搜索失败'
        },
      },
    )
    if (!final && !showInvalidModal.value) {
      error.value = '搜索未返回结果'
    }
  } catch (err) {
    if (!showInvalidModal.value) {
      error.value = error.value || err?.message || '搜索失败'
    }
  } finally {
    loading.value = false
  }
}

function closeInvalidModal() {
  showInvalidModal.value = false
}

onMounted(async () => {
  try {
    health.value = await fetchHealth()
  } catch {
    health.value = { status: 'offline' }
  }
})
</script>

<template>
  <div class="page" :class="{ 'page--results': hasSearched && result }">
    <header class="topbar">
      <div class="brand-mark" aria-hidden="true"></div>
      <div class="brand-text">
        <strong>ScholarAgent</strong>
        <span>科研场景智能论文搜索</span>
      </div>
      <div class="status" :data-online="health?.status === 'ok' || health?.status === 'degraded'">
        {{ healthText }}
      </div>
      <button class="paper-test-button" type="button" @click="showPaperTest = true">
        test-paper
      </button>
    </header>

    <SearchHero
      v-model="query"
      :loading="loading"
      :compact="Boolean(hasSearched && result)"
      @submit="runSearch"
    />

    <p v-if="error" class="error">{{ error }}</p>

    <PaperTestPanel v-if="showPaperTest" @close="showPaperTest = false" />

    <div v-if="showInvalidModal" class="modal-mask" @click.self="closeInvalidModal">
      <div class="modal" role="dialog" aria-modal="true">
        <h3>提示</h3>
        <p>{{ invalidMessage }}</p>
        <button type="button" @click="closeInvalidModal">知道了</button>
      </div>
    </div>

    <section v-if="showWorkspace" class="workspace">
      <div class="workspace-head">
        <div>
          <h2>{{ loading ? 'Agent 执行中…' : '检索结果' }}</h2>
          <p>
            <template v-if="loading">实时展示思考 / 工具调用进度</template>
            <template v-else>
              Top {{ result.papers.length }} · 高度相关 {{ highCount }} · 部分相关
              {{ partialCount }} · 待核验 {{ verifyCount }} · 已排除 {{ excludedCount }}
              <template v-if="result.session_id"> · session {{ result.session_id.slice(0, 8) }}</template>
            </template>
          </p>
        </div>
        <div class="meta-pills">
          <span>{{ result?.meta?.mode || 'streaming' }}</span>
          <span v-if="loading" class="live">LIVE</span>
          <span>意图 {{ result?.meta?.intent_provider || '-' }}</span>
          <span>向量 {{ result?.meta?.embedding_provider || result?.meta?.model_versions?.embedding || '-' }}</span>
          <span>召回 {{ result?.meta?.retriever || (result?.meta?.ranking_run_id ? 'Dense+Sparse RRF' : '-') }}</span>
          <span>精排 {{ result?.meta?.reranker || result?.meta?.model_versions?.reranker || '-' }}</span>
          <span v-if="result?.meta?.ranking_run_id">EvidRank {{ result.meta.ranking_run_id.slice(0, 8) }}</span>
          <span v-if="result?.meta?.scimap_artifact_hash">SciMap {{ result.meta.scimap_artifact_hash.slice(0, 8) }}</span>
          <span>PG {{ result?.meta?.postgres || '-' }}</span>
        </div>
      </div>

      <div class="grid">
        <AgentTrace :trace="liveTrace.length ? liveTrace : result?.agent_trace || []" />
        <UnderstandingPanel :understanding="currentUnderstanding" />
        <SummaryPanel v-if="!loading || result?.summary" :summary="result.summary" />
        <RelationGraphView v-if="!loading && result?.graph?.nodes?.length" :graph="result.graph" />
        <PaperList
          v-if="!loading && result?.papers?.length"
          :papers="result.papers"
          :decision-policy="result.meta?.evidrank_decision_policy"
        />
        <SciMapPanel
          v-if="!loading && result?.output_artifact"
          :artifact="result.output_artifact"
        />
      </div>
    </section>

    <footer class="footer">
      <span>赛题：科研场景下复杂学术查询的智能论文搜索与推荐</span>
      <span>技术栈：Vue3 · FastAPI · BGE-M3 · 智谱 GLM</span>
    </footer>
  </div>
</template>

<style scoped>
.page {
  width: min(1180px, calc(100% - 2rem));
  margin: 0 auto;
  padding: 1.25rem 0 3rem;
}

.topbar {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  margin-bottom: 1.5rem;
}

.brand-mark {
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 0.7rem;
  background:
    radial-gradient(circle at 70% 30%, #c47a3a 0 28%, transparent 29%),
    linear-gradient(145deg, #1f6f78, #0f2a33);
  box-shadow: 0 8px 20px rgba(31, 111, 120, 0.28);
}

.brand-text {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.brand-text strong {
  font-family: 'Fraunces', serif;
  font-size: 1.15rem;
  letter-spacing: 0.01em;
}

.brand-text span {
  color: var(--muted);
  font-size: 0.82rem;
}

.status {
  margin-left: auto;
  font-size: 0.78rem;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.35rem 0.75rem;
  background: rgba(255, 255, 255, 0.45);
}

.status[data-online='true'] {
  color: var(--high);
  border-color: rgba(31, 122, 77, 0.25);
}

.paper-test-button {
  border: 1px solid rgba(31, 111, 120, 0.3);
  border-radius: 999px;
  padding: 0.45rem 0.85rem;
  color: #1f6f78;
  background: rgba(255, 255, 255, 0.58);
  font-weight: 800;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.2s ease;
}

.paper-test-button:hover {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.9);
}

.error {
  margin: 1rem 0 0;
  color: #9b2c2c;
  background: rgba(155, 44, 44, 0.08);
  border: 1px solid rgba(155, 44, 44, 0.18);
  border-radius: 12px;
  padding: 0.8rem 1rem;
}

.modal-mask {
  position: fixed;
  inset: 0;
  z-index: 50;
  background: rgba(15, 42, 51, 0.45);
  display: grid;
  place-items: center;
  padding: 1rem;
}

.modal {
  width: min(420px, 100%);
  background: #fffaf2;
  border-radius: 16px;
  padding: 1.25rem 1.3rem 1.1rem;
  box-shadow: var(--shadow);
  border: 1px solid var(--line);
}

.modal h3 {
  margin: 0;
  font-family: 'Fraunces', serif;
}

.modal p {
  margin: 0.75rem 0 1.1rem;
  color: var(--ink-soft);
  line-height: 1.55;
}

.modal button {
  border: none;
  border-radius: 999px;
  padding: 0.7rem 1.2rem;
  background: linear-gradient(135deg, #d08a45, #c47a3a);
  color: #fff8ef;
  font-weight: 700;
  cursor: pointer;
}

.workspace {
  margin-top: 1.75rem;
  animation: rise 0.55s ease both;
}

.workspace-head {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 1rem;
  margin-bottom: 1rem;
}

.workspace-head h2 {
  margin: 0;
  font-family: 'Fraunces', serif;
  font-size: 1.6rem;
}

.workspace-head p {
  margin: 0.35rem 0 0;
  color: var(--muted);
}

.meta-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.meta-pills span {
  font-size: 0.75rem;
  color: var(--ink-soft);
  background: rgba(255, 255, 255, 0.55);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.3rem 0.65rem;
}

.meta-pills .live {
  color: #fff;
  background: linear-gradient(135deg, #c47a3a, #a85f28);
  border-color: transparent;
  animation: pulse 1.2s ease-in-out infinite;
}

.grid {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 1rem;
}

.grid > :first-child,
.grid > :nth-child(n + 3) {
  grid-column: 1 / -1;
}

.footer {
  margin-top: 2.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.5rem;
  color: var(--muted);
  font-size: 0.8rem;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.65; }
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }

  .workspace-head {
    flex-direction: column;
    align-items: start;
  }
}
</style>
