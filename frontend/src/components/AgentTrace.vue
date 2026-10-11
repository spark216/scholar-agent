<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  trace: {
    type: Array,
    default: () => [],
  },
})

const open = ref(true)

const phaseLabel = {
  think: '思考',
  plan: '计划',
  tool: '工具调用',
  observe: '观察',
  rank: '排序',
  persist: '落库',
}

const steps = computed(() => props.trace || [])

const hasMetric = (value) => value !== null && value !== undefined

const intentUsage = computed(() =>
  [...steps.value]
    .reverse()
    .find((step) => step.step === 'understand' && step.status === 'done'),
)

const roundUsage = computed(() =>
  steps.value
    .filter((step) => String(step.step || '').startsWith('agent:retrieve:'))
    .map((step) => ({
      iteration: step.iteration,
      apiCalls: step.api_calls ?? 0,
      tokenUsage: step.token_usage ?? 0,
      cumulativeApiCalls: step.cumulative_api_calls ?? 0,
      cumulativeTokenUsage: step.cumulative_token_usage ?? 0,
    })),
)
</script>

<template>
  <article class="panel">
    <button class="head" type="button" @click="open = !open">
      <div>
        <h3>Agent 执行流程</h3>
        <p>思考 / 计划 / 工具调用 / 观察 / 排序 / 落库（可折叠）</p>
      </div>
      <span class="toggle">{{ open ? '收起' : '展开' }}</span>
    </button>

    <div v-if="open" class="body">
      <section v-if="intentUsage || roundUsage.length" class="usage-panel">
        <div class="usage-title">
          <strong>API 与 Token 消耗</strong>
          <span>Token 仅统计 LLM API，本地 BGE 编码不计入</span>
        </div>
        <div class="usage-grid">
          <div v-if="intentUsage" class="usage-card">
            <strong>意图识别</strong>
            <span>API {{ intentUsage.api_calls ?? 0 }}</span>
            <span>Token {{ intentUsage.token_usage ?? 0 }}</span>
          </div>
          <div
            v-for="round in roundUsage"
            :key="`usage-${round.iteration}`"
            class="usage-card"
          >
            <strong>第 {{ round.iteration || '-' }} 轮检索</strong>
            <span>API {{ round.apiCalls }}</span>
            <span>Token {{ round.tokenUsage }}</span>
            <small>
              检索累计 API {{ round.cumulativeApiCalls }} ·
              Token {{ round.cumulativeTokenUsage }}
            </small>
          </div>
        </div>
      </section>

      <details
        v-for="(step, idx) in steps"
        :key="idx"
        class="step"
        :data-phase="step.phase || 'tool'"
        :open="step.status === 'running' || idx < 2"
      >
        <summary>
          <span class="badge">{{ phaseLabel[step.phase] || step.phase || '步骤' }}</span>
          <strong>{{ step.step }}</strong>
          <em :data-status="step.status">{{ step.status || 'done' }}</em>
        </summary>
        <p class="detail">{{ step.detail }}</p>
        <div
          v-if="
            hasMetric(step.api_calls) ||
            hasMetric(step.token_usage) ||
            hasMetric(step.cumulative_api_calls) ||
            hasMetric(step.cumulative_token_usage)
          "
          class="metrics"
        >
          <span v-if="hasMetric(step.api_calls)">本步 API {{ step.api_calls }}</span>
          <span v-if="hasMetric(step.token_usage)">本步 Token {{ step.token_usage }}</span>
          <span v-if="hasMetric(step.cumulative_api_calls)">
            累计 API {{ step.cumulative_api_calls }}
          </span>
          <span v-if="hasMetric(step.cumulative_token_usage)">
            累计 Token {{ step.cumulative_token_usage }}
          </span>
        </div>
        <div v-if="step.tool" class="kv"><span>工具</span><code>{{ step.tool }}</code></div>
        <div v-if="step.input" class="kv"><span>输入</span><pre>{{ step.input }}</pre></div>
        <div v-if="step.output" class="kv"><span>输出</span><pre>{{ step.output }}</pre></div>
      </details>
    </div>
  </article>
</template>

<style scoped>
.panel {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1rem 1.15rem;
  backdrop-filter: blur(8px);
  grid-column: 1 / -1;
}

.head {
  width: 100%;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  border: none;
  background: transparent;
  padding: 0;
  cursor: pointer;
  text-align: left;
  color: inherit;
}

h3 {
  margin: 0;
  font-family: 'Fraunces', serif;
  font-size: 1.2rem;
}

.head p {
  margin: 0.3rem 0 0;
  color: var(--muted);
  font-size: 0.85rem;
}

.toggle {
  font-size: 0.82rem;
  color: var(--sea);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.3rem 0.7rem;
  background: rgba(255, 255, 255, 0.7);
}

.body {
  margin-top: 0.9rem;
  display: grid;
  gap: 0.55rem;
}

.usage-panel {
  border: 1px solid rgba(31, 111, 120, 0.18);
  border-radius: 12px;
  padding: 0.75rem;
  background: rgba(31, 111, 120, 0.05);
}

.usage-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.6rem;
}

.usage-title > span {
  color: var(--muted);
  font-size: 0.74rem;
  text-align: right;
}

.usage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.5rem;
}

.usage-card {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.25rem 0.55rem;
  border-radius: 9px;
  padding: 0.55rem 0.65rem;
  background: rgba(255, 255, 255, 0.72);
  font-size: 0.78rem;
}

.usage-card strong,
.usage-card small {
  grid-column: 1 / -1;
}

.usage-card small {
  color: var(--muted);
  font-size: 0.69rem;
}

.step {
  border: 1px solid rgba(15, 42, 51, 0.08);
  border-radius: 12px;
  background: rgba(247, 243, 234, 0.45);
  padding: 0.55rem 0.75rem;
}

.step[data-phase='think'] { border-left: 3px solid #c47a3a; }
.step[data-phase='plan'] { border-left: 3px solid #2a9aa4; }
.step[data-phase='tool'] { border-left: 3px solid #1f6f78; }
.step[data-phase='observe'] { border-left: 3px solid #1f7a4d; }
.step[data-phase='rank'] { border-left: 3px solid #9a6b2f; }
.step[data-phase='persist'] { border-left: 3px solid #5b737a; }

summary {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  cursor: pointer;
  list-style: none;
}

summary::-webkit-details-marker {
  display: none;
}

.badge {
  font-size: 0.7rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--sea);
  background: rgba(31, 111, 120, 0.1);
  border-radius: 999px;
  padding: 0.18rem 0.45rem;
}

strong {
  font-size: 0.9rem;
}

em {
  margin-left: auto;
  font-style: normal;
  font-size: 0.72rem;
  color: var(--muted);
}

em[data-status='error'] { color: #9b2c2c; }
em[data-status='skipped'] { color: #9a6b2f; }
em[data-status='running'] {
  color: #c47a3a;
  animation: blink 1s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.detail {
  margin: 0.55rem 0 0.35rem;
  color: var(--ink-soft);
  line-height: 1.5;
  font-size: 0.9rem;
}

.metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.4rem 0;
}

.metrics span {
  border-radius: 999px;
  padding: 0.2rem 0.5rem;
  color: var(--sea);
  background: rgba(31, 111, 120, 0.09);
  font-size: 0.72rem;
}

.kv {
  margin-top: 0.35rem;
}

.kv span {
  display: block;
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 0.2rem;
}

.kv code,
.kv pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  background: rgba(15, 42, 51, 0.05);
  border-radius: 8px;
  padding: 0.45rem 0.55rem;
  font-size: 0.78rem;
  line-height: 1.45;
}
</style>
