<script setup>
defineProps({
  loading: { type: Boolean, default: false },
  compact: { type: Boolean, default: false },
})

const query = defineModel({ type: String, default: '' })
const emit = defineEmits(['submit'])

function onSubmit() {
  emit('submit')
}
</script>

<template>
  <section class="hero" :class="{ 'hero--compact': compact }">
    <div class="hero-copy">
      <p class="eyebrow">Complex Scholarly Query · Multi-strategy Retrieval</p>
      <h1>ScholarAgent</h1>
      <p class="lead">
        面向复杂学术查询的智能论文搜索与推荐：查询理解、子问题分解、迭代检索与结构化归纳。
      </p>
    </div>

    <form class="search" @submit.prevent="onSubmit">
      <label class="sr-only" for="query">学术查询</label>
      <textarea
        id="query"
        v-model="query"
        rows="3"
        placeholder="描述你的研究问题，例如：LLM Agent 如何做查询分解与引文网络探索以提升论文召回？"
      />
      <div class="actions">
        <p>流程：意图识别 → 自适应多源检索 → 覆盖分析与查询演化 → BGE-M3 + RRF → Reranker。</p>
        <button type="submit" :disabled="loading || !query.trim()">
          {{ loading ? '检索中…' : '开始智能检索' }}
        </button>
      </div>
    </form>

    <div class="hero-visual" aria-hidden="true">
      <div class="orbit orbit-a"></div>
      <div class="orbit orbit-b"></div>
      <div class="node n1">Query</div>
      <div class="node n2">Decompose</div>
      <div class="node n3">Retrieve</div>
      <div class="node n4">Rank</div>
    </div>
  </section>
</template>

<style scoped>
.hero {
  position: relative;
  min-height: calc(100vh - 7rem);
  display: grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 1.5rem;
  align-items: center;
  overflow: hidden;
  border-radius: 28px;
  padding: 2.2rem 2rem;
  background:
    linear-gradient(135deg, rgba(15, 42, 51, 0.92), rgba(31, 111, 120, 0.88)),
    url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  color: #f4f8f7;
  box-shadow: var(--shadow);
  transition: min-height 0.45s ease, padding 0.45s ease;
}

.hero--compact {
  min-height: auto;
  grid-template-columns: 1fr;
  padding: 1.4rem 1.35rem;
}

.hero--compact .hero-visual,
.hero--compact .eyebrow,
.hero--compact .lead {
  display: none;
}

.hero--compact h1 {
  font-size: 1.7rem;
  margin-bottom: 0.8rem;
}

.hero-copy {
  position: relative;
  z-index: 2;
  max-width: 36rem;
}

.eyebrow {
  margin: 0 0 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: rgba(244, 248, 247, 0.72);
}

h1 {
  margin: 0;
  font-family: 'Fraunces', serif;
  font-size: clamp(2.6rem, 6vw, 4.2rem);
  line-height: 1.05;
  font-weight: 700;
}

.lead {
  margin: 1rem 0 0;
  font-size: 1.05rem;
  line-height: 1.65;
  color: rgba(244, 248, 247, 0.86);
  max-width: 32rem;
}

.search {
  position: relative;
  z-index: 2;
  grid-column: 1 / -1;
  display: grid;
  gap: 0.85rem;
}

textarea {
  width: 100%;
  resize: vertical;
  min-height: 96px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  background: rgba(247, 243, 234, 0.95);
  color: var(--ink);
  outline: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}

textarea:focus {
  border-color: rgba(196, 122, 58, 0.65);
  box-shadow: 0 0 0 3px rgba(196, 122, 58, 0.22);
}

.actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.actions p {
  margin: 0;
  font-size: 0.82rem;
  color: rgba(244, 248, 247, 0.7);
}

button {
  border: none;
  border-radius: 999px;
  padding: 0.85rem 1.4rem;
  background: linear-gradient(135deg, #d08a45, #c47a3a);
  color: #fff8ef;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
  box-shadow: 0 10px 24px rgba(196, 122, 58, 0.35);
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
}

button:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.hero-visual {
  position: absolute;
  right: 4%;
  top: 12%;
  width: min(340px, 36vw);
  height: 280px;
  pointer-events: none;
}

.orbit {
  position: absolute;
  inset: 20px;
  border: 1px dashed rgba(255, 255, 255, 0.22);
  border-radius: 50%;
  animation: spin 18s linear infinite;
}

.orbit-b {
  inset: 55px;
  animation-duration: 12s;
  animation-direction: reverse;
}

.node {
  position: absolute;
  padding: 0.4rem 0.7rem;
  border-radius: 999px;
  background: rgba(247, 243, 234, 0.14);
  border: 1px solid rgba(255, 255, 255, 0.22);
  backdrop-filter: blur(6px);
  font-size: 0.75rem;
  letter-spacing: 0.02em;
}

.n1 { left: 38%; top: 8%; animation: float 4s ease-in-out infinite; }
.n2 { right: 8%; top: 36%; animation: float 4.8s ease-in-out infinite 0.4s; }
.n3 { left: 18%; bottom: 22%; animation: float 5.2s ease-in-out infinite 0.8s; }
.n4 { right: 22%; bottom: 8%; animation: float 4.4s ease-in-out infinite 0.2s; }

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
    min-height: auto;
  }

  .hero-visual {
    display: none;
  }
}
</style>
