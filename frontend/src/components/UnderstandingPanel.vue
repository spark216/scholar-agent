<script setup>
defineProps({
  understanding: {
    type: Object,
    required: true,
  },
})
</script>

<template>
  <article class="panel">
    <h3>查询意图识别</h3>
    <p class="intent">{{ understanding.intent }}</p>

    <div class="kv">
      <div><span>领域</span><strong>{{ understanding.domain || '未识别' }}</strong></div>
      <div><span>时间范围</span><strong>{{ understanding.time_range || '未指定' }}</strong></div>
      <div><span>语言</span><strong>{{ understanding.language || 'zh+en' }}</strong></div>
    </div>

    <div class="block">
      <h4>主题 / 关键词</h4>
      <div class="tags">
        <span v-for="item in understanding.topics || []" :key="`t-${item}`">{{ item }}</span>
        <span v-for="item in understanding.keywords || []" :key="`k-${item}`" class="method">{{ item }}</span>
      </div>
    </div>

    <div class="block">
      <h4>关键实体</h4>
      <div class="tags">
        <span v-for="item in understanding.entities" :key="item">{{ item }}</span>
      </div>
    </div>

    <div class="block">
      <h4>方法 / 数据集 / Venue</h4>
      <div class="tags">
        <span v-for="item in understanding.methods" :key="`m-${item}`" class="method">{{ item }}</span>
        <span v-for="item in understanding.datasets" :key="`d-${item}`" class="dataset">{{ item }}</span>
        <span v-for="item in understanding.venues || []" :key="`v-${item}`">{{ item }}</span>
      </div>
    </div>

    <div class="block">
      <h4>子查询分解</h4>
      <ol>
        <li v-for="(q, idx) in understanding.sub_queries" :key="idx">{{ q }}</li>
      </ol>
    </div>

    <div class="block">
      <h4>改写检索词</h4>
      <ol>
        <li v-for="(q, idx) in understanding.rewritten_queries || []" :key="`r-${idx}`">{{ q }}</li>
      </ol>
    </div>

    <details v-if="understanding.raw_json && Object.keys(understanding.raw_json).length" class="raw">
      <summary>查看小模型原始 JSON</summary>
      <pre>{{ JSON.stringify(understanding.raw_json, null, 2) }}</pre>
    </details>
  </article>
</template>

<style scoped>
.panel {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem;
  backdrop-filter: blur(8px);
}

h3 {
  margin: 0;
  font-family: 'Fraunces', serif;
  font-size: 1.2rem;
}

.intent {
  margin: 0.7rem 0 1rem;
  color: var(--ink-soft);
  line-height: 1.55;
}

.kv {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.55rem;
  margin-bottom: 0.9rem;
}

.kv div {
  background: rgba(31, 111, 120, 0.06);
  border-radius: 12px;
  padding: 0.55rem 0.65rem;
}

.kv span {
  display: block;
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.kv strong {
  display: block;
  margin-top: 0.2rem;
  font-size: 0.9rem;
}

.block + .block {
  margin-top: 0.9rem;
}

h4 {
  margin: 0 0 0.45rem;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.tags span {
  font-size: 0.78rem;
  padding: 0.28rem 0.55rem;
  border-radius: 999px;
  background: rgba(31, 111, 120, 0.1);
  color: var(--sea);
}

.tags .method {
  background: rgba(196, 122, 58, 0.12);
  color: var(--copper);
}

.tags .dataset {
  background: rgba(31, 122, 77, 0.1);
  color: var(--high);
}

ol {
  margin: 0;
  padding-left: 1.1rem;
  color: var(--text);
  line-height: 1.55;
  font-size: 0.9rem;
}

.raw {
  margin-top: 1rem;
  border-top: 1px dashed var(--line);
  padding-top: 0.75rem;
}

.raw summary {
  cursor: pointer;
  color: var(--sea);
  font-size: 0.85rem;
}

.raw pre {
  margin: 0.6rem 0 0;
  max-height: 220px;
  overflow: auto;
  background: rgba(15, 42, 51, 0.06);
  border-radius: 10px;
  padding: 0.75rem;
  font-size: 0.75rem;
  line-height: 1.45;
}

@media (max-width: 700px) {
  .kv {
    grid-template-columns: 1fr;
  }
}
</style>
