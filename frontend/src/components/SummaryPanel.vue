<script setup>
defineProps({
  summary: {
    type: Object,
    required: true,
  },
})
</script>

<template>
  <article class="panel">
    <h3>结果归纳</h3>
    <p>{{ summary.overview }}</p>

    <div class="cols">
      <div>
        <h4>关键发现</h4>
        <ul>
          <li v-for="(item, idx) in summary.key_findings" :key="idx">{{ item }}</li>
        </ul>
      </div>
      <div>
        <h4>研究缺口</h4>
        <ul>
          <li v-for="(item, idx) in summary.research_gaps" :key="idx">{{ item }}</li>
        </ul>
      </div>
    </div>
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
  margin: 0 0 0.7rem;
  font-family: 'Fraunces', serif;
  font-size: 1.2rem;
}

p {
  margin: 0;
  line-height: 1.6;
  color: var(--ink-soft);
}

.cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.9rem;
  margin-top: 1rem;
}

h4 {
  margin: 0 0 0.4rem;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}

ul {
  margin: 0;
  padding-left: 1.05rem;
  font-size: 0.9rem;
  line-height: 1.55;
}

@media (max-width: 700px) {
  .cols {
    grid-template-columns: 1fr;
  }
}
</style>
