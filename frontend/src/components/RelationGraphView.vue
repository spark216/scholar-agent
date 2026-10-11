<script setup>
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { GraphChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import VChart from 'vue-echarts'

use([CanvasRenderer, GraphChart, TooltipComponent, LegendComponent])

const props = defineProps({
  graph: {
    type: Object,
    default: () => ({ nodes: [], edges: [] }),
  },
})

const categoryMap = {
  query: '查询',
  subquery: '子查询',
  method: '方法',
  dataset: '数据集',
  topic: '主题',
  entity: '实体',
  venue: '来源',
  paper: '论文',
}

const option = computed(() => {
  const categories = ['查询', '子查询', '方法', '数据集', '主题', '实体', '来源', '论文']
  const categoryIndex = Object.fromEntries(categories.map((name, idx) => [name, idx]))

  const nodes = (props.graph?.nodes || []).map((node) => {
    const categoryName = categoryMap[node.category] || node.category
    return {
      id: node.id,
      name: node.name,
      value: node.value,
      category: categoryIndex[categoryName] ?? categoryIndex['论文'],
      symbolSize: Math.max(28, Math.min(64, (node.value || 1) * 3.2)),
      paperId: node.paper_id,
    }
  })

  const links = (props.graph?.edges || []).map((edge) => ({
    source: edge.source,
    target: edge.target,
    value: edge.weight,
    label: {
      show: true,
      formatter: edge.relation,
      fontSize: 10,
      color: '#5b737a',
    },
    lineStyle: {
      width: Math.max(1, (edge.weight || 1) * 1.8),
      curveness: 0.18,
      color: 'rgba(31, 111, 120, 0.35)',
    },
  }))

  return {
    tooltip: {
      formatter: (params) => {
        if (params.dataType === 'edge') {
          return `${params.data.source} → ${params.data.target}`
        }
        return `<strong>${params.name}</strong><br/>权重 ${params.value ?? '-'}`
      },
    },
    legend: [
      {
        data: categories,
        bottom: 0,
        textStyle: { color: '#5b737a' },
      },
    ],
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        categories: categories.map((name) => ({ name })),
        data: nodes,
        links,
        label: {
          show: true,
          position: 'bottom',
          fontSize: 11,
          color: '#17343c',
        },
        force: {
          repulsion: 220,
          edgeLength: [70, 150],
          gravity: 0.08,
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 3 },
        },
        color: [
          '#c47a3a', '#2a9aa4', '#1f6f78', '#1f7a4d',
          '#6d7f31', '#8b5f8d', '#97733b', '#0f2a33',
        ],
      },
    ],
  }
})
</script>

<template>
  <section class="panel">
    <div class="head">
      <h3>关系图</h3>
      <p>查询 · 子问题 · 方法 · 数据集 · 主题 · 实体 · 来源 · 论文之间的关联结构</p>
    </div>
    <VChart class="chart" :option="option" autoresize />
  </section>
</template>

<style scoped>
.panel {
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1.15rem 1.2rem 0.5rem;
  backdrop-filter: blur(8px);
}

.head {
  margin-bottom: 0.4rem;
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

.chart {
  width: 100%;
  height: 460px;
}
</style>
