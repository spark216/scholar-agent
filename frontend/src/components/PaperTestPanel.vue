<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import {
  fetchLatestPaperTest,
  fetchPaperTestDatasets,
  fetchPaperTestHistory,
  fetchPaperTestHistoryDetail,
  runPaperTestStream,
} from '../api/search'

const emit = defineEmits(['close'])

const datasets = ref([])
const selectedDataset = ref('pasa-real')
const sampleLimit = ref(3)
const selectedK = ref(10)
const providerCacheMode = ref('record')
const loading = ref(false)
const loadingData = ref(true)
const error = ref('')
const report = ref(null)
const progress = ref({ current: 0, total: 0, query: '' })
const partialQueries = ref([])
const history = ref([])
const historyTotal = ref(0)
const historyLoading = ref(false)
const historyPageSize = 20
const stageLabels = {
  provider: '来源召回',
  merged: '合并候选',
  budget: '候选预算',
  ranking: '排序候选',
  final: '最终结果',
}

const availableDatasets = computed(() => datasets.value.filter((item) => item.available))
const selectedDatasetInfo = computed(
  () => datasets.value.find((item) => item.id === selectedDataset.value) || null,
)
const sampleMaximum = computed(() => Number(selectedDatasetInfo.value?.sample_count) || 0)
const sampleLimitValid = computed(() => {
  const value = Number(sampleLimit.value)
  return Number.isInteger(value)
    && value >= 1
    && (!sampleMaximum.value || value <= sampleMaximum.value)
})
const selectedKValid = computed(() => {
  const value = Number(selectedK.value)
  return Number.isInteger(value) && value >= 1 && value <= 50
})
const controlsValid = computed(() => sampleLimitValid.value && selectedKValid.value)
const metrics = computed(() => {
  if (!report.value?.aggregate) return []
  return Object.values(report.value.aggregate).sort((a, b) => a.k - b.k)
})
const successfulMetrics = computed(() => {
  if (!report.value?.aggregate_successful) return []
  return Object.values(report.value.aggregate_successful).sort((a, b) => a.k - b.k)
})
const displayedQueries = computed(() => report.value?.queries || partialQueries.value)
const progressPercent = computed(() => {
  if (!progress.value.total) return 0
  return Math.round((progress.value.current / progress.value.total) * 100)
})

function percent(value) {
  return `${((value || 0) * 100).toFixed(2)}%`
}

function seconds(value) {
  return `${((value || 0) / 1000).toFixed(2)} s`
}

function optionalSeconds(value) {
  return value == null ? '—' : seconds(value)
}

function firstSeenLabel(item) {
  if (item.first_seen_stage) return stageLabels[item.first_seen_stage] || item.first_seen_stage
  if (Object.keys(item.stage_metrics || {}).length) return '未召回'
  return '旧报告无记录'
}

function stageDetail(item) {
  const provider = item.stage_metrics?.provider
  const final = item.stage_metrics?.final
  const ranks = item.stage_metrics?.ranking?.matched_ranks || []
  const parts = []
  if (provider || final) {
    parts.push(`来源 ${provider?.gold_hits || 0} → 最终 ${final?.gold_hits || 0}`)
  }
  if (ranks.length) parts.push(`完整排序 #${ranks.join('、#')}`)
  if (item.loss_stage) parts.push(`损失于${stageLabels[item.loss_stage] || item.loss_stage}`)
  return parts.join('；')
}

function providerLabel(item) {
  const groups = new Map()
  for (const diagnostic of item.provider_diagnostics || []) {
    const key = `R${diagnostic.iteration} ${diagnostic.provider}`
    const current = groups.get(key) || {
      returned: 0,
      errors: [],
      circuitOpen: false,
      cacheHits: 0,
      skipped: [],
    }
    current.returned += diagnostic.returned_count || 0
    if (diagnostic.error) current.errors.push(diagnostic.error)
    current.circuitOpen ||= Boolean(diagnostic.circuit_open)
    current.cacheHits += diagnostic.cache_hit ? 1 : 0
    if (diagnostic.skipped) current.skipped.push(diagnostic.skipped)
    groups.set(key, current)
  }
  return [...groups.entries()].map(([name, value]) => (
    `${name} ${value.circuitOpen
      ? '暂时熔断'
      : value.errors.length
        ? `失败(${value.errors.join(',')})`
        : value.skipped.length
          ? '按覆盖情况跳过'
          : `${value.returned}篇${value.cacheHits ? '（缓存）' : ''}`}`
  ))
}

function datasetCount(dataset) {
  if (!dataset.available) return '未找到'
  return dataset.sample_count ? `${dataset.sample_count} 条` : '运行时加载'
}

function formatDate(value) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(new Date(value))
}

function historyMetric(item) {
  const k = item.max_k || Math.max(...(item.ks || [5]))
  return item.aggregate?.[String(k)] || null
}

function comparablePrevious(item, index) {
  return history.value.slice(index + 1).find((previous) => (
    previous.dataset_id === item.dataset_id
    && previous.sample_count === item.sample_count
    && previous.seed === item.seed
    && previous.dataset_sha256 === item.dataset_sha256
    && previous.config_fingerprint === item.config_fingerprint
    && Boolean(item.provider_snapshot_sha256)
    && previous.provider_snapshot_sha256 === item.provider_snapshot_sha256
    && (previous.max_k || Math.max(...(previous.ks || [5])))
      === (item.max_k || Math.max(...(item.ks || [5])))
  )) || null
}

function shortHash(value, length = 12) {
  return value ? value.slice(0, length) : '—'
}

function historyDelta(item, index, metricName) {
  const previous = comparablePrevious(item, index)
  const currentValue = historyMetric(item)?.[metricName]
  const previousValue = previous ? historyMetric(previous)?.[metricName] : null
  if (currentValue == null || previousValue == null) return null
  return currentValue - previousValue
}

function signedPercent(value) {
  if (value == null) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${(value * 100).toFixed(2)}pp`
}

function applyReport(value) {
  if (!value) return
  report.value = value
  selectedDataset.value = value.dataset.id
  sampleLimit.value = value.sample_limit
  selectedK.value = Math.max(...value.ks)
  providerCacheMode.value = value.run_metadata?.provider_cache_mode || 'record'
}

function useFullDataset() {
  if (sampleMaximum.value) sampleLimit.value = sampleMaximum.value
}

async function refreshHistory(reset = true) {
  if (historyLoading.value) return
  historyLoading.value = true
  try {
    const offset = reset ? 0 : history.value.length
    const data = await fetchPaperTestHistory({ limit: historyPageSize, offset })
    history.value = reset ? data.items : [...history.value, ...data.items]
    historyTotal.value = data.total
  } catch (err) {
    error.value = error.value || err?.message || '无法加载评测历史'
  } finally {
    historyLoading.value = false
  }
}

async function viewHistory(runId) {
  if (loading.value) return
  error.value = ''
  try {
    applyReport(await fetchPaperTestHistoryDetail(runId))
  } catch (err) {
    error.value = err?.message || '无法加载历史报告'
  }
}

async function runTest() {
  if (loading.value || !selectedDataset.value) return
  if (!controlsValid.value) {
    error.value = !sampleLimitValid.value
      ? `抽样数量请输入 1–${sampleMaximum.value || '测试集总量'} 的整数`
      : '最大 K 请输入 1–50 的整数'
    return
  }
  loading.value = true
  error.value = ''
  report.value = null
  partialQueries.value = []
  progress.value = { current: 0, total: sampleLimit.value, query: '' }
  try {
    const completedReport = await runPaperTestStream(
      {
        dataset: selectedDataset.value,
        sampleLimit: Number(sampleLimit.value),
        ks: Number(selectedK.value) > 5
          ? [5, Number(selectedK.value)]
          : [Number(selectedK.value)],
        seed: 42,
        useMock: false,
        providerCacheMode: providerCacheMode.value,
      },
      {
        onStart: (data) => {
          progress.value = { current: 0, total: data.total, query: '' }
        },
        onProgress: (data) => {
          progress.value = {
            current: data.current,
            total: data.total,
            query: data.query,
          }
        },
        onQueryResult: (data) => {
          partialQueries.value = [...partialQueries.value, data.query_result]
        },
        onError: (data) => {
          error.value = data?.detail || '评测失败'
        },
      },
    )
    applyReport(completedReport)
    await refreshHistory(true)
  } catch (err) {
    error.value = error.value || err?.message || '评测失败'
  } finally {
    loading.value = false
  }
}

watch(selectedDataset, () => {
  if (sampleMaximum.value && Number(sampleLimit.value) > sampleMaximum.value) {
    sampleLimit.value = sampleMaximum.value
  }
  error.value = ''
})

onMounted(async () => {
  try {
    const [datasetValues, latest] = await Promise.all([
      fetchPaperTestDatasets(),
      fetchLatestPaperTest(),
    ])
    datasets.value = datasetValues
    if (latest) {
      applyReport(latest)
    } else if (availableDatasets.value.length) {
      selectedDataset.value = availableDatasets.value[0].id
    }
    await refreshHistory(true)
  } catch (err) {
    error.value = err?.message || '无法加载评测配置'
  } finally {
    loadingData.value = false
  }
})
</script>

<template>
  <div class="paper-test-mask" @click.self="emit('close')">
    <section class="paper-test-panel" role="dialog" aria-modal="true" aria-label="论文检索评测">
      <header class="panel-head">
        <div>
          <span class="eyebrow">PAPER FINDING BENCHMARKS</span>
          <h2>ScholarAgent 检索效果测试</h2>
          <p>优先使用 arXiv、PMID、DOI 和 Corpus ID 精确匹配，规范化标题仅作后备。</p>
        </div>
        <button class="close" type="button" aria-label="关闭" @click="emit('close')">×</button>
      </header>

      <div class="controls">
        <label>
          测试集
          <select v-model="selectedDataset" :disabled="loading || loadingData">
            <option
              v-for="dataset in datasets"
              :key="dataset.id"
              :value="dataset.id"
              :disabled="!dataset.available"
            >
              {{ dataset.name }}（{{ datasetCount(dataset) }}）
            </option>
          </select>
        </label>
        <label>
          抽样数量
          <div class="number-control">
            <input
              v-model.number="sampleLimit"
              type="number"
              min="1"
              :max="sampleMaximum || undefined"
              step="1"
              inputmode="numeric"
              :disabled="loading"
              aria-label="自定义抽样数量"
            >
            <button
              type="button"
              :disabled="loading || !sampleMaximum"
              @click="useFullDataset"
            >全量</button>
          </div>
          <small :class="{ invalid: !sampleLimitValid }">
            最大 {{ sampleMaximum || '加载后确定' }}，支持自定义
          </small>
        </label>
        <label>
          最大 K
          <input
            v-model.number="selectedK"
            type="number"
            min="1"
            max="50"
            step="1"
            inputmode="numeric"
            :disabled="loading"
            aria-label="自定义最大 K"
          >
          <small :class="{ invalid: !selectedKValid }">范围 1–50，支持自定义</small>
        </label>
        <label>
          Provider 数据
          <select v-model="providerCacheMode" :disabled="loading">
            <option value="record">建立/更新快照</option>
            <option value="frozen">冻结快照复测</option>
          </select>
          <small>
            {{ providerCacheMode === 'frozen'
              ? '仅用已有缓存；缺项将停止，不会联网降级'
              : '允许联网并保存本次 Provider 返回' }}
          </small>
        </label>
        <button
          class="run"
          type="button"
          :disabled="loading || loadingData || !availableDatasets.length || !controlsValid"
          @click="runTest"
        >
          {{ loading ? '评测运行中…' : report ? '重新运行' : '开始评测' }}
        </button>
      </div>

      <p v-if="selectedDatasetInfo?.evaluation_family === 'astabench'" class="asta-notice">
        AstaBench 将从固定 revision 加载官方分割，先把 DOI/arXiv 批量解析为 Semantic
        Scholar Corpus ID，再按查询类型执行精确匹配或相关性 Judge。首次运行需要网络和
        Hugging Face 权限。
      </p>
      <p v-else-if="selectedDatasetInfo?.evaluation_family === 'trec-pm'" class="asta-notice">
        TREC Precision Medicine 使用 PMID 精确匹配，适合检查 PubMed 医学论文召回。
        金标准论文数量通常较多，请在相同 K 下比较不同运行。
      </p>
      <p v-else-if="selectedDatasetInfo?.evaluation_family === 'beir-open'" class="asta-notice">
        当前以开放学术 Provider 检索并按规范化标题匹配，属于跨来源泛化评测，不是
        BEIR 封闭语料官方分数。TREC-COVID 等多答案任务建议重点观察 nDCG 与同配置增量。
      </p>

      <div v-if="loading" class="progress-card">
        <div class="progress-line">
          <span>正在测试 {{ progress.current }}/{{ progress.total }}</span>
          <strong>{{ progressPercent }}%</strong>
        </div>
        <div class="track"><span :style="{ width: `${progressPercent}%` }"></span></div>
        <p>{{ progress.query || '正在准备测试集…' }}</p>
        <small>每个样本只检索一次，再同时计算多个 K；完整检索链路可能需要数分钟。</small>
      </div>

      <p v-if="error" class="panel-error">{{ error }}</p>
      <p v-if="loadingData" class="empty">正在检查本地论文检索测试集…</p>
      <p v-else-if="!availableDatasets.length" class="empty">
        未找到本地测试数据。请设置 PAPER_TEST_DATASET_ROOT 或把数据放入 backend/paper-test/datasets。
      </p>

      <section class="history-section">
        <div class="history-head">
          <div>
            <span class="eyebrow">LOCAL EVALUATION HISTORY</span>
            <h3>历史测试记录</h3>
          </div>
          <span>本地共 {{ historyTotal }} 次</span>
        </div>
        <p v-if="!history.length && !historyLoading" class="empty history-empty">
          暂无历史记录，完成第一次评测后会自动保存在本地。
        </p>
        <div v-else class="table-wrap history-table">
          <table>
            <thead>
              <tr>
                <th>运行时间</th>
                <th>测试集</th>
                <th>样本</th>
                <th>最大 K</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ΔF1</th>
                <th>nDCG</th>
                <th>MRR</th>
                <th>命中/Gold</th>
                <th>来源召回</th>
                <th>Provider 快照</th>
                <th>平均耗时</th>
                <th>P95 耗时</th>
                <th>API/Token</th>
                <th>Asta Adjusted F1</th>
                <th>成功/失败</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(item, index) in history"
                :key="item.run_id"
                :class="{ selected: report?.run_id === item.run_id }"
              >
                <td>{{ formatDate(item.generated_at) }}</td>
                <td>{{ item.dataset_name }}</td>
                <td>{{ item.sample_count }}</td>
                <td>@{{ item.max_k || Math.max(...(item.ks || [5])) }}</td>
                <td>{{ percent(historyMetric(item)?.precision) }}</td>
                <td>{{ percent(historyMetric(item)?.recall) }}</td>
                <td><strong>{{ percent(historyMetric(item)?.f1) }}</strong></td>
                <td
                  class="metric-delta"
                  :class="{
                    positive: historyDelta(item, index, 'f1') > 0,
                    negative: historyDelta(item, index, 'f1') < 0,
                  }"
                >
                  {{ signedPercent(historyDelta(item, index, 'f1')) }}
                </td>
                <td>{{ percent(historyMetric(item)?.ndcg) }}</td>
                <td>{{ percent(historyMetric(item)?.mrr) }}</td>
                <td>{{ historyMetric(item)?.hits ?? '—' }}/{{ item.total_relevant || '—' }}</td>
                <td>{{ item.stage_recall?.provider == null ? '—' : percent(item.stage_recall.provider) }}</td>
                <td :title="item.provider_snapshot_sha256">{{ shortHash(item.provider_snapshot_sha256, 8) }}</td>
                <td>{{ seconds(item.mean_latency_ms) }}</td>
                <td>{{ optionalSeconds(item.p95_latency_ms) }}</td>
                <td>{{ item.total_api_calls ?? 0 }}/{{ item.total_token_usage ?? 0 }}</td>
                <td>
                  {{ item.asta_official?.adjusted_f1 == null
                    ? '—'
                    : percent(item.asta_official.adjusted_f1) }}
                </td>
                <td>{{ item.successful_queries }}/{{ item.failed_queries }}</td>
                <td>
                  <button type="button" class="view-run" @click="viewHistory(item.run_id)">
                    {{ report?.run_id === item.run_id ? '正在查看' : '查看' }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <button
          v-if="history.length < historyTotal"
          type="button"
          class="load-more"
          :disabled="historyLoading"
          @click="refreshHistory(false)"
        >
          {{ historyLoading ? '加载中…' : `加载更多（${history.length}/${historyTotal}）` }}
        </button>
      </section>

      <template v-if="report">
        <div class="current-report-head">
          <h3>当前查看报告</h3>
          <span>run {{ report.run_id.slice(0, 10) }}</span>
        </div>
        <div class="run-summary">
          <span>{{ report.dataset.name }}</span>
          <span>样本 {{ report.sample_count }}</span>
          <span>成功 {{ report.successful_queries }}</span>
          <span>失败 {{ report.failed_queries }}</span>
          <span>失败率 {{ percent(report.failure_rate) }}</span>
          <span>平均耗时 {{ seconds(report.mean_latency_ms) }}</span>
          <span>P50 {{ optionalSeconds(report.p50_latency_ms) }}</span>
          <span>P95 {{ optionalSeconds(report.p95_latency_ms) }}</span>
          <span>API {{ report.total_api_calls ?? 0 }}</span>
          <span>Token {{ report.total_token_usage ?? 0 }}</span>
        </div>
        <details v-if="Object.keys(report.failure_breakdown || {}).length" class="reproducibility-card">
          <summary>失败查询分类</summary>
          <dl>
            <div v-for="(count, category) in report.failure_breakdown" :key="category">
              <dt>{{ category }}</dt><dd>{{ count }}</dd>
            </div>
          </dl>
        </details>
        <details v-if="report.run_metadata" class="reproducibility-card">
          <summary>实验复现信息</summary>
          <dl>
            <div><dt>数据集 SHA-256</dt><dd :title="report.dataset_sha256">{{ shortHash(report.dataset_sha256) }}</dd></div>
            <div><dt>配置指纹</dt><dd :title="report.run_metadata.config_fingerprint">{{ shortHash(report.run_metadata.config_fingerprint) }}</dd></div>
            <div><dt>Provider 快照</dt><dd :title="report.provider_snapshot_sha256">{{ shortHash(report.provider_snapshot_sha256) }}</dd></div>
            <div><dt>Provider 模式</dt><dd>{{ report.run_metadata.provider_cache_mode === 'frozen' ? '冻结复测' : '建立/更新快照' }}</dd></div>
            <div><dt>评测协议</dt><dd>{{ report.dataset.evaluation_protocol || '旧报告未记录' }}</dd></div>
            <div><dt>排行榜可比</dt><dd>{{ report.dataset.leaderboard_comparable ? '是' : '否（内部诊断）' }}</dd></div>
            <div><dt>快照完整性</dt><dd>{{ report.run_metadata.provider_snapshot_complete ? '完整' : `不完整（${report.run_metadata.provider_snapshot_query_count || 0}/${report.sample_count}）` }}</dd></div>
            <div><dt>代码提交</dt><dd :title="report.run_metadata.code_commit">{{ shortHash(report.run_metadata.code_commit) }}{{ report.run_metadata.code_dirty ? '（有未提交改动）' : '' }}</dd></div>
            <div><dt>查询轮次</dt><dd>最多 {{ report.run_metadata.configuration?.retrieval?.max_iterations }} 轮，至少 {{ report.run_metadata.configuration?.retrieval?.minimum_evaluation_variants }} 个变体</dd></div>
            <div><dt>候选预算</dt><dd>{{ report.run_metadata.configuration?.models?.bge_m3?.candidate_limit }} / {{ report.run_metadata.configuration?.models?.bge_m3?.max_candidate_limit }}</dd></div>
            <div><dt>Reranker Top-N</dt><dd>{{ report.run_metadata.configuration?.models?.bge_reranker?.top_n }}</dd></div>
            <div><dt>本地索引</dt><dd>{{ report.run_metadata.configuration?.retrieval?.local_index_version }}</dd></div>
            <div><dt>快速评测</dt><dd>{{ report.run_metadata.fast_mode ? '开启' : '关闭' }}</dd></div>
          </dl>
        </details>

        <div class="metric-grid">
          <article v-for="metric in metrics" :key="metric.k" class="metric-card">
            <h3>@{{ metric.k }}</h3>
            <dl>
              <div><dt>Precision</dt><dd>{{ percent(metric.precision) }}</dd></div>
              <div><dt>Recall</dt><dd>{{ percent(metric.recall) }}</dd></div>
              <div><dt>F1</dt><dd>{{ percent(metric.f1) }}</dd></div>
              <div><dt>nDCG</dt><dd>{{ percent(metric.ndcg) }}</dd></div>
              <div><dt>MRR</dt><dd>{{ percent(metric.mrr) }}</dd></div>
              <div><dt>命中数</dt><dd>{{ metric.hits }}</dd></div>
            </dl>
          </article>
        </div>
        <details v-if="report.failed_queries && successfulMetrics.length" class="reproducibility-card">
          <summary>仅成功查询指标（诊断口径，不替代总体成绩）</summary>
          <dl>
            <div v-for="metric in successfulMetrics" :key="metric.k">
              <dt>@{{ metric.k }}</dt>
              <dd>F1 {{ percent(metric.f1) }} / Recall {{ percent(metric.recall) }} / nDCG {{ percent(metric.ndcg) }}</dd>
            </div>
          </dl>
        </details>

        <section v-if="report.asta_official" class="asta-official">
          <div class="official-head">
            <div>
              <span class="eyebrow">ASTABENCH OFFICIAL METRICS</span>
              <h3>AstaBench 官方指标</h3>
            </div>
            <span class="judge-status">Judge {{ report.asta_official.judge_status }}</span>
          </div>
          <div class="official-grid">
            <div>
              <span>Adjusted F1</span>
              <strong>{{ report.asta_official.adjusted_f1 == null ? '—' : percent(report.asta_official.adjusted_f1) }}</strong>
              <small>仅 Semantic 查询</small>
            </div>
            <div>
              <span>Estimated Recall</span>
              <strong>{{ percent(report.asta_official.estimated_recall) }}</strong>
              <small>使用官方 normalizer</small>
            </div>
            <div>
              <span>nDCG / Rank</span>
              <strong>{{ percent(report.asta_official.ndcg) }}</strong>
              <small>相关论文排序质量</small>
            </div>
            <div>
              <span>Corpus ID 有效率</span>
              <strong>{{ percent(report.asta_official.corpus_id_validity) }}</strong>
              <small>{{ report.asta_official.valid_corpus_ids }}/{{ report.asta_official.submitted_papers }} 篇</small>
            </div>
            <div v-if="report.asta_official.standard_f1 != null">
              <span>Standard F1</span>
              <strong>{{ percent(report.asta_official.standard_f1) }}</strong>
              <small>Specific / Metadata 查询</small>
            </div>
            <div>
              <span>Perfectly Relevant</span>
              <strong>{{ report.asta_official.perfectly_relevant }}</strong>
              <small>Judge 判为完全相关</small>
            </div>
          </div>
        </section>
      </template>

      <div v-if="displayedQueries.length" class="query-results">
        <h3>逐查询结果</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>查询</th>
                <th>返回/相关</th>
                <th>Gold 首次出现</th>
                <th>Provider 诊断</th>
                <th>F1@{{ selectedK }}</th>
                <th>Recall@{{ selectedK }}</th>
                <th>nDCG@{{ selectedK }}</th>
                <th>耗时</th>
                <th v-if="report?.asta_official">AstaBench</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in displayedQueries" :key="item.qid">
                <td>
                  <strong>{{ item.qid }}</strong>
                  <span>{{ item.query }}</span>
                  <em v-if="item.error">{{ item.error }}</em>
                </td>
                <td>{{ item.returned_count }}/{{ item.relevant_count }}</td>
                <td>
                  <strong>{{ firstSeenLabel(item) }}</strong>
                  <span v-if="stageDetail(item)">{{ stageDetail(item) }}</span>
                </td>
                <td class="provider-cell">
                  <template v-if="providerLabel(item).length">
                    <span v-for="label in providerLabel(item)" :key="label">{{ label }}</span>
                    <details>
                      <summary>查看查询表达式</summary>
                      <small
                        v-for="(diagnostic, index) in item.provider_diagnostics"
                        :key="`${diagnostic.iteration}-${diagnostic.provider}-${index}`"
                      >
                        R{{ diagnostic.iteration }} {{ diagnostic.provider }}：
                        {{ diagnostic.queries?.join(' | ') || diagnostic.query || '—' }}
                      </small>
                    </details>
                  </template>
                  <template v-else>旧报告无记录</template>
                </td>
                <td>{{ percent(item.metrics?.[String(selectedK)]?.f1) }}</td>
                <td>{{ percent(item.metrics?.[String(selectedK)]?.recall) }}</td>
                <td>{{ percent(item.metrics?.[String(selectedK)]?.ndcg) }}</td>
                <td>{{ seconds(item.latency_ms) }}</td>
                <td v-if="report?.asta_official">
                  <template v-if="item.asta_official">
                    <strong>
                      {{ item.asta_official.adjusted_f1 == null
                        ? `Standard F1 ${percent(item.asta_official.standard_f1)}`
                        : `Adjusted F1 ${percent(item.asta_official.adjusted_f1)}` }}
                    </strong>
                    <span>Corpus ID {{ percent(item.asta_official.corpus_id_validity) }}</span>
                  </template>
                  <template v-else>—</template>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.paper-test-mask {
  position: fixed;
  inset: 0;
  z-index: 80;
  padding: clamp(0.75rem, 3vw, 2rem);
  overflow-y: auto;
  background: rgba(10, 31, 38, 0.62);
  backdrop-filter: blur(8px);
}

.paper-test-panel {
  width: min(1120px, 100%);
  min-height: 80vh;
  margin: 0 auto;
  padding: clamp(1rem, 3vw, 2rem);
  border: 1px solid rgba(31, 111, 120, 0.2);
  border-radius: 24px;
  background: #f8f7f1;
  box-shadow: 0 30px 90px rgba(4, 25, 31, 0.35);
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
}

.eyebrow { color: #1f6f78; font-size: 0.74rem; font-weight: 800; letter-spacing: 0.09em; }
.panel-head h2 { margin: 0.35rem 0; font-family: 'Fraunces', serif; font-size: 1.8rem; }
.panel-head p { margin: 0; color: var(--muted); }
.close { border: 0; background: transparent; color: var(--muted); font-size: 2rem; cursor: pointer; }

.controls {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1.25fr auto;
  gap: 0.8rem;
  align-items: end;
  margin: 1.5rem 0 1rem;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.55);
}

.controls label { display: grid; gap: 0.4rem; color: var(--muted); font-size: 0.78rem; }
.controls select,
.controls input {
  width: 100%;
  min-height: 2.6rem;
  padding: 0 0.7rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: white;
  color: var(--ink);
}
.controls small { min-height: 1em; font-size: 0.68rem; }
.controls small.invalid { color: #9b2c2c; }
.number-control { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.4rem; }
.number-control button {
  min-height: 2.6rem;
  padding: 0 0.75rem;
  border: 1px solid rgba(31, 111, 120, 0.25);
  border-radius: 10px;
  color: #1f6f78;
  background: white;
  font-weight: 700;
  cursor: pointer;
}
.number-control button:disabled { cursor: not-allowed; opacity: 0.5; }
.run {
  min-height: 2.6rem;
  padding: 0 1.2rem;
  border: 0;
  border-radius: 999px;
  color: white;
  background: linear-gradient(135deg, #1f6f78, #174d56);
  font-weight: 800;
  cursor: pointer;
}
.run:disabled { cursor: not-allowed; opacity: 0.5; }

.progress-card, .panel-error, .empty {
  margin: 1rem 0;
  padding: 1rem;
  border-radius: 14px;
  background: rgba(31, 111, 120, 0.07);
}
.asta-notice {
  margin: 0 0 1rem;
  padding: 0.8rem 1rem;
  border-left: 3px solid #d08a45;
  border-radius: 8px;
  color: var(--ink-soft);
  background: rgba(208, 138, 69, 0.09);
  font-size: 0.82rem;
  line-height: 1.55;
}
.progress-line { display: flex; justify-content: space-between; }
.track { height: 8px; margin: 0.65rem 0; overflow: hidden; border-radius: 99px; background: rgba(31, 111, 120, 0.12); }
.track span { display: block; height: 100%; background: linear-gradient(90deg, #1f6f78, #d08a45); transition: width 0.3s ease; }
.progress-card p { margin: 0.45rem 0; color: var(--ink-soft); }
.progress-card small { color: var(--muted); }
.panel-error { color: #9b2c2c; background: rgba(155, 44, 44, 0.08); }
.empty { color: var(--muted); }

.run-summary { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
.run-summary span { padding: 0.35rem 0.7rem; border: 1px solid var(--line); border-radius: 99px; background: white; font-size: 0.78rem; }
.reproducibility-card { margin: 0 0 1rem; padding: 0.8rem 1rem; border: 1px solid var(--line); border-radius: 16px; background: rgba(255, 255, 255, 0.68); }
.reproducibility-card summary { cursor: pointer; color: var(--teal); font-weight: 700; }
.reproducibility-card dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.7rem 1rem; margin: 0.9rem 0 0; }
.reproducibility-card dl div { min-width: 0; }
.reproducibility-card dt { color: var(--muted); font-size: 0.72rem; }
.reproducibility-card dd { overflow-wrap: anywhere; margin: 0.2rem 0 0; font-size: 0.8rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
.metric-card { padding: 1rem; border: 1px solid var(--line); border-radius: 16px; background: white; }
.metric-card h3 { margin: 0 0 0.8rem; color: #1f6f78; }
.metric-card dl { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin: 0; }
.metric-card dl div { padding: 0.65rem; border-radius: 10px; background: #f4f5f1; }
.metric-card dt { color: var(--muted); font-size: 0.72rem; }
.metric-card dd { margin: 0.25rem 0 0; font-weight: 800; }

.history-section {
  margin: 1rem 0 1.25rem;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.5);
}
.history-head, .current-report-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.history-head h3, .current-report-head h3 { margin: 0.25rem 0 0; }
.history-head > span, .current-report-head > span { color: var(--muted); font-size: 0.75rem; }
.history-empty { margin-bottom: 0; }
.history-table { margin-top: 0.8rem; max-height: 310px; overflow: auto; }
.history-table td { white-space: nowrap; vertical-align: middle; }
.history-table td:nth-child(2) { min-width: 210px; white-space: normal; }
.history-table tr.selected td { background: rgba(31, 111, 120, 0.07); }
.metric-delta { font-weight: 700; color: var(--muted); }
.metric-delta.positive { color: #257a4f; }
.metric-delta.negative { color: #a44b43; }
.view-run, .load-more {
  border: 1px solid rgba(31, 111, 120, 0.25);
  border-radius: 999px;
  padding: 0.35rem 0.7rem;
  color: #1f6f78;
  background: white;
  font-weight: 700;
  cursor: pointer;
}
.load-more { display: block; margin: 0.8rem auto 0; }
.view-run:disabled, .load-more:disabled { cursor: not-allowed; opacity: 0.5; }
.current-report-head { margin: 1.25rem 0 0.65rem; }

.asta-official {
  margin-top: 1rem;
  padding: 1rem;
  border: 1px solid rgba(208, 138, 69, 0.3);
  border-radius: 16px;
  background: linear-gradient(145deg, rgba(255, 250, 241, 0.95), rgba(240, 247, 245, 0.95));
}
.official-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.official-head h3 { margin: 0.25rem 0 0; }
.judge-status { padding: 0.3rem 0.65rem; border-radius: 99px; background: white; color: var(--muted); font-size: 0.72rem; }
.official-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.7rem; margin-top: 0.9rem; }
.official-grid > div { display: grid; gap: 0.25rem; padding: 0.8rem; border: 1px solid rgba(31, 111, 120, 0.12); border-radius: 12px; background: rgba(255, 255, 255, 0.7); }
.official-grid span, .official-grid small { color: var(--muted); font-size: 0.72rem; }
.official-grid strong { color: #174d56; font-size: 1.25rem; }

.query-results { margin-top: 1.5rem; }
.query-results h3 { margin-bottom: 0.7rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }
table { width: 100%; border-collapse: collapse; background: white; font-size: 0.82rem; }
th, td { padding: 0.75rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--muted); background: #f1f3ef; white-space: nowrap; }
td:first-child { min-width: 320px; }
td strong, td span, td em { display: block; }
td span { margin-top: 0.25rem; color: var(--ink-soft); line-height: 1.4; }
td em { margin-top: 0.25rem; color: #9b2c2c; font-style: normal; }
.provider-cell { min-width: 240px; }
.provider-cell > span { margin-top: 0; font-size: 0.72rem; }
.provider-cell details { margin-top: 0.4rem; color: var(--muted); }
.provider-cell summary { cursor: pointer; font-weight: 700; }
.provider-cell small { display: block; max-width: 360px; margin-top: 0.35rem; line-height: 1.35; }

@media (max-width: 760px) {
  .controls { grid-template-columns: 1fr 1fr; }
  .controls label:first-child, .run { grid-column: 1 / -1; }
  .metric-card dl { grid-template-columns: repeat(2, 1fr); }
}
</style>
