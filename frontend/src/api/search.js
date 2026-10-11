import axios from 'axios'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 60000,
})

export async function searchPapers(query, options = {}) {
  const { data } = await client.post('/api/search', {
    query,
    use_mock: options.useMock ?? false,
    max_papers: options.maxPapers ?? 5,
  })
  return data
}

/**
 * Stream search progress via SSE over POST.
 * handlers: { onStep, onUnderstanding, onResult, onError }
 */
export async function searchPapersStream(query, options = {}, handlers = {}) {
  const res = await fetch(`${import.meta.env.VITE_API_BASE || ''}/api/search/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      query,
      use_mock: options.useMock ?? false,
      max_papers: options.maxPapers ?? 5,
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  if (!res.body) {
    throw new Error('浏览器不支持流式响应')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalResult = null
  let invalidHandled = false
  let streamError = ''

  const dispatch = (eventName, data) => {
    if (eventName === 'step') handlers.onStep?.(data)
    else if (eventName === 'understanding') handlers.onUnderstanding?.(data)
    else if (eventName === 'invalid_query') {
      invalidHandled = true
      handlers.onInvalidQuery?.(data)
    } else if (eventName === 'result') {
      finalResult = data.result
      handlers.onResult?.(data)
    } else if (eventName === 'error') {
      streamError = data?.detail || '搜索失败'
      handlers.onError?.(data)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)

      let eventName = 'message'
      const dataLines = []
      for (const line of rawEvent.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      try {
        const data = JSON.parse(dataLines.join('\n'))
        dispatch(eventName, data)
      } catch {
        // ignore malformed chunk
      }
    }
  }

  if (invalidHandled) {
    return null
  }
  if (!finalResult) {
    throw new Error(streamError || '流式检索未返回最终结果')
  }
  return finalResult
}

export async function fetchHealth() {
  const { data } = await client.get('/api/health')
  return data
}

export async function fetchPaperTestDatasets() {
  const { data } = await client.get('/api/paper-test/datasets')
  return data
}

export async function fetchLatestPaperTest() {
  const { data } = await client.get('/api/paper-test/latest')
  return data
}

export async function fetchPaperTestHistory(options = {}) {
  const { data } = await client.get('/api/paper-test/history', {
    params: {
      limit: options.limit ?? 20,
      offset: options.offset ?? 0,
    },
  })
  return data
}

export async function fetchPaperTestHistoryDetail(runId) {
  const { data } = await client.get(`/api/paper-test/history/${encodeURIComponent(runId)}`)
  return data
}

/** Stream PaSa evaluation progress. One ScholarAgent search is run per sample. */
export async function runPaperTestStream(options = {}, handlers = {}) {
  const res = await fetch(`${import.meta.env.VITE_API_BASE || ''}/api/paper-test/run/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      dataset: options.dataset || 'pasa-real',
      sample_limit: options.sampleLimit ?? 3,
      ks: options.ks?.length ? options.ks : [5, 10],
      seed: options.seed ?? 42,
      use_mock: options.useMock ?? false,
      provider_cache_mode: options.providerCacheMode || 'record',
    }),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  if (!res.body) throw new Error('浏览器不支持流式响应')

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let finalReport = null
  let streamError = ''

  const dispatch = (eventName, data) => {
    if (eventName === 'start') handlers.onStart?.(data)
    else if (eventName === 'progress') handlers.onProgress?.(data)
    else if (eventName === 'query_result') handlers.onQueryResult?.(data)
    else if (eventName === 'result') {
      finalReport = data.report
      handlers.onResult?.(data)
    } else if (eventName === 'error') {
      streamError = data?.detail || '评测失败'
      handlers.onError?.(data)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let separator
    while ((separator = buffer.indexOf('\n\n')) >= 0) {
      const rawEvent = buffer.slice(0, separator)
      buffer = buffer.slice(separator + 2)
      let eventName = 'message'
      const dataLines = []
      for (const line of rawEvent.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      try {
        dispatch(eventName, JSON.parse(dataLines.join('\n')))
      } catch {
        // Ignore malformed chunks; the final-report guard below remains fail-closed.
      }
    }
  }

  if (!finalReport) throw new Error(streamError || '评测未返回最终报告')
  return finalReport
}
