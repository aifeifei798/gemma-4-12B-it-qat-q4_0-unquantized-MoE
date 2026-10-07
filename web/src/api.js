export async function streamChat(prompt, max_tokens, onToken, extra, signal) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens, ...(extra || {}) }),
    signal,
  })
  if (!res.ok) throw new Error(`chat ${res.status}`)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = '', full = '', meta = null, stopped = false, err = null
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop()
      for (const p of parts) {
        if (!p.startsWith('data: ')) continue
        const msg = JSON.parse(p.slice(6))
        if (msg.token) { full += msg.token; onToken(msg.token) }
        if (msg.error && !full) err = msg.error
        if (msg.done) return { text: msg.text ?? full, meta: msg.meta || null, stopped: !!msg.stopped, error: msg.error || err }
      }
    }
  } catch (e) {
    if (e && e.name === 'AbortError') return { text: full, meta: null, stopped: true, aborted: true }
    throw e
  }
  return { text: full, meta, stopped, error: err }
}

export async function fetchProbe(prompt, response, domain) {
  const res = await fetch('/api/probe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, response, domain }),
  })
  const data = await res.json()
  if (!res.ok || !Array.isArray(data.macro_hist)) {
    throw new Error(data.error || data.detail ? JSON.stringify(data.detail || data.error) : `probe ${res.status}`)
  }
  return data
}

export async function fetchHealth() {
  return (await fetch('/api/health')).json()
}

export async function hotswap(weight) {
  const res = await fetch('/api/hotswap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weight }),
  })
  return res.json()
}

export async function getSteering() {
  return (await fetch('/admin/steering')).json()
}

export async function setSteering(patch) {
  const res = await fetch('/admin/steering', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  return res.json()
}

export async function gpuClear() {
  const res = await fetch('/admin/gpu-clear', { method: 'POST' })
  return res.json()
}

export async function reloadWeights(weight) {
  const res = await fetch('/admin/reload-weights', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weight }),
  })
  return res.json()
}

export async function markBadcase(payload) {
  const res = await fetch('/data/mark-badcase', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return res.json()
}

export async function fetchBadcases(limit = 20) {
  return (await fetch(`/data/badcases?limit=${limit}`)).json()
}

export async function uploadFile(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/upload', { method: 'POST', body: fd })
  return res.json()
}

export async function analyzeIntent(prompt) {
  const res = await fetch('/v1/analyze/intent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })
  return res.json()
}

export async function fetchDomains() {
  return (await fetch('/api/domains')).json()
}
