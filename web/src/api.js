export async function streamChat(prompt, max_tokens, onToken) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens }),
  })
  if (!res.ok) throw new Error(`chat ${res.status}`)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = '', full = ''
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
      if (msg.done) return msg.text ?? full
    }
  }
  return full
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
