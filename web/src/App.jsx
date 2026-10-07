import React, { useState, useEffect } from 'react'
import { streamChat, fetchProbe, fetchHealth, hotswap } from './api.js'
import { ArchPanel } from './components.jsx'
import { STR } from './i18n.js'

const DOMAINS_ZH = ['代码工程Code', '严密数学Math', '自然科学Science', '文学创意Creative_Arts',
  '商务对话Business_Dialogue', '逻辑思辨Logic_Philosophy', '严格约束Constraint_Rules', '中文特区Chinese_Slots']
const DOMAINS_EN = ['Code', 'Math', 'Science', 'Creative_Arts', 'Business_Dialogue',
  'Logic_Philosophy', 'Constraint_Rules', 'Chinese_Slots']

export default function App() {
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('Write a Python quicksort function, code only')
  const [domain, setDomain] = useState(0)
  const [busy, setBusy] = useState(false)
  const [probe, setProbe] = useState(null)
  const [probing, setProbing] = useState(false)
  const [probeErr, setProbeErr] = useState('')
  const [health, setHealth] = useState(null)
  const [patch, setPatch] = useState('myriad_moe_patch_poetry.pt')
  const [swapMsg, setSwapMsg] = useState('')
  const [lang, setLang] = useState(() => localStorage.getItem('mm-lang') || 'en')
  const t = STR[lang]
  const DOMAINS = lang === 'en' ? DOMAINS_EN : DOMAINS_ZH
  useEffect(() => { fetchHealth().then(setHealth).catch(() => {}) }, [])
  function toggleLang() {
    const next = lang === 'en' ? 'zh' : 'en'
    setLang(next)
    try { localStorage.setItem('mm-lang', next) } catch (e) { /* ignore */ }
  }

  async function doSwap() {
    setSwapMsg(t.swapping)
    try {
      const r = await hotswap(patch.trim())
      setSwapMsg(r.ok ? t.swapped(r) : `${t.probeFail('')}: ${r.error}`)
    } catch (e) { setSwapMsg(t.probeFail(e.message)) }
  }

  async function send() {
    const prompt = input.trim()
    if (!prompt || busy) return
    setBusy(true); setProbe(null); setProbeErr('')
    setMsgs(m => [...m, { role: 'user', text: prompt }, { role: 'ai', text: '' }])
    let full = ''
    try {
      full = await streamChat(prompt, 128, tok =>
        setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: 'ai', text: c[c.length - 1].text + tok }; return c }))
      // done时用服务端清洗过的完整文本替换气泡 (去掉流式中漏出的停止符碎片)
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: 'ai', text: full }; return c })
    } catch (e) {
      setMsgs(m => { const c = [...m]; c[c.length - 1] = { role: 'ai', text: t.genFail(e.message) }; return c })
      setBusy(false)
      return
    }
    setBusy(false)
    setProbing(true)
    try { setProbe(await fetchProbe(prompt, full, domain)) }
    catch (e) { setProbeErr(t.probeFail(e.message)) }
    finally { setProbing(false) }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#111', color: '#eee', fontFamily: 'sans-serif' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 16 }}>
        <h2 style={{ margin: '0 0 8px' }}>{t.title} <small style={{ color: '#888' }}>{t.sub}</small>
          <button onClick={toggleLang} style={{ float: 'right', fontSize: 12, padding: '4px 10px', borderRadius: 6 }}>
            {lang === 'en' ? '中文' : 'EN'}
          </button>
        </h2>
        <div style={{ fontSize: 12, color: health?.ok ? '#7c7' : '#c77', marginBottom: 8 }}>
          {health ? (health.ok ? t.online(health) : t.offline) : t.connecting}
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <input value={patch} onChange={e => setPatch(e.target.value)} title={t.swapPh} placeholder={t.swapPh}
                 style={{ flex: 1, background: '#222', color: '#eee', border: '1px solid #444', borderRadius: 6, padding: 6, fontSize: 12 }} />
          <button onClick={doSwap} style={{ padding: '6px 12px', borderRadius: 6, fontSize: 12 }}>{t.swapBtn}</button>
        </div>
        {swapMsg && <div style={{ fontSize: 12, color: '#fc6', marginBottom: 8 }}>{swapMsg}</div>}
        <div style={{ flex: 1, overflowY: 'auto', border: '1px solid #333', borderRadius: 8, padding: 12, marginBottom: 8 }}>
          {msgs.map((m, i) => (
            <div key={i} style={{ margin: '8px 0', textAlign: m.role === 'user' ? 'right' : 'left' }}>
              <span style={{ display: 'inline-block', maxWidth: '85%', textAlign: 'left', whiteSpace: 'pre-wrap',
                background: m.role === 'user' ? '#254' : '#223', padding: '8px 12px', borderRadius: 8,
                fontFamily: m.role === 'ai' ? 'monospace' : 'inherit', fontSize: 13 }}>{m.text || '…'}</span>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <select value={domain} onChange={e => setDomain(+e.target.value)}
                  style={{ background: '#222', color: '#eee', borderRadius: 6 }}>
            {DOMAINS.map((d, i) => <option key={i} value={i}>{i}·{d}</option>)}
          </select>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()}
                 placeholder={t.inputPh}
                 style={{ flex: 1, background: '#222', color: '#eee', border: '1px solid #444', borderRadius: 6, padding: 8 }} />
          <button onClick={send} disabled={busy} style={{ padding: '8px 20px', borderRadius: 6 }}>{t.send}</button>
        </div>
      </div>
      <div style={{ width: 420, borderLeft: '1px solid #333', padding: 16, overflowY: 'auto' }}>
        <h2 style={{ margin: '0 0 8px' }}>{t.probeTitle}</h2>
        {probeErr && <div style={{ color: '#f66', fontSize: 13, marginBottom: 8 }}>{probeErr}</div>}
        <ArchPanel probe={probe} probing={probing} t={t} lang={lang} />
      </div>
    </div>
  )
}
