import React, { useState, useEffect, useRef } from 'react'
import { streamChat, fetchProbe, fetchHealth, hotswap, getSteering, setSteering,
         gpuClear, markBadcase, fetchDomains, fetchBadcases, uploadFile,
         fetchStrength } from './api.js'
import { ArchPanel, SteeringPanel, MetaBadge, Md, PerReqPanel, IntentBox,
         BadcasePanel, GatewayPanel, StrengthPanel, PatchPanel } from './components.jsx'
import { STR } from './i18n.js'
import './styles.css'

const DOMAINS_ZH = ['代码工程Code', '严密数学Math', '自然科学Science', '文学创意Creative_Arts',
  '商务对话Business_Dialogue', '逻辑思辨Logic_Philosophy', '严格约束Constraint_Rules', '中文特区Chinese_Slots']
const DOMAINS_EN = ['Code', 'Math', 'Science', 'Creative_Arts', 'Business_Dialogue',
  'Logic_Philosophy', 'Constraint_Rules', 'Chinese_Slots']

const LS_SESS = 'mm-sessions'
const LS_ACTIVE = 'mm-active'
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
const PER_REQ_DEFAULT = {
  armed: false, routing_temperature: 1.0, force_macro: -1,
  disabled_macros: [], disabled_clusters: [],
  disable_shared: false, disable_macro: false,
  disable_micro: false, disable_all_moe: false, max_tokens: 128,
}

function loadSessions() {
  try {
    const arr = JSON.parse(localStorage.getItem(LS_SESS) || '[]')
    return Array.isArray(arr) ? arr : []
  } catch (e) { return [] }
}

export default function App() {
  const [sessions, setSessions] = useState(loadSessions)
  const [activeId, setActiveId] = useState(() => localStorage.getItem(LS_ACTIVE) || null)
  const [input, setInput] = useState('')
  const [domain, setDomain] = useState(0)
  const [busy, setBusy] = useState(false)
  const [probe, setProbe] = useState(null)
  const [probing, setProbing] = useState(false)
  const [probeErr, setProbeErr] = useState('')
  const [health, setHealth] = useState(null)
  const [patch, setPatch] = useState('myriad_moe_patch_poetry.pt')
  const [swapMsg, setSwapMsg] = useState('')
  const [steer, setSteer] = useState(null)
  const [cores, setCores] = useState([])
  const [clusters, setClusters] = useState([])
  const [badMsg, setBadMsg] = useState('')
  const [cockpit, setCockpit] = useState(true)
  const [cockpitTab, setCockpitTab] = useState('steer')
  const [perReq, setPerReq] = useState(PER_REQ_DEFAULT)
  const [badcases, setBadcases] = useState({ total: 0, cases: [] })
  const [strength, setStrength] = useState(null)
  const [atts, setAtts] = useState([])  // 待发送附件 [{id, kind, name, url}]
  const fileRef = useRef(null)
  const [lang, setLang] = useState(() => localStorage.getItem('mm-lang') || 'en')
  const ctrlRef = useRef(null)
  const chatRef = useRef(null)
  const t = STR[lang]
  const DOMAINS = lang === 'en' ? DOMAINS_EN : DOMAINS_ZH

  const active = sessions.find(s => s.id === activeId) || null
  const msgs = active ? active.msgs : []

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => {})
    getSteering().then(setSteer).catch(() => {})
    fetchDomains().then(d => { setCores(d.cores || []); setClusters(d.clusters || []) }).catch(() => {})
    fetchBadcases(20).then(r => { if (r.ok) setBadcases({ total: r.total, cases: r.cases }) }).catch(() => {})
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem(LS_SESS, JSON.stringify(sessions.slice(0, 50)))
      if (activeId) localStorage.setItem(LS_ACTIVE, activeId)
    } catch (e) { /* ignore */ }
  }, [sessions, activeId])
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight })
  }, [msgs, activeId])

  function toggleLang() {
    const next = lang === 'en' ? 'zh' : 'en'
    setLang(next)
    try { localStorage.setItem('mm-lang', next) } catch (e) { /* ignore */ }
  }

  function touchSession(id, fn) {
    setSessions(ss => ss.map(s => s.id === id
      ? { ...s, msgs: fn(s.msgs), updatedAt: Date.now() } : s))
  }

  function newChat() {
    if (busy) return
    const id = uid()
    setSessions(ss => [{ id, title: t.newChat, msgs: [], updatedAt: Date.now() }, ...ss])
    setActiveId(id)
    setProbe(null); setProbeErr('')
  }

  function delSession(e, id) {
    e.stopPropagation()
    setSessions(ss => {
      const rest = ss.filter(s => s.id !== id)
      if (id === activeId) setActiveId(rest[0]?.id || null)
      return rest
    })
  }

  async function doSwap() {
    setSwapMsg(t.swapping)
    try {
      const r = await hotswap(patch.trim())
      setSwapMsg(r.ok ? t.swapped(r) : `${t.probeFail('')}: ${r.error}`)
    } catch (e) { setSwapMsg(t.probeFail(e.message)) }
  }

  async function patchSteer(p) {
    try {
      const r = await setSteering(p)
      if (r.routing_temperature !== undefined) setSteer(r)
      else setBadMsg(t.probeFail(JSON.stringify(r)))
    } catch (e) { setBadMsg(t.probeFail(e.message)) }
  }

  async function doClear() {
    try {
      const r = await gpuClear()
      setSwapMsg(r.ok ? t.cleared(r) : t.probeFail(r.error || ''))
      setHealth(await fetchHealth())
    } catch (e) { setSwapMsg(t.probeFail(e.message)) }
  }

  async function loadBad() {
    try {
      const r = await fetchBadcases(20)
      if (r.ok) setBadcases({ total: r.total, cases: r.cases })
    } catch (e) { /* ignore */ }
  }

  async function loadStrength() {
    try {
      const r = await fetchStrength()
      if (r.cores) setStrength(r)
      else setBadMsg(t.probeFail(r.error || ''))
    } catch (e) { setBadMsg(t.probeFail(e.message)) }
  }

  async function markBad(i) {
    const m = msgs[i]
    if (!m || !m.prompt) return
    setBadMsg('')
    try {
      const r = await markBadcase({
        prompt: m.prompt, response: m.text,
        pred_macro: m.meta?.macro_top?.[0] ?? null,
        pred_cluster: m.meta?.cluster_top ?? null,
        correct_macro: domain, correct_cluster: -1,
      })
      setBadMsg(r.ok ? t.badOk(r.total) : t.badFail(r.error || ''))
      if (r.ok) loadBad()
    } catch (e) { setBadMsg(t.badFail(e.message)) }
  }

  async function pickFiles(ev) {
    const files = [...(ev.target.files || [])]
    ev.target.value = ''
    for (const f of files) {
      try {
        const r = await uploadFile(f)
        if (r.ok) {
          setAtts(a => [...a, { id: r.id, kind: r.kind, name: r.name,
                                 url: URL.createObjectURL(f) }])
        } else {
          setBadMsg(t.uploadFail(r.error || ''))
        }
      } catch (e) { setBadMsg(t.uploadFail(e.message)) }
    }
  }

  async function send(prefill) {
    const prompt = (typeof prefill === 'string' ? prefill : input).trim()
    if ((!prompt && atts.length === 0) || busy) return
    const myAtts = [...atts]
    setAtts([])
    const title = (prompt || myAtts.map(a => a.name).join(',')).slice(0, 28)
    let sid = activeId
    if (!sid) {
      sid = uid()
      setSessions(ss => [{ id: sid, title, msgs: [], updatedAt: Date.now() }, ...ss])
      setActiveId(sid)
    } else {
      setSessions(ss => ss.map(s => s.id === sid && s.msgs.length === 0
        ? { ...s, title } : s))
    }
    setBusy(true); setProbe(null); setProbeErr('')
    setInput('')
    const history = msgs.filter(m => m.text).slice(-20)
      .map(m => ({ role: m.role === 'ai' ? 'assistant' : 'user', content: m.text }))
    touchSession(sid, m => [...m, { role: 'user', text: prompt, atts: myAtts },
      { role: 'ai', text: '', prompt, meta: null }])
    const ctrl = new AbortController()
    ctrlRef.current = ctrl
    // 单次覆盖: 武装时把tab里的值打包进本次请求, 发完自动卸装 (全局不动)
    const one = perReq.armed ? perReq : null
    const extra = { ...(history.length ? { history } : null),
                    ...(myAtts.length ? { files: myAtts.map(a => a.id) } : null) }
    let maxTok = 128
    if (one) {
      maxTok = one.max_tokens
      if (one.routing_temperature !== 1.0) extra.routing_temperature = one.routing_temperature
      if (one.force_macro >= 0) extra.force_macro = one.force_macro
      if (one.disabled_macros.length) extra.disabled_macros = one.disabled_macros
      if (one.disabled_clusters.length) extra.disabled_clusters = one.disabled_clusters
      for (const k of ['disable_shared', 'disable_macro', 'disable_micro', 'disable_all_moe'])
        if (one[k]) extra[k] = true
    }
    let res
    try {
      res = await streamChat(prompt, maxTok, tok =>
        touchSession(sid, m => {
          const c = [...m]
          c[c.length - 1] = { ...c[c.length - 1], text: c[c.length - 1].text + tok }
          return c
        }),
        extra, ctrl.signal)
      if (res.error && !res.text) throw new Error(res.error)
      touchSession(sid, m => {
        const c = [...m]
        c[c.length - 1] = { ...c[c.length - 1], text: res.text, meta: res.meta, stopped: res.stopped }
        return c
      })
    } catch (e) {
      touchSession(sid, m => {
        const c = [...m]
        c[c.length - 1] = { role: 'ai', text: t.genFail(e.message) }
        return c
      })
      setBusy(false)
      ctrlRef.current = null
      return
    }
    setBusy(false)
    ctrlRef.current = null
    if (one) setPerReq(PER_REQ_DEFAULT)  // 一次性: 发完卸装
    if (!res.text) return  // 中断且无输出: 不跑探针
    setProbing(true)
    try { setProbe(await fetchProbe(prompt, res.text, domain)) }
    catch (e) { setProbeErr(t.probeFail(e.message)) }
    finally { setProbing(false) }
  }

  function stop() {
    try { ctrlRef.current?.abort() } catch (e) { /* ignore */ }
  }

  const examples = lang === 'en'
    ? ['Write a Python quicksort, code only', 'Explain Newton\'s laws simply', '白日依山尽是什么意思?']
    : ['用Python写快排,只给代码', '简单解释牛顿定律', '说说白日依山尽']

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-btn" onClick={newChat}>＋ {t.newChat}</button>
        <h4>{t.sessions}</h4>
        <div className="sess-list">
          {sessions.map(s => (
            <div key={s.id} className={'sess' + (s.id === activeId ? ' active' : '')}
                 onClick={() => { if (!busy) { setActiveId(s.id); setProbe(null); setProbeErr('') } }}>
              <span className="title">{s.title || t.newChat}</span>
              <button className="del" title={t.del} onClick={e => delSession(e, s.id)}>×</button>
            </div>
          ))}
        </div>
        <div className="side-foot">
          Myriad-MoE v3<br />
          {health ? (health.ok
            ? `⚡ ${health.vram_gb}G · ${health.layers.length}${t.layersUnit}`
            : t.offline) : t.connecting}
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          {t.title}
          <span className={'status-dot' + (health?.ok ? ' ok' : '')}>
            {health ? (health.ok ? '●' : '○') : '…'}
          </span>
          <span className="spacer" />
          <button className="icon-btn" onClick={toggleLang}>{lang === 'en' ? '中文' : 'EN'}</button>
          <button className={'icon-btn' + (cockpit ? ' active' : '')} onClick={() => setCockpit(!cockpit)}>
            🎛 {t.cockpitBtn}
          </button>
        </div>

        <div className="chat" ref={chatRef}>
          <div className="col">
            {msgs.length === 0 && (
              <div className="hero">
                <h1>{t.hero}</h1>
                <div className="chips">
                  {examples.map((ex, i) => (
                    <button key={i} className="chip" onClick={() => send(ex)}>{ex}</button>
                  ))}
                </div>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className="msg-row">
                <div className={'avatar' + (m.role === 'ai' ? ' ai' : ' user')}>
                  {m.role === 'ai' ? 'M' : (lang === 'en' ? 'U' : '你')}
                </div>
                <div className="msg-body">
                  {m.role === 'user' ? (
                    <div className="user-bubble">
                      {(m.atts || []).length > 0 && (
                        <div style={{ marginBottom: 6, display: 'flex', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                          {(m.atts || []).map((a, k) => a.kind === 'image' && a.url ? (
                            <img key={k} src={a.url} alt={a.name}
                                 style={{ maxWidth: 180, maxHeight: 140, borderRadius: 10 }} />
                          ) : (
                            <span key={k} className="kill">♪ {a.name}</span>
                          ))}
                        </div>
                      )}
                      {m.text && <span className="ub">{m.text}</span>}
                    </div>
                  ) : (
                    <div className={m.text ? '' : 'typing'}>
                      {m.text ? <Md text={m.text} /> : ''}
                    </div>
                  )}
                  {m.role === 'ai' && m.text && <MetaBadge meta={m.meta} />}
                  {m.role === 'ai' && m.stopped && (
                    <div className="stopped-line">{t.stopped}</div>
                  )}
                  {m.role === 'ai' && m.text && (
                    <button className="bad-btn" onClick={() => markBad(i)} title={t.markTitle}>👎</button>
                  )}
                </div>
              </div>
            ))}
            {(swapMsg || badMsg || probeErr) && (
              <div className="flash">{swapMsg} {badMsg} {probeErr}</div>
            )}
          </div>
        </div>

        <div className="composer-wrap">
          <div className="composer">
            {atts.length > 0 && (
              <div className="att-chips">
                {atts.map(a => (
                  <span key={a.id} className="att-chip">
                    {a.kind === 'image' && a.url
                      ? <img src={a.url} alt={a.name} />
                      : <span>♪</span>}
                    {a.name.length > 18 ? a.name.slice(0, 16) + '…' : a.name}
                    <button onClick={() => setAtts(x => x.filter(y => y.id !== a.id))}>×</button>
                  </span>
                ))}
              </div>
            )}
            <div className="comp-bar">
              <button className="clip-btn" onClick={() => fileRef.current?.click()} title={t.attach}>📎</button>
              <input ref={fileRef} type="file" multiple accept="image/*,audio/*" style={{ display: 'none' }}
                     onChange={pickFiles} />
              <input value={input} onChange={e => setInput(e.target.value)}
                     onKeyDown={e => e.key === 'Enter' && send()}
                     placeholder={t.inputPh} />
              {perReq.armed && <span className="armed" title={t.onceTitle}>⚡</span>}
              {busy
                ? <button className="send-btn stop" onClick={stop} title={t.stop}>■</button>
                : <button className="send-btn" onClick={() => send()} disabled={!input.trim() && atts.length === 0} title={t.send}>↑</button>}
            </div>
          </div>
        </div>
      </div>

      <aside className={'cockpit' + (cockpit ? '' : ' hidden')}>
        <div className="tabs">
          {[['steer', t.tabSteer], ['once', t.tabOnce], ['probe', t.tabProbe],
            ['power', t.tabStrength], ['patch', t.tabPatch],
            ['data', t.tabData], ['gw', t.tabGateway]].map(([k, label]) => (
            <button key={k} className={'tab' + (cockpitTab === k ? ' active' : '')}
                    onClick={() => { setCockpitTab(k); if (k === 'power' && !strength) loadStrength() }}>
              {label}{k === 'once' && perReq.armed ? ' ●' : ''}
            </button>
          ))}
        </div>
        {cockpitTab === 'steer' && (
          <SteeringPanel steer={steer} cores={cores} clusters={clusters} onPatch={patchSteer} t={t} lang={lang} />
        )}
        {cockpitTab === 'once' && (
          <PerReqPanel perReq={perReq} setPerReq={setPerReq} cores={cores} clusters={clusters}
                       onPrefix={px => setInput(input ? (input.startsWith('/') ? input : `${px} ${input}`) : `${px} `)}
                       t={t} lang={lang} />
        )}
        {cockpitTab === 'probe' && (
          <>
            <div className="panel">
              <h3>🎯 {t.domainPick}:
                <select value={domain} onChange={e => setDomain(+e.target.value)}
                        title={t.domainTitle}
                        style={{ background: '#2f2f2f', color: '#ececec', border: '1px solid #424242',
                          borderRadius: 6, marginLeft: 6, fontSize: 12 }}>
                  {DOMAINS.map((d, i) => <option key={i} value={i}>{i}·{d}</option>)}
                </select>
              </h3>
            </div>
            <IntentBox t={t} />
            <h3 className="probe-h" style={{ color: '#ececec' }}>{t.probeTitle}</h3>
            <ArchPanel probe={probe} probing={probing} t={t} lang={lang} />
          </>
        )}
        {cockpitTab === 'power' && (
          <StrengthPanel data={strength} onRefresh={loadStrength} steer={steer} onPatch={patchSteer} t={t} lang={lang} />
        )}
        {cockpitTab === 'patch' && (
          <PatchPanel t={t} />
        )}
        {cockpitTab === 'data' && (
          <>
            <div className="panel">
              <h3>💾 {t.hwTitle}</h3>
              <div className="row">
                ⚡ {health ? `${health.vram_gb} / ${health.vram_reserved_gb}G` : '…'}
                <button className="tgl" onClick={doClear}>{t.vramClear}</button>
              </div>
              <div className="row">
                {t.ctxCap}:
                <input key={steer?.max_context_tokens} defaultValue={steer?.max_context_tokens ?? 4096}
                       onBlur={e => {
                         const v = parseInt(e.target.value, 10)
                         if (v >= 256 && v <= 16384) patchSteer({ max_context_tokens: v })
                       }}
                       style={{ width: 70, background: '#2f2f2f', color: '#ececec', border: '1px solid #424242',
                         borderRadius: 6, padding: 3, fontSize: 11, marginLeft: 6 }} />
              </div>
              <div className="swap-row">
                <input value={patch} onChange={e => setPatch(e.target.value)} placeholder={t.swapPh} />
                <button className="tgl" onClick={doSwap}>{t.swapBtn}</button>
              </div>
            </div>
            <BadcasePanel cases={badcases.cases} total={badcases.total} onRefresh={loadBad} t={t} />
          </>
        )}
        {cockpitTab === 'gw' && <GatewayPanel t={t} />}
      </aside>
    </div>
  )
}
