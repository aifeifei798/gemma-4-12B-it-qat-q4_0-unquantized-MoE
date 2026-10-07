import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'
import { CLAN_EN, coreName } from './i18n.js'

export function Md({ text }) {
  return (
    <div className="md" style={{ fontSize: 14, lineHeight: 1.7, overflowWrap: 'anywhere' }}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code({ className, children }) {
            const raw = String(children ?? '').replace(/\n$/, '')
            const m = /language-([\w+-]+)/.exec(className || '')
            let html = null
            try {
              html = (m && hljs.getLanguage(m[1]))
                ? hljs.highlight(raw, { language: m[1] }).value
                : hljs.highlightAuto(raw).value
            } catch (e) { html = null }
            if (html && (className || raw.includes('\n'))) {
              return (
                <span style={{ position: 'relative', display: 'block' }}>
                  <button
                    onClick={ev => {
                      navigator.clipboard?.writeText(raw).catch(() => {})
                      ev.target.textContent = '✓'
                      setTimeout(() => { ev.target.textContent = '⧉' }, 1200)
                    }}
                    style={{ position: 'absolute', right: 6, top: 6, fontSize: 11,
                      background: '#333', color: '#ddd', border: '1px solid #555',
                      borderRadius: 5, cursor: 'pointer', padding: '1px 7px' }}>
                    ⧉
                  </button>
                  <code className={className}
                    dangerouslySetInnerHTML={{ __html: html }}
                    style={{ display: 'block', background: '#0d1117', padding: '10px 12px',
                      borderRadius: 6, overflowX: 'auto', fontFamily: 'monospace', fontSize: 12 }} />
                </span>
              )
            }
            return <code style={{ background: '#333', padding: '1px 5px', borderRadius: 4,
              fontFamily: 'monospace', fontSize: 12 }}>{children}</code>
          },
          pre({ children }) { return <>{children}</> },
          a({ href, children }) {
            return <a href={href} target="_blank" rel="noreferrer" style={{ color: '#8cf' }}>{children}</a>
          },
          table({ children }) {
            return <table style={{ borderCollapse: 'collapse', margin: '8px 0' }}>{children}</table>
          },
          th({ children }) {
            return <th style={{ border: '1px solid #555', padding: '4px 8px', background: '#222' }}>{children}</th>
          },
          td({ children }) {
            return <td style={{ border: '1px solid #555', padding: '4px 8px' }}>{children}</td>
          },
        }}>
        {text}
      </ReactMarkdown>
    </div>
  )
}

export function Bar({ label, value, max = 1, color = '#19c37d' }) {
  return (
    <div className="bar-row">
      <div className="bar-label">{label}</div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color }} />
      </div>
      <div className="bar-val">{(value * 100).toFixed(1)}%</div>
    </div>
  )
}

export function MetaBadge({ meta }) {
  if (!meta) return null
  const [, b] = meta.macro_top || [-1, -1]
  const nm = n => (meta.macro_names && meta.macro_names[n]) || ''
  return (
    <div className="meta-line">
      🏷 {nm(0)}{b >= 0 ? ` + ${nm(1)}` : ''}｜{meta.cluster_name}｜⚡{((meta.energy?.micro || 0) * 100).toFixed(0)}%
    </div>
  )
}

export function PerReqPanel({ perReq, setPerReq, cores, clusters, onPrefix, t, lang }) {
  const p = perReq
  const set = (k, v) => setPerReq({ ...p, [k]: v })
  const temp = p.routing_temperature ?? 1.0
  const clanName = c => `${c.id}·${lang === 'en' ? c.en : c.zh}`
  return (
    <div className="panel">
      <h3>⚡ {t.onceTitle}</h3>
      <div className="row">
        <button onClick={() => set('armed', !p.armed)} className={'tgl' + (p.armed ? ' on' : '')}
                style={{ fontSize: 12, padding: '5px 14px' }}>
          {p.armed ? t.armedOn : t.armedOff}
        </button>
      </div>
      <div className={'once-body' + (p.armed ? '' : ' dim')}>
        <div className="row">
          🌡 {t.temp}: <b style={{ color: '#fff' }}>{Number(temp).toFixed(1)}</b>
          <input type="range" min="0.1" max="2.5" step="0.1" value={temp}
                 onChange={e => set('routing_temperature', parseFloat(e.target.value))} />
        </div>
        <div className="row">
          🎯 {t.forceMacro}:
          <select value={p.force_macro} onChange={e => set('force_macro', parseInt(e.target.value, 10))}
                  style={{ background: '#2f2f2f', color: '#ececec', border: '1px solid #424242',
                    borderRadius: 6, marginLeft: 6, fontSize: 12 }}>
            <option value={-1}>{t.forceAuto}</option>
            {(cores || []).map(c => (
              <option key={c.id} value={c.id}>{c.id}·{lang === 'en' ? c.en : c.zh}</option>
            ))}
          </select>
        </div>
        <div className="row">
          📏 {t.maxTok}:
          <input type="number" min="8" max="1024" value={p.max_tokens}
                 onChange={e => set('max_tokens', Math.min(1024, Math.max(8, parseInt(e.target.value, 10) || 128)))}
                 style={{ width: 64, background: '#2f2f2f', color: '#ececec', border: '1px solid #424242',
                   borderRadius: 6, padding: 3, fontSize: 11, marginLeft: 6 }} />
        </div>
        <div className="row">{t.disableMacros}:
          <div style={{ marginTop: 4 }}>
            {(cores || []).map(c => {
              const off = (p.disabled_macros || []).includes(c.id)
              return (
                <span key={c.id} className={'kill' + (off ? ' off' : '')}
                  onClick={() => {
                    const cur = p.disabled_macros || []
                    set('disabled_macros', off ? cur.filter(x => x !== c.id) : [...cur, c.id])
                  }}>
                  {c.id}·{lang === 'en' ? c.en : c.zh}
                </span>
              )
            })}
          </div>
        </div>
        <div className="row">{t.disableClusters}:
          <div style={{ marginTop: 4 }}>
            {(clusters || []).map(cl => {
              const off = (p.disabled_clusters || []).includes(cl.id)
              return (
                <span key={cl.id} title={cl.en} className={'kill' + (off ? ' off' : '')}
                  onClick={() => {
                    const cur = p.disabled_clusters || []
                    set('disabled_clusters', off ? cur.filter(x => x !== cl.id) : [...cur, cl.id])
                  }}>
                  {clanName(cl)}
                </span>
              )
            })}
          </div>
        </div>
        {[['disable_micro', t.swMicro], ['disable_macro', t.swMacro],
          ['disable_shared', t.swShared], ['disable_all_moe', t.swBase]].map(([k, label]) => (
          <div key={k} className="row">
            {label}
            <button onClick={() => set(k, !p[k])} className={'tgl' + (p[k] ? ' on' : '')}>
              {p[k] ? t.off : t.on}
            </button>
          </div>
        ))}
        <div className="row">{t.prefixes}:
          <div style={{ marginTop: 4 }}>
            {['/force-code', '/no-micro', '/base-only', '/cold', '/wild'].map(px => (
              <span key={px} className="kill" onClick={() => onPrefix(px)}>{px}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function IntentBox({ t }) {
  const [q, setQ] = React.useState('')
  const [res, setRes] = React.useState(null)
  const [busy, setBusy] = React.useState(false)
  async function go() {
    if (!q.trim() || busy) return
    setBusy(true); setRes(null)
    try { setRes(await (await import('./api.js')).analyzeIntent(q.trim())) }
    catch (e) { setRes({ error: String(e.message || e) }) }
    setBusy(false)
  }
  return (
    <div className="panel">
      <h3>🔍 {t.intentTitle}</h3>
      <div className="swap-row">
        <input value={q} onChange={e => setQ(e.target.value)}
               onKeyDown={e => e.key === 'Enter' && go()} placeholder={t.intentPh} />
        <button className="tgl" onClick={go}>{busy ? '…' : t.intentBtn}</button>
      </div>
      {res && (res.error
        ? <div className="row" style={{ color: '#f88' }}>{res.error}</div>
        : <div className="row">
            {t.macroLabel}: <b style={{ color: '#fff' }}>{res.macro_name}</b> {(res.macro_conf * 100).toFixed(0)}%
            <br />{t.clanLabel}: <b style={{ color: '#fff' }}>{res.cluster_name}</b> {(res.cluster_conf * 100).toFixed(0)}%
            <br /><span style={{ color: '#8e8e8e' }}>{res.ms}ms · {t.intentNote}</span>
          </div>)}
    </div>
  )
}

export function BadcasePanel({ cases, total, onRefresh, t }) {
  return (
    <div className="panel">
      <h3>📝 {t.badTitle} ({total})
        <button className="tgl" onClick={onRefresh} style={{ marginLeft: 8 }}>↻</button>
      </h3>
      {(cases || []).length === 0 && <div className="row">{t.badEmpty}</div>}
      {(cases || []).map((c, i) => (
        <div key={i} className="row" style={{ borderTop: '1px solid #333', paddingTop: 6 }}>
          <div style={{ color: '#ececec' }}>❓ {(c.prompt || '').slice(0, 60)}</div>
          <div>✖ {t.macroLabel}{c.pred_macro ?? '?'} → ✔ {t.macroLabel}{c.correct_macro}
            {c.note ? <span style={{ color: '#8e8e8e' }}> · {c.note}</span> : null}</div>
        </div>
      ))}
    </div>
  )
}

export function GatewayPanel({ t }) {
  const copy = (s) => { try { navigator.clipboard?.writeText(s).catch(() => {}) } catch (e) { /* ignore */ } }
  const rows = [
    [t.gwBase, 'http://localhost:8000/v1'],
    [t.gwModel, 'myriad-moe-v3'],
    [t.gwKey, '(empty)'],
  ]
  return (
    <div className="panel">
      <h3>🔌 {t.gwTitle}</h3>
      {rows.map(([k, v], i) => (
        <div key={i} className="row">
          {k}: <code className="codebox">{v}</code>
          <button className="tgl" onClick={() => copy(v)}>⧉</button>
        </div>
      ))}
      <div className="row" style={{ lineHeight: 1.7 }}>{t.gwNote}</div>
    </div>
  )
}

export function SteeringPanel({ steer, cores, clusters, onPatch, t, lang }) {
  if (!steer) return <div className="panel"><h3>{t.steerTitle}</h3><div className="row">{t.steerLoading}</div></div>
  const temp = steer.routing_temperature ?? 1.0
  const toggle = (k) => onPatch({ [k]: !steer[k] })
  const tempLabel = temp <= 0.3 ? t.tempCold : temp >= 2.0 ? t.tempWild : t.tempMid
  const clanName = c => `${c.id}·${lang === 'en' ? c.en : c.zh}`
  const toggleClan = (id) => {
    const cur = steer.disabled_clusters || []
    onPatch({ disabled_clusters: cur.includes(id) ? cur.filter(x => x !== id) : [...cur, id] })
  }
  return (
    <div className="panel">
      <h3>🎛 {t.steerTitle}</h3>
      <div className="row">
        🌡 {t.temp}: <b style={{ color: '#fff' }}>{Number(temp).toFixed(1)}</b> {tempLabel}
        <input type="range" min="0.1" max="2.5" step="0.1" value={temp}
               onChange={e => onPatch({ routing_temperature: parseFloat(e.target.value) })} />
      </div>
      <div className="row">
        {t.disableMacros}:
        <div style={{ marginTop: 4 }}>
          {(cores || []).map(c => {
            const off = (steer.disabled_macros || []).includes(c.id)
            return (
              <span key={c.id} title={c.name} className={'kill' + (off ? ' off' : '')}
                onClick={() => {
                  const cur = steer.disabled_macros || []
                  onPatch({ disabled_macros: off ? cur.filter(x => x !== c.id) : [...cur, c.id] })
                }}>
                {c.id}·{lang === 'en' ? c.en : c.zh}
              </span>
            )
          })}
        </div>
      </div>
      <div className="row">
        {t.disableClusters}:
        <div style={{ marginTop: 4 }}>
          {(cores || []).map(c => (
            <div key={c.id} style={{ marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: '#8e8e8e', marginRight: 4 }}>
                {c.id}·{lang === 'en' ? c.en : c.zh}
              </span>
              {(clusters || []).filter(cl => cl.parent === c.id).map(cl => {
                const off = (steer.disabled_clusters || []).includes(cl.id)
                return (
                  <span key={cl.id} title={cl.en} className={'kill' + (off ? ' off' : '')}
                        onClick={() => toggleClan(cl.id)}>
                    {clanName(cl)}
                  </span>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      {[
        ['disable_micro', t.swMicro], ['disable_macro', t.swMacro],
        ['disable_shared', t.swShared], ['disable_all_moe', t.swBase],
      ].map(([k, label]) => (
        <div key={k} className="row">
          {label}
          <button onClick={() => toggle(k)} className={'tgl' + (steer[k] ? ' on' : '')}>
            {steer[k] ? t.off : t.on}
          </button>
        </div>
      ))}
    </div>
  )
}

export function ArchPanel({ probe, probing, t, lang }) {
  if (probing) return <div className="panel"><div className="row typing">{t.probing}</div></div>
  if (!probe) return <div className="panel"><div className="row">{t.waiting}</div></div>
  const mMax = Math.max(...probe.macro_hist, 0.01)
  const cMax = Math.max(...probe.cluster_hist, 0.01)
  const clanName = i => lang === 'en' ? `${i}·${CLAN_EN[i]}` : `${i}·${probe.cluster_names[i]}`;
  return (
    <div className="panel">
      <h3 className="probe-h">{t.macro}</h3>
      {probe.macro_hist.map((v, i) => (
        <Bar key={i} label={coreName(probe.core_names[i], lang, i)} value={v} max={mMax} color="#19c37d" />
      ))}
      <h3 className="probe-h">{t.clan}</h3>
      {probe.cluster_hist.map((v, i) => (
        <Bar key={i} label={clanName(i)} value={v} max={cMax} color="#7cc4ff" />
      ))}
      <h3 className="probe-h">{t.energy}</h3>
      <Bar label="shared" value={probe.energy.shared} max={1} color="#b18cff" />
      <Bar label="macro" value={probe.energy.macro} max={1} color="#19c37d" />
      <Bar label="micro" value={probe.energy.micro} max={1} color="#7cc4ff" />
      {probe.domain_hit != null && (
        <p className="row">{t.hit}：<b style={{ color: '#fff' }}>{(probe.domain_hit * 100).toFixed(1)}%</b>
          {probe.domain_hit > 0.3 ? t.aligned : t.misaligned}</p>
      )}
      <h3 className="probe-h">{t.bnorms}</h3>
      <table className="bnorm">
        <thead><tr><th>#</th><th>shared</th><th>macro</th><th>micro</th></tr></thead>
        <tbody>
          {probe.b_norms.map(r => (
            <tr key={r.layer}><td>L{r.layer}</td><td>{r.shared.toFixed(2)}</td>
              <td>{r.macro.toFixed(2)}</td><td>{r.micro.toFixed(2)}</td></tr>
          ))}
        </tbody>
      </table>
      <h3 className="probe-h">{t.tokens}</h3>
      {probe.tokens.map((t2, i) => (
        <div key={i} className="tok-line">
          [{i}] {JSON.stringify(t2.tok)} → macro{t2.macro.map((m, k) => `${m}:${t2.macro_w[k]}`).join(' ')}｜clan{t2.cluster.join(',')}
        </div>
      ))}
    </div>
  )
}
