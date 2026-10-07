import React from 'react'
import { CLAN_EN, coreName } from './i18n.js'

export function Bar({ label, value, max = 1, color = '#4f8cff' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '2px 0' }}>
      <div style={{ width: 150, fontSize: 12, color: '#aaa', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
      <div style={{ flex: 1, height: 10, background: '#222', borderRadius: 4 }}>
        <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <div style={{ width: 52, fontSize: 12, textAlign: 'right', color: '#ddd' }}>{(value * 100).toFixed(1)}%</div>
    </div>
  )
}

export function ArchPanel({ probe, probing, t, lang }) {
  if (probing) return <div style={{ color: '#888' }}>{t.probing}</div>
  if (!probe) return <div style={{ color: '#666' }}>{t.waiting}</div>
  const mMax = Math.max(...probe.macro_hist, 0.01)
  const cMax = Math.max(...probe.cluster_hist, 0.01)
  const clanName = i => lang === 'en' ? `${i}·${CLAN_EN[i]}` : `${i}·${probe.cluster_names[i]}`;
  return (
    <div>
      <h3>{t.macro}</h3>
      {probe.macro_hist.map((v, i) => (
        <Bar key={i} label={coreName(probe.core_names[i], lang, i)} value={v} max={mMax} color="#4f8cff" />
      ))}
      <h3>{t.clan}</h3>
      {probe.cluster_hist.map((v, i) => (
        <Bar key={i} label={clanName(i)} value={v} max={cMax} color="#38c172" />
      ))}
      <h3>{t.energy}</h3>
      <Bar label="shared" value={probe.energy.shared} max={1} color="#b18cff" />
      <Bar label="macro" value={probe.energy.macro} max={1} color="#4f8cff" />
      <Bar label="micro" value={probe.energy.micro} max={1} color="#38c172" />
      {probe.domain_hit != null && (
        <p>{t.hit}：<b>{(probe.domain_hit * 100).toFixed(1)}%</b>
          {probe.domain_hit > 0.3 ? t.aligned : t.misaligned}</p>
      )}
      <h3>{t.bnorms}</h3>
      <table style={{ fontSize: 12, borderCollapse: 'collapse' }}>
        <thead><tr><th>#</th><th>shared</th><th>macro</th><th>micro</th></tr></thead>
        <tbody>
          {probe.b_norms.map(r => (
            <tr key={r.layer}><td>L{r.layer}</td><td>{r.shared.toFixed(2)}</td>
              <td>{r.macro.toFixed(2)}</td><td>{r.micro.toFixed(2)}</td></tr>
          ))}
        </tbody>
      </table>
      <h3>{t.tokens}</h3>
      {probe.tokens.map((t2, i) => (
        <div key={i} style={{ fontSize: 12, fontFamily: 'monospace', color: '#ccc' }}>
          [{i}] {JSON.stringify(t2.tok)} → macro{t2.macro.map((m, k) => `${m}:${t2.macro_w[k]}`).join(' ')}｜clan{t2.cluster.join(',')}
        </div>
      ))}
    </div>
  )
}
