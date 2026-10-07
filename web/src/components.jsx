import React from 'react'

export function Bar({ label, value, max = 1, color = '#4f8cff' }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '2px 0' }}>
      <div style={{ width: 110, fontSize: 12, color: '#aaa', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</div>
      <div style={{ flex: 1, height: 10, background: '#222', borderRadius: 4 }}>
        <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <div style={{ width: 52, fontSize: 12, textAlign: 'right', color: '#ddd' }}>{(value * 100).toFixed(1)}%</div>
    </div>
  )
}

export function ArchPanel({ probe, probing }) {
  if (probing) return <div style={{ color: '#888' }}>路由探测中…（一次prefill前向）</div>
  if (!probe) return <div style={{ color: '#666' }}>发送一条消息后，这里会显示8天王×16宗门的路由决策。</div>
  const mMax = Math.max(...probe.macro_hist, 0.01)
  const cMax = Math.max(...probe.cluster_hist, 0.01)
  return (
    <div>
      <h3>八大天王 Top1（response段聚合）</h3>
      {probe.macro_hist.map((v, i) => (
        <Bar key={i} label={`${i}·${probe.core_names[i]}`} value={v} max={mMax} color="#4f8cff" />
      ))}
      <h3>十六宗门 Top1</h3>
      {probe.cluster_hist.map((v, i) => (
        <Bar key={i} label={`${i}·${probe.cluster_names[i]}`} value={v} max={cMax} color="#38c172" />
      ))}
      <h3>三分支能量 / 基座</h3>
      <Bar label="shared主宰" value={probe.energy.shared} max={1} color="#b18cff" />
      <Bar label="macro天王" value={probe.energy.macro} max={1} color="#4f8cff" />
      <Bar label="micro微核" value={probe.energy.micro} max={1} color="#38c172" />
      {probe.domain_hit != null && (
        <p>期望领域命中率：<b>{(probe.domain_hit * 100).toFixed(1)}%</b>
          {probe.domain_hit > 0.3 ? '（对齐）' : '（未对齐/待训）'}</p>
      )}
      <h3>LoRA分支B范数（0=没学到）</h3>
      <table style={{ fontSize: 12, borderCollapse: 'collapse' }}>
        <thead><tr><th>层</th><th>shared</th><th>macro</th><th>micro</th></tr></thead>
        <tbody>
          {probe.b_norms.map(r => (
            <tr key={r.layer}><td>L{r.layer}</td><td>{r.shared.toFixed(2)}</td>
              <td>{r.macro.toFixed(2)}</td><td>{r.micro.toFixed(2)}</td></tr>
          ))}
        </tbody>
      </table>
      <h3>逐token路由（中间层，前12个response token）</h3>
      {probe.tokens.map((t, i) => (
        <div key={i} style={{ fontSize: 12, fontFamily: 'monospace', color: '#ccc' }}>
          [{i}] {JSON.stringify(t.tok)} → 天王{t.macro.map((m, k) => `${m}:${t.macro_w[k]}`).join(' ')}｜宗门{t.cluster.join(',')}
        </div>
      ))}
    </div>
  )
}
