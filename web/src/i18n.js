export const CLAN_EN = [
  'Syntax & Algorithms', 'Systems Engineering', 'Numerical Algebra', 'Word Problems',
  'Basic Physics & Chemistry', 'Experimental Science', 'Long-form Narrative', 'Poetry & Imagery',
  'Expert QA', 'Summarization', 'Causal Reasoning', 'Critical Thinking',
  'Format Compliance', 'Safety Refusals', 'Chinese Writing', 'Chinese QA',
]

const CORE_EN = ['Code', 'Math', 'Science', 'Creative_Arts', 'Business_Dialogue',
  'Logic_Philosophy', 'Constraint_Rules', 'Chinese_Slots']

export function coreName(raw, lang, i) {
  if (lang === 'en') {
    const m = /[A-Za-z_]+$/.exec(raw || '')
    return m ? `${i}·${m[0]}` : `${i}·${CORE_EN[i] || ''}`
  }
  return `${i}·${raw}`
}

export const STR = {
  en: {
    title: 'Myriad-MoE Chat', sub: '1 shared + 8 sovereigns + 16 clans × 16 micro-cores',
    probeTitle: 'Architecture Probe', send: 'Send', inputPh: 'Type a message…',
    waiting: 'Send a message to see Top-2 routing of the 8 sovereigns × 16 clans.',
    probing: 'Probing routes… (one prefill forward)',
    online: h => `● API online · VRAM ${h.vram_gb}G · ${h.layers.length} MoE layers`,
    offline: '● API not ready', connecting: '● Connecting to API…',
    macro: 'Macro Top-1 (response segment)', clan: 'Clan Top-1',
    energy: 'Branch energy / base', hit: 'Expected-domain hit rate', aligned: '(aligned)', misaligned: '(misaligned/training)',
    bnorms: 'LoRA branch B-norms (0 = learned nothing)', tokens: 'Per-token routing (mid layer, first 12 response tokens)',
    swapPh: 'patch weights path (server-local)', swapBtn: 'Hot-swap',
    swapping: 'Hot-swapping…', swapped: r => `Swapped to ${r.weight} (${r.secs}s, ${r.layers} layers)`,
    genFail: e => `Generation failed: ${e}`, probeFail: e => `Probe failed: ${e}`,
  },
  zh: {
    title: 'Myriad-MoE Chat', sub: '1共享 + 8天王 + 16宗门×16微核',
    probeTitle: '架构探针', send: '发送', inputPh: '输入消息…',
    waiting: '发送一条消息后，这里会显示8天王×16宗门的路由决策。',
    probing: '路由探测中…（一次prefill前向）',
    online: h => `● API在线 · 显存${h.vram_gb}G · ${h.layers.length}层MoE`,
    offline: '● API未就绪', connecting: '● 连接API…',
    macro: '八大天王 Top1（response段聚合）', clan: '十六宗门 Top1',
    energy: '三分支能量 / 基座', hit: '期望领域命中率', aligned: '（对齐）', misaligned: '（未对齐/待训）',
    bnorms: 'LoRA分支B范数（0=没学到）', tokens: '逐token路由（中间层，前12个response token）',
    swapPh: '补丁权重路径（服务端本地）', swapBtn: '热插拔',
    swapping: '热插拔中…', swapped: r => `已换装 ${r.weight}（${r.secs}s，${r.layers}层）`,
    genFail: e => `生成失败: ${e}`, probeFail: e => `探测失败: ${e}`,
  },
}
