# Myriad-MoE: Hierarchical Fine-Grained MoE on Gemma-4-12B

A hierarchical Mixture-of-Experts grafted onto a frozen **Gemma-4-12B** backbone
(48 layers, hidden 3840): **1 shared sovereign + 8 macro cores + 16 clans × 16
micro-experts**, trained with supervised domain routing, 4-bit QLoRA-style
adapters, and a hot-swappable patch workflow. Includes a FastAPI backend and a
pnpm (Vite + React) chat UI with a live routing probe.

## Architecture

MoE adapters are grafted onto layers **18–29** (12 layers, **405.1M**
trainable params). The base model stays frozen (4-bit NF4).

```
token hidden state ──┬── Shared sovereign (Rank-32, always on)
                     ├── Macro router: Top-2 of 8 cores (Rank-16 each)
                     └── Clan router: Top-2 of 16 clans × per-clan Top-2 of 16 micro-experts (Rank-16)
                           joint weight = P(clan) · P(expert | clan)
output = base_mlp(x) + shared + macro + micro      # gamma = 1.0, LoRA alpha/r only
```

- **True Top-K sparsity**: unselected experts contribute exactly `0.0`
  (hard scatter mask, re-normalized Top-2 weights).
- **Load balancing**: Switch-style `density × p_mean` aux loss on all three
  routing levels (micro loss is computed per-clan, then averaged).
- **Supervised routing**: `core_id → macro (8-way)` + `cluster_id → clan
  (16-way)` cross-entropy on **response tokens only** (prompt templates are
  identical across domains and would otherwise feed contradictory labels).

## Data pipeline

| Version | Size | Design |
|---|---|---|
| v2 | 24k (8×3000, 16×1500) | Per-domain sources, but clans split by `idx % 2` (= random labels), cores 5/6/7 arbitrary alpaca slices |
| v2.4 | 24k, same shape | Semantic score-rank + fixed-quota clustering; no_robots split by `category`; alpaca split by constraint/logic keyword scores |
| **v3 (current)** | **28k (8×3500, 16×1750)** | Dedicated source per core; Chinese slots (Belle 0.5M CN); real refusal pairs for the safety clan (`mlabonne/harmful_behaviors` + fixed refusal templates) |

v3 core map: `0 Code · 1 Math · 2 Science · 3 Creative · 4 Dialogue ·
5 Logic · 6 Constraint · 7 Chinese-Slots`. All splits are quota-enforced
(`assert` locked) so the balance loss sees uniform classes.

## Training (`2.train_myriad_v2_24g_safe.py`)

- 4-bit NF4 frozen base + bf16 adapters + `PagedAdamW8bit`, `B=1 × ACCUM=16`,
  gradient checkpointing — peak **~13 GB**, fits 24 GB cards.
- Loss: `LM + 0.005·aux + 0.3·sup`, linear warmup (100 steps) + cosine decay
  to 3e-5 (`LambdaLR` — note: `SequentialLR` silently freezes on
  `load_state_dict` resume, hence the single-function scheduler).
- 240-sample held-out split with periodic `[val]` snapshots (LM / sup-acc).
- Long sequences are chunked (≤256 tokens, 1-token overlap) — a display
  watchdog (`Xid 8`) on desktop GPUs kills unsplit 448-token backward bursts.
- Deterministic seed + absolute batch offsets: crash-safe resume, multi-epoch
  rollover, emergency checkpoint on exception.

v3 result (5205 steps, 3 epochs): **val macro Top-1 66.2 %** (chance 12.5 %),
**clan Top-1 45.6 %** (chance 6.25 %), val LM 1.35.

## Inference, probe, serving

- `3.infer_probe.py` — generate + routing probe: per-layer LoRA B-norms
  (0 = dead branch), branch/base energy ratios, macro/clan histograms,
  per-token Top-2 decisions, domain hit-rate. `--base-only` compares against
  the frozen base to attribute quality issues.
- `server.py` — FastAPI: `POST /api/chat` (SSE stream),
  `POST /api/probe` (response-segment diagnostics), `POST /api/hotswap`
  (load new weights into the live server, no restart), `GET /api/health`.
- `web/` — pnpm + Vite + React chat UI with an architecture panel
  (`pnpm install && pnpm build`; dev via `pnpm dev` with `/api` proxy).

## Micro-patch workflow (`4.micro_patch.py`, `poetry_patch.jsonl`)

Targeted hotfix without full retraining, demonstrated on a real failure
(model called 白日依山尽 "a mountaineering technique"):

1. Write ~tens of Q&A pairs covering the failure mode.
2. Micro-fine-tune on v3 weights (73 samples, 15 epochs, lr 6e-5, ~10 min).
3. Verify offline (`--weight patch`), then `POST /api/hotswap` — 0.6 s,
   zero downtime, quicksort regression-checked.

## Repo layout

```
1.prepare_myriad_data.py   # v2.4 semantic data pipeline (offline cache)
1.prepare_myriad_v3.py     # v3 pipeline: dedicated sources + Chinese + refusals
2.train_myriad_v2_24g_safe.py  # training (DATA_TAG=v3)
3.infer_probe.py           # inference + routing probe
4.micro_patch.py           # targeted micro-fine-tune
poetry_patch.jsonl         # example patch data
server.py                  # FastAPI backend
web/                       # pnpm chat UI
```

Large artifacts (`*.pt` weights/checkpoints, the base model dir) are not
committed — use Git LFS or external storage. You need
`../gemma-4-12B-it-qat-q4_0-unquantized` (Gemma-4-12B tokenizer + weights)
next to this repo.

## Quickstart

```bash
# 1. data (v3)
python3 1.prepare_myriad_v3.py
# 2. train (~6 h on RTX 5090-class, resumes from checkpoint_v3.pt)
python3 -u 2.train_myriad_v2_24g_safe.py
# 3. probe
python3 3.infer_probe.py --prompt "..." --domain 0 --probe-only
# 4. serve + UI
python3 server.py &                 # :8000
cd web && pnpm install && pnpm build
# 5. targeted fix, then hot-swap (no restart)
python3 4.micro_patch.py --data poetry_patch.jsonl --out myriad_moe_patch_poetry.pt
curl -X POST localhost:8000/api/hotswap -H 'Content-Type: application/json' \
  -d '{"weight":"myriad_moe_patch_poetry.pt"}'
```
