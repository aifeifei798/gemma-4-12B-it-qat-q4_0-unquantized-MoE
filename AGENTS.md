# AGENTS.md — Myriad-MoE repo conventions (for coding agents)

## What this is
Hierarchical MoE (1 shared + 8 macro + 16×16 micro, layers 18–29) grafted on
frozen Gemma-4-12B (4-bit NF4). FastAPI backend (`server.py`) + pnpm/Vite/React
chat UI (`web/`). Single-user local rig with a 24G card that also drives the
desktop (Xorg on GPU0 — respect the display watchdog).

## Never do
- Never commit `*.pt` / checkpoints / caches / `.venv` / `node_modules` /
  `web/dist` (see `.gitignore`). Weights travel via external storage, not git.
- Never rewrite git history (timestamps matter for prior-art disclosure).
- Never `pkill -f <literal-filename>` from a shell whose own command line
  contains that filename — it kills your own shell. Kill by PID from `pgrep`.
- Never run two `server.py` instances (port + GPU collision). Check
  `pgrep -af "server\.py"` first; old process may ignore SIGTERM, use -9.
- Don't touch `train/` and `bak_v22/` (stale snapshots, git-ignored).

## Before changing routing math
- Steering defaults must stay identity (T=1.0, no masks, all branches on) so
  training (`2.train_*.py`, `4.micro_patch.py`) is unaffected.
- `diagnose_layer` in `3.infer_probe.py` must mirror `forward` exactly —
  panel numbers must equal generation behavior. CPU-verify with tiny dims
  before pushing (see README quickstart patterns).
- Names come only from `domains.yaml` (`3.infer_probe` loads it at import).
  Never hardcode core/clan names in backend or frontend.

## Server workflow
- After editing `server.py`: `py_compile`, rebuild web if touched
  (`cd web && pnpm build`), restart server, poll `GET /api/health` till ok.
- Keep one server on :8000. Generation/probe/hotswap serialize on
  `_MODEL_LOCK` — concurrent smoke tests will queue, that's expected.
- Smoke tests: `/api/chat` (SSE), `/v1/chat/completions` non-stream,
  `/admin/steering` set + reset, `/data/mark-badcase` writes must be cleaned
  up afterwards (don't pollute `hard_cases_v4.jsonl`).

## Frontend
- JS (not TS), no component lib — plain CSS in `web/src/styles.css`.
  All strings via `web/src/i18n.js` (en + zh), default en.
- After UI changes: `pnpm build` and grep `dist/assets/*.js` for new strings.
- Screenshots: playwright + system Chrome against localhost:8000 into
  `images/ui-*.png` only.

## Docs
- README numbers must be measured, not invented (arch params, VRAM, val acc).
- `## Public Prior Art & Disclosures`: technical record only, not legal
  advice; corrections welcome via issues.
