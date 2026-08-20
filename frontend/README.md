# SentinelFlow console

Next.js operations console from `pro_report` (navy `#1D3557`, slate `#457B9D`, crimson `#E63946`). The paper says Next.js 14; this app is **Next.js 16 / React 19 / Tailwind 4**. Behaviour matches the paper’s three-pane SecOps console, not the older Streamlit sketch in `phase2_roadmap.md`.

## Pages

| Route | What it shows |
|---|---|
| `/` | Dashboard — live stream, HITL queue, `POST /score` simulator |
| `/analysis/[id]` | Case file — calibrated *P*, Taylor-scaled TreeSHAP, ego graph, Redis L7 profile, latency waterfall, I4C tickets, escrow webhook log |
| `/registry` | M1–M4 arena, feature control board, SMOTE slider, `POST /retrain` |

## Talk to the API

`next.config.ts` rewrites `/sf-api/:path*` → `http://127.0.0.1:8000/:path*`. Override with `NEXT_PUBLIC_API_URL` if the engine is elsewhere.

On boot the console:

1. `GET /health` and `GET /registry`
2. If that succeeds: `EventSource('/sf-api/stream')` and live `POST /score` / `POST /registry/active` / `POST /retrain`
3. If the API is down: seeded mock feed (`lib/demo.ts`) so the UI still demo

Header reads **API** when the engine is up, **Demo** otherwise.

`lib/store.tsx` is the single client store. `lib/api.ts` maps FastAPI JSON onto the same `Transaction` type the mock uses.

## Run

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Start the backend first if you want live scores (see `backend/README.md`). Do not start uvicorn with `--reload` on Windows.

Sample accounts in the simulator: `XXXX2203` (mule + seeded I4C), `XXXX2411`, `XXXX2688`, `XXXX2904`.

## Layout

```
app/page.tsx                 dashboard
app/analysis/[id]/page.tsx   case file
app/registry/page.tsx        orchestrator
components/views/            Dashboard / Analysis / Registry
components/ego-graph.tsx     2D canvas 1-hop / 2-hop (not WebGL)
components/shell.tsx         navy chrome, live/demo pill
lib/store.tsx                feed, score, swap, retrain
lib/api.ts                   FastAPI client + SSE mapper
lib/demo.ts                  mock stream if API is down
lib/types.ts                 shared shapes
```

## Gaps vs report / roadmap

Paper calls for libraries this app does **not** depend on:

- `react-virtuoso` virtualized feed — the table is a plain scroll
- `react-force-graph-2d` WebGL ego graph — `ego-graph.tsx` draws on a 2D canvas
- Retrain progress **SSE** — the console **polls** `GET /retrain/{job_id}` every 1.6 s

Other gaps:

- I4C / NCRP list on the analysis page is still the seeded `SEED_GOV` demo, not `GET /feeds/gov`.
- Feature toggles (moments, gaps, cats, leaky flags) change local state. Retrain currently sends only `include_elapsed`, `include_leaky`, and `smote_ratio` — not a full `features_on` list.
- No hyperparameter fields (learning rate) in the UI; trees are hardcoded to 120 in `lib/api.ts`.
- Registry hot-swap of M3 does nothing until the backend has M3 weights (`python -m app.bootstrap --all`).
