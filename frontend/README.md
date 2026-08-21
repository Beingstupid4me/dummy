# SentinelFlow SecOps Console

Modern, high-performance cybersecurity operations console built with **Next.js 16 (Turbopack) / React 19 / Tailwind CSS** matching the institutional palette of Bank of India (Navy `#1D3557`, Slate `#457B9D`, Crimson `#E63946`, Foam `#F1FAEE`).

## Console Architecture & Views

### 1. Threat Dashboard (`/`)
- **Live Transaction Stream**: Real-time SSE streaming table with full-text search (account, ID, beneficiary) and multi-attribute channel/status filtering.
- **Incident Inspector**: Focus card showing calibrated fraud risk percentage, transaction breakdown, latency metrics, and TMS flag alerts.
- **HITL Priority Queue**: Human-in-the-loop review queue highlighting non-clear and escrow-quarantined accounts.
- **Interactive Simulation Sandbox**: Form to submit real-time test transactions against any account via `POST /score`.

### 2. Incident Dossier & Case Analysis (`/analysis/[id]`)
- **Calibrated Risk Gauge**: Circular SVG meter showing mathematically calibrated posterior probability $P_{\text{calib}}$.
- **Taylor-Scaled TreeSHAP Explanations**: Local feature attribution reason codes scaled by the continuous monotonic PCHIP derivative.
- **Ego-Network Graph Decomposition**: Interactive 2D canvas displaying 1-hop and 2-hop transaction layering topology with 24h / 7d TTL edge indicators.
- **Redis Profile Rollup**: Real-time $L7D$ channel debit/credit balances and velocity acceleration factors.
- **Latency Waterfall**: Microsecond-accurate latency breakdown (Redis, Reconstruction, GBDT, PCHIP, SHAP).
- **I4C / NCRP Intelligence Ingestion**: Live regulatory ticket viewer with integrated form to ingest new blacklist entities.
- **Autonomous Escrow Outbound Webhooks**: Audit stream of 15-minute escrow hold events and dispatch status.

### 3. Model Registry & Orchestrator (`/registry`)
- **4-Slot Memory Registry**: Live performance scorecard (PR-AUC, ROC-AUC, Mule F1, Macro F1, p99 latency, memory footprint).
- **Zero-Downtime Hot-Swap**: Click any active slot (M1, M2, M3, M4) to execute an instantaneous pointer cutover with 0 ms downtime.
- **Feature Composition Board**: Toggle individual production and audited leaky tracks fed into retrain requests.
- **Automated Retrain Pipeline**: Trigger on-demand GBDT retraining with SMOTE control, 90-day label-delay enforcement, and live SSE event stream logs.

## Running the Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### API Proxy Configuration
`next.config.ts` automatically proxies `/sf-api/:path*` to `http://127.0.0.1:8000/:path*`.
- When the backend is online, the console operates in **Live API** mode.
- If the backend is offline, the console seamlessly falls back to a deterministic **Mock Stream** (`lib/demo.ts`) to maintain complete demo functionality.

## Project Structure

```
app/
  page.tsx                 Dashboard view route
  analysis/[id]/page.tsx   Case analysis dossier route
  registry/page.tsx        Model registry route
  layout.tsx               Root layout, fonts, and dark theme
  globals.css              Custom styling & responsive scrollbars
components/
  shell.tsx                Institutional header, status badge, and routing tabs
  ui.tsx                   Panels, KPI cards, status pills, risk meter, TreeSHAP list
  ego-graph.tsx            2D Canvas ego-network topology renderer
  views/                   DashboardView, AnalysisView, RegistryView
lib/
  store.tsx                Global React Context, SSE listeners, polling fallbacks
  api.ts                   Typed FastAPI client & REST/SSE endpoint bindings
  demo.ts                  Deterministic simulation dataset & mock stream generator
  types.ts                 TypeScript data schemas
```
