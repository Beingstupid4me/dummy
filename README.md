# SentinelFlow

Cybershield Hackathon 2026 — Bank of India Mule Account Detection & Autonomous Real-Time Intervention (`F3924`).

SentinelFlow is an end-to-end AI/ML detection and decision-support architecture featuring an offline GBDT + PCHIP feature engineering pipeline (**Phase 1.2**), a high-throughput sub-100ms FastAPI scoring engine (**Backend**), and an institutional SecOps operations console built in Next.js 16 (**Frontend**).

---

## Benchmark Performance Highlights

Evaluated across the exact step 5 chronological rolling forward-windows on account-open date `F3888` (zero target leakage):

| Component / Metric | PS2 Report (Paper) | Locked Phase 1 Stable | Phase 1.2 (Our Final Solution) | Operational Gain |
|---|---|---|---|---|
| **Chronological PR-AUC** | `0.7097` | `0.7381` | **`0.8111`** (mean) / **`0.8508`** (holdout) | **+0.0730 to +0.1014 PR-AUC** |
| **Chronological ROC-AUC** | `0.8552` | `0.8671` | **`0.9902`** | **+0.1231 ROC-AUC** |
| **Honest Mule F1** | `0.3941` | `0.6696` | **`0.7924`** | **+0.1228 F1 score** |
| **Macro F1** | `0.6958` | `0.8337` | **`0.8956`** | High class separation |
| **Recall @ Top 1% Alert Queue** | `0.6955` | `0.6955` | **`0.7677`** | **76.8% of all mules captured in top 1%** |
| **Recall @ Top 5% Alert Queue** | `0.8081` | `0.8081` | **`0.8970`** | **89.7% of all mules captured in top 5%** |
| **Operational Review Cost ($)** | `$43.00` | `$43.00` | **`$18.60`** | **56.7% operational cost reduction** |
| **p99 Scoring Latency** | `< 100.0 ms` | — | **`46.86 ms`** | **SLA passed (p50: 14.53 ms)** |

*Strict Zero-Leakage Policy: `F3912`, `F2230`, and post-alert status variables are excluded from production slots.*

---

## Repository Structure

```
├── DataSet.csv                      9,082 accounts × 3,925 variables (81 verified mules)
├── Description.xlsx                 Bank variable dictionary (Data_Dicitionary)
│
├── Phase_1.2/                       ★ COMPLETE REPRODUCIBLE ML SOLUTION
│   ├── SentinelFlow_v2.ipynb        End-to-end Master Jupyter Notebook (Cleansing → Model → Scorecard)
│   ├── data_layer.py                Cached data ingestion & Description.xlsx grammar parser
│   ├── semantic.py                  Domain-specific mule typology feature factory
│   ├── advanced.py                  Peer-cohort normalization & prototype geometry
│   ├── experiment.py                Model ablation harness & chronological walk-forward
│   ├── scorecard.py                 Multi-objective scorecard (PR-AUC, F1, ECE, Brier, Cost)
│   ├── cost.py                      Asymmetric banking loss optimizer ($R \in [2, 100]$)
│   ├── run_bakeoff.py               Staged ablation runners (features, capacity, combo, calibration)
│   ├── audit_leakage.py             Empirical target leakage & purity audit
│   ├── audit_timeaxis.py            Observation window & chronological cohort audit
│   └── results/                     Final ablation logs and scorecards
│
├── backend/                         ★ HIGH-THROUGHPUT FASTAPI INFERENCE ENGINE
│   ├── app/
│   │   ├── main.py                  ASGI entrypoint with startup matrix pre-warming
│   │   ├── api.py                   REST endpoints & SSE streams (/score, /retrain, /registry)
│   │   ├── config.py                Runtime settings (Redis, webhooks, SLAs)
│   │   ├── bootstrap.py             Pre-trains & seeds registry slots (M1, M2, M3)
│   │   └── services/                Scoring, registry, graph, TMS rules, webhooks, store
│   ├── benchmark.py                 N=10,000 /score latency distribution benchmark harness
│   └── registry/                    M1.joblib, M2.joblib, M3.joblib, M4.joblib (<5 MB each)
│
├── frontend/                        ★ INSTITUTIONAL SECOPS CONSOLE (NEXT.JS 16)
│   ├── app/                         Dashboard (/), Analysis (/analysis/[id]), Registry (/registry)
│   ├── components/views/            DashboardView, AnalysisView, RegistryView
│   ├── components/                  Ego-network canvas, risk meter, TreeSHAP, KPI cards
│   └── lib/                         React Context, SSE listeners, REST API client, mock fallback
│
├── Phase_1_stable/                  Locked baseline reference
├── Phase_1_enhancement/             Exploration reference
├── README.md                        Global system documentation
└── remaining.md                     Roadmap & feature coverage verification
```

---

## Quickstart: Running the System

### 1. Execute ML Pipeline & Master Notebook
Open and run `Phase_1.2/SentinelFlow_v2.ipynb` in any Jupyter environment. It loads `DataSet.csv`, reconstructs the feature typology, trains the regularized GBDT ensemble, calibrates with PCHIP splines, evaluates the economic loss engine, and exports the bundle.

### 2. Start the Backend API (FastAPI)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m app.bootstrap --all --trees 150
.\.venv\Scripts\uvicorn app.main:app --port 8000
```
- API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Run Latency Benchmark: `.\.venv\Scripts\python benchmark.py --n 10000`

### 3. Start the SecOps Console (Next.js)
```powershell
cd frontend
npm install
npm run dev
```
- Operations Console: [http://localhost:3000](http://localhost:3000)

---

## Detailed Documentation per Component

- [Phase_1.2/README.md](Phase_1.2/README.md) — Comprehensive ML research paper, feature factory breakdown, and chronological audit.
- [backend/README.md](backend/README.md) — FastAPI engine, TMS rules, escrow webhooks, registry hot-swap, and latency benchmark results.
- [frontend/README.md](frontend/README.md) — Next.js 16 SecOps interface, real-time SSE stream, case analysis, and ego-graph topology.
- [remaining.md](remaining.md) — Complete specification coverage checklist against `pro_report.pdf` & `phase2_roadmap.md`.
