# SentinelFlow

Cybershield Hackathon 2026 — Bank of India mule-account detection (`F3924`). Phase 1 is an offline GBDT recipe. Phase 2 is a live `/score` engine plus a Next.js SecOps console.

Do **not** edit `Phase_1/` or `Phase_1_stable/`. New modelling lives in `Phase_1_enhancement/`. The engine and console live in `backend/` and `frontend/`.

## Honest scores (quote these)

Same step5 chronological windows on account open date `F3888`. Raw 0.6 XGB + 0.4 LGB blend. No SMOTE. No `elapsed_days` on the production model.

| Split | PR-AUC | Notes |
|---|---|---|
| PS2 report chrono | 0.710 | paper ranking protocol |
| Locked `Phase_1_stable` | **0.738** | production baseline |
| `Phase_1_enhancement` chrono | **0.732** | beats PS2, 0.006 under locked |
| Enhancement future-core (70→90%) | **0.766** | nested mule F1 0.79 |

Do not quote shuffled ~0.87 or PR-AUC 1.0. Those were `F2230` (`Sep25`/`Nov25`) leaking through cyclical date parsing. Confirmed leaks stay out of M1: `F3912`, `F2230`, resolution flags `F3913–F3915`.

A live `/score` of ~0.995 on sample `XXXX2203` is a known mule + I4C hit, not a model metric.

## Layout

```
DataSet.csv                  9082 accounts × 3925 cols (81 mules)
Description.xlsx             bank dictionary (sheet Data_Dicitionary)
Phase_1_stable/              locked recipe — do not edit
Phase_1_enhancement/         dictionary tracks + chrono bake-off
backend/                    FastAPI /score /retrain /registry
frontend/                    Next.js 16 console (Dashboard, Analysis, Registry)
phase2_roadmap.md            original Streamlit plan (console is Next.js in the paper)
pro_report.tex / .pdf        Phase 2 write-up
```

## Run the prototype

Two processes. Do **not** use `uvicorn --reload` on Windows — WatchFiles hits locked venv files (`WinError 32`).

```powershell
# API  —  http://127.0.0.1:8000/docs
cd backend
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m app.bootstrap
.\.venv\Scripts\uvicorn app.main:app --port 8000

# Console  —  http://localhost:3000
cd frontend
npm install
npm run dev
```

`bootstrap` copies `Phase_1_enhancement/models/enhance_m1.joblib` into registry slot **M1**. M2 (elapsed_days) and M3 (leaky demo) need `python -m app.bootstrap --all` and take several minutes.

The console proxies `/sf-api/*` to `:8000`. If the API is down it falls back to a mock stream so the UI still demo.

## What “a mule” is in this data

`DataSet.csv` is **one row per account**, not a raw payment log. L7 / L14 / L31 columns already summarize that account’s history. Chrono splits are by `F3888` (open date). Live scoring overlays the new payment onto the stored L7 profile (Redis or in-memory), then runs the GBDT blend + PCHIP.

## Docs per component

- [backend/README.md](backend/README.md) — FastAPI surface, reconstruction, registry, Redis
- [frontend/README.md](frontend/README.md) — console pages, API wiring, mock fallback
- [Phase_1_enhancement/README.md](Phase_1_enhancement/README.md) — chrono protocol and scorecard

Remaining work vs `pro_report.pdf` / `phase2_roadmap.md` is in [remaining.md](remaining.md). Component READMEs also have a **Gaps** section.
