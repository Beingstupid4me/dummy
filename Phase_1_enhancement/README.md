# Phase 1 enhancement

Time-first follow-on to `Phase_1_stable`. **Does not edit the locked baseline.** The FastAPI registry loads `models/enhance_m1.joblib` as slot M1.

Shared reconstruction code lives in `backend/app/ml/pipeline.py` (imported from here). Do not copy a second pipeline into this folder.

## Scorecard (step5 `F3888` windows)

Same protocol as the locked baseline and the PS2 report: sort by account open date; `split = 0.8n + (fold-2)*0.04n` clamped to `[0.5n, 0.95n]`; train `[:split]`, val `[split : split+0.2n]`. Ranking metric is the **raw** 0.6/0.4 blend PR-AUC (PCHIP is the probability map, not the ranker).

| | Chrono PR-AUC |
|---|---|
| PS2 report | 0.710 |
| Locked `Phase_1_stable` | **0.738** |
| This folder (best honest, 500 trees) | **0.732** |
| Future-core 70→90% (exported bundle) | **0.766** |
| Beats PS2 | yes |
| Beats locked stable | no (0.006 short; fold 4 has 3 mules) |

Do not quote shuffled ~0.87. Do not quote PR-AUC 1.0 — that was `F2230` leaking through cyclical parsing of `Sep25`/`Nov25`.

Confirmed out of the clean spec: `F3912`, `F2230`, resolution `F3913–F3915`, post-alert block `F3895–F3923` unless `include_tms`. No `elapsed_days`. No SMOTE.

## Dictionary tracks vs locked baseline

On top of locked `V_cross` / `txn_accel` / `F3889_comp_lag`, from `Description.xlsx`:

- POS / NET / CASH debit vs UPI credit (L7)
- L7−L31 burst on UPI credit, ATM debit, ELEC debit
- Channel-mix entropy and UPI debit–credit imbalance
- Occupation-relative balances `F3880–F3885`
- Fold-safe cats `F3886` / `F3889` / `F3891` / `F3892` (occupation stays categorical, not numeric)
- Cyclical **F3888 only**, Isolation Forest (normals), PCA-3 + KMeans-3, 14 row moments

## Scripts

```powershell
# from repo root, using backend venv (has xgboost / lightgbm)
.\backend\.venv\Scripts\python Phase_1_enhancement\run_enhance.py
.\backend\.venv\Scripts\python Phase_1_enhancement\run_ideas.py
```

| File | Role |
|---|---|
| `run_enhance.py` | 5-fold chrono at 500 trees; writes `metrics.json`, `scorecard.json`, `models/enhance_m1.joblib` |
| `run_ideas.py` | faster bake-off (280 trees): curvature, `V_cross` percentile, `V_cross × txn_accel`, no-PCA, eval-only blends |
| `metrics.json` | last `run_enhance` fold table |
| `scorecard.json` | headline numbers for the report |
| `ideas.json` | last idea bake-off (did **not** beat 0.738; bundle not overwritten) |

Latest `run_ideas.py` (280 trees, not comparable 1:1 to the 500-tree locked run): no-PCA mean chrono PR-AUC **0.727** (beats PS2, still under 0.738). Rank-blend and 0.7/0.3 were indistinguishable from 0.6/0.4. Extra tracks were **not** written into `enhance_m1.joblib`.
