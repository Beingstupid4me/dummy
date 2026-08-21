# Phase 1.2 — improving every stage of the PS2 solution

Phase 1.2 rebuilds the pipeline described in `ps2_mule_account_detection_report.pdf` stage by
stage. It does **not** edit `Phase_1/`, `Phase_1_stable/` or `Phase_1_enhancement/`, and it does
not touch `backend/` or `frontend/` — this is model work only.

Every number here is measured on the **same step5 chronological windows** the report and the
locked baseline used (`split = 0.8n + (fold-2)·0.04n`, clamped to `[0.5n, 0.95n]`, ordered by
account-open date `F3888`), so a delta is a delta and not a change of protocol.

## Reference points

| | chrono PR-AUC | ROC-AUC | mule F1 | macro F1 | cost @R=5 | source |
|---|---|---|---|---|---|---|
| PS2 report | 0.7097 | 0.8552 | 0.8104 (shuf) | 0.9044 (shuf) | 3.40 (shuf) | recorded in the paper |
| Locked `Phase_1_stable` | 0.7381 | 0.8671 | 0.6696 | 0.8337 | - | recorded in `phase1_benchmark_metrics.json` |
| Locked recipe re-run in this harness | 0.7243 | 0.8672 | 0.3941 | 0.6958 | 43.0 (norm 0.754) | `results/final_recipe.json` |
| **Phase 1.2 (step5 chrono mean)** | **0.8111** | **0.9902** | **0.7924** | **0.8956** | **18.6 (norm 0.326)** | `results/final_recipe.json` |
| **Phase 1.2 (future-core holdout)** | **0.8508** | **0.9840** | **0.7857** | **0.8916** | **30.0 (norm 0.353)** | `results/final_recipe.json` |

The locked recipe reproduces to 0.7243 here rather than 0.7381 — same 157 features, same
hyperparameters, same folds, but a different library generation moves the mutual-information
selection on the noisiest fold. Phase 1.2 is therefore compared against **both**: it beats the
in-harness control by +0.087 and the recorded locked number by +0.073 on PR-AUC, while
more than doubling minority F1 and halving operational review cost.

## Final Head-to-Head Scorecard (Step 5 Chronological Windows)

| Metric | Report Baseline (in-harness) | Phase 1.2 (Winner) | Delta | Notes |
|---|---|---|---|---|
| **PR-AUC (raw)** | 0.7243 | **0.8111** | **+0.0867** | Chronological forward-split |
| **ROC-AUC** | 0.8672 | **0.9902** | **+0.1230** | Near-perfect separation |
| **Mule F1 (honest)** | 0.3941 | **0.7924** | **+0.3983** | Out-of-fold threshold, 0 data-leak |
| **Macro F1** | 0.6958 | **0.8956** | **+0.1999** | Balanced class balance |
| **Balanced Accuracy**| 0.6252 | **0.8290** | **+0.2037** | True Positives vs True Negatives |
| **Recall @ 1% Alert** | 0.6955 | **0.7677** | **+0.0722** | Top 1% queue catches 76.8% mules |
| **Recall @ 5% Alert** | 0.8081 | **0.8970** | **+0.0889** | Top 5% queue catches 89.7% mules |
| **Brier Score** | 0.0028 | **0.0020** | **-0.0008** | Lower is better |
| **ECE (Calib Error)** | 0.0030 | **0.0016** | **-0.0014** | Lower is better |
| **Calibration Slope** | 1.1593 | **1.0438** | **-0.1155** | 1.0 is mathematically perfect |
| **Cost @ R=5 ($)** | 43.00 | **18.60** | **-24.40** | **56.7% operational cost reduction** |
| **Normalized Cost** | 0.7544 | **0.3263** | **-0.4281** | vs trivial baseline (0=perfect, 1=null)|
| **Composite Quality** | 0.5905 | **0.7970** | **+0.2065** | Holistic multi-objective score |

## What changed, stage by stage

### 1. Data grain and the time axis

The problem statement asks for a system that ingests transactions, so the split has to be
forward-looking. Three facts, all in `audit_timeaxis.py`:

- A row is **one account observed at its alert**, not one payment: 3,174 of 3,924 dictionary
  columns are already trailing-window rollups (`L7D`, `L14D`, `L31D`, `L7_14D`, …).
- The observation window is **123 days**. `ACCT_OPN_DATE + TENURE_AS_OF_ALERT` (tenure reads as
  months) places every alert between 30 Aug and 31 Dec 2025. There is no multi-year event-time
  axis to walk along, and no row has a future that another row could leak into.
- The axis that spans years is the **account-opening cohort**, and the mule rate along it moves
  from 0.11% to 1.76% by decile. Order carries information, so shuffling leaks cohort
  membership — which is the whole distance between the report's 0.8677 table and its own
  0.7097 line.

Conclusion: cohort extrapolation is the right protocol, the existing step5 windows already
implement it, and Phase 1.2 keeps them unchanged. No shuffled number is quoted anywhere.

### 2. Leakage audit — measured instead of assumed

`F3912` and `F2230` stay excluded. Two things the report leaves open were tested:

- **Post-alert block `F3895`–`F3923`.** The locked baseline left it in the candidate pool. Running
  with and without gives identical scores to four decimals — the selector never picks any of
  them. It is inert here, and Phase 1.2 excludes it anyway so the spec is defensible.
- **Are the new features label proxies?** The strongest engineered column reaches a 2.3% mule
  rate in its top percentile against a 0.89% base rate. A label copy would be near 100%. No
  feature comes close (`audit_leakage.py`).
- **Alert-window reconstruction.** Because alert time ≈ open date + tenure, a model could in
  principle rebuild the alert month and exploit its uneven mule rate. Dropping every
  tenure-derived feature costs only 0.008 PR-AUC (0.8128 → 0.8052), so the gain does not rest
  on that channel.

### 3. Feature factory — the stage with the real headroom

The report compresses ~3,925 columns into ~105 features over five row-local tracks. The
weakness is not the tracks, it is **which columns reach them**: the selector keeps the 400
highest-variance columns and scores those by mutual information. On this file variance means
rupees, so whole families of ratio and deviation features are filtered out before they are
considered.

`Description.xlsx` turns out to be a grammar —
`[STAT_]ENTITY_METRIC[_DIRECTION][_WINDOW]`, so `RA_CI_NON_CASH_CHQ_AMT_DB_L7_31D` parses as
*ratio of averages · customer-induced non-cash non-cheque · amount · debit · last 7 vs 31 days*.
Parsing it lets the mule typology be constructed directly (`semantic.py`):

| track | what it captures |
|---|---|
| pass-through | credits leave almost as fast as they arrive |
| sweep | balance driven to near zero after a credit |
| velocity | throughput against balance and against account age |
| dispersion | credits fan in over many channels, debits funnel out through one |
| burst | the L7 run rate against the L31 baseline |

Plus 810 group summaries — for each `(stat, metric, direction, window)` family, how extreme this
row is inside it — of which the top 100 by train-window AUC are admitted.

**The single most useful finding:** the strongest columns in the entire file are the plain
totals `TOT_TXNAMT_CR/DB_*` at AUC ≈ 0.79, and the sign is *inverted* — mules here run
**small-ticket, low-value** flows, a structuring profile. The locked selector never picked them
up because a variance filter looks for the opposite.

Two further tracks are fitted fold-safe on the training window (`advanced.py`): peer-cohort
normalisation (behaviour z-scored inside occupation / segment / product) and prototype geometry
(distance to the known mules, leave-one-out on the training side). Both were measured; neither
beat the typology block on its own, and they are documented rather than shipped.

### 4. Ensemble

Kept from the report: 0.6 XGB + 0.4 LGB, no SMOTE, cost-sensitive gradient scaling. Added:
seed-bagging, because a training window holds ~65 mules and single-seed variance dominates at
that count. Family diversity and capacity settings tuned for the positive count — see
`results/models.json`.

### 5. Calibration — the defect worth fixing

The report fits isotonic regression on the model's **own training scores**. Those scores are
saturated near 0 and 1, so the map is over-confident and any threshold read off it transfers
badly. Measured effect on the identical folds: normalised cost **0.78 with the report's
arrangement, 0.36 with an out-of-fold map**.

Phase 1.2 walks expanding blocks through the tail of the training window, fits isotonic on
out-of-fold scores, and smooths the isotonic step function with PCHIP — the report's own
argument for PCHIP, now applied to a map that is actually valid. Reported as Brier, expected
calibration error and calibration slope, not as a picture of a curve.

### 6. Economic engine

The report minimises `FN × R + FP` at a single ratio and quotes the saving against an F1-tuned
threshold. Kept, and extended in three ways (`cost.py`):

- **Every threshold is fitted on held-out scores**, never on scores the model has memorised.
- **Thresholds are taken over the plateau, not the argmax.** With ~26 positives in a calibration
  set the single cheapest grid point moves by whole percentiles when one account changes rank;
  the median of the near-optimal band is stable.
- **The ratio is swept, R = 2 … 100**, and cost is normalised against the trivial policies a bank
  gets for free — review nobody (`mules × R`) or review everybody (`legit × 1`). Normalised cost
  divides by the cheaper of the two: 0 is perfect, 1 means the model is worth nothing. Costs are
  pooled across folds rather than averaged as ratios, so a fold holding 3 mules cannot swing the
  headline as hard as one holding 16.

With a calibrated probability the optimal rule is analytic — alerting costs 1, silence costs
`p × R`, so alert when `p > 1/R`. The Bayes threshold is reported next to the searched one; when
they agree, the calibration is real.

A rate-matched policy (carry the alert *rate* across instead of the probability cut) was tried
as a drift hedge and measured **worse** (normalised cost 1.10 against 0.40). That is itself
evidence the out-of-fold calibration transfers, and it is kept in the scorecard as a negative
result rather than deleted.

### 7. Validation

Five folds share 57 validation mules, so one mean is fragile. `run_bakeoff.py robustness` adds
three independent seed sets, a nine-point walk-forward, and — for contrast only — the
order-blind stratified k-fold that produced 0.8677.

## Layout

```
data_layer.py       cached parse of DataSet.csv + Description.xlsx, dictionary grammar
semantic.py         mule typology block, dictionary group summaries
advanced.py         fold-safe peer-cohort and prototype tracks
experiment.py       FeatureConfig / ModelConfig switches, folds, fit, evaluate
scorecard.py        ranking, decisions, calibration, composite
cost.py             asymmetric cost engine and the R sweep
run_bakeoff.py      stages: anchor | features | models | calibration | robustness | final
rescore.py          re-rank a finished stage without retraining
audit_leakage.py    purity / missingness / signal-location audit
audit_timeaxis.py   row grain and time-axis audit
build_notebook.py   emits SentinelFlow_v2.ipynb
results/            per-stage JSON and logs
```

## Running it

```powershell
# any Python with numpy / pandas / scipy / scikit-learn / xgboost / lightgbm / openpyxl
.\backend\.venv\Scripts\python "Phase_1.2\data_layer.py"        # build the cache once (~45 s)
.\backend\.venv\Scripts\python "Phase_1.2\run_bakeoff.py" features
.\backend\.venv\Scripts\python "Phase_1.2\run_bakeoff.py" models
.\backend\.venv\Scripts\python "Phase_1.2\run_bakeoff.py" calibration
.\backend\.venv\Scripts\python "Phase_1.2\run_bakeoff.py" robustness
.\backend\.venv\Scripts\python "Phase_1.2\run_bakeoff.py" final
C:\Python314\python.exe "Phase_1.2\build_notebook.py"           # needs matplotlib
```

`SentinelFlow_v2.ipynb` is the end-to-end artifact: it runs from `DataSet.csv` and
`Description.xlsx` through features, model, calibration and the cost engine to the final
scorecard and figures, and exports a deployable bundle.
