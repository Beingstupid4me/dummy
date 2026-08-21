# Technical Review of `enhancement.md` and the Phase 1.2 Pipeline

**Reviewer perspective:** senior ML researcher, financial-crime modelling
**Scope reviewed:** `enhancement.md`, `Phase_1.2/` (code + `results/`), `backend/`, `pro_report.tex`, `mule_account_detection_report.tex`, `README.md`, `remaining.md`, and `DataSet.csv` / `Description.xlsx` directly
**Date:** 21 Aug 2026

---

## Verdict

`enhancement.md` is well-written and its productionisation instincts (streaming feature store, value-weighted loss, statutory limits on autonomous freezing, adversarial drift) are broadly sound. But it reviews the *marketing layer*: it accepts the headline metrics as given and critiques how to deploy them.

The material problems are underneath those metrics. There are four that would terminate a model-risk review before any of the Flink / VaR / cross-institutional discussion begins:

| # | Finding | Severity |
|---|---|---|
| 1 | The label is perfectly confounded with the monthly data extract (`F2230`) | **Blocking** |
| 2 | The labelled positives do not behave like mules on any canonical typology | **Blocking** |
| 3 | The winning recipe was selected from 28 candidates on the same folds it is reported on | **Critical** |
| 4 | The five "chronological folds" are nested and resolve to 17 distinct positive accounts | **Critical** |

Findings 1 and 2 mean the central empirical claim — that the model detects mule accounts — is **not established by this dataset as constructed**. Findings 3 and 4 mean the reported improvement (+0.087 PR-AUC) is inside the noise band of the evaluation.

Separately, several specific factual claims in `enhancement.md` do not survive contact with the repository (§B). Since the document's stated purpose is to defend the work to evaluators, those are liabilities in their own right.

**Phase 1.2 is the strongest and most honest work in the repository** (§D). The criticism below is aimed at the conclusions drawn from it, not at its craftsmanship.

---

## A. Blocking and critical findings

### A1. The label is the extract month (`F2230` = `MNTH`)

`F2230` is `MNTH` in the data dictionary. Its crosstab against the target `F3924`:

```
F2230    legit   mule
Sep25        0     48
Oct25     9001      0
Nov25        0     23
Dec25        0     10
```

**All 9,001 negatives are the October 2025 extract. All 81 positives come from the September, November and December extracts. There is zero overlap.**

The dataset was assembled by stitching mules from three monthly pulls together with non-mules from a fourth. In this file, "is a mule" and "came from a non-October extract" are the same variable.

**Why dropping the column does not fix it.** `data_layer.py` correctly excludes `F2230` and `F3912`:

```python
CONFIRMED_LEAKY = ["F3912", "F2230"]
```

But per `results/timeaxis.log`, **3,174 of 3,924 dictionary columns are trailing-window rollups** (`L7D`, `L14D`, `L31D`, `AVG_BAL_31DAYS`, …) computed as of the extract date. If the two classes come from different extract months, every one of those columns carries a systematic between-class difference unrelated to behaviour — seasonality, month-end batch effects, festival-period spending (Diwali fell ~20 Oct 2025), or simply a different source population. Removing `F2230` deletes the *name* of the confound and leaves the confound itself smeared across thousands of features, each individually weak.

**Why the leakage audit could not catch it.** `audit_leakage.py` asks two questions — is any single feature a label proxy (top-1% purity), and is the block merely detecting missingness. Both are the right questions. Its conclusion is accurate and irrelevant here:

> "the strongest engineered column reaches a 2.3% mule rate in its top percentile against a 0.89% base rate. A label copy would be near 100%."

A diffuse confound distributed over thousands of moderately-weak columns passes a per-feature purity test trivially. The audit tests for *proxies*; this is a *confound*.

**Why it explains the ROC/PR signature.** ROC-AUC is dominated by the negative class; here the negative class is one homogeneous extract. PR-AUC is driven by the 81 positives. ROC-AUC 0.9902 alongside PR-AUC 0.8111 is the signature of a model separating two data pulls rather than two behaviours.

**Why the chronological protocol offers no protection.** `step5_windows` splits on `F3888` (account-open date), which is orthogonal to extract month. Both sides of every fold carry the confound in equal measure. The PS2 paper's 0.8677, Phase 1 Stable's 0.7381 and Phase 1.2's 0.8111 are all measuring the same confound with varying efficiency. `semantic.py` is built entirely from the amount rollups that carry it most strongly, which is a plausible reading of where the +0.09 came from.

**Reproduce:**
```python
df = pd.read_csv('DataSet.csv', usecols=['F2230','F3924'], low_memory=False)
pd.crosstab(df['F2230'], df['F3924'])
```

---

### A2. The labelled positives do not behave like mules

`Phase_1.2/README.md` §3 calls this "the single most useful finding":

> "the strongest columns in the entire file are the plain totals `TOT_TXNAMT_CR/DB_*` at AUC ≈ 0.79, and the sign is *inverted* — mules here run **small-ticket, low-value** flows, a structuring profile."

`enhancement.md` §2.1 builds on it, describing "smurfing amounts between ₹500 and ₹5,000 to evade statutory ₹50,000 PMLA reporting thresholds."

Measured behaviour of the labelled classes:

| Metric (L31D, median) | legit | mule | ratio |
|---|---|---|---|
| Credit transaction **count** | 25 | 13 | 0.52× |
| Average **credit ticket** | ₹37,840 | ₹7,192 | 0.19× |
| Average **debit ticket** | ₹24,488 | ₹2,736 | 0.11× |
| Monthly **turnover ÷ average balance** | 1.41 | 0.46 | 0.32× |
| **Average balance** | ₹3,99,517 | ₹3,38,197 | 0.85× |
| Min balance ÷ average balance (sweep) | 0.00 | 0.00 | — |

Univariate AUC against the target (0.5 = no signal):

| Variable | AUC | reads as |
|---|---|---|
| `TOT_TXNAMT_CR_L7D` | 0.2087 | mules much lower |
| `TOT_TXNAMT_DB_L31D` | 0.1982 | mules much lower |
| `TOT_TXNS_CR_L7D` | 0.4556 | ~no signal |
| `TOT_TXNS_DB_L31D` | 0.4767 | ~no signal |
| `AVG_BAL_31DAYS` | 0.4753 | ~no signal |
| `TENURE_AS_OF_ALERT` | 0.5236 | ~no signal |

**The structuring reading fails on its own terms.** Smurfing means *more* transactions of *smaller* size. These accounts make **half as many** transactions. More decisively, their turnover ratio is **0.46** — they move less than half their balance in a month, against 1.41 for legitimate accounts — while holding a comparable ₹3.4 lakh balance. A pass-through mule has turnover far above 1 on a balance near zero. What the labels describe is a **quiet, low-activity retail account**.

**This directly undermines the feature factory's premise.** `semantic.py` is constructed around five typologies — pass-through, sweep, velocity, dispersion, burst. The labelled positives exhibit these in the *opposite* direction or not at all. The grammar-parsing of `Description.xlsx` is genuinely good engineering, but what it is efficiently encoding is "low monetary activity," which is also the most likely fingerprint of a different extract population (A1).

**A whole typology is absent from the labels.** From `results/timeaxis.log`, `ACCT_OPN_DAYS` buckets: 7 days → 386 accounts, **0 mules**; 14 days → 87, **0**; 90 days → 207, **0**. The canonical mule — freshly opened, rented, burst, abandoned — does not appear in the label set at all.

**Action:** retire the structuring narrative, or restate it with the transaction-count and turnover figures alongside.

---

### A3. The winner was selected on the folds it is reported on

`run_bakeoff.py` carries stage winners forward by composite score:

```python
def best_model_config() -> ModelConfig:
    rows = []
    for name in ("models", "capacity", "combo", "calibration"):
        ...
    best = max(rows, key=lambda r: r["composite"])
```

`stage_final` then evaluates that winner on **the same step5 folds every candidate was scored on**. There is no inner selection split, and the composite's components (`pr_auc`, `mule_f1`, `cost_efficiency`) are the same quantities that appear in the headline table.

Completed candidates in `results/`:

| Stage | n | PR-AUC range |
|---|---|---|
| `features` | 9 | 0.7243 – 0.8176 |
| `capacity` | 8 | 0.8127 – 0.8826 |
| `combo` | 7 | 0.8037 – 0.8780 |
| `calibration` | 4 | 0.8111 – 0.8111 |
| `models` | — | **`models.json` missing** (see A3.1) |
| **Total** | **28** | mean 0.8192, **sd 0.0362**, max 0.8826 |

On identical data, config choice alone moves PR-AUC by 0.16. The claimed improvement of +0.087 over the control sits comfortably inside that band.

**Corroborating evidence from `results/robustness.json`** — the same winner and control under three protocols:

| Protocol | winner | control | delta |
|---|---|---|---|
| step5 (5 nested folds) — **the quoted number** | 0.8111 | 0.7243 | **+0.087** |
| dense 9-point walk-forward | 0.8495 | 0.8118 | **+0.038** |
| order-blind shuffled | 0.8697 | 0.8200 | **+0.050** |

The quoted +0.087 is the largest of the three and comes from the most overlapping protocol. The dense-9 gap of +0.038 is roughly the shrinkage expected once selection pressure is partly relieved.

**A vivid illustration**, from `results/models.log`, config `m2_shallow`:

```
fold 4 n_feat= 414 PR=1.0000 ROC=1.0000 muleF1=0.500 R@1%=1.000 normcost=0.667
```

A fold containing three mules returning a perfect PR-AUC of 1.000, averaged into a headline.

#### A3.1 The `models` stage never completed
`results/models.log` stops mid-run at `m3_shallow_reg` and `models.json` was never written. `best_model_config()` reads the file list with `if path.exists()`, so that entire stage is **silently skipped** with no warning. Nothing downstream indicates the selection pool was incomplete.

---

### A4. Nested folds — 17 distinct positive accounts

`step5_windows` produces splits at 0.72n, 0.76n, 0.80n, 0.84n, 0.88n with a fixed 0.2n validation slice. Measured (n = 9,082):

| fold | val range | val n | val mules | val rate |
|---|---|---|---|---|
| 0 | [6539 : 8355] | 1816 | 16 | 0.881% |
| 1 | [6902 : 8718] | 1816 | 16 | 0.881% |
| 2 | [7265 : 9081] | 1816 | 13 | 0.716% |
| 3 | [7628 : 9082] | 1454 | 9 | 0.619% |
| 4 | [7991 : 9082] | 1091 | **3** | 0.275% |

Pairwise Jaccard overlap ranges 0.14 – 0.80. Fold 4's validation set is a subset of fold 3's, which is a subset of fold 2's.

- **Unique validation rows across all five folds: 2,543** (of 9,082)
- **Unique validation mules across all five folds: 17** (of 81)

`Phase_1.2/README.md` states "Five folds share 57 validation mules." 57 is the *sum* of per-fold counts (16+16+13+9+3); the **distinct count is 17**. The fragility is understated by 3.4×.

Every headline number — PR-AUC 0.8111, mule F1 0.7924, cost $18.60 — is a function of how the model ranks 17 accounts, each counted up to five times. `experiment.py` pools costs by summing across these overlapping populations, double-counting the same accounts.

Combined with A3: **selecting the best of 28 configurations on 17 positive accounts.**

---

## B. Factual errors in `enhancement.md`

These are credibility liabilities in front of evaluators.

### B1. The escrow claim is fabricated — but the underlying bug is real and worse

`enhancement.md` §3.2: *"Section 5 of the PS2 paper proposes an automated 15-minute escrow fund freeze when $P_{calib} \ge 0.62$."*

In `mule_account_detection_report.tex`:
- §5 is the **cost framework**; it never mentions escrow.
- §1 states the opposite: *"SentinelFlow does not seek to fully automate account suspension."*
- The cost curve places the minimum at **threshold 0.32**, where the **F1 score is 0.62**.

Someone read a value off the y-axis of a figure and hardcoded it as a probability cutoff:

```python
# backend/app/services/decision.py
def status(p_calib: float, threshold: float) -> AlertStatus:
    if p_calib >= max(0.62, threshold):
        return "ESCROW"
    if p_calib >= max(threshold, 0.32):
        return "QUEUE"
    return "CLEAR"
```

Every shipped bundle carries `threshold = 0.08` (verified by loading M1/M2/M3). The `max()` calls mean that threshold is **never used**. Everything between 0.08 and 0.32 returns `CLEAR` — and that band contains both the cost-optimal operating point and the R=5 Bayes cut of 1/R = 0.20. The entire asymmetric cost engine in `cost.py` is overridden by two constants lifted from a chart.

### B2. The $18.60 / −56.7% figure measures a broken control

In `results/final_recipe.json`, the baseline's threshold search collapses to `cost_threshold: 1.0` in **all five folds**, producing TP = 2.8, FP = 0.0 and an identical `normcost` of 0.75 at *every* R from 2 to 100 — a threshold that never moves.

Against a working comparator, the same file reports:

```json
"savings_r5_vs_f1": 0.021052631578947323
```

**The honest saving at R=5 is 2.1%, not 56.7%.** It does reach 40% at R=20 and 52% at R=100 — that is the number worth leading with, and it reframes the contribution correctly as *"the cost engine matters when FN cost is high."*

### B3. The ego-graph does not exist

§4.1 critiques the graph service for being limited to Bank of India's internal ledger. There is **no counterparty data in `DataSet.csv` at all**; `graph.py` synthesises node IDs from the account string:

```python
if not hop1:
    hop1 = [f"B{account[-4:]}{i}" for i in range(3)]
...
hop2 = f"C{account[-4:]}"
```

Framing this as "blind to hops 2 and 3" implies hops 0 and 1 are real. Under one follow-up question this is far worse than stating "the graph is a UI placeholder; the dataset has no counterparty field."

### B4. Other corrections

| Claim in `enhancement.md` | Reality |
|---|---|
| "R = 5" attributed to the PS2 paper | Neither `.tex` states a numeric R; both use symbolic `COST_FN_RATIO`. R=5 is Phase 1.2's own choice (`cost.py: R_HEADLINE = 5.0`). |
| §2.2 titled "The 3,925-to-157 Dimensionality Collapse" | Its own comparison matrix says Phase 1.2 uses **414** columns; 157 is the *baseline's* count. Self-contradictory. |
| "p99 46.86 ms … high-throughput inference engine" | Artifact is **N = 2,000** (not the N=10,000 claimed in `README.md` and `pro_report.tex`); measured throughput is **60.8 req/s single-threaded**; `shap_explanation` p50 = 0.0 means **TreeSHAP never fired**, so the most expensive component in `pro_report.tex`'s own latency table (24.1 ms p99) is excluded from the measured p99. |
| "Section 102 of the Code of Criminal Procedure (Cr.P.C.)" | CrPC was repealed 1 July 2024; the provision is now **Section 106 BNSS**. Current case law (Kerala HC *Headstar Global*, followed by Bombay HC; SC declined to interfere) holds that debit-freezing for proceeds of crime **cannot** be done under §106 at all — it requires a Magistrate's order under **§107 BNSS**. This makes the document's own recommendation (reframe as an interim settlement-verification hold under the bank's T&Cs) *stronger* than it currently argues. |
| SR 11-7 / OCC cited as the governing MRM frame | US guidance. For a Bank of India panel the relevant frames are the RBI Master Directions and the RBI FREE-AI framework. |

---

## C. Additional issues not raised in `enhancement.md`

### C1. The reported operating point is a threshold of exactly 1.0
In `results/final_recipe.json`, `f1_threshold = 1.0` in **4 of 5 folds** and `cost_threshold = 1.0` in **3 of 5**, yielding precision exactly 1.000 and FP exactly 0. That is the top isotonic knot, where PCHIP maps to y = 1.0 — the "threshold" is floating-point equality with certainty, not a probability cut.

The headline **Mule F1 of 0.7924 is read at p ≥ 1.0**. No model-risk function will accept P(mule) = 1.0 derived from 65 training positives, and an operating point pinned to the saturation boundary cannot be tuned or governed.

### C2. Mean calibration slope 1.0438 is an averaging artifact
Per-fold slopes: **1.119, 1.269, 0.977, 1.156, 0.698**. The mean lands near 1.0 because errors in opposite directions cancel; the spread is 0.70 – 1.27.

Additionally, **ECE is near-uninformative at 0.89% prevalence** — a constant "never a mule" predictor scores ECE ≈ 0.0089, so 0.0016 is respectable but not the "mathematically perfect" framing it receives. Calibration should be reported *within the alert tail*.

### C3. The cohort-risk narrative runs backwards
`Phase_1.2/README.md` justifies the protocol with *"the mule rate along it moves from 0.11% to 1.76% by decile. Order carries information."*

`results/timeaxis.log`, oldest → newest decile: **0.77, 0.77, 0.66, 1.21, 1.76, 0.99, 0.77, 0.55, 1.32, 0.11**. Non-monotone, and the **newest** decile is the **lowest**. First-half 1.035% vs second-half 0.749%. Risk *falls* with recency here, so "forward-cohort extrapolation" validates on the cleanest population, not the riskiest.

### C4. `future_core` is neither future nor an independent holdout
```python
def future_core_split(open_dates):
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    return order[: int(0.7 * n)], order[int(0.7 * n) : int(0.9 * n)]
```
The newest 10% — **909 accounts, 1 mule** — is silently discarded. Its training window (6,357) is a **subset of every step5 training window**, and its validation slice overlaps all five step5 validation sets. The 0.8508 figure is therefore not independent confirmation of 0.8111; it is largely the same accounts scored again.

### C5. Scope — this is post-alert triage, not detection
Every row carries `TENURE_AS_OF_ALERT` and every reconstructed alert falls inside a 123-day window (30 Aug – 31 Dec 2025). All 9,082 accounts had **already tripped an alert**. The problem statement asks for detection across the book.

Two consequences:
- On the full customer base, prevalence is orders of magnitude below 0.89%, and precision at any fixed threshold falls proportionally. **No prevalence-shift analysis exists anywhere in the repo.**
- The 9,001 negatives are *alerted-but-unconfirmed*, so they contain undetected mules. Recall is upward-biased and the cost model understates FN.

### C6. Deployment handoff is incomplete
- `Phase_1.2/SentinelFlow_v2.ipynb` §8 ("Train the deployable bundle and export") writes `Phase_1.2/models/phase12_bundle.joblib` with per-ratio cost thresholds. **`Phase_1.2/models/` is empty** — the cell has never been run to completion.
- `backend/app/bootstrap.py` loads M1 from `Phase_1_enhancement/models/enhance_m1.joblib`, **not** from Phase 1.2.

The served model's own recorded metrics differ materially from every published headline:

| | Documented as "our solution" | Actually served by `/score` |
|---|---|---|
| Chrono PR-AUC (step5 mean) | 0.8111 | **0.7261** |
| Future-core PR-AUC | 0.8508 | **0.7660** |
| ROC-AUC | 0.9902 | **0.9140** |
| Features | 414 | **186** |

`Phase_1_enhancement/metrics.json` further records `"beats_stable": false` — the served model does not beat the locked Phase 1 baseline of 0.7381.

### C7. TMS flags read the post-alert block the model excludes as leaky
`engine.py::_TMS_MAP` maps **F3900–F3911 and F3916–F3919**. `data_layer.py` defines `POST_ALERT = [f"F{i}" for i in range(3895, 3924)]` and excludes it. The offline claim is "F3895–F3923 excluded, zero leakage"; the live console displays exactly those fields as the reasons for the alert. Defensible as a separate rule engine, but it must be stated explicitly.

### C8. Demo scores are in-sample for ~70% of accounts
`engine.py::materialize()` builds the served matrix over `all_idx` using the bundle's `train_idx`. Any account inside the training window is scored by a model that memorised it. Acceptable for a demo; dangerous if any screenshot or number is quoted.

### C9. Documentation claims that do not match the repository

| Claim | Location | Reality |
|---|---|---|
| "M4 … **Shipped** … generated via `bootstrap --all`" | `remaining.md` | `bootstrap.py` loops over only `("M2", "M3")`; `backend/registry/` contains only M1, M2, M3. |
| "N=10,000 Latency Benchmark … **Shipped**" | `README.md`, `pro_report.tex` | `latency_benchmark_results.json` records `n_samples: 2000`. |
| Leakage table: F3886 = "Freeze flag", F3892 = "Post-event tag" | `pro_report.tex` | F3886 is **`PRODUCT_NAME`** (17 values); F3892 is **`GENDER`**. Descriptions are wrong. |
| PS2 random 5-fold PR-AUC | `mule_account_detection_report.tex` | Reported as **0.8335** in `tab:chroncomp` and **0.867695** in `tab:step4_results` for the same stage, unexplained. |

### C10. Minor engineering notes
- `experiment.py::_FOLD_CACHE` and `semantic.py::_CACHE` key on `id(base)`. CPython reuses object ids after GC; a stale-cache collision is possible in a long bake-off session.
- `advanced.py::COHORT_COLS` includes `F3886`, which both `.tex` reports list as excluded-leaky. Not shipped (`peer_relative=False` in the final recipe), but the candidate pool and the published audit disagree.
- `Phase_1.2/README.md` concedes the locked recipe reproduces to 0.7243 rather than 0.7381 due to a library-generation change. No pinned versions exist in the Phase 1.2 tree.

---

## D. What Phase 1.2 gets right

This should be said plainly, because the craft is not the problem — the conclusion is.

- **Correctly excluded `F2230` and `F3912`** when the published audit table in `pro_report.tex` mislabels other fields entirely.
- **Replaced in-sample isotonic with a genuine out-of-fold map** (`calib_mode="oof"`, expanding blocks through the training tail). This was a real defect in the baseline and it was correctly diagnosed and fixed.
- **Swept R from 2 to 100** rather than quoting one flattering ratio, and normalised cost against the trivial policies a bank gets for free.
- **Reported the Bayes threshold 1/R next to the searched threshold** as a calibration cross-check — genuinely good practice.
- **Reported the rate-matched policy as a negative result** rather than deleting it.
- **Ran a dense 9-point walk-forward and an order-blind control** specifically to test its own win. That the dense-9 delta is half the headline is visible only because they ran the experiment.
- **Parsed `Description.xlsx` as a formal grammar.** Independent of whether the resulting signal is real, this is the most creative piece of engineering in the project.
- **`Phase_1.2/README.md` is by a wide margin the most honest document in the repository** — it volunteers the 0.7243 reproduction gap and documents peer-cohort and prototype tracks as measured-and-rejected.

The two documents downstream of it — the top-level `README.md` and `enhancement.md` — strip out nearly every caveat Phase 1.2 was careful to record.

---

## E. Remediation, in priority order

### E1. Raise the `F2230` finding with the data provider — **do this first**
You need negatives drawn from the same extract months as the positives. Until then, no metric from this file separates mule behaviour from extract cohort. Stating that clearly is a **stronger** position in front of evaluators than any PR-AUC: it demonstrates you audited the data, not just the model.

### E2. Run the confound diagnostic
Train the current Phase 1.2 recipe to predict `F2230 == "Oct25"` with the target and `F2230` both removed. If it reaches the same ~0.99 ROC-AUC, you have proof and a slide. **This is the single highest-value experiment remaining.**

### E3. Retire the structuring narrative
Or restate it alongside the transaction-count (0.52×) and turnover (0.32×) figures, which contradict it.

### E4. Add an inner selection split
Choose configurations on folds you do not report, or promote the dense-9 numbers to primary. Publishing **+0.038 with a bootstrap interval over the 17 unique positives** is worth more than +0.087 without one. Also fix the silent skip in `best_model_config()` when a stage JSON is missing.

### E5. Replace step5 with non-overlapping blocked walk-forward
Report the number of *distinct* validation positives next to every metric.

### E6. Remove the degenerate operating point
Drop threshold = 1.0. Force the decision point onto the Bayes cut 1/R and report precision/recall there. Delete the `max(0.62, …)` / `max(…, 0.32)` constants in `decision.py` and drive thresholds from the bundle.

### E7. Complete the handoff
Run the notebook's export cell, finish `stage_models`, and load `phase12_bundle.joblib` in `bootstrap.py` so the served model and the benchmarked model are the same object.

### E8. Reframe the deliverable as post-alert triage
Add a prevalence-shift curve: precision as a function of deployment base rate. This converts the biggest unstated weakness into a demonstration of domain judgement.

### E9. Restate the economics honestly
Savings versus the F1 policy across the R sweep (2.1% at R=5, 40% at R=20, 52% at R=100), with R stated explicitly and owned as your choice.

### E10. Housekeeping
Relabel the ego-graph as a placeholder in both UI and docs. Re-run the latency benchmark at N=10,000 with SHAP forced on and report throughput alongside percentiles. Correct the M4 and N=10,000 claims in `remaining.md` and `README.md`. Pin library versions in `Phase_1.2/`.

---

## F. Recommended framing for the presentation

Lead with the data audit, not the model.

> *"We found that the positive and negative classes were drawn from disjoint monthly extracts, which makes the headline metrics in the prior work unsafe. Here is how we detected it, here is why the standard leakage audit misses it, and here is what we would need to fix it."*

That is a stronger claim to methodological seriousness than any number currently in the comparison matrix — and it is defensible under questioning, which the present headline numbers are not.
