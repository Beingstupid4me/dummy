"""Emit Phase_1.2/SentinelFlow_v2.ipynb — the end-to-end improved solution.

The notebook walks the same seven stages as `ps2_mule_account_detection_report.pdf`
and, for each one, states what the report did, what Phase_1.2 changed and what
the change measured on the identical chronological folds. It runs from
`DataSet.csv` + `Description.xlsx` to a trained, calibrated, cost-optimised
bundle with every figure regenerated.

Heavy lifting lives in the sibling modules (`data_layer`, `semantic`,
`advanced`, `experiment`, `scorecard`, `cost`) so the notebook and the tested
code cannot drift apart; the appendix prints their source inline.

    python Phase_1.2/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "SentinelFlow_v2.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def recipe() -> dict:
    """Winning configs from the bake-off, so the notebook trains what we picked."""
    path = HERE / "results" / "final_recipe.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "feature_config": {
            "include_post_alert": False,
            "semantic": True,
            "force_semantic": True,
        },
        "model_config": {"n_seeds": 5, "calib_mode": "oof"},
        "note": "defaults; run run_bakeoff.py stages to refresh",
    }


def cells() -> list[dict]:
    rec = recipe()
    fcfg = json.dumps(rec["feature_config"], indent=4)
    mcfg = json.dumps(rec["model_config"], indent=4)

    out: list[dict] = []

    out.append(md(r"""
# SentinelFlow v2 — Phase 1.2

**Improving every stage of `ps2_mule_account_detection_report.pdf`, not just the headline score.**

The report describes a seven-stage system: a leakage audit, a feature factory, a GBDT
ensemble, isotonic calibration, an asymmetric cost engine, chronological validation and a
serving path. This notebook rebuilds the first six of those stages and measures each change
on **the identical chronological folds** the report and the locked Phase 1 baseline used, so
every number below is a like-for-like delta rather than a new protocol flattering a new model.

| Stage | What the report does | What v2 does instead |
|---|---|---|
| 1 Data & time axis | splits on `F3888`, quotes a shuffled 5-fold table too | establishes the row grain and the real observation window, drops shuffled numbers |
| 2 Leakage audit | excludes `F3912` / `F2230` | keeps that, and *measures* the post-alert block instead of assuming it |
| 3 Feature factory | 5 row-local tracks, ~105 features | adds a dictionary-derived mule typology, peer-cohort normalisation, prototype geometry |
| 4 Ensemble | one seed, 0.6 XGB + 0.4 LGB | seed-bagged, diversified, capacity tuned for 65 positives |
| 5 Calibration | isotonic fitted on in-sample training scores | isotonic + PCHIP fitted on out-of-fold scores, then measured (Brier / ECE / slope) |
| 6 Economic engine | one cost ratio, saving quoted against an F1 threshold | nested thresholds, ratio sweep R = 2…100, cost normalised against trivial policies |
| 7 Validation | 5 rolling windows | 5 rolling windows + 9-point walk-forward + seed repeats |

Everything runs from `DataSet.csv` and `Description.xlsx`.
"""))

    out.append(md("## 0. Environment"))
    out.append(code("""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path.cwd()
PKG = HERE if (HERE / "data_layer.py").exists() else HERE / "Phase_1.2"
ROOT = PKG.parent
sys.path.insert(0, str(PKG))

FIG = PKG / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.titlesize": 12, "font.size": 9})
NAVY, SLATE, CRIMSON, GOLD = "#1D3557", "#457B9D", "#E63946", "#F4A261"

print("data :", (ROOT / "DataSet.csv").exists(), "| dictionary:", (ROOT / "Description.xlsx").exists())
"""))

    out.append(code("""
from data_layer import load_base, parse_grammar
from experiment import (
    FeatureConfig, ModelConfig, build_fold, evaluate, fit_predict,
    step5_windows, dense_windows, shuffled_folds, future_core_split,
    LOCKED_CHRONO_PR, PS2_CHRONO_PR,
)
from scorecard import (
    apply_calibrator, composite, expected_calibration_error, fit_calibrator,
    calibration_slope, score_fold,
)
from cost import R_GRID, pick_cost_threshold, pick_f1_threshold, total_cost, trivial_cost
from semantic import semantic_block, ratio_physics_block

t0 = time.time()
base = load_base()          # parses the CSV once, then caches
print(f"loaded in {time.time() - t0:.1f}s")
print(f"rows {len(base.y)}   mules {int(base.y.sum())}   base rate {base.y.mean():.4%}")
print(f"numeric columns after scrubbing: {base.numeric.shape[1]}")
"""))

    # ---------------- stage 1 ----------------
    out.append(md(r"""
## 1. What is a row, and what is the time axis?

The problem statement asks for a system that ingests **transactions**. It matters a great deal
whether this file is a transaction log or something else, because it decides what a valid
split looks like. Three facts settle it.

1. Every row is one **account observed at its alert**, not one payment. The dictionary shows
   3,174 of 3,924 columns are already trailing-window rollups (`L7D`, `L14D`, `L31D`,
   `L7_14D`, …), and `TENURE_AS_OF_ALERT` / `AGE_IN_YRS` are stated "as of the alert date".
2. The **observation window is short**. `ACCT_OPN_DATE + TENURE_AS_OF_ALERT` (tenure reads as
   months) puts every alert inside a single 123-day extract. There is no multi-year event-time
   axis to walk forward along, and no row has a future that another row could leak.
3. The axis that *does* span years is the **account-opening cohort**, and the mule rate along
   it is strongly non-stationary. Order therefore carries information, and shuffling destroys
   it — which is exactly why the report's shuffled 0.8677 is not a deployment estimate.

So the honest protocol is **cohort extrapolation**: train on older-vintage accounts, validate
on newer ones. That is the step5 protocol, and Phase 1.2 keeps it unchanged so the comparison
stays fair.
"""))
    out.append(code("""
opn = base.open_dates
ten = pd.to_numeric(base.work["F3887"], errors="coerce")
alert = opn + pd.to_timedelta(np.clip(ten * 30.44, 0, 60000), unit="D")

print(f"account-open span : {opn.min():%Y-%m-%d} -> {opn.max():%Y-%m-%d}")
print(f"alert-time span   : {alert.min():%Y-%m-%d} -> {alert.max():%Y-%m-%d}"
      f"  ({(alert.max() - alert.min()).days} days)")
print(f"window rollup cols: {(base.grammar['window'] != '').sum()} of {len(base.grammar)}")

order = opn.argsort(kind="mergesort").to_numpy()
n = len(order)
oy = base.y.to_numpy()[order]
dec = pd.DataFrame({
    "decile": range(10),
    "n": [len(oy[int(n*i/10):int(n*(i+1)/10)]) for i in range(10)],
    "mules": [int(oy[int(n*i/10):int(n*(i+1)/10)].sum()) for i in range(10)],
    "cohort_start": [opn.iloc[order[int(n*i/10)]].date() for i in range(10)],
})
dec["rate"] = dec["mules"] / dec["n"]
display(dec)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].bar(dec["decile"], dec["rate"] * 100, color=SLATE)
ax[0].axhline(base.y.mean() * 100, color=CRIMSON, ls="--", label="overall")
ax[0].set(title="Mule rate by account-open cohort decile", xlabel="decile", ylabel="mule rate (%)")
ax[0].legend()
ax[1].hist(alert.dt.dayofyear, bins=40, color=NAVY)
ax[1].set(title=f"Reconstructed alert dates span {(alert.max()-alert.min()).days} days",
          xlabel="day of year 2025", ylabel="accounts")
fig.tight_layout(); fig.savefig(FIG / "01_time_axis.png", bbox_inches="tight"); plt.show()
"""))

    # ---------------- stage 2 ----------------
    out.append(md(r"""
## 2. Leakage audit — measured, not assumed

The report excludes `F3912` (`FRAUD_SUSPECTED`) and `F2230`. Phase 1.2 keeps both exclusions
and then asks two questions the report leaves open.

**Does the post-alert block matter?** Columns `F3895`–`F3923` are populated around the alert:
some are genuine inputs the problem statement asks us to ingest (TMS alert types such as
`RCVING_FUNDS_FROM_MULITPLE_USERS`, `ONE_TO_MANY_UPI_PAYMENTS`), others are outcomes
(`MIN_RESOLVE_DAYS`, `FALSE_POSITIVE`, `UNATTENDED`). The locked baseline left the whole block
in the candidate pool. Rather than argue about it, we ran the recipe with and without: the
selector never picks any of them, and the chrono score is identical to four decimals. The
block is inert here, and Phase 1.2 excludes it anyway so the spec is defensible.

**Are the new engineered features label proxies?** A feature that isolates mules would show a
near-pure top slice. The strongest engineered column reaches a 2.3% mule rate in its top
percentile against a 0.89% base rate — a real but weak signal, which is what an honest
behavioural feature looks like.
"""))
    out.append(code("""
from scipy.stats import rankdata

sem = semantic_block(base)
y = base.y.to_numpy()

def top_slice_rate(frame, q=0.99):
    rows = []
    for c in frame.columns:
        v = frame[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 100:
            continue
        top = ok & (v >= np.nanquantile(v, q))
        if top.sum() >= 20:
            rows.append({"feature": c, "top1pct_mule_rate": float(y[top].mean()), "n": int(top.sum())})
    return pd.DataFrame(rows).sort_values("top1pct_mule_rate", ascending=False)

pur = top_slice_rate(sem)
print(f"base rate {y.mean():.4%};  a label copy would show ~100% here")
display(pur.head(8))
print("engineered features whose top percentile is >=50% mule:",
      int((pur['top1pct_mule_rate'] >= 0.5).sum()))
"""))

    # ---------------- stage 3 ----------------
    out.append(md(r"""
## 3. Feature factory v2

The report compresses ~3,925 columns into ~105 features over five row-local tracks: statistical
moments, missingness gaps, categorical encoding, cyclical dates, and an isolation-forest score.
That factory has one structural weakness: **which raw columns reach it**. The selector keeps
the 400 highest-variance columns and then scores those by mutual information. On this file
variance means rupees, so entire families of ratio and deviation features are filtered out
before they are ever considered.

`Description.xlsx` fixes this, because the column names are a grammar:

`[STAT_]ENTITY_METRIC[_DIRECTION][_WINDOW]` — e.g. `RA_CI_NON_CASH_CHQ_AMT_DB_L7_31D` is
*ratio of averages, customer-induced non-cash non-cheque, amount, debit, last 7 vs 31 days*.

Parsing it lets us construct the mule typology directly instead of hoping a variance filter
stumbles onto it:

- **pass-through** — credits leave almost as fast as they arrive
- **sweep** — balance driven to near zero after a credit
- **velocity** — throughput against balance and against account age
- **dispersion** — credits fan in over many channels, debits funnel out through one
- **burst** — the L7 run rate against the L31 baseline

Two further tracks need the training window and so are fitted fold-safe:

- **peer-cohort normalisation** — the same turnover is unremarkable for a trader and loud for
  a pensioner, so behaviour is z-scored inside its occupation / segment / product cohort
- **prototype geometry** — with only 65 known mules in a training window, where a row sits
  relative to those 65 in behaviour space is information axis-aligned splits cannot recover
  (leave-one-out on the training side, so no row is its own nearest mule)
"""))
    out.append(code("""
g = base.grammar
print("dictionary grammar:")
display(g.head(6)[["name", "stat", "entity", "metric", "dir", "window"]])
print("\\nwindow families:", g["window"].value_counts().to_dict())
print("stat families   :", g["stat"].value_counts().to_dict())

phys = ratio_physics_block(base)
print(f"\\ntypology block      : {sem.shape[1]} features")
print(f"group-summary block : {phys.shape[1]} features")
"""))
    out.append(code("""
# univariate strength: |AUC - 0.5| of each engineered feature against the label
def col_auc(frame):
    arr = frame.to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), np.nanmedian(arr, axis=0), arr)
    arr = np.nan_to_num(arr)
    ranks = rankdata(arr, axis=0)
    n1 = float(y.sum()); n0 = len(y) - n1
    return pd.Series((ranks[y == 1].sum(axis=0) - n1*(n1+1)/2) / (n1*n0), index=frame.columns)

auc_sem = col_auc(sem)
strength = (auc_sem - 0.5).abs().sort_values(ascending=False).head(14)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
vals = auc_sem[strength.index]
ax.barh(range(len(vals)), vals.values - 0.5,
        color=[CRIMSON if v > 0.5 else SLATE for v in vals.values])
ax.set_yticks(range(len(vals)))
ax.set_yticklabels([c.replace("sem_", "") for c in vals.index], fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set(title="Typology features: AUC - 0.5  (left = mules score LOW)", xlabel="AUC - 0.5")
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(FIG / "02_typology_strength.png", bbox_inches="tight"); plt.show()

print("The strongest signals are negative: mules here run SMALL-ticket, LOW-value flows.")
print("That is a structuring / smurfing profile, and it is why a variance filter missed it.")
"""))

    # ---------------- stage 4-6 config ----------------
    out.append(md(r"""
## 4–6. Ensemble, calibration and the economic engine

Three changes, each measured separately in `run_bakeoff.py` and combined here.

**Ensemble.** A training window holds ~65 mules, so a single-seed GBDT is dominated by seed
variance. v2 bags several seeds per family and averages, which costs nothing at inference
because the trees are merged into one score.

**Calibration.** The report fits isotonic regression on the model's *own training scores*.
Those scores are saturated near 0 and 1, so the resulting map is over-confident and any
threshold read off it transfers badly — in our runs the report's arrangement pushed
normalised cost to 0.75 where an honest map reaches 0.33. v2 walks expanding blocks through
the tail of the training window, fits isotonic on **out-of-fold** scores, and smooths the
isotonic step function with PCHIP so the derivative stays positive and continuous (the report's
own argument for PCHIP, applied to a map that is now valid).

**Economic engine.** The report minimises `FN x R + FP` at a single `R = COST_FN_RATIO` and
quotes the saving against an F1-tuned threshold. v2 keeps that comparison and adds the two
things that make it decision-grade: every threshold is fitted on held-out scores, and the
engine is scored across **R = 2 … 100** against the trivial policies a bank gets for free
(review nobody: `mules x R`; review everybody: `legit accounts x 1`). Normalised cost divides
by the cheaper of the two, so 0 is perfect and 1 means the model is worth nothing.

With a calibrated probability the optimal rule is analytic — alerting costs 1, silence costs
`p x R`, so alert when `p > 1/R`. We report that Bayes threshold next to the searched one;
when they agree, the calibration is real.
"""))
    out.append(code(f"""
FEATURES_V2 = FeatureConfig(**{fcfg})
MODEL_V2 = ModelConfig(**{mcfg})

FEATURES_REPORT = FeatureConfig()                      # variance-MI selection, PCA/KMeans, post-alert pool
MODEL_REPORT = ModelConfig(calib_mode="insample")      # single seed, in-sample isotonic

print("report recipe :", FEATURES_REPORT, "\\n              ", MODEL_REPORT)
print("\\nv2 recipe     :", FEATURES_V2, "\\n              ", MODEL_V2)
"""))

    out.append(md("### Run both recipes on the same five chronological folds"))
    out.append(code("""
folds = step5_windows(base.open_dates)
for i, tr, va in folds:
    print(f"fold {i}: train {len(tr):5d} rows / {int(base.y.iloc[tr].sum()):3d} mules"
          f"   ->   val {len(va):5d} rows / {int(base.y.iloc[va].sum()):3d} mules")

t0 = time.time()
res_report = evaluate(base, FEATURES_REPORT, MODEL_REPORT, label="report_recipe")
res_v2 = evaluate(base, FEATURES_V2, MODEL_V2, label="phase_1_2")
print(f"\\ntotal {time.time() - t0:.0f}s")
"""))

    out.append(md("### Head-to-head scorecard"))
    out.append(code("""
def scorecard_frame(results):
    rows = []
    for r in results:
        m = r["mean"]
        rows.append({
            "recipe": r["label"],
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "mule F1": m["f1_mule_f1"],
            "macro F1": m["f1_macro_f1"],
            "balanced acc": m["f1_balanced_accuracy"],
            "accuracy": m["f1_accuracy"],
            "recall@1%": m["recall_at_1pct"],
            "recall@5%": m["recall_at_5pct"],
            "Brier": m["brier"],
            "ECE": m["ece"],
            "calib slope": m["calibration_slope"],
            "cost @R=5": m["cost_r5_opt"],
            "norm cost": m["normcost_r5"],
            "composite": r["composite"],
        })
    return pd.DataFrame(rows).set_index("recipe").T

board = scorecard_frame([res_report, res_v2])
board["delta"] = board["phase_1_2"] - board["report_recipe"]
display(board.round(4))

print(f"recorded PS2 chronological PR-AUC        : {PS2_CHRONO_PR:.4f}")
print(f"recorded locked Phase_1_stable PR-AUC    : {LOCKED_CHRONO_PR:.4f}")
print(f"Phase_1.2 chronological PR-AUC           : {res_v2['mean']['pr_auc']:.4f}")
"""))

    out.append(md("### Where the ranking gain shows up: per-fold precision–recall"))
    out.append(code("""
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 5, figsize=(16, 3.1), sharey=True)
for ax, (i, tr, va) in zip(axes, folds):
    y_va = base.y.iloc[va].to_numpy()
    for cfg_f, cfg_m, name, colour in (
        (FEATURES_REPORT, MODEL_REPORT, "report", SLATE),
        (FEATURES_V2, MODEL_V2, "v2", CRIMSON),
    ):
        X_tr, X_va, y_tr, _ = build_fold(base, tr, va, cfg_f)
        p = fit_predict(X_tr, y_tr, X_va, cfg_m)["va_blend"]
        pr, rc, _ = precision_recall_curve(y_va, p)
        ax.plot(rc, pr, color=colour, label=name, lw=1.6)
    ax.axhline(y_va.mean(), color="k", ls=":", lw=0.8)
    ax.set(title=f"fold {i} ({int(y_va.sum())} mules)", xlabel="recall")
axes[0].set_ylabel("precision"); axes[0].legend()
fig.tight_layout(); fig.savefig(FIG / "03_pr_curves.png", bbox_inches="tight"); plt.show()
"""))

    out.append(md("### Calibration: what the probability is worth"))
    out.append(code("""
fold_id, tr, va = folds[0]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (cfg_f, cfg_m, name, colour) in zip(axes, (
    (FEATURES_REPORT, MODEL_REPORT, "report: in-sample isotonic", SLATE),
    (FEATURES_V2, MODEL_V2, "v2: out-of-fold isotonic + PCHIP", CRIMSON),
)):
    X_tr, X_va, y_tr, y_va = build_fold(base, tr, va, cfg_f)
    pred = fit_predict(X_tr, y_tr, X_va, cfg_m)
    px, py = fit_calibrator(pred["cal_scores"], pred["cal_y"])
    cal = apply_calibrator(pred["va_blend"], px, py)
    yv = y_va.to_numpy()
    order_ = np.argsort(cal)
    groups = np.array_split(order_, 10)
    xs = [cal[gp].mean() for gp in groups]
    ys = [yv[gp].mean() for gp in groups]
    ax.plot([0, max(xs + ys)], [0, max(xs + ys)], "k:", lw=0.8, label="ideal")
    ax.plot(xs, ys, "o-", color=colour, lw=1.5, label="observed")
    ax.set(title=name, xlabel="predicted probability", ylabel="observed rate")
    ax.text(0.04, 0.92, f"Brier {((cal - yv) ** 2).mean():.5f}\\n"
                        f"ECE {expected_calibration_error(yv, cal):.5f}\\n"
                        f"slope {calibration_slope(yv, cal):.2f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=8)
fig.suptitle(f"Reliability on fold {fold_id}", y=1.04)
fig.tight_layout(); fig.savefig(FIG / "04_calibration.png", bbox_inches="tight"); plt.show()
"""))

    out.append(md("### The economic engine across cost ratios"))
    out.append(code("""
rows = []
for r in R_GRID:
    tag = f"r{int(r)}"
    for res, name in ((res_report, "report"), (res_v2, "v2")):
        m = res["mean"]
        rows.append({
            "R": r, "recipe": name,
            "cost_opt": m[f"cost_{tag}_opt"],
            "cost_f1": m[f"cost_{tag}_f1"],
            "cost_bayes": m[f"cost_{tag}_bayes"],
            "trivial": m[f"cost_{tag}_trivial"],
            "norm_cost": m[f"normcost_{tag}"],
            "saving_vs_trivial": m[f"savings_{tag}_vs_trivial"],
            "saving_vs_f1": m[f"savings_{tag}_vs_f1"],
        })
econ = pd.DataFrame(rows)
display(econ.pivot(index="R", columns="recipe",
                   values=["cost_opt", "norm_cost", "saving_vs_f1"]).round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, colour in (("report", SLATE), ("v2", CRIMSON)):
    s = econ[econ["recipe"] == name]
    ax[0].plot(s["R"], s["cost_opt"], "o-", color=colour, label=name)
    ax[1].plot(s["R"], s["norm_cost"], "o-", color=colour, label=name)
s = econ[econ["recipe"] == "report"]
ax[0].plot(s["R"], s["trivial"], "k:", label="trivial policy")
ax[0].set(xscale="log", title="Total operational cost per fold", xlabel="R = cost(FN) / cost(FP)",
          ylabel="cost")
ax[1].axhline(1.0, color="k", ls=":")
ax[1].set(xscale="log", title="Normalised cost (0 = perfect, 1 = worthless)",
          xlabel="R = cost(FN) / cost(FP)", ylabel="cost / trivial")
ax[0].legend(); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "05_cost_curves.png", bbox_inches="tight"); plt.show()
"""))

    out.append(md(r"""
## 7. Is the win real?

Five folds hold 57 validation mules between them, so a single mean is fragile. Three checks:
independent seed sets on the same folds, a denser nine-point walk-forward, and — for contrast
only — the order-blind stratified k-fold that produced the report's 0.8677.
"""))
    out.append(code("""
rob = []
for off in (0, 1000, 2000):
    rob.append(evaluate(base, FEATURES_V2, ModelConfig(**{**MODEL_V2.__dict__, "seed_offset": off}),
                        label=f"v2 seed{off}", verbose=False))
pr = [r["mean"]["pr_auc"] for r in rob]
print(f"v2 across three seed sets: {np.mean(pr):.4f} +/- {np.std(pr):.4f}   {[round(p, 4) for p in pr]}")

dense = dense_windows(base.open_dates, n_folds=9)
d_rep = evaluate(base, FEATURES_REPORT, MODEL_REPORT, label="report dense9", verbose=False, folds=dense)
d_v2 = evaluate(base, FEATURES_V2, MODEL_V2, label="v2 dense9", verbose=False, folds=dense)
print(f"9-point walk-forward   : report {d_rep['mean']['pr_auc']:.4f}   v2 {d_v2['mean']['pr_auc']:.4f}")

wins = sum(a["pr_auc"] > b["pr_auc"] for a, b in zip(d_v2["rows"], d_rep["rows"]))
print(f"v2 wins {wins} of {len(d_v2['rows'])} walk-forward folds")

shuf = shuffled_folds(base.y, n_splits=5, seed=0)
s_rep = evaluate(base, FEATURES_REPORT, MODEL_REPORT, label="report shuffled", verbose=False, folds=shuf)
print(f"\\nshuffled 5-fold, report recipe: PR-AUC {s_rep['mean']['pr_auc']:.4f}"
      f"  vs chronological {res_report['mean']['pr_auc']:.4f}"
      f"  -> shuffling inflates by {s_rep['mean']['pr_auc'] - res_report['mean']['pr_auc']:+.4f}")
print("That inflation is the gap between the report's 0.8677 table and its own 0.7097 line.")
"""))

    out.append(md("## 8. Train the deployable bundle and export"))
    out.append(code("""
import joblib
from sklearn.impute import SimpleImputer

tr_idx, va_idx = future_core_split(base.open_dates)
X_tr, X_va, y_tr, y_va = build_fold(base, tr_idx, va_idx, FEATURES_V2)
pred = fit_predict(X_tr, y_tr, X_va, MODEL_V2)
final = score_fold(pred["cal_y"], pred["cal_scores"], y_va.to_numpy(), pred["va_blend"])
print(f"future-core holdout: PR-AUC {final['pr_auc']:.4f}  ROC {final['roc_auc']:.4f}  "
      f"mule F1 {final['f1_mule_f1']:.3f}  normalised cost {final['normcost_r5']:.3f}")

px, py = fit_calibrator(pred["cal_scores"], pred["cal_y"])
bundle = {
    "feature_config": FEATURES_V2.__dict__,
    "model_config": MODEL_V2.__dict__,
    "columns": list(X_tr.columns),
    "pchip_x": px, "pchip_y": py,
    "cost_thresholds": {int(r): pick_cost_threshold(pred["cal_y"],
                                                    apply_calibrator(pred["cal_scores"], px, py), r)
                        for r in R_GRID},
    "holdout_metrics": final,
}
joblib.dump(bundle, PKG / "models" / "phase12_bundle.joblib", compress=3)
print("wrote", PKG / "models" / "phase12_bundle.joblib")
print("cost-optimal thresholds by ratio:", {k: round(v, 4) for k, v in bundle["cost_thresholds"].items()})
"""))

    out.append(md(r"""
## 9. Appendix — implementation source

The notebook imports the tested modules rather than restating them, so the narrative above and
the code that produced the numbers cannot drift apart. The source is printed here for review.
"""))
    out.append(code("""
import inspect
import data_layer, semantic, advanced, experiment, scorecard, cost

for mod in (data_layer, semantic, advanced, experiment, scorecard, cost):
    src = inspect.getsource(mod)
    print("=" * 100)
    print(f"### {mod.__name__}.py   ({len(src.splitlines())} lines)")
    print("=" * 100)
    print(src)
"""))

    return out


def main() -> None:
    nb = {
        "cells": cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
