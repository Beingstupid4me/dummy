"""Apples-to-apples chrono test vs Phase_1/step5 (quoted PR-AUC 0.7097).

Uses step5's exact rolling windows, step2/4 feature tracks, report HPs,
salvaged bank cats, dictionary V_cross, and an XGB+LGB+HGB ensemble.
No F3912/F2230, no SMOTE, no elapsed_days, train-window MI only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
TARGET, ID_COL = "F3924", "Unnamed: 0"
LEAKS = ["F3912", "F2230"]
BANK = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]
F3889_LAG = {"L7D": 7, "L14D": 14, "L31D": 31, "L90D": 90, "L180D": 180, "L365D": 365, "G365D": 400}
PLACEHOLDERS = {-99999999, 99999999, -9999999, 9999999, -999999, 999999, -9999, 9999}
TOP_MI, TOP_GAP, MI_CAP = 25, 25, 400
EPS = 1e-6

print("loading...")
raw = pd.read_csv(ROOT / "DataSet.csv", low_memory=False)
y = raw[TARGET].astype(int)
work = raw.drop(columns=[ID_COL, TARGET], errors="ignore").replace(list(PLACEHOLDERS), np.nan)
work = work.replace([np.inf, -np.inf], np.nan)
work = work.drop(columns=[c for c in LEAKS if c in work.columns])

desc = pd.read_excel(ROOT / "Description.xlsx")
feat_of = dict(zip(desc["Variable Name"].fillna("").astype(str), desc["Feature"].astype(str).str.strip()))


def feat(name: str) -> str | None:
    return feat_of.get(name)


work["F3889_comp_lag"] = work["F3889"].astype("string").map(F3889_LAG).astype(float)
work = work.drop(columns=["F3889"], errors="ignore")
age = work["F3889_comp_lag"].fillna(400.0) + 1.0
channel = []
for win in ("L7D", "L14D", "L31D"):
    atm = pd.to_numeric(work[feat(f"ATM_AMT_DB_{win}")], errors="coerce")
    elec = pd.to_numeric(work[feat(f"ELEC_XFER_AMT_DB_{win}")], errors="coerce")
    upi = pd.to_numeric(work[feat(f"UPI_AMT_CR_{win}")], errors="coerce")
    ratio = (atm.fillna(0) + elec.fillna(0)) / (upi.fillna(0) + EPS)
    work[f"V_cross_{win}"] = np.log1p(np.clip(ratio, 0, 1e6))
    work[f"log_atm_db_{win}"] = np.log1p(atm.clip(lower=0))
    work[f"log_elec_db_{win}"] = np.log1p(elec.clip(lower=0))
    work[f"log_upi_cr_{win}"] = np.log1p(upi.clip(lower=0))
    atm_txn = pd.to_numeric(work[feat(f"ATM_TXNS_{win}")], errors="coerce")
    upi_txn = pd.to_numeric(work[feat(f"UPI_XFER_TXNS_{win}")], errors="coerce")
    elec_txn = pd.to_numeric(work[feat(f"ELEC_XFER_TXNS_{win}")], errors="coerce")
    work[f"txn_accel_{win}"] = np.log1p(np.clip((atm_txn.fillna(0) + upi_txn.fillna(0) + elec_txn.fillna(0)) / age, 0, 1e4))
    channel += [f"V_cross_{win}", f"log_atm_db_{win}", f"log_elec_db_{win}", f"log_upi_cr_{win}", f"txn_accel_{win}"]
work["V_cross"] = work[["V_cross_L7D", "V_cross_L14D", "V_cross_L31D"]].mean(axis=1)
work["txn_accel"] = work[["txn_accel_L7D", "txn_accel_L14D", "txn_accel_L31D"]].mean(axis=1)
channel += ["V_cross", "txn_accel", "F3889_comp_lag"]

object_cols = work.select_dtypes(include=["object", "string", "category"]).columns.tolist()
temporal_cols, categorical_cols = [], []
for col in object_cols:
    parsed = pd.to_datetime(work[col], errors="coerce")
    if parsed.notna().mean() >= 0.7 and parsed.nunique(dropna=True) > 1:
        temporal_cols.append(col)
    else:
        categorical_cols.append(col)

numeric_base = work.apply(pd.to_numeric, errors="coerce")
extreme = numeric_base.abs() >= 1e7
for col in numeric_base.columns:
    s = numeric_base[col]
    ext = s[s.abs() >= 1e7].dropna()
    if ext.empty:
        continue
    vc = ext.value_counts()
    if vc.iloc[0] / s.notna().sum() >= 0.002:
        numeric_base[col] = s.replace(vc.index[0], np.nan)

vals = numeric_base.to_numpy(dtype=float)
mask = ~np.isnan(vals)
nm = mask.sum(axis=1)
tot = max(vals.shape[1], 1)
with np.errstate(all="ignore"):
    q25 = np.nanpercentile(vals, 25, axis=1)
    q75 = np.nanpercentile(vals, 75, axis=1)
    row_stats = pd.DataFrame({
        "row_non_missing_count": nm,
        "row_missing_rate": 1.0 - nm / tot,
        "row_zero_rate": np.where(nm > 0, np.nansum(vals == 0, axis=1) / nm, 0),
        "row_positive_rate": np.where(nm > 0, np.nansum(vals > 0, axis=1) / nm, 0),
        "row_negative_rate": np.where(nm > 0, np.nansum(vals < 0, axis=1) / nm, 0),
        "row_mean": np.nanmean(vals, axis=1),
        "row_std": np.nanstd(vals, axis=1),
        "row_min": np.nanmin(vals, axis=1),
        "row_max": np.nanmax(vals, axis=1),
        "row_median": np.nanmedian(vals, axis=1),
        "row_q25": q25,
        "row_q75": q75,
        "row_iqr": q75 - q25,
        "row_abs_mean": np.nanmean(np.abs(vals), axis=1),
    }, index=numeric_base.index)

open_dates = pd.to_datetime(work["F3888"], errors="coerce")
date_sorted = open_dates.argsort().to_numpy()
n = len(date_sorted)
print("cats", categorical_cols, "temporal", temporal_cols, "channel", len(channel))


def step5_windows():
    """Exact rolling windows from Phase_1/step5_calibration_cost_chronological.ipynb."""
    out = []
    for fold in range(5):
        split_point = int(0.8 * n) + (fold - 2) * int(0.04 * n)
        split_point = max(int(0.5 * n), min(int(0.95 * n), split_point))
        tr = date_sorted[:split_point]
        va = date_sorted[split_point: split_point + int(0.2 * n)]
        if len(va) < 50:
            continue
        out.append((fold, tr, va))
    return out


def select_spec(idx, target):
    frame = numeric_base.iloc[idx]
    forced = [c for c in BANK + channel if c in frame.columns and frame[c].notna().any()]
    nunique = frame.nunique(dropna=True)
    miss = frame.isna().mean()
    usable = [c for c in frame.columns if nunique.get(c, 0) > 1 and miss.get(c, 1) < 0.95]
    cand = frame[usable]
    if cand.shape[1] > MI_CAP:
        cand = cand[cand.var(skipna=True).sort_values(ascending=False).head(MI_CAP).index]
    mi = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(cand), target, random_state=RANDOM_STATE)
    top_mi = pd.Series(mi, index=cand.columns).sort_values(ascending=False).head(TOP_MI).index.tolist()
    gap = (frame.loc[target == 1].isna().mean() - frame.loc[target == 0].isna().mean()).abs().sort_values(ascending=False)
    top_gap = gap.head(TOP_GAP).index.tolist()
    selected = []
    for c in forced + top_mi + top_gap:
        if c not in selected and c in frame.columns:
            selected.append(c)
    return selected, top_mi, top_gap


def encode_cats(tr_raw, va_raw, y_tr):
    parts_tr, parts_va = [], []
    for col in categorical_cols:
        tr = tr_raw[col].astype("string").fillna("MISSING")
        va = va_raw[col].astype("string").fillna("MISSING")
        if tr.nunique(dropna=False) <= 12:
            mapping = {v: i for i, v in enumerate(sorted(tr.unique().tolist()))}
            parts_tr.append(tr.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
            parts_va.append(va.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
        else:
            g = float(y_tr.mean())
            stats = pd.DataFrame({"c": tr, "y": y_tr.to_numpy()}).groupby("c")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + 20.0 * g) / (stats["count"] + 20.0)
            parts_tr.append(tr.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
            parts_va.append(va.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
    if parts_tr:
        return pd.concat(parts_tr, axis=1), pd.concat(parts_va, axis=1)
    return pd.DataFrame(index=tr_raw.index), pd.DataFrame(index=va_raw.index)


def cyclical(raw, cols):
    parts = []
    for col in cols:
        p = pd.to_datetime(raw[col], errors="coerce")
        dow, month = p.dt.dayofweek.astype(float), p.dt.month.astype(float)
        parts.append(pd.DataFrame({
            f"{col}_dow_sin": np.sin(2 * np.pi * dow / 7),
            f"{col}_dow_cos": np.cos(2 * np.pi * dow / 7),
            f"{col}_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            f"{col}_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        }, index=raw.index))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)


def add_if_pca(tr, va, y_tr):
    imp, scaler = SimpleImputer(strategy="median"), RobustScaler()
    tr_s = scaler.fit_transform(imp.fit_transform(tr))
    va_s = scaler.transform(imp.transform(va))
    iso = IsolationForest(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    normal = (y_tr == 0).to_numpy()
    iso.fit(tr_s[normal] if normal.sum() >= 10 else tr_s)
    tr, va = tr.copy(), va.copy()
    tr["iso_anomaly_score"] = -iso.score_samples(tr_s)
    va["iso_anomaly_score"] = -iso.score_samples(va_s)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    tr_p, va_p = pca.fit_transform(tr_s), pca.transform(va_s)
    for i in range(3):
        tr[f"feature_pc_{i+1}"] = tr_p[:, i]
        va[f"feature_pc_{i+1}"] = va_p[:, i]
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    tr_d, va_d = km.fit_transform(tr_s), km.transform(va_s)
    for i in range(3):
        tr[f"feature_kmeans_dist_c{i+1}"] = tr_d[:, i]
        va[f"feature_kmeans_dist_c{i+1}"] = va_d[:, i]
    return tr, va


def make_xy(tr_idx, va_idx):
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    selected, top_mi, top_gap = select_spec(tr_idx, y_tr)
    flags_tr = numeric_base.iloc[tr_idx][top_gap].isna().astype(int).add_prefix("miss_")
    flags_va = numeric_base.iloc[va_idx][top_gap].isna().astype(int).add_prefix("miss_")
    base_tr = pd.concat([numeric_base.iloc[tr_idx][selected], row_stats.iloc[tr_idx], flags_tr], axis=1)
    base_va = pd.concat([numeric_base.iloc[va_idx][selected], row_stats.iloc[va_idx], flags_va], axis=1)
    cat_tr, cat_va = encode_cats(work.iloc[tr_idx], work.iloc[va_idx], y_tr)
    tmp_tr, tmp_va = cyclical(work.iloc[tr_idx], temporal_cols), cyclical(work.iloc[va_idx], temporal_cols)
    tr = pd.concat([base_tr, cat_tr, tmp_tr], axis=1)
    va = pd.concat([base_va, cat_va, tmp_va], axis=1)
    # train-only interactions like step3, limited
    for mi_col in top_mi[:6]:
        if mi_col in tr.columns:
            for stat in ("row_missing_rate", "row_zero_rate", "row_iqr"):
                tr[f"ix_{mi_col}_{stat}"] = tr[mi_col] * tr[stat]
                va[f"ix_{mi_col}_{stat}"] = va[mi_col] * va[stat]
    tr, va = add_if_pca(tr, va, y_tr)
    tr = tr.loc[:, tr.isna().mean() < 1]
    va = va.reindex(columns=tr.columns)
    return tr, va, y_tr, y_va


def fit_models(X_tr, y_tr, X_va):
    spw = float((y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1))
    imp = SimpleImputer(strategy="median")
    cols = list(X_tr.columns)
    Xtr = pd.DataFrame(imp.fit_transform(X_tr), columns=cols, index=X_tr.index)
    Xva = pd.DataFrame(imp.transform(X_va), columns=cols, index=X_va.index)
    xgb_m = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", scale_pos_weight=spw, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
    )
    lgb_m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        objective="binary", scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    hgb_m = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, max_iter=350, l2_regularization=1.0,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    xgb_m.fit(Xtr, y_tr)
    lgb_m.fit(Xtr, y_tr)
    hgb_m.fit(Xtr, y_tr)
    px = np.vstack([
        xgb_m.predict_proba(Xtr)[:, 1],
        lgb_m.predict_proba(Xtr)[:, 1],
        hgb_m.predict_proba(Xtr)[:, 1],
    ])
    vx = np.vstack([
        xgb_m.predict_proba(Xva)[:, 1],
        lgb_m.predict_proba(Xva)[:, 1],
        hgb_m.predict_proba(Xva)[:, 1],
    ])
    return px, vx, imp, cols


def blend_on_inner(p_inner, y_inner, p_va):
    """Search mix on an inner time tail; fallback to report 0.6/0.4/0."""
    best, best_w = -1.0, (0.6, 0.4, 0.0)
    grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    for wx in grid:
        for wl in grid:
            wh = 1.0 - wx - wl
            if wh < -1e-9 or wh > 1.0 + 1e-9:
                continue
            wh = max(0.0, wh)
            s = wx * p_inner[0] + wl * p_inner[1] + wh * p_inner[2]
            ap = average_precision_score(y_inner, s)
            if ap > best:
                best, best_w = ap, (wx, wl, wh)
    wx, wl, wh = best_w
    return wx * p_va[0] + wl * p_va[1] + wh * p_va[2], best_w, best


def report_blend(p):
    return 0.6 * p[0] + 0.4 * p[1]


print("\n=== Phase_1 step5 windows (the 0.7097 protocol) ===")
rows = []
for fold, tr, va in step5_windows():
    y_va = y.iloc[va]
    print(f"fold {fold}: train {len(tr)} mules={int(y.iloc[tr].sum())} | val {len(va)} mules={int(y_va.sum())}", flush=True)
    # Train on the full past window, same as step5. Blend weights are declared
    # up front (not tuned on val): report 0.6/0.4, plus HGB mixes.
    X_tr, X_va, y_tr, y_va = make_xy(tr, va)
    p_tr, p_va, _, _ = fit_models(X_tr, y_tr, X_va)
    rpt_va = report_blend(p_va)
    mix_hgb = 0.45 * p_va[0] + 0.35 * p_va[1] + 0.20 * p_va[2]
    mix_eq = (p_va[0] + p_va[1] + p_va[2]) / 3.0
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    cal.fit(report_blend(p_tr), y_tr.to_numpy())
    rpt_cal = cal.predict(rpt_va)
    row = {
        "fold": fold,
        "val_mules": int(y_va.sum()),
        "n_features": int(X_tr.shape[1]),
        "pr_auc_xgb": float(average_precision_score(y_va, p_va[0])),
        "pr_auc_lgb": float(average_precision_score(y_va, p_va[1])),
        "pr_auc_hgb": float(average_precision_score(y_va, p_va[2])),
        "pr_auc_report_blend": float(average_precision_score(y_va, rpt_va)),
        "pr_auc_report_blend_cal": float(average_precision_score(y_va, rpt_cal)),
        "pr_auc_mix_hgb": float(average_precision_score(y_va, mix_hgb)),
        "pr_auc_equal": float(average_precision_score(y_va, mix_eq)),
        "roc_report_blend": float(roc_auc_score(y_va, rpt_va)),
    }
    rows.append(row)
    print(
        f"  xgb={row['pr_auc_xgb']:.4f} lgb={row['pr_auc_lgb']:.4f} hgb={row['pr_auc_hgb']:.4f} "
        f"0.6/0.4={row['pr_auc_report_blend']:.4f} mixHGB={row['pr_auc_mix_hgb']:.4f} eq={row['pr_auc_equal']:.4f}",
        flush=True,
    )

df = pd.DataFrame(rows)
print("\nper-fold")
print(df[["fold", "pr_auc_report_blend", "pr_auc_mix_hgb", "pr_auc_equal", "roc_report_blend"]].to_string(index=False))
print("\nmeans vs PS2 chrono 0.7097")
for col in ["pr_auc_xgb", "pr_auc_lgb", "pr_auc_hgb", "pr_auc_report_blend", "pr_auc_report_blend_cal", "pr_auc_mix_hgb", "pr_auc_equal"]:
    m = df[col].mean()
    print(f"  {col:28s} {m:.4f}  beat={m >= 0.7097}")

out = Path(__file__).with_name("beat_chrono_results.json")
out.write_text(json.dumps({"rows": rows, "means": df.mean(numeric_only=True).to_dict()}, indent=2, default=float), encoding="utf-8")
print("wrote", out)
