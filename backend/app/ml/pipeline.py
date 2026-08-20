"""SentinelFlow feature reconstruction and GBDT blend.

Used by Phase_1_enhancement (chrono bake-off) and the FastAPI registry.
Confirmed leaks stay out of M1: F3912, F2230. Resolution flags F3913–F3915
are also excluded. Dictionary channel physics come from Description.xlsx.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.calibration import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb

RANDOM_STATE = 42
TARGET, ID_COL = "F3924", "Unnamed: 0"
CONFIRMED_LEAKY = ["F3912", "F2230"]
RESOLUTION_LEAKY = ["F3913", "F3914", "F3915"]
M3_LEAKY = ["F3912", "F2230", "F3886", "F3889", "F3891", "F3892"]
BANK_FINALIZED = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3894",
]
OCC_BAL = [f"F{i}" for i in range(3880, 3886)]
DEV_COLS = [f"F{i}" for i in range(3862, 3871)]
TMS_FLAGS = [f"F{i}" for i in range(3900, 3912)] + [f"F{i}" for i in range(3916, 3924)]
POST_ALERT = [f"F{i}" for i in range(3895, 3924)]
PLACEHOLDERS = {-99999999, 99999999, -9999999, 9999999, -999999, 999999, -9999, 9999}
F3889_LAG = {"L7D": 7, "L14D": 14, "L31D": 31, "L90D": 90, "L180D": 180, "L365D": 365, "G365D": 400}
TOP_MI, TOP_GAP, MI_CAP = 25, 25, 400
EPS = 1e-6
BLEND = (0.6, 0.4)
COST_FN_RATIO, COST_FP_BASE = 5.0, 1.0
WINDOWS = ("L7D", "L14D", "L31D")
CHANNEL_AMT = {
    "atm_db": "ATM_AMT_DB_{w}",
    "atm_cr": "ATM_AMT_CR_{w}",
    "elec_db": "ELEC_XFER_AMT_DB_{w}",
    "elec_cr": "ELEC_XFER_AMT_CR_{w}",
    "upi_db": "UPI_AMT_DB_{w}",
    "upi_cr": "UPI_AMT_CR_{w}",
    "pos_db": "POS_PYMT_AMT_DB_{w}",
    "net_db": "NET_BNKING_AMT_DB_{w}",
    "cash_db": "CASH_AMT_DB_{w}",
    "mob": "MOB_BNKING_AMT_{w}",
}
CHANNEL_TXN = {
    "atm": "ATM_TXNS_{w}",
    "elec": "ELEC_XFER_TXNS_{w}",
    "upi": "UPI_XFER_TXNS_{w}",
    "pos": "POS_PYMT_TXNS_{w}",
    "net": "NET_BNKING_TXNS_{w}",
    "cash": "CASH_TXNS_{w}",
    "mob": "MOB_BNKING_TXNS_{w}",
}
DICT_FORCE = [
    "F3889_comp_lag",
    "V_cross",
    "txn_accel",
    "V_cross_L7D",
    "V_cross_L14D",
    "V_cross_L31D",
    "log_atm_db_L7D",
    "log_elec_db_L7D",
    "log_upi_cr_L7D",
    "txn_accel_L7D",
    "txn_accel_L14D",
    "txn_accel_L31D",
    "V_pos_L7D",
    "V_net_L7D",
    "V_cash_L7D",
    "burst_upi_cr",
    "burst_atm_db",
    "burst_elec_db",
    "ch_entropy_L7D",
    "upi_db_cr_imb_L7D",
    "v_cross_curv",
    "upi_cr_curv",
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "DataSet.csv").exists():
            return p
    return Path.cwd()


@dataclass
class FeatureSpec:
    include_elapsed: bool = False
    include_leaky: bool = False
    include_tms: bool = False
    use_pca: bool = True
    feature_allow: list[str] | None = None


@dataclass
class FittedBundle:
    xgb: xgb.XGBClassifier
    lgb: lgb.LGBMClassifier
    imputer: SimpleImputer
    iso: IsolationForest | None
    iso_scaler: RobustScaler | None
    columns: list[str]
    cat_maps: dict
    te_maps: dict
    te_global: float
    pchip_x: np.ndarray
    pchip_y: np.ndarray
    threshold: float
    metrics: dict = field(default_factory=dict)
    spec: FeatureSpec = field(default_factory=FeatureSpec)
    train_idx: np.ndarray | None = None


class PchipAbort(RuntimeError):
    pass


def load_dictionary(root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root()
    return pd.read_excel(root / "Description.xlsx", sheet_name="Data_Dicitionary")


def name_to_col(desc: pd.DataFrame) -> dict[str, str]:
    return dict(
        zip(
            desc["Variable Name"].fillna("").astype(str),
            desc["Feature"].astype(str).str.strip(),
        )
    )


def _num(frame: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def add_dictionary_tracks(work: pd.DataFrame, lookup: dict[str, str]) -> list[str]:
    """Channel physics from Description.xlsx. No elapsed_days."""
    made: list[str] = []

    if "F3889" in work.columns:
        work["F3889_comp_lag"] = work["F3889"].astype("string").map(F3889_LAG).astype(float)
        made.append("F3889_comp_lag")
    age = work.get("F3889_comp_lag", pd.Series(400.0, index=work.index)).fillna(400.0) + 1.0

    def grab(template: str, win: str) -> pd.Series:
        return _num(work, lookup.get(template.format(w=win)))

    for win in WINDOWS:
        amt = {k: grab(t, win).fillna(0).clip(lower=0) for k, t in CHANNEL_AMT.items()}
        txn = {k: grab(t, win).fillna(0).clip(lower=0) for k, t in CHANNEL_TXN.items()}

        work[f"V_cross_{win}"] = np.log1p(
            np.clip((amt["atm_db"] + amt["elec_db"]) / (amt["upi_cr"] + EPS), 0, 1e6)
        )
        work[f"V_pos_{win}"] = np.log1p(np.clip(amt["pos_db"] / (amt["upi_cr"] + EPS), 0, 1e6))
        work[f"V_net_{win}"] = np.log1p(np.clip(amt["net_db"] / (amt["upi_cr"] + EPS), 0, 1e6))
        work[f"V_cash_{win}"] = np.log1p(np.clip(amt["cash_db"] / (amt["upi_cr"] + EPS), 0, 1e6))
        work[f"log_atm_db_{win}"] = np.log1p(amt["atm_db"])
        work[f"log_elec_db_{win}"] = np.log1p(amt["elec_db"])
        work[f"log_upi_cr_{win}"] = np.log1p(amt["upi_cr"])
        work[f"log_upi_db_{win}"] = np.log1p(amt["upi_db"])
        work[f"log_pos_db_{win}"] = np.log1p(amt["pos_db"])
        work[f"log_net_db_{win}"] = np.log1p(amt["net_db"])
        work[f"txn_accel_{win}"] = np.log1p(
            np.clip((txn["atm"] + txn["upi"] + txn["elec"]) / age, 0, 1e4)
        )
        upi_imb = (amt["upi_db"] - amt["upi_cr"]) / (amt["upi_db"] + amt["upi_cr"] + EPS)
        work[f"upi_db_cr_imb_{win}"] = upi_imb.clip(-1, 1)
        stack = np.vstack(
            [txn["atm"], txn["upi"], txn["elec"], txn["pos"], txn["net"], txn["cash"]]
        ).T
        tot = stack.sum(axis=1, keepdims=True) + EPS
        shares = stack / tot
        work[f"ch_entropy_{win}"] = -(shares * np.log(shares + EPS)).sum(axis=1)
        made += [
            f"V_cross_{win}", f"V_pos_{win}", f"V_net_{win}", f"V_cash_{win}",
            f"log_atm_db_{win}", f"log_elec_db_{win}", f"log_upi_cr_{win}",
            f"log_upi_db_{win}", f"log_pos_db_{win}", f"log_net_db_{win}",
            f"txn_accel_{win}", f"upi_db_cr_imb_{win}", f"ch_entropy_{win}",
        ]

    work["V_cross"] = work[[f"V_cross_{w}" for w in WINDOWS]].mean(axis=1)
    work["txn_accel"] = work[[f"txn_accel_{w}" for w in WINDOWS]].mean(axis=1)
    work["burst_upi_cr"] = work["log_upi_cr_L7D"] - work["log_upi_cr_L31D"]
    work["burst_atm_db"] = work["log_atm_db_L7D"] - work["log_atm_db_L31D"]
    work["burst_elec_db"] = work["log_elec_db_L7D"] - work["log_elec_db_L31D"]
    work["v_cross_curv"] = (work["V_cross_L7D"] - work["V_cross_L14D"]) - (
        work["V_cross_L14D"] - work["V_cross_L31D"]
    )
    work["upi_cr_curv"] = (work["log_upi_cr_L7D"] - work["log_upi_cr_L14D"]) - (
        work["log_upi_cr_L14D"] - work["log_upi_cr_L31D"]
    )
    made += [
        "V_cross", "txn_accel", "burst_upi_cr", "burst_atm_db", "burst_elec_db",
        "v_cross_curv", "upi_cr_curv",
    ]
    return made


def row_moments(numeric: pd.DataFrame) -> pd.DataFrame:
    vals = numeric.to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    nm = mask.sum(axis=1)
    tot = max(vals.shape[1], 1)
    with np.errstate(all="ignore"):
        q25 = np.nanpercentile(vals, 25, axis=1)
        q75 = np.nanpercentile(vals, 75, axis=1)
        return pd.DataFrame(
            {
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
            },
            index=numeric.index,
        )


def load_frame(root: Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str], list[str]]:
    root = root or repo_root()
    raw = pd.read_csv(root / "DataSet.csv", low_memory=False)
    y = raw[TARGET].astype(int)
    work = raw.drop(columns=[ID_COL, TARGET], errors="ignore").replace(list(PLACEHOLDERS), np.nan)
    work = work.replace([np.inf, -np.inf], np.nan)
    desc = load_dictionary(root)
    lookup = name_to_col(desc)
    made = add_dictionary_tracks(work, lookup)
    drop = [c for c in RESOLUTION_LEAKY if c in work.columns]
    work = work.drop(columns=drop, errors="ignore")
    open_dates = pd.to_datetime(work["F3888"], errors="coerce")
    return work, y, open_dates, lookup, made


def classify_columns(work: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    object_cols = work.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    temporal = ["F3888"] if "F3888" in work.columns else []
    categorical = [c for c in object_cols if c not in temporal]
    numeric = work.drop(columns=temporal, errors="ignore").apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        s = numeric[col]
        ext = s[s.abs() >= 1e7].dropna()
        if ext.empty:
            continue
        vc = ext.value_counts()
        if vc.iloc[0] / max(int(s.notna().sum()), 1) >= 0.002:
            numeric[col] = s.replace(vc.index[0], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    return categorical, temporal, numeric


def step5_windows(open_dates: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray]]:
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    out = []
    for fold in range(5):
        split = int(0.8 * n) + (fold - 2) * int(0.04 * n)
        split = max(int(0.5 * n), min(int(0.95 * n), split))
        tr = order[:split]
        va = order[split: split + int(0.2 * n)]
        if len(va) < 50:
            continue
        out.append((fold, tr, va))
    return out


def future_core_split(open_dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    return order[: int(0.7 * n)], order[int(0.7 * n): int(0.9 * n)]


def cyclical(raw: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    parts = []
    for col in cols:
        p = pd.to_datetime(raw[col], errors="coerce")
        dow, month = p.dt.dayofweek.astype(float), p.dt.month.astype(float)
        parts.append(
            pd.DataFrame(
                {
                    f"{col}_dow_sin": np.sin(2 * np.pi * dow / 7),
                    f"{col}_dow_cos": np.cos(2 * np.pi * dow / 7),
                    f"{col}_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
                    f"{col}_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
                },
                index=raw.index,
            )
        )
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)


def encode_cats(
    tr_raw: pd.DataFrame,
    va_raw: pd.DataFrame,
    y_tr: pd.Series,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, float]:
    parts_tr, parts_va = [], []
    ord_maps, te_maps = {}, {}
    g = float(y_tr.mean())
    for col in categorical_cols:
        tr = tr_raw[col].astype("string").fillna("MISSING")
        va = va_raw[col].astype("string").fillna("MISSING")
        if tr.nunique(dropna=False) <= 12:
            mapping = {v: i for i, v in enumerate(sorted(tr.unique().tolist()))}
            ord_maps[col] = mapping
            parts_tr.append(tr.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
            parts_va.append(va.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
        else:
            stats = pd.DataFrame({"c": tr, "y": y_tr.to_numpy()}).groupby("c")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + 20.0 * g) / (stats["count"] + 20.0)
            te_maps[col] = smooth.to_dict()
            parts_tr.append(tr.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
            parts_va.append(va.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
    if parts_tr:
        return pd.concat(parts_tr, axis=1), pd.concat(parts_va, axis=1), ord_maps, te_maps, g
    return pd.DataFrame(index=tr_raw.index), pd.DataFrame(index=va_raw.index), {}, {}, g


def select_spec(frame: pd.DataFrame, target: pd.Series, forced: list[str]) -> tuple[list[str], list[str], list[str]]:
    nunique = frame.nunique(dropna=True)
    miss = frame.isna().mean()
    usable = [c for c in frame.columns if nunique.get(c, 0) > 1 and miss.get(c, 1) < 0.95]
    cand = frame[usable]
    if cand.shape[1] > MI_CAP:
        cand = cand[cand.var(skipna=True).sort_values(ascending=False).head(MI_CAP).index]
    mi = mutual_info_classif(
        SimpleImputer(strategy="median").fit_transform(cand),
        target,
        random_state=RANDOM_STATE,
    )
    top_mi = pd.Series(mi, index=cand.columns).sort_values(ascending=False).head(TOP_MI).index.tolist()
    gap = (frame.loc[target == 1].isna().mean() - frame.loc[target == 0].isna().mean()).abs().sort_values(
        ascending=False
    )
    top_gap = gap.head(TOP_GAP).index.tolist()
    selected = []
    for c in list(forced) + top_mi + top_gap:
        if c not in selected and c in frame.columns:
            selected.append(c)
    return selected, top_mi, top_gap


def oversample(X: pd.DataFrame, y: pd.Series, ratio: float) -> tuple[pd.DataFrame, pd.Series]:
    if ratio <= 0:
        return X, y
    pos = y[y == 1].index
    neg_n = int((y == 0).sum())
    need = max(int(ratio * neg_n) - len(pos), 0)
    if need <= 0 or len(pos) == 0:
        return X, y
    rng = np.random.default_rng(RANDOM_STATE)
    extra = rng.choice(pos.to_numpy(), size=need, replace=True)
    idx = np.concatenate([X.index.to_numpy(), extra])
    return X.loc[idx], y.loc[idx]


def fit_pchip(raw: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y)
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    uniq_x, idx = np.unique(xs, return_index=True)
    uniq_y = ys[idx]
    if len(uniq_x) < 2:
        raise PchipAbort("isotonic collapsed; aborting retrain")
    spline = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
    grid = np.linspace(float(uniq_x.min()), float(uniq_x.max()), 64)
    deriv = spline.derivative()(grid)
    if not np.all(deriv >= -1e-7):
        raise PchipAbort("PCHIP f'(x) not strictly positive")
    return uniq_x, uniq_y


def apply_pchip(raw: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    spline = PchipInterpolator(px, py, extrapolate=True)
    return np.clip(spline(raw), 0.0, 1.0)


def pchip_deriv(raw: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    spline = PchipInterpolator(px, py, extrapolate=True)
    return spline.derivative()(raw)


def cost_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.clip(np.quantile(p, np.linspace(0.05, 0.95, 40)), 0.08, 0.45))
    best_t, best_c = 0.32, 1e18
    for t in grid:
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        c = fn * COST_FN_RATIO * COST_FP_BASE + fp * COST_FP_BASE
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def _fit_trees(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    n_estimators: int,
    early_stop: bool,
) -> tuple[xgb.XGBClassifier, lgb.LGBMClassifier, SimpleImputer]:
    spw = float((y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1))
    imp = SimpleImputer(strategy="median")
    cols = list(X_tr.columns)
    Xtr = pd.DataFrame(imp.fit_transform(X_tr), columns=cols, index=X_tr.index)
    fit_x, fit_y = Xtr, y_tr
    eval_set_x = None
    if early_stop and len(Xtr) > 200:
        cut = int(0.88 * len(Xtr))
        fit_x, ev_x = Xtr.iloc[:cut], Xtr.iloc[cut:]
        fit_y, ev_y = y_tr.iloc[:cut], y_tr.iloc[cut:]
        if int(ev_y.sum()) >= 2:
            eval_set_x = (ev_x, ev_y)
        else:
            fit_x, fit_y = Xtr, y_tr

    xgb_kw = dict(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    if eval_set_x is not None:
        xgb_m = xgb.XGBClassifier(**xgb_kw, early_stopping_rounds=40)
        xgb_m.fit(fit_x, fit_y, eval_set=[eval_set_x], verbose=False)
        lgb_m = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=-1,
        )
        lgb_m.fit(
            fit_x,
            fit_y,
            eval_set=[eval_set_x],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
    else:
        xgb_m = xgb.XGBClassifier(**xgb_kw)
        xgb_m.fit(fit_x, fit_y)
        lgb_m = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=-1,
        )
        lgb_m.fit(fit_x, fit_y)
    return xgb_m, lgb_m, imp


def overlay_txn(row: pd.DataFrame, channel: str, amount: float) -> pd.DataFrame:
    """Fold a live payment into the account's L7 snapshot (history + this txn)."""
    out = row.copy()
    a = float(np.log1p(max(amount, 0.0)))
    ch = channel.upper()

    def bump(col: str, delta: float) -> None:
        if col in out.columns:
            out[col] = out[col].astype(float).fillna(0.0) + delta

    if ch == "UPI":
        bump("log_upi_cr_L7D", a * 0.18)
        bump("burst_upi_cr", a * 0.10)
        bump("V_cross", -a * 0.04)
    elif ch == "ATM":
        bump("log_atm_db_L7D", a * 0.18)
        bump("burst_atm_db", a * 0.10)
        bump("V_cross", a * 0.10)
    elif ch in ("IMPS", "NEFT"):
        bump("log_elec_db_L7D", a * 0.18)
        bump("burst_elec_db", a * 0.10)
        bump("V_cross", a * 0.10)
    elif ch == "NETBANK":
        bump("log_net_db_L7D", a * 0.18)
        bump("V_net_L7D", a * 0.08)
    bump("txn_accel", 0.06)
    bump("txn_accel_L7D", 0.06)
    if "V_cross" in out.columns and "txn_accel" in out.columns:
        out["v_cross_x_accel"] = out["V_cross"].astype(float) * out["txn_accel"].astype(float)
    return out


def blend_proba(xgb_m, lgb_m, X: pd.DataFrame, weights: tuple[float, float] = BLEND) -> np.ndarray:
    return weights[0] * xgb_m.predict_proba(X)[:, 1] + weights[1] * lgb_m.predict_proba(X)[:, 1]


def rank_blend(xgb_m, lgb_m, X: pd.DataFrame, weights: tuple[float, float] = BLEND) -> np.ndarray:
    px = pd.Series(xgb_m.predict_proba(X)[:, 1]).rank(method="average", pct=True).to_numpy()
    pl = pd.Series(lgb_m.predict_proba(X)[:, 1]).rank(method="average", pct=True).to_numpy()
    return weights[0] * px + weights[1] * pl


def make_xy(
    work: pd.DataFrame,
    numeric: pd.DataFrame,
    moments: pd.DataFrame,
    y: pd.Series,
    open_dates: pd.Series,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    categorical_cols: list[str],
    temporal_cols: list[str],
    dict_cols: list[str],
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, dict, float, IsolationForest | None, RobustScaler | None]:
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    numeric = numeric.copy()
    leak_cols = list(CONFIRMED_LEAKY)
    if not spec.include_leaky:
        numeric = numeric.drop(columns=[c for c in leak_cols if c in numeric.columns], errors="ignore")
    if not spec.include_tms:
        numeric = numeric.drop(columns=[c for c in POST_ALERT if c in numeric.columns], errors="ignore")
    forced = [c for c in BANK_FINALIZED + OCC_BAL + DICT_FORCE if c in numeric.columns]
    if spec.include_tms:
        forced += [c for c in TMS_FLAGS if c in numeric.columns]
    if spec.include_elapsed:
        elapsed = (open_dates - open_dates.min()).dt.days.astype(float)
        numeric = numeric.copy()
        numeric["elapsed_days"] = elapsed
        forced.append("elapsed_days")
    if spec.include_leaky:
        for c in M3_LEAKY:
            if c in work.columns:
                numeric = numeric.copy()
                numeric[c] = pd.to_numeric(work[c], errors="coerce")
                forced.append(c)

    selected, top_mi, top_gap = select_spec(numeric.iloc[tr_idx], y_tr, forced)
    cats = list(categorical_cols)
    if not spec.include_leaky:
        cats = [c for c in cats if c not in leak_cols]
    if not spec.include_tms:
        cats = [c for c in cats if c not in POST_ALERT]
    cat_tr, cat_va, ord_maps, te_maps, g = encode_cats(
        work.iloc[tr_idx], work.iloc[va_idx], y_tr, cats
    )
    flags_tr = numeric.iloc[tr_idx][top_gap].isna().astype(int).add_prefix("miss_")
    flags_va = numeric.iloc[va_idx][top_gap].isna().astype(int).add_prefix("miss_")
    base_tr = pd.concat([numeric.iloc[tr_idx][selected], moments.iloc[tr_idx], flags_tr], axis=1)
    base_va = pd.concat([numeric.iloc[va_idx][selected], moments.iloc[va_idx], flags_va], axis=1)
    temps = [c for c in temporal_cols if c not in leak_cols and c not in POST_ALERT and c != "F2230"]
    temps = [c for c in temps if c == "F3888"]
    if "F3888" in work.columns and "F3888" not in temps:
        temps = ["F3888"]
    tmp_tr, tmp_va = cyclical(work.iloc[tr_idx], temps), cyclical(work.iloc[va_idx], temps)
    tr = pd.concat([base_tr, cat_tr, tmp_tr], axis=1)
    va = pd.concat([base_va, cat_va, tmp_va], axis=1)
    if "V_cross" in tr.columns:
        ref = tr["V_cross"].dropna().sort_values().to_numpy()
        if len(ref) > 8:
            tr = tr.copy()
            va = va.copy()
            tr["V_cross_pct"] = np.searchsorted(ref, tr["V_cross"].fillna(ref[0]).to_numpy(), side="right") / len(ref)
            va["V_cross_pct"] = np.searchsorted(ref, va["V_cross"].fillna(ref[0]).to_numpy(), side="right") / len(ref)
    if "V_cross" in tr.columns and "txn_accel" in tr.columns:
        tr = tr.copy()
        va = va.copy()
        tr["v_cross_x_accel"] = tr["V_cross"].astype(float) * tr["txn_accel"].astype(float)
        va["v_cross_x_accel"] = va["V_cross"].astype(float) * va["txn_accel"].astype(float)
    for mi_col in top_mi[:5]:
        if mi_col in tr.columns:
            for stat in ("row_missing_rate", "row_zero_rate"):
                tr[f"ix_{mi_col}_{stat}"] = tr[mi_col] * tr[stat]
                va[f"ix_{mi_col}_{stat}"] = va[mi_col] * va[stat]

    iso_m, iso_scaler = None, None
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    tr_s = scaler.fit_transform(imp.fit_transform(tr))
    va_s = scaler.transform(imp.transform(va))
    iso_m = IsolationForest(n_estimators=150, random_state=RANDOM_STATE, n_jobs=4)
    normal = (y_tr == 0).to_numpy()
    iso_m.fit(tr_s[normal] if normal.sum() >= 10 else tr_s)
    tr = tr.copy()
    va = va.copy()
    tr["iso_anomaly_score"] = -iso_m.score_samples(tr_s)
    va["iso_anomaly_score"] = -iso_m.score_samples(va_s)
    iso_scaler = scaler
    if spec.use_pca:
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

    tr = tr.loc[:, tr.notna().any()]
    tr = tr.loc[:, tr.isna().mean() < 1]
    if spec.feature_allow:
        keep = [c for c in tr.columns if c in spec.feature_allow or c.startswith("miss_") or c.startswith("ix_")]
        keep += [c for c in ("iso_anomaly_score",) if c in tr.columns]
        tr = tr[[c for c in dict.fromkeys(keep) if c in tr.columns]]
    va = va.reindex(columns=tr.columns)
    return tr, va, y_tr, y_va, ord_maps, te_maps, g, iso_m, iso_scaler


def fit_bundle(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame | None,
    y_va: pd.Series | None,
    spec: FeatureSpec,
    n_estimators: int = 400,
    smote_ratio: float = 0.0,
    hp: dict | None = None,
    strict_pchip: bool = True,
) -> tuple[FittedBundle, np.ndarray | None]:
    hp = hp or {}
    n_estimators = int(hp.get("n_estimators", n_estimators))
    X_fit, y_fit = oversample(X_tr, y_tr, smote_ratio)
    xgb_m, lgb_m, imp = _fit_trees(X_fit, y_fit, n_estimators=n_estimators, early_stop=False)
    cols = list(X_tr.columns)
    Xtr_i = pd.DataFrame(imp.transform(X_tr), columns=cols, index=X_tr.index)
    raw_tr = blend_proba(xgb_m, lgb_m, Xtr_i)
    try:
        px, py = fit_pchip(raw_tr, y_tr.to_numpy())
    except PchipAbort:
        if strict_pchip:
            raise
        px, py = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cal_tr = apply_pchip(raw_tr, px, py)
    thr = cost_threshold(y_tr.to_numpy(), cal_tr)
    va_raw = None
    metrics = {}
    if X_va is not None and y_va is not None:
        Xva_i = pd.DataFrame(imp.transform(X_va), columns=cols, index=X_va.index)
        va_raw = blend_proba(xgb_m, lgb_m, Xva_i)
        va_cal = apply_pchip(va_raw, px, py)
        pred = (va_cal >= thr).astype(int)
        metrics = {
            "pr_auc": float(average_precision_score(y_va, va_raw)),
            "pr_auc_cal": float(average_precision_score(y_va, va_cal)),
            "roc_auc": float(roc_auc_score(y_va, va_raw)),
            "macro_f1": float(f1_score(y_va, pred, average="macro")),
            "minority_f1": float(f1_score(y_va, pred, pos_label=1, zero_division=0)),
            "n_features": int(X_tr.shape[1]),
            "threshold": thr,
        }
    bundle = FittedBundle(
        xgb=xgb_m,
        lgb=lgb_m,
        imputer=imp,
        iso=None,
        iso_scaler=None,
        columns=cols,
        cat_maps={},
        te_maps={},
        te_global=float(y_tr.mean()),
        pchip_x=px,
        pchip_y=py,
        threshold=thr,
        metrics=metrics,
        spec=spec,
    )
    return bundle, va_raw


def shap_top(bundle: FittedBundle, X_row: pd.DataFrame, k: int = 6) -> list[dict]:
    dmat = xgb.DMatrix(X_row[bundle.columns])
    contrib = bundle.xgb.get_booster().predict(dmat, pred_contribs=True)[0]
    raw = float(blend_proba(bundle.xgb, bundle.lgb, X_row[bundle.columns])[0])
    deriv = float(pchip_deriv(np.array([raw]), bundle.pchip_x, bundle.pchip_y)[0])
    pairs = []
    for name, val in zip(bundle.columns, contrib[:-1]):
        pairs.append({"feature": name, "label": name, "value": float(val * deriv)})
    pairs.sort(key=lambda d: abs(d["value"]), reverse=True)
    return pairs[:k]
