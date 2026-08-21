"""Phase_1.2 bake-off harness.

One code path, two dataclasses of switches. `FeatureConfig()` and
`ModelConfig()` with no arguments reproduce the locked Phase_1_stable recipe
(mean chrono PR-AUC 0.7381); every improvement is a flipped switch so a
scorecard row is always an apples-to-apples delta on the same folds.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb

from data_layer import (
    Base,
    CONFIRMED_LEAKY,
    EPS,
    POST_ALERT,
    RESOLUTION_LEAKY,
    WINDOWS_POINT,
    WINDOWS_RATIO,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LOCKED_CHRONO_PR = 0.7381
LOCKED_FOLDS = [0.7503, 0.7853, 0.7810, 0.7056, 0.6684]
PS2_CHRONO_PR = 0.7097
BANK_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
def step5_windows(open_dates: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The Phase_1/step5 rolling windows every previous phase reported on."""
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    folds = []
    for fold in range(5):
        split = int(0.8 * n) + (fold - 2) * int(0.04 * n)
        split = max(int(0.5 * n), min(int(0.95 * n), split))
        tr, va = order[:split], order[split : split + int(0.2 * n)]
        if len(va) >= 50:
            folds.append((fold, tr, va))
    return folds


def dense_windows(open_dates: pd.Series, n_folds: int = 9) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Wider walk-forward grid used only to check a win is not fold-specific."""
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    folds = []
    for fold, frac in enumerate(np.linspace(0.60, 0.88, n_folds)):
        split = int(frac * n)
        tr, va = order[:split], order[split : split + int(0.12 * n)]
        folds.append((fold, tr, va))
    return folds


def shuffled_folds(y: pd.Series, n_splits: int = 5, seed: int = 0) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Stratified k-fold. Only used to show how much order-blind CV over-reports."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(i, tr, va) for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y))]


def future_core_split(open_dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    order = open_dates.argsort(kind="mergesort").to_numpy()
    n = len(order)
    return order[: int(0.7 * n)], order[int(0.7 * n) : int(0.9 * n)]


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    # locked-baseline defaults
    selector: str = "variance_mi"      # variance_mi | auc_screen
    top_mi: int = 25
    top_gap: int = 25
    mi_cap: int = 400
    n_select: int = 60                 # auc_screen only: kept univariate columns
    corr_prune: float = 0.0            # auc_screen only: 0 disables redundancy prune
    interactions: int = 6              # top-MI columns crossed with row moments
    use_pca_kmeans: bool = True
    use_isoforest: bool = True
    use_row_stats: bool = True
    use_cyclical: bool = True
    include_post_alert: bool = True    # locked baseline left F3895-F3923 in the pool
    semantic: bool = False             # mule-typology block
    force_semantic: bool = False       # bypass the screen for the typology block
    drop_tenure: bool = False          # remove tenure-derived typology columns
    peer_relative: bool = False        # cohort-normalised behaviour
    prototype: bool = False            # distance geometry to known mules
    ratio_physics: bool = False        # dictionary group summaries
    extra_screen_n: int = 0            # extra columns drawn from the new blocks by AUC
    extra_prune: float = 0.95


@dataclass
class ModelConfig:
    families: tuple[str, ...] = ("xgb", "lgb")
    weights: tuple[float, ...] = (0.6, 0.4)
    n_seeds: int = 1
    combine: str = "prob"              # prob | rank
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    num_leaves: int = 31
    spw_power: float = 1.0             # scale_pos_weight ** power
    seed_offset: int = 0               # shifts every seed, for repeat runs
    depths: tuple[int, ...] = ()       # if set, each family is bagged over these depths
    calib_mode: str = "oof"            # oof (honest) | insample (what the report did)
    inner_folds: int = 2               # oof: expanding blocks at the end of the train window
    inner_block: float = 0.20          # oof: size of each block as a fraction of the window


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
def _auc_screen(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| per column, rank-based, vectorised over all 3.9k columns.

    The locked selector kept the 400 highest-variance columns before scoring
    them, which on this frame means raw rupee amounts crowd out every ratio
    and deviation family. Ranking is scale free, so nothing is pre-filtered.
    """
    med = np.nanmedian(X, axis=0)
    filled = np.where(np.isnan(X), med, X)
    filled = np.where(np.isnan(filled), 0.0, filled)
    ranks = rankdata(filled, axis=0)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 < 1 or n0 < 1:
        return np.zeros(X.shape[1])
    r1 = ranks[y == 1].sum(axis=0)
    auc = (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.abs(auc - 0.5)


def _corr_prune(frame: pd.DataFrame, ordered: list[str], limit: float, keep: int) -> list[str]:
    if limit <= 0:
        return ordered[:keep]
    sub = frame[ordered[: min(len(ordered), 4 * keep)]]
    mat = sub.corr(method="spearman").abs().fillna(0.0)
    picked: list[str] = []
    for col in sub.columns:
        if len(picked) >= keep:
            break
        if picked and mat.loc[col, picked].max() > limit:
            continue
        picked.append(col)
    return picked


def select_columns(
    numeric: pd.DataFrame, tr_idx: np.ndarray, y_tr: pd.Series, forced: list[str], cfg: FeatureConfig
) -> tuple[list[str], list[str], list[str]]:
    frame = numeric.iloc[tr_idx]
    nunique, miss = frame.nunique(dropna=True), frame.isna().mean()
    usable = [c for c in frame.columns if nunique.get(c, 0) > 1 and miss.get(c, 1) < 0.95]
    cand = frame[usable]

    if cfg.selector == "variance_mi":
        if cand.shape[1] > cfg.mi_cap:
            cand = cand[cand.var(skipna=True).sort_values(ascending=False).head(cfg.mi_cap).index]
        mi = mutual_info_classif(
            SimpleImputer(strategy="median").fit_transform(cand), y_tr, random_state=RANDOM_STATE
        )
        ranked = pd.Series(mi, index=cand.columns).sort_values(ascending=False)
        top_score = ranked.head(cfg.top_mi).index.tolist()
    else:
        score = _auc_screen(cand.to_numpy(dtype=float), y_tr.to_numpy())
        ranked = pd.Series(score, index=cand.columns).sort_values(ascending=False)
        top_score = _corr_prune(cand, ranked.index.tolist(), cfg.corr_prune, cfg.n_select)

    gap = (frame.loc[y_tr == 1].isna().mean() - frame.loc[y_tr == 0].isna().mean()).abs()
    top_gap = gap.sort_values(ascending=False).head(cfg.top_gap).index.tolist()

    selected: list[str] = []
    for c in forced + top_score + top_gap:
        if c not in selected and c in frame.columns:
            selected.append(c)
    return selected, top_score, top_gap


def encode_cats(tr_raw: pd.DataFrame, va_raw: pd.DataFrame, y_tr: pd.Series, cols: list[str]):
    parts_tr, parts_va = [], []
    g = float(y_tr.mean())
    for col in cols:
        tr = tr_raw[col].astype("string").fillna("MISSING")
        va = va_raw[col].astype("string").fillna("MISSING")
        if tr.nunique(dropna=False) <= 12:
            mapping = {v: i for i, v in enumerate(sorted(tr.unique().tolist()))}
            parts_tr.append(tr.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
            parts_va.append(va.map(mapping).fillna(-1).astype(float).rename(f"{col}_ord"))
        else:
            stats = pd.DataFrame({"c": tr, "y": y_tr.to_numpy()}).groupby("c")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + 20.0 * g) / (stats["count"] + 20.0)
            parts_tr.append(tr.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
            parts_va.append(va.map(smooth).fillna(g).astype(float).rename(f"{col}_te"))
    if parts_tr:
        return pd.concat(parts_tr, axis=1), pd.concat(parts_va, axis=1)
    return pd.DataFrame(index=tr_raw.index), pd.DataFrame(index=va_raw.index)


def cyclical(raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    parts = []
    for col in cols:
        p = pd.to_datetime(raw[col], errors="coerce", format="mixed")
        dow, month = p.dt.dayofweek.astype(float), p.dt.month.astype(float)
        parts.append(
            pd.DataFrame(
                {
                    f"{col}_dow_sin": np.sin(2 * np.pi * dow / 7.0),
                    f"{col}_dow_cos": np.cos(2 * np.pi * dow / 7.0),
                    f"{col}_month_sin": np.sin(2 * np.pi * (month - 1.0) / 12.0),
                    f"{col}_month_cos": np.cos(2 * np.pi * (month - 1.0) / 12.0),
                },
                index=raw.index,
            )
        )
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=raw.index)


def add_unsup(tr: pd.DataFrame, va: pd.DataFrame, y_tr: pd.Series, cfg: FeatureConfig):
    if not (cfg.use_isoforest or cfg.use_pca_kmeans):
        return tr, va
    imp, scaler = SimpleImputer(strategy="median"), RobustScaler()
    tr_s = scaler.fit_transform(imp.fit_transform(tr))
    va_s = scaler.transform(imp.transform(va))
    tr, va = tr.copy(), va.copy()
    if cfg.use_isoforest:
        iso = IsolationForest(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
        normal = (y_tr == 0).to_numpy()
        iso.fit(tr_s[normal] if normal.sum() >= 10 else tr_s)
        tr["iso_anomaly_score"] = -iso.score_samples(tr_s)
        va["iso_anomaly_score"] = -iso.score_samples(va_s)
    if cfg.use_pca_kmeans:
        pca = PCA(n_components=3, random_state=RANDOM_STATE)
        tr_p, va_p = pca.fit_transform(tr_s), pca.transform(va_s)
        km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
        tr_d, va_d = km.fit_transform(tr_s), km.transform(va_s)
        for i in range(3):
            tr[f"feature_pc_{i+1}"] = tr_p[:, i]
            va[f"feature_pc_{i+1}"] = va_p[:, i]
            tr[f"feature_kmeans_dist_c{i+1}"] = tr_d[:, i]
            va[f"feature_kmeans_dist_c{i+1}"] = va_d[:, i]
    return tr, va


_FOLD_CACHE: dict[tuple, tuple] = {}


def build_fold(base: Base, tr_idx: np.ndarray, va_idx: np.ndarray, cfg: FeatureConfig):
    """Feature matrices for one fold. Cached: a model sweep rebuilds them otherwise.

    The build is a pure function of (fold, FeatureConfig) — selection, encoding
    and the fold-safe blocks all read the training window only — so a stage that
    varies ModelConfig can reuse one matrix instead of recomputing a 400-column
    mutual-information screen seven times.
    """
    key = (
        id(base), int(tr_idx[0]), int(tr_idx[-1]), len(tr_idx),
        int(va_idx[0]), int(va_idx[-1]), len(va_idx), repr(cfg),
    )
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    out = _build_fold_uncached(base, tr_idx, va_idx, cfg)
    _FOLD_CACHE[key] = out
    return out


def _build_fold_uncached(base: Base, tr_idx: np.ndarray, va_idx: np.ndarray, cfg: FeatureConfig):
    from semantic import ratio_physics_block, semantic_block

    y_tr, y_va = base.y.iloc[tr_idx], base.y.iloc[va_idx]
    numeric = base.numeric
    if not cfg.include_post_alert:
        drop = [c for c in POST_ALERT + RESOLUTION_LEAKY if c in numeric.columns]
        numeric = numeric.drop(columns=drop, errors="ignore")

    forced = [c for c in BANK_FEATURES + base.channel_cols if c in numeric.columns and numeric[c].notna().any()]
    selected, top_score, top_gap = select_columns(numeric, tr_idx, y_tr, forced, cfg)

    # The new dictionary blocks are additive: the locked selection is left
    # intact and the extras arrive either whole (typology) or as the top-K by
    # train-window AUC (group summaries). That keeps every delta interpretable.
    extra_blocks: list[pd.DataFrame] = []
    sem_cols: list[str] = []
    if cfg.semantic:
        sem = semantic_block(base)
        if cfg.drop_tenure:
            sem = sem.drop(columns=[c for c in sem.columns if "tenure" in c or "young" in c])
        sem_cols = list(sem.columns)
        extra_blocks.append(sem)
    if cfg.ratio_physics:
        extra_blocks.append(ratio_physics_block(base))
    if extra_blocks:
        extra = pd.concat(extra_blocks, axis=1)
        extra = extra.loc[:, ~extra.columns.duplicated()]
        take: list[str] = []
        if cfg.semantic and cfg.force_semantic:
            take += [c for c in sem_cols if extra[c].notna().any()]
        pool = [c for c in extra.columns if c not in take and extra[c].notna().any()]
        if cfg.extra_screen_n and pool:
            sub = extra.iloc[tr_idx][pool]
            score = _auc_screen(sub.to_numpy(dtype=float), y_tr.to_numpy())
            ordered = pd.Series(score, index=pool).sort_values(ascending=False).index.tolist()
            take += _corr_prune(sub, ordered, cfg.extra_prune, cfg.extra_screen_n)
        numeric = pd.concat([numeric, extra[take]], axis=1)
        selected = selected + [c for c in take if c not in selected]

    blocks_tr = [numeric.iloc[tr_idx][selected]]
    blocks_va = [numeric.iloc[va_idx][selected]]
    if cfg.use_row_stats:
        blocks_tr.append(base.row_stats.iloc[tr_idx])
        blocks_va.append(base.row_stats.iloc[va_idx])
    if cfg.top_gap:
        blocks_tr.append(numeric.iloc[tr_idx][top_gap].isna().astype(int).add_prefix("miss_"))
        blocks_va.append(numeric.iloc[va_idx][top_gap].isna().astype(int).add_prefix("miss_"))
    cat_tr, cat_va = encode_cats(base.work.iloc[tr_idx], base.work.iloc[va_idx], y_tr, base.categorical_cols)
    blocks_tr.append(cat_tr)
    blocks_va.append(cat_va)
    if cfg.use_cyclical:
        blocks_tr.append(cyclical(base.work.iloc[tr_idx], base.temporal_cols))
        blocks_va.append(cyclical(base.work.iloc[va_idx], base.temporal_cols))

    tr = pd.concat(blocks_tr, axis=1)
    va = pd.concat(blocks_va, axis=1)
    tr = tr.loc[:, ~tr.columns.duplicated()]
    va = va.loc[:, ~va.columns.duplicated()]

    if cfg.interactions and cfg.use_row_stats:
        for col in top_score[: cfg.interactions]:
            if col in tr.columns:
                for stat in ("row_missing_rate", "row_zero_rate", "row_iqr"):
                    tr[f"ix_{col}_{stat}"] = tr[col] * tr[stat]
                    va[f"ix_{col}_{stat}"] = va[col] * va[stat]

    if cfg.peer_relative:
        from advanced import peer_relative_block

        behaviour = [c for c in selected if c.startswith("sem_")][:14] or selected[:14]
        p_tr, p_va = peer_relative_block(base.work, numeric, tr_idx, va_idx, behaviour)
        tr = pd.concat([tr, p_tr], axis=1)
        va = pd.concat([va, p_va], axis=1)

    tr, va = add_unsup(tr, va, y_tr, cfg)

    if cfg.prototype:
        from advanced import prototype_block

        q_tr, q_va = prototype_block(tr, y_tr, va)
        tr = pd.concat([tr, q_tr], axis=1)
        va = pd.concat([va, q_va], axis=1)

    tr = tr.loc[:, tr.isna().mean() < 1]
    va = va.reindex(columns=tr.columns)
    return tr, va, y_tr, y_va


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def _make_model(family: str, seed: int, spw: float, m: ModelConfig):
    if family == "xgb":
        return xgb.XGBClassifier(
            n_estimators=m.n_estimators, max_depth=m.max_depth, learning_rate=m.learning_rate,
            subsample=m.subsample, colsample_bytree=m.colsample, min_child_weight=m.min_child_weight,
            reg_lambda=m.reg_lambda, eval_metric="aucpr", scale_pos_weight=spw,
            tree_method="hist", random_state=seed, n_jobs=-1,
        )
    if family == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=m.n_estimators, learning_rate=m.learning_rate, num_leaves=m.num_leaves,
            max_depth=m.max_depth, subsample=m.subsample, subsample_freq=1,
            colsample_bytree=m.colsample, min_child_samples=max(int(m.min_child_weight), 5),
            reg_lambda=m.reg_lambda, objective="binary", scale_pos_weight=spw,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    if family == "xgb_dart":
        return xgb.XGBClassifier(
            n_estimators=max(m.n_estimators // 2, 120), max_depth=max(m.max_depth - 1, 2),
            learning_rate=m.learning_rate * 2, subsample=m.subsample, colsample_bytree=0.5,
            min_child_weight=m.min_child_weight, reg_lambda=m.reg_lambda * 4,
            eval_metric="aucpr", scale_pos_weight=spw, tree_method="hist",
            grow_policy="lossguide", max_leaves=15, random_state=seed, n_jobs=-1,
        )
    if family == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if family == "logit":
        return LogisticRegression(
            C=0.05, penalty="l2", class_weight="balanced", max_iter=2000, random_state=seed
        )
    raise ValueError(family)


def _rank01(v: np.ndarray) -> np.ndarray:
    return rankdata(v) / len(v)


def _fit_family_scores(
    Xtr: pd.DataFrame, y_tr: pd.Series, targets: list[pd.DataFrame], spw: float, m: ModelConfig
) -> list[dict[str, np.ndarray]]:
    """Fit each seed-bagged family once and score every requested matrix."""
    if any(f == "logit" for f in m.families):
        sc = RobustScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
        targets_s = [pd.DataFrame(sc.transform(t), columns=t.columns, index=t.index) for t in targets]
    out: list[dict[str, np.ndarray]] = [{} for _ in targets]
    # Depth is the strongest capacity knob at 65 positives, and the best depth
    # for ranking is not the best depth for the decision threshold. Bagging a
    # family across depths keeps both rather than choosing.
    depth_set = m.depths or (m.max_depth,)
    for family in m.families:
        acc = [[] for _ in targets]
        for depth in depth_set:
            cfg = replace(m, max_depth=depth, num_leaves=min(2 ** depth - 1, m.num_leaves))
            for k in range(m.n_seeds):
                model = _make_model(family, RANDOM_STATE + m.seed_offset + 101 * k, spw, cfg)
                if family == "logit":
                    model.fit(Xtr_s, y_tr)
                    for i, t in enumerate(targets_s):
                        acc[i].append(model.predict_proba(t)[:, 1])
                else:
                    model.fit(Xtr, y_tr)
                    for i, t in enumerate(targets):
                        acc[i].append(model.predict_proba(t)[:, 1])
                if family == "extratrees":
                    break  # already an ensemble; extra seeds buy nothing
        for i in range(len(targets)):
            out[i][family] = np.mean(acc[i], axis=0)
    return out


def _blend(per_family: dict[str, np.ndarray], m: ModelConfig) -> np.ndarray:
    weights = np.asarray(m.weights, dtype=float)[: len(m.families)]
    weights = weights / weights.sum()
    if m.combine == "rank":
        stack = np.vstack([_rank01(per_family[f]) for f in m.families])
    else:
        stack = np.vstack([per_family[f] for f in m.families])
    return (weights[:, None] * stack).sum(axis=0)


def fit_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, m: ModelConfig) -> dict:
    """Validation scores plus the training-side scores the calibrator is fitted on.

    `insample` reuses the fitted model's own training scores, which is what the
    report did — and those scores are saturated near 0 and 1, so any threshold
    read off them transfers badly. `oof` walks expanding blocks forward through
    the tail of the training window, so the probability map and the operating
    thresholds are built on scores no model in the ensemble has memorised.
    """
    spw = float((y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1)) ** m.spw_power
    imp = SimpleImputer(strategy="median")
    cols = list(X_tr.columns)
    Xtr = pd.DataFrame(imp.fit_transform(X_tr), columns=cols, index=X_tr.index)
    Xva = pd.DataFrame(imp.transform(X_va), columns=cols, index=X_va.index)

    (va_scores, tr_scores) = _fit_family_scores(Xtr, y_tr, [Xva, Xtr], spw, m)
    va_blend, tr_blend = _blend(va_scores, m), _blend(tr_scores, m)

    cal_y, cal_scores = y_tr.to_numpy(), tr_blend
    if m.calib_mode == "oof":
        n = len(Xtr)
        ys, ps = [], []
        for b in range(m.inner_folds, 0, -1):
            lo = int(n * (1.0 - b * m.inner_block))
            hi = int(n * (1.0 - (b - 1) * m.inner_block))
            if lo < 200 or hi - lo < 50:
                continue
            in_tr, in_va = Xtr.iloc[:lo], Xtr.iloc[lo:hi]
            in_y, in_yva = y_tr.iloc[:lo], y_tr.iloc[lo:hi]
            if int(in_y.sum()) < 10:
                continue
            in_spw = float((in_y == 0).sum() / max(int((in_y == 1).sum()), 1)) ** m.spw_power
            (scores,) = _fit_family_scores(in_tr, in_y, [in_va], in_spw, m)
            ys.append(in_yva.to_numpy())
            ps.append(_blend(scores, m))
        if ys and sum(int(a.sum()) for a in ys) >= 5:
            cal_y, cal_scores = np.concatenate(ys), np.concatenate(ps)

    return {
        "va_blend": va_blend,
        "va_families": va_scores,
        "cal_y": cal_y,
        "cal_scores": cal_scores,
    }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(
    base: Base,
    fcfg: FeatureConfig,
    mcfg: ModelConfig,
    label: str = "run",
    folds: list[tuple[int, np.ndarray, np.ndarray]] | None = None,
    verbose: bool = True,
) -> dict:
    from scorecard import composite, score_fold

    folds = folds if folds is not None else step5_windows(base.open_dates)
    rows = []
    for fold, tr, va in folds:
        X_tr, X_va, y_tr, y_va = build_fold(base, tr, va, fcfg)
        pred = fit_predict(X_tr, y_tr, X_va, mcfg)
        row = score_fold(
            pred["cal_y"], pred["cal_scores"], y_va.to_numpy(), pred["va_blend"],
            tail_frac=0.28 if mcfg.calib_mode == "insample" else 1.0,
        )
        row["fold"] = fold
        row["n_features"] = int(X_tr.shape[1])
        for f, p in pred["va_families"].items():
            row[f"pr_{f}"] = float(average_precision_score(y_va, p))
        rows.append(row)
        if verbose:
            print(
                f"  fold {fold} n_feat={row['n_features']:4d} PR={row['pr_auc']:.4f} "
                f"ROC={row['roc_auc']:.4f} muleF1={row['f1_mule_f1']:.3f} "
                f"R@1%={row['recall_at_1pct']:.3f} normcost={row['normcost_r5']:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    mean_row = {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind in "fi"}
    # Cost is a total, not an average of ratios: a fold holding 3 mules would
    # otherwise swing the headline as hard as one holding 16. Pool numerator
    # and denominator across folds instead.
    for tag in (f"r{int(r)}" for r in (2, 5, 10, 20, 50, 100)):
        if f"cost_{tag}_opt" not in df.columns:
            continue
        triv = float(df[f"cost_{tag}_trivial"].sum())
        opt = float(df[f"cost_{tag}_opt"].sum())
        f1c = float(df[f"cost_{tag}_f1"].sum())
        mean_row[f"normcost_{tag}"] = opt / max(triv, 1e-9)
        mean_row[f"savings_{tag}_vs_trivial"] = 1.0 - opt / max(triv, 1e-9)
        mean_row[f"savings_{tag}_vs_f1"] = 1.0 - opt / max(f1c, 1e-9)
    total, parts = composite(mean_row)
    mean_pr = mean_row["pr_auc"]
    out = {
        "label": label,
        "rows": rows,
        "mean": mean_row,
        "composite": total,
        "composite_parts": parts,
        "mean_pr_auc": mean_pr,
        "mean_roc_auc": mean_row["roc_auc"],
        "delta_vs_locked": mean_pr - LOCKED_CHRONO_PR,
        "beats_locked": mean_pr > LOCKED_CHRONO_PR,
    }
    if verbose:
        print(
            f"  MEAN  PR {mean_pr:.4f} ({out['delta_vs_locked']:+.4f} vs locked)  "
            f"ROC {mean_row['roc_auc']:.4f}  muleF1 {mean_row['f1_mule_f1']:.3f}  "
            f"macroF1 {mean_row['f1_macro_f1']:.3f}  bAcc {mean_row['f1_balanced_accuracy']:.3f}\n"
            f"        Brier {mean_row['brier']:.4f}  ECE {mean_row['ece']:.4f}  "
            f"slope {mean_row['calibration_slope']:.2f}  R@1% {mean_row['recall_at_1pct']:.3f}\n"
            f"        cost@R5 {mean_row['cost_r5_opt']:.0f} (F1-policy {mean_row['cost_r5_f1']:.0f}, "
            f"trivial {mean_row['cost_r5_trivial']:.0f})  normcost {mean_row['normcost_r5']:.3f}  "
            f"COMPOSITE {total:.4f}",
            flush=True,
        )
    return out
