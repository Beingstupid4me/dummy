"""Fold-safe feature tracks that need the training window to be built.

The report's factory has five tracks that are all row-local: moments,
missingness, categorical encoding, cyclical dates, isolation forest. Two things
an AML analyst uses every day are missing, and neither can be precomputed
because both need statistics estimated on the past only:

  peer-cohort normalisation   a 40k monthly turnover is unremarkable for a
                              trader and very loud for a pensioner, so
                              behaviour is scored against the account's own
                              occupation / segment / product cohort

  prototype geometry          with 65 known mules, where a row sits relative
                              to those 65 in behaviour space is information the
                              trees cannot recover on their own from axis
                              aligned splits

Both are fitted on `tr_idx` only. Prototype distances for training rows are
leave-one-out, otherwise every mule would be its own nearest mule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

EPS = 1e-9
COHORT_COLS = ("F3891", "F3893", "F3886", "F3890")  # occupation, segment, product, area
MIN_COHORT = 25


def peer_relative_block(
    work: pd.DataFrame,
    values: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    behaviour_cols: list[str],
    cohort_cols: tuple[str, ...] = COHORT_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust z-score of each behaviour against its cohort, cohort stats from train."""
    tr_out, va_out = {}, {}
    y_tr_rows = values.iloc[tr_idx]
    for ckey in cohort_cols:
        if ckey not in work.columns:
            continue
        keys_all = work[ckey].astype("string").fillna("MISSING")
        keys_tr = keys_all.iloc[tr_idx]
        counts = keys_tr.value_counts()
        valid = set(counts[counts >= MIN_COHORT].index)
        for col in behaviour_cols:
            if col not in values.columns:
                continue
            g = pd.DataFrame({"k": keys_tr.to_numpy(), "v": y_tr_rows[col].to_numpy()})
            stats = g.groupby("k")["v"].agg(
                med="median",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
            )
            stats = stats.loc[[k for k in stats.index if k in valid]]
            if stats.empty:
                continue
            gmed = float(np.nanmedian(y_tr_rows[col].to_numpy()))
            giqr = float(
                np.nanquantile(y_tr_rows[col].to_numpy(), 0.75)
                - np.nanquantile(y_tr_rows[col].to_numpy(), 0.25)
            )
            med_map, iqr_map = stats["med"].to_dict(), (stats["q75"] - stats["q25"]).to_dict()
            name = f"peer_{ckey}_{col}"
            for idx, store in ((tr_idx, tr_out), (va_idx, va_out)):
                k = keys_all.iloc[idx]
                med = k.map(med_map).astype(float).fillna(gmed).to_numpy()
                iqr = k.map(iqr_map).astype(float).fillna(giqr).to_numpy()
                x = values.iloc[idx][col].to_numpy(dtype=float)
                store[name] = np.clip((x - med) / (np.abs(iqr) + giqr * 0.1 + EPS), -20, 20)
    tr = pd.DataFrame(tr_out, index=values.index[tr_idx])
    va = pd.DataFrame(va_out, index=values.index[va_idx])
    return tr, va


def prototype_block(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    k_pos: int = 5,
    k_neg: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distance geometry against the known mules and the normal cloud."""
    imp = SimpleImputer(strategy="median")
    sc = RobustScaler()
    A = sc.fit_transform(imp.fit_transform(X_tr))
    B = sc.transform(imp.transform(X_va))
    A = np.clip(np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), -20, 20)
    B = np.clip(np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0), -20, 20)

    pos = np.flatnonzero(y_tr.to_numpy() == 1)
    neg = np.flatnonzero(y_tr.to_numpy() == 0)
    if len(pos) < k_pos + 2:
        empty_tr = pd.DataFrame(index=X_tr.index)
        return empty_tr, pd.DataFrame(index=X_va.index)

    centroid = A[pos].mean(axis=0)
    norm_c = np.linalg.norm(centroid) + EPS

    def to_centroid(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.linalg.norm(M - centroid, axis=1)
        cos = (M @ centroid) / ((np.linalg.norm(M, axis=1) + EPS) * norm_c)
        return d, cos

    # +1 neighbour on the train side so the row itself can be dropped
    nn_pos = NearestNeighbors(n_neighbors=min(k_pos + 1, len(pos))).fit(A[pos])
    nn_neg = NearestNeighbors(n_neighbors=min(k_neg + 1, len(neg))).fit(A[neg])

    def knn_stats(M: np.ndarray, drop_self: bool) -> tuple[np.ndarray, np.ndarray]:
        dp, _ = nn_pos.kneighbors(M)
        dn, _ = nn_neg.kneighbors(M)
        dp = dp[:, 1:] if drop_self else dp[:, : min(k_pos, dp.shape[1])]
        dn = dn[:, 1:] if drop_self else dn[:, : min(k_neg, dn.shape[1])]
        return dp.mean(axis=1), dn.mean(axis=1)

    out = {}
    for tag, M, drop in (("tr", A, True), ("va", B, False)):
        d, cos = to_centroid(M)
        dp, dn = knn_stats(M, drop)
        out[tag] = pd.DataFrame(
            {
                "proto_dist_mule_centroid": d,
                "proto_cos_mule_centroid": cos,
                "proto_knn_mule_dist": dp,
                "proto_knn_normal_dist": dn,
                "proto_knn_ratio": dp / (dn + EPS),
                "proto_knn_margin": dn - dp,
            },
            index=(X_tr.index if tag == "tr" else X_va.index),
        )
    return out["tr"], out["va"]
