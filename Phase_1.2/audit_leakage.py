"""Leakage audit for the Phase_1.2 typology block.

A +0.09 PR-AUC jump has to be explained before it is quoted. Three questions:

  1. Is any single engineered feature a near-copy of the label (purity)?
  2. Does the block only work because mules are the rows that have data at all
     (a missingness artefact rather than behaviour)?
  3. Does the gain survive when the block is restricted to columns whose
     underlying raw inputs are populated for both classes?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from data_layer import load_base
from semantic import semantic_block

pd.set_option("display.width", 220)


def col_auc(frame: pd.DataFrame, y: np.ndarray) -> pd.Series:
    arr = frame.to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    filled = np.where(np.isnan(arr), med, arr)
    filled = np.where(np.isnan(filled), 0.0, filled)
    ranks = rankdata(filled, axis=0)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return pd.Series(auc, index=frame.columns)


def main() -> None:
    base = load_base()
    y = base.y.to_numpy()
    sem = semantic_block(base)

    print("=" * 78)
    print("1. PURITY — can any single engineered feature isolate the mules?")
    print("=" * 78)
    auc = col_auc(sem, y)
    rows = []
    for c in sem.columns:
        v = sem[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 100:
            continue
        thr = np.nanquantile(v, 0.99)
        top = ok & (v >= thr)
        rows.append(
            {
                "feature": c,
                "auc": float(auc[c]),
                "top1pct_fraud_rate": float(y[top].mean()) if top.sum() else np.nan,
                "top1pct_n": int(top.sum()),
                "missing": float(np.isnan(v).mean()),
            }
        )
    tab = pd.DataFrame(rows).sort_values("top1pct_fraud_rate", ascending=False)
    print(f"base rate {y.mean():.4%}")
    print(tab.head(12).to_string(index=False))
    pure = tab[(tab["top1pct_fraud_rate"] >= 0.5) & (tab["top1pct_n"] >= 20)]
    print(f"\nfeatures whose top 1% is >=50% mule: {len(pure)}  (a label copy would be ~1.0)")

    print()
    print("=" * 78)
    print("2. MISSINGNESS — is the block just detecting 'this row has data'?")
    print("=" * 78)
    for var in (
        "TOT_TXNAMT_CR_L7D", "TOT_TXNAMT_DB_L7D", "TOT_TXNS_CR_L31D",
        "AVG_BAL_7DAYS", "TENURE_AS_OF_ALERT",
    ):
        code = base.col_of.get(var)
        if code is None or code not in base.numeric.columns:
            continue
        s = base.numeric[code]
        present = s.notna().to_numpy()
        nonzero = (s.fillna(0).to_numpy() != 0)
        print(
            f"{var:22s} present legit {present[y == 0].mean():.3f} / mule {present[y == 1].mean():.3f}"
            f"   nonzero legit {nonzero[y == 0].mean():.3f} / mule {nonzero[y == 1].mean():.3f}"
        )

    print()
    print("=" * 78)
    print("3. WHERE THE SIGNAL SITS — top typology features by |AUC-0.5|")
    print("=" * 78)
    top = (auc - 0.5).abs().sort_values(ascending=False).head(20)
    out = pd.DataFrame({"auc": auc[top.index].round(4), "missing": sem[top.index].isna().mean().round(3)})
    print(out.to_string())

    print()
    print("Same statistic on the raw dictionary columns those features are built from,")
    print("to show the information was already in the file and only needed the right form:")
    raw_vars = ["TOT_TXNAMT_CR_L7D", "TOT_TXNAMT_DB_L7D", "TOT_TXNS_CR_L7D", "AVG_BAL_7DAYS"]
    codes = [base.col_of[v] for v in raw_vars if base.col_of.get(v) in base.numeric.columns]
    raw_auc = col_auc(base.numeric[codes], y)
    print(pd.DataFrame({"var": raw_vars[: len(codes)], "code": codes, "auc": raw_auc.values.round(4)}).to_string(index=False))


if __name__ == "__main__":
    main()
