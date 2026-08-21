"""Dictionary-driven features for Phase_1.2.

`Description.xlsx` names decompose into stat / entity / metric / direction /
window, so instead of hand-picking three channel ratios we can build the whole
mule typology from the grammar:

  pass-through   money in and straight back out within the window
  sweep          balance driven to near zero after a credit
  velocity       throughput large relative to balance and to account age
  dispersion     credits arrive on many channels, debits leave on one
  burst          L7 activity far above the L31 run rate

Everything is a pure row-wise transform of already-rolled-up columns, so it is
computed once for all 9082 accounts and stays fold-safe (no label, no fit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_layer import EPS, Base

POINT = ("L7D", "L14D", "L31D")
BAL_WIN = {"L7D": "7DAYS", "L14D": "14DAYS", "L31D": "31DAYS"}
DAYS = {"L7D": 7.0, "L14D": 14.0, "L31D": 31.0}
CORE_CHANNELS = (
    "ATM", "UPI", "ELEC_XFER", "NET_BNKING", "POS_PYMT", "CASH", "CHQ",
    "BBPS", "APB", "MBNKING", "STDNG_INSTR",
)
_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def _series(base: Base, var: str) -> pd.Series:
    code = base.col_of.get(var)
    if code is None or code not in base.numeric.columns:
        return pd.Series(np.nan, index=base.numeric.index, dtype=float)
    return base.numeric[code].astype(float)


def _pos(s: pd.Series) -> pd.Series:
    return s.fillna(0.0).clip(lower=0.0)


def _lr(num: pd.Series, den: pd.Series) -> pd.Series:
    """Bounded log ratio — a zero denominator must not become 1e16."""
    return np.log1p(np.clip(_pos(num) / (_pos(den) + EPS), 0, 1e6))


def _share_stats(mat: np.ndarray, prefix: str, index: pd.Index) -> pd.DataFrame:
    """Concentration of a non-negative vector across channels."""
    tot = mat.sum(axis=1, keepdims=True)
    share = np.divide(mat, tot + EPS)
    hhi = (share ** 2).sum(axis=1)
    ent = -(share * np.log(share + EPS)).sum(axis=1)
    order = np.argsort(-mat, axis=1)
    top = np.take_along_axis(share, order[:, :1], axis=1)[:, 0]
    second = np.take_along_axis(share, order[:, 1:2], axis=1)[:, 0] if mat.shape[1] > 1 else np.zeros(len(mat))
    return pd.DataFrame(
        {
            f"{prefix}_hhi": hhi,
            f"{prefix}_entropy": ent,
            f"{prefix}_top_share": top,
            f"{prefix}_top2_gap": top - second,
            f"{prefix}_active": (mat > 0).sum(axis=1).astype(float),
            f"{prefix}_argmax": order[:, 0].astype(float),
            f"{prefix}_log_total": np.log1p(tot[:, 0]),
        },
        index=index,
    )


def _channel_matrix(base: Base, metric: str, direction: str, win: str) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for ch in CORE_CHANNELS:
        var = f"{ch}_{metric}_{direction}_{win}" if direction else f"{ch}_{metric}_{win}"
        s = _series(base, var)
        if s.notna().any():
            cols.append(_pos(s).to_numpy())
            names.append(ch)
    if not cols:
        return np.zeros((len(base.numeric), 1)), []
    return np.vstack(cols).T, names


def semantic_block(base: Base) -> pd.DataFrame:
    """Mule typology: pass-through, sweep, velocity, dispersion, burst."""
    key = ("semantic", id(base))
    if key in _CACHE:
        return _CACHE[key]
    idx = base.numeric.index
    out: dict[str, pd.Series] = {}
    frames: list[pd.DataFrame] = []

    tenure = _series(base, "TENURE_AS_OF_ALERT").fillna(0).clip(lower=0)
    age_days = tenure + 1.0

    for win in POINT:
        bw = BAL_WIN[win]
        cr_amt = _series(base, f"TOT_TXNAMT_CR_{win}")
        db_amt = _series(base, f"TOT_TXNAMT_DB_{win}")
        cr_txn = _series(base, f"TOT_TXNS_CR_{win}")
        db_txn = _series(base, f"TOT_TXNS_DB_{win}")
        avg_bal = _series(base, f"AVG_BAL_{bw}")
        min_bal = _series(base, f"MIN_BAL_{bw}")
        max_bal = _series(base, f"MAX_BAL_{bw}")

        # pass-through: credits leave almost as fast as they arrive
        out[f"pt_ratio_{win}"] = _lr(db_amt, cr_amt)
        out[f"pt_net_{win}"] = (
            (_pos(cr_amt) - _pos(db_amt)) / (_pos(cr_amt) + _pos(db_amt) + EPS)
        ).clip(-1, 1)
        out[f"pt_residual_{win}"] = _lr(_pos(cr_amt) - _pos(db_amt).clip(upper=_pos(cr_amt)), avg_bal)

        # sweep: balance emptied relative to what moved through it
        out[f"sweep_min_over_max_{win}"] = (_pos(min_bal) / (_pos(max_bal) + EPS)).clip(0, 1)
        out[f"sweep_range_over_avg_{win}"] = (
            (_pos(max_bal) - _pos(min_bal)) / (_pos(avg_bal) + EPS)
        ).clip(0, 1e4)
        out[f"sweep_bal_over_cr_{win}"] = _lr(avg_bal, cr_amt)

        # velocity: throughput vs balance and vs account age
        out[f"vel_turnover_{win}"] = _lr(_pos(cr_amt) + _pos(db_amt), avg_bal)
        out[f"vel_cr_per_day_{win}"] = np.log1p(_pos(cr_amt) / DAYS[win])
        out[f"vel_txn_per_day_{win}"] = np.log1p((_pos(cr_txn) + _pos(db_txn)) / DAYS[win])
        out[f"vel_cr_per_tenure_{win}"] = np.log1p(_pos(cr_amt) / age_days)

        # ticket size and its asymmetry — layering uses many similar amounts
        out[f"ticket_cr_{win}"] = _lr(cr_amt, cr_txn)
        out[f"ticket_db_{win}"] = _lr(db_amt, db_txn)
        out[f"ticket_gap_{win}"] = out[f"ticket_db_{win}"] - out[f"ticket_cr_{win}"]
        out[f"txn_imbalance_{win}"] = (
            (_pos(db_txn) - _pos(cr_txn)) / (_pos(db_txn) + _pos(cr_txn) + EPS)
        ).clip(-1, 1)

        # dispersion: fan-in on credits, funnel-out on debits
        cr_mat, cr_names = _channel_matrix(base, "AMT", "CR", win)
        db_mat, db_names = _channel_matrix(base, "AMT", "DB", win)
        frames.append(_share_stats(cr_mat, f"chcr_{win}", idx))
        frames.append(_share_stats(db_mat, f"chdb_{win}", idx))
        if cr_names and db_names:
            out[f"ch_route_switch_{win}"] = (
                np.asarray([cr_names[i] for i in np.argmax(cr_mat, axis=1)])
                != np.asarray([db_names[i] for i in np.argmax(db_mat, axis=1)])
            ).astype(float)

        ct_mat, _ = _channel_matrix(base, "TXNS", "CR", win)
        dt_mat, _ = _channel_matrix(base, "TXNS", "DB", win)
        frames.append(_share_stats(ct_mat, f"chcrt_{win}", idx))
        frames.append(_share_stats(dt_mat, f"chdbt_{win}", idx))

        # customer-induced share: mule movement is user driven, not standing instruction
        ci = _series(base, f"CI_NON_CASH_CHQ_AMT_DB_{win}")
        ncc = _series(base, f"NON_CASH_CHQ_AMT_DB_{win}")
        out[f"ci_share_db_{win}"] = (_pos(ci) / (_pos(ncc) + EPS)).clip(0, 2)
        ci_cr = _series(base, f"CI_NON_CASH_CHQ_AMT_CR_{win}")
        ncc_cr = _series(base, f"NON_CASH_CHQ_AMT_CR_{win}")
        out[f"ci_share_cr_{win}"] = (_pos(ci_cr) / (_pos(ncc_cr) + EPS)).clip(0, 2)

        # cash-out leg: digital in, physical out
        cash_db = _series(base, f"CASH_AMT_DB_{win}")
        atm_db = _series(base, f"ATM_AMT_DB_{win}")
        digital_cr = (
            _pos(_series(base, f"UPI_AMT_CR_{win}"))
            + _pos(_series(base, f"ELEC_XFER_AMT_CR_{win}"))
            + _pos(_series(base, f"NET_BNKING_AMT_CR_{win}"))
        )
        out[f"cashout_ratio_{win}"] = _lr(_pos(cash_db) + _pos(atm_db), digital_cr)
        out[f"digital_in_share_{win}"] = (digital_cr / (_pos(cr_amt) + EPS)).clip(0, 2)

    # burst: L7 run rate against the L31 baseline
    for stem, var in (
        ("cr_amt", "TOT_TXNAMT_CR_{w}"),
        ("db_amt", "TOT_TXNAMT_DB_{w}"),
        ("cr_txn", "TOT_TXNS_CR_{w}"),
        ("db_txn", "TOT_TXNS_DB_{w}"),
    ):
        l7 = _pos(_series(base, var.format(w="L7D"))) / 7.0
        l14 = _pos(_series(base, var.format(w="L14D"))) / 14.0
        l31 = _pos(_series(base, var.format(w="L31D"))) / 31.0
        out[f"burst_{stem}_7_31"] = np.log1p(l7) - np.log1p(l31)
        out[f"burst_{stem}_7_14"] = np.log1p(l7) - np.log1p(l14)
        out[f"burst_{stem}_curv"] = (np.log1p(l7) - np.log1p(l14)) - (np.log1p(l14) - np.log1p(l31))

    bal7, bal31 = _series(base, "AVG_BAL_7DAYS"), _series(base, "AVG_BAL_31DAYS")
    out["bal_drift_7_31"] = np.log1p(_pos(bal7)) - np.log1p(_pos(bal31))
    out["tenure_log"] = np.log1p(tenure)
    out["young_high_throughput"] = np.log1p(
        _pos(_series(base, "TOT_TXNAMT_CR_L31D")) / (tenure + 30.0)
    )
    out["age_over_tenure"] = _series(base, "AGE_IN_YRS").astype(float) / (np.log1p(tenure) + 1.0)

    block = pd.concat([pd.DataFrame(out, index=idx)] + frames, axis=1)
    block = block.replace([np.inf, -np.inf], np.nan).astype("float32")
    block.columns = [f"sem_{c}" for c in block.columns]
    _CACHE[key] = block
    return block


def _nan_safe(fn, mat: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmax/nanmean on an all-NaN row warns and returns NaN; we want NaN quietly."""
    with np.errstate(all="ignore"):
        empty = np.isnan(mat).all(axis=axis)
        filled = np.where(np.isnan(mat) & empty[:, None], 0.0, mat)
        out = fn(filled, axis=axis)
        return np.where(empty, np.nan, out)


def ratio_physics_block(base: Base) -> pd.DataFrame:
    """Grammar group summaries: how extreme is this row inside each named family.

    The dictionary ships ~2500 ratio / deviation columns organised as
    (stat, metric, direction, window) x entity. A GBDT with 65 positives cannot
    look at all of them, but the shape of each family - its max, its spread,
    how many members fired - is a stable low-dimensional summary of the same
    information.
    """
    key = ("physics", id(base))
    if key in _CACHE:
        return _CACHE[key]
    g = base.grammar
    idx = base.numeric.index
    cols: dict[str, np.ndarray] = {}
    groups = g[(g["stat"] != "") & (g["entity"] != "") & (g["window"] != "")].groupby(
        ["stat", "metric", "dir", "window"]
    )
    for (stat, metric, direction, window), sub in groups:
        members = [c for c in sub.index if c in base.numeric.columns]
        if len(members) < 6:
            continue
        mat = base.numeric[members].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            finite = np.where(np.isfinite(mat), mat, np.nan)
            name = f"grp_{stat}_{metric or 'NA'}_{direction}_{window}"
            cols[f"{name}_max"] = _nan_safe(np.max, finite, 1)
            cols[f"{name}_mean"] = _nan_safe(np.mean, finite, 1)
            cols[f"{name}_std"] = _nan_safe(np.std, finite, 1)
            cols[f"{name}_nz"] = np.nansum(np.abs(finite) > 1e-9, axis=1).astype(float)
            cols[f"{name}_miss"] = np.isnan(finite).mean(axis=1)
    block = pd.DataFrame(cols, index=idx).replace([np.inf, -np.inf], np.nan).astype("float32")
    _CACHE[key] = block
    return block


if __name__ == "__main__":
    from data_layer import load_base

    b = load_base()
    s = semantic_block(b)
    r = ratio_physics_block(b)
    print("semantic", s.shape, "ratio_physics", r.shape)
    y = b.y
    from scipy.stats import rankdata

    for name, blk in (("semantic", s), ("physics", r)):
        arr = blk.to_numpy(dtype=float)
        med = np.nanmedian(arr, axis=0)
        arr = np.where(np.isnan(arr), med, arr)
        arr = np.where(np.isnan(arr), 0.0, arr)
        ranks = rankdata(arr, axis=0)
        n1 = float(y.sum())
        n0 = len(y) - n1
        auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        top = pd.Series(np.abs(auc - 0.5), index=blk.columns).sort_values(ascending=False).head(12)
        print(f"\ntop |AUC-0.5| in {name} (global, sanity only):")
        print((top + 0.5).round(3).to_string())
