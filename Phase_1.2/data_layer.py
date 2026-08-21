"""Cached load of DataSet.csv + Description.xlsx for Phase_1.2 experiments.

The raw CSV is 9082 x 3925 and takes ~40 s to parse. Everything downstream is
a bake-off loop, so the parsed frame, the dictionary grammar and the row
moments are built once and cached under Phase_1.2/cache.

Preprocessing here is byte-identical to the locked Phase_1_stable recipe
(placeholder scrub, sentinel scrub, F3889 -> comp_lag, drop F3912/F2230) so a
Phase_1.2 number can be compared with 0.7381 without an asterisk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
CACHE_VERSION = "v3"

TARGET, ID_COL = "F3924", "Unnamed: 0"
CONFIRMED_LEAKY = ["F3912", "F2230"]
RESOLUTION_LEAKY = ["F3913", "F3914", "F3915"]
POST_ALERT = [f"F{i}" for i in range(3895, 3924)]
BANK_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]
PLACEHOLDERS = [-99999999, 99999999, -9999999, 9999999, -999999, 999999, -9999, 9999]
F3889_LAG_MAP = {"L7D": 7, "L14D": 14, "L31D": 31, "L90D": 90, "L180D": 180, "L365D": 365, "G365D": 400}
EPS = 1e-6

WINDOWS_POINT = ("L7D", "L14D", "L31D")
WINDOWS_RATIO = ("L7_14D", "L7_31D", "L14_31D")
STAT_PREFIXES = ("D_TA", "DA", "RA", "MAX", "MIN", "AVG", "MM", "TOT", "R", "D")
DIRECTIONS = ("CR", "DB")
METRICS = ("TXNS", "TXN", "AMT")


@dataclass
class Base:
    """Everything an experiment needs, already scrubbed and aligned."""

    numeric: pd.DataFrame          # all numeric columns, sentinel-scrubbed
    work: pd.DataFrame             # original columns (objects kept) + engineered
    y: pd.Series
    open_dates: pd.Series
    categorical_cols: list[str]
    temporal_cols: list[str]
    channel_cols: list[str]        # locked-baseline V_cross / txn_accel track
    row_stats: pd.DataFrame
    grammar: pd.DataFrame          # parsed Description.xlsx, indexed by F-code
    name_of: dict[str, str]        # F-code -> variable name
    col_of: dict[str, str]         # variable name -> F-code


def parse_grammar(desc: pd.DataFrame) -> pd.DataFrame:
    """Decompose every variable name into stat / entity / metric / dir / window.

    Names follow `[STAT_]ENTITY_METRIC[_DIR][_WINDOW]`, e.g.
    `RA_CI_NON_CASH_CHQ_AMT_DB_L7_31D` -> stat RA, entity CI_NON_CASH_CHQ,
    metric AMT, dir DB, window L7_31D. Rows that do not fit (AGE_IN_YRS,
    CUST_OCCP, ...) come back with empty parts and are simply not grouped.
    """
    rows = []
    for code, name in zip(desc["Feature"], desc["Variable Name"]):
        s = str(name).strip()
        rest = s
        window = ""
        occ = rest.endswith("_OCC")
        if occ:
            rest = rest[: -len("_OCC")]
        m = re.search(r"_(L?\d+_\d+D|L?\d+D|G\d+D)$", rest)
        if m:
            window = m.group(1)
            if not window.startswith(("L", "G")):
                window = "L" + window
            rest = rest[: m.start()]
        stat = ""
        for p in STAT_PREFIXES:
            if rest.startswith(p + "_"):
                stat = p
                rest = rest[len(p) + 1 :]
                break
        direction = ""
        for d in DIRECTIONS:
            if rest.endswith("_" + d):
                direction = d
                rest = rest[: -(len(d) + 1)]
                break
        metric = ""
        for mt in METRICS:
            if rest.endswith("_" + mt):
                metric = "TXN" if mt in ("TXN", "TXNS") else mt
                rest = rest[: -(len(mt) + 1)]
                break
        rows.append(
            {
                "col": str(code).strip(),
                "name": s,
                "stat": stat,
                "entity": rest,
                "metric": metric,
                "dir": direction or "TOT",
                "window": window,
                "occ": occ,
            }
        )
    return pd.DataFrame(rows).set_index("col")


def _scrub_sentinels(numeric: pd.DataFrame) -> pd.DataFrame:
    """Locked-baseline sentinel rule: a repeated |x| >= 1e7 value is a code."""
    for col in numeric.columns:
        s = numeric[col]
        ext = s[s.abs() >= 1e7].dropna()
        if ext.empty:
            continue
        counts = ext.value_counts()
        if counts.iloc[0] / max(int(s.notna().sum()), 1) >= 0.002:
            numeric[col] = s.replace(counts.index[0], np.nan)
    return numeric


def _row_stats(numeric: pd.DataFrame) -> pd.DataFrame:
    values = numeric.to_numpy(dtype=float)
    mask = ~np.isnan(values)
    nm = mask.sum(axis=1)
    tot = max(values.shape[1], 1)
    with np.errstate(all="ignore"):
        q25 = np.nanpercentile(values, 25, axis=1)
        q75 = np.nanpercentile(values, 75, axis=1)
        return pd.DataFrame(
            {
                "row_non_missing_count": nm,
                "row_missing_rate": 1.0 - nm / tot,
                "row_zero_rate": np.where(nm > 0, np.nansum(values == 0, axis=1) / nm, 0),
                "row_positive_rate": np.where(nm > 0, np.nansum(values > 0, axis=1) / nm, 0),
                "row_negative_rate": np.where(nm > 0, np.nansum(values < 0, axis=1) / nm, 0),
                "row_mean": np.nanmean(values, axis=1),
                "row_std": np.nanstd(values, axis=1),
                "row_min": np.nanmin(values, axis=1),
                "row_max": np.nanmax(values, axis=1),
                "row_median": np.nanmedian(values, axis=1),
                "row_q25": q25,
                "row_q75": q75,
                "row_iqr": q75 - q25,
                "row_abs_mean": np.nanmean(np.abs(values), axis=1),
            },
            index=numeric.index,
        )


def _build() -> Base:
    raw = pd.read_csv(ROOT / "DataSet.csv", low_memory=False)
    y = raw[TARGET].astype(int)
    work = raw.drop(columns=[ID_COL, TARGET], errors="ignore")
    work = work.replace([np.inf, -np.inf], np.nan).replace(PLACEHOLDERS, np.nan)

    desc = pd.read_excel(ROOT / "Description.xlsx", sheet_name="Data_Dicitionary")
    desc["Feature"] = desc["Feature"].astype(str).str.strip()
    desc["Variable Name"] = desc["Variable Name"].fillna("").astype(str)
    grammar = parse_grammar(desc)
    col_of = dict(zip(desc["Variable Name"], desc["Feature"]))
    name_of = dict(zip(desc["Feature"], desc["Variable Name"]))

    # Only the two confirmed label proxies go here. The post-alert block
    # F3895-F3923 stays in the frame because the locked baseline kept it and
    # Phase_1.2 needs to reproduce that number before it can beat it; the
    # experiment layer decides whether to expose it.
    work = work.drop(columns=[c for c in CONFIRMED_LEAKY if c in work.columns], errors="ignore")

    work["F3889_comp_lag"] = work["F3889"].astype("string").map(F3889_LAG_MAP).astype(float)
    work = work.drop(columns=["F3889"], errors="ignore")
    age = work["F3889_comp_lag"].fillna(400.0) + 1.0

    def get(var: str) -> pd.Series:
        code = col_of.get(var)
        if code is None or code not in work.columns:
            return pd.Series(np.nan, index=work.index, dtype=float)
        return pd.to_numeric(work[code], errors="coerce")

    channel_cols: list[str] = []
    for win in WINDOWS_POINT:
        atm = get(f"ATM_AMT_DB_{win}")
        elec = get(f"ELEC_XFER_AMT_DB_{win}")
        upi = get(f"UPI_AMT_CR_{win}")
        work[f"V_cross_{win}"] = np.log1p(
            np.clip((atm.fillna(0) + elec.fillna(0)) / (upi.fillna(0) + EPS), 0, 1e6)
        )
        work[f"log_atm_db_{win}"] = np.log1p(atm.clip(lower=0))
        work[f"log_elec_db_{win}"] = np.log1p(elec.clip(lower=0))
        work[f"log_upi_cr_{win}"] = np.log1p(upi.clip(lower=0))
        txns = get(f"ATM_TXNS_{win}").fillna(0) + get(f"UPI_XFER_TXNS_{win}").fillna(0) + get(f"ELEC_XFER_TXNS_{win}").fillna(0)
        work[f"txn_accel_{win}"] = np.log1p(np.clip(txns / age, 0, 1e4))
        channel_cols += [
            f"V_cross_{win}", f"log_atm_db_{win}", f"log_elec_db_{win}",
            f"log_upi_cr_{win}", f"txn_accel_{win}",
        ]
    work["V_cross"] = work[[f"V_cross_{w}" for w in WINDOWS_POINT]].mean(axis=1)
    work["txn_accel"] = work[[f"txn_accel_{w}" for w in WINDOWS_POINT]].mean(axis=1)
    channel_cols += ["V_cross", "txn_accel", "F3889_comp_lag"]

    object_cols = work.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    temporal_cols, categorical_cols = [], []
    for col in object_cols:
        parsed = pd.to_datetime(work[col], errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.7 and parsed.nunique(dropna=True) > 1:
            temporal_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric = work.apply(pd.to_numeric, errors="coerce")
    numeric = _scrub_sentinels(numeric)
    row_stats = _row_stats(numeric)
    open_dates = pd.to_datetime(work["F3888"], errors="coerce")

    return Base(
        numeric=numeric,
        work=work,
        y=y,
        open_dates=open_dates,
        categorical_cols=categorical_cols,
        temporal_cols=temporal_cols,
        channel_cols=channel_cols,
        row_stats=row_stats,
        grammar=grammar,
        name_of=name_of,
        col_of=col_of,
    )


def load_base(refresh: bool = False) -> Base:
    """Cache as a plain dict — a pickled dataclass would bind to __main__."""
    path = CACHE / f"base_{CACHE_VERSION}.joblib"
    if path.exists() and not refresh:
        return Base(**joblib.load(path))
    base = _build()
    joblib.dump(base.__dict__, path, compress=0)
    return base


if __name__ == "__main__":
    import time

    t0 = time.time()
    b = load_base(refresh=True)
    print(f"built in {time.time() - t0:.1f}s")
    print("numeric", b.numeric.shape, "mules", int(b.y.sum()), "rate", f"{b.y.mean():.3%}")
    print("categorical", b.categorical_cols)
    print("temporal", b.temporal_cols)
    print("open dates", b.open_dates.min(), "->", b.open_dates.max())
    g = b.grammar
    print("grammar parsed:", (g["entity"] != "").sum(), "of", len(g))
    print(g.groupby(["stat", "metric", "dir"]).size().sort_values(ascending=False).head(15))
