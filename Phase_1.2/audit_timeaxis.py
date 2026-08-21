"""What is a row, and what is the time axis?

The problem statement says the solution ingests financial transactions, so the
validation protocol has to be forward-in-time, never shuffled. Before trusting
any split we have to establish what one row actually is in this file and which
column orders rows in observation time.

Checks:
  1. Is a row a transaction or an account snapshot?
  2. Is F3888 (ACCT_OPN_DATE) an event time or a cohort label?
  3. Can the alert time be reconstructed from F3888 + F3887 (TENURE_AS_OF_ALERT)?
  4. Does the mule rate drift along each candidate axis (is order informative)?
  5. How much does shuffled validation over-report against ordered validation?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from data_layer import load_base

pd.set_option("display.width", 220)


def main() -> None:
    base = load_base()
    y = base.y
    n = len(y)

    print("=" * 78)
    print("1. ROW GRAIN")
    print("=" * 78)
    print(f"rows {n}   mules {int(y.sum())}   rate {y.mean():.4%}")
    opn = base.open_dates
    print(f"unique ACCT_OPN_DATE values: {opn.nunique()}  (rows per date {n / max(opn.nunique(), 1):.2f})")
    dup = base.numeric.duplicated().sum()
    print(f"exactly duplicated numeric rows: {dup}")
    print(
        "columns are trailing-window rollups (L7D / L14D / L31D / L7_14D ...), so one row is\n"
        "an account observed at its alert, not a single transaction."
    )
    print(f"window-suffixed columns in the dictionary: {(base.grammar['window'] != '').sum()} of {len(base.grammar)}")

    print()
    print("=" * 78)
    print("2. F3888 AS A TIME AXIS")
    print("=" * 78)
    print(f"range {opn.min()} -> {opn.max()}   nulls {int(opn.isna().sum())}")
    ordered = y.to_numpy()[opn.argsort(kind='mergesort').to_numpy()]
    dec = pd.DataFrame(
        {
            "decile": range(10),
            "n": [len(ordered[int(n * i / 10): int(n * (i + 1) / 10)]) for i in range(10)],
            "mules": [int(ordered[int(n * i / 10): int(n * (i + 1) / 10)].sum()) for i in range(10)],
        }
    )
    dec["rate"] = dec["mules"] / dec["n"]
    dec["start"] = [
        opn.iloc[opn.argsort(kind="mergesort").to_numpy()[int(n * i / 10)]] for i in range(10)
    ]
    print(dec.to_string(index=False))
    print("\nnon-stationary mule rate along this axis means the order carries information:")
    print(f"  first-half rate {ordered[: n // 2].mean():.4%}   second-half rate {ordered[n // 2:].mean():.4%}")

    print()
    print("=" * 78)
    print("3. CAN THE ALERT TIME BE RECONSTRUCTED?  F3888 + F3887")
    print("=" * 78)
    ten = pd.to_numeric(base.work.get("F3887"), errors="coerce")
    print(f"TENURE_AS_OF_ALERT  min {ten.min():.2f}  median {ten.median():.2f}  max {ten.max():.2f}")
    for unit, days in (("days", 1.0), ("months", 30.44), ("years", 365.25)):
        delta = pd.to_timedelta(np.clip(ten * days, 0, 60000), unit="D")
        alert = opn + delta
        share = float(((alert >= pd.Timestamp("2025-01-01")) & (alert <= pd.Timestamp("2026-12-31"))).mean())
        print(
            f"  if tenure is in {unit:6s}: alert range {alert.min()} -> {alert.max()}"
            f"   share landing in 2025-26: {share:.3f}"
        )

    alert = opn + pd.to_timedelta(np.clip(ten * 30.44, 0, 60000), unit="D")
    print("\nreading tenure as months, the reconstructed alert dates are:")
    print(alert.describe().to_string())
    span = (alert.max() - alert.min()).days
    print(f"total span of the alert window: {span} days")
    monthly = pd.DataFrame({"m": alert.dt.to_period("M").astype(str), "y": y})
    agg = monthly.groupby("m")["y"].agg(["size", "sum", "mean"]).rename(
        columns={"size": "n", "sum": "mules", "mean": "rate"}
    )
    print("\nmule rate by reconstructed alert month:")
    print(agg.to_string())
    print(
        "\nEvery account is observed inside one extract window, so there is no per-row future\n"
        "to leak and no event-time axis long enough to split on. The only ordering with real\n"
        "span is the account-opening cohort, which is what the step5 protocol uses."
    )

    lag = base.work.get("F3889_comp_lag")
    if lag is not None:
        print("\nACCT_OPN_DAYS buckets vs target (mule rate per bucket):")
        tab = pd.DataFrame({"bucket": lag, "y": y}).groupby("bucket")["y"].agg(
            ["size", "sum", "mean"]
        ).rename(columns={"size": "n", "sum": "mules", "mean": "rate"})
        print(tab.to_string())

    print()
    print("=" * 78)
    print("4. WHY SHUFFLED VALIDATION OVER-REPORTS")
    print("=" * 78)
    order = opn.argsort(kind="mergesort").to_numpy()
    tr_c, va_c = order[: int(0.8 * n)], order[int(0.8 * n):]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr_s, va_s = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    print(f"chronological split: train mules {int(y.iloc[tr_c].sum()):3d}  val mules {int(y.iloc[va_c].sum()):3d}"
          f"  val rate {y.iloc[va_c].mean():.4%}")
    print(f"shuffled split     : train mules {int(y.iloc[tr_s].sum()):3d}  val mules {int(y.iloc[va_s].sum()):3d}"
          f"  val rate {y.iloc[va_s].mean():.4%}")
    print(
        "\nA shuffled fold puts 2016 and 2025 accounts on both sides, so the model interpolates\n"
        "inside a cohort it has already seen. That is the 0.8677 number in the report. Every\n"
        "Phase_1.2 figure stays on the ordered protocol."
    )


if __name__ == "__main__":
    main()
