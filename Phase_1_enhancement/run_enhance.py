from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.pipeline import (  # noqa: E402
    FeatureSpec,
    apply_pchip,
    future_core_split,
    load_frame,
    step5_windows,
)
from app.ml.train import DatasetContext, train_on_indices  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "metrics.json"
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)

PS2_CHRONO_PR = 0.7097
STABLE_CHRONO_PR = 0.7381


def eval_spec(ctx: DatasetContext, spec: FeatureSpec, label: str, n_estimators: int = 400) -> dict:
    rows = []
    print(f"\n=== {label} ===", flush=True)
    for fold, tr, va in step5_windows(ctx.open_dates):
        y_va = ctx.y.iloc[va]
        print(
            f"fold {fold}: train={len(tr)} mules={int(ctx.y.iloc[tr].sum())} | "
            f"val={len(va)} mules={int(y_va.sum())}",
            flush=True,
        )
        bundle, X_va, raw_va = train_on_indices(ctx, tr, va, spec, n_estimators=n_estimators)
        cal = apply_pchip(raw_va, bundle.pchip_x, bundle.pchip_y)
        pred = (cal >= bundle.threshold).astype(int)
        row = {
            "fold": fold,
            "val_mules": int(y_va.sum()),
            **bundle.metrics,
            "pr_auc_raw": float(average_precision_score(y_va, raw_va)),
            "roc_auc_raw": float(roc_auc_score(y_va, raw_va)),
            "macro_f1": float(f1_score(y_va, pred, average="macro")),
            "minority_f1": float(f1_score(y_va, pred, pos_label=1, zero_division=0)),
        }
        rows.append(row)
        print(
            f"  PR-AUC {row['pr_auc_raw']:.4f}  ROC {row['roc_auc_raw']:.4f}  "
            f"mule-F1 {row['minority_f1']:.3f}  n_feat {row['n_features']}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    mean = {
        "pr_auc": float(df["pr_auc_raw"].mean()),
        "roc_auc": float(df["roc_auc_raw"].mean()),
        "macro_f1": float(df["macro_f1"].mean()),
        "minority_f1": float(df["minority_f1"].mean()),
    }
    print(
        f"MEAN PR-AUC {mean['pr_auc']:.4f} vs PS2 {PS2_CHRONO_PR} vs stable {STABLE_CHRONO_PR} "
        f"beat_stable={mean['pr_auc'] >= STABLE_CHRONO_PR}",
        flush=True,
    )
    return {"label": label, "rows": rows, "mean": mean}


def main() -> None:
    print("loading DataSet.csv + Description.xlsx", flush=True)
    work, y, open_dates, _lookup, dict_cols = load_frame(ROOT)
    ctx = DatasetContext(work, y, open_dates, dict_cols)
    print("dict tracks", len(dict_cols), "cats", ctx.categorical_cols, "temporal", ctx.temporal_cols, flush=True)

    specs = [
        ("dict_channels_no_tms", FeatureSpec()),
    ]
    results = [eval_spec(ctx, spec, name, n_estimators=500) for name, spec in specs]
    winner = max(results, key=lambda r: r["mean"]["pr_auc"])
    print("winner", winner["label"], winner["mean"]["pr_auc"], flush=True)

    spec = FeatureSpec(include_tms=winner["label"].endswith("tms"))
    tr, va = future_core_split(open_dates)
    bundle, _, _ = train_on_indices(ctx, tr, va, spec, n_estimators=500)
    joblib.dump(
        {
            "bundle": bundle,
            "spec": spec,
            "dict_cols": dict_cols,
            "winner": winner["label"],
            "chrono_mean": winner["mean"],
        },
        MODEL_DIR / "enhance_m1.joblib",
        compress=3,
    )
    payload = {
        "protocol": "step5 chronological rolling windows (F3888)",
        "baseline_stable_pr_auc": STABLE_CHRONO_PR,
        "ps2_chrono_pr_auc": PS2_CHRONO_PR,
        "winner": winner["label"],
        "results": results,
        "future_core": bundle.metrics,
        "beats_ps2": winner["mean"]["pr_auc"] >= PS2_CHRONO_PR,
        "beats_stable": winner["mean"]["pr_auc"] >= STABLE_CHRONO_PR,
        "notes": [
            "Does not edit Phase_1_stable.",
            "Confirmed leaks F3912/F2230 and resolution flags F3913-F3915 stay out of the clean spec.",
            "Dictionary tracks: V_cross, V_pos, V_net, V_cash, channel entropy, L7-L31 burst, UPI imbalance.",
            "Quote raw-blend PR-AUC (same as locked baseline). PCHIP is the probability map.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
