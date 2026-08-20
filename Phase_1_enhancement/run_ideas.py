"""Honest chrono bake-off of new tracks. Does not edit Phase_1_stable.

Ideas (all leak-free):
  - L7/L14/L31 curvature on V_cross and UPI credit
  - train-only V_cross percentile
  - V_cross × txn_accel
  - no-PCA ablation
  - eval-only blends: 0.7/0.3, 0.5/0.5, rank-average, XGB-only, LGB-only

Headline stays raw-blend PR-AUC on step5 F3888 windows. Quote 0.738 locked / 0.710 PS2.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.pipeline import (  # noqa: E402
    FeatureSpec,
    blend_proba,
    future_core_split,
    rank_blend,
    step5_windows,
)
from app.ml.train import DatasetContext, train_on_indices  # noqa: E402
from app.ml.pipeline import load_frame  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "ideas.json"
PS2, STABLE = 0.7097, 0.7381
TREES = 280


def _imputed(bundle, X: pd.DataFrame) -> pd.DataFrame:
    cols = bundle.columns
    return pd.DataFrame(bundle.imputer.transform(X.reindex(columns=cols)), columns=cols, index=X.index)


def extra_scores(bundle, X_va: pd.DataFrame, y_va: pd.Series) -> dict[str, float]:
    Xi = _imputed(bundle, X_va)
    px = bundle.xgb.predict_proba(Xi)[:, 1]
    pl = bundle.lgb.predict_proba(Xi)[:, 1]
    recipes = {
        "blend_06_04": blend_proba(bundle.xgb, bundle.lgb, Xi, (0.6, 0.4)),
        "blend_07_03": blend_proba(bundle.xgb, bundle.lgb, Xi, (0.7, 0.3)),
        "blend_05_05": blend_proba(bundle.xgb, bundle.lgb, Xi, (0.5, 0.5)),
        "rank_06_04": rank_blend(bundle.xgb, bundle.lgb, Xi, (0.6, 0.4)),
        "xgb_only": px,
        "lgb_only": pl,
        "max_tree": np.maximum(px, pl),
    }
    return {k: float(average_precision_score(y_va, v)) for k, v in recipes.items()}


def eval_spec(ctx: DatasetContext, spec: FeatureSpec, label: str) -> dict:
    rows = []
    print(f"\n=== {label} trees={TREES} ===", flush=True)
    for fold, tr, va in step5_windows(ctx.open_dates):
        y_va = ctx.y.iloc[va]
        print(
            f"fold {fold}: train={len(tr)} mules={int(ctx.y.iloc[tr].sum())} | "
            f"val={len(va)} mules={int(y_va.sum())}",
            flush=True,
        )
        bundle, X_va, raw_va = train_on_indices(ctx, tr, va, spec, n_estimators=TREES)
        extras = extra_scores(bundle, X_va, y_va)
        row = {
            "fold": fold,
            "val_mules": int(y_va.sum()),
            "n_features": bundle.metrics.get("n_features"),
            "pr_auc_raw": float(average_precision_score(y_va, raw_va)),
            "roc_auc_raw": float(roc_auc_score(y_va, raw_va)),
            **{f"pr_{k}": v for k, v in extras.items()},
        }
        rows.append(row)
        print(
            f"  PR {row['pr_auc_raw']:.4f}  ROC {row['roc_auc_raw']:.4f}  "
            f"rank {extras['rank_06_04']:.4f}  0.7/0.3 {extras['blend_07_03']:.4f}  "
            f"n_feat {row['n_features']}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    mean = {c: float(df[c].mean()) for c in df.columns if c not in ("fold", "val_mules")}
    print(
        f"MEAN PR-AUC {mean['pr_auc_raw']:.4f} vs PS2 {PS2} vs stable {STABLE} "
        f"beat_stable={mean['pr_auc_raw'] >= STABLE}",
        flush=True,
    )
    return {"label": label, "rows": rows, "mean": mean}


def main() -> None:
    print("loading DataSet.csv", flush=True)
    work, y, open_dates, _lookup, dict_cols = load_frame(ROOT)
    ctx = DatasetContext(work, y, open_dates, dict_cols)
    print(
        "dict", len(dict_cols), "cats", ctx.categorical_cols, "temporal", ctx.temporal_cols,
        "F3891 in numeric", "F3891" in ctx.numeric.columns,
        flush=True,
    )
    results = [
        eval_spec(ctx, FeatureSpec(), "curv_pct_interact"),
        eval_spec(ctx, FeatureSpec(use_pca=False), "curv_pct_interact_no_pca"),
    ]
    winner = max(results, key=lambda r: r["mean"]["pr_auc_raw"])
    print("winner", winner["label"], winner["mean"]["pr_auc_raw"], flush=True)

    tr, va = future_core_split(open_dates)
    spec = FeatureSpec(use_pca=winner["label"] != "curv_pct_interact_no_pca")
    bundle, _, _ = train_on_indices(ctx, tr, va, spec, n_estimators=TREES)
    payload = {
        "protocol": "step5 chronological rolling windows (F3888)",
        "trees": TREES,
        "baseline_stable_pr_auc": STABLE,
        "ps2_chrono_pr_auc": PS2,
        "winner": winner["label"],
        "results": results,
        "future_core": bundle.metrics,
        "beats_ps2": winner["mean"]["pr_auc_raw"] >= PS2,
        "beats_stable": winner["mean"]["pr_auc_raw"] >= STABLE,
        "notes": [
            "Does not edit Phase_1_stable.",
            "Confirmed leaks F3912/F2230 stay out.",
            "Quote raw-blend PR-AUC. Rank-blend is an eval-only ablation.",
            "280 trees for bake-off speed; locked baseline used 500.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
