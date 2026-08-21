"""Re-rank a finished stage from its stored per-fold metrics.

Every scorecard field is written to results/<stage>.json, so changing which
operating point counts as production is a pure post-processing step — there is
no need to retrain to compare policies.

    python Phase_1.2/rescore.py features
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from experiment import LOCKED_CHRONO_PR, PS2_CHRONO_PR
from scorecard import composite

RESULTS = Path(__file__).resolve().parent / "results"


def main(stage: str) -> None:
    path = RESULTS / f"{stage}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for r in rows:
        # normcost was stored under whichever policy was production at run time;
        # recompute it per fold from the raw costs so policies stay comparable.
        for tag in ("r2", "r5", "r10", "r20", "r50", "r100"):
            key = f"normcost_{tag}"
            if f"cost_{tag}_opt" not in r["rows"][0]:
                continue
            opt = sum(f[f"cost_{tag}_opt"] for f in r["rows"])
            triv = sum(f[f"cost_{tag}_trivial"] for f in r["rows"])
            f1c = sum(f[f"cost_{tag}_f1"] for f in r["rows"])
            r["mean"][key] = float(opt / max(triv, 1e-9))
            r["mean"][f"savings_{tag}_vs_trivial"] = 1.0 - r["mean"][key]
            r["mean"][f"savings_{tag}_vs_f1"] = 1.0 - float(opt / max(f1c, 1e-9))
        r["composite"], r["composite_parts"] = composite(r["mean"])
    rows.sort(key=lambda r: -r["composite"])

    table = pd.DataFrame(
        [
            {
                "config": r["label"],
                "composite": r["composite"],
                "PR-AUC": r["mean"]["pr_auc"],
                "d_locked": r["mean"]["pr_auc"] - LOCKED_CHRONO_PR,
                "ROC": r["mean"]["roc_auc"],
                "muleF1": r["mean"]["f1_mule_f1"],
                "oracleF1": r["mean"]["oracle_mule_f1"],
                "macroF1": r["mean"]["f1_macro_f1"],
                "bAcc": r["mean"]["f1_balanced_accuracy"],
                "ECE": r["mean"]["ece"],
                "R@1%": r["mean"]["recall_at_1pct"],
                "cost@5": r["mean"]["cost_r5_opt"],
                "normcost": r["mean"]["normcost_r5"],
                "n_feat": r["rows"][0]["n_features"],
            }
            for r in rows
        ]
    )
    pd.set_option("display.width", 230)
    print(f"recorded PS2 {PS2_CHRONO_PR:.4f} | recorded locked {LOCKED_CHRONO_PR:.4f}\n")
    print(table.round(4).to_string(index=False))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nre-ranked {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "features")
