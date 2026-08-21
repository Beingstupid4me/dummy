"""Phase_1.2 staged bake-off.

    python Phase_1.2/run_bakeoff.py anchor      reproduce locked 0.7381, measure leak assist
    python Phase_1.2/run_bakeoff.py features    feature-side ablations
    python Phase_1.2/run_bakeoff.py models      model-side ablations
    python Phase_1.2/run_bakeoff.py combo       stack the winners

Every stage writes results/<stage>.json and prints a delta against the locked
mean chrono PR-AUC of 0.7381 on the identical step5 windows.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data_layer import load_base
from experiment import (
    LOCKED_CHRONO_PR,
    LOCKED_FOLDS,
    PS2_CHRONO_PR,
    FeatureConfig,
    ModelConfig,
    evaluate,
)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def run_stage(name: str, cases: list[tuple[str, FeatureConfig, ModelConfig]]) -> list[dict]:
    base = load_base()
    out = []
    for label, fcfg, mcfg in cases:
        print(f"\n=== {label} ===", flush=True)
        t0 = time.time()
        res = evaluate(base, fcfg, mcfg, label=label)
        res["seconds"] = round(time.time() - t0, 1)
        res["feature_config"] = asdict(fcfg)
        res["model_config"] = asdict(mcfg)
        out.append(res)
        print(f"  ({res['seconds']}s)", flush=True)
        # checkpoint after every case; these stages run for tens of minutes
        (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    out.sort(key=lambda r: -r["composite"])
    print(f"\n---------- {name} scorecard (sorted by composite) ----------")
    print(
        f"{'config':28s} {'COMPOS':>7s} {'PR-AUC':>7s} {'d-lock':>7s} {'ROC':>6s} "
        f"{'muleF1':>7s} {'macroF1':>8s} {'bAcc':>6s} {'ECE':>6s} {'R@1%':>6s} "
        f"{'cost@5':>7s} {'normc':>6s} {'sec':>5s}"
    )
    print(f"{'locked stable (recorded)':28s} {'-':>7s} {LOCKED_CHRONO_PR:7.4f} {0.0:+7.4f}")
    print(f"{'PS2 report (recorded)':28s} {'-':>7s} {PS2_CHRONO_PR:7.4f} {PS2_CHRONO_PR - LOCKED_CHRONO_PR:+7.4f}")
    for r in out:
        m = r["mean"]
        print(
            f"{r['label']:28s} {r['composite']:7.4f} {m['pr_auc']:7.4f} {r['delta_vs_locked']:+7.4f} "
            f"{m['roc_auc']:6.4f} {m['f1_mule_f1']:7.3f} {m['f1_macro_f1']:8.3f} "
            f"{m['f1_balanced_accuracy']:6.3f} {m['ece']:6.4f} {m['recall_at_1pct']:6.3f} "
            f"{m['cost_r5_opt']:7.1f} {m['normcost_r5']:6.3f} {r['seconds']:5.0f}"
        )

    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return out


def stage_anchor() -> None:
    """Is 0.7381 reproducible, and how much of it is the post-alert block?"""
    cases = [
        ("locked_repro", FeatureConfig(), ModelConfig()),
        ("locked_minus_post_alert", FeatureConfig(include_post_alert=False), ModelConfig()),
    ]
    run_stage("anchor", cases)


def stage_features() -> None:
    """Model held at the locked 0.6 XGB + 0.4 LGB so deltas are feature deltas."""
    gate = {"include_post_alert": False}
    sem = dict(gate, semantic=True, force_semantic=True)
    m = ModelConfig()
    cases = [
        ("f0_control", FeatureConfig(**gate), m),
        ("f1_semantic", FeatureConfig(**sem), m),
        ("f2_semantic_no_tenure", FeatureConfig(**sem, drop_tenure=True), m),
        ("f3_semantic_nopca", FeatureConfig(**sem, use_pca_kmeans=False), m),
        ("f4_semantic_phys100", FeatureConfig(**sem, ratio_physics=True, extra_screen_n=100), m),
        ("f5_phys150_only", FeatureConfig(**gate, ratio_physics=True, extra_screen_n=150), m),
        ("f6_semantic_peer", FeatureConfig(**sem, peer_relative=True), m),
        ("f7_semantic_proto", FeatureConfig(**sem, prototype=True), m),
        (
            "f8_semantic_all",
            FeatureConfig(**sem, ratio_physics=True, extra_screen_n=100,
                          peer_relative=True, prototype=True, use_pca_kmeans=False),
            m,
        ),
    ]
    run_stage("features", cases)


def best_feature_config() -> FeatureConfig:
    """Winner of the feature stage, so later stages build on it instead of guessing."""
    path = RESULTS / "features.json"
    if not path.exists():
        return FeatureConfig(include_post_alert=False, semantic=True, force_semantic=True)
    rows = json.loads(path.read_text(encoding="utf-8"))
    best = max(rows, key=lambda r: r["composite"])
    print(f"[carrying forward feature winner: {best['label']}]")
    return FeatureConfig(**best["feature_config"])


def stage_models() -> None:
    """Ensemble cog: variance reduction, family diversity, capacity for 65 positives."""
    f = best_feature_config()
    cases = [
        ("m0_report_blend", f, ModelConfig()),
        ("m1_seedbag3", f, ModelConfig(n_seeds=3)),
        ("m2_shallow", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15)),
        (
            "m3_shallow_reg",
            f,
            ModelConfig(n_seeds=3, max_depth=3, num_leaves=15, min_child_weight=5,
                        reg_lambda=5.0, colsample=0.5),
        ),
        ("m4_spw_sqrt", f, ModelConfig(n_seeds=3, spw_power=0.5)),
        ("m5_rank_blend", f, ModelConfig(n_seeds=3, combine="rank")),
        (
            "m6_four_family",
            f,
            ModelConfig(families=("xgb", "lgb", "xgb_dart", "extratrees"),
                        weights=(0.35, 0.25, 0.20, 0.20), n_seeds=3, combine="rank"),
        ),
    ]
    run_stage("models", cases)


def stage_capacity() -> None:
    """Depth 5 -> 3 was the single biggest model-side gain; find the real optimum."""
    f = best_feature_config()
    cases = [
        ("cap_d5_report", f, ModelConfig(n_seeds=3, max_depth=5, num_leaves=31)),
        ("cap_d2_l7", f, ModelConfig(n_seeds=3, max_depth=2, num_leaves=7)),
        ("cap_d3_l15", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15)),
        ("cap_d4_l31", f, ModelConfig(n_seeds=3, max_depth=4, num_leaves=31)),
        ("cap_d3_l15_lr02", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15,
                                           learning_rate=0.02, n_estimators=1200)),
        ("cap_d3_l15_cs05", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15, colsample=0.5)),
        ("cap_d3_l15_mcw5", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15,
                                           min_child_weight=5, reg_lambda=5.0)),
        ("cap_d3_l15_seed7", f, ModelConfig(n_seeds=7, max_depth=3, num_leaves=15)),
    ]
    run_stage("capacity", cases)


def stage_combo() -> None:
    """Depth 3 ranks best, depth 2 decides best — can one model have both?"""
    f = best_feature_config()
    cases = [
        ("combo_d2", f, ModelConfig(n_seeds=3, max_depth=2, num_leaves=7)),
        ("combo_d3", f, ModelConfig(n_seeds=3, max_depth=3, num_leaves=15)),
        ("combo_d23", f, ModelConfig(n_seeds=3, depths=(2, 3))),
        ("combo_d234", f, ModelConfig(n_seeds=3, depths=(2, 3, 4))),
        ("combo_d23_rank", f, ModelConfig(n_seeds=3, depths=(2, 3), combine="rank")),
        ("combo_d23_seed5", f, ModelConfig(n_seeds=5, depths=(2, 3))),
        (
            "combo_d23_4fam",
            f,
            ModelConfig(families=("xgb", "lgb", "xgb_dart", "extratrees"),
                        weights=(0.35, 0.25, 0.20, 0.20), n_seeds=2, depths=(2, 3),
                        combine="rank"),
        ),
    ]
    run_stage("combo", cases)


def best_model_config() -> ModelConfig:
    """Best across every model-side stage, so earlier results are not lost."""
    rows: list[dict] = []
    for name in ("models", "capacity", "combo", "calibration"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            rows += json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return ModelConfig(n_seeds=3, max_depth=3, num_leaves=15)
    best = max(rows, key=lambda r: r["composite"])
    print(f"[carrying forward model winner: {best['label']}]")
    return ModelConfig(**best["model_config"])


def stage_calibration() -> None:
    """Calibration cog: what the report's in-sample isotonic actually costs."""
    f = best_feature_config()
    m = best_model_config()
    cases = [
        ("cal_report_insample", f, replace(m, calib_mode="insample")),
        ("cal_oof_1block", f, replace(m, calib_mode="oof", inner_folds=1)),
        ("cal_oof_2block", f, replace(m, calib_mode="oof", inner_folds=2)),
        ("cal_oof_3block", f, replace(m, calib_mode="oof", inner_folds=3)),
    ]
    run_stage("calibration", cases)


def stage_robustness() -> None:
    """Is the win real, or is it five folds and one lucky seed?"""
    from data_layer import load_base
    from experiment import dense_windows, shuffled_folds

    base = load_base()
    f_win, m_win = best_feature_config(), best_model_config()
    f_ctl, m_ctl = FeatureConfig(include_post_alert=False), ModelConfig()
    out = []

    print("\n### A. same recipe, three independent seed sets (step5 folds)")
    for off in (0, 1000, 2000):
        res = evaluate(base, f_win, replace(m_win, seed_offset=off), label=f"winner_seed{off}")
        out.append(res)

    print("\n### B. dense walk-forward, 9 cut points instead of 5")
    dense = dense_windows(base.open_dates, n_folds=9)
    out.append(evaluate(base, f_ctl, m_ctl, label="control_dense9", folds=dense))
    out.append(evaluate(base, f_win, m_win, label="winner_dense9", folds=dense))

    print("\n### C. order-blind stratified 5-fold — the protocol that produced 0.8677")
    shuf = shuffled_folds(base.y, n_splits=5, seed=0)
    out.append(evaluate(base, f_ctl, m_ctl, label="control_shuffled", folds=shuf))
    out.append(evaluate(base, f_win, m_win, label="winner_shuffled", folds=shuf))

    out.sort(key=lambda r: -r["composite"])
    print("\n---------- robustness ----------")
    for r in out:
        m = r["mean"]
        print(
            f"{r['label']:22s} PR {m['pr_auc']:.4f}  ROC {m['roc_auc']:.4f}  "
            f"muleF1 {m['f1_mule_f1']:.3f}  normcost {m['normcost_r5']:.3f}  "
            f"composite {r['composite']:.4f}"
        )
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'robustness.json'}")


def stage_final() -> None:
    """Lock the winning recipe, score it head-to-head, write the scorecard."""
    from data_layer import load_base
    from experiment import future_core_split

    base = load_base()
    f_win, m_win = best_feature_config(), best_model_config()
    f_rep, m_rep = FeatureConfig(), ModelConfig(calib_mode="insample")

    print("\n=== report recipe (as published) ===")
    rep = evaluate(base, f_rep, m_rep, label="report_recipe")
    print("\n=== Phase_1.2 ===")
    v2 = evaluate(base, f_win, m_win, label="phase_1_2")

    print("\n=== Phase_1.2 on the future-core holdout ===")
    tr, va = future_core_split(base.open_dates)
    core = evaluate(base, f_win, m_win, label="phase_1_2_future_core", folds=[(0, tr, va)])

    keys = [
        ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"),
        ("mule F1", "f1_mule_f1"), ("macro F1", "f1_macro_f1"),
        ("balanced acc", "f1_balanced_accuracy"), ("accuracy", "f1_accuracy"),
        ("recall@1%", "recall_at_1pct"), ("recall@5%", "recall_at_5pct"),
        ("Brier", "brier"), ("ECE", "ece"), ("calib slope", "calibration_slope"),
        ("cost @R=5", "cost_r5_opt"), ("normalised cost", "normcost_r5"),
        ("saving vs F1 policy", "savings_r5_vs_f1"),
    ]
    print("\n---------- final scorecard (step5 chronological folds) ----------")
    print(f"{'metric':22s} {'report':>10s} {'phase_1.2':>10s} {'delta':>10s}")
    lines = []
    for name, key in keys:
        a, b = rep["mean"][key], v2["mean"][key]
        print(f"{name:22s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
        lines.append({"metric": name, "report": a, "phase_1_2": b, "delta": b - a})
    print(f"{'composite':22s} {rep['composite']:10.4f} {v2['composite']:10.4f} "
          f"{v2['composite'] - rep['composite']:+10.4f}")

    payload = {
        "protocol": "step5 chronological rolling windows on F3888 (identical to PS2 and Phase_1_stable)",
        "recorded_ps2_chrono_pr_auc": PS2_CHRONO_PR,
        "recorded_locked_chrono_pr_auc": LOCKED_CHRONO_PR,
        "feature_config": asdict(f_win),
        "model_config": asdict(m_win),
        "report_recipe": rep,
        "phase_1_2": v2,
        "future_core": core,
        "scorecard": lines,
        "beats_recorded_locked": v2["mean_pr_auc"] > LOCKED_CHRONO_PR,
        "beats_recorded_ps2": v2["mean_pr_auc"] > PS2_CHRONO_PR,
    }
    (RESULTS / "final_recipe.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'final_recipe.json'}")


STAGES = {
    "anchor": stage_anchor,
    "features": stage_features,
    "models": stage_models,
    "capacity": stage_capacity,
    "combo": stage_combo,
    "calibration": stage_calibration,
    "robustness": stage_robustness,
    "final": stage_final,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "anchor"
    if which not in STAGES:
        raise SystemExit(f"unknown stage {which!r}; pick one of {sorted(STAGES)}")
    STAGES[which]()
