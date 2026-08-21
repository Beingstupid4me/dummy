"""Full report scorecard for Phase_1.2 — ranking, decisions, calibration, cost.

A mule model is not one number. The PS2 report quoted PR-AUC, ROC-AUC, macro
F1, minority F1 and an isotonic calibration curve, and the console needs a
probability that means something at a threshold. So every candidate recipe is
scored on four families:

  ranking      PR-AUC, ROC-AUC, recall inside a 1% / 5% alert budget
  decisions    macro F1, mule F1, balanced accuracy, precision, recall, cost
  calibration  Brier, log loss, expected calibration error, calibration slope
  composite    one transparent weighted number so there is a single winner

Thresholds and the calibration map are fitted on the training window only —
never on the validation slice. `matched` variants (threshold tuned on val) are
carried purely because the earlier reports quoted them.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from cost import (
    R_HEADLINE,
    cost_report,
    pick_cost_threshold,
    pick_f1_threshold,
    rate_matched_threshold,
)

COST_FN, COST_FP = R_HEADLINE, 1.0
ALERT_BUDGETS = (0.01, 0.05)
# Ranking, decision quality, queue economics, probability quality. Weights are
# stated here rather than buried so the "overall winner" is auditable.
COMPOSITE_WEIGHTS = {
    "pr_auc": 0.30,
    "mule_f1": 0.20,
    "recall_at_1pct": 0.15,
    "cost_efficiency": 0.20,
    "calibration": 0.15,
}


class PchipAbort(RuntimeError):
    """Isotonic collapsed or the smoothed map is not monotone."""


def fit_calibrator(raw: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic knots smoothed by PCHIP, the map the reports used."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y)
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    uniq_x, idx = np.unique(xs, return_index=True)
    uniq_y = ys[idx]
    if len(uniq_x) < 2:
        raise PchipAbort("isotonic collapsed to a constant")
    spline = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
    grid = np.linspace(float(uniq_x.min()), float(uniq_x.max()), 64)
    if not np.all(spline.derivative()(grid) >= -1e-7):
        raise PchipAbort("smoothed map is not monotone")
    return uniq_x, uniq_y


def apply_calibrator(raw: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.clip(PchipInterpolator(px, py, extrapolate=True)(raw), 0.0, 1.0)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_slope(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Observed rate regressed on predicted rate across equal-count bins.

    Binning by rank, not by value: at a 0.9% base rate almost every calibrated
    probability sits near zero, so value quantiles collapse to one edge.
    """
    order = np.argsort(p)
    groups = np.array_split(order, bins)
    xs, ys = [], []
    for g in groups:
        if len(g) < 5:
            continue
        xs.append(float(p[g].mean()))
        ys.append(float(y[g].mean()))
    if len(xs) < 3 or np.std(xs) < 1e-12:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def cost_threshold(y: np.ndarray, p: np.ndarray, lo: float = 0.02, hi: float = 0.60) -> float:
    """Cheapest threshold under FN:FP = 5:1, searched on training-side scores."""
    grid = np.unique(np.clip(np.quantile(p, np.linspace(0.50, 0.999, 80)), lo, hi))
    best_t, best_c = float(np.clip(COST_FP / (COST_FP + COST_FN), lo, hi)), np.inf
    for t in grid:
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        c = fn * COST_FN + fp * COST_FP
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 120)))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.min() == pred.max():
            continue
        f = f1_score(y, pred, pos_label=1, zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t


def _decision_metrics(y: np.ndarray, p: np.ndarray, thr: float, prefix: str) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_threshold": float(thr),
        f"{prefix}_accuracy": float((tp + tn) / max(len(y), 1)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        f"{prefix}_macro_f1": float(f1_score(y, pred, average="macro")),
        f"{prefix}_mule_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        f"{prefix}_precision": float(tp / max(tp + fp, 1)),
        f"{prefix}_recall": float(tp / max(tp + fn, 1)),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_cost": float(fn * COST_FN + fp * COST_FP),
    }


def _budget_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    out = {}
    n_pos = max(int(y.sum()), 1)
    order = np.argsort(-score)
    for b in ALERT_BUDGETS:
        k = max(int(round(b * len(y))), 1)
        top = order[:k]
        hits = int(y[top].sum())
        tag = f"{int(b * 100)}pct"
        out[f"precision_at_{tag}"] = float(hits / k)
        out[f"recall_at_{tag}"] = float(hits / n_pos)
        out[f"lift_at_{tag}"] = float((hits / k) / max(y.mean(), 1e-9))
    return out


def score_fold(
    y_tr: np.ndarray,
    raw_tr: np.ndarray,
    y_va: np.ndarray,
    raw_va: np.ndarray,
    tail_frac: float = 1.0,
) -> dict:
    """Every metric for one validation slice. Nothing here is fitted on y_va."""
    y_tr = np.asarray(y_tr).astype(int)
    y_va = np.asarray(y_va).astype(int)

    row: dict = {
        "pr_auc": float(average_precision_score(y_va, raw_va)),
        "roc_auc": float(roc_auc_score(y_va, raw_va)),
        "val_mules": int(y_va.sum()),
        "val_n": int(len(y_va)),
    }
    row.update(_budget_metrics(y_va, raw_va))

    try:
        px, py = fit_calibrator(raw_tr, y_tr)
        cal_tr = apply_calibrator(raw_tr, px, py)
        cal_va = apply_calibrator(raw_va, px, py)
        row["pchip_ok"] = True
    except PchipAbort:
        cal_tr, cal_va = raw_tr, raw_va
        row["pchip_ok"] = False

    row["pr_auc_calibrated"] = float(average_precision_score(y_va, cal_va))
    row["brier"] = float(brier_score_loss(y_va, np.clip(cal_va, 1e-6, 1 - 1e-6)))
    row["log_loss"] = float(log_loss(y_va, np.clip(cal_va, 1e-6, 1 - 1e-6), labels=[0, 1]))
    row["ece"] = expected_calibration_error(y_va, cal_va)
    row["calibration_slope"] = calibration_slope(y_va, cal_va)
    # a base-rate-only model is the honest calibration reference
    prior = float(y_tr.mean())
    row["brier_skill"] = float(
        1.0 - row["brier"] / max(brier_score_loss(y_va, np.full(len(y_va), prior)), 1e-12)
    )

    # Three operating points off one score, each threshold fitted on
    # training-side scores only. They answer different questions and must not
    # be mixed: a cost-optimal cut deliberately buys recall with 5 false
    # positives per extra catch, so judging it by F1 would be a category error.
    cut = int(len(y_tr) * (1.0 - tail_frac))
    tail_y, tail_p = y_tr[cut:], cal_tr[cut:]
    if int(tail_y.sum()) < 4:
        tail_y, tail_p = y_tr, cal_tr
    thr_cost = pick_cost_threshold(tail_y, tail_p, COST_FN)
    thr_f1 = pick_f1_threshold(tail_y, tail_p)
    row.update(_decision_metrics(y_va, cal_va, thr_cost, "cost"))
    row.update(_decision_metrics(y_va, cal_va, thr_f1, "f1"))
    # same two policies, carried across as an alert rate instead of a raw cut
    row.update(_decision_metrics(y_va, cal_va, rate_matched_threshold(tail_p, cal_va, thr_cost), "costrate"))
    row.update(_decision_metrics(y_va, cal_va, rate_matched_threshold(tail_p, cal_va, thr_f1), "f1rate"))
    row.update(_decision_metrics(y_va, cal_va, np.quantile(cal_va, 1.0 - 0.01), "budget1"))
    # oracle threshold: what the earlier reports quoted, kept only for comparison
    row.update(_decision_metrics(y_va, cal_va, f1_threshold(y_va, cal_va), "oracle"))
    row.update(cost_report(tail_y, tail_p, y_va, cal_va))
    return row


def composite(mean_row: dict) -> tuple[float, dict]:
    """One transparent number: ranking, decisions, queue economics, probability."""
    parts = {
        "pr_auc": float(np.clip(mean_row["pr_auc"], 0, 1)),
        "mule_f1": float(np.clip(mean_row["f1_mule_f1"], 0, 1)),
        "recall_at_1pct": float(np.clip(mean_row["recall_at_1pct"], 0, 1)),
        "cost_efficiency": float(np.clip(1.0 - mean_row[f"normcost_r{int(R_HEADLINE)}"], 0, 1)),
        "calibration": float(np.clip(1.0 - mean_row["ece"] / 0.05, 0, 1)),
    }
    total = sum(COMPOSITE_WEIGHTS[k] * v for k, v in parts.items())
    return float(total), parts
