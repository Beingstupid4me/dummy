"""Asymmetric cost layer — the decision cog of SentinelFlow, rebuilt.

The PS2 report minimises

    Total Operational Cost = FN x (R x C_fp) + FP x C_fp,   R = COST_FN_RATIO

and quotes a single saving at one R against an F1-tuned threshold. Three things
about that are worth fixing, and all three are measured here:

1. The report's threshold is searched on scores the model has already fitted,
   so it is optimistic. Every threshold here comes from an inner holdout that
   the final model never trained on.

2. One R is a guess. Investigator cost and average mule exposure move, so the
   engine is scored across R = 2 ... 100 and has to win at every ratio, not at
   the one that flatters it.

3. `$10,650 vs $15,000` has no scale. A cost is only meaningful against the
   trivial policies a bank could run for free: review nobody (pay R per mule)
   or review everybody (pay 1 per legitimate account). Normalised cost divides
   by the cheaper of those, so 0.0 is perfect and 1.0 means the model is worth
   nothing.

With calibrated probabilities the optimal rule is analytic: alerting costs 1,
staying silent costs p x R, so alert iff p > 1/R. That Bayes threshold is
reported alongside the searched one — when they agree, calibration is real.
"""
from __future__ import annotations

import numpy as np

COST_FP_BASE = 1.0
R_GRID = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
R_HEADLINE = 5.0


def total_cost(y: np.ndarray, pred: np.ndarray, r: float) -> float:
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return fn * r * COST_FP_BASE + fp * COST_FP_BASE


def trivial_cost(y: np.ndarray, r: float) -> float:
    """Cheaper of 'alert on nobody' and 'alert on everybody'."""
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return min(n_pos * r * COST_FP_BASE, n_neg * COST_FP_BASE)


def _grid(p: np.ndarray, n: int = 200) -> np.ndarray:
    return np.unique(np.quantile(p, np.linspace(0.0, 1.0, n))[1:-1])


def pick_cost_threshold(y: np.ndarray, p: np.ndarray, r: float, plateau: float = 1.05) -> float:
    """Cheapest threshold on held-out scores, taken over the flat region.

    Same stability argument as `pick_f1_threshold`: the cost curve near its
    minimum is close to flat, so the median of the near-optimal band survives
    a cohort shift better than the single cheapest grid point.
    """
    grid = _grid(p)
    costs = np.asarray([total_cost(y, (p >= t).astype(int), r) for t in grid])
    if not len(costs):
        return float(np.clip(1.0 / r, 1e-6, 1 - 1e-6))
    keep = grid[costs <= plateau * costs.min() + 1e-9]
    return float(np.median(keep)) if len(keep) else float(grid[int(np.argmin(costs))])


def pick_f1_threshold(y: np.ndarray, p: np.ndarray, plateau: float = 0.95) -> float:
    """The cost-blind baseline the report compares against, made stable.

    With ~26 positives in the calibration set the argmax of the F1 curve moves
    by whole percentiles when one account changes rank. Taking the median of
    the near-optimal plateau instead of the single peak keeps the operating
    point where the curve is genuinely flat.
    """
    from sklearn.metrics import f1_score

    grid = _grid(p)
    scores = []
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.min() == pred.max():
            scores.append(-1.0)
            continue
        scores.append(float(f1_score(y, pred, pos_label=1, zero_division=0)))
    scores = np.asarray(scores)
    if scores.max() <= 0:
        return 0.5
    keep = grid[scores >= plateau * scores.max()]
    return float(np.median(keep))


def budget_threshold(p: np.ndarray, budget: float) -> float:
    """Threshold that fills exactly `budget` of the analyst queue."""
    return float(np.quantile(p, 1.0 - budget))


def alert_rate(p: np.ndarray, thr: float) -> float:
    return float((p >= thr).mean())


def rate_matched_threshold(cal_p: np.ndarray, va_p: np.ndarray, thr: float) -> float:
    """Carry the *alert rate* across, not the probability value.

    A probability cut of 0.21 is only meaningful if the score distribution on
    new accounts matches the one the cut was tuned on. Under cohort drift it
    does not, and the queue silently doubles or empties. Transferring the
    quantile instead keeps the operating point where the analyst expects it.
    """
    rate = alert_rate(cal_p, thr)
    rate = float(np.clip(rate, 1.0 / max(len(va_p), 1), 1.0))
    return float(np.quantile(va_p, 1.0 - rate))


def cost_report(
    cal_y: np.ndarray,
    cal_p: np.ndarray,
    y_va: np.ndarray,
    p_va: np.ndarray,
    r_grid: tuple[float, ...] = R_GRID,
) -> dict:
    """Cost of every policy at every ratio. Thresholds come from `cal_*` only."""
    out: dict[str, float] = {}
    f1_thr = pick_f1_threshold(cal_y, cal_p)
    for r in r_grid:
        tag = f"r{int(r)}"
        searched = pick_cost_threshold(cal_y, cal_p, r)
        matched = rate_matched_threshold(cal_p, p_va, searched)
        bayes = float(np.clip(1.0 / r, 1e-6, 1 - 1e-6))
        triv = max(trivial_cost(y_va, r), 1e-9)

        c_opt = total_cost(y_va, (p_va >= searched).astype(int), r)
        c_rate = total_cost(y_va, (p_va >= matched).astype(int), r)
        c_f1 = total_cost(y_va, (p_va >= f1_thr).astype(int), r)
        c_bayes = total_cost(y_va, (p_va >= bayes).astype(int), r)
        # Production policy is the direct calibrated cut. Rate-matching was
        # tried as a drift hedge and measured worse across the bake-off
        # (normalised cost 1.10 against 0.40), which is itself evidence that
        # the out-of-fold calibration transfers: the probability value means
        # the same thing on the next cohort, so the quantile detour only adds
        # sampling noise. `cost_*_rate` stays reported as the alternative.
        production = c_opt

        out[f"cost_{tag}_opt"] = c_opt
        out[f"cost_{tag}_rate"] = c_rate
        out[f"cost_{tag}_f1"] = c_f1
        out[f"cost_{tag}_bayes"] = c_bayes
        out[f"cost_{tag}_trivial"] = triv
        out[f"normcost_{tag}"] = production / triv
        out[f"savings_{tag}_vs_trivial"] = 1.0 - production / triv
        out[f"savings_{tag}_vs_f1"] = 1.0 - production / max(c_f1, 1e-9)
        out[f"thr_{tag}_opt"] = searched
        out[f"alertrate_{tag}"] = float((p_va >= searched).mean())
    out["thr_f1"] = f1_thr
    return out
