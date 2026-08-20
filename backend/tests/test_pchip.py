from app.ml.pipeline import PchipAbort, apply_pchip, fit_pchip
import numpy as np


def test_pchip_monotone():
    rng = np.random.default_rng(0)
    raw = np.clip(rng.normal(0.3, 0.2, 400), 0, 1)
    y = (raw + rng.normal(0, 0.05, 400) > 0.4).astype(int)
    px, py = fit_pchip(raw, y)
    cal = apply_pchip(np.array([0.1, 0.5, 0.9]), px, py)
    assert cal[0] <= cal[1] <= cal[2]


def test_pchip_abort_on_constant():
    raw = np.zeros(50)
    y = np.zeros(50)
    try:
        fit_pchip(raw, y)
        raise AssertionError("expected abort")
    except PchipAbort:
        pass
