"""Train a FittedBundle on a time slice and score a holdout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.pipeline import (
    FeatureSpec,
    FittedBundle,
    blend_proba,
    classify_columns,
    fit_bundle,
    make_xy,
    row_moments,
)


class DatasetContext:
    def __init__(self, work: pd.DataFrame, y: pd.Series, open_dates: pd.Series, dict_cols: list[str]):
        self.work = work
        self.y = y
        self.open_dates = open_dates
        self.dict_cols = dict_cols
        self.categorical_cols, self.temporal_cols, self.numeric = classify_columns(work)
        self.moments = row_moments(self.numeric)


def train_on_indices(
    ctx: DatasetContext,
    tr_idx: np.ndarray,
    va_idx: np.ndarray | None,
    spec: FeatureSpec,
    n_estimators: int = 400,
    smote_ratio: float = 0.0,
    hp: dict | None = None,
    strict_pchip: bool = False,
) -> tuple[FittedBundle, pd.DataFrame | None, np.ndarray | None]:
    if va_idx is None:
        va_idx = tr_idx[-min(len(tr_idx) // 8, 400) :]
    tr, va, y_tr, y_va, ord_maps, te_maps, g, iso, iso_scaler = make_xy(
        ctx.work,
        ctx.numeric,
        ctx.moments,
        ctx.y,
        ctx.open_dates,
        tr_idx,
        va_idx,
        ctx.categorical_cols,
        ctx.temporal_cols,
        ctx.dict_cols,
        spec,
    )
    bundle, raw_va = fit_bundle(
        tr,
        y_tr,
        va,
        y_va,
        spec,
        n_estimators=n_estimators,
        smote_ratio=smote_ratio,
        hp=hp,
        strict_pchip=strict_pchip,
    )
    bundle.cat_maps = ord_maps
    bundle.te_maps = te_maps
    bundle.te_global = g
    bundle.iso = iso
    bundle.iso_scaler = iso_scaler
    bundle.train_idx = np.asarray(tr_idx)
    return bundle, va, raw_va


def transform_matrix(ctx: DatasetContext, idx: np.ndarray, bundle: FittedBundle) -> pd.DataFrame:
    dummy_va = idx
    spec = bundle.spec
    tr, va, *_ = make_xy(
        ctx.work,
        ctx.numeric,
        ctx.moments,
        ctx.y,
        ctx.open_dates,
        idx,
        dummy_va,
        ctx.categorical_cols,
        ctx.temporal_cols,
        ctx.dict_cols,
        spec,
    )
    return va.reindex(columns=bundle.columns)


def score_matrix(bundle: FittedBundle, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer

    cols = bundle.columns
    Xi = pd.DataFrame(bundle.imputer.transform(X.reindex(columns=cols)), columns=cols, index=X.index)
    raw = blend_proba(bundle.xgb, bundle.lgb, Xi)
    from app.ml.pipeline import apply_pchip

    cal = apply_pchip(raw, bundle.pchip_x, bundle.pchip_y)
    return raw, cal
