"""Score path: Redis/profile lookup → reconstruct → GBDT → PCHIP → cost/escrow."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import get_settings
from app.ml.pipeline import apply_pchip, blend_proba, load_frame, overlay_txn, shap_top
from app.ml.train import DatasetContext, train_on_indices
from app.schemas import RedisProfile, ScoreRequest, ScoreResponse
from app.services import decision
from app.services.graph import ego_graph
from app.services.gov import is_blacklisted, seed_tickets
from app.services.registry import get_registry
from app.services.store import TTL_24H, get_store
from app.services.webhook import record_and_dispatch_webhook

_TMS_MAP = {
    "F3900": "HIGH_VALUE_UPI_DB_TXNS",
    "F3901": "MULTI_DBS_FROM_ACCOUNT",
    "F3902": "MULTI_PG_TXNS",
    "F3903": "PWD_CHANGED_LARGE_FUND_XFERS",
    "F3904": "RCVING_FUNDS_FROM_MULITPLE_USERS",
    "F3905": "RISKY_COUNTRY_TXNS",
    "F3906": "STATUS_CHANGE_AFTER_WD",
    "F3907": "TXN_AT_UNUSUAL_TIME",
    "F3908": "MULTI_UPI_DB_TXNS",
    "F3909": "FAILED_UPI_TXNS",
    "F3910": "ONE_TO_MANY_UPI_PAYMENTS",
    "F3911": "OTHER_ALERT_TYPES",
    "F3916": "L3_FLG",
    "F3917": "L2_FLG",
    "F3918": "L1_FLG",
    "F3919": "COUNT_ALERTS",
}


def evaluate_tms_flags(req: ScoreRequest, idx: int, profile: RedisProfile) -> list[str]:
    """Evaluate TMS rules from historical indicators and real-time transaction telemetry."""
    flags: list[str] = []
    try:
        w = context().work.iloc[idx]
        for col, label in _TMS_MAP.items():
            if col in w.index and pd.notna(w[col]):
                try:
                    val = float(w[col])
                    if val > 0 and label not in flags:
                        flags.append(label)
                except (ValueError, TypeError):
                    pass
    except Exception:  # noqa: BLE001
        pass

    # Real-time event rules
    if req.channel == "UPI" and req.amount >= 100000 and "HIGH_VALUE_UPI_DB_TXNS" not in flags:
        flags.append("HIGH_VALUE_UPI_DB_TXNS")
    if req.amount >= 200000 and "LARGE_FUND_XFERS" not in flags:
        flags.append("LARGE_FUND_XFERS")
    if profile.accel >= 4.5 and "RAPID_TXN_BURST" not in flags:
        flags.append("RAPID_TXN_BURST")
    if profile.vCross >= 3.5 and "CROSS_CHANNEL_PASS_THROUGH" not in flags:
        flags.append("CROSS_CHANNEL_PASS_THROUGH")

    return flags

_ctx: DatasetContext | None = None
_X: dict[str, pd.DataFrame] = {}
_accounts: dict[str, int] = {}
_row_account: list[str] = []


def context() -> DatasetContext:
    global _ctx
    if _ctx is None:
        work, y, open_dates, _lookup, dict_cols = load_frame()
        _ctx = DatasetContext(work, y, open_dates, dict_cols)
        _index_accounts(work)
    return _ctx


def _index_accounts(work: pd.DataFrame) -> None:
    global _accounts, _row_account
    _accounts = {}
    _row_account = []
    mules = []
    legits = []
    for i in range(len(work)):
        alias = f"XXXX{(2200 + i) % 10000:04d}"
        key = f"A{i:06d}"
        _accounts[alias] = i
        _accounts[key] = i
        _accounts[str(i)] = i
        _row_account.append(alias)
        yv = int(context().y.iloc[i])
        (mules if yv == 1 else legits).append(i)
    samples = {
        "XXXX2203": mules[0] if mules else 0,
        "XXXX2411": legits[10] if len(legits) > 10 else 1,
        "XXXX2688": mules[1] if len(mules) > 1 else 0,
        "XXXX2904": legits[50] if len(legits) > 50 else 2,
    }
    for name, idx in samples.items():
        _accounts[name] = idx
        _row_account[idx] = name


def resolve_account(account: str, row_id: int | None) -> int:
    if row_id is not None:
        return int(row_id)
    if account in _accounts:
        return _accounts[account]
    raise KeyError(f"unknown account {account}")


def materialize(model_id: str) -> pd.DataFrame:
    if model_id in _X:
        return _X[model_id]
    ctx = context()
    slot = get_registry().get(model_id)
    if slot.bundle is None:
        raise RuntimeError(f"{model_id} has no weights")
    bundle = slot.bundle
    tr_idx = getattr(bundle, "train_idx", None)
    if tr_idx is None:
        n = len(ctx.y)
        tr_idx = np.arange(int(0.7 * n))
    from app.ml.pipeline import make_xy

    all_idx = np.arange(len(ctx.y))
    _tr, va, *_rest = make_xy(
        ctx.work,
        ctx.numeric,
        ctx.moments,
        ctx.y,
        ctx.open_dates,
        np.asarray(tr_idx),
        all_idx,
        ctx.categorical_cols,
        ctx.temporal_cols,
        ctx.dict_cols,
        bundle.spec,
    )
    X = pd.DataFrame(
        bundle.imputer.transform(va.reindex(columns=bundle.columns)),
        columns=bundle.columns,
        index=va.index,
    )
    _X[model_id] = X
    return X


def _g(row: pd.Series, *names: str) -> float:
    for col in names:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return 0.0


def _profile_from_row(row: pd.DataFrame) -> RedisProfile:
    r = row.iloc[0]
    return RedisProfile(
        upiL7=_g(r, "log_upi_cr_L7D"),
        atmL7=_g(r, "log_atm_db_L7D"),
        cardL7=_g(r, "log_pos_db_L7D"),
        netL7=_g(r, "log_net_db_L7D"),
        vCross=_g(r, "V_cross"),
        accel=_g(r, "txn_accel", "txn_accel_L7D"),
    )


def _apply_cached(row: pd.DataFrame, cached: dict) -> pd.DataFrame:
    out = row.copy()
    idx = out.index[0]
    for col, val in cached.items():
        if col not in out.columns or val is None:
            continue
        try:
            out.at[idx, col] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _row_cache(row: pd.DataFrame) -> dict[str, float]:
    r = row.iloc[0]
    out: dict[str, float] = {}
    for col in row.columns:
        v = r[col]
        if pd.notna(v):
            try:
                out[col] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def score(req: ScoreRequest) -> ScoreResponse:
    t0 = time.perf_counter()
    context()
    seed_tickets()
    store = get_store()
    t_redis = time.perf_counter()
    idx = resolve_account(req.account, req.row_id)
    account = _row_account[idx]
    gov_hit = is_blacklisted(req.account) or is_blacklisted(account)
    feat_key = f"feat:{account}"
    cached = store.get(feat_key)
    redis_ms = (time.perf_counter() - t_redis) * 1000

    t_rec = time.perf_counter()
    slot = get_registry().get_active()
    if slot.bundle is None:
        raise RuntimeError("no active model — run python -m app.bootstrap")
    bundle = slot.bundle
    mid = get_registry().active_id
    X = materialize(mid)
    row = X.iloc[[idx]].copy()
    if isinstance(cached, dict):
        row = _apply_cached(row, cached)
    row = overlay_txn(row, req.channel, req.amount)
    profile = _profile_from_row(row)
    store.set(feat_key, _row_cache(row), TTL_24H)
    store.set(f"profile:{account}", profile.model_dump(), TTL_24H)
    reconstruct_ms = (time.perf_counter() - t_rec) * 1000

    t_gbdt = time.perf_counter()
    row_model = row.reindex(columns=bundle.columns)
    raw = float(blend_proba(bundle.xgb, bundle.lgb, row_model)[0])
    gbdt_ms = (time.perf_counter() - t_gbdt) * 1000

    t_pchip = time.perf_counter()
    calib = float(apply_pchip(np.array([raw]), bundle.pchip_x, bundle.pchip_y)[0])
    pchip_ms = (time.perf_counter() - t_pchip) * 1000

    tms = evaluate_tms_flags(req, idx, profile)
    shap_ms = 0.0
    shap_bars = []
    if calib >= get_settings().shap_min_p:
        t_s = time.perf_counter()
        shap_bars = shap_top(bundle, row_model, k=6)
        shap_ms = (time.perf_counter() - t_s) * 1000

    rt = decision.route(gov_hit, tms, calib)
    st = decision.status(calib, bundle.threshold)
    hold = decision.hold_until(st)
    graph = ego_graph(account, req.amount, req.channel, gov_hit)
    tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
    if st == "ESCROW":
        record_and_dispatch_webhook(
            tx_id=tx_id,
            account=account,
            amount=req.amount,
            channel=req.channel,
            hold_min=get_settings().escrow_minutes,
        )
    total = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        id=tx_id,
        ts=datetime.now(timezone.utc),
        account=account,
        channel=req.channel,
        amount=req.amount,
        beneficiary=req.beneficiary,
        pRaw=raw,
        pCalib=calib,
        latencyMs=total,
        latency={
            "redis": redis_ms,
            "reconstruct": reconstruct_ms,
            "gbdt": gbdt_ms,
            "pchip": pchip_ms,
            "shap": shap_ms,
        },
        route=rt,
        status=st,
        holdUntil=hold,
        tmsFlags=tms,
        govHit=gov_hit,
        shap=shap_bars,
        graph=graph,
        profile=profile,
        modelId=mid,
        unitCost=decision.unit_cost(rt),
    )


def ensure_m1_loaded() -> None:
    """Load enhancement bundle or train a compact M1 so /score never 503s on a fresh clone."""
    from pathlib import Path

    import joblib

    from app.ml.pipeline import FeatureSpec, future_core_split
    from app.services.registry import get_registry

    reg = get_registry()
    if reg.get_active().bundle is not None:
        return
    settings = get_settings()
    path = settings.enhance_bundle
    if path.exists():
        obj = joblib.load(path)
        bundle = obj["bundle"]
        get_registry().put("M1", bundle, bundle.metrics, path)
        dest = settings.registry_dir / "M1.joblib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"bundle": bundle, "metrics": bundle.metrics}, dest, compress=3)
        return
    ctx = context()
    tr, va = future_core_split(ctx.open_dates)
    bundle, _, _ = train_on_indices(ctx, tr, va, FeatureSpec(), n_estimators=80)
    bundle.train_idx = tr
    get_registry().put("M1", bundle, bundle.metrics)
    settings.registry_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bundle": bundle, "metrics": bundle.metrics}, settings.registry_dir / "M1.joblib", compress=3)
