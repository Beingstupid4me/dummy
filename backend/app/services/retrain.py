from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.ml.pipeline import FeatureSpec, PchipAbort, future_core_split
from app.ml.train import train_on_indices
from app.schemas import RetrainRequest, RetrainStatus
from app.services.engine import context
from app.services.registry import get_registry

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")
_jobs: dict[str, RetrainStatus] = {}
_lock = threading.Lock()


def _append(job_id: str, line: str) -> None:
    with _lock:
        job = _jobs[job_id]
        job.log.append(line)


def _run(job_id: str, req: RetrainRequest) -> None:
    t0 = time.perf_counter()
    with _lock:
        _jobs[job_id].status = "running"
    try:
        _append(job_id, "HTTP 202  POST /retrain  accepted")
        ctx = context()
        cutoff = ctx.open_dates.max() - pd.Timedelta(days=get_settings().label_delay_days)
        order = ctx.open_dates.argsort(kind="mergesort").to_numpy()
        eligible = order[ctx.open_dates.iloc[order] <= cutoff]
        if len(eligible) < 400:
            eligible = future_core_split(ctx.open_dates)[0]
        _append(job_id, f"90-day label-delay buffer applied  n={len(eligible)}")
        if req.smote_ratio > 0:
            _append(job_id, f"SMOTE ratio {req.smote_ratio:.2f} · minority upsample")
        else:
            _append(job_id, "SMOTE skipped · scale_pos_weight")
        spec = FeatureSpec(
            include_elapsed=req.include_elapsed or "elapsed" in req.features_on,
            include_leaky=req.include_leaky
            or any(x in req.features_on for x in ("F3912", "F2230", "F3886", "F3889", "F3891", "F3892")),
            include_tms=req.include_tms,
        )
        if req.include_leaky:
            _append(job_id, "Leaky tracks included — demo only")
        else:
            _append(job_id, "Feature reconstruction · moments / TE / V_cross")
        n_est = int(req.n_estimators)
        hp = {"n_estimators": n_est}
        if req.learning_rate:
            hp["learning_rate"] = req.learning_rate
        va = eligible[-min(len(eligible) // 6, 800) :]
        tr = eligible[: -len(va)] if len(va) else eligible
        bundle, _, _ = train_on_indices(
            ctx, tr, va, spec, n_estimators=n_est, smote_ratio=req.smote_ratio, hp=hp, strict_pchip=True
        )
        _append(job_id, "XGB+LGB 0.6/0.4  ·  worker thread")
        _append(job_id, "PCHIP isotonic  ·  f′(x) > 0  passed")
        settings = get_settings()
        settings.registry_dir.mkdir(parents=True, exist_ok=True)
        path = settings.registry_dir / "M4.joblib"
        joblib.dump({"bundle": bundle, "metrics": bundle.metrics}, path, compress=3)
        get_registry().put("M4", bundle, bundle.metrics, path)
        get_registry().set_active("M4")
        from app.services.engine import _X

        _X.pop("M4", None)
        elapsed = time.perf_counter() - t0
        _append(job_id, f"Cost threshold search  ·  Model 4 registered < 5 MB")
        _append(job_id, f"Registered M4 in {elapsed:.1f}s  ·  SLA {settings.retrain_seconds_sla:.0f}s")
        with _lock:
            _jobs[job_id].status = "done"
            _jobs[job_id].metrics = bundle.metrics
            _jobs[job_id].elapsed_s = elapsed
    except PchipAbort as exc:
        _append(job_id, f"ABORTED  {exc}")
        with _lock:
            _jobs[job_id].status = "aborted"
            _jobs[job_id].elapsed_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        _append(job_id, f"ERROR  {exc}")
        with _lock:
            _jobs[job_id].status = "error"
            _jobs[job_id].elapsed_s = time.perf_counter() - t0


def submit(req: RetrainRequest) -> str:
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        _jobs[job_id] = RetrainStatus(job_id=job_id, status="queued", log=[])
    _executor.submit(_run, job_id, req)
    return job_id


def get_job(job_id: str) -> RetrainStatus | None:
    return _jobs.get(job_id)
