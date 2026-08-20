from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import (
    ActiveModelRequest,
    GovTicket,
    RetrainRequest,
    ScoreRequest,
)
from app.services.engine import context, ensure_m1_loaded, score
from app.services.gov import ingest_ticket, list_tickets
from app.services.registry import get_registry
from app.services.retrain import get_job, submit
from app.services.store import get_store

router = APIRouter()


@router.get("/health")
def health() -> dict:
    ensure_m1_loaded()
    return {"ok": True, "active": get_registry().active_id}


@router.post("/score")
def post_score(req: ScoreRequest):
    ensure_m1_loaded()
    try:
        return score(req)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/registry")
def registry():
    ensure_m1_loaded()
    return get_registry().snapshot()


@router.post("/registry/active")
def set_active(req: ActiveModelRequest):
    ensure_m1_loaded()
    try:
        get_registry().set_active(req.model_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"active": get_registry().active_id, "downtime_ms": 0}


@router.post("/retrain", status_code=202)
def post_retrain(req: RetrainRequest):
    ensure_m1_loaded()
    job_id = submit(req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/retrain/{job_id}")
def retrain_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@router.post("/feeds/gov")
def post_gov(ticket: GovTicket):
    ingest_ticket(ticket)
    return {"ok": True}


@router.get("/feeds/gov")
def get_gov():
    return list_tickets()


@router.get("/webhooks/escrow")
def list_webhooks():
    store = get_store()
    keys = store.keys("webhook:")
    return [store.get(k) for k in keys]


@router.get("/stream")
async def stream():
    """SSE replay of /score on rotating sample accounts."""
    ensure_m1_loaded()
    context()
    samples = ["XXXX2203", "XXXX2411", "XXXX2688", "XXXX2904"]

    async def gen():
        i = 0
        while True:
            acct = samples[i % len(samples)]
            ch = ["UPI", "ATM", "IMPS", "NEFT"][i % 4]
            amt = 8000 + (i * 137) % 240000
            try:
                out = await asyncio.to_thread(
                    score, ScoreRequest(account=acct, channel=ch, amount=amt)
                )
                payload = out.model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                payload = {"error": str(exc), "ts": datetime.now(timezone.utc).isoformat()}
            yield f"data: {json.dumps(payload, default=str)}\n\n"
            i += 1
            await asyncio.sleep(1.6)

    return StreamingResponse(gen(), media_type="text/event-stream")
