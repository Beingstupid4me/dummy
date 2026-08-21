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
from app.services.webhook import list_escrow_webhooks

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


@router.get("/retrain/{job_id}/stream")
async def retrain_stream(job_id: str):
    """SSE streaming of retrain job logs in real time."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        last_idx = 0
        while True:
            current_job = get_job(job_id)
            if current_job is None:
                break
            if len(current_job.log) > last_idx:
                for line in current_job.log[last_idx:]:
                    payload = {
                        "job_id": job_id,
                        "status": current_job.status,
                        "line": line,
                        "metrics": current_job.metrics,
                        "elapsed_s": current_job.elapsed_s,
                    }
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                last_idx = len(current_job.log)

            if current_job.status in ("done", "aborted", "error"):
                done_payload = {
                    "job_id": job_id,
                    "status": current_job.status,
                    "metrics": current_job.metrics,
                    "elapsed_s": current_job.elapsed_s,
                    "done": True,
                }
                yield f"data: {json.dumps(done_payload, default=str)}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/feeds/gov")
def post_gov(ticket: GovTicket):
    ingest_ticket(ticket)
    return {"ok": True}


@router.get("/feeds/gov")
def get_gov():
    return list_tickets()


@router.get("/webhooks/escrow")
def list_webhooks():
    return list_escrow_webhooks()


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
