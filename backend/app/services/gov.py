from __future__ import annotations

from datetime import datetime, timezone

from app.schemas import GovTicket
from app.services.store import TTL_7D, get_store


def ingest_ticket(ticket: GovTicket) -> None:
    store = get_store()
    store.set(f"gov:{ticket.kind}:{ticket.value}", ticket.model_dump(mode="json"), TTL_7D)
    hist = store.get("gov:list") or []
    hist = [ticket.model_dump(mode="json"), *hist][:50]
    store.set("gov:list", hist, TTL_7D)


def list_tickets() -> list[dict]:
    return get_store().get("gov:list") or []


def is_blacklisted(account: str) -> bool:
    store = get_store()
    if store.get(f"gov:ACCOUNT:{account}"):
        return True
    return False


def seed_tickets() -> None:
    if list_tickets():
        return
    now = datetime.now(timezone.utc)
    samples = [
        GovTicket(id="I4C-8000", ts=now, kind="ACCOUNT", value="XXXX2203", src="I4C"),
        GovTicket(id="I4C-8001", ts=now, kind="IP_SUBNET", value="103.21.12.0/24", src="NCRP"),
        GovTicket(id="I4C-8002", ts=now, kind="DEVICE", value="IMEI-35100211", src="I4C"),
    ]
    for t in samples:
        ingest_ticket(t)
