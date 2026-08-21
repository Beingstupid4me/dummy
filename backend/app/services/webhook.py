from __future__ import annotations

import logging
from datetime import datetime, timezone
import json
import urllib.request
import urllib.error
from app.config import get_settings
from app.services.store import TTL_7D, get_store

logger = logging.getLogger("sentinelflow.webhook")


def record_and_dispatch_webhook(tx_id: str, account: str, amount: float, channel: str, hold_min: int) -> dict:
    """Record an escrow event in store and dispatch outbound webhook if configured."""
    store = get_store()
    now_iso = datetime.now(timezone.utc).isoformat()
    endpoint = get_settings().escrow_webhook_url or "POST /webhooks/escrow"
    payload = {
        "id": f"WH-{tx_id}",
        "txId": tx_id,
        "account": account,
        "amount": amount,
        "channel": channel,
        "ts": now_iso,
        "endpoint": endpoint,
        "holdMin": hold_min,
        "dispatched": False,
        "status_code": None,
    }

    url = get_settings().escrow_webhook_url
    if url and url.startswith("http"):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                payload["dispatched"] = True
                payload["status_code"] = resp.status
        except urllib.error.HTTPError as exc:
            payload["dispatched"] = True
            payload["status_code"] = exc.code
        except Exception as exc:  # noqa: BLE001
            logger.warning("Escrow webhook dispatch to %s failed: %s", url, exc)
            payload["dispatched"] = False
            payload["error"] = str(exc)

    store.set(f"webhook:{tx_id}", payload, TTL_7D)

    # maintain recent webhook list in store
    hist = store.get("webhooks:list") or []
    hist = [payload, *[h for h in hist if h.get("txId") != tx_id]][:50]
    store.set("webhooks:list", hist, TTL_7D)

    return payload


def list_escrow_webhooks() -> list[dict]:
    """Retrieve all recorded escrow webhook events."""
    store = get_store()
    hist = store.get("webhooks:list")
    if hist is not None:
        return hist
    keys = store.keys("webhook:")
    items = []
    for k in keys:
        item = store.get(k)
        if isinstance(item, dict):
            items.append(item)
    items.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return items
