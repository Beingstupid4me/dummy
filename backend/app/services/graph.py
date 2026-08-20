from __future__ import annotations

from app.services.store import TTL_24H, TTL_7D, get_store


def remember_edge(src: str, dst: str, amount: float, channel: str) -> None:
    store = get_store()
    store.set(f"edge:{src}:{dst}", {"amount": amount, "channel": channel, "ttl": "24h"}, TTL_24H)
    store.set(f"edge7:{src}:{dst}", {"amount": amount, "channel": channel, "ttl": "7d"}, TTL_7D)


def ego_graph(account: str, amount: float, channel: str, gov_hit: bool) -> dict:
    store = get_store()
    remember_edge(account, f"B-{account[-4:]}", amount, channel)
    hop1 = []
    for key in store.keys("edge:" + account):
        hop1.append(key.split(":")[-1])
    if not hop1:
        hop1 = [f"B{account[-4:]}{i}" for i in range(3)]
    nodes = [{"id": account, "kind": "ego", "label": account, "risk": 0.8 if gov_hit else 0.2}]
    edges = []
    for i, nid in enumerate(hop1[:4]):
        nodes.append({"id": nid, "kind": "hop1", "label": nid, "risk": 0.3})
        ttl = "24h" if i % 2 == 0 else "7d"
        payload = store.get(f"edge:{account}:{nid}") or {"amount": amount / (i + 1), "channel": channel}
        edges.append(
            {
                "from": account,
                "to": nid,
                "amount": payload.get("amount", amount),
                "channel": payload.get("channel", channel),
                "ttl": ttl,
            }
        )
    hop2 = f"C{account[-4:]}"
    nodes.append(
        {
            "id": hop2,
            "kind": "blacklist" if gov_hit else "hop2",
            "label": hop2,
            "risk": 0.95 if gov_hit else 0.12,
        }
    )
    edges.append(
        {
            "from": hop1[0],
            "to": hop2,
            "amount": max(amount * 0.4, 1000),
            "channel": channel,
            "ttl": "7d",
        }
    )
    return {"nodes": nodes, "edges": edges}
