"""TTL key-value store. Redis when SF_REDIS_URL is set, otherwise process memory.

Memory backend is the default so the prototype runs without ops overhead.
The Redis path uses the same key layout (profile / edge / tx) and TTLs.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

from app.config import get_settings


class KV(Protocol):
    def set(self, key: str, value: Any, ttl: int) -> None: ...
    def get(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> None: ...
    def keys(self, prefix: str) -> list[str]: ...


class MemoryKV:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}

    def _purge(self) -> None:
        now = time.time()
        dead = [k for k, (exp, _) in self._data.items() if exp < now]
        for k in dead:
            self._data.pop(k, None)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, value)

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        exp, val = item
        if exp < time.time():
            self._data.pop(key, None)
            return None
        return val

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self, prefix: str) -> list[str]:
        self._purge()
        return [k for k in self._data if k.startswith(prefix)]


class RedisKV:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: Any, ttl: int) -> None:
        self._r.set(key, json.dumps(value), ex=ttl)

    def get(self, key: str) -> Any | None:
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def keys(self, prefix: str) -> list[str]:
        return [k for k in self._r.scan_iter(f"{prefix}*")]


_store: KV | None = None


def get_store() -> KV:
    global _store
    if _store is None:
        url = get_settings().redis_url
        _store = RedisKV(url) if url else MemoryKV()
    return _store


TTL_24H = 24 * 3600
TTL_7D = 7 * 24 * 3600
