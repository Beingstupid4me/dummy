from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.schemas import AlertStatus, RoutePath


def route(gov_hit: bool, tms_flags: list[str], p_calib: float) -> RoutePath:
    if gov_hit or tms_flags or p_calib >= 0.55:
        return "FIREWALL_FIRST"
    return "ML_FIRST"


def status(p_calib: float, threshold: float) -> AlertStatus:
    if p_calib >= max(0.62, threshold):
        return "ESCROW"
    if p_calib >= max(threshold, 0.32):
        return "QUEUE"
    return "CLEAR"


def hold_until(st: AlertStatus) -> datetime | None:
    if st != "ESCROW":
        return None
    return datetime.now(timezone.utc) + timedelta(minutes=get_settings().escrow_minutes)


def unit_cost(rt: RoutePath) -> float:
    s = get_settings()
    return s.c_search if rt == "FIREWALL_FIRST" else s.c_inference
