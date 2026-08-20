from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Channel = Literal["UPI", "ATM", "IMPS", "NEFT", "NETBANK"]
ModelId = Literal["M1", "M2", "M3", "M4"]
RoutePath = Literal["ML_FIRST", "FIREWALL_FIRST"]
AlertStatus = Literal["CLEAR", "QUEUE", "ESCROW"]


class ScoreRequest(BaseModel):
    account: str
    channel: Channel = "UPI"
    amount: float = Field(gt=0)
    beneficiary: str = "UPI@okhdfc"
    row_id: int | None = None


class LatencyBreak(BaseModel):
    redis: float
    reconstruct: float
    gbdt: float
    pchip: float
    shap: float


class ShapBar(BaseModel):
    feature: str
    label: str
    value: float


class GraphNode(BaseModel):
    id: str
    kind: Literal["ego", "hop1", "hop2", "blacklist"]
    label: str
    risk: float


class GraphEdge(BaseModel):
    from_: str = Field(serialization_alias="from")
    to: str
    amount: float
    channel: str
    ttl: Literal["24h", "7d"]

    model_config = {"populate_by_name": True}


class RedisProfile(BaseModel):
    upiL7: float
    atmL7: float
    cardL7: float
    netL7: float
    vCross: float
    accel: float


class ScoreResponse(BaseModel):
    id: str
    ts: datetime
    account: str
    channel: Channel
    amount: float
    beneficiary: str
    pRaw: float
    pCalib: float
    latencyMs: float
    latency: LatencyBreak
    route: RoutePath
    status: AlertStatus
    holdUntil: datetime | None = None
    tmsFlags: list[str]
    govHit: bool
    shap: list[ShapBar]
    graph: dict
    profile: RedisProfile
    modelId: ModelId
    unitCost: float


class RetrainRequest(BaseModel):
    features_on: list[str] = Field(default_factory=list)
    features_off: list[str] = Field(default_factory=list)
    include_elapsed: bool = False
    include_leaky: bool = False
    include_tms: bool = False
    smote_ratio: float = 0.0
    n_estimators: int = 220
    learning_rate: float | None = None


class RetrainAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class RetrainStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "aborted", "error"]
    log: list[str]
    metrics: dict | None = None
    elapsed_s: float | None = None


class ModelCard(BaseModel):
    id: ModelId
    name: str
    tag: str
    sizeMb: float
    prAuc: float
    rocAuc: float
    macroF1: float
    minorityF1: float
    cost: float
    p99Ms: float
    notes: str
    active: bool = False


class ActiveModelRequest(BaseModel):
    model_id: ModelId


class GovTicket(BaseModel):
    id: str
    ts: datetime
    kind: Literal["IP_SUBNET", "DEVICE", "ACCOUNT"]
    value: str
    src: Literal["I4C", "NCRP"] = "I4C"


class WebhookEvent(BaseModel):
    id: str
    txId: str
    ts: datetime
    endpoint: str
    holdMin: int
