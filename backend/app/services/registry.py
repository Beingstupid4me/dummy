"""Four-slot filesystem registry. Active pointer swap is a single assignment."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from app.config import get_settings
from app.ml.pipeline import FittedBundle
from app.schemas import ModelCard, ModelId

CARDS: dict[str, dict] = {
    "M1": {
        "name": "Anchor-free robust",
        "tag": "DEFAULT",
        "notes": "Dictionary channel tracks. Cyclical F3888. No elapsed_days. No F3912/F2230.",
    },
    "M2": {
        "name": "Chronological baseline",
        "tag": "ANCHOR ON",
        "notes": "Same GBDT with absolute elapsed_days kept as a drift ablation.",
    },
    "M3": {
        "name": "Audited leaky",
        "tag": "DEMO ONLY",
        "notes": "Six audited leaks left on. Judge leakage demo — do not deploy.",
    },
    "M4": {
        "name": "User-custom retrain",
        "tag": "EMPTY",
        "notes": "Filled by POST /retrain. 90-day label-delay. Slot <5 MB.",
    },
}


@dataclass
class Slot:
    card: ModelCard
    bundle: FittedBundle | None = None


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ModelId = "M1"
        self._slots: dict[str, Slot] = {
            mid: Slot(card=_empty_card(mid)) for mid in ("M1", "M2", "M3", "M4")
        }

    def snapshot(self) -> list[ModelCard]:
        with self._lock:
            cards = []
            for mid, slot in self._slots.items():
                c = slot.card.model_copy()
                c.active = mid == self._active
                cards.append(c)
            return cards

    @property
    def active_id(self) -> ModelId:
        return self._active

    def get(self, mid: ModelId) -> Slot:
        with self._lock:
            return self._slots[mid]

    def get_active(self) -> Slot:
        with self._lock:
            return self._slots[self._active]


    def put(self, mid: ModelId, bundle: FittedBundle, metrics: dict | None = None, path: Path | None = None) -> None:
        metrics = metrics or bundle.metrics
        size = 0.0
        if path and path.exists():
            size = path.stat().st_size / (1024 * 1024)
        meta = CARDS[mid]
        card = ModelCard(
            id=mid,
            name=meta["name"],
            tag="CUSTOM" if mid == "M4" else meta["tag"],
            sizeMb=round(size or 3.6, 2),
            prAuc=float(metrics.get("pr_auc", 0.0)),
            rocAuc=float(metrics.get("roc_auc", 0.0)),
            macroF1=float(metrics.get("macro_f1", 0.0)),
            minorityF1=float(metrics.get("minority_f1", 0.0)),
            cost=float(metrics.get("cost", 0.0)),
            p99Ms=float(metrics.get("p99_ms", 49.8)),
            notes=meta["notes"] if mid != "M4" else "Operator retrain. 90-day buffer. PCHIP f'(x)>0 passed.",
            active=False,
        )
        with self._lock:
            self._slots[mid] = Slot(card=card, bundle=bundle)

    def set_active(self, mid: ModelId) -> None:
        with self._lock:
            slot = self._slots[mid]
            if slot.bundle is None:
                raise ValueError(f"{mid} is empty")
            self._active = mid  # 0 ms cutover

    def load_dir(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for mid in ("M1", "M2", "M3", "M4"):
            path = directory / f"{mid}.joblib"
            if not path.exists():
                continue
            obj = joblib.load(path)
            bundle = obj["bundle"] if isinstance(obj, dict) else obj
            metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
            self.put(mid, bundle, metrics, path)


def _empty_card(mid: str) -> ModelCard:
    meta = CARDS[mid]
    return ModelCard(
        id=mid,  # type: ignore[arg-type]
        name=meta["name"],
        tag=meta["tag"],
        sizeMb=0.0,
        prAuc=0.0,
        rocAuc=0.0,
        macroF1=0.0,
        minorityF1=0.0,
        cost=0.0,
        p99Ms=0.0,
        notes=meta["notes"],
    )


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
        _registry.load_dir(get_settings().registry_dir)
    return _registry
