from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import router
from app.services.engine import ensure_m1_loaded
from app.services.gov import seed_tickets


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings().registry_dir.mkdir(parents=True, exist_ok=True)
    seed_tickets()
    try:
        ensure_m1_loaded()
        from app.services.engine import context, materialize
        from app.services.registry import get_registry

        context()
        materialize(get_registry().active_id)
    except Exception as exc:  # noqa: BLE001
        print("startup: model load deferred", exc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="SentinelFlow",
        version="2.0.0",
        description="Mule-account prevention engine. /score /retrain /registry.",
        lifespan=lifespan,
    )
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
