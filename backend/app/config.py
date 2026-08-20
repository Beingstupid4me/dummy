from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    app_name = "SentinelFlow"
    redis_url = os.environ.get("SF_REDIS_URL", "")
    cors_origins = os.environ.get(
        "SF_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    backend_dir = Path(__file__).resolve().parents[1]
    repo_dir = backend_dir.parent
    registry_dir = backend_dir / "registry"
    enhance_bundle = repo_dir / "Phase_1_enhancement" / "models" / "enhance_m1.joblib"
    data_csv = repo_dir / "DataSet.csv"
    cost_fn_ratio = 5.0
    cost_fp_base = 1.0
    c_search = 0.4
    c_inference = 2.8
    escrow_minutes = 15
    shap_min_p = 0.32
    label_delay_days = 90
    retrain_seconds_sla = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
