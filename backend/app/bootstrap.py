"""Train M1 (required). M2/M3 are optional and slow — pass --all to fit them.

    python -m app.bootstrap
    python -m app.bootstrap --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.ml.pipeline import FeatureSpec, future_core_split, load_frame  # noqa: E402
from app.ml.train import DatasetContext, train_on_indices  # noqa: E402
from app.services.registry import get_registry  # noqa: E402


def _put(mid: str, bundle, path: Path) -> None:
    joblib.dump({"bundle": bundle, "metrics": bundle.metrics}, path, compress=3)
    get_registry().put(mid, bundle, bundle.metrics, path)  # type: ignore[arg-type]
    print(mid, bundle.metrics, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Also train M2 (elapsed_days) and M3 (leaky)")
    parser.add_argument("--trees", type=int, default=120)
    args = parser.parse_args()

    settings = get_settings()
    settings.registry_dir.mkdir(parents=True, exist_ok=True)
    enhance = settings.enhance_bundle
    m1_path = settings.registry_dir / "M1.joblib"

    if enhance.exists():
        obj = joblib.load(enhance)
        _put("M1", obj["bundle"], m1_path)
        print("M1 copied from Phase_1_enhancement")
    elif not m1_path.exists():
        work, y, open_dates, _lookup, dict_cols = load_frame()
        ctx = DatasetContext(work, y, open_dates, dict_cols)
        tr, va = future_core_split(open_dates)
        print("training M1 (no enhance bundle)", flush=True)
        bundle, _, _ = train_on_indices(ctx, tr, va, FeatureSpec(), n_estimators=args.trees)
        _put("M1", bundle, m1_path)
    else:
        print("M1 already on disk", m1_path)

    if args.all:
        work, y, open_dates, _lookup, dict_cols = load_frame()
        ctx = DatasetContext(work, y, open_dates, dict_cols)
        tr, va = future_core_split(open_dates)
        for mid, spec in (
            ("M2", FeatureSpec(include_elapsed=True)),
            ("M3", FeatureSpec(include_leaky=True)),
        ):
            print("training", mid, flush=True)
            bundle, _, _ = train_on_indices(ctx, tr, va, spec, n_estimators=args.trees)
            dest = settings.registry_dir / f"{mid}.joblib"
            _put(mid, bundle, dest)
    print("registry", settings.registry_dir)


if __name__ == "__main__":
    main()
