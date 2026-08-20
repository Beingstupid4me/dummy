# SentinelFlow FastAPI engine

Inference gateway from `pro_report` and `phase2_roadmap.md`. The Next.js console talks to this process over REST + SSE. Redis is optional; the default store is process memory with the same key layout and TTLs.

## What it does

On `POST /score` the engine:

1. Resolves the account (sample aliases `XXXX2203` / `XXXX2411` / `XXXX2688` / `XXXX2904`, or `A000000`-style keys).
2. Loads the reconstructed feature row for the **active** registry model (imputed, fold-safe cats, cyclical `F3888`, Isolation Forest, optional PCA).
3. If Redis/memory already holds a live profile for that account, that snapshot is the history **up to now**.
4. Overlays this payment onto L7 channel tracks (`overlay_txn`): UPI → UPI credit L7, ATM → ATM debit, IMPS/NEFT → ELEC debit, NETBANK → net banking. Writes the snapshot back with a 24h TTL.
5. Scores 0.6 XGB + 0.4 LGB, maps through PCHIP, applies the cost-sensitive threshold, routes `ML_FIRST` vs `FIREWALL_FIRST`, and may escrow 15 minutes.
6. Conditional Taylor-scaled TreeSHAP when calibrated *P* ≥ 0.32.

That overlay is the operational stand-in for “the stream until this mule txn.” The CSV itself is not a txn log.

## Layout

```
app/
  main.py              ASGI app, CORS, startup warmup
  api.py               routes
  config.py            SF_REDIS_URL, SF_CORS_ORIGINS, paths
  schemas.py           request/response models
  bootstrap.py         M1 (default); --all also fits M2/M3
  ml/pipeline.py       dictionary tracks, moments, PCHIP, GBDT blend
  ml/train.py          train_on_indices on a time slice
  services/engine.py   /score path, account index, Redis write-back
  services/registry.py four filesystem slots, pointer swap
  services/retrain.py  HTTP 202 worker (90-day buffer, PCHIP abort)
  services/decision.py route / escrow / unit cost
  services/graph.py    1-hop / 2-hop edges with 24h and 7d TTL
  services/gov.py      simulated I4C / NCRP tickets
  services/store.py    MemoryKV or RedisKV
  tests/test_pchip.py  monotonicity + abort
registry/              M1–M4 *.joblib  (each < 5 MB)
```

Reconstruction (shared with `Phase_1_enhancement`):

- 14 row moments (count, missing/zero/sign rates, mean, std, min/max, median, q25/q75, IQR, abs mean)
- Cyclical sin/cos on **F3888 only** (never `F2230`)
- Fold-safe ordinal + Laplace target encoding
- Isolation Forest on the normal class
- Dictionary physics: `V_cross`, `txn_accel`, L7 bursts, channel entropy, occupation-relative balances `F3880–F3885`
- M1 excludes `elapsed_days`, `F3912`, `F2230`, `F3913–F3915`

## Run

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m app.bootstrap
.\.venv\Scripts\uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000/docs

**Windows:** do not pass `--reload`. WatchFiles tries to unlink locked files under `.venv` and raises `WinError 32`.

```powershell
python -m app.bootstrap --all --trees 120   # also fit M2 (elapsed_days) and M3 (leaky)
```

M1 is copied from `Phase_1_enhancement/models/enhance_m1.joblib` when that file exists.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{ ok, active }` |
| POST | `/score` | reconstruct + overlay + blend + PCHIP + cost/escrow |
| POST | `/retrain` | HTTP 202. 90-day label delay. Aborts if PCHIP \(f'(x) \le 0\) |
| GET | `/retrain/{job_id}` | log + metrics + elapsed |
| GET | `/registry` | four `ModelCard`s |
| POST | `/registry/active` | `{ model_id }` — 0 ms pointer swap |
| GET | `/stream` | SSE scored sample accounts (~1.6 s) |
| GET/POST | `/feeds/gov` | I4C / NCRP tickets |
| GET | `/webhooks/escrow` | keys stored on ESCROW (no outbound HTTP yet) |

Env:

| Variable | Default | Meaning |
|---|---|---|
| `SF_REDIS_URL` | unset (memory) | e.g. `redis://localhost:6379/0` |
| `SF_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | comma-separated |

AOF `appendfsync everysec` is a **Redis server** setting, not application code. Point `SF_REDIS_URL` at a Redis started with that policy if you need crash recovery.

## Latency

After the feature matrix is warm, `/score` is ~50 ms in-process (SLA < 100 ms; paper target p99 49.8 ms). The **first** score after process start rebuilds all rows (~tens of seconds). `main.py` warms M1 on startup so a restarted server should not stall the first click.

Retrain uses a `ThreadPoolExecutor` (one worker), not `ProcessPoolExecutor` as written in the paper. Scoring stays on the ASGI thread; retrain does not block `/score` except for GIL/CPU contention.

## Gaps vs report / roadmap

- No Docker Compose (paper: FastAPI + Redis + console as compose services).
- Registry slots **M2 and M3 are empty** until `bootstrap --all`. Hot-swap M1↔M3 is coded but has nothing to swap to.
- Retrain worker is a thread pool, not a process pool.
- `/retrain` honours `include_elapsed` / `include_leaky` / `include_tms` / `smote_ratio` / `n_estimators`. It does **not** apply a full per-column `features_on` allow-list from the console checklist.
- `tmsFlags` on `/score` is always `[]` — no TMS ingest stream.
- Escrow writes a Redis/memory key; it does not POST to a bank webhook URL.
- No `N=10_000` latency harness; the paper’s 49.8 ms p99 is not produced by a script in this repo.
- Retrain < 15 s SLA is **not verified** at 220–500 trees; the console currently requests 120 estimators.
