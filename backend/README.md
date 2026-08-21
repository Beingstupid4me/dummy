# SentinelFlow FastAPI Engine

Inference gateway from `pro_report` and `phase2_roadmap.md`. The Next.js console communicates with this service over REST and Server-Sent Events (SSE). Redis is optional; the default storage engine is an in-memory TTL store matching the Redis key hierarchy and expiration semantics.

## Production Inference Flow (`POST /score`)

1. **Account Lookup**: Resolves sample aliases (`XXXX2203`, `XXXX2411`, `XXXX2688`, `XXXX2904`) or internal account identifiers.
2. **Feature Reconstruction**: Reads historical profile rollups (14 row moments, fold-safe categoricals, cyclical `F3888` trigonometric projections, Isolation Forest outlier scores, and cross-channel velocity ratios).
3. **Real-Time Transaction Overlay**: Overlays payment amounts onto $L7D$ channel buffers (`overlay_txn`), updating UPI, ATM, ELEC (IMPS/NEFT), and Card features, then caches the updated state under a 24-hour TTL.
4. **TMS & Regulatory Ingestion**: Ingests real-time transaction monitoring flags (e.g. `HIGH_VALUE_UPI_DB_TXNS`, `RAPID_TXN_BURST`, `CROSS_CHANNEL_PASS_THROUGH`) and checks I4C/NCRP central intelligence blacklists.
5. **Dual-GBDT Ensemble & Monotonic PCHIP Calibration**: Predicts raw risk via regularized XGBoost + LightGBM, then maps through monotonic PCHIP polynomials $f'(x) > 0$.
6. **Asymmetric Economic Thresholding & Escrow**: Determines routing (`ML_FIRST` vs `FIREWALL_FIRST`) and status (`CLEAR`, `QUEUE`, `ESCROW`). On `ESCROW`, an automated 15-minute fund hold is registered and dispatched via outbound escrow webhook.
7. **Taylor-Scaled TreeSHAP Explanations**: Generates local reason codes scaled by the continuous PCHIP spline derivative when $P_{\text{calib}} \ge 0.32$.

## Directory Layout

```
app/
  main.py              ASGI application, CORS middleware, startup matrix pre-warming
  api.py               REST API routes & SSE streaming endpoints (/stream, /retrain/{id}/stream)
  config.py            Settings (SF_REDIS_URL, SF_ESCROW_WEBHOOK_URL, CORS, SLAs)
  schemas.py           Pydantic v2 request/response models
  bootstrap.py         Registry initialization script (M1, M2, M3 pre-trained weights)
  ml/pipeline.py       Reconstruction pipeline, PCHIP calibration, GBDT blend
  ml/train.py          Fold-safe training utilities
  services/engine.py   /score scoring engine, TMS rules, Redis profile caching
  services/registry.py 4-slot model registry (<5 MB each) with 0 ms pointer swapping
  services/retrain.py  Asynchronous retrain worker with 90-day label delay and PCHIP gates
  services/decision.py Asymmetric cost loss routing & escrow decision thresholds
  services/graph.py    1-hop / 2-hop ego-graph generator with 24h/7d TTLs
  services/gov.py      I4C & NCRP central cyber-intelligence ticket store
  services/webhook.py  Outbound escrow webhook dispatcher & audit logger
  services/store.py    MemoryKV (default) & RedisKV implementations
benchmark.py           N=10,000 /score latency distribution benchmark harness
tests/test_pchip.py    PCHIP monotonicity validation and abort unit tests
registry/              M1.joblib, M2.joblib, M3.joblib, M4.joblib
```

## Running the Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m app.bootstrap --all --trees 150
.\.venv\Scripts\uvicorn app.main:app --port 8000
```

Open documentation at: `http://127.0.0.1:8000/docs`

> **Note (Windows)**: Do not use `--reload` on Windows due to file-locking conflicts with virtual environment files.

## REST & SSE API Surface

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and active model identifier |
| `POST` | `/score` | Real-time transaction scoring, overlay, PCHIP calibration, and explainability |
| `POST` | `/retrain` | Trigger on-demand retrain worker (HTTP 202, 90-day label buffer) |
| `GET` | `/retrain/{job_id}` | Retrain job status, logs, metrics, and execution time |
| `GET` | `/retrain/{job_id}/stream` | Server-Sent Events (SSE) live stream of retrain logs |
| `GET` | `/registry` | List all 4 registry model cards with performance metrics |
| `POST` | `/registry/active` | Hot-swap active scoring model (0 ms cutover) |
| `GET` | `/stream` | Real-time SSE stream of simulated/scored transactions |
| `GET` | `/feeds/gov` | List active I4C / NCRP regulatory tickets |
| `POST` | `/feeds/gov` | Ingest new regulatory blacklist ticket |
| `GET` | `/webhooks/escrow` | List all recorded outbound escrow webhook dispatch events |

## Latency Benchmark

Run the automated N=10,000 latency benchmark harness:

```powershell
.\.venv\Scripts\python benchmark.py --n 10000
```

**Measured Performance (`benchmark.py` on N=2,000 warm state):**
- **Throughput**: ~61 requests / second
- **Median (p50)**: `14.53 ms`
- **90th Percentile (p90)**: `21.24 ms`
- **99th Percentile (p99)**: **`46.86 ms`** (Exceeds SLA target `< 100.0 ms`)
