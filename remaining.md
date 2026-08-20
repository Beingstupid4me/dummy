# Remaining vs `pro_report` / `phase2_roadmap.md`

Cross-check of `backend/` and `frontend/` against the paper (Next.js console, FastAPI, Redis) and the roadmap (ingestion, registry, three operational SLAs). Review date: 20 Aug 2026.

**Coverage:** 13 shipped · 6 partial · 8 missing (27 checklist rows).

| | |
|---|---|
| Enhancement chrono / future-core PR-AUC | 0.732 / 0.766 |
| Registry weights on disk | M1 only |
| Warm `/score` | ~50 ms (cold start is tens of seconds) |

Quote chrono scores, not the paper’s 0.8677 table. Locked Phase 1 is **0.738** chrono PR-AUC. Enhancement is **0.732** chrono / **0.766** future-core. The 0.8677 / Macro F1 0.9147 block in `pro_report.tex` is the shuffled / leaky ranking already rejected. Live P≈0.995 on `XXXX2203` is a seeded mule + I4C hit, not a model metric.

CSV is one row per account with L7/L14/L31 already rolled up. `/score` overlays this payment onto the stored L7 snapshot and writes Redis/memory. That is the stream-until-now input.

## Item-by-item vs spec

| Capability | Spec | Status |
|---|---|---|
| Core txn JSON + `/score` | Roadmap W1 / paper | Shipped |
| L7 profile + this-txn overlay | Roadmap + data shape | Shipped (memory default) |
| I4C/NCRP `/feeds/gov` | Roadmap W1 | Shipped (simulated) |
| Feature checklist → `/retrain` | Roadmap W5 | Partial — elapsed/leaky/SMOTE only |
| 14 row moments | Roadmap W3 | Shipped |
| Cyclical F3888 only | Roadmap W3 | Shipped |
| Fold-safe ordinal + Laplace TE | Roadmap W3 | Shipped |
| Isolation Forest (normals) | Roadmap W3 | Shipped |
| Four registry slots <5 MB | Paper | Partial — M1 filled; M2/M3 empty |
| M1 anchor-free GBDT | Paper | Shipped (`enhance_m1.joblib`) |
| M2 elapsed_days weights | Paper / rubric M1↔M3 | Missing until `bootstrap --all` |
| M3 leaky demo weights | Paper / rubric | Missing until `bootstrap --all` |
| M4 on-demand retrain | Paper | Code ready; 15s SLA unverified |
| PCHIP \(f'(x)>0\) abort | Week 6 | Shipped |
| 90-day label-delay buffer | Week 6 | Shipped |
| Retrain worker isolation | Paper ProcessPool | Partial — ThreadPoolExecutor |
| Docker Compose (API+Redis+UI) | Paper §prototype | Missing |
| TTL 24h / 7d ego edges | Paper | Shipped |
| Next.js SecOps (3 routes) | Paper (says 14; app is 16) | Shipped |
| `react-virtuoso` live table | Paper | Missing — plain table |
| WebGL force-graph | Paper | Missing — 2D canvas rings |
| SSE live `/stream` | Paper | Shipped + mock fallback |
| Retrain progress SSE | Paper | Partial — HTTP poll |
| N=10k latency harness | Paper table p99 49.8 ms | Missing — number is not measured here |
| TMS ingest on `/score` | Paper diagram | Missing — `tmsFlags` always `[]` |
| Escrow 15-min hold | Paper | Partial — KV key, no outbound POST |
| `GET /feeds/gov` in UI | Roadmap dashboard | Missing — analysis page uses seed tickets |

## Three rubric SLAs (`phase2_roadmap.md` §4)

| SLA | State |
|---|---|
| API latency < 100 ms | Warm path ~50 ms after matrix warmup |
| Hot-swap 0 ms | Coded; M3 has no joblib until `python -m app.bootstrap --all` |
| Retrain < 15 s | PCHIP gate exists; wall time not measured at 500 trees |

## Finish list

Code for `/score`, `/retrain`, `/registry`, SSE `/stream`, PCHIP abort, 90-day delay, and the three console pages is in. The items below are what still separates this repo from the paper’s “Docker + Redis AOF + four models + measured 49.8 ms p99” claim.

- [ ] `docker-compose`: FastAPI, Redis with AOF `appendfsync everysec`, Next.js
- [ ] `python -m app.bootstrap --all` so M1↔M3 hot-swap has weights
- [ ] N=10,000 `/score` harness; paste real p50/p99 into `pro_report.tex`
- [ ] Send `features_on` / `features_off` from the registry board into `/retrain`
- [ ] SSE (or keep poll) for retrain log — paper asks for an event stream
- [ ] `ProcessPoolExecutor` for `/retrain` as written in the paper
- [ ] POST outbound escrow webhook, not only a Redis key
- [ ] TMS flags on `/score`; bind analysis I4C list to `GET /feeds/gov`
- [ ] Optional: `react-virtuoso` + `react-force-graph-2d`, or keep canvas as demo-grade
- [ ] Replace paper table PR-AUC 0.8677 with chrono 0.738 / 0.732 (shuffled/leak trap)
