"""N=10,000 latency benchmark harness for SentinelFlow /score engine.

Measures cold-start vs warm-state latency distributions across accounts,
including Redis lookup, feature reconstruction, GBDT blend, PCHIP calibration,
and Taylor-scaled TreeSHAP reason code extraction.

    python benchmark.py --n 10000
    python benchmark.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd

from app.schemas import ScoreRequest
from app.services.engine import context, ensure_m1_loaded, score
from app.services.registry import get_registry


def run_benchmark(n: int = 10000, warmup: int = 100) -> dict:
    ensure_m1_loaded()
    ctx = context()
    total_accounts = len(ctx.y)
    print(f"Loaded dataset context with {total_accounts} accounts.")
    active_slot = get_registry().get_active()
    print(f"Active Model: {get_registry().active_id} ({active_slot.card.name})")

    channels = ["UPI", "ATM", "IMPS", "NEFT", "NETBANK"]
    sample_aliases = ["XXXX2203", "XXXX2411", "XXXX2688", "XXXX2904"]

    print(f"\nWarming up scoring engine ({warmup} iterations)...")
    for i in range(warmup):
        acct = sample_aliases[i % len(sample_aliases)] if i < 20 else str(i % total_accounts)
        req = ScoreRequest(
            account=acct,
            channel=channels[i % len(channels)],
            amount=float(1000 + (i * 73) % 200000),
        )
        score(req)

    print(f"Executing N={n:,} /score benchmark iterations...")
    latencies: list[float] = []
    t_redis_list: list[float] = []
    t_rec_list: list[float] = []
    t_gbdt_list: list[float] = []
    t_pchip_list: list[float] = []
    t_shap_list: list[float] = []

    t_start_total = time.perf_counter()
    for i in range(n):
        # Mix of seeded alias accounts and random account indexes
        if i % 10 == 0:
            acct = sample_aliases[(i // 10) % len(sample_aliases)]
        else:
            acct = str((i * 17) % total_accounts)

        amt = float(random.randint(500, 250000))
        ch = channels[i % len(channels)]
        req = ScoreRequest(account=acct, channel=ch, amount=amt)

        resp = score(req)
        latencies.append(resp.latencyMs)
        t_redis_list.append(resp.latency.redis)
        t_rec_list.append(resp.latency.reconstruct)
        t_gbdt_list.append(resp.latency.gbdt)
        t_pchip_list.append(resp.latency.pchip)
        t_shap_list.append(resp.latency.shap)

        if (i + 1) % (n // 5) == 0:
            pct = ((i + 1) / n) * 100
            cur_p50 = np.percentile(latencies, 50)
            cur_p99 = np.percentile(latencies, 99)
            print(f"  {pct:3.0f}% complete ({i + 1:,}/{n:,}) — current p50: {cur_p50:.2f}ms, p99: {cur_p99:.2f}ms")

    elapsed_wall = time.perf_counter() - t_start_total
    lat_arr = np.array(latencies)

    results = {
        "n_samples": n,
        "wall_time_seconds": round(elapsed_wall, 2),
        "throughput_rps": round(n / elapsed_wall, 1),
        "latency_ms": {
            "mean": round(float(np.mean(lat_arr)), 2),
            "std": round(float(np.std(lat_arr)), 2),
            "min": round(float(np.min(lat_arr)), 2),
            "p50": round(float(np.percentile(lat_arr, 50)), 2),
            "p90": round(float(np.percentile(lat_arr, 90)), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
            "p99": round(float(np.percentile(lat_arr, 99)), 2),
            "p99_9": round(float(np.percentile(lat_arr, 99.9)), 2),
            "max": round(float(np.max(lat_arr)), 2),
        },
        "component_p50_ms": {
            "redis_lookup": round(float(np.percentile(t_redis_list, 50)), 3),
            "reconstruction": round(float(np.percentile(t_rec_list, 50)), 3),
            "gbdt_blend": round(float(np.percentile(t_gbdt_list, 50)), 3),
            "pchip_calib": round(float(np.percentile(t_pchip_list, 50)), 3),
            "shap_explanation": round(float(np.percentile(t_shap_list, 50)), 3),
        },
        "sla_passed": bool(np.percentile(lat_arr, 99) < 100.0),
    }

    out_json = Path(__file__).parent / "latency_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("           SENTINELFLOW /SCORE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Iterations (N)      : {n:,}")
    print(f"Wall Clock Time     : {elapsed_wall:.2f} s")
    print(f"Throughput          : {results['throughput_rps']} req/sec")
    print("-" * 60)
    print(f"Latency p50 (median): {results['latency_ms']['p50']:6.2f} ms")
    print(f"Latency p90         : {results['latency_ms']['p90']:6.2f} ms")
    print(f"Latency p95         : {results['latency_ms']['p95']:6.2f} ms")
    print(f"Latency p99 (SLA)   : {results['latency_ms']['p99']:6.2f} ms  (SLA < 100 ms: {'PASSED' if results['sla_passed'] else 'FAILED'})")
    print(f"Latency p99.9       : {results['latency_ms']['p99_9']:6.2f} ms")
    print("-" * 60)
    print("Component Breakdown (p50):")
    for k, v in results["component_p50_ms"].items():
        print(f"  • {k:20s}: {v:6.3f} ms")
    print("=" * 60)
    print(f"Wrote benchmark results to {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelFlow /score latency benchmark")
    parser.add_argument("--n", type=int, default=10000, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    args = parser.parse_args()
    run_benchmark(n=args.n, warmup=args.warmup)
