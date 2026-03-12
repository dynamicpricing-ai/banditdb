#!/usr/bin/env python3
"""
BanditDB Throughput Benchmark
==============================
Measures sustained predict throughput (requests/sec) and latency percentiles
against a live BanditDB instance.

Usage:
    # Start BanditDB first (no auth, dev mode):
    #   docker run -d -p 8080:8080 simeonlukov/banditdb:latest

    python3 benchmark/throughput/bench.py
    python3 benchmark/throughput/bench.py --url http://localhost:8080 --duration 10

The benchmark creates a throwaway campaign, runs a warmup pass, then sweeps
concurrency levels [1, 4, 16, 64, 128] and reports p50/p99 latency and
throughput at each level. The best result is printed at the end.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time

import aiohttp

BASE_CAMPAIGN = "bench_throughput"

ARMS = ["arm_a", "arm_b", "arm_c", "arm_d"]
FEATURE_DIM = 5
CONTEXT = [0.5, 0.35, 0.70, 0.20, 0.80]

CONCURRENCY_LEVELS = [1, 4, 16, 64, 128]
WARMUP_REQUESTS = 200
DURATION_SECS = 10


async def setup(session: aiohttp.ClientSession, url: str, api_key: str | None) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key

    # Delete if exists (ignore 404)
    await session.delete(f"{url}/campaign/{BASE_CAMPAIGN}", headers=headers)

    payload = {
        "campaign_id": BASE_CAMPAIGN,
        "arms": ARMS,
        "feature_dim": FEATURE_DIM,
    }
    async with session.post(
        f"{url}/campaign", headers=headers, json=payload
    ) as resp:
        if resp.status not in (200, 201):
            body = await resp.text()
            print(f"[error] Failed to create campaign: {resp.status} {body}", file=sys.stderr)
            sys.exit(1)
    print(f"  Campaign '{BASE_CAMPAIGN}' ready.")


async def predict_once(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
) -> float:
    """Fire one predict request; return latency in milliseconds."""
    t0 = time.perf_counter()
    async with session.post(f"{url}/predict", headers=headers, json=payload) as resp:
        await resp.read()
    return (time.perf_counter() - t0) * 1000.0


async def run_concurrent(
    url: str,
    headers: dict,
    payload: dict,
    concurrency: int,
    duration: float,
) -> list[float]:
    """
    Run predict requests at the given concurrency level for `duration` seconds.
    Returns a list of latencies in milliseconds.
    """
    latencies: list[float] = []
    stop_at = time.perf_counter() + duration

    async def worker():
        async with aiohttp.ClientSession() as session:
            while time.perf_counter() < stop_at:
                ms = await predict_once(session, url, headers, payload)
                latencies.append(ms)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return latencies


def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (data_sorted[hi] - data_sorted[lo]) * (k - lo)


def fmt_ms(ms: float) -> str:
    if ms < 1.0:
        return f"{ms * 1000:.0f}µs"
    return f"{ms:.1f}ms"


async def main():
    parser = argparse.ArgumentParser(description="BanditDB throughput benchmark")
    parser.add_argument("--url", default="http://localhost:8080", help="BanditDB base URL")
    parser.add_argument("--api-key", default=None, help="API key (if auth is enabled)")
    parser.add_argument("--duration", type=int, default=DURATION_SECS, help="Seconds per concurrency level")
    args = parser.parse_args()

    url: str = args.url.rstrip("/")
    api_key: str | None = args.api_key
    duration: int = args.duration

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Api-Key"] = api_key

    predict_payload = {"campaign_id": BASE_CAMPAIGN, "context": CONTEXT}

    print(f"\nBanditDB Throughput Benchmark")
    print(f"  Target : {url}")
    print(f"  Arms   : {len(ARMS)}   Feature dim: {FEATURE_DIM}")
    print(f"  Duration per level: {duration}s\n")

    # Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{url}/health") as resp:
                if resp.status != 200:
                    print(f"[error] Health check failed: {resp.status}", file=sys.stderr)
                    sys.exit(1)
        except aiohttp.ClientConnectorError:
            print(
                f"[error] Cannot connect to {url}. Is BanditDB running?\n"
                "  docker run -d -p 8080:8080 simeonlukov/banditdb:latest",
                file=sys.stderr,
            )
            sys.exit(1)

        print("  Health check passed.")
        await setup(session, url, api_key)

    # Warmup
    print(f"\n  Warmup ({WARMUP_REQUESTS} requests at concurrency=16)...")
    await run_concurrent(url, headers, predict_payload, concurrency=16, duration=max(2, duration // 3))
    print("  Warmup done.\n")

    # Header
    col = 12
    print(
        f"  {'Concurrency':>{col}}  {'Requests':>{col}}  {'RPS':>{col}}  "
        f"{'p50':>{col}}  {'p99':>{col}}"
    )
    print("  " + "-" * (col * 5 + 8))

    best_rps = 0.0
    best_concurrency = 1

    for concurrency in CONCURRENCY_LEVELS:
        latencies = await run_concurrent(
            url, headers, predict_payload,
            concurrency=concurrency,
            duration=duration,
        )
        if not latencies:
            continue

        rps = len(latencies) / duration
        p50 = percentile(latencies, 50)
        p99 = percentile(latencies, 99)

        if rps > best_rps:
            best_rps = rps
            best_concurrency = concurrency

        print(
            f"  {concurrency:>{col}}  {len(latencies):>{col},}  {rps:>{col},.0f}  "
            f"  {fmt_ms(p50):>{col}}  {fmt_ms(p99):>{col}}"
        )

    print()
    print(f"  Peak throughput: {best_rps:,.0f} predictions/sec  (concurrency={best_concurrency})")
    print()


if __name__ == "__main__":
    asyncio.run(main())
