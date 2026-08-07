#!/usr/bin/env python3
"""
Measure the scale limits published in docs/PRODUCTION_STAGE1.md §1.

The SLA table there had placeholders. A ceiling that has not been measured cannot
be documented, so this produces the four numbers it needs:

  1. predict throughput / p99   — the read path, which no longer touches the disk
  2. reward throughput / p99    — the durable path; each caller now awaits the fsync
                                  covering its record, so this is the number that
                                  changed most and the one that bounds write rate
  3. RPO vs fsync interval      — the latency/durability trade, measured rather than
                                  assumed
  4. RTO: recovery time vs WAL  — how long a restart takes, which sets the RTO claim
  5. memory per pending interaction — sizes BANDITDB_MAX_PENDING_INTERACTIONS

Usage:
    cargo build --release --features neural
    python3 benchmark/scale/limits.py                  # full sweep
    python3 benchmark/scale/limits.py --quick          # short run for a smoke check
"""

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp

REPO = Path(__file__).resolve().parents[2]
BIN = os.environ.get("BANDITDB_BIN", str(REPO / "target/release/banditdb"))
KEY = "scale-bench"


class Server:
    """A BanditDB process on its own data directory."""

    def __init__(self, port, env=None, data_dir=None):
        self.port = port
        self.dir = data_dir or tempfile.mkdtemp(prefix="banditdb_scale_")
        self.env = {**os.environ, "DATA_DIR": self.dir, "PORT": str(port),
                    "BANDITDB_API_KEY": KEY, "BANDITDB_RATE_LIMIT_PER_SEC": "1000000",
                    **(env or {})}
        self.proc = None

    def start(self, timeout=60):
        self.proc = subprocess.Popen([BIN], env=self.env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return True
            except Exception:
                if self.proc.poll() is not None:
                    return False
                time.sleep(0.2)
        return False

    def stop(self, keep_dir=False):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
        if not keep_dir:
            shutil.rmtree(self.dir, ignore_errors=True)

    def rss_mb(self):
        try:
            import psutil
            return psutil.Process(self.proc.pid).memory_info().rss / 1024 / 1024
        except Exception:
            return float("nan")


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p / 100))] * 1000  # ms


async def _create(session, port, cid, dim=8):
    await session.post(f"http://127.0.0.1:{port}/campaign",
        json={"campaign_id": cid, "arms": ["A", "B", "C"], "feature_dim": dim, "alpha": 1.0},
        headers={"X-Api-Key": KEY})


async def measure_path(port, cid, concurrency, duration, do_reward, dim=8):
    """Drive predict (and optionally reward) and return (ops/sec, latencies)."""
    ctx = [0.1 * (i % 9) for i in range(dim)]
    latencies = []
    stop_at = time.time() + duration
    count = 0

    async def worker(session):
        nonlocal count
        while time.time() < stop_at:
            t0 = time.perf_counter()
            async with session.post(f"http://127.0.0.1:{port}/predict",
                                    json={"campaign_id": cid, "context": ctx},
                                    headers={"X-Api-Key": KEY}) as r:
                if r.status != 200:
                    continue
                body = await r.json()
            if do_reward:
                # Timed separately below; here we time the reward leg only.
                t0 = time.perf_counter()
                async with session.post(f"http://127.0.0.1:{port}/reward",
                                        json={"interaction_id": body["interaction_id"], "reward": 1.0},
                                        headers={"X-Api-Key": KEY}) as r:
                    await r.read()
                    if r.status != 200:
                        continue
            latencies.append(time.perf_counter() - t0)
            count += 1

    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*[worker(session) for _ in range(concurrency)])
    elapsed = duration
    return count / elapsed, latencies


async def sweep(title, port, cid, levels, duration, do_reward):
    print(f"\n### {title}")
    print(f"{'concurrency':>12}{'ops/sec':>12}{'p50 ms':>10}{'p99 ms':>10}")
    best = (0, 0)
    for c in levels:
        rate, lat = await measure_path(port, cid, c, duration, do_reward)
        print(f"{c:>12}{rate:>12,.0f}{pct(lat,50):>10.2f}{pct(lat,99):>10.2f}")
        if rate > best[0]:
            best = (rate, c)
    print(f"  peak: {best[0]:,.0f} ops/sec at concurrency {best[1]}")
    return best


async def fsync_tradeoff(duration):
    """Reward latency and throughput against the configured commit window."""
    print("\n### RPO vs reward latency (BANDITDB_FSYNC_INTERVAL_MS)")
    print(f"{'interval ms':>12}{'ops/sec':>12}{'p50 ms':>10}{'p99 ms':>10}   worst-case RPO")
    for interval in ["0", "50", "200", "1000"]:
        srv = Server(18620, {"BANDITDB_FSYNC_INTERVAL_MS": interval})
        if not srv.start():
            print(f"{interval:>12}   (server failed to start)")
            srv.stop(); continue
        async with aiohttp.ClientSession() as s:
            await _create(s, srv.port, "fs")
        rate, lat = await measure_path(srv.port, "fs", 16, duration, do_reward=True)
        rpo = "~0 (idle sync)" if interval == "0" else f"≤{interval} ms"
        print(f"{interval:>12}{rate:>12,.0f}{pct(lat,50):>10.2f}{pct(lat,99):>10.2f}   {rpo}")
        srv.stop()


async def memory_envelope():
    """Bytes of RSS per pending (predicted, unrewarded) interaction."""
    print("\n### Memory per pending interaction")
    dim = 64
    srv = Server(18621, {"BANDITDB_MAX_PENDING_INTERACTIONS": "500000"})
    if not srv.start():
        print("  (server failed to start)"); srv.stop(); return
    async with aiohttp.ClientSession() as s:
        await _create(s, srv.port, "mem", dim=dim)
        time.sleep(1)
        base = srv.rss_mb()
        ctx = [0.01 * i for i in range(dim)]
        n = 50_000
        # Predict without rewarding so nothing is invalidated.
        sem = asyncio.Semaphore(64)
        async def one():
            async with sem:
                async with s.post(f"http://127.0.0.1:{srv.port}/predict",
                                  json={"campaign_id": "mem", "context": ctx},
                                  headers={"X-Api-Key": KEY}) as r:
                    await r.read()
        await asyncio.gather(*[one() for _ in range(n)])
        time.sleep(2)
        grown = srv.rss_mb()
    per = (grown - base) * 1024 * 1024 / n
    print(f"  context_dim={dim}, pending={n:,}")
    print(f"  RSS {base:,.0f} MB -> {grown:,.0f} MB   ≈ {per:,.0f} bytes/interaction")
    print(f"  => 100k default cap ≈ {per * 100_000 / 1024 / 1024:,.0f} MB")
    srv.stop()


async def recovery_time():
    """Restart duration against WAL size — the input to the RTO claim."""
    print("\n### RTO: recovery time vs WAL size")
    print(f"{'events':>10}{'WAL MB':>10}{'recovery s':>12}")
    for n in [5_000, 25_000, 100_000]:
        srv = Server(18622)
        if not srv.start():
            srv.stop(); continue
        async with aiohttp.ClientSession() as s:
            await _create(s, srv.port, "rec")
            ctx = [0.1] * 8
            sem = asyncio.Semaphore(64)
            async def one():
                async with sem:
                    async with s.post(f"http://127.0.0.1:{srv.port}/predict",
                                      json={"campaign_id": "rec", "context": ctx},
                                      headers={"X-Api-Key": KEY}) as r:
                        b = await r.json()
                    async with s.post(f"http://127.0.0.1:{srv.port}/reward",
                                      json={"interaction_id": b["interaction_id"], "reward": 1.0},
                                      headers={"X-Api-Key": KEY}) as r:
                        await r.read()
            await asyncio.gather(*[one() for _ in range(n)])
        wal = Path(srv.dir) / "bandit_wal.jsonl"
        wal_mb = wal.stat().st_size / 1024 / 1024 if wal.exists() else 0
        data_dir = srv.dir
        srv.stop(keep_dir=True)

        restarted = Server(18623, data_dir=data_dir)
        t0 = time.time()
        ok = restarted.start(timeout=180)
        elapsed = time.time() - t0
        print(f"{n:>10,}{wal_mb:>10.1f}{elapsed if ok else float('nan'):>12.1f}")
        restarted.stop()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short durations for a smoke check")
    args = ap.parse_args()

    if not Path(BIN).exists():
        sys.exit(f"binary not found: {BIN} (cargo build --release --features neural)")

    dur = 3 if args.quick else 8
    print("=" * 62)
    print("BanditDB scale limits")
    print(f"binary: {BIN}")
    print(f"host:   {os.cpu_count()} cores")
    print("=" * 62)

    levels = [1, 8, 32] if args.quick else [1, 8, 32, 64, 128]

    srv = Server(18610)
    if not srv.start():
        sys.exit("server failed to start")
    async with aiohttp.ClientSession() as s:
        await _create(s, srv.port, "bench")
    await sweep("Predict (read path — no disk write)", srv.port, "bench", levels, dur, False)
    await sweep("Reward (durable — awaits fsync)", srv.port, "bench", levels, dur, True)
    srv.stop()

    await fsync_tradeoff(dur)
    await memory_envelope()
    if not args.quick:
        await recovery_time()

    print("\nDone. Fold these into docs/PRODUCTION_STAGE1.md §1.")


if __name__ == "__main__":
    asyncio.run(main())
