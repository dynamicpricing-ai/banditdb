#!/usr/bin/env python3
"""
UCI contextual-bandit benchmark for BanditDB.

Runs the classification-as-bandit protocol from Zhang, Zhou, Li & Gu,
"Neural Thompson Sampling" (ICLR 2021), against a live BanditDB server:

    each round  -> one data point, context sent to /predict
    reward 1    -> the chosen arm matches the true class, else 0
    regret_t    -> t - cumulative_reward   (the optimal policy always scores 1)

Unlike a logged-feedback replay, this evaluation is exact: the label is known
for every arm, so no counterfactual estimation or rejection sampling is needed
and no events are discarded.

Prerequisites:
    python benchmark/uci/convert.py
    cargo build --release --features neural
    ./target/release/banditdb

Usage:
    python benchmark/uci/evaluate.py --dataset mushroom
    python benchmark/uci/evaluate.py --dataset magic --algorithms linucb,thompson_sampling
    python benchmark/uci/evaluate.py --all --rounds 10000 --repeats 3
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

import requests

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "uci_bandit"
RESULTS = HERE / "results"

DATASETS = ["mushroom", "magic", "shuttle", "adult"]
DEFAULT_ALGOS = ["linucb", "thompson_sampling", "neural_lin_ucb", "neural_thompson_sampling"]
NEURAL = {"neural_lin_ucb", "neural_thompson_sampling"}


# --- data ---------------------------------------------------------------

def load(name: str):
    meta = json.loads((DATA / f"{name}.meta.json").read_text())
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]
    return meta, rows


# --- server ------------------------------------------------------------

class Server:
    def _post(self, path: str, payload: dict) -> dict:
        r = self.session.post(f"{self.url}{path}", json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            return {}

    def health(self) -> bool:
        try:
            return self.session.get(f"{self.url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def __init__(self, url: str, api_key: str | None, neural_cfg: dict | None = None):
        self.url = url.rstrip("/")
        self.neural_cfg = neural_cfg or {}
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-Api-Key"] = api_key
        self.session.headers["Content-Type"] = "application/json"

    def create_campaign(self, cid: str, arms: list[str], dim: int, alpha: float, algo: str):
        if algo in NEURAL:
            algorithm = {algo: {
                "context_dim":   dim,
                "embed_dim":     self.neural_cfg.get("embed_dim") or min(32, max(8, dim // 2)),
                "hidden_dim":    128,
                "hidden_layers": 2,
                "retrain_every": self.neural_cfg.get("retrain_every", 200),
                "retrain_steps": self.neural_cfg.get("retrain_steps", 100),
                "learning_rate": 1e-3,
                "lambda":        1.0,
            }}
        else:
            # Unit-variant enums serialise as their bare snake_case name.
            algorithm = algo

        self._post("/campaign", {
            "campaign_id": cid,
            "arms":        arms,
            # Neural campaigns size their arm matrices from embed_dim, so
            # feature_dim is ignored — matches the examples/ convention.
            "feature_dim": 0 if algo in NEURAL else dim,
            "alpha":       alpha,
            "algorithm":   algorithm,
        })

    def predict(self, cid: str, context: list[float]):
        r = self._post("/predict", {"campaign_id": cid, "context": context})
        return r["arm_id"], r["interaction_id"]

    def reward(self, interaction_id: str, value: float):
        self._post("/reward", {"interaction_id": interaction_id, "reward": value})

    def checkpoint(self):
        self._post("/checkpoint", {})

    def delete_campaign(self, cid: str):
        try:
            self.session.delete(f"{self.url}/campaign/{cid}", timeout=30)
        except requests.RequestException:
            pass


# --- benchmark ----------------------------------------------------------

def run_one(srv: Server, name: str, meta: dict, rows: list, algo: str,
            rounds: int, alpha: float, seed: int, checkpoint_every: int,
            log_every: int) -> dict:
    arms = [f"arm_{i}" for i in range(meta["n_arms"])]
    cid = f"uci_{name}_{algo}_{seed}_{uuid.uuid4().hex[:6]}"

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    order = order[:rounds] if rounds <= len(order) else \
        [order[i % len(order)] for i in range(rounds)]

    srv.create_campaign(cid, arms, meta["dim"], alpha, algo)

    cumulative = 0.0
    curve = []
    started = time.time()
    failures = 0

    try:
        for t, idx in enumerate(order, 1):
            row = rows[idx]
            try:
                arm, iid = srv.predict(cid, row["context"])
            except RuntimeError as e:
                failures += 1
                if failures <= 3:
                    print(f"    [warn] predict failed at t={t}: {e}", file=sys.stderr)
                if failures > 50:
                    raise
                continue

            reward = 1.0 if arm == f"arm_{row['label']}" else 0.0
            cumulative += reward
            srv.reward(iid, reward)

            if checkpoint_every and t % checkpoint_every == 0:
                srv.checkpoint()

            if t % log_every == 0:
                curve.append({"t": t, "regret": t - cumulative,
                              "avg_reward": cumulative / t})
                print(f"    t={t:>6,}  regret={t - cumulative:>8,.0f}  "
                      f"avg_reward={cumulative / t:.4f}", flush=True)
    finally:
        srv.delete_campaign(cid)

    n = len(order)
    return {
        "dataset": name, "algorithm": algo, "seed": seed, "rounds": n,
        "cumulative_reward": cumulative,
        "cumulative_regret": n - cumulative,
        "avg_reward": cumulative / n if n else 0.0,
        "accuracy": cumulative / n if n else 0.0,
        "elapsed_secs": round(time.time() - started, 1),
        "predict_failures": failures,
        "curve": curve,
    }


def random_baseline(meta: dict, rounds: int) -> float:
    """Expected regret of uniform-random arm choice."""
    return rounds * (1.0 - 1.0 / meta["n_arms"])


def majority_baseline(meta: dict, rounds: int) -> float:
    """Expected regret of always playing the most frequent class."""
    best = max(meta["class_balance"]) / sum(meta["class_balance"])
    return rounds * (1.0 - best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mushroom")
    ap.add_argument("--all", action="store_true", help="run every dataset")
    ap.add_argument("--algorithms", default=",".join(DEFAULT_ALGOS))
    ap.add_argument("--rounds", type=int, default=10000, help="matches Zhang et al. (T=10000)")
    ap.add_argument("--repeats", type=int, default=1, help="seeds per configuration")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="neural retrain runs at checkpoint; 0 disables")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--embed-dim", type=int, default=0, help="0 = auto (min(32, dim//2))")
    ap.add_argument("--retrain-every", type=int, default=200,
                    help="rewards accumulated before a retrain becomes due")
    ap.add_argument("--retrain-steps", type=int, default=100,
                    help="gradient steps per retrain")
    ap.add_argument("--url", default=os.environ.get("BANDITDB_URL", "http://localhost:8080"))
    ap.add_argument("--api-key", default=os.environ.get("BANDITDB_API_KEY"))
    args = ap.parse_args()

    srv = Server(args.url, args.api_key, {
        "embed_dim":     args.embed_dim or None,
        "retrain_every": args.retrain_every,
        "retrain_steps": args.retrain_steps,
    })
    if not srv.health():
        sys.exit(f"BanditDB not reachable at {args.url} — start it with "
                 f"./target/release/banditdb")

    datasets = DATASETS if args.all else [args.dataset]
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    results = []

    for name in datasets:
        if not (DATA / f"{name}.meta.json").exists():
            print(f"skipping {name} — run convert.py first", file=sys.stderr)
            continue
        meta, rows = load(name)
        rounds = min(args.rounds, len(rows)) if args.rounds > 0 else len(rows)
        print(f"\n=== {name}  (rows={meta['n_rows']:,}  dim={meta['dim']}  "
              f"arms={meta['n_arms']}  T={rounds:,}) ===")
        print(f"  baselines: random regret={random_baseline(meta, rounds):,.0f}  "
              f"majority-class regret={majority_baseline(meta, rounds):,.0f}")

        for algo in algos:
            for seed in range(args.repeats):
                print(f"\n  -- {algo} (seed {seed})")
                ckpt = args.checkpoint_every if algo in NEURAL else 0
                try:
                    res = run_one(srv, name, meta, rows, algo, rounds,
                                  args.alpha, seed, ckpt, args.log_every)
                except Exception as e:
                    print(f"    [error] {algo} failed: {e}", file=sys.stderr)
                    continue
                res["random_regret"] = random_baseline(meta, rounds)
                res["majority_regret"] = majority_baseline(meta, rounds)
                results.append(res)
                print(f"    => regret={res['cumulative_regret']:,.0f}  "
                      f"accuracy={res['accuracy']:.4f}  ({res['elapsed_secs']}s)")

    if not results:
        sys.exit("no results")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"uci_benchmark_{stamp}.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print(f"{'dataset':<10} {'algorithm':<26} {'regret':>10} {'accuracy':>10} {'vs random':>11}")
    print("-" * 78)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        grouped.setdefault((r["dataset"], r["algorithm"]), []).append(r)
    for (ds, algo), runs in grouped.items():
        regret = statistics.mean(r["cumulative_regret"] for r in runs)
        acc = statistics.mean(r["accuracy"] for r in runs)
        lift = 100.0 * (runs[0]["random_regret"] - regret) / runs[0]["random_regret"]
        print(f"{ds:<10} {algo:<26} {regret:>10,.0f} {acc:>10.4f} {lift:>10.1f}%")
    print("=" * 78)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
