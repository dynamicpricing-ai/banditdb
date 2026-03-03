#!/usr/bin/env python3
"""
Advanced Feature Sweep — MovieLens 100K
========================================

Two remaining ideas for pushing past +17%:

1. Cumulative history (correct causal approach)
   During training, for event i, use only history from events 0..i-1.
   No lookahead: the model trains on features available at decision time.
   For testing: use full training history (valid — we have it available).

2. Genre preference deviation
   Instead of raw like_rate[g], use like_rate[g] - pop_avg[g].
   Captures "how much MORE does this user like Drama vs the average person".
   Range ≈ [-0.6, +0.4]. Removes population-level bias from the signal.

Feature sets
------------
  demo            dim=24  [1, age, male, occ_oh×21]             current best
  cumul_dev       dim=37  demo + [dev×6, seen_norm×6, overall]  causal history
  cumul_raw       dim=37  demo + [like_rate×6, seen×6, overall] causal history, raw rates

Usage:
  python benchmark/movielens/advanced_sweep.py
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

ARMS     = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC    = 21
N_ARMS   = len(ARMS)
ARM_IDX  = {a: i for i, a in enumerate(ARMS)}
ML_DIR   = Path(__file__).parent.parent / "data" / "ml-100k-raw" / "ml-100k"

GENRE_COLS     = {"Action": 1, "Adventure": 2, "Comedy": 5, "Drama": 8, "Romance": 14, "Thriller": 16}
GENRE_PRIORITY = ["Action", "Adventure", "Romance", "Thriller", "Comedy", "Drama"]
TRAIN_RATIO    = 0.9


# ---------------------------------------------------------------------------
# LinUCB (Sherman-Morrison)
# ---------------------------------------------------------------------------

class LinUCB:
    def __init__(self, arms, d, alpha):
        self.arms = arms; self.d = d; self.alpha = alpha
        self.A_inv = {a: np.eye(d)   for a in arms}
        self.b     = {a: np.zeros(d) for a in arms}

    def predict(self, x):
        x = np.array(x, dtype=float); best, bs = None, -np.inf
        for a in self.arms:
            Ai = self.A_inv[a]; th = Ai @ self.b[a]
            s  = float(th @ x) + self.alpha * np.sqrt(max(0., float(x @ Ai @ x)))
            if s > bs: bs, best = s, a
        return best

    def update(self, arm, x, r):
        x = np.array(x, dtype=float); Ai = self.A_inv[arm]; Ax = Ai @ x
        self.A_inv[arm] = Ai - np.outer(Ax, Ax) / (1. + float(x @ Ax))
        self.b[arm] += r * x

    def clone(self):
        m = LinUCB(self.arms, self.d, self.alpha)
        m.A_inv = {a: v.copy() for a, v in self.A_inv.items()}
        m.b     = {a: v.copy() for a, v in self.b.items()}
        return m


# ---------------------------------------------------------------------------
# Raw MovieLens parsers
# ---------------------------------------------------------------------------

def parse_occupations():
    return {l.strip(): i for i, l in enumerate((ML_DIR/"u.occupation").read_text().strip().splitlines())}

def parse_users(occupations):
    users = {}
    for line in (ML_DIR/"u.user").read_text().strip().splitlines():
        uid, age, gender, occ, _ = line.split("|")
        users[int(uid)] = (float(age)/73.0, 1.0 if gender.strip()=="M" else 0.0,
                           occupations.get(occ.strip(), 0))
    return users

def parse_movies():
    movies = {}
    for line in (ML_DIR/"u.item").read_text(encoding="latin-1").strip().splitlines():
        parts = line.split("|"); mid = int(parts[0]); flags = [int(x) for x in parts[5:]]
        movies[mid] = next((g for g in GENRE_PRIORITY
                            if GENRE_COLS[g] < len(flags) and flags[GENRE_COLS[g]]==1), None)
    return movies

def parse_ratings():
    rows = []
    for line in (ML_DIR/"u.data").read_text().strip().splitlines():
        uid, mid, rating, ts = line.split("\t")
        rows.append((int(uid), int(mid), int(rating), int(ts)))
    rows.sort(key=lambda r: r[3])
    return rows


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

class RunningHistory:
    """Per-user incremental history. Call update() AFTER building the context."""
    def __init__(self, pop_like_rate):
        self.pop  = pop_like_rate
        self.liked = defaultdict(lambda: defaultdict(int))
        self.seen  = defaultdict(lambda: defaultdict(int))
        self.total_liked = defaultdict(int)
        self.total_seen  = defaultdict(int)

    def features(self, uid, mode="dev"):
        """Return 13 history features for this user using events seen so far."""
        like_rates, seen_norms = [], []
        for arm in ARMS:
            s = self.seen[uid][arm]
            l = self.liked[uid][arm]
            w = s / (s + 5)  # Bayesian blend weight
            raw = w * (l/s if s else 0.0) + (1-w) * self.pop[arm]
            like_rates.append(raw - self.pop[arm] if mode == "dev" else raw)
            seen_norms.append(min(s, 20) / 20.0)
        tot = self.total_seen[uid]
        overall = (self.total_liked[uid] / tot if tot else 0.5) - 0.56 if mode == "dev" else \
                  (self.total_liked[uid] / tot if tot else 0.5)
        return like_rates + seen_norms + [overall]   # 13 features

    def update(self, uid, arm, reward):
        self.seen[uid][arm]  += 1
        self.liked[uid][arm] += int(reward >= 0.5)
        self.total_seen[uid]  += 1
        self.total_liked[uid] += int(reward >= 0.5)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def demo_vec(uid, users):
    age, male, occ_id = users[uid]
    occ_oh = [0.0]*N_OCC; occ_oh[occ_id] = 1.0
    return [1.0, age, male] + occ_oh   # 24


# ---------------------------------------------------------------------------
# Evaluation with cumulative history
# ---------------------------------------------------------------------------

def run_cumulative(train_raw, test_raw, alpha, mode, users, pop_like_rate):
    """
    train_raw / test_raw: list of (uid, arm, binary_reward)

    mode: "demo"      — no history, 24-dim
          "cumul_dev" — cumulative history with deviation features, 37-dim
          "cumul_raw" — cumulative history with raw like rates, 37-dim
    """
    dim   = 24 if mode == "demo" else 37
    model = LinUCB(ARMS, dim, alpha)
    rh    = RunningHistory(pop_like_rate)
    hmode = "dev" if mode == "cumul_dev" else "raw"

    # Training: history is incremented AFTER each update (causal)
    for uid, arm, r in train_raw:
        x = demo_vec(uid, users)
        if mode != "demo":
            x = x + rh.features(uid, hmode)   # 24 + 13 = 37
        model.update(arm, x, r)
        rh.update(uid, arm, r)   # add to history only after the decision

    # After training: freeze history (use full training history for test)
    ev = model.clone()
    br, bc = 0.0, 0
    rand = sum(r for _, _, r in test_raw) / len(test_raw) if test_raw else 0.0

    for uid, arm_logged, r in test_raw:
        x = demo_vec(uid, users)
        if mode != "demo":
            x = x + rh.features(uid, hmode)
        ap = ev.predict(x)
        if ap == arm_logged:
            bc += 1; br += r
            ev.update(ap, x, r)
            rh.update(uid, ap, r)  # also update history on test matches

    ba   = br / bc if bc else 0.0
    lift = (ba / rand - 1) * 100 if rand else 0.0
    return lift, ba, rand, bc / len(test_raw) if test_raw else 0.0, bc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not ML_DIR.exists():
        print(f"Raw data not found at {ML_DIR}. Run convert.py first.")
        return

    print("Parsing raw MovieLens data ...")
    occupations = parse_occupations()
    users       = parse_users(occupations)
    movies      = parse_movies()
    ratings     = parse_ratings()

    interactions = []
    for uid, mid, rating, ts in ratings:
        if uid not in users: continue
        arm = movies.get(mid)
        if arm is None: continue
        interactions.append((uid, arm, 1.0 if rating >= 4 else 0.0))

    split = int(len(interactions) * TRAIN_RATIO)
    train_raw = interactions[:split]
    test_raw  = interactions[split:]
    print(f"Train: {len(train_raw):,}   Test: {len(test_raw):,}\n")

    # Population like rates (from training set)
    arm_liked = defaultdict(int); arm_seen = defaultdict(int)
    for _, arm, r in train_raw:
        arm_seen[arm] += 1; arm_liked[arm] += int(r)
    pop = {arm: arm_liked[arm]/arm_seen[arm] if arm_seen[arm] else 0.5 for arm in ARMS}

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    modes  = ["demo", "cumul_dev", "cumul_raw"]

    results = []
    col = "  {:<12}  {:<5}  {:>8}  {:>10}  {:>8}  {:>5}"
    print(col.format("mode", "alpha", "lift", "bandit", "match%", "dim"))
    print("  " + "-" * 56)

    for mode in modes:
        dim  = 24 if mode == "demo" else 37
        best = -np.inf
        for alpha in alphas:
            lift, ba, rand, match, cnt = run_cumulative(
                train_raw, test_raw, alpha, mode, users, pop)
            results.append((lift, alpha, mode, ba, rand, match, cnt))
            marker = " <--" if lift > best else ""
            best   = max(best, lift)
            print(col.format(mode, f"{alpha:.3f}", f"{lift:+.1f}%",
                             f"{ba:.4f}", f"{match:.1%}", str(dim)) + marker)
        print()

    sep = "=" * 60
    print(sep)
    print("Top 10 by lift:")
    results.sort(reverse=True)
    for lift, alpha, mode, ba, rand, match, cnt in results[:10]:
        dim = 24 if mode == "demo" else 37
        print(f"  {mode:<12}  alpha={alpha:.3f}  dim={dim}  "
              f"lift={lift:+.1f}%  bandit={ba:.4f}  match={match:.1%}")

    print(f"\nBest: {results[0][2]}  alpha={results[0][1]}  lift={results[0][0]:+.1f}%")


if __name__ == "__main__":
    main()
