#!/usr/bin/env python3
"""
User History Feature Sweep — MovieLens 100K
============================================

Demographic features (age, gender, occupation) explain only a small fraction
of genre preference variance. Per-user genre history is much more predictive:
a user who has liked 9 of their last 10 Drama films is almost certainly going
to like the next one — regardless of their age.

This script re-reads the raw MovieLens files (the WAL discards user IDs)
and adds per-user genre history as context features.

History features (computed from training set, used at all prediction times)
---------------------------------------------------------------------------
  genre_like_rate[6]   fraction of liked (>= 4) ratings per genre
                       → default = population avg for that genre (better prior than 0.5)
  genre_seen_norm[6]   min(count, 20) / 20  — confidence weight per genre
  overall_like_rate    user's binary like rate across all genres

Feature sets compared
---------------------
  demo          dim=24  [1, age, male, occ_oh×21]                     current best
  history       dim=14  [1, like_rate×6, seen_norm×6, overall]        history only
  demo+history  dim=37  demo (24) + history without bias (13)          combined

Usage:
  python benchmark/movielens/history_sweep.py
"""

import numpy as np
from collections import defaultdict
from pathlib import Path

ARMS     = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC    = 21
N_ARMS   = len(ARMS)
ARM_IDX  = {a: i for i, a in enumerate(ARMS)}

ML_DIR   = Path(__file__).parent.parent / "data" / "ml-100k-raw" / "ml-100k"

GENRE_COLS = {
    "Action": 1, "Adventure": 2, "Comedy": 5,
    "Drama": 8,  "Romance": 14, "Thriller": 16,
}
GENRE_PRIORITY = ["Action", "Adventure", "Romance", "Thriller", "Comedy", "Drama"]

TRAIN_RATIO = 0.9


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
    return {
        line.strip(): i
        for i, line in enumerate((ML_DIR / "u.occupation").read_text().strip().splitlines())
    }

def parse_users(occupations):
    """Returns {user_id: (age_norm, is_male, occ_id)}."""
    users = {}
    for line in (ML_DIR / "u.user").read_text().strip().splitlines():
        uid, age, gender, occ, _ = line.split("|")
        occ_id = occupations.get(occ.strip(), 0)
        users[int(uid)] = (float(age) / 73.0, 1.0 if gender.strip() == "M" else 0.0, occ_id)
    return users

def parse_movies():
    """Returns {movie_id: arm_id | None}."""
    movies = {}
    for line in (ML_DIR / "u.item").read_text(encoding="latin-1").strip().splitlines():
        parts = line.split("|"); mid = int(parts[0])
        flags = [int(x) for x in parts[5:]]
        arm   = next(
            (g for g in GENRE_PRIORITY if GENRE_COLS[g] < len(flags) and flags[GENRE_COLS[g]] == 1),
            None,
        )
        movies[mid] = arm
    return movies

def parse_ratings():
    """Returns list of (user_id, movie_id, rating, timestamp) sorted by timestamp."""
    rows = []
    for line in (ML_DIR / "u.data").read_text().strip().splitlines():
        uid, mid, rating, ts = line.split("\t")
        rows.append((int(uid), int(mid), int(rating), int(ts)))
    rows.sort(key=lambda r: r[3])
    return rows


# ---------------------------------------------------------------------------
# User history feature computation
# ---------------------------------------------------------------------------

def compute_user_history(interactions, pop_like_rate):
    """
    interactions: list of (uid, arm, binary_reward)
    Returns {uid: {"like": [6 floats], "seen": [6 floats], "overall": float}}
    """
    liked = defaultdict(lambda: defaultdict(int))   # uid -> arm -> liked_count
    seen  = defaultdict(lambda: defaultdict(int))   # uid -> arm -> total_count
    for uid, arm, reward in interactions:
        seen[uid][arm]  += 1
        liked[uid][arm] += int(reward)

    history = {}
    for uid in seen:
        all_seen  = sum(seen[uid].values())
        all_liked = sum(liked[uid].values())
        overall   = all_liked / all_seen if all_seen else 0.5

        like_rates, seen_norms = [], []
        for arm in ARMS:
            s = seen[uid].get(arm, 0)
            l = liked[uid].get(arm, 0)
            # Bayesian-ish: blend user evidence with population prior
            weight    = s / (s + 5)                          # trust grows with evidence
            like_rate = weight * (l / s if s else 0.0) + (1 - weight) * pop_like_rate[arm]
            like_rates.append(like_rate)
            seen_norms.append(min(s, 20) / 20.0)             # capped exposure weight

        history[uid] = {"like": like_rates, "seen": seen_norms, "overall": overall}
    return history


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def demo_ctx(age, male, occ_id):
    occ_oh = [0.0] * N_OCC; occ_oh[occ_id] = 1.0
    return [1.0, age, male] + occ_oh                          # dim=24

def history_ctx(hist):
    return [1.0] + hist["like"] + hist["seen"] + [hist["overall"]]  # dim=14

def combined_ctx(age, male, occ_id, hist):
    occ_oh = [0.0] * N_OCC; occ_oh[occ_id] = 1.0
    # No second bias; history features appended to demo
    return [1.0, age, male] + occ_oh + hist["like"] + hist["seen"] + [hist["overall"]]  # dim=37

EMPTY_HIST = {"like": [0.5] * N_ARMS, "seen": [0.0] * N_ARMS, "overall": 0.5}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run(train, test, alpha, dim, ctx_fn):
    model = LinUCB(ARMS, dim, alpha)
    for arm, x, r in train:
        model.update(arm, x, r)

    ev = model.clone()
    br, bc = 0.0, 0
    rand = sum(r for _, _, r in test) / len(test) if test else 0.0

    for arm_logged, x, r in test:
        ap = ev.predict(x)
        if ap == arm_logged:
            bc += 1; br += r
            ev.update(ap, x, r)

    ba   = br / bc if bc else 0.0
    lift = (ba / rand - 1) * 100 if rand else 0.0
    return lift, ba, rand, bc / len(test) if test else 0.0, bc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not ML_DIR.exists():
        print(f"Raw MovieLens data not found at {ML_DIR}")
        print("Run benchmark/movielens/convert.py first.")
        return

    print("Parsing raw MovieLens data ...")
    occupations = parse_occupations()
    users       = parse_users(occupations)
    movies      = parse_movies()
    ratings     = parse_ratings()

    # Build interaction list (uid, arm, binary_reward, ts)
    interactions = []
    for uid, mid, rating, ts in ratings:
        if uid not in users: continue
        arm = movies.get(mid)
        if arm is None: continue
        interactions.append((uid, arm, 1.0 if rating >= 4 else 0.0, ts))

    split = int(len(interactions) * TRAIN_RATIO)
    train_raw = interactions[:split]
    test_raw  = interactions[split:]
    print(f"Total: {len(interactions):,}   Train: {len(train_raw):,}   Test: {len(test_raw):,}")

    # Population like rates per arm (from training set)
    arm_liked = defaultdict(int); arm_seen = defaultdict(int)
    for _, arm, r, _ in train_raw:
        arm_seen[arm] += 1; arm_liked[arm] += int(r)
    pop_like_rate = {arm: arm_liked[arm] / arm_seen[arm] if arm_seen[arm] else 0.5 for arm in ARMS}
    print("\nPopulation like rates (training):")
    for arm in ARMS:
        print(f"  {arm:<12} {pop_like_rate[arm]:.3f}")

    # Compute user history from training set
    print("\nComputing user history features ...")
    history = compute_user_history(
        [(uid, arm, r) for uid, arm, r, _ in train_raw],
        pop_like_rate,
    )
    print(f"  Users with history: {len(history):,}")

    # Build context vectors for each feature set
    def make_dataset(raw, feature_set):
        rows = []
        for uid, arm, r, _ in raw:
            age, male, occ_id = users[uid]
            hist = history.get(uid, EMPTY_HIST)
            if feature_set == "demo":
                x = demo_ctx(age, male, occ_id)
            elif feature_set == "history":
                x = history_ctx(hist)
            else:  # combined
                x = combined_ctx(age, male, occ_id, hist)
            rows.append((arm, x, r))
        return rows

    dims = {"demo": 24, "history": 14, "demo+history": 37}
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]

    results = []
    col = "  {:<14}  {:<5}  {:>8}  {:>10}  {:>8}  {:>5}"
    print()
    print(col.format("feature_set", "alpha", "lift", "bandit", "match%", "dim"))
    print("  " + "-" * 58)

    for fset in ["demo", "history", "demo+history"]:
        train_data = make_dataset(train_raw, fset)
        test_data  = make_dataset(test_raw,  fset)
        dim        = dims[fset]
        best       = -np.inf

        for alpha in alphas:
            lift, ba, rand, match, cnt = run(train_data, test_data, alpha, dim, None)
            results.append((lift, alpha, fset, ba, rand, match, cnt))
            marker = " <--" if lift > best else ""
            best   = max(best, lift)
            print(col.format(fset, f"{alpha:.3f}", f"{lift:+.1f}%",
                             f"{ba:.4f}", f"{match:.1%}", str(dim)) + marker)
        print()

    sep = "=" * 62
    print(sep)
    print("Top 10 by lift:")
    results.sort(reverse=True)
    for lift, alpha, fset, ba, rand, match, cnt in results[:10]:
        print(f"  {fset:<14}  alpha={alpha:.3f}  dim={dims[fset]:<3}  "
              f"lift={lift:+.1f}%  bandit={ba:.4f}  match={match:.1%}")

    best = results[0]
    print(f"\nBest: {best[2]}  alpha={best[1]}  dim={dims[best[2]]}  lift={best[0]:+.1f}%")

    # Show what the model learned about genre preferences
    print("\nSample: avg genre_like_rate for Drama-heavy users vs Comedy-heavy users")
    drama_users  = [uid for uid, h in history.items() if h["like"][ARM_IDX["Drama"]] > 0.75]
    comedy_users = [uid for uid, h in history.items() if h["like"][ARM_IDX["Comedy"]] > 0.75]
    print(f"  Drama-heavy users ({len(drama_users):,}): "
          f"drama_like={np.mean([history[u]['like'][ARM_IDX['Drama']] for u in drama_users]):.3f}  "
          f"comedy_like={np.mean([history[u]['like'][ARM_IDX['Comedy']] for u in drama_users]):.3f}")
    print(f"  Comedy-heavy users ({len(comedy_users):,}): "
          f"drama_like={np.mean([history[u]['like'][ARM_IDX['Drama']] for u in comedy_users]):.3f}  "
          f"comedy_like={np.mean([history[u]['like'][ARM_IDX['Comedy']] for u in comedy_users]):.3f}")


if __name__ == "__main__":
    main()
