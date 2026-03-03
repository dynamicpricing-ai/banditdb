#!/usr/bin/env python3
"""
Cross-Feature Sweep — MovieLens 100K
======================================

LinUCB is a linear model: it cannot learn that "young males prefer Action
differently to young females" unless we manually create an age × is_male
feature. This script evaluates progressively richer cross-feature sets to
find the configuration that maximises lift over random.

Feature sets tested
-------------------
  baseline       dim=24   [1, age, male, occ×21]
  x_age_gender   dim=25   baseline + age×male
  x_age_sq       dim=25   baseline + age²
  x_2way_simple  dim=26   baseline + age×male + age²
  x_gender_occ   dim=45   baseline + male×occ[i]×21
  x_age_occ      dim=45   baseline + age×occ[i]×21
  x_2way_full    dim=68   baseline + age×male + age² + male×occ×21 + age×occ×21
  x_3way         dim=89   x_2way_full + age×male×occ[i]×21

All configs use binary reward (1.0 if rating >= 4) and 90/10 split.

Usage:
  python benchmark/movielens/cross_feature_sweep.py
"""

import json
import numpy as np
from pathlib import Path

ARMS      = ["Drama", "Comedy", "Action", "Romance", "Thriller", "Adventure"]
N_OCC     = 21
DATA_DIR  = Path(__file__).parent.parent / "data"
TRAIN_WAL = DATA_DIR / "movielens_train.jsonl"
TEST_WAL  = DATA_DIR / "movielens_test.jsonl"


# ---------------------------------------------------------------------------
# LinUCB (Sherman-Morrison)
# ---------------------------------------------------------------------------

class LinUCB:
    def __init__(self, arms, d, alpha):
        self.arms = arms; self.d = d; self.alpha = alpha
        self.A_inv = {a: np.eye(d)     for a in arms}
        self.b     = {a: np.zeros(d)   for a in arms}

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
# WAL loader
# ---------------------------------------------------------------------------

def load_wal(path):
    pred, rew = {}, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        ev = json.loads(line)
        if "Predicted" in ev:
            p = ev["Predicted"]; pred[p["interaction_id"]] = (p["arm_id"], p["context"])
        elif "Rewarded" in ev:
            r = ev["Rewarded"]; rew[r["interaction_id"]] = r["reward"]
    return [(a, c, rew[i]) for i, (a, c) in pred.items() if i in rew]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_context(raw_ctx, feature_set):
    """
    raw_ctx may be either:
      - 3-dim [age_norm, is_male, occ_norm]  (old WAL format)
      - 24-dim [1.0, age, male, occ_oh×21]  (current WAL format, one-hot+bias)

    We unpack whichever is present and rebuild the full cross-feature vector.
    """
    if len(raw_ctx) == 3:
        age, male, occ_n = raw_ctx
        occ_id = min(max(int(round(occ_n * N_OCC)), 0), N_OCC - 1)
        occ_oh = [0.0] * N_OCC
        occ_oh[occ_id] = 1.0
    else:
        # 24-dim: [bias, age, male, occ_oh×21]
        age    = raw_ctx[1]
        male   = raw_ctx[2]
        occ_oh = list(raw_ctx[3:3 + N_OCC])
        occ_id = int(np.argmax(occ_oh))

    # Current best baseline (dim=24)
    base = [1.0, age, male] + occ_oh

    if feature_set == "baseline":
        return base                                                          # 24

    if feature_set == "x_age_gender":
        return base + [age * male]                                          # 25

    if feature_set == "x_age_sq":
        return base + [age * age]                                           # 25

    if feature_set == "x_2way_simple":
        return base + [age * male, age * age]                               # 26

    if feature_set == "x_gender_occ":
        # Does male/female interact with occupation? (e.g. female librarian vs male)
        return base + [male * occ_oh[i] for i in range(N_OCC)]              # 45

    if feature_set == "x_age_occ":
        # Does age interact with occupation? (e.g. young student vs old student)
        return base + [age * occ_oh[i] for i in range(N_OCC)]               # 45

    if feature_set == "x_2way_full":
        # All pairwise interactions
        cross  = [age * male, age * age]
        cross += [male * occ_oh[i] for i in range(N_OCC)]
        cross += [age  * occ_oh[i] for i in range(N_OCC)]
        return base + cross                                                  # 68

    if feature_set == "x_3way":
        # Add age × male × occ (triple interaction)
        cross  = [age * male, age * age]
        cross += [male * occ_oh[i] for i in range(N_OCC)]
        cross += [age  * occ_oh[i] for i in range(N_OCC)]
        cross += [age  * male * occ_oh[i] for i in range(N_OCC)]
        return base + cross                                                  # 89

    raise ValueError(f"Unknown feature_set: {feature_set}")


FEATURE_DIMS = {
    "baseline":      24,
    "x_age_gender":  25,
    "x_age_sq":      25,
    "x_2way_simple": 26,
    "x_gender_occ":  45,
    "x_age_occ":     45,
    "x_2way_full":   68,
    "x_3way":        89,
}


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def run(train, test, alpha, feature_set):
    dim   = FEATURE_DIMS[feature_set]
    model = LinUCB(ARMS, dim, alpha)

    # Binary reward: rating >= 4 → 1.0 (stored as >= 0.8 in continuous WAL)
    for arm, ctx, raw_r in train:
        r = 1.0 if raw_r >= 0.8 else 0.0
        model.update(arm, build_context(ctx, feature_set), r)

    # Replay evaluation with online learning on matches
    ev   = model.clone()
    br, bc = 0.0, 0
    test_r = [1.0 if r >= 0.8 else 0.0 for _, _, r in test]
    rand   = sum(test_r) / len(test_r) if test_r else 0.0

    for (al, ctx, _), r in zip(test, test_r):
        xv = build_context(ctx, feature_set)
        ap = ev.predict(xv)
        if ap == al:
            bc += 1; br += r
            ev.update(ap, xv, r)

    ba   = br / bc if bc else 0.0
    lift = (ba / rand - 1) * 100 if rand else 0.0
    return lift, ba, rand, bc / len(test) if test else 0.0, bc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load all data, re-split at 90/10 (same as current best)
    print("Loading WAL data ...")
    all_data = load_wal(TRAIN_WAL) + load_wal(TEST_WAL)
    split    = int(len(all_data) * 0.9)
    train, test = all_data[:split], all_data[split:]
    print(f"Total: {len(all_data):,}   Train: {len(train):,}   Test: {len(test):,}\n")

    feature_sets = [
        "baseline",
        "x_age_gender",
        "x_age_sq",
        "x_2way_simple",
        "x_gender_occ",
        "x_age_occ",
        "x_2way_full",
        "x_3way",
    ]
    alphas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

    results = []
    col = "  {:<14}  {:<5}  {:>8}  {:>10}  {:>7}  {:>5}"
    print(col.format("feature_set", "alpha", "lift", "bandit", "match%", "dim"))
    print("  " + "-" * 55)

    for feature_set in feature_sets:
        dim = FEATURE_DIMS[feature_set]
        best_lift = -np.inf
        for alpha in alphas:
            lift, ba, rand, match, matched = run(train, test, alpha, feature_set)
            results.append((lift, alpha, feature_set, ba, rand, match, matched))
            marker = " <--" if lift > best_lift else ""
            best_lift = max(best_lift, lift)
            print(col.format(feature_set, f"{alpha:.1f}", f"{lift:+.1f}%",
                             f"{ba:.4f}", f"{match:.1%}", str(dim)) + marker)
        print()

    sep = "=" * 58
    print(sep)
    print("Top 10 by lift:")
    results.sort(reverse=True)
    for lift, alpha, fset, ba, rand, match, matched in results[:10]:
        dim = FEATURE_DIMS[fset]
        print(
            f"  {fset:<14}  alpha={alpha:.1f}  dim={dim:<3}  "
            f"lift={lift:+.1f}%  bandit={ba:.4f}  match={match:.1%}"
        )

    print()
    best = results[0]
    lift, alpha, fset = best[0], best[1], best[2]
    dim = FEATURE_DIMS[fset]
    print(f"Best: {fset}  alpha={alpha}  dim={dim}  lift={lift:+.1f}%")
    print()
    print("To validate with BanditDB, update convert.py:")
    print(f"  FEATURE_DIM = {dim}")
    print(f"  ALPHA       = {alpha}")
    print(f"  TRAIN_RATIO = 0.9")
    print(f"  Use build_context(ctx, '{fset}') in parse_users()")


if __name__ == "__main__":
    main()
